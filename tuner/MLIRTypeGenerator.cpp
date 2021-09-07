//
// Created by parsa on 5/14/21.
//
#include "MLIRTypeGenerator.h"

namespace clang::tuner {

/// This function is the main procedure of the type generator,
/// if the clang type 'qualType' is an array type, it will
/// go over the dimensions and collect the size (or -1 if size is unknown) of
/// the array. It returns the shape in 'shape', and returns the base type in
/// 't'.
static void
findArrayShape(mlir::MLIRContext *mlirContext, QualType *qualType,
               SmallVector<int64_t> &shape, mlir::Type &t,
               mlir::LLVM::TypeFromLLVMIRTranslator &llvmTypeTranlator,
               CodeGen::CodeGenTypes &CGTypes) {

  // Declaring the lambda first so that we can call it recursively
  std::function<void(mlir::MLIRContext *, QualType *, SmallVector<int64_t> &,
                     mlir::Type &)>
      findShapeRecursively;

  // Recurses through an array type with 'qualType'
  // dimension by dimension and writes its shape to 'shape'.
  findShapeRecursively = [&findShapeRecursively, &llvmTypeTranlator, &CGTypes](
                             mlir::MLIRContext *mlirContext, QualType *qualType,
                             SmallVector<int64_t> &shape, mlir::Type &t) {
    const auto *type = qualType->getTypePtr();

    if (const auto *array = dyn_cast<ArrayType>(type)) {
      if (const auto *constantArray = dyn_cast<ConstantArrayType>(type)) {
        uint64_t size = *(constantArray->getSize().getRawData());
        shape.push_back(static_cast<int64_t>(size));
      } else {
        shape.push_back(-1);
      }
      auto elementType = array->getElementType();
      findShapeRecursively(mlirContext, &elementType, shape, t);
    } else {
      auto llvmType = CGTypes.ConvertType(*qualType);
      t = llvmTypeTranlator.translateType(llvmType);
    }
  };

  const auto *type = qualType->getTypePtr();

  if (isa<ArrayType>(type)) {
    findShapeRecursively(mlirContext, qualType, shape, t);
  } else if (auto *adjustedType = dyn_cast<AdjustedType>(type)) {
    // In c/c++ multi-dimensional arrays decay when passed to functions;
    // that behavior is handled here.

    // Adjusted has two QualTypes: i) the original type ii) the adjusted type;
    // we need the original to infer shape
    auto qualType = adjustedType->getOriginalType();
    findShapeRecursively(mlirContext, &qualType, shape, t);
  } else {
    shape.push_back(1);
    auto *llvmType = CGTypes.ConvertType(*qualType);
    t = llvmTypeTranlator.translateType(llvmType);
  }
}

static mlir::MemRefType
createMemref(mlir::MLIRContext *mlirContext, QualType qualType,
             mlir::LLVM::TypeFromLLVMIRTranslator &llvmTypeTranlator,
             CodeGen::CodeGenTypes &CGTypes) {
  SmallVector<int64_t> shape;
  mlir::MemRefType type;
  findArrayShape(mlirContext, &qualType, shape, type, llvmTypeTranlator,
                 CGTypes);
  if (shape.size() == 1 && shape[0] == 1)
    return mlir::MemRefType::get({}, type);
  else
    return mlir::MemRefType::get(shape, type);
}

mlir::Type
MLIRTypeGenerator::VisitImplicitCastExpr(ImplicitCastExpr *castExpr) {
  return Visit(castExpr->getSubExpr());
}

mlir::Type MLIRTypeGenerator::VisitDeclRefExpr(DeclRefExpr *declRef) {
  auto qualType = declRef->getType();
  auto llvmType = CGTypes.ConvertType(qualType);
  auto type = qualType.getTypePtr();
  if (type->isArrayType()) {
    SmallVector<int64_t> shape;
    mlir::Type t;
    findArrayShape(mlirContext, &qualType, shape, t, llvmTypeTranlator,
                   CGTypes);
    return mlir::MemRefType::get(shape, t);
  } else {
    auto t = llvmTypeTranlator.translateType(llvmType);
    return mlir::MemRefType::get({1}, t);
  }
}

mlir::MemRefType MLIRTypeGenerator::getMemRefType(VarDecl *varDecl) {
  auto memref =
      createMemref(mlirContext, varDecl->getType(), llvmTypeTranlator, CGTypes);
  return memref;
}

mlir::MemRefType MLIRTypeGenerator::getMemRefType(Decl *decl) {
  switch (decl->getKind()) {
  case Decl::Var:
    return getMemRefType(dyn_cast<VarDecl>(decl));
  default:
    assert(false && "only this table is supported for now");
    return nullptr;
  }
}

mlir::Type
MLIRTypeGenerator::VisitArraySubscriptExpr(ArraySubscriptExpr *arraySubscript) {
  return Visit(arraySubscript->getBase());
}

mlir::Type
MLIRTypeGenerator::VisitIntegerLiteral(IntegerLiteral *integerLiteral) {
  return mlir::IntegerType::get(mlirContext, 32);
}

} // namespace clang::tuner