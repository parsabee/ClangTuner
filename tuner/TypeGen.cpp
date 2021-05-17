//
// Created by parsa on 5/14/21.
//
#include "TypeGen.h"

namespace clang::tuner {

static mlir::Type getBuiltinType(const Type *type,
                                  mlir::MLIRContext *mlirContext) {
  if (type->isFloatingType()) {
    return mlir::Float32Type::get(mlirContext);
  } else if (type->isIntegerType()) {
    return mlir::IntegerType::get(mlirContext, 32);
  } else if (type->isBooleanType()) {
    return mlir::IntegerType::get(mlirContext, 1);
  } else if (type->isCharType()) {
    return mlir::IntegerType::get(mlirContext, 1);
  }
  return nullptr;
}

static void findArrayShape(mlir::MLIRContext *mlirContext, QualType *qualType,
                           SmallVector<int64_t> &shape, mlir::Type &t) {
  auto type = qualType->getTypePtr();
  if (auto array = dyn_cast<ArrayType>(type)) {
    if (auto constantArray = dyn_cast<ConstantArrayType>(type)) {
      const llvm::APInt &apInt = constantArray->getSize();
      uint64_t size = *(constantArray->getSize().getRawData());
      shape.push_back(static_cast<int64_t>(size));
    } else {
      shape.push_back(-1);
    }
    auto elementType = array->getElementType();
    findArrayShape(mlirContext, &elementType, shape, t);
  } else {
    t = getBuiltinType(type, mlirContext);
  }
}

static mlir::MemRefType createMemref(mlir::MLIRContext *mlirContext, QualType *qualType) {
  SmallVector<int64_t> shape;
  mlir::MemRefType type;
  findArrayShape(mlirContext, qualType, shape, type);
  if (shape.empty())
    return type;
  return mlir::MemRefType::get(shape, type);
}

mlir::Type TypeGen::VisitImplicitCastExpr(ImplicitCastExpr *castExpr) {
  return Visit(castExpr->getSubExpr());
}

mlir::Type TypeGen::VisitDeclRefExpr(DeclRefExpr *declRef) {
  auto qualType = declRef->getType();
  auto type = qualType.getTypePtr();
  if (type->isArrayType()) {
    SmallVector<int64_t> shape;
    mlir::Type t;
    findArrayShape(mlirContext, &qualType, shape, t);
    return mlir::MemRefType::get(shape, t);
  } else {
    mlir::Type t = getBuiltinType(type, mlirContext);
    return mlir::MemRefType::get({1}, t);
  }
}

mlir::MemRefType TypeGen::getType(VarDecl *varDecl) {
  auto type = varDecl->getType();
  return createMemref(mlirContext, &type);
}

mlir::MemRefType TypeGen::getType(Decl *decl) {
  switch(decl->getKind()) {
  case Decl::Var:
    return getType(dyn_cast<VarDecl>(decl));
  default:
    assert(false && "only this table is supported for now");
    return nullptr;
  }
}

mlir::Type TypeGen::VisitArraySubscriptExpr(ArraySubscriptExpr *arraySubscript) {
  return Visit(arraySubscript->getLHS());
}

mlir::Type TypeGen::VisitIntegerLiteral(IntegerLiteral *integerLiteral) {
  return mlir::IntegerType::get(mlirContext, 32);
}

mlir::Attribute TypeGen::getAttr(VarDecl *varDecl) {
//  return mlir::IntegerAttr::get()
}


} // namespace clang::tuner