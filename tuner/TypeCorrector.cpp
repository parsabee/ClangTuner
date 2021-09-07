//
// Created by Parsa Bagheri on 6/16/21.
//

#include "TypeCorrector.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

namespace clang {
namespace tuner {

static const std::string getBuiltinType(const Type *type) {
  return type->getBaseElementTypeUnsafe()
      ->getCanonicalTypeInternal()
      .getAsString();
}

static void findShapeRecursively(QualType *qualType,
                                 SmallVector<int64_t> &shape) {
  const auto *type = qualType->getTypePtr();
  if (const auto *array = dyn_cast<ArrayType>(type)) {
    if (const auto *constantArray = dyn_cast<ConstantArrayType>(type)) {
      const llvm::APInt &apInt = constantArray->getSize();
      uint64_t size = *(constantArray->getSize().getRawData());
      shape.push_back(static_cast<int64_t>(size));
    } else {
      shape.push_back(-1);
    }
    auto elementType = array->getElementType();
    findShapeRecursively(&elementType, shape);
  }
}

static void findArrayShape(QualType *qualType, SmallVector<int64_t> &shape) {
  const auto *type = qualType->getTypePtr();
  if (isa<ArrayType>(type)) {
    findShapeRecursively(qualType, shape);
  }
}

void TypeCorrector::insertCorrection(
    QualType type, llvm::SmallVector<std::string> &insertedDecls, int offset) {
  llvm::SmallVector<std::string> args;
  auto st = getBuiltinType(type.getTypePtr());
  std::stringstream ss;

  if (type.getTypePtr()->isArrayType()) {
    SmallVector<int64_t> shape;
    findArrayShape(&type, shape);
    st += "*";

    args.push_back(st);
    args.push_back(st);

    args.push_back("std::size_t"); // offset

    for (int i = 0; i < shape.size(); i++) {
      args.push_back("std::size_t"); // shape
      args.push_back("std::size_t"); // stride
    }
  } else {
    args.push_back(st);
  }
}

llvm::SmallVector<std::string> TypeCorrector::getExpandedTypes(QualType type) {
  llvm::SmallVector<std::string> args;
  if (const auto *decayedType = dyn_cast<clang::DecayedType>(type)) {
    type = decayedType->getOriginalType();
  }

  auto st = getBuiltinType(type.getTypePtr());
  if (type.getTypePtr()->isArrayType()) {
    SmallVector<int64_t> shape;
    findArrayShape(&type, shape);
    st += "*";

    args.push_back(st);
    args.push_back(st);

    args.push_back("std::size_t"); // offset

    for (int i = 0; i < shape.size(); i++) {
      args.push_back("std::size_t"); // shape
      args.push_back("std::size_t"); // stride
    }
  } else {
    args.push_back(st);
  }

  return args;
}

llvm::SmallVector<std::string>
TypeCorrector::getExpandedArgs(const DeclRefExpr *decl) {
  llvm::SmallVector<std::string> args;
  auto type = decl->getDecl()->getType();
  if (const auto *decayedType = dyn_cast<clang::DecayedType>(type)) {
    type = decayedType->getOriginalType();
  }
  if (type.getTypePtr()->isArrayType()) {
    SmallVector<int64_t> shape;
    findArrayShape(&type, shape);

    args.push_back(decl->getDecl()->getNameAsString());
    args.push_back(decl->getDecl()->getNameAsString());

    args.push_back("0"); // offset

    for (auto s : shape) {
      args.push_back(std::to_string(s)); // shape
    }

    for (int i = 0; i < shape.size(); i++) {
      args.push_back("1"); // stride
    }

  } else {

    args.push_back(decl->getDecl()->getNameAsString());
  }

  return args;
}

} // namespace tuner
} // namespace clang