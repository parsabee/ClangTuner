//
// Created by parsa on 5/9/21.
//

#include "ClangTune/MLIRGen.h"
#include "ClangTune/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>
//
// using namespace mlir::clang_tune;
// using namespace toy;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace clang_tune {

class MLIRGenImpl {};
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context) {
//  return MLIRGenImpl(context).mlirGen();
}

} // namespace clang_tune