//
// Created by parsa on 5/9/21.
//

#ifndef MLIR_TEST_MLIRGEN_H
#define MLIR_TEST_MLIRGEN_H

#include <memory>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace clang_tune {
/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context);
} // namespace clang_tune

#endif // MLIR_TEST_MLIRGEN_H
