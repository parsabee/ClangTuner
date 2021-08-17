//
// Created by Parsa Bagheri on 7/2/21.
//

#ifndef TUNER__OPTIMIZATIONDRIVER_H
#define TUNER__OPTIMIZATIONDRIVER_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace clang {
namespace tuner {
class OptimizationDriver {
  mlir::MLIRContext &context;

public:
  OptimizationDriver(mlir::MLIRContext &context) : context(context) {}
  std::unique_ptr<mlir::ModuleOp> run(mlir::ModuleOp *module,
                                      llvm::ArrayRef<llvm::StringRef> args);
  std::unique_ptr<mlir::ModuleOp> run(mlir::ModuleOp *module,
                                      llvm::SmallVector<llvm::StringRef> &args);
};
} // namespace tuner
} // namespace clang

#endif // TUNER__OPTIMIZATIONDRIVER_H
