//
// Created by Parsa Bagheri on 7/3/21.
//
/// This class defines a driver for the compiler operations

#ifndef TUNER__DRIVER_H
#define TUNER__DRIVER_H

#include <array>
#include <condition_variable>
#include <fstream>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

//#include "AttrForLoopFinder.h"
#include "LockableObject.h"
#include "MLIRCodeGenerator.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "clang/AST/Attr.h"
#include "clang/AST/AttrVisitor.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/CommonOptionsParser.h"

namespace clang {
namespace tuner {

/// The llvm modules produced in FindAttrStmtsVisitor will be added to this
/// map, keys: name of the generated function, vals: llvm module
using MLIRModuleOps = llvm::StringMap<std::unique_ptr<mlir::ModuleOp>>;

using MLIRPair =
std::pair<FuturePtr<mlir::ModuleOp>, std::unique_ptr<mlir::MLIRContext>>;

using MLIRQueueProducer = QueueProducer<MLIRPair, std::mutex>;
using MLIRQueueConsumer = QueueConsumer<MLIRPair, std::mutex>;

using LLVMQueueProducer = QueueProducer<FuturePtr<llvm::Module>, std::mutex>;
using LLVMQueueConsumer = QueueConsumer<FuturePtr<llvm::Module>, std::mutex>;

/// ===-------------------------------------------------------------------===///
/// The Compiler Driver
/// ===-------------------------------------------------------------------===///
/// Topological pipeline of the compiler:
/// i) Generate mlir code for the attributed statements with the
/// MLIRCodeGenerator. (optionally save to file with -m0)
/// ii) Perform additional optimizations with mlir-opt. (optionally save to
/// file with -m1)
/// iii) Lower to llvm ir dialect. (optionally save to file with -m2)
/// iv) Convert llvm dialect to llvm ir module. (optionally save to file with
/// -m3)
/// v) Refactor the original source code(extract the mlir attributed stmts
/// into functions). (optionally save to file with -rs)
/// vi) Compile the refactored code with clang down to llvm bit code.
/// vii) Link the two llvm modules modules with the llvm::Linker
/// viii) Write bitcode to file
/// ===-------------------------------------------------------------------===///
class Driver {
public:
  enum Status : unsigned { Fail = 0, Success, Unfinished };

private:
  // Address of main (or any symbol in executable), used for getting the path of
  // the executable
  void *mainAddr;
  const char *argv0; // The name of the executable

  // Clang stuff
  DiagnosticsEngine diags;
  clang::driver::Driver clangDriver;

  // LLVMContext isn't thread safe, this is a mutex wrapper to make it lockable
  Lockable<llvm::LLVMContext, std::mutex> lockableLLVMContext;

  // State of the compiler
  Lockable<Status, std::mutex> MLIRCodeGenStatus;
  Lockable<Status, std::mutex> MLIRCompilationStatus;
  Lockable<Status, std::mutex> RefactoringStatus;
  Lockable<Status, std::mutex> ClangCompilationStatus;
  Lockable<Status, std::mutex> LLVMLinkingStatus;

public:
  Driver(int argc, const char **argv, void *mainAddr);
  ~Driver();

  // Driver is immovable and uncopyable
  Driver(const Driver &other) = delete;
  Driver &operator=(const Driver &other) = delete;

  // ===-------------------------------------------------------------------===//
  // Functions for querying state of driver
  // ===-------------------------------------------------------------------===//

  Status getMLIRCodeGenStatus();
  Status getMLIRCompilationStatus();
  Status getRefactoringStatus();
  Status getClangCompilationStatus();
  Status getLLVMLinkingStatus();

  bool isMLIRCodeGenFinished();
  bool isMLIRCompilationFinished();
  bool isRefactoringFinished();
  bool isClangCompilationFinished();
  bool isLLVMLinkingFinished();

  // ===-------------------------------------------------------------------===//
  // Asynchronous tasks
  // ===-------------------------------------------------------------------===//

  /// Refactors source files and invokes clang on the refactored files
  /// returns a future llvm module
  FuturePtr<llvm::Module> PerformClangCompilation();

  /// Generates mlir code for mlir attributed stmts
  /// Fills the mlir module queue with generated mlir module ops
  /// returns a future bool indicating the status of thread upon termination
  std::future<bool> PerformMLIRCodeGeneration(MLIRQueueProducer &);

  /// Translates mlir modules to llvm modules
  /// takes a lockable list of mlir modules and an empty lockable list of
  /// llvm modules and fills it
  /// returns a future bool indicating the status of thread upon termination
  std::future<bool> PerformMLIRCompilation(MLIRQueueConsumer &,
                                           LLVMQueueProducer &);

  /// returns a future llvm module; the final module
  FuturePtr<llvm::Module> PerformModuleLinking(FuturePtr<llvm::Module>,
                                               LLVMQueueConsumer &);

private:
  std::unique_ptr<llvm::Module>
  convertToLLVMIR(FuturePtr<mlir::ModuleOp> futureModuleOp);

  // ===-------------------------------------------------------------------===//
  // Worker threads will use these functions to set the status of the driver
  // ===-------------------------------------------------------------------===//

  void setMLIRCodeGenStatus(Status status);
  void setMLIRCompilationStatus(Status status);
  void setRefactoringStatus(Status status);
  void setClangCompilationStatus(Status status);
  void setLLVMLinkingStatus(Status status);

  /// Gets rid of debug symbols in llvm modules
  void runStripSymbolsPass(llvm::Module &module);
};

} // namespace tuner
} // namespace clang

#endif // TUNER__DRIVER_H
