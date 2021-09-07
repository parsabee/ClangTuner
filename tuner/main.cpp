//
// Created by parsa on 4/24/21.
//

#include "Driver.h"
#include "IOUtils.h"

// Defining the command line options, these options are declared in
// CommandLineOpts.h
llvm::cl::OptionCategory ClangTuneCategory("clang-tune options");

llvm::cl::opt<bool>
    MLIRCodeGenOnly("mlir-code-gen-only",
                    llvm::cl::desc("Only performs the mlir code "
                                   "generation stage"),
                    llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool> MLIRCompilationOnly(
    "mlir-compilation-only",
    llvm::cl::desc("Only performs the mlir compilation stage"),
    llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool> ClangCompilationOnly(
    "clang-compilation-only",
    llvm::cl::desc("Only performs the clang compilation stage"),
    llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool>
    SaveInitialMLIRFile("m0",
                        llvm::cl::desc("Write the initial mlir code to "
                                       "file."),
                        llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool>
    SaveOptimizedMLIRFile("m1",
                          llvm::cl::desc("Write the optimized mlir code to "
                                         "file."),
                          llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool> SaveLLVMDialectMLIRFile(
    "m2",
    llvm::cl::desc("Write the llvm dialect mlir code to "
                   "file."),
    llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool>
    SaveLLVMModule("m3",
                   llvm::cl::desc("Write the final llvm module generated "
                                  "by lowering mlir code file."),
                   llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool> SaveRefactoredSourceFile(
    "rs", llvm::cl::desc("Write the refactored source to file."),
    llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool> SaveDebugSymbols(
    "dbg",
    llvm::cl::desc("Save source information for creating debug symbols."),
    llvm::cl::cat(ClangTuneCategory));

clang::tooling::CommonOptionsParser *OptionsParser;

using namespace clang::tuner;

int main(int argc, const char **argv) {

  auto ExpectedParser = clang::tooling::CommonOptionsParser::create(
      argc, argv, ClangTuneCategory);

  if (!ExpectedParser) {
    // Fail gracefully for unsupported options.
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }

  // Global, is accessed throughout the program
  OptionsParser = &ExpectedParser.get();

  clang::tuner::Driver driver(argc, argv, (void *)main);

  LockableQueue<MLIRPair, std::mutex> mlirModules;
  LockableQueue<FuturePtr<llvm::Module>, std::mutex> llvmModules;

  LLVMQueueProducer llvmProducer(llvmModules);
  LLVMQueueConsumer llvmConsumer(llvmModules);
  MLIRQueueProducer mlirProducer(mlirModules);
  MLIRQueueConsumer mlirConsumer(mlirModules);

  bool performClangComp = ClangCompilationOnly.getValue(),
       performMLIRCodeGen = MLIRCodeGenOnly.getValue(),
       performMLIRComp = MLIRCompilationOnly.getValue();

  // If none of the stages of the pipeline is specified, all of them are
  // performed
  if (!performClangComp && !performMLIRComp && !performMLIRCodeGen) {
    performClangComp = performMLIRCodeGen = performMLIRComp = true;
  }

  // Holds the status of compiler operations
  std::vector<std::future<bool>> statusOfOperations;

  if (performMLIRCodeGen || performMLIRComp) {
    auto futureMLIRCodeGenStatus =
        driver.PerformMLIRCodeGeneration(mlirProducer);
    statusOfOperations.push_back(std::move(futureMLIRCodeGenStatus));
  }

  if (performMLIRComp) {
    auto futureMLIRCompilationStatus =
        driver.PerformMLIRCompilation(mlirConsumer, llvmProducer);
    statusOfOperations.push_back(std::move(futureMLIRCompilationStatus));
  }

  FuturePtr<llvm::Module> futureLinkedModule;
  if (performClangComp) {
    auto futureLLVMModule = driver.PerformClangCompilation();
    if (!futureLLVMModule.valid())
      return 1;

    // wait until the module is available then hand it to the linker
    futureLinkedModule =
        driver.PerformModuleLinking(std::move(futureLLVMModule),
                                    llvmConsumer);
  }

  // Wait until async jobs are finished
  for (auto &futureStatus : statusOfOperations) {
    if (!futureStatus.get())
      return 1;
  }

  if (!futureLinkedModule.valid())
    return 1;

  auto finalModule = futureLinkedModule.get();

  auto sources = OptionsParser->getSourcePathList();
  llvm::SmallString<256> generatedFilesName;
  writeModuleToFile(sources[0] + ".ll", generatedFilesName,
                    *finalModule);

  return 0;
}