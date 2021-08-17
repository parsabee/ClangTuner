//
// Created by Parsa Bagheri on 7/2/21.
//
#include "OptimizationDriver.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Parser.h"


namespace clang {
namespace tuner {

std::unique_ptr<mlir::ModuleOp> OptimizationDriver::run(mlir::ModuleOp *module,
                                    llvm::SmallVector<llvm::StringRef> &args) {
  llvm::SmallString<32> inputFile;
  llvm::SmallString<32> resultFile;
  llvm::sys::fs::createTemporaryFile("tmp-input-file", "mlir", inputFile);
  llvm::sys::fs::createTemporaryFile("tmp-result-file", "mlir", resultFile);
  llvm::FileRemover inputFileRemover(inputFile.c_str());
  llvm::FileRemover resultFileRemover(resultFile.c_str());

  std::error_code EC;
  llvm::raw_fd_ostream rawStream(inputFile, EC, llvm::sys::fs::F_None);
  if (rawStream.has_error()) {
    return nullptr;
  }
  rawStream << *module;
  rawStream.flush();
  rawStream.close();

  llvm::Optional<llvm::StringRef> Redirects[] = {llvm::StringRef(resultFile),
                                                 llvm::StringRef(resultFile),
                                                 llvm::StringRef(resultFile)};
  auto pathOrErr = llvm::sys::findProgramByName(
      "/Users/parsabagheri/Development/llvm-project/build/bin/mlir-opt");
  if (!pathOrErr)
    return nullptr;

  const std::string &path = *pathOrErr;
  args.insert(args.begin(), path);
  args.push_back(inputFile);
  int RunResult = llvm::sys::ExecuteAndWait(path, args, llvm::None, Redirects);
  if (RunResult != 0)
    return nullptr;

  auto OutputBuf = llvm::MemoryBuffer::getFile(resultFile.c_str());
  if (!OutputBuf)
    return nullptr;

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(OutputBuf.get()), llvm::SMLoc());

  auto optimizedModule = mlir::parseSourceFile(sourceMgr, &this->context);
  return std::make_unique<mlir::ModuleOp>(optimizedModule.release());

//  llvm::SmallVector<llvm::StringRef, 64> Lines;
//  Output.split(Lines, "\n");
//  llvm::errs() << "result " << resultFile << "\n";
//  llvm::errs() << "stored " << inputFile << "\n";
//  llvm::errs() << "code: ";
//  for (auto &l : Lines) {
//    llvm::errs() << l << "\n";
//  }
//  llvm::errs() << "payan\n";
}

std::unique_ptr<mlir::ModuleOp> OptimizationDriver::run(mlir::ModuleOp *module,
                             llvm::ArrayRef<llvm::StringRef> args) {
  llvm::SmallVector<llvm::StringRef> newArgs(args.begin(), args.end());
  return run(module, newArgs);
}
} // namespace tuner
} // namespace clang
