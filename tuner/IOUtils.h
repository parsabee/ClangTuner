//
// Created by Parsa Bagheri on 8/25/21.
//

#ifndef TUNER__IOUTILS_H
#define TUNER__IOUTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"

/// Writes the module to file with filename
/// It is generic because it should work on llvm and mlir modules
template <typename Module>
bool writeModuleToFile(const std::string &name,
                       llvm::SmallString<256> &fileName, Module &module,
                       bool deleteIfExists = true) {

  // Delete the file if it already exist and create a new one
  if (llvm::sys::fs::exists(name)) {
    if (deleteIfExists) {
      llvm::FileRemover fileRemover(name.c_str());
    } else {
      return false; // already exists
    }
  }

  int MLIROptimizedFileFD;
  llvm::sys::fs::createUniqueFile(name, MLIROptimizedFileFD, fileName);

  std::error_code EC;
  llvm::raw_fd_ostream rawStream(fileName, EC, llvm::sys::fs::F_None);
  if (rawStream.has_error()) {
    return false;
  }

  rawStream << module;
  rawStream.flush();
  rawStream.close();
  return true;
}

std::unique_ptr<mlir::ModuleOp> readModule(llvm::SmallString<256> loopMLIRFile,
                                           mlir::MLIRContext *mlirContext) {
  llvm::SourceMgr sourceMgr;
  auto OutputBuf = llvm::MemoryBuffer::getFile(loopMLIRFile.c_str());
  if (!OutputBuf)
    return nullptr;
  sourceMgr.AddNewSourceBuffer(std::move(OutputBuf.get()), llvm::SMLoc());
  auto theModule = mlir::parseSourceFile(sourceMgr, mlirContext);
  return std::make_unique<mlir::ModuleOp>(theModule.release());
}

#endif // TUNER__IOUTILS_H
