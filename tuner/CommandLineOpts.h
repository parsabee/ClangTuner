//
// Created by Parsa Bagheri on 8/15/21.
//

/// Command line options, global variables defined in main

#ifndef TUNER__COMMANDLINEOPTS_H
#define TUNER__COMMANDLINEOPTS_H

#include "clang/Tooling/CommonOptionsParser.h"

extern llvm::cl::OptionCategory ClangTuneCategory;

extern llvm::cl::opt<bool> ClangCompilationOnly;

extern llvm::cl::opt<bool> MLIRCompilationOnly;

extern llvm::cl::opt<bool> MLIRCodeGenOnly;

extern llvm::cl::opt<bool> SaveInitialMLIRFile;

extern llvm::cl::opt<bool> SaveOptimizedMLIRFile;

extern llvm::cl::opt<bool> SaveLLVMDialectMLIRFile;

extern llvm::cl::opt<bool> SaveLLVMModule;

extern llvm::cl::opt<bool> SaveRefactoredSourceFile;

extern llvm::cl::opt<bool> SaveDebugSymbols;

extern clang::tooling::CommonOptionsParser *OptionsParser;

#endif // TUNER__COMMANDLINEOPTS_H
