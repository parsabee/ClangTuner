//===-- MLIRCodeGenerator.h -------------------------------------*- C++ -*-===//
//
// Author: Parsa Bagheri
//
//===----------------------------------------------------------------------===//
//
// This module will visit Stmt nodes in Clang's Stmt AST and generates 
// corresponding MLIR code
//
//===----------------------------------------------------------------------===//
// 
// Copyright (c) 2021 Parsa Bagheri
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//===----------------------------------------------------------------------===//

#ifndef TUNER_CODEGEN_H
#define TUNER_CODEGEN_H

#include "clangtuner/ADT/LockableObject.h"
#include "clangtuner/MLIRCodeGen/MLIRTypeGenerator.h"

#include "../../clang/lib/CodeGen/CodeGenModule.h"

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/PreprocessorOptions.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <condition_variable>
#include <mutex>

namespace clang::tuner {

struct ForLoopArgs {
private:
  std::vector<mlir::Type> argTypes;
  std::vector<StringRef> argNames;
  std::vector<StringRef> mlirArgNames;
  llvm::SmallDenseMap<StringRef, size_t> nameArgMap;

public:
  bool append(mlir::Type type, StringRef name);
  const mlir::Type lookUp(StringRef name) const;
  const std::vector<mlir::Type> &getArgTypes() const;
  const std::vector<StringRef> &getArgNames() const;
};

class MLIRCodeGenerator : public StmtVisitor<MLIRCodeGenerator, mlir::Value> {
  SourceManager &sourceManager;
  Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext;
  llvm::Module llvmModule;
  mlir::MLIRContext *mlirContext;
  mlir::ModuleOp &moduleOp;
  mlir::OpBuilder &opBuilder;
  ASTContext &astContext;
  const ForStmt *forStmt;
  ForLoopArgs loopArgs;

  // Use this for generating llvm types from c types
  ::clang::CodeGen::CodeGenModule CGModule;

  // This module produces the MLIR type associated with an expression
  MLIRTypeGenerator typeGen;

  // This table stores the variables names (key) and the MLIR allocations
  // (value) corresponding to them.
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> scopedHashTable;
  llvm::DenseMap<int, mlir::Value> indexConstants;

  void declare(llvm::StringRef var, mlir::Value value);

  mlir::Value handleVarDecl(VarDecl *);

  mlir::Value handleAssignment(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value handleAssignment(Expr *lhs, Expr *rhs);

  mlir::Value handleAddition(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value handleSubtraction(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value handleMultiplication(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value handleDivision(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value createConstantIndex(unsigned int index);

public:
  MLIRCodeGenerator(
      ForStmt *forStmt, ASTContext &context, mlir::ModuleOp &moduleOp,
      Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext,
      mlir::OpBuilder &opBuilder, SourceManager &sourceManager,
      DiagnosticsEngine &diags);

  //===----------------------------------------------------------------------===//
  //
  // X-macro pattern, will expand to Visit method signatures
  // for example:
  //    mlir::Value VisitStmt(Stmt *);
  //
  //===----------------------------------------------------------------------===//
  
  #define FIELD(AST_STMT_NODE) \
    mlir::Value Visit##AST_STMT_NODE(AST_STMT_NODE *);

  #include "clangtuner/SupportedStmts.def"

  //===----------------------------------------------------------------------===//

  mlir::Value forLoopHandler(ForStmt *, bool isParallel);

  std::unique_ptr<mlir::ModuleOp>
  performLoweringAndOptimizationPipeline(llvm::SmallVector<StringRef> &);

  bool lowerToMLIR();

  static std::unique_ptr<mlir::ModuleOp> runOpt(mlir::ModuleOp *,
                                                llvm::SmallVector<StringRef> &);

  static bool runParallelizingPass(mlir::ModuleOp &, mlir::MLIRContext *);

  static bool lowerToLLVMDialect(mlir::ModuleOp &moduleOp,
                                 mlir::MLIRContext *mlirContext);

  /// Converts mlir llvm dialect to llvm ir.
  /// LLVMContext is not thread safe, so in threaded settings we need to pass
  /// a condition variable and mutex to this function
  static std::unique_ptr<llvm::Module>
  convertToLLVMIR(mlir::ModuleOp &moduleOp, mlir::MLIRContext *mlirContext,
                  llvm::LLVMContext &llvmContext, std::condition_variable *cond,
                  std::mutex *mtx);
};
} // namespace clang::tuner

#endif // TUNER_CODEGEN_H
