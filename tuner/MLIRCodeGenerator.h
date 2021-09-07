//
// Created by parsa on 5/14/21.
//

#ifndef TUNER_CODEGEN_H
#define TUNER_CODEGEN_H

#include "LockableObject.h"
#include "MLIRTypeGenerator.h"

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

  // generates the mlir type of expr
  MLIRTypeGenerator typeGen;

  // Storing the stack allocations
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

  mlir::Value VisitImplicitCastExpr(ImplicitCastExpr *);

  mlir::Value VisitDeclRefExpr(DeclRefExpr *);

  mlir::Value VisitDeclStmt(DeclStmt *);

  mlir::Value VisitArraySubscriptExpr(ArraySubscriptExpr *);

  mlir::Value VisitBinaryOperator(BinaryOperator *);

  mlir::Value forLoopHandler(ForStmt *, bool isParallel);

  mlir::Value VisitForStmt(ForStmt *);

  //  mlir::Value VisitParallelForStmt(ForStmt *);

  mlir::Value VisitIntegerLiteral(IntegerLiteral *);

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
