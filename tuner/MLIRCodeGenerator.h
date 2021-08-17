//
// Created by parsa on 5/14/21.
//

#ifndef TUNER_CODEGEN_H
#define TUNER_CODEGEN_H

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
  llvm::LLVMContext &llvmContext;
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
  MLIRCodeGenerator(ForStmt *forStmt, ASTContext &context, mlir::ModuleOp &moduleOp,
                    llvm::LLVMContext &llvmContext,
                    mlir::OpBuilder &opBuilder, SourceManager &sourceManager,
                    DiagnosticsEngine &diags)
      : sourceManager(sourceManager), llvmContext(llvmContext),
        llvmModule("llvm_mod", llvmContext),
        mlirContext(moduleOp->getContext()), moduleOp(moduleOp),
        opBuilder(opBuilder), astContext(context), forStmt(forStmt),
        CGModule(context, {}, {}, {}, llvmModule, diags),
        typeGen(moduleOp, CGModule.getTypes()) {}

  mlir::Value VisitImplicitCastExpr(ImplicitCastExpr *);

  mlir::Value VisitDeclRefExpr(DeclRefExpr *);

  mlir::Value VisitDeclStmt(DeclStmt *);

  mlir::Value VisitArraySubscriptExpr(ArraySubscriptExpr *);

  mlir::Value VisitBinaryOperator(BinaryOperator *);

  mlir::Value forLoopHandler(ForStmt *, bool isParallel);

  mlir::Value VisitForStmt(ForStmt *);

  //  mlir::Value VisitParallelForStmt(ForStmt *);

  mlir::Value VisitIntegerLiteral(IntegerLiteral *);

  std::unique_ptr<llvm::Module> performLoweringAndOptimizationPipeline(llvm::SmallVector<StringRef> &);

  bool lowerToMLIR();
};
}// namespace clang::tuner

#endif// TUNER_CODEGEN_H
