//
// Created by parsa on 5/14/21.
//

#ifndef TUNER_CODEGEN_H
#define TUNER_CODEGEN_H

#include "TypeGen.h"
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
  SmallVector<mlir::Type> argTypes;
  SmallVector<StringRef> argNames;
  SmallVector<StringRef> mlirArgNames;
  llvm::SmallDenseMap<StringRef, size_t> nameArgMap;

public:
  bool append(mlir::Type type, StringRef name);
  const mlir::Type lookUp(StringRef name) const;
  const SmallVector<mlir::Type> &getArgTypes() const;
  const SmallVector<StringRef> &getArgNames() const;
};

class CodeGen : public StmtVisitor<CodeGen, mlir::Value> {
  llvm::LLVMContext llvmContext;
  llvm::Module llvmModule;
  mlir::MLIRContext *mlirContext;
  mlir::ModuleOp &moduleOp;
  mlir::OpBuilder &opBuilder;
  ASTContext &astContext;
  const ForStmt *forStmt;
  ForLoopArgs loopArgs;

  // generates the mlir type of expr
  TypeGen typeGen;

  // Storing the stack allocations
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> scopedHashTable;
  llvm::ScopedHashTable<int, mlir::Value> constants;

  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value);

  mlir::Value handleVarDecl(VarDecl *);

  mlir::Value handleAssignment(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value handleAssignment(Expr *lhs, Expr *rhs);

  mlir::Value handleAddition(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value handleSubtraction(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value handleMultiplication(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value handleDivision(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value createConstantIndex(unsigned int index);

public:
  CodeGen(ForStmt *forStmt, ASTContext &context, mlir::ModuleOp &moduleOp,
          mlir::OpBuilder &opBuilder)
      : llvmContext(), llvmModule("llvm_mod", llvmContext),
        mlirContext(moduleOp->getContext()), moduleOp(moduleOp),
        opBuilder(opBuilder), astContext(context), forStmt(forStmt),
        typeGen(forStmt, astContext, moduleOp, opBuilder, llvmContext,
                llvmModule) {}

  mlir::Value VisitImplicitCastExpr(ImplicitCastExpr *);

  mlir::Value VisitDeclRefExpr(DeclRefExpr *);

  mlir::Value VisitDeclStmt(DeclStmt *);

  mlir::Value VisitArraySubscriptExpr(ArraySubscriptExpr *);

  mlir::Value VisitBinaryOperator(BinaryOperator *);

  mlir::Value VisitForStmt(ForStmt *);

  mlir::Value VisitIntegerLiteral(IntegerLiteral *);

  void run();
};
} // namespace clang::tuner

#endif // TUNER_CODEGEN_H
