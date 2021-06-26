//
// Created by parsa on 5/14/21.
//

#ifndef TUNER_TYPEGEN_H
#define TUNER_TYPEGEN_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace clang::tuner {
class TypeGen : public StmtVisitor<TypeGen, mlir::Type> {
  llvm::LLVMContext &llvmContext;
  llvm::Module &llvmModule;
  mlir::MLIRContext *mlirContext;
  mlir::ModuleOp &moduleOp;
  mlir::OpBuilder &opBuilder;
  ASTContext &astContext;
  const ForStmt *forStmt;
public:
  TypeGen(const ForStmt *forStmt, ASTContext &context, mlir::ModuleOp &moduleOp,
          mlir::OpBuilder &opBuilder, llvm::LLVMContext &llvmContext,
          llvm::Module &llvmModule)
  : llvmContext(llvmContext), llvmModule(llvmModule),
  mlirContext(moduleOp->getContext()), moduleOp(moduleOp),
  opBuilder(opBuilder), astContext(context), forStmt(forStmt) {}

  mlir::Type VisitImplicitCastExpr(ImplicitCastExpr *);

  mlir::Type VisitDeclRefExpr(DeclRefExpr *);

  mlir::Type VisitArraySubscriptExpr(ArraySubscriptExpr *);

  mlir::Type VisitIntegerLiteral(IntegerLiteral *);

  mlir::MemRefType getType(Decl *);

  mlir::MemRefType getType(VarDecl *);

  mlir::Attribute getAttr(VarDecl *);

  };
} // namespace clang::tuner
#endif // TUNER_TYPEGEN_H
