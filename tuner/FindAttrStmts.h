//
// Created by parsa on 4/25/21.
//

#ifndef CLANG_FINDATTRSTMTS_H
#define CLANG_FINDATTRSTMTS_H

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include <iostream>

namespace clang {
namespace tuner {

class FindAttrStmtsVisitor : public RecursiveASTVisitor<FindAttrStmtsVisitor> {
  bool isInTuneAttr = false;
  mlir::ModuleOp theModule = nullptr;
  mlir::OpBuilder opBuilder;
  ASTContext &astContext;

public:
  FindAttrStmtsVisitor(mlir::MLIRContext &context, ASTContext &astContext)
      : opBuilder(&context), astContext(astContext) {}
  bool VisitAttributedStmt(AttributedStmt *attributedStmt);
  bool TraverseAttributedStmt(AttributedStmt *attributedStmt);
  bool VisitForStmt(ForStmt *forStmt);
};

class FindAttrStmtsConsumer : public ASTConsumer {
  FindAttrStmtsVisitor Visitor;

public:
  FindAttrStmtsConsumer(mlir::MLIRContext &context, ASTContext &astContext) : Visitor(context, astContext) {}
  virtual void HandleTranslationUnit(ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class FindAttrStmts : public ASTFrontendAction {
  mlir::MLIRContext &context;

public:
  FindAttrStmts(mlir::MLIRContext &context) : context(context) {}
  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler, llvm::StringRef InFile) {
    return std::make_unique<FindAttrStmtsConsumer>(context, Compiler.getASTContext());
  }
};

} // namespace tuner
} // namespace clang
#endif // CLANG_FINDATTRSTMTS_H
