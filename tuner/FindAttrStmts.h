//
// Created by parsa on 4/25/21.
//

#ifndef CLANG_FINDATTRSTMTS_H
#define CLANG_FINDATTRSTMTS_H

#include "ForLoopRefactorer.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
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

/// Visits AttrStmts and if the attribute is a "tune" attribute, it will extract
/// the ForStmt within it into its own function. An object of this class changes
/// the ASTContext
class FindAttrStmtsVisitor : public RecursiveASTVisitor<FindAttrStmtsVisitor> {
  bool isInTuneAttr = false;
  mlir::ModuleOp theModule;
  std::map<std::string, mlir::ModuleOp> modules;
  mlir::OpBuilder opBuilder;
  ASTContext &astContext;
  SourceManager &sourceManager;
  ForLoopRefactorer loopRefactorer;
  DiagnosticsEngine &diags;

public:
  FindAttrStmtsVisitor(mlir::MLIRContext &context, ASTContext &astContext,
                       SourceManager &sm, Rewriter &rewriter,
                       DiagnosticsEngine &diags)
      : opBuilder(&context), astContext(astContext), sourceManager(sm),
        loopRefactorer(sm, astContext, rewriter), diags(diags) {}
  bool VisitAttributedStmt(AttributedStmt *attributedStmt);

private:
  /// This function changes the AST by extracting the for loop within
  /// attributedStmt into its own function
  bool extractForLoopIntoFunction(AttributedStmt *attributedStmt);
};

class FindAttrStmtsConsumer : public ASTConsumer {
  FindAttrStmtsVisitor Visitor;

public:
  FindAttrStmtsConsumer(mlir::MLIRContext &context, ASTContext &astContext,
                        SourceManager &sm, Rewriter &rewriter,
                        DiagnosticsEngine &diags)
      : Visitor(context, astContext, sm, rewriter, diags) {}
  virtual void HandleTranslationUnit(ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class FindAttrStmts : public ASTFrontendAction {
  mlir::MLIRContext &context;
  Rewriter &rewriter;
  DiagnosticsEngine &diags;

public:
  FindAttrStmts(mlir::MLIRContext &context, Rewriter &rewriter,
                DiagnosticsEngine &diags)
      : context(context), rewriter(rewriter), diags(diags) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    rewriter.setSourceMgr(Compiler.getSourceManager(), Compiler.getLangOpts());
    return std::make_unique<FindAttrStmtsConsumer>(
        context, Compiler.getASTContext(), Compiler.getSourceManager(),
        rewriter, diags);
  }
};

} // namespace tuner
} // namespace clang
#endif // CLANG_FINDATTRSTMTS_H
