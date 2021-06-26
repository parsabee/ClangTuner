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

#include "llvm/IR/Module.h"

#include <iostream>

extern bool getStdHeaders(llvm::SmallVector<std::string> &headers,
                          bool addFlag); // defined
// in main

namespace clang {
namespace tuner {

/// The llvm modules produced in FindAttrStmtsVisitor will be added to this
/// map, keys: name of the generated function, vals: llvm module
/// Note: we have save the LLVMContext that was used to create the module
/// otherwise we'll get really freaky errors, I slaved to figure it out
using Modules = llvm::StringMap<std::pair<std::unique_ptr<llvm::LLVMContext>,
                                          std::unique_ptr<llvm::Module>>>;

/// Visits AttrStmts and if the attribute is a "tune" attribute, it will extract
/// the ForStmt within it into its own function. An object of this class changes
/// the ASTContext
class FindAttrStmtsVisitor : public RecursiveASTVisitor<FindAttrStmtsVisitor> {
  bool isInTuneAttr = false;
  mlir::ModuleOp theModule;
  mlir::OpBuilder opBuilder;
  ASTContext &astContext;
  SourceManager &sourceManager;
  ForLoopRefactorer loopRefactorer;
  DiagnosticsEngine &diags;
  Modules &modules;

public:
  FindAttrStmtsVisitor(mlir::MLIRContext &context, ASTContext &astContext,
                       SourceManager &sm, Rewriter &rewriter,
                       DiagnosticsEngine &diags, Modules &modules)
      : opBuilder(&context), astContext(astContext), sourceManager(sm),
        loopRefactorer(sm, astContext, rewriter), diags(diags),
        modules(modules) {}
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
                        DiagnosticsEngine &diags, Modules &modules)
      : Visitor(context, astContext, sm, rewriter, diags, modules) {}
  virtual void HandleTranslationUnit(ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class FindAttrStmts : public ASTFrontendAction {
  mlir::MLIRContext &context;
  Rewriter &rewriter;
  DiagnosticsEngine &diags;
  Modules &modules;

public:
  FindAttrStmts(mlir::MLIRContext &context, Rewriter &rewriter,
                DiagnosticsEngine &diags, Modules &modules)
      : context(context), rewriter(rewriter), diags(diags), modules(modules) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    rewriter.setSourceMgr(Compiler.getSourceManager(), Compiler.getLangOpts());
    SmallVector<std::string> headers;
    if (getStdHeaders(headers, false)) {
      for (const auto &h : headers) {
        Compiler.getHeaderSearchOptsPtr()->AddPath(
            h, frontend::IncludeDirGroup::After, false, false);
      }
    }

    return std::make_unique<FindAttrStmtsConsumer>(
        context, Compiler.getASTContext(), Compiler.getSourceManager(),
        rewriter, diags, modules);
  }
};

std::unique_ptr<tooling::FrontendActionFactory>
newFindAttrStmtsFrontendActionFactory(mlir::MLIRContext &context,
                                      Rewriter &rewriter,
                                      DiagnosticsEngine &diags,
                                      Modules &modules);

} // namespace tuner
} // namespace clang
#endif // CLANG_FINDATTRSTMTS_H
