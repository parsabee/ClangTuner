//
// Created by parsa on 4/25/21.
//

#ifndef CLANG_FINDATTRSTMTS_H
#define CLANG_FINDATTRSTMTS_H

#include "AttrForLoopRefactorer.h"

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

namespace clang {
namespace tuner {

/// The llvm modules produced in FindAttrStmtsVisitor will be added to this
/// map, keys: name of the generated function, vals: llvm module
using Modules = llvm::StringMap<std::unique_ptr<llvm::Module>>;

class MLIRCodeGenAttrVisitor
    : public RecursiveASTVisitor<MLIRCodeGenAttrVisitor> {
  mlir::ModuleOp theModule;
  mlir::OpBuilder opBuilder;
  ASTContext &astContext;
  SourceManager &sourceManager;
  DiagnosticsEngine &diags;
  Modules &modules;
  llvm::LLVMContext &llvmContext;

public:
  MLIRCodeGenAttrVisitor(mlir::MLIRContext &context,
                         llvm::LLVMContext &llvmContext, ASTContext &astContext,
                         SourceManager &sm, DiagnosticsEngine &diags,
                         Modules &modules)
      : opBuilder(&context), astContext(astContext), sourceManager(sm),
        diags(diags), modules(modules), llvmContext(llvmContext) {}

  bool VisitAttributedStmt(AttributedStmt *attributedStmt);

private:
  bool handleMLIROptAttr(const MLIROptAttr *, ForStmt *);
};

class RewriterAttrVisitor : public RecursiveASTVisitor<RewriterAttrVisitor> {
  ASTContext &astContext;
  SourceManager &sourceManager;
  AttrForLoopRefactorer loopRefactorer;
  DiagnosticsEngine &diags;

public:
  RewriterAttrVisitor(ASTContext &astContext, SourceManager &sm,
                      Rewriter &rewriter, DiagnosticsEngine &diags)
      : astContext(astContext), sourceManager(sm),
        loopRefactorer(sm, astContext, rewriter), diags(diags) {}

  bool VisitAttributedStmt(AttributedStmt *attributedStmt);

private:
  /// This function changes the AST by extracting the for loop within
  /// attributedStmt into its own function
  bool extractForLoopIntoFunction(AttributedStmt *attributedStmt);
};

template <typename AttrStmtVisitor>
class AttrStmtConsumer : public ASTConsumer {
  AttrStmtVisitor Visitor;

public:
  AttrStmtConsumer(AttrStmtVisitor &&Visitor) : Visitor(std::move(Visitor)) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class RewriterFrontendAction : public ASTFrontendAction {
  Rewriter &rewriter;
  DiagnosticsEngine &diags;

public:
  RewriterFrontendAction(Rewriter &rewriter, DiagnosticsEngine &diags)
      : rewriter(rewriter), diags(diags) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    rewriter.setSourceMgr(Compiler.getSourceManager(), Compiler.getLangOpts());
    return std::make_unique<AttrStmtConsumer<RewriterAttrVisitor>>(
        RewriterAttrVisitor(Compiler.getASTContext(),
                            Compiler.getSourceManager(), rewriter, diags));
  }
};

class MLIRCodeGenFrontendAction : public ASTFrontendAction {
  mlir::MLIRContext &mlirContext;
  llvm::LLVMContext &llvmContext;
  DiagnosticsEngine &diags;
  Modules &modules;

public:
  MLIRCodeGenFrontendAction(mlir::MLIRContext &mlirContext,
                            llvm::LLVMContext &llvmContext,
                            DiagnosticsEngine &diags, Modules &modules)
      : mlirContext(mlirContext), llvmContext(llvmContext), diags(diags),
        modules(modules) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<AttrStmtConsumer<MLIRCodeGenAttrVisitor>>(
        MLIRCodeGenAttrVisitor(mlirContext, llvmContext,
                               Compiler.getASTContext(),
                               Compiler.getSourceManager(), diags, modules));
  }
};

std::unique_ptr<tooling::FrontendActionFactory>
newRewriterFrontendActionFactory(Rewriter &rewriter, DiagnosticsEngine &diags);

std::unique_ptr<tooling::FrontendActionFactory>
newMLIRCodeGenFrontendActionFactory(mlir::MLIRContext &mlirContext,
                                    llvm::LLVMContext &llvmContext,
                                    Modules &modules, DiagnosticsEngine &diags);

} // namespace tuner
} // namespace clang
#endif // CLANG_FINDATTRSTMTS_H
