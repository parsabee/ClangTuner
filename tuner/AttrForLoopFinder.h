//
// Created by parsa on 4/25/21.
//

#ifndef CLANG_FINDATTRSTMTS_H
#define CLANG_FINDATTRSTMTS_H

#include "AttrForLoopRefactorer.h"
#include "Driver.h"
#include "LockableObject.h"

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

class MLIRCodeGenAttrVisitor
    : public RecursiveASTVisitor<MLIRCodeGenAttrVisitor> {

  ASTContext &astContext;
  SourceManager &sourceManager;
  DiagnosticsEngine &diags;
  MLIRQueueProducer &moduleProducer;
  Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext;
  Lockable<tuner::Driver::Status, std::mutex> &MLIRCodeGenStatus;

public:
  MLIRCodeGenAttrVisitor(
      Lockable<tuner::Driver::Status, std::mutex> &MLIRCodeGenStatus,
      Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext,
      ASTContext &astContext, SourceManager &sm, DiagnosticsEngine &diags,
      MLIRQueueProducer &modules)
      : astContext(astContext), sourceManager(sm), diags(diags),
        moduleProducer(modules), lockableLLVMContext(lockableLLVMContext),
        MLIRCodeGenStatus(MLIRCodeGenStatus) {}

  MLIRCodeGenAttrVisitor(const MLIRCodeGenAttrVisitor &) = delete;

  MLIRCodeGenAttrVisitor(MLIRCodeGenAttrVisitor &&other)
      : astContext(other.astContext), sourceManager(other.sourceManager),
        diags(other.diags), moduleProducer(other.moduleProducer),
        lockableLLVMContext(other.lockableLLVMContext),
        MLIRCodeGenStatus(other.MLIRCodeGenStatus) {}

  bool VisitAttributedStmt(AttributedStmt *attributedStmt);

private:
  bool handleMLIRAttr(ForStmt *, mlir::ModuleOp &, mlir::OpBuilder &);
  std::unique_ptr<mlir::ModuleOp>
  handleMLIROptAttr(ForStmt *, mlir::ModuleOp &, const MLIROptAttr *);
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
  std::function<void(void)> callback;

public:
  AttrStmtConsumer(AttrStmtVisitor Visitor, std::function<void(void)> callback)
      : Visitor(std::move(Visitor)), callback(std::move(callback)) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    callback();
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
    return std::unique_ptr<AttrStmtConsumer<RewriterAttrVisitor>>(
        new AttrStmtConsumer(RewriterAttrVisitor(Compiler.getASTContext(),
                                                 Compiler.getSourceManager(),
                                                 rewriter, diags),
                             [] {}));
  }
};

class MLIRCodeGenFrontendAction : public ASTFrontendAction {
  Lockable<Driver::Status, std::mutex> &MLIRCodeGenStatus;
  Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext;
  DiagnosticsEngine &diags;
  MLIRQueueProducer &modules;
  std::function<void(void)> callback;

public:
  MLIRCodeGenFrontendAction(
      Lockable<Driver::Status, std::mutex> &MLIRCodeGenStatus,
      Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext,
      DiagnosticsEngine &diags, MLIRQueueProducer &modules,
      std::function<void(void)> callback)
      : MLIRCodeGenStatus(MLIRCodeGenStatus),
        lockableLLVMContext(lockableLLVMContext), diags(diags),
        modules(modules), callback(std::move(callback)) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<AttrStmtConsumer<MLIRCodeGenAttrVisitor>>(
        MLIRCodeGenAttrVisitor(MLIRCodeGenStatus, lockableLLVMContext,
                               Compiler.getASTContext(),
                               Compiler.getSourceManager(), diags, modules),
        std::move(callback));
  }
};

std::unique_ptr<tooling::FrontendActionFactory>
newRewriterFrontendActionFactory(Rewriter &rewriter, DiagnosticsEngine &diags);

std::unique_ptr<tooling::FrontendActionFactory>
newMLIRCodeGenFrontendActionFactory(
    Lockable<Driver::Status, std::mutex> &MLIRCodeGenStatus,
    Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext,
    MLIRQueueProducer &, DiagnosticsEngine &,
    std::function<void(void)> callback);

} // namespace tuner
} // namespace clang
#endif // CLANG_FINDATTRSTMTS_H
