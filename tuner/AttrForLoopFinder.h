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
/// Note: we have save the LLVMContext that was used to create the module
/// otherwise we'll get really freaky errors, I slaved to figure it out
using Modules = llvm::StringMap<std::pair<std::unique_ptr<llvm::LLVMContext>,
                                          std::unique_ptr<llvm::Module>>>;

struct TunableMLIRModule {
  mlir::ModuleOp theModule;
};

using TunableMLIRModules = llvm::SmallVector<TunableMLIRModule>;

using TuningUnsignedParameters = llvm::StringMap<llvm::SmallVector<unsigned>>;

/// Visits AttrStmts and if the attribute is a "tune" attribute, it will extract
/// the ForStmt within it into its own function. An object of this class changes
/// the ASTContext
class FindAttrStmtsVisitor : public RecursiveASTVisitor<FindAttrStmtsVisitor> {
public:
  enum class VisitorTy {
    Refactorer,
    MLIRCodeGenerator
  };

private:
  bool isInTuneAttr = false;
  mlir::ModuleOp theModule;
  mlir::OpBuilder opBuilder;
  ASTContext &astContext;
  SourceManager &sourceManager;
  AttrForLoopRefactorer loopRefactorer;
  DiagnosticsEngine &diags;
  Modules &modules;
  VisitorTy visitorTy;

public:
  FindAttrStmtsVisitor(mlir::MLIRContext &context, ASTContext &astContext,
                       SourceManager &sm, Rewriter &rewriter,
                       DiagnosticsEngine &diags, Modules &modules,
                       VisitorTy visitorTy)
      : opBuilder(&context), astContext(astContext), sourceManager(sm),
        loopRefactorer(sm, astContext, rewriter), diags(diags),
        modules(modules), visitorTy(visitorTy) {}

  bool handleMLIROptAttr(AttributedStmt *, const MLIROptAttr *, ForStmt *);
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
                        DiagnosticsEngine &diags, Modules &modules,
                        FindAttrStmtsVisitor::VisitorTy visitorTy)
      : Visitor(context, astContext, sm, rewriter, diags, modules, visitorTy) {}
  virtual void HandleTranslationUnit(ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class AttrForLoopFinder : public ASTFrontendAction {
  mlir::MLIRContext &context;
  Rewriter &rewriter;
  DiagnosticsEngine &diags;
  Modules &modules;
  FindAttrStmtsVisitor::VisitorTy visitorTy;

public:
  AttrForLoopFinder(mlir::MLIRContext &context, Rewriter &rewriter,
                    DiagnosticsEngine &diags, Modules &modules,
                    FindAttrStmtsVisitor::VisitorTy visitorTy)
      : context(context), rewriter(rewriter), diags(diags), modules(modules),
        visitorTy(visitorTy)
  {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    rewriter.setSourceMgr(Compiler.getSourceManager(), Compiler.getLangOpts());
    return std::make_unique<FindAttrStmtsConsumer>(
        context, Compiler.getASTContext(), Compiler.getSourceManager(),
        rewriter, diags, modules, visitorTy);
  }
};

std::unique_ptr<tooling::FrontendActionFactory>
newFindAttrStmtsFrontendActionFactory(
    mlir::MLIRContext &context, Rewriter &rewriter, DiagnosticsEngine &diags,
    Modules &modules,
    FindAttrStmtsVisitor::VisitorTy visitorTy);

} // namespace tuner
} // namespace clang
#endif // CLANG_FINDATTRSTMTS_H
