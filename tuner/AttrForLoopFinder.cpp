//
// Created by parsa on 4/25/21.
//
#include "AttrForLoopFinder.h"
#include "SemaCheck.h"
#include "TypeCorrector.h"

#include "clang/Tooling/Refactoring/ASTSelection.h"

#include "ClangTune/Dialect.h"
#include "MLIRCodeGenerator.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "clang/AST/ASTDumper.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/IR/DerivedTypes.h"

namespace clang::tuner {

static mlir::ModuleOp initializeMlirModule(mlir::OpBuilder &opBuilder) {
  auto theModule = mlir::ModuleOp::create(opBuilder.getUnknownLoc());

#define LOAD_DIALECT(DIALECT)                                                  \
  theModule->getContext()->getOrLoadDialect<DIALECT>()

  //  LOAD_DIALECT(mlir::clang_tune::ClangTuneDialect);
  LOAD_DIALECT(mlir::AffineDialect);
  LOAD_DIALECT(mlir::memref::MemRefDialect);
  LOAD_DIALECT(mlir::omp::OpenMPDialect);
  LOAD_DIALECT(mlir::LLVM::LLVMDialect);
  LOAD_DIALECT(mlir::scf::SCFDialect);
  LOAD_DIALECT(mlir::StandardOpsDialect);
  mlir::registerOpenMPDialectTranslation(*theModule->getContext());
  //  theModule->getContext()->getOrLoadDialect<mlir::omp::OpenMPDialect>();
#undef LOAD_DIALECT
  return theModule;
}

using namespace ast_matchers;

using Declarations = std::map<llvm::StringRef, const DeclRefExpr *>;

class DeclCollector : public clang::RecursiveASTVisitor<DeclCollector> {
  Declarations &varDecls;
  Declarations &declRefs;

public:
  DeclCollector(Declarations &varDecls, Declarations &declRefs)
      : varDecls(varDecls), declRefs(declRefs) {}

  bool VisitVarDecl(VarDecl *varDecl) {
    varDecls[varDecl->getName()] = nullptr;
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *declRefExpr) {
    declRefs[declRefExpr->getDecl()->getName()] = declRefExpr;
    return true;
  }
};

bool MLIRCodeGenAttrVisitor::VisitAttributedStmt(
    AttributedStmt *attributedStmt) {

  // checking if attr is a tune attr
  for (auto attr : attributedStmt->getAttrs()) {
    if (isATuneAttr(attr)) {
      auto forStmt = dyn_cast<ForStmt>(attributedStmt->getSubStmt());
      if (!forStmt)
        return false;

      if (auto mlirOpt = dyn_cast<MLIROptAttr>(attr)) {
        return handleMLIROptAttr(mlirOpt, forStmt);
      }
    }
  }

  return true;
}

bool MLIRCodeGenAttrVisitor::handleMLIROptAttr(const MLIROptAttr *mlirOpt,
                                               ForStmt *forStmt) {
  llvm::SmallVector<StringRef> attrArgs;

  for (auto it : mlirOpt->argv()) {
    auto pair = it.split(" ");

    while (true) {
      attrArgs.push_back(pair.first);
      if (pair.second == "")
        break;
      pair = pair.second.split(" ");
    }
  }

  auto forloopName = createFunctionName(forStmt, sourceManager);
  auto module = initializeMlirModule(opBuilder);
  {
    MLIRCodeGenerator codeGen(forStmt, astContext, module, llvmContext,
                              opBuilder, sourceManager, diags);

    auto llvmModule = codeGen.performLoweringAndOptimizationPipeline(attrArgs);
    if (llvmModule) {
      //      auto pair = std::make_pair(std::move(llvmCntx),
      //      std::move(llvmModule));
      modules.insert({forloopName, std::move(llvmModule)});
    } else {
      llvm::errs() << "error while compiling with MLIR\n";
      return false;
    }
  }

  return true;
}

bool RewriterAttrVisitor::VisitAttributedStmt(AttributedStmt *attributedStmt) {

  // checking if attr is a tune attr
  for (auto attr : attributedStmt->getAttrs()) {
    if (isATuneAttr(attr)) {
      if (!isa<ForStmt>(attributedStmt->getSubStmt())) {
        return false;
      }
      if (isa<MLIROptAttr>(attr)) {
        loopRefactorer.performExtraction(attributedStmt);
      }
    }
  }

  return true;
}

std::unique_ptr<tooling::FrontendActionFactory>
newRewriterFrontendActionFactory(Rewriter &rewriter, DiagnosticsEngine &diags) {
  class RewriterFrontendActionFactory : public tooling::FrontendActionFactory {
    Rewriter &rewriter;
    DiagnosticsEngine &diags;

  public:
    RewriterFrontendActionFactory(Rewriter &rewriter, DiagnosticsEngine &diags)
        : rewriter(rewriter), diags(diags) {}

    std::unique_ptr<FrontendAction> create() override {
      return std::make_unique<RewriterFrontendAction>(rewriter, diags);
    }
  };

  return std::unique_ptr<RewriterFrontendActionFactory>(
      new RewriterFrontendActionFactory(rewriter, diags));
}

std::unique_ptr<tooling::FrontendActionFactory>
newMLIRCodeGenFrontendActionFactory(mlir::MLIRContext &mlirContext,
                                    llvm::LLVMContext &llvmContext,
                                    Modules &modules,
                                    DiagnosticsEngine &diags) {

  class MLIRCodeGenFrontendActionFactory
      : public tooling::FrontendActionFactory {
    mlir::MLIRContext &mlirContext;
    llvm::LLVMContext &llvmContext;
    Modules &modules;
    DiagnosticsEngine &diags;

  public:
    MLIRCodeGenFrontendActionFactory(mlir::MLIRContext &mlirContext,
                                     llvm::LLVMContext &llvmContext,
                                     Modules &modules, DiagnosticsEngine &diags)
        : mlirContext(mlirContext), llvmContext(llvmContext), modules(modules),
          diags(diags) {}

    std::unique_ptr<FrontendAction> create() override {
      return std::make_unique<MLIRCodeGenFrontendAction>(
          mlirContext, llvmContext, diags, modules);
    }
  };

  return std::unique_ptr<MLIRCodeGenFrontendActionFactory>(
      new MLIRCodeGenFrontendActionFactory(mlirContext, llvmContext, modules,
                                           diags));
}

} // namespace clang::tuner