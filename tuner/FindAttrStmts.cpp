//
// Created by parsa on 4/25/21.
//
#include "FindAttrStmts.h"
#include "SemaCheck.h"
#include "TypeCorrector.h"

#include "clang/Tooling/Refactoring/ASTSelection.h"

#include "ClangTune/Dialect.h"
#include "CodeGen.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "clang/AST/ASTDumper.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Refactoring/Extract/Extract.h"
#include "clang/Tooling/Refactoring/RefactoringRuleContext.h"
#include "llvm/IR/DerivedTypes.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"

namespace clang::tuner {

static bool isATuneAttr(AttributedStmt *AttributedStmt) {
  const auto attrs = AttributedStmt->getAttrs();
  for (auto attr : attrs) {
    if (std::strcmp(attr->getSpelling(), "block_dim") == 0) {
      return true;
    }
  }
  return false;
}

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

/// Finds all declaration references within a for loop to be passed as the loops
/// arguments
/// TODO: exclude the decl refs that their declaration is within the loop
static void findLoopInputs(ForStmt *forStmt, ASTContext &context,
                           Declarations &inputArgs) {
  Declarations declRefs, varDecls;
  DeclCollector declCollector(varDecls, declRefs);
  declCollector.TraverseForStmt(forStmt);

  for (const auto &it : declRefs) {
    if (varDecls.find(it.first) == varDecls.end()) {
      inputArgs[it.first] = it.second;
    }
  }
}

bool FindAttrStmtsVisitor::extractForLoopIntoFunction(
    AttributedStmt *attributedStmt) {

  // selecting the nodes in the source range of attributedStmt
  auto sourceRange = attributedStmt->getSourceRange();

  auto selectedNodes =
      clang::tooling::findSelectedASTNodes(astContext, sourceRange);

  auto codeRange = clang::tooling::CodeRangeASTSelection::create(
      sourceRange, selectedNodes.getValue());

  if (codeRange.getPointer() == nullptr) {
    llvm::errs() << "failed to create code range\n";
    return false;
  }

  // creating refactoring rule for extracting the for loop within attributedStmt
  clang::tooling::RefactoringRuleContext RRC(sourceManager);
  RRC.setASTContext(astContext);

  auto extractFunc = clang::tooling::ExtractFunction::initiate(
      RRC, std::move(*codeRange.getPointer()), None);

  if (auto err = extractFunc.takeError()) {
    llvm::logAllUnhandledErrors(std::move(err), llvm::errs());
    return false;
  }

  // refactoring
  Rewriter rewriter(sourceManager, {});
  //  ExtractFuncConsumer consumer(rewriter);
  //  extractFunc->invoke(consumer, RRC);

  auto rewriteBuffer =
      rewriter.getRewriteBufferFor(sourceManager.getMainFileID());
  llvm::outs() << std::string(rewriteBuffer->begin(), rewriteBuffer->end());

  return true;
}

bool FindAttrStmtsVisitor::VisitAttributedStmt(AttributedStmt *attributedStmt) {
  auto attrs = attributedStmt->getAttrs();
  ForStmt *forStmt = nullptr;

  // checking if attr is a tune attr
  for (auto attr : attrs) {
    if (isATuneAttr(attr->getSpelling())) {
      if (auto blockDim = dyn_cast<TuneBlockDimAttr>(attr)) {
        forStmt = dyn_cast<ForStmt>(attributedStmt->getSubStmt());
      }
    }
  }

  if (forStmt) {
    auto forloopName = createFunctionName(forStmt, sourceManager);
    auto module = initializeMlirModule(opBuilder);
    auto llvmCntx = std::make_unique<llvm::LLVMContext>();
    {
      CodeGen codeGen(forStmt, astContext, module, *llvmCntx, opBuilder,
                      sourceManager, diags);
      auto llvmModule = codeGen.generateLLVM();
      if (llvmModule) {
        loopRefactorer.performExtraction(attributedStmt);
        auto pair = std::make_pair(std::move(llvmCntx), std::move(llvmModule));
        modules.insert({forloopName, std::move(pair)});
      }
    }
  }
  return true;
}

std::unique_ptr<tooling::FrontendActionFactory>
newFindAttrStmtsFrontendActionFactory(mlir::MLIRContext &context,
                                      Rewriter &rewriter,
                                      DiagnosticsEngine &diags,
                                      Modules &modules) {

  class FindAttrStmtsFrontendActionFactory
      : public tooling::FrontendActionFactory {
    mlir::MLIRContext &context;
    Rewriter &rewriter;
    DiagnosticsEngine &diags;
    Modules &modules;

  public:
    FindAttrStmtsFrontendActionFactory(mlir::MLIRContext &context,
                                       Rewriter &rewriter,
                                       DiagnosticsEngine &diags, Modules &modules)
        : context(context), rewriter(rewriter), diags(diags), modules(modules) {}

    std::unique_ptr<FrontendAction> create() override {
      return std::make_unique<FindAttrStmts>(context, rewriter, diags, modules);
    }
  };

  return std::unique_ptr<FindAttrStmtsFrontendActionFactory>(
      new FindAttrStmtsFrontendActionFactory(context, rewriter, diags,
                                             modules));
}

} // namespace clang::tuner
