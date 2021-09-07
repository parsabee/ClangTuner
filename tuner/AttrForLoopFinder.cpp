//
// Created by parsa on 4/25/21.
//
#include "AttrForLoopFinder.h"
#include "SemaCheck.h"
#include "TypeCorrector.h"

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

/// Constructs the moduleOp, mlirContext, and opBuilder
/// Loads builtin dialects
static std::tuple<std::unique_ptr<mlir::ModuleOp>,
                  std::unique_ptr<mlir::MLIRContext>, mlir::OpBuilder>
initializeMlirModule() {

  auto mlirContext = std::make_unique<mlir::MLIRContext>();
  mlir::OpBuilder opBuilder(mlirContext.get());
  auto theModule = mlir::ModuleOp::create(opBuilder.getUnknownLoc());

  // Loading the necessary dialects
#define LOAD_DIALECT(DIALECT)                                                  \
  theModule->getContext()->getOrLoadDialect<DIALECT>()

  LOAD_DIALECT(mlir::AffineDialect);
  LOAD_DIALECT(mlir::memref::MemRefDialect);
  LOAD_DIALECT(mlir::omp::OpenMPDialect);
  LOAD_DIALECT(mlir::LLVM::LLVMDialect);
  LOAD_DIALECT(mlir::scf::SCFDialect);
  LOAD_DIALECT(mlir::StandardOpsDialect);

#undef LOAD_DIALECT
  mlir::registerOpenMPDialectTranslation(*theModule->getContext());
  return {std::make_unique<mlir::ModuleOp>(theModule), std::move(mlirContext),
          std::move(opBuilder)};
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

  bool hasMLIRAttr = false;
  // checking if attr is a tune attr
  for (auto attr : attributedStmt->getAttrs()) {
    if (isAnMLIRAttr(attr)) {
      hasMLIRAttr = true;
    }
  }

  if (hasMLIRAttr) {
    auto forStmt = dyn_cast<ForStmt>(attributedStmt->getSubStmt());
    if (!forStmt)
      return false;

    auto &&[moduleOp, mlirContext, opBuilder] = initializeMlirModule();

    // This generates mlir code for the for loop
    if (handleMLIRAttr(forStmt, *moduleOp, opBuilder)) {

      // Handling other mlir attributes
      for (auto attr : attributedStmt->getAttrs()) {

        if (const auto *mlirOpt = dyn_cast<MLIROptAttr>(attr)) {
          if (auto optimized = handleMLIROptAttr(forStmt, *moduleOp, mlirOpt)) {
            moduleOp = std::move(optimized);
          } else {
            llvm::errs() << "error producing optimized file\n";
            return false;
          }
        }

        if (llvm::isa<MLIRParallelAttr>(attr)) {
          if (!MLIRCodeGenerator::runParallelizingPass(*moduleOp,
                                                       mlirContext.get())) {
            llvm::errs() << "error parallelizing the module\n";
            return false;
          }
        }
      }

      // Lower to llvm dialect and add to the mlir queue
      if (MLIRCodeGenerator::lowerToLLVMDialect(*moduleOp, mlirContext.get())) {
        std::promise<std::unique_ptr<mlir::ModuleOp>> promise;
        promise.set_value(std::move(moduleOp));
        auto futureVal = promise.get_future();
        moduleProducer.lockAndEnqueue(
            {std::move(futureVal), std::move(mlirContext)});
      } else {
        llvm::errs() << "error while compiling with MLIR\n";
        return false;
      }
    }
  }

  return true;
}

bool MLIRCodeGenAttrVisitor::handleMLIRAttr(ForStmt *forStmt,
                                            mlir::ModuleOp &module,
                                            mlir::OpBuilder &opBuilder) {

  auto forloopName = createFunctionName(forStmt, sourceManager);

  MLIRCodeGenerator codeGen(forStmt, astContext, module, lockableLLVMContext,
                            opBuilder, sourceManager, diags);

  // Writes to module
  return codeGen.lowerToMLIR();
}

std::unique_ptr<mlir::ModuleOp> MLIRCodeGenAttrVisitor::handleMLIROptAttr(
    ForStmt *forStmt, mlir::ModuleOp &module, const MLIROptAttr *mlirOpt) {

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

  return MLIRCodeGenerator::runOpt(&module, attrArgs);
}

// bool MLIRCodeGenAttrVisitor::handleMLIROptAttr(const MLIROptAttr *mlirOpt,
//                                               mlir::ModuleOp module,
//                                               ForStmt *forStmt) {
//  llvm::SmallVector<StringRef> attrArgs;
//
//  for (auto it : mlirOpt->argv()) {
//    auto pair = it.split(" ");
//
//    while (true) {
//      attrArgs.push_back(pair.first);
//      if (pair.second == "")
//        break;
//      pair = pair.second.split(" ");
//    }
//  }
//
//  auto forloopName = createFunctionName(forStmt, sourceManager);
//
//  auto mlirContext = std::make_unique<mlir::MLIRContext>();
//
//  mlir::OpBuilder opBuilder(mlirContext.get());
//
//  auto module = initializeMlirModule(opBuilder);
//
//  MLIRCodeGenerator codeGen(forStmt, astContext, module, lockableLLVMContext,
//                            opBuilder, sourceManager, diags);
//
//  auto mlirModule = codeGen.performLoweringAndOptimizationPipeline(attrArgs);
//  if (mlirModule) {
//    std::promise<std::unique_ptr<mlir::ModuleOp>> promise;
//    promise.set_value(std::move(mlirModule));
//    auto futureVal = promise.get_future();
//    moduleProducer.lockAndEnqueue(
//        {std::move(futureVal), std::move(mlirContext)});
//  } else {
//    llvm::errs() << "error while compiling with MLIR\n";
//    return false;
//  }
//
//  return true;
//}

bool RewriterAttrVisitor::VisitAttributedStmt(AttributedStmt *attributedStmt) {

  // checking if attr is a tune attr
  for (auto attr : attributedStmt->getAttrs()) {
    if (isAnMLIRAttr(attr)) {
      loopRefactorer.performExtraction(attributedStmt);
      break;
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
newMLIRCodeGenFrontendActionFactory(
    Lockable<Driver::Status, std::mutex> &MLIRCodeGenStatus,
    Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext,
    MLIRQueueProducer &modules, DiagnosticsEngine &diags,
    std::function<void(void)> callback) {

  class MLIRCodeGenFrontendActionFactory
      : public tooling::FrontendActionFactory {
    Lockable<Driver::Status, std::mutex> &MLIRCodeGenStatus;
    Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext;
    MLIRQueueProducer &modules;
    DiagnosticsEngine &diags;
    std::function<void(void)> callback;

  public:
    MLIRCodeGenFrontendActionFactory(
        Lockable<Driver::Status, std::mutex> &MLIRCodeGenStatus,
        Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext,
        MLIRQueueProducer &modules, DiagnosticsEngine &diags,
        std::function<void(void)> callback)
        : MLIRCodeGenStatus(MLIRCodeGenStatus),
          lockableLLVMContext(lockableLLVMContext), modules(modules),
          diags(diags), callback(std::move(callback)) {}

    std::unique_ptr<FrontendAction> create() override {
      return std::make_unique<MLIRCodeGenFrontendAction>(
          MLIRCodeGenStatus, lockableLLVMContext, diags, modules,
          std::move(callback));
    }
  };

  return std::unique_ptr<MLIRCodeGenFrontendActionFactory>(
      new MLIRCodeGenFrontendActionFactory(MLIRCodeGenStatus,
                                           lockableLLVMContext, modules, diags,
                                           std::move(callback)));
}

} // namespace clang::tuner