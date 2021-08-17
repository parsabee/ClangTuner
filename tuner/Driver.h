////
//// Created by Parsa Bagheri on 7/3/21.
////
///// This class defines a driver for the compiler operations
//
//#ifndef TUNER__DRIVER_H
//#define TUNER__DRIVER_H
//
//#include <fstream>
//#include <memory>
//#include <vector>
//
//#include "clang/AST/Attr.h"
//#include "clang/AST/AttrVisitor.h"
//#include "clang/Frontend/TextDiagnosticPrinter.h"
//#include "clang/Tooling/CommonOptionsParser.h"
//
//#include "mlir/Dialect/Affine/IR/AffineOps.h"
//#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
//#include "mlir/Dialect/MemRef/IR/MemRef.h"
//#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
//#include "mlir/Dialect/SCF/SCF.h"
//#include "mlir/Dialect/StandardOps/IR/Ops.h"
//#include "mlir/IR/BuiltinOps.h"
//#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
//#include "mlir/Target/LLVMIR/ModuleTranslation.h"
//
//#include "AttrForLoopFinder.h"
//#include "MLIRCodeGenerator.h"
//
//namespace clang {
//namespace tuner {
//
//using MLIRTranslationUnit = AttributedStmt;
//
//class Driver {
//public:
//  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
//  TextDiagnosticPrinter *DiagClient;
//  IntrusiveRefCntPtr<DiagnosticIDs> DiagIDs;
//  DiagnosticsEngine Diags;
//  clang::tooling::ClangTool Tool;
//
//  Driver(clang::tooling::CommonOptionsParser &optionsParser)
//      : DiagOpts(new DiagnosticOptions()),
//        DiagClient(new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts)),
//        DiagIDs(new DiagnosticIDs()), Diags(DiagIDs, DiagOpts, DiagClient),
//        Tool(optionsParser.getCompilations(),
//             optionsParser.getSourcePathList()) {}
//
//public:
//  std::unique_ptr<llvm::SmallVector<MLIRTranslationUnit *>>
//  collectTranslationUnits() {
//
//    clang::tuner::Modules modules;
//    mlir::MLIRContext context;
//    clang::Rewriter rewriter;
//
//    std::unique_ptr<llvm::SmallVector<AttributedStmt *>> mlirTranslationUnits =
//        std::make_unique<llvm::SmallVector<AttributedStmt *>>();
//
//    auto fronendAction = clang::tuner::newFindAttrStmtsFrontendActionFactory(
//        context, rewriter, Diags, modules,
//        [&mlirTranslationUnits](AttributedStmt *attrStmt) -> bool {
//          mlirTranslationUnits->push_back(attrStmt);
//          return true;
//        });
//
//    if (Tool.run(fronendAction.get()) == 1) {
//      return nullptr;
//    }
//
//    return mlirTranslationUnits;
//  }
//
//  /// extracts the forloop within the attribute, performs the necessary type
//  /// corrections and writes the resulting code to a new file. it returns a
//  /// string to the location of the file
//  bool
//  extractAttributedForLoops(clang::tooling::CommonOptionsParser &optionsParser,
//                            SourceManager &sourceManager,
//                            ASTContext &astContext,
//                            llvm::SmallString<32> &filename) {
//    clang::tuner::Modules modules;
//    mlir::MLIRContext context;
//    clang::Rewriter rewriter;
//
//    AttrForLoopRefactorer loopRefactorer(sourceManager, astContext, rewriter);
//
//    std::unique_ptr<llvm::SmallVector<AttributedStmt *>> mlirTranslationUnits =
//        std::make_unique<llvm::SmallVector<AttributedStmt *>>();
//
//    auto fronendAction = clang::tuner::newFindAttrStmtsFrontendActionFactory(
//        context, rewriter, Diags, modules,
//        [&loopRefactorer](AttributedStmt *attrStmt) -> bool {
//          loopRefactorer.performExtraction(attrStmt);
//          return true;
//        });
//
//    clang::tooling::ClangTool Tool(optionsParser.getCompilations(),
//                                   optionsParser.getSourcePathList());
//    if (Tool.run(fronendAction.get()) == 1) {
//      return false;
//    }
//
//    const auto *RewriteBuf =
//        rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());
//
//    if (!RewriteBuf) {
//      return false;
//    }
//
//    llvm::sys::fs::createTemporaryFile("temp-refactored-file", ".cpp",
//                                       filename);
//
//    std::ofstream out(filename.c_str());
//    if (out.bad())
//      return false;
//
//    out << std::string(RewriteBuf->begin(), RewriteBuf->end());
//    out.close();
//
//    return true;
//  }
//
//#define LOAD_DIALECT(MODULE, DIALECT) \
//                                      \
//  MODULE->getContext()->getOrLoadDialect<DIALECT>()
//
//  mlir::ModuleOp initializeMlirModule(mlir::OpBuilder &opBuilder) {
//    auto theModule = mlir::ModuleOp::create(opBuilder.getUnknownLoc());
//    LOAD_DIALECT(theModule, mlir::AffineDialect);
//    LOAD_DIALECT(theModule, mlir::memref::MemRefDialect);
//    LOAD_DIALECT(theModule, mlir::omp::OpenMPDialect);
//    LOAD_DIALECT(theModule, mlir::LLVM::LLVMDialect);
//    LOAD_DIALECT(theModule, mlir::scf::SCFDialect);
//    LOAD_DIALECT(theModule, mlir::StandardOpsDialect);
//    mlir::registerOpenMPDialectTranslation(*theModule->getContext());
//    return theModule;
//  }
//
//  std::unique_ptr<SmallVector<mlir::ModuleOp>> lowerAttrForStmtToMLIRModules(
//      std::unique_ptr<llvm::SmallVector<MLIRTranslationUnit *>>
//          translationUnits,
//      SourceManager &sourceManager, mlir::OpBuilder &opBuilder,
//      ASTContext &astContext) {
//    auto modules = std::make_unique<SmallVector<mlir::ModuleOp>>();
//    for (auto *TU : *translationUnits) {
//      auto forStmt = dyn_cast<ForStmt>(TU->getSubStmt());
//      if (!forStmt)
//        return nullptr;
//
//      auto forloopName = createFunctionName(forStmt, sourceManager);
//      auto module = initializeMlirModule(opBuilder);
//      auto llvmCntx = std::make_unique<llvm::LLVMContext>();
//      {
//        MLIRCodeGenerator codeGen(forStmt, astContext, module, *llvmCntx,
//                                  opBuilder, sourceManager, Diags);
//      }
//    }
//    assert(modules->size() == translationUnits->size());
//    return modules;
//  }
//
//public:
//  ~Driver() = default;
//  Driver(const Driver &) = delete;
//};// namespace tuner
//
//}
//}// namespace clang::tuner
//
//#endif// TUNER__DRIVER_H
