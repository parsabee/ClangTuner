//
// Created by parsa on 4/25/21.
//
#include "FindAttrStmts.h"

#include "CodeGen.h"
#include "ClangTune/Dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/IR/DerivedTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"


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

#define LOAD_DIALECT(DIALECT) theModule->getContext()->getOrLoadDialect<DIALECT>()

static mlir::ModuleOp initializeMlirModule(mlir::OpBuilder &opBuilder) {
  auto theModule = mlir::ModuleOp::create(opBuilder.getUnknownLoc());
  LOAD_DIALECT(mlir::clang_tune::ClangTuneDialect);
  LOAD_DIALECT(mlir::AffineDialect);
  LOAD_DIALECT(mlir::memref::MemRefDialect);
  LOAD_DIALECT(mlir::LLVM::LLVMDialect);
  LOAD_DIALECT(mlir::scf::SCFDialect);
  LOAD_DIALECT(mlir::StandardOpsDialect);
  return theModule;
}

bool FindAttrStmtsVisitor::VisitAttributedStmt(AttributedStmt *attributedStmt) {
  auto attrs = attributedStmt->getAttrs();
  for (auto attr : attrs) {
    if (std::strcmp(attr->getSpelling(), "block_dim") == 0) {
      if (auto blockDim = dyn_cast<TuneBlockDimAttr>(attr)) {
        for (auto i : blockDim->blockDim()) {
          std::cout << i << "\n";
        }
      }

      if (auto forStmt = dyn_cast<ForStmt>(attributedStmt->getSubStmt())) {
        theModule = initializeMlirModule(opBuilder);
        CodeGen codeGen(forStmt, astContext, theModule, opBuilder);
        codeGen.run();
      }
    }
  }
  return true;
}

} // namespace clang::tuner