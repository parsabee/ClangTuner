//
// Created by Parsa Bagheri on 7/18/21.
//

#include "ParallelizingPass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace clang {
namespace tuner {

void ParallelizingPass::runOnOperation() {
  // Get the current FuncOp operation being operated on.
  auto convertForToParallel = [](mlir::Operation &op){
    mlir::RewritePatternSet patterns(op.getContext());
    patterns.add<ParallelizingRewritePattern>(op.getContext());
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));
    if (mlir::failed(mlir::applyOpPatternsAndFold(&op, frozen))) {
      llvm::errs() << "Failed to convert scf.ForOp to scf.ParallelOp @" <<
                   op.getLoc() << "\n";
    }
  };

  auto funOp = getOperation();
  mlir::Operation *theForOp = nullptr;

  for (auto &block : funOp.getRegion()) {
    for (auto &op: block.getOperations()) {
      if (llvm::isa<mlir::scf::ForOp>(op)) {
        theForOp = &op;
      }
    }
  }

  if (theForOp)
    convertForToParallel(*theForOp);


//  funOp.walk([](mlir::Operation *inst) {
//    if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(inst)) {
//      mlir::RewritePatternSet patterns(forOp.getContext());
//      patterns.add<ParallelizingRewritePattern>(inst->getContext());
//      mlir::FrozenRewritePatternSet frozen(std::move(patterns));
//      if (mlir::failed(mlir::applyOpPatternsAndFold(inst, frozen))) {
//        llvm::errs() << "Failed to convert scf.ForOp to scf.ParallelOp @" <<
//            inst->getLoc() << "\n";
//      }
//    }
//  });
}

mlir::LogicalResult ParallelizingRewritePattern::matchAndRewrite(
    mlir::scf::ForOp op, mlir::PatternRewriter &rewriter) const {
  // Look through the input of the current transpose.

  auto parallelOp = rewriter.create<mlir::scf::ParallelOp>(
      op.getLoc(), mlir::ValueRange{op.lowerBound()}, mlir::ValueRange{op.upperBound()},
      mlir::ValueRange{op.step()}, llvm::None);

  auto &loopBody = parallelOp.getLoopBody();
  auto *lastBlock = &loopBody.getBlocks().back();
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(op->getBlock());
  rewriter.inlineRegionBefore(op.region(), parallelOp.region(),
                              parallelOp.region().begin());
//  rewriter.create<mlir::scf::YieldOp>(rewriter.getUnknownLoc());
  rewriter.eraseBlock(lastBlock);

  rewriter.eraseOp(op);
  return mlir::success();
}
} // namespace tuner
} // namespace clang