//
// Created by Parsa Bagheri on 6/30/21.
//

#include "OpenMPConfigurer.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace clang {
namespace tuner {

void ConfigureOpenMPPass::runOnOperation() {
  // Get the current FuncOp operation being operated on.
  mlir::FuncOp op = getOperation();

  op.walk([this](mlir::Operation *inst) {
    if (auto ompParallelOp = llvm::dyn_cast<OMPParallelOp>(inst)) {
      mlir::RewritePatternSet patterns(ompParallelOp.getContext());
      patterns.add<ConfigureOpenMPRewritePattern>(inst->getContext());
      mlir::FrozenRewritePatternSet frozen(std::move(patterns));
      mlir::applyOpPatternsAndFold(inst, frozen);
    }
  });
}

mlir::LogicalResult ConfigureOpenMPRewritePattern::matchAndRewrite(
    mlir::omp::ParallelOp op, mlir::PatternRewriter &rewriter) const {
  // Look through the input of the current transpose.
  auto newOp = rewriter.create<OMPParallelOp>(
      op->getLoc(), /*if_expr_var=*/nullptr,
      /*num_threads_var=*/
      rewriter.create<mlir::ConstantIndexOp>(rewriter.getUnknownLoc(), 4),
      /*default_val=*/nullptr, /*private_vars=*/mlir::ValueRange(),
      /*firstprivate_vars=*/mlir::ValueRange(),
      /*shared_vars=*/mlir::ValueRange(),
      /*copyin_vars=*/mlir::ValueRange(),
      /*allocate_vars=*/mlir::ValueRange(),
      /*allocators_vars=*/mlir::ValueRange(), /*proc_bind_val=*/nullptr);
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(op->getBlock());
  rewriter.inlineRegionBefore(op.region(), newOp.region(),
                              newOp.region().begin());
  rewriter.eraseOp(op);
  return mlir::success();
}
}
}