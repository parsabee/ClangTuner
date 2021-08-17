//
// Created by Parsa Bagheri on 7/18/21.
//

#ifndef TUNER__PARALLELIZINGPASS_HPP
#define TUNER__PARALLELIZINGPASS_HPP


#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace clang {
namespace tuner {

using OMPParallelOp = mlir::omp::ParallelOp;

class ParallelizingRewritePattern
    : public mlir::OpRewritePattern<mlir::scf::ForOp> {

public:
  explicit ParallelizingRewritePattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::scf::ForOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

class ParallelizingPass
    : public mlir::PassWrapper<ParallelizingPass,
                               mlir::OperationPass<mlir::FuncOp>> {
public:
  ParallelizingPass() = default;
  void runOnOperation() override;
};

} // namespace tuner
} // namespace clang

#endif // TUNER__PARALLELIZINGPASS_HPP
