//
// Created by Parsa Bagheri on 6/30/21.
//

#ifndef TUNER_MLIR_CONFIGUREOPENMP_H
#define TUNER_MLIR_CONFIGUREOPENMP_H

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
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

class ConfigureOpenMPRewritePattern
    : public mlir::OpRewritePattern<OMPParallelOp> {

public:
  explicit ConfigureOpenMPRewritePattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::omp::ParallelOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::omp::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

class ConfigureOpenMPPass
    : public mlir::PassWrapper<ConfigureOpenMPPass,
                               mlir::OperationPass<mlir::FuncOp>> {
  mlir::OpBuilder &opBuilder;

public:
  explicit ConfigureOpenMPPass(mlir::OpBuilder &opBuilder)
      : opBuilder(opBuilder) {}

  void runOnOperation() override;
};

} // namespace tuner
} // namespace clang

#endif // TUNER_MLIR_CONFIGUREOPENMP_H
