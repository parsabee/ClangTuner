//===- MemoryFootprintReduction.cpp - Implementation of linalg Tiling -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements linalg's generic op memory footprint reduction pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/Analysis/MemoryFootprintReductionAnalysis.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

static void
applyTilingToLoopPatterns(LinalgTilingLoopType loopType, FuncOp funcOp,
                          TileSizeComputationFunction tileCompFunc,
                          ArrayRef<StringRef> distributionTypes = {}) {
  auto options = LinalgTilingOptions()
                     .setTileSizeComputationFunction(tileCompFunc)
                     .setLoopType(loopType)
                     .setDistributionTypes(distributionTypes);
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LinalgTilingPattern<GenericOp>>(
      ctx, options,
      LinalgTransformationFilter(ArrayRef<Identifier>{},
                                 Identifier::get("tiled-size", ctx)));
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

namespace {

struct LinalgMemoryFootprintReductionPass
    : public LinalgMemoryFootprintReductionBase<
          LinalgMemoryFootprintReductionPass> {
  LinalgMemoryFootprintReductionPass() = default;
  LinalgMemoryFootprintReductionPass(int64_t maxFootprint) {
    maxMemFootprint = maxFootprint;
  }

  void runOnFunction() override {
    // Apply tiling patterns for each linalg op here
    if (maxMemFootprint <= 0)
      return;

    auto tileCompFunc = [this](OpBuilder &builder,
                               Operation *op) -> SmallVector<Value, 4> {
      auto genOp = dyn_cast<GenericOp>(op);
      if (!genOp)
        return {};

      // Get the appropriate tiling shape for this generic op, map them to mlir
      // value
      return llvm::to_vector<4>(llvm::map_range(
          computeTileSizesForMemoryFootprintReduction(genOp, maxMemFootprint),
          [&](int64_t s) -> Value {
            return builder.create<ConstantIndexOp>(op->getLoc(), s);
          }));
    };

    applyTilingToLoopPatterns(LinalgTilingLoopType::ParallelLoops,
                              getFunction(), tileCompFunc);
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgMemoryFootprintReductionPass(int64_t maxFootprint) {
  return std::make_unique<LinalgMemoryFootprintReductionPass>(maxFootprint);
}