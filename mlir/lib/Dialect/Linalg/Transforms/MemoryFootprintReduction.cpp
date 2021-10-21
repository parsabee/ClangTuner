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
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/PassManager.h"
#include <numeric>

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::scf;

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

/// Calculates the size (in bytes) of a ranked tensor
static inline size_t getSizeFromShape(llvm::ArrayRef<int64_t> shape,
                                      size_t elementBitWidth) {
  assert(elementBitWidth % 8 == 0 && "BitWidth has to be divisible by 8");
  auto elementSize = elementBitWidth / 8;
  return std::accumulate(shape.begin(), shape.end(), 
                         elementSize, std::multiplies<>());
}

struct RankedOperands {
  struct Operand {
    mlir::ArrayRef<int64_t> shape;
    size_t bitWidth;
  };
  SmallVector<Operand> inputs;
  SmallVector<Operand> outputs;
};

RankedOperands getOperands(linalg::LinalgOp op) {
  auto findRankedOperands = [](linalg::OpOperandVector operandVector)
      -> SmallVector<RankedOperands::Operand> {
    SmallVector<RankedOperands::Operand> rankedOperands;
    for (auto op : operandVector) {
      if (auto memRef =
              op->get().getType().dyn_cast<mlir::MemRefType>()) {
        if (!memRef.hasStaticShape())
          return {};
        rankedOperands.push_back(
            {memRef.getShape(), memRef.getElementTypeBitWidth()});
      } else {
        return {};
      }
    }
    return rankedOperands;
  };
  return {findRankedOperands(op.getInputOperands()),
          findRankedOperands(op.getOutputOperands())};
}

static llvm::SmallVector<int64_t> getNewShape(llvm::ArrayRef<int64_t> oldShape,
                                              size_t bitWidth, size_t maxSize) {

  llvm::SmallVector<int64_t> newShape(oldShape.begin(), oldShape.end());
  for (size_t i = 0, end = oldShape.size(); i < end; i++) {
    auto curSize = getSizeFromShape(newShape, bitWidth);
    int64_t coefficient = curSize / maxSize;
    if (oldShape[i] < coefficient)
      newShape[i] = 1;
    else {
      newShape[i] = oldShape[i] / coefficient;
      break;
    }
  }
  for (auto i : newShape)
    llvm::errs() << i << " ";
  llvm::errs() << "\n";
  return newShape;
}

static llvm::SmallVector<int64_t> getTilingShape(LinalgOp op,
                                                 int64_t maxMemoryFootprint) {
  auto operands = getOperands(op);

  // We can't tile if we have unranked stuff
  if (operands.inputs.empty() || operands.outputs.empty())
    return {};

  auto opSizeFold = [](size_t size, RankedOperands::Operand &op) {
    return size + getSizeFromShape(op.shape, op.bitWidth);
  };

  auto getOperandsSize =
      [opSizeFold](SmallVector<RankedOperands::Operand> &ops) {
        return std::accumulate(ops.begin(), ops.end(), 0, opSizeFold);
      };

  int64_t inputSize = getOperandsSize(operands.inputs),
          outputSize = getOperandsSize(operands.outputs);
  int64_t totalSize = inputSize + outputSize;
  // If the total mem footprint is already below max, we're done
  if (totalSize < maxMemoryFootprint)
    return {};

  double normalizedOutputSize = (double)outputSize / totalSize;
  int64_t desiredOutputSize = maxMemoryFootprint * normalizedOutputSize;
  if (outputSize < desiredOutputSize)
    return {};

  // TODO: Right now it is assumed that the generic op only has one output
  auto &rankedOp = *operands.outputs.begin();
  return getNewShape(rankedOp.shape, rankedOp.bitWidth, desiredOutputSize);
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
          getTilingShape(genOp, maxMemFootprint), [&](int64_t s) -> Value {
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