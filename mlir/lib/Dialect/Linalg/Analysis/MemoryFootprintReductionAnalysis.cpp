//===- MemoryFootprintReductionAnalysis.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Analysis/MemoryFootprintReductionAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

using namespace mlir;
using namespace mlir::linalg;

static int64_t getDivisorCeil(int64_t n, int64_t divisor) {
  assert(divisor <= n && divisor >= 1);
  while (n % divisor)
    divisor++;
  return divisor;
}

/// Calculates the size (in bytes) of a ranked tensor
static inline size_t getSizeFromShape(llvm::ArrayRef<int64_t> shape,
                                      size_t elementBitWidth) {
  assert(elementBitWidth % 8 == 0 && "BitWidth has to be divisible by 8");
  auto elementSize = elementBitWidth / 8;
  return std::accumulate(shape.begin(), shape.end(), elementSize,
                         std::multiplies<>());
}

struct RankedOperands {
  struct Operand {
    mlir::ArrayRef<int64_t> shape;
    size_t bitWidth;
  };
  SmallVector<Operand, 2> inputs;
  SmallVector<Operand, 2> outputs;
};

RankedOperands getOperands(LinalgOp op) {
  auto findRankedOperands = [](OpOperandVector operandVector)
      -> SmallVector<RankedOperands::Operand, 2> {
    SmallVector<RankedOperands::Operand, 2> rankedOperands;
    for (auto *op : operandVector) {
      if (auto memRef = op->get().getType().dyn_cast<mlir::MemRefType>()) {
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

using StaticallyShapedOpSizeCompFn =
    std::function<size_t(ArrayRef<int64_t>, int64_t)>;

static llvm::SmallVector<int64_t, 4>
getNewShape(llvm::ArrayRef<int64_t> oldShape, size_t bitWidth, size_t maxSize,
            StaticallyShapedOpSizeCompFn cmpFn) {

  llvm::SmallVector<int64_t, 4> newShape(oldShape.begin(), oldShape.end());
  for (size_t i = 0, end = oldShape.size(); i < end; i++) {
    auto curSize = cmpFn(newShape, bitWidth);
    int64_t reductionFactor = (curSize + maxSize - 1) / maxSize;
    if (oldShape[i] < reductionFactor)
      newShape[i] = 1;
    else {
      auto divisor = getDivisorCeil(oldShape[i], reductionFactor);
      newShape[i] = oldShape[i] / divisor;
      break;
    }
  }

  return newShape;
}

static llvm::SmallVector<int64_t, 4>
getMatMulTileSizes(RankedOperands &rankedOps, int64_t maxMemoryFootprint) {
  assert(rankedOps.inputs.size() == 2 && rankedOps.outputs.size() == 1);
  StaticallyShapedOpSizeCompFn cmpFn = [](ArrayRef<int64_t> shape,
                                          int64_t bitwidth) {
    int64_t nbytes = bitwidth / 8;
    return (shape[0] * shape[1] * nbytes) + (shape[1] * shape[2] * nbytes) +
           (shape[0] * shape[2] * nbytes);
  };
  auto m = rankedOps.inputs[0].shape[0], n = rankedOps.inputs[0].shape[1],
       k = rankedOps.inputs[1].shape[1];
  return getNewShape({m, k, n}, rankedOps.inputs[0].bitWidth,
                     maxMemoryFootprint, cmpFn);
}

static SmallVector<int64_t> getGenericTileSizes(RankedOperands &operands,
                                                int64_t maxMemoryFootprint) {

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

  auto &rankedOp = *operands.outputs.begin();
  return getNewShape(rankedOp.shape, rankedOp.bitWidth, desiredOutputSize,
                     getSizeFromShape);
}

namespace mlir {
namespace linalg {
llvm::SmallVector<int64_t, 4>
computeTileSizesForMemoryFootprintReduction(LinalgOp op,
                                            int64_t maxMemoryFootprint) {

  auto operands = getOperands(op);
  // TODO: At the moment it is assumed that the linalg op only has one output
  // We can't tile if we have unranked stuff
  if (operands.inputs.empty() || operands.outputs.empty() ||
      operands.outputs.size() > 1)
    return {};

  if (isa<MatmulOp>(op)) {
    return getMatMulTileSizes(operands, maxMemoryFootprint);
  }
  return getGenericTileSizes(operands, maxMemoryFootprint);
}
} // namespace linalg
} // namespace mlir