//===- MemoryFootprintReductionAnalysis.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Analysis/MemoryFootprintReductionAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

using namespace mlir;
using namespace mlir::linalg;

static int64_t getDivisor(int64_t n, int64_t divisor,
                          std::function<void(int64_t &)> fn) {
  assert(divisor <= n && divisor >= 1);
  while (n % divisor)
    fn(divisor);
  return divisor;
}

static int64_t getDivisorFloor(int64_t n, int64_t divisor) {
  return getDivisor(n, divisor, [](int64_t &d) { d--; });
}

static int64_t getDivisorCeil(int64_t n, int64_t divisor) {
  return getDivisor(n, divisor, [](int64_t &d) { d++; });
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
  SmallVector<Operand> inputs;
  SmallVector<Operand> outputs;
};

RankedOperands getOperands(LinalgOp op) {
  auto findRankedOperands = [](OpOperandVector operandVector)
      -> SmallVector<RankedOperands::Operand> {
    SmallVector<RankedOperands::Operand> rankedOperands;
    for (auto op : operandVector) {
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

static llvm::SmallVector<int64_t> getNewShape(llvm::ArrayRef<int64_t> oldShape,
                                              size_t bitWidth, size_t maxSize) {

  llvm::SmallVector<int64_t> newShape(oldShape.begin(), oldShape.end());
  for (size_t i = 0, end = oldShape.size(); i < end; i++) {
    auto curSize = getSizeFromShape(newShape, bitWidth);
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

namespace mlir {
namespace linalg {
llvm::SmallVector<int64_t>
computeTileSizesForMemoryFootprintReduction(LinalgOp op,
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
} // namespace linalg
} // namespace mlir