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
static size_t getSizeFromShape(llvm::ArrayRef<int64_t> shape,
                                      size_t elementBitWidth) {
  assert(elementBitWidth % 8 == 0 && "BitWidth has to be divisible by 8");
  auto elementSize = elementBitWidth / 8;
  return std::accumulate(shape.begin(), shape.end(), elementSize,
                         std::multiplies<>());
}

static SmallVector<int64_t, 4> getLinalgOpLoopShape(LinalgOp op) {
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  return shapeSizesToLoopsMap.compose(op.getStaticShape());
}

static int64_t getBitWidth(LinalgOp op) {
  for (auto operand: op->getOperands()) {
    if (auto shapedType = operand.getType().dyn_cast<ShapedType>()) {
      return shapedType.getElementTypeBitWidth();
    }
  }
  return -1;
}

namespace mlir {
namespace linalg {

size_t LinalgOpShape::computeSize() {
  return std::accumulate(indexingMaps.begin(), indexingMaps.end(), 0,
                         [this](size_t curSize, const AffineMap &curMap) {
                           auto curOperandsShape = curMap.compose(opShape);
                           return curSize +
                                  getSizeFromShape(curOperandsShape, bitWidth);
                         });
}

void reduceLinalgOpFootprintGreedily(LinalgOpShape &linalgOpShape,
                                     size_t maxSize) {
  auto &shape = linalgOpShape.getShape();
  for (size_t i = 0, end = shape.size(); i < end; i++) {
    auto curSize = linalgOpShape.computeSize();
    if (curSize < maxSize)
      return;
    int64_t reductionFactor = (curSize + maxSize - 1) / maxSize;
    if (shape[i] < reductionFactor)
      shape[i] = 1;
    else {
      auto divisor = getDivisorCeil(shape[i], reductionFactor);
      shape[i] /= divisor;
    }
  }
}

SmallVector<int64_t, 4> computeTileSizesForMemoryFootprintReduction(
    LinalgOp op, int64_t maxMemoryFootprint,
    std::function<void(LinalgOpShape &, int64_t)> reduceFn) {
  LinalgOpShape linalgOpShape(getLinalgOpLoopShape(op), op.getIndexingMaps(),
                              getBitWidth(op));
  reduceFn(linalgOpShape, maxMemoryFootprint);
  return linalgOpShape.getShape();
}
} // namespace linalg
} // namespace mlir