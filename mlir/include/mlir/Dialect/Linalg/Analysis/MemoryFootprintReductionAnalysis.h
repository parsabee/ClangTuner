//===- MemoryFootprintReductionAnalysis.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ANALYSIS_MEMORYFOOTPRINTREDUCTIONANALYSIS_H_
#define MLIR_DIALECT_ANALYSIS_MEMORYFOOTPRINTREDUCTIONANALYSIS_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace linalg {

/// Encapsulates the information that we need for size reduction of a linalg op
class LinalgOpShape {
  SmallVector<int64_t, 4> opShape;
  const SmallVector<AffineMap, 4> indexingMaps;
  const SmallVector<int64_t, 4> bitWidths;

public:
  LinalgOpShape(SmallVector<int64_t, 4> s, ArrayRef<AffineMap> iMaps,
                SmallVector<int64_t, 4> bWidths);

  SmallVector<int64_t, 4> &getShape() { return opShape; }

  /// Computes the size of this linalg operation based on the sizes of its
  /// operands (which are calculated by looking at their corresponding
  /// indexingMap)
  size_t computeSize() const;
};

void reduceLinalgOpFootprintGreedily(LinalgOpShape &, size_t maxSize);

llvm::SmallVector<int64_t, 4> computeTileSizesForMemoryFootprintReduction(
    LinalgOp op, int64_t maxMemoryFootprint,
    std::function<void(LinalgOpShape &, int64_t)> reduceFn =
        reduceLinalgOpFootprintGreedily);

} // namespace linalg
} // namespace mlir
#endif