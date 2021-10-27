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

llvm::SmallVector<int64_t>
computeTileSizesForMemoryFootprintReduction(LinalgOp op,
                                            int64_t maxMemoryFootprint);

}
} // namespace mlir
#endif