//===- SCFToLuminous.cpp - SCF to Luminous conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.parallel operations into luminous
// dispatches.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToLuminous/SCFToLuminous.h"
#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// A pass converting SCF operations to OpenMP operations.
struct SCFToLuminousPass
    : public ConvertParallelLoopToLuminousDispatchBase<SCFToLuminousPass> {
  /// Pass entry point.
  void runOnFunction() override {
//    if (failed(applyPatterns(getOperation())))
//      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertParallelForToLuminousDispatchPass() {
  return std::make_unique<SCFToLuminousPass>();
}