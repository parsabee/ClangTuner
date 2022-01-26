//===- SCFToLuminous.cpp - SCF to Luminous conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.parallel operations into luminous
// dispatches. This file contains copies of some static functions and types
// from mlir/lib/Dialect/Async/Transform/AsyncParallelFor.cpp
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToLuminous/SCFToLuminous.h"
#include "../../../lib/Dialect/Async/Transforms/PassDetail.h" /* cloneConstantsIntoTheRegion */
#include "../PassDetail.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::async;
using namespace mlir::luminous;
using namespace mlir::linalg;

static constexpr char luminousModuleSymbol[] = "device_module";
static constexpr char luminousAsyncFnSymbol[] = "async_fn";

/// Copied from mlir/lib/Dialect/Async/Transform/AsyncParallelFor.cpp
struct AsyncFunctionType {
  FunctionType type;
  llvm::SmallVector<Value> captures;
};

struct AsyncDispatchFunction {
  LuminousFuncOp func;
  llvm::SmallVector<Value> captures;
};

static auto filterLinalgOps(scf::ParallelOp parallelOp) {
  return llvm::make_filter_range(
      parallelOp, [](Operation &op) { return isa<linalg::LinalgOp>(op); });
}

static SmallVector<Value, 4> getCaptures(scf::ParallelOp op) {
  llvm::SetVector<Value> capturesSet;
  getUsedValuesDefinedAbove(op.region(), op.region(), capturesSet);
  return llvm::to_vector<4>(capturesSet);
}

static AsyncFunctionType getDispatchFunctionType(scf::ParallelOp op,
                                                 PatternRewriter &rewriter) {
  // Values implicitly captured by the parallel operation.
  llvm::SetVector<Value> capturesSet;
  auto loopBodyArgs = op.getLoopBody().getArguments();
  capturesSet.insert(loopBodyArgs.begin(), loopBodyArgs.end());
  getUsedValuesDefinedAbove(op.region(), op.region(), capturesSet);
  auto captures = llvm::to_vector<6>(capturesSet);
  auto inputs = llvm::to_vector<6>(
      llvm::map_range(captures, [](Value &val) { return val.getType(); }));
  return {rewriter.getFunctionType(inputs, TypeRange()), std::move(captures)};
}

static LuminousModuleOp findLuminousModule(ModuleOp module) {
  auto *op = module.lookupSymbol(luminousModuleSymbol);
  assert(op && "couldn't find luminous module");
  return cast<LuminousModuleOp>(op);
}

static LuminousModuleOp createLuminousModule(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  ImplicitLocOpBuilder b(module.getLoc(), OpBuilder{ctx});
  auto luminousModule = b.create<LuminousModuleOp>(luminousModuleSymbol);
  module->setAttr(LuminousDialect::getContainerModuleAttrName(),
                  UnitAttr::get(ctx));

  // Insert luminous module into the module symbol table and assign it unique
  // name.
  SymbolTable symbolTable(module);
  symbolTable.insert(luminousModule);

  return luminousModule;
}

/// Finds the LuminousModuleOp in `op's parent module if it exists,
/// otherwise creates a new LuminousModuleOp.
static LuminousModuleOp getLuminousModule(scf::ParallelOp op) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (module->hasAttr(LuminousDialect::getContainerModuleAttrName()))
    return findLuminousModule(module);

  return createLuminousModule(module);
}

static AsyncDispatchFunction
getAsyncDispatchFunction(scf::ParallelOp op, PatternRewriter &rewriter) {

  OpBuilder::InsertionGuard guard(rewriter);
  ModuleOp module = op->getParentOfType<ModuleOp>();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  // Make sure that all constants will be inside the parallel operation body to
  // reduce the number of parallel compute function arguments.
  cloneConstantsIntoTheRegion(op.getLoopBody(), rewriter);

  auto luminousModule = getLuminousModule(op);
  rewriter.setInsertionPointToStart(luminousModule.getBody());

  auto parallelFuncType = getDispatchFunctionType(op, rewriter);
  auto &type = parallelFuncType.type;
  auto luminousFunc =
      rewriter.create<LuminousFuncOp>(op.getLoc(), luminousAsyncFnSymbol, type);

  // Create function entry block.
  //  Block *block = b.createBlock(&luminousFunc.getBody(),
  //  luminousFunc.begin(),
  //                               type.getInputs());

  //  block.addArguments(type.getInputs());
  //  b.setInsertionPointToStart(&block);
  llvm::SetVector<Value> capturesSet;
  getUsedValuesDefinedAbove(op.region(), op.region(), capturesSet);
  rewriter.inlineRegionBefore(op.getLoopBody(), luminousFunc.getBody(),
                              luminousFunc.getBody().end());
  Block *block = &luminousFunc.body().back();
  b.setInsertionPointToEnd(block);
  //  block->addArguments()
  // Copy the body of the parallel op into the inner-most loop.
  //  BlockAndValueMapping mapping;
  //  for (auto &operation : op.getLoopBody().getOps())
  //    if (!isa<scf::YieldOp>(operation)) {
  //      auto *clone = b.clone(operation, mapping);
  //      mapping.map(op.getResults(), clone->getResults());
  //      mapping.map(op.getOperands(), clone->getOperands());
  //    }
  //  b.setInsertionPointToEnd(block);
  b.create<luminous::ReturnOp>();
  block->dump();
  luminousModule.dump();
  return {luminousFunc, std::move(parallelFuncType.captures)};
}

// Dispatch parallel compute functions by submitting all async compute tasks
// from a simple for loop in the caller thread.
static void doSequantialDispatch(ImplicitLocOpBuilder &b,
                                 PatternRewriter &rewriter,
                                 AsyncDispatchFunction &asyncFunction,
                                 scf::ParallelOp op, Value blockSize,
                                 Value blockCount,
                                 const SmallVector<Value> &tripCounts) {
  MLIRContext *ctx = op->getContext();

  auto &func = asyncFunction.func;

  Value c0 = b.create<ConstantIndexOp>(0);
  Value c1 = b.create<ConstantIndexOp>(1);

  // Create an async.group to wait on all async tokens from the concurrent
  // execution of multiple parallel compute function. First block will be
  // executed synchronously in the caller thread.
  Value groupSize = b.create<SubIOp>(blockCount, c1);
  Value group = b.create<CreateGroupOp>(GroupType::get(ctx), groupSize);

  // Call parallel compute function for all blocks.
  using LoopBodyBuilder =
      std::function<void(OpBuilder &, Location, Value, ValueRange)>;

  // Induction variable is the index of the block: [0, blockCount).
  LoopBodyBuilder loopBuilder = [&](OpBuilder &loopBuilder, Location loc,
                                    Value iv, ValueRange args) {
    ImplicitLocOpBuilder nb(loc, loopBuilder);
    auto dispatch =
        nb.create<DispatchOp>(func, ValueRange(), asyncFunction.captures);
    nb.create<AddToGroupOp>(rewriter.getIndexType(), dispatch.asyncToken(),
                            group);
    nb.create<scf::YieldOp>();
  };

  // Iterate over all compute blocks and launch parallel compute operations.
  b.create<scf::ForOp>(c0, blockCount, c1, ValueRange(), loopBuilder);

  // Wait for the completion of all async compute operations.
  b.create<AwaitAllOp>(group);
}

namespace mlir {
namespace luminous {
struct ConversionTarget : public ::mlir::ConversionTarget {
  ConversionTarget(MLIRContext &ctx) : ::mlir::ConversionTarget(ctx) {
    addLegalDialect<LuminousDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, async::AsyncDialect>();
    this->addIllegalOp<scf::ParallelOp>();
  }
};
} // namespace luminous
} // namespace mlir

namespace {

struct LuminousDispatchParallelRewrite
    : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override;
};

/// A pass converting SCF operations to OpenMP operations.
struct SCFToLuminousPass
    : public ConvertParallelLoopToLuminousDispatchBase<SCFToLuminousPass> {
  /// Pass entry point.
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LuminousDispatchParallelRewrite>(ctx);
    mlir::luminous::ConversionTarget target(*ctx);
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, frozen)))
      signalPassFailure();
  }
};

LogicalResult LuminousDispatchParallelRewrite::matchAndRewrite(
    scf::ParallelOp op, PatternRewriter &rewriter) const {
  // TODO: add an attribute to the parallel op and check its presence here
  // We do not currently support rewrite for parallel op with reductions.

  auto module = op->getParentOfType<ModuleOp>();

  if (op.getNumReductions() != 0)
    return failure();

  Region &parallelLoopBody = op.getLoopBody();
  auto captures = getCaptures(op);
  auto launchOp = rewriter.create<LaunchOp>(
      op.getLoc(), op.upperBound(), op.step(), std::move(captures));
  // Replace SCF yield with luminous return.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(op.getBody());
    assert(llvm::hasSingleElement(op.region()) &&
           "expected scf.parallel to have one block");
    rewriter.replaceOpWithNewOp<luminous::ReturnOp>(
        op.getBody()->getTerminator(), ValueRange());
  }
  Block &block = launchOp.body().front();
  {
    ImplicitLocOpBuilder b(launchOp.getLoc(), rewriter);
    OpBuilder::InsertionGuard guard(b);

    //    b.setInsertionPointToEnd(&block);
    //    b.create<luminous::ReturnOp>();
    b.setInsertionPointToStart(&block);

    BlockAndValueMapping mapping;
    mapping.map(&parallelLoopBody.front(), &block);

    for (auto &operation : parallelLoopBody.getOps())
      b.clone(operation, mapping);
  }
//  for (auto &use: op->getUses())

  //  op->erase();
  //  for (auto argMap :
  //       llvm::zip(op.getLoopBody().getOps(), block.getArguments())) {
  //    auto &loopArg = std::get<0>(argMap);
  //    auto &blockArg = std::get<1>(argMap);
  //  }

  //
  //  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  //
  //  auto launchOp = rewriter.create<luminous::LaunchOp>(
  //      op.getLoc(), op.upperBound(), op.step(), )
  //
  //                  // Compute trip count for each loop induction variable:
  //                  //   tripCount = ceil_div(upperBound - lowerBound, step);
  //                  SmallVector<Value>
  //                      tripCounts(op.getNumLoops());
  //  for (size_t i = 0; i < op.getNumLoops(); ++i) {
  //    auto lb = op.lowerBound()[i];
  //    auto ub = op.upperBound()[i];
  //    auto step = op.step()[i];
  //    auto range = b.create<SubIOp>(ub, lb);
  //    tripCounts[i] = b.create<SignedCeilDivIOp>(range, step);
  //  }
  //
  //  // Compute a product of trip counts to get the 1-dimensional iteration
  //  space
  //  // for the scf.parallel operation.
  //  Value tripCount = tripCounts[0];
  //  for (size_t i = 1; i < tripCounts.size(); ++i)
  //    tripCount = b.create<MulIOp>(tripCount, tripCounts[i]);
  //
  //  // Compute balanced block size for the estimated block count.
  //  Value blockSize = b.create<ConstantIndexOp>(1);
  //
  //  auto asyncFunc = getAsyncDispatchFunction(op, rewriter);
  //  b.setInsertionPointAfter(op);
  //  doSequantialDispatch(b, rewriter, asyncFunc, op, blockSize, tripCount,
  //                       tripCounts);
  //  rewriter.eraseOp(op);

  module->dump();
  return success();
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertParallelForToLuminousDispatchPass() {
  return std::make_unique<SCFToLuminousPass>();
}