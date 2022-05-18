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

#include "PassDetail.h"
#include "mlir/Conversion/SCFToLuminous/SCFToLuminous.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/Dialect/Luminous/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::async;
using namespace mlir::luminous;
using namespace mlir::linalg;

static constexpr char luminousModuleSymbol[] = "device_module";
static constexpr char kernelOutliningVisited[] = "kerenels_outlined";

static const std::string getNextFunctionName(const std::string &name) {
  static constexpr char luminousAsyncFnSymbol[] = "async_fn";
  static unsigned fnId = 0;
  return luminousAsyncFnSymbol + name + std::to_string(fnId++);
}

namespace mlir {
namespace luminous {

namespace detail {
struct DispatchBlockImpl {
  std::string name;
  OpBuilder builder;
  LaunchOp launchOp;
  std::unique_ptr<Block> block;
  SmallVector<Value> args;
  SmallVector<Operation *> ops;
  SetVector<Value> opResults;
  BlockAndValueMapping cloningMap;
  DispatchBlockImpl(LaunchOp op, const std::string &name)
      : name(name), builder(op.getContext()), launchOp(op), block(new Block) {}
  Block *releaseBlock() { return block.release(); }
};

struct DispatchBlocksImpl {
  LaunchOp launchOp;
  using Blocks = std::vector<DispatchBlockImpl>;
  Blocks blocks;
  DispatchBlocksImpl(LaunchOp op) : launchOp(op) {}
};

} // namespace detail

/// Clones `op' and inserts it in the basic block; it keeps track of
/// the op for performing replacement with dispatch call.
void DispatchBlock::pushBack(Operation *op) {
  assert(op->getParentOp() == impl.launchOp && "can only push back operations "
                                               "within launch op");
  // handling operands
  for (auto operand : op->getOperands()) {
    // if the operands of the current op have been visited before then
    // continue, otherwise they are arguments to this block.
    if (impl.opResults.contains(operand))
      continue;

    impl.args.push_back(operand);
    auto blockArg =
        impl.block->addArgument(operand.getType(), operand.getLoc());
    impl.cloningMap.map(operand, blockArg);
  }

  // keeping track of ops results to determine whether the value is an
  // argument to the block or has been produced in the block
  for (auto result : op->getOpResults()) {
    assert(!impl.opResults.contains(result));
    impl.opResults.insert(result);
  }

  // keeping track of ops to later replace them with a dispatch call
  impl.ops.push_back(op);
  auto *clone = impl.builder.clone(*op, impl.cloningMap);
  impl.block->push_back(clone);
}

/// Creates a dispatch block implementation in the container and returns a
/// DispatchBlock wrapper of the implementation.
DispatchBlock DispatchBlocks::addNewBlock(const std::string &name) {
  impl.blocks.push_back(detail::DispatchBlockImpl(impl.launchOp, name));
  return DispatchBlock(impl.blocks.back());
}

/// Outlines the luminous function
static LuminousFuncOp outline(PatternRewriter &rewriter, Location loc,
                              LuminousModuleOp luminousModule, Block *block, const std::string &name) {

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(luminousModule.getBody()->getTerminator());

  // return type is void and arguments are the basic blocks arguments
  auto funcType =
      rewriter.getFunctionType(block->getArgumentTypes(), TypeRange());

  auto func =
      rewriter.create<LuminousFuncOp>(loc, getNextFunctionName(name), funcType);

  rewriter.setInsertionPoint(block, block->end());
  rewriter.create<luminous::ReturnOp>(loc);

  // transferring ownership of the block to the funcs region
  func.body().push_front(block);

  return func;
}

/// Replaces the ops in the target launch op with dispatch calls to the
/// outlined luminous functions
static Value replaceOpsWithDispatchCall(PatternRewriter &rewriter,
                                        LuminousFuncOp func,
                                        ValueRange dependencies,
                                        SmallVector<Operation *> &ops,
                                        SmallVector<Value> &args) {
  OpBuilder::InsertionGuard guard(rewriter);

  // Insert dispatch call after the last op in the dispatchable region
  auto *lastOp = ops.back();
  rewriter.setInsertionPoint(lastOp);
  auto dispatchOp =
      rewriter.create<DispatchOp>(lastOp->getLoc(), func, dependencies, args);

  // removing the dispatchable regions ops from the launch ops body
  for (auto *op : ops)
    rewriter.eraseOp(op);

  return dispatchOp;
}

/// Outlines kernels for each dispatchable blocks
/// this function consumes and destroys dispatchBlocks
void outlineKernels(PatternRewriter &rewriter, Location loc,
                    LuminousModuleOp luminousModule,
                    detail::DispatchBlocksImpl &dispatchBlocks) {
  auto &blocks = dispatchBlocks.blocks;
  Value dispOpToken = nullptr;

  // outlining the kernel and dispatching every block within dispatch blocks
  for (auto &block : blocks) {
    auto fn = outline(rewriter, loc, luminousModule, block.releaseBlock(), block.name);
    if (dispOpToken)
      dispOpToken = replaceOpsWithDispatchCall(rewriter, fn, {dispOpToken},
                                               block.ops, block.args);
    else
      dispOpToken =
          replaceOpsWithDispatchCall(rewriter, fn, {}, block.ops, block.args);
  }

  // waiting on the dispatches
  if (dispOpToken) {
    ImplicitLocOpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(dispOpToken);
    rewriter.create<AwaitOp>(loc, dispOpToken);
  }
}

static LuminousModuleOp findLuminousModule(ModuleOp module) {
  auto *op = module.lookupSymbol(luminousModuleSymbol);
  assert(op && isa<LuminousModuleOp>(op) && "couldn't find luminous module");
  return cast<LuminousModuleOp>(op);
}

static LuminousModuleOp createLuminousModule(PatternRewriter &rewriter,
                                             Location loc, ModuleOp module) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(module.getBody(), module.getBody()->begin());
  MLIRContext *ctx = module.getContext();
  auto luminousModule =
      rewriter.create<LuminousModuleOp>(loc, luminousModuleSymbol);
  module->setAttr(LuminousDialect::getContainerModuleAttrName(),
                  UnitAttr::get(ctx));
  return luminousModule;
}

/// Finds the LuminousModuleOp in `op's parent module if it exists,
/// otherwise creates a new LuminousModuleOp.
static LuminousModuleOp getLuminousModule(Operation *op, Location loc,
                                          PatternRewriter &rewriter) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (module->hasAttr(LuminousDialect::getContainerModuleAttrName()))
    return findLuminousModule(module);

  return createLuminousModule(rewriter, loc, module);
}

void defaultDispatchBuilderFn(LaunchOp launchOp,
                              DispatchBlocks &dispatchableBlocks) {
  auto &body = launchOp.body().back();
  for (auto &op : body) {
    if (!op.hasAttr(luminous::maxMemoryAttrName))
      continue;

    // add a new block to the dispatchable blocks and fill it with ops
    auto block = dispatchableBlocks.addNewBlock();
    block.pushBack(&op);
  }
}

} // namespace luminous
} // namespace mlir

namespace {

struct KernelOutliningRewritePattern : public OpRewritePattern<LaunchOp> {
  DispatchBuilderFn dispatchBuilderFn;
  KernelOutliningRewritePattern(MLIRContext *ctx, DispatchBuilderFn fn)
      : OpRewritePattern<LaunchOp>(ctx), dispatchBuilderFn(fn) {}
  LogicalResult matchAndRewrite(LaunchOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult applyPatterns(func::FuncOp funcOp, DispatchBuilderFn fn) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<KernelOutliningRewritePattern>(ctx, fn);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

/// A pass converting SCF operations to OpenMP operations.
struct LuminousKernelOutliningPass
    : public LuminousKernelOutliningBase<LuminousKernelOutliningPass> {
  DispatchBuilderFn dispatchBuilderFn;

  LuminousKernelOutliningPass(DispatchBuilderFn fn)
      : dispatchBuilderFn(std::move(fn)) {}

  /// Pass entry point.
  void runOnOperation() override {
    if (failed(applyPatterns(getOperation(), dispatchBuilderFn)))
      signalPassFailure();
  }
};

LogicalResult KernelOutliningRewritePattern::matchAndRewrite(
    LaunchOp op, PatternRewriter &rewriter) const {

  // stop if already visited
  if (op->hasAttr(kernelOutliningVisited))
    return failure();

  auto loc = op.getLoc();
  auto luminousModule = getLuminousModule(op, loc, rewriter);
  op->setAttr(kernelOutliningVisited, rewriter.getUnitAttr());
  mlir::luminous::detail::DispatchBlocksImpl dispatchBlocksImpl(op);
  DispatchBlocks dispatchableBlocks(dispatchBlocksImpl);
  dispatchBuilderFn(op, dispatchableBlocks);
  outlineKernels(rewriter, loc, luminousModule, dispatchBlocksImpl);

  return success();
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLuminousKernelOutliningPass(DispatchBuilderFn fn) {
  return std::make_unique<LuminousKernelOutliningPass>(fn);
}
