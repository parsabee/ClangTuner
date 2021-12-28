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
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"


using namespace mlir;
using namespace mlir::async;

struct ParallelComputeFunction {
  FuncOp func;
  llvm::SmallVector<Value> captures;
};

// Create a parallel compute fuction from the parallel operation.
static ParallelComputeFunction
createParallelComputeFunction(scf::ParallelOp op, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  ModuleOp module = op->getParentOfType<ModuleOp>();

  // Make sure that all constants will be inside the parallel operation body to
  // reduce the number of parallel compute function arguments.
  cloneConstantsIntoTheRegion(op.getLoopBody(), rewriter);

  ParallelComputeFunctionType computeFuncType =
      getParallelComputeFunctionType(op, rewriter);

  FunctionType type = computeFuncType.type;
  FuncOp func = FuncOp::create(op.getLoc(), "parallel_compute_fn", type);
  func.setPrivate();

  // Insert function into the module symbol table and assign it unique name.
  SymbolTable symbolTable(module);
  symbolTable.insert(func);
  rewriter.getListener()->notifyOperationInserted(func);

  // Create function entry block.
  Block *block = b.createBlock(&func.getBody(), func.begin(), type.getInputs());
  b.setInsertionPointToEnd(block);

  unsigned offset = 0; // argument offset for arguments decoding

  // Returns `numArguments` arguments starting from `offset` and updates offset
  // by moving forward to the next argument.
  auto getArguments = [&](unsigned numArguments) -> ArrayRef<Value> {
    auto args = block->getArguments();
    auto slice = args.drop_front(offset).take_front(numArguments);
    offset += numArguments;
    return {slice.begin(), slice.end()};
  };

  // Block iteration position defined by the block index and size.
  Value blockIndex = block->getArgument(offset++);
  Value blockSize = block->getArgument(offset++);

  // Constants used below.
  Value c0 = b.create<ConstantIndexOp>(0);
  Value c1 = b.create<ConstantIndexOp>(1);

  // Multi-dimensional parallel iteration space defined by the loop trip counts.
  ArrayRef<Value> tripCounts = getArguments(op.getNumLoops());

  // Compute a product of trip counts to get the size of the flattened
  // one-dimensional iteration space.
  Value tripCount = tripCounts[0];
  for (unsigned i = 1; i < tripCounts.size(); ++i)
    tripCount = b.create<MulIOp>(tripCount, tripCounts[i]);

  // Parallel operation lower bound and step.
  ArrayRef<Value> lowerBound = getArguments(op.getNumLoops());
  offset += op.getNumLoops(); // skip upper bound arguments
  ArrayRef<Value> step = getArguments(op.getNumLoops());

  // Remaining arguments are implicit captures of the parallel operation.
  ArrayRef<Value> captures = getArguments(block->getNumArguments() - offset);

  // Find one-dimensional iteration bounds: [blockFirstIndex, blockLastIndex]:
  //   blockFirstIndex = blockIndex * blockSize
  Value blockFirstIndex = b.create<MulIOp>(blockIndex, blockSize);

  // The last one-dimensional index in the block defined by the `blockIndex`:
  //   blockLastIndex = max(blockFirstIndex + blockSize, tripCount) - 1
  Value blockEnd0 = b.create<AddIOp>(blockFirstIndex, blockSize);
  Value blockEnd1 = b.create<CmpIOp>(CmpIPredicate::sge, blockEnd0, tripCount);
  Value blockEnd2 = b.create<SelectOp>(blockEnd1, tripCount, blockEnd0);
  Value blockLastIndex = b.create<SubIOp>(blockEnd2, c1);

  // Convert one-dimensional indices to multi-dimensional coordinates.
  auto blockFirstCoord = delinearize(b, blockFirstIndex, tripCounts);
  auto blockLastCoord = delinearize(b, blockLastIndex, tripCounts);

  // Compute loops upper bounds derived from the block last coordinates:
  //   blockEndCoord[i] = blockLastCoord[i] + 1
  //
  // Block first and last coordinates can be the same along the outer compute
  // dimension when inner compute dimension contains multiple blocks.
  SmallVector<Value> blockEndCoord(op.getNumLoops());
  for (size_t i = 0; i < blockLastCoord.size(); ++i)
    blockEndCoord[i] = b.create<AddIOp>(blockLastCoord[i], c1);

  // Construct a loop nest out of scf.for operations that will iterate over
  // all coordinates in [blockFirstCoord, blockLastCoord] range.
  using LoopBodyBuilder =
      std::function<void(OpBuilder &, Location, Value, ValueRange)>;
  using LoopNestBuilder = std::function<LoopBodyBuilder(size_t loopIdx)>;

  // Parallel region induction variables computed from the multi-dimensional
  // iteration coordinate using parallel operation bounds and step:
  //
  //   computeBlockInductionVars[loopIdx] =
  //       lowerBound[loopIdx] + blockCoord[loopIdx] * step[loopDdx]
  SmallVector<Value> computeBlockInductionVars(op.getNumLoops());

  // We need to know if we are in the first or last iteration of the
  // multi-dimensional loop for each loop in the nest, so we can decide what
  // loop bounds should we use for the nested loops: bounds defined by compute
  // block interval, or bounds defined by the parallel operation.
  //
  // Example: 2d parallel operation
  //                   i   j
  //   loop sizes:   [50, 50]
  //   first coord:  [25, 25]
  //   last coord:   [30, 30]
  //
  // If `i` is equal to 25 then iteration over `j` should start at 25, when `i`
  // is between 25 and 30 it should start at 0. The upper bound for `j` should
  // be 50, except when `i` is equal to 30, then it should also be 30.
  //
  // Value at ith position specifies if all loops in [0, i) range of the loop
  // nest are in the first/last iteration.
  SmallVector<Value> isBlockFirstCoord(op.getNumLoops());
  SmallVector<Value> isBlockLastCoord(op.getNumLoops());

  // Builds inner loop nest inside async.execute operation that does all the
  // work concurrently.
  LoopNestBuilder workLoopBuilder = [&](size_t loopIdx) -> LoopBodyBuilder {
    return [&, loopIdx](OpBuilder &nestedBuilder, Location loc, Value iv,
                        ValueRange args) {
      ImplicitLocOpBuilder nb(loc, nestedBuilder);

      // Compute induction variable for `loopIdx`.
      computeBlockInductionVars[loopIdx] = nb.create<AddIOp>(
          lowerBound[loopIdx], nb.create<MulIOp>(iv, step[loopIdx]));

      // Check if we are inside first or last iteration of the loop.
      isBlockFirstCoord[loopIdx] =
          nb.create<CmpIOp>(CmpIPredicate::eq, iv, blockFirstCoord[loopIdx]);
      isBlockLastCoord[loopIdx] =
          nb.create<CmpIOp>(CmpIPredicate::eq, iv, blockLastCoord[loopIdx]);

      // Check if the previous loop is in its first or last iteration.
      if (loopIdx > 0) {
        isBlockFirstCoord[loopIdx] = nb.create<AndOp>(
            isBlockFirstCoord[loopIdx], isBlockFirstCoord[loopIdx - 1]);
        isBlockLastCoord[loopIdx] = nb.create<AndOp>(
            isBlockLastCoord[loopIdx], isBlockLastCoord[loopIdx - 1]);
      }

      // Keep building loop nest.
      if (loopIdx < op.getNumLoops() - 1) {
        // Select nested loop lower/upper bounds depending on out position in
        // the multi-dimensional iteration space.
        auto lb = nb.create<SelectOp>(isBlockFirstCoord[loopIdx],
                                      blockFirstCoord[loopIdx + 1], c0);

        auto ub = nb.create<SelectOp>(isBlockLastCoord[loopIdx],
                                      blockEndCoord[loopIdx + 1],
                                      tripCounts[loopIdx + 1]);

        nb.create<scf::ForOp>(lb, ub, c1, ValueRange(),
                              workLoopBuilder(loopIdx + 1));
        nb.create<scf::YieldOp>(loc);
        return;
      }

      // Copy the body of the parallel op into the inner-most loop.
      BlockAndValueMapping mapping;
      mapping.map(op.getInductionVars(), computeBlockInductionVars);
      mapping.map(computeFuncType.captures, captures);

      for (auto &bodyOp : op.getLoopBody().getOps())
        nb.clone(bodyOp, mapping);
    };
  };

  b.create<scf::ForOp>(blockFirstCoord[0], blockEndCoord[0], c1, ValueRange(),
                       workLoopBuilder(0));
  b.create<ReturnOp>(ValueRange());

  return {func, std::move(computeFuncType.captures)};
}

// Dispatch parallel compute functions by submitting all async compute tasks
// from a simple for loop in the caller thread.
static void
doSequantialDispatch(ImplicitLocOpBuilder &b, PatternRewriter &rewriter,
                     ParallelComputeFunction &parallelComputeFunction,
                     scf::ParallelOp op, Value blockSize, Value blockCount,
                     const SmallVector<Value> &tripCounts) {
  MLIRContext *ctx = op->getContext();

  FuncOp compute = parallelComputeFunction.func;

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

  // Returns parallel compute function operands to process the given block.
  auto computeFuncOperands = [&](Value blockIndex) -> SmallVector<Value> {
    SmallVector<Value> computeFuncOperands = {blockIndex, blockSize};
    computeFuncOperands.append(tripCounts);
    computeFuncOperands.append(op.lowerBound().begin(), op.lowerBound().end());
    computeFuncOperands.append(op.upperBound().begin(), op.upperBound().end());
    computeFuncOperands.append(op.step().begin(), op.step().end());
    computeFuncOperands.append(parallelComputeFunction.captures);
    return computeFuncOperands;
  };

  // Induction variable is the index of the block: [0, blockCount).
  LoopBodyBuilder loopBuilder = [&](OpBuilder &loopBuilder, Location loc,
                                    Value iv, ValueRange args) {
    ImplicitLocOpBuilder nb(loc, loopBuilder);

    // Call parallel compute function inside the async.execute region.
    auto executeBodyBuilder = [&](OpBuilder &executeBuilder,
                                  Location executeLoc, ValueRange executeArgs) {
      executeBuilder.create<CallOp>(executeLoc, compute.sym_name(),
                                    compute.getCallableResults(),
                                    computeFuncOperands(iv));
      executeBuilder.create<async::YieldOp>(executeLoc, ValueRange());
    };

    // Create async.execute operation to launch parallel computate function.
    auto execute = nb.create<ExecuteOp>(TypeRange(), ValueRange(), ValueRange(),
                                        executeBodyBuilder);
    nb.create<AddToGroupOp>(rewriter.getIndexType(), execute.token(), group);
    nb.create<scf::YieldOp>();
  };

  // Iterate over all compute blocks and launch parallel compute operations.
  b.create<scf::ForOp>(c1, blockCount, c1, ValueRange(), loopBuilder);

  // Call parallel compute function for the first block in the caller thread.
  b.create<CallOp>(compute.sym_name(), compute.getCallableResults(),
                   computeFuncOperands(c0));

  // Wait for the completion of all async compute operations.
  b.create<AwaitAllOp>(group);
}

namespace {

struct LuminousDispatchParallelRewrite
    : public OpRewritePattern<scf::ParallelOp> {
public:
  LuminousDispatchParallelRewrite(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override;
};

/// A pass converting SCF operations to OpenMP operations.
struct SCFToLuminousPass
    : public ConvertParallelLoopToLuminousDispatchBase<SCFToLuminousPass> {
  /// Pass entry point.
  void runOnFunction() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LuminousDispatchParallelRewrite>(ctx);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

LogicalResult LuminousDispatchParallelRewrite::matchAndRewrite(
    scf::ParallelOp op, PatternRewriter &rewriter) const {

  // TODO: add an attribute to the parallel op and check its presence here

  // We do not currently support rewrite for parallel op with reductions.
  if (op.getNumReductions() != 0)
    return failure();

  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  // Compute trip count for each loop induction variable:
  //   tripCount = ceil_div(upperBound - lowerBound, step);
  SmallVector<Value> tripCounts(op.getNumLoops());
  for (size_t i = 0; i < op.getNumLoops(); ++i) {
    auto lb = op.lowerBound()[i];
    auto ub = op.upperBound()[i];
    auto step = op.step()[i];
    auto range = b.create<SubIOp>(ub, lb);
    tripCounts[i] = b.create<SignedCeilDivIOp>(range, step);
  }

  // Compute a product of trip counts to get the 1-dimensional iteration space
  // for the scf.parallel operation.
  Value tripCount = tripCounts[0];
  for (size_t i = 1; i < tripCounts.size(); ++i)
    tripCount = b.create<MulIOp>(tripCount, tripCounts[i]);

  // Compute the parallel block size and dispatch concurrent tasks computing
  // results for each block.
  auto dispatch = [&](OpBuilder &nestedBuilder, Location loc) {
    ImplicitLocOpBuilder nb(loc, nestedBuilder);

    // With large number of threads the value of creating many compute blocks
    // is reduced because the problem typically becomes memory bound. For small
    // number of threads it helps with stragglers.
    float overshardingFactor = numWorkerThreads <= 4    ? 8.0
                               : numWorkerThreads <= 8  ? 4.0
                               : numWorkerThreads <= 16 ? 2.0
                               : numWorkerThreads <= 32 ? 1.0
                               : numWorkerThreads <= 64 ? 0.8
                                                        : 0.6;

    // Do not overload worker threads with too many compute blocks.
    Value maxComputeBlocks = b.create<ConstantIndexOp>(
        std::max(1, static_cast<int>(numWorkerThreads * overshardingFactor)));

    // Target block size from the pass parameters.
    Value targetComputeBlock = b.create<ConstantIndexOp>(targetBlockSize);

    // Compute parallel block size from the parallel problem size:
    //   blockSize = min(tripCount,
    //                   max(ceil_div(tripCount, maxComputeBlocks),
    //                       targetComputeBlock))
    Value bs0 = b.create<SignedCeilDivIOp>(tripCount, maxComputeBlocks);
    Value bs1 = b.create<CmpIOp>(CmpIPredicate::sge, bs0, targetComputeBlock);
    Value bs2 = b.create<SelectOp>(bs1, bs0, targetComputeBlock);
    Value bs3 = b.create<CmpIOp>(CmpIPredicate::sle, tripCount, bs2);
    Value blockSize0 = b.create<SelectOp>(bs3, tripCount, bs2);
    Value blockCount0 = b.create<SignedCeilDivIOp>(tripCount, blockSize0);

    // Compute balanced block size for the estimated block count.
    Value blockSize = b.create<SignedCeilDivIOp>(tripCount, blockCount0);
    Value blockCount = b.create<SignedCeilDivIOp>(tripCount, blockSize);

    // Create a parallel compute function that takes a block id and computes the
    // parallel operation body for a subset of iteration space.
    ParallelComputeFunction parallelComputeFunction =
        createParallelComputeFunction(op, rewriter);

    doSequantialDispatch(b, rewriter, parallelComputeFunction, op, blockSize,
                        blockCount, tripCounts);

    nb.create<scf::YieldOp>();
  };

}

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertParallelForToLuminousDispatchPass() {
  return std::make_unique<SCFToLuminousPass>();
}