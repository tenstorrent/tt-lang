// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#define GEN_PASS_DEF_TTLLOWERTOLOOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttl {
namespace {

/// Generate loop body that processes tiles at given indices.
/// Extracts tiles from inputs, clones compute body, inserts results back.
static scf::ValueVector generateTileProcessing(OpBuilder &b, Location loc,
                                               ComputeOp op, ValueRange ivs,
                                               ValueRange iterArgs) {
  // Extract tiles from inputs at current indices.
  SmallVector<Value> extractedInputs;
  for (Value input : op.getInputs()) {
    Value tile = b.create<tensor::ExtractOp>(loc, input, ivs);
    extractedInputs.push_back(tile);
  }

  // Extract tiles from outputs (iter_args) at current indices.
  SmallVector<Value> extractedOutputs;
  for (Value output : iterArgs) {
    Value tile = b.create<tensor::ExtractOp>(loc, output, ivs);
    extractedOutputs.push_back(tile);
  }

  // Clone body operations with block args mapped to extracted tiles.
  Block &bodyBlock = op.getBody().front();
  IRMapping mapping;
  for (auto [idx, arg] : llvm::enumerate(op.getInputs())) {
    mapping.map(bodyBlock.getArgument(idx), extractedInputs[idx]);
  }
  size_t numInputs = op.getInputs().size();
  for (auto [idx, arg] : llvm::enumerate(op.getOutputs())) {
    mapping.map(bodyBlock.getArgument(numInputs + idx), extractedOutputs[idx]);
  }
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    b.clone(bodyOp, mapping);
  }

  // Get yielded values from terminator.
  auto yieldOp = cast<YieldOp>(bodyBlock.getTerminator());
  SmallVector<Value> results;
  for (auto [idx, yieldVal] : llvm::enumerate(yieldOp.getValues())) {
    Value result = mapping.lookupOrDefault(yieldVal);
    // Insert result tile back into output tensor.
    Value updated =
        b.create<tensor::InsertOp>(loc, result, iterArgs[idx], ivs);
    results.push_back(updated);
  }
  return results;
}

struct LowerComputeToLoops : OpRewritePattern<ComputeOp> {
  using OpRewritePattern<ComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComputeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Range> iterDomain = op.getIterationDomain(rewriter);
    if (iterDomain.empty()) {
      return failure();
    }

    // Get tile batch size per dimension.
    SmallVector<int64_t> batchSize;
    if (auto attr = op.getTileBatchSize()) {
      batchSize.assign(attr->begin(), attr->end());
    } else {
      // Default: 1 tile per iteration for each dimension
      auto outTy = cast<RankedTensorType>(op.getOutputs().front().getType());
      batchSize.assign(outTy.getRank(), 1);
    }

    // Check if all batch sizes are 1 (simple case, no batching).
    bool isUnitBatch = llvm::all_of(batchSize, [](int64_t n) { return n == 1; });

    // Build loop bounds from iteration domain.
    SmallVector<Value> lowerBounds, upperBounds, steps;
    for (auto [idx, range] : llvm::enumerate(iterDomain)) {
      Value lb = getValueOrCreateConstantIndexOp(rewriter, loc, range.offset);
      Value ub = getValueOrCreateConstantIndexOp(rewriter, loc, range.size);
      // Use batch size as step when batching, otherwise step by 1.
      int64_t stepVal = isUnitBatch ? 1 : batchSize[idx];
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, stepVal);
      lowerBounds.push_back(lb);
      upperBounds.push_back(ub);
      steps.push_back(step);
    }

    // Initial values for iter_args are the output tensors.
    SmallVector<Value> initValues(op.getOutputs());

    if (isUnitBatch) {
      // Simple case: 1 tile per iteration, use buildLoopNest directly.
      scf::LoopNest loopNest = scf::buildLoopNest(
          rewriter, loc, lowerBounds, upperBounds, steps, initValues,
          [&](OpBuilder &b, Location loc, ValueRange ivs,
              ValueRange iterArgs) -> scf::ValueVector {
            return generateTileProcessing(b, loc, op, ivs, iterArgs);
          });
      rewriter.replaceOp(op, loopNest.results);
    } else {
      // Batched case: generate outer loops with step=num_tiles, then inner
      // loops to iterate over tiles within each batch.
      scf::LoopNest outerLoops = scf::buildLoopNest(
          rewriter, loc, lowerBounds, upperBounds, steps, initValues,
          [&](OpBuilder &b, Location loc, ValueRange outerIvs,
              ValueRange outerIterArgs) -> scf::ValueVector {
            // Build inner loops to iterate over tiles within the batch.
            // For each dimension, add an inner loop from 0 to batch_size.
            SmallVector<Value> innerLbs, innerUbs, innerSteps;
            for (size_t i = 0; i < batchSize.size(); ++i) {
              innerLbs.push_back(b.create<arith::ConstantIndexOp>(loc, 0));
              innerUbs.push_back(
                  b.create<arith::ConstantIndexOp>(loc, batchSize[i]));
              innerSteps.push_back(b.create<arith::ConstantIndexOp>(loc, 1));
            }

            scf::LoopNest innerLoops = scf::buildLoopNest(
                b, loc, innerLbs, innerUbs, innerSteps, outerIterArgs,
                [&](OpBuilder &ib, Location iloc, ValueRange innerIvs,
                    ValueRange innerIterArgs) -> scf::ValueVector {
                  // Compute actual tile indices: outer_iv + inner_iv.
                  SmallVector<Value> tileIvs;
                  for (size_t i = 0; i < outerIvs.size(); ++i) {
                    Value idx = ib.create<arith::AddIOp>(iloc, outerIvs[i],
                                                         innerIvs[i]);
                    tileIvs.push_back(idx);
                  }
                  return generateTileProcessing(ib, iloc, op, tileIvs,
                                                innerIterArgs);
                });
            return innerLoops.results;
          });
      rewriter.replaceOp(op, outerLoops.results);
    }
    return success();
  }
};

struct TTLLowerToLoopsPass
    : public ::impl::TTLLowerToLoopsBase<TTLLowerToLoopsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    patterns.add<LowerComputeToLoops>(func.getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTTLLowerToLoops() {
  return std::make_unique<TTLLowerToLoopsPass>();
}

} // namespace mlir::tt::ttl
