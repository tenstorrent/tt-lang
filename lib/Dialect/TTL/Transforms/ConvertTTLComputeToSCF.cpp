// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#define GEN_PASS_DEF_TTLLOWERTOLOOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttl {
namespace {

/// Apply an indexing map to the induction variables with canonicalization to
/// drop unused/duplicate operands and fold constants (mirrors linalg lowering).
static SmallVector<Value> applyIndexingMap(OpBuilder &b, Location loc,
                                           AffineMap map, ValueRange ivs) {
  SmallVector<Value> operands(ivs.begin(), ivs.end());
  if (operands.size() < map.getNumDims()) {
    operands.append(map.getNumDims() - operands.size(),
                    b.create<arith::ConstantIndexOp>(loc, 0));
  } else if (operands.size() > map.getNumDims()) {
    operands.resize(map.getNumDims());
  }

  AffineMap canonMap = map;
  affine::canonicalizeMapAndOperands(&canonMap, &operands);

  SmallVector<Value> mapped;
  mapped.reserve(canonMap.getNumResults());
  for (AffineExpr expr : canonMap.getResults()) {
    AffineMap single =
        AffineMap::get(canonMap.getNumDims(), canonMap.getNumSymbols(), expr);
    mapped.push_back(
        b.create<affine::AffineApplyOp>(loc, single, operands).getResult());
  }
  return mapped;
}

/// Generate loop body that processes tiles at given indices.
/// Extracts tiles from inputs, clones compute body, inserts results back.
static scf::ValueVector generateTileProcessing(OpBuilder &b, Location loc,
                                               ComputeOp op,
                                               ArrayRef<AffineMap> indexingMaps,
                                               ValueRange ivs,
                                               ValueRange iterArgs) {
  // Extract tiles from inputs at current mapped indices.
  SmallVector<Value> extractedInputs;
  for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
    SmallVector<Value> indices =
        applyIndexingMap(b, loc, indexingMaps[idx], ivs);
    Value tile = b.create<tensor::ExtractOp>(loc, input, indices);
    extractedInputs.push_back(tile);
  }

  // Extract tiles from outputs (iter_args) at current indices.
  SmallVector<Value> extractedOutputs;
  size_t numInputs = op.getInputs().size();
  for (auto [idx, output] : llvm::enumerate(iterArgs)) {
    SmallVector<Value> indices =
        applyIndexingMap(b, loc, indexingMaps[numInputs + idx], ivs);
    Value tile = b.create<tensor::ExtractOp>(loc, output, indices);
    extractedOutputs.push_back(tile);
  }

  // Clone body operations with block args mapped to extracted tiles.
  Block &bodyBlock = op.getBody().front();
  IRMapping mapping;
  for (auto [idx, arg] : llvm::enumerate(op.getInputs())) {
    mapping.map(bodyBlock.getArgument(idx), extractedInputs[idx]);
  }
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
    SmallVector<Value> indices =
        applyIndexingMap(b, loc, indexingMaps[numInputs + idx], ivs);
    Value updated =
        b.create<tensor::InsertOp>(loc, result, iterArgs[idx], indices);
    results.push_back(updated);
  }
  return results;
}

struct LowerComputeToLoops : OpRewritePattern<ComputeOp> {
  using OpRewritePattern<ComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComputeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(op.getIndexingMaps().size());
    for (Attribute attr : op.getIndexingMaps()) {
      indexingMaps.push_back(cast<AffineMapAttr>(attr).getValue());
    }

    SmallVector<Range> iterDomain = op.getIterationDomain(rewriter);
    if (iterDomain.empty()) {
      return failure();
    }

    SmallVector<StringRef> iteratorTypes;
    iteratorTypes.reserve(op.getIteratorTypes().size());
    for (Attribute attr : op.getIteratorTypes()) {
      iteratorTypes.push_back(cast<StringAttr>(attr).getValue());
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

    // Reduction dimensions are not batched.
    SmallVector<int64_t> effectiveBatch;
    effectiveBatch.reserve(batchSize.size());
    for (auto [idx, itType] : llvm::enumerate(iteratorTypes)) {
      effectiveBatch.push_back(itType == "reduction" ? 1 : batchSize[idx]);
    }

    // Check if all batch sizes are 1 (simple case, no batching).
    bool isUnitBatch =
        llvm::all_of(effectiveBatch, [](int64_t n) { return n == 1; });

    // Build loop bounds from iteration domain.
    SmallVector<Value> lowerBounds, upperBounds, steps;
    for (auto [idx, range] : llvm::enumerate(iterDomain)) {
      Value lb = getValueOrCreateConstantIndexOp(rewriter, loc, range.offset);
      Value ub = getValueOrCreateConstantIndexOp(rewriter, loc, range.size);
      // Use batch size as step when batching, otherwise step by 1.
      int64_t stepVal = isUnitBatch ? 1 : effectiveBatch[idx];
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
            return generateTileProcessing(b, loc, op, indexingMaps, ivs,
                                          iterArgs);
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
              Value batchConst =
                  b.create<arith::ConstantIndexOp>(loc, effectiveBatch[i]);
              Value remaining =
                  b.create<arith::SubIOp>(loc, upperBounds[i], outerIvs[i]);
              Value useRemaining = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sle, remaining, batchConst);
              Value boundedUb = b.create<arith::SelectOp>(
                  loc, useRemaining, remaining, batchConst);
              innerUbs.push_back(boundedUb);
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
                  return generateTileProcessing(ib, iloc, op, indexingMaps,
                                                tileIvs, innerIterArgs);
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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    RewritePatternSet patterns(func.getContext());
    patterns.add<LowerComputeToLoops>(func.getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(func, frozen))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTTLLowerToLoops() {
  return std::make_unique<TTLLowerToLoopsPass>();
}

} // namespace mlir::tt::ttl
