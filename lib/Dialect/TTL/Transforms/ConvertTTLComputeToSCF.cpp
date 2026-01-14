// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "ttl-lower-to-loops"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLLOWERTOLOOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"
namespace {

/// Get the iteration domain for a ComputeOp. The verifier ensures that the
/// maximum tensor rank equals iterator_types.size(). Use the max-rank tensor's
/// shape for loop bounds (handles reductions/broadcasts where other tensors
/// have lower rank).
static SmallVector<Range> getIterationDomain(OpBuilder &b, ComputeOp op) {
  SmallVector<Range> domain;
  Location loc = op.getLoc();

  // Find the tensor with maximum rank (matches iterator domain per verifier).
  Value maxRankTensor;
  int64_t maxRank = 0;
  for (Value operand : llvm::concat<Value>(op.getInputs(), op.getOutputs())) {
    int64_t rank = cast<RankedTensorType>(operand.getType()).getRank();
    if (rank > maxRank) {
      maxRank = rank;
      maxRankTensor = operand;
    }
  }

  if (!maxRankTensor) {
    return domain;
  }

  auto refTy = cast<RankedTensorType>(maxRankTensor.getType());
  for (int64_t i = 0; i < refTy.getRank(); ++i) {
    OpFoldResult offset = b.getIndexAttr(0);
    OpFoldResult stride = b.getIndexAttr(1);
    OpFoldResult size;
    if (refTy.isDynamicDim(i)) {
      size = b.create<tensor::DimOp>(loc, maxRankTensor, i).getResult();
    } else {
      size = b.getIndexAttr(refTy.getDimSize(i));
    }
    domain.push_back(Range{offset, size, stride});
  }
  return domain;
}

/// Apply an indexing map to the induction variables using MLIR's
/// makeComposedFoldedAffineApply utility for automatic composition and folding.
static SmallVector<Value> applyIndexingMap(OpBuilder &b, Location loc,
                                           AffineMap map, ValueRange ivs) {
  SmallVector<OpFoldResult> operands(ivs.begin(), ivs.end());
  assert(operands.size() == map.getNumDims() &&
         "IV count must match map dimensions (verifier ensures this)");

  SmallVector<Value> mapped;
  mapped.reserve(map.getNumResults());

  for (AffineExpr expr : map.getResults()) {
    AffineMap singleResultMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr);
    OpFoldResult result = affine::makeComposedFoldedAffineApply(
        b, loc, singleResultMap, operands);
    mapped.push_back(getValueOrCreateConstantIndexOp(b, loc, result));
  }
  return mapped;
}

/// Generate loop body that processes tiles at given indices.
/// Extracts tiles from inputs, clones compute body, inserts results back.
/// Returns failure if copy_tile encounters unsupported tensor rank/shape.
static FailureOr<scf::ValueVector>
generateTileProcessing(OpBuilder &b, Location loc, ComputeOp op,
                       ArrayRef<AffineMap> indexingMaps, ValueRange ivs,
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

  // Build map from block arguments to input tensor info for O(1) lookup.
  DenseMap<Value, std::pair<size_t, RankedTensorType>> blockArgToInput;
  for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
    blockArgToInput[bodyBlock.getArgument(idx)] = {
        idx, cast<RankedTensorType>(input.getType())};
  }

  // Pre-pass: materialize ttl.linearized_index ops as affine.apply
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (auto linIdx = dyn_cast<LinearizedIndexOp>(&bodyOp)) {
      AffineMap indexMap = linIdx.getIndexMap();

      // Check rank matches
      if (static_cast<int64_t>(ivs.size()) != indexMap.getNumDims()) {
        return failure();
      }

      // Apply the index_map to IVs to get linear index
      // TODO: Add symbol handling for dynamic dimensions using getMixedSizes()
      // to query tensor dimensions and pass as affine map symbols
      SmallVector<OpFoldResult> operands(ivs.begin(), ivs.end());
      OpFoldResult result =
          affine::makeComposedFoldedAffineApply(b, loc, indexMap, operands);
      Value linearIdx = getValueOrCreateConstantIndexOp(b, loc, result);

      // Add to mapping so cloning will use the computed value
      mapping.map(linIdx.getResult(), linearIdx);
    }
  }

  // Compute linearized CB tile index from IVs for StoreOp.
  // For output indexing map (d0, d1) -> (d0, d1), linearize to d0 * cols + d1.
  // We use the first output's indexing map since all outputs should have
  // compatible shapes within a compute block.
  Value cbTileIndex;
  if (!indexingMaps.empty() && numInputs < indexingMaps.size()) {
    AffineMap outputMap = indexingMaps[numInputs]; // First output's map
    SmallVector<Value> outputIndices = applyIndexingMap(b, loc, outputMap, ivs);

    // Linearize: for 2D, idx = row * numCols + col
    if (outputIndices.size() == 2) {
      // Get CB shape from the output tensor type (it matches CB shape).
      auto outputTy =
          cast<RankedTensorType>(iterArgs.front().getType());
      int64_t numCols = outputTy.getDimSize(1);
      Value numColsVal = b.create<arith::ConstantIndexOp>(loc, numCols);
      Value rowOffset =
          b.create<arith::MulIOp>(loc, outputIndices[0], numColsVal);
      cbTileIndex = b.create<arith::AddIOp>(loc, rowOffset, outputIndices[1]);
    } else if (outputIndices.size() == 1) {
      cbTileIndex = outputIndices[0];
    } else {
      // Fallback for higher-rank: just use 0 (shouldn't happen for CBs).
      cbTileIndex = b.create<arith::ConstantIndexOp>(loc, 0);
    }
  } else {
    cbTileIndex = b.create<arith::ConstantIndexOp>(loc, 0);
  }

  // Clone body operations (skip linearized_index since it's already
  // materialized). Force-clone constants into the loop body to enable
  // per-iteration updates during unrolling.
  // For StoreOp, update the cb_tile_index to the computed linearized index.
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (isa<LinearizedIndexOp>(&bodyOp)) {
      continue;
    }

    if (isa<arith::ConstantOp>(&bodyOp)) {
      Operation *cloned = b.clone(bodyOp);
      mapping.map(&bodyOp, cloned);
      for (auto [origResult, clonedResult] :
           llvm::zip(bodyOp.getResults(), cloned->getResults())) {
        mapping.map(origResult, clonedResult);
      }
    } else if (auto storeOp = dyn_cast<StoreOp>(&bodyOp)) {
      // Clone StoreOp with updated cb_tile_index from adjusted IVs.
      Value mappedTile = mapping.lookupOrDefault(storeOp.getTile());
      Value mappedView = mapping.lookupOrDefault(storeOp.getView());
      b.create<StoreOp>(loc, mappedTile, mappedView, cbTileIndex);
    } else {
      b.clone(bodyOp, mapping);
    }
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
  bool enableUnroll;

  LowerComputeToLoops(MLIRContext *context, bool enableUnroll)
      : OpRewritePattern<ComputeOp>(context), enableUnroll(enableUnroll) {}

  LogicalResult matchAndRewrite(ComputeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(op.getIndexingMaps().size());
    for (Attribute attr : op.getIndexingMaps()) {
      indexingMaps.push_back(cast<AffineMapAttr>(attr).getValue());
    }

    SmallVector<Range> iterDomain = getIterationDomain(rewriter, op);
    if (iterDomain.empty()) {
      return failure();
    }

    // Check for unroll factor on the innermost dimension (only if pass option
    // enables unrolling).
    int64_t unrollFactor = 1;
    if (enableUnroll) {
      if (auto unrollAttr =
              op->getAttrOfType<IntegerAttr>(kUnrollFactorAttrName)) {
        unrollFactor = unrollAttr.getInt();
      }
    }

    // Build loop bounds from iteration domain.
    // If unrolling, adjust the innermost loop step.
    SmallVector<Value> lowerBounds, upperBounds, steps;
    for (auto [idx, range] : llvm::enumerate(iterDomain)) {
      Value lb = getValueOrCreateConstantIndexOp(rewriter, loc, range.offset);
      Value ub = getValueOrCreateConstantIndexOp(rewriter, loc, range.size);
      // For the innermost loop (last dimension), apply unroll factor to step.
      int64_t stepValue = (idx == iterDomain.size() - 1) ? unrollFactor : 1;
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, stepValue);
      lowerBounds.push_back(lb);
      upperBounds.push_back(ub);
      steps.push_back(step);
    }

    // Initial values for iter_args are the output tensors.
    SmallVector<Value> initValues(op.getOutputs());

    // Track whether generateTileProcessing fails inside the lambda.
    bool processingFailed = false;
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps, initValues,
        [&](OpBuilder &b, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          // Generate unrolled iterations in the loop body.
          // For each unroll iteration i in [0, unrollFactor), adjust the
          // innermost IV by +i and process tiles at those indices.
          SmallVector<Value> currentIterArgs(iterArgs.begin(), iterArgs.end());
          for (int64_t i = 0; i < unrollFactor; ++i) {
            // Compute adjusted IVs for this unrolled iteration.
            SmallVector<Value> adjustedIvs(ivs.begin(), ivs.end());
            if (!ivs.empty()) {
              // Adjust the innermost IV: iv + i
              Value innermostIv = ivs.back();
              if (i > 0) {
                Value offset = b.create<arith::ConstantIndexOp>(loc, i);
                adjustedIvs.back() =
                    b.create<arith::AddIOp>(loc, innermostIv, offset);
              }
            }

            // Generate tile processing for this iteration.
            auto result = generateTileProcessing(b, loc, op, indexingMaps,
                                                 adjustedIvs, currentIterArgs);
            if (failed(result)) {
              processingFailed = true;
              return {};
            }
            currentIterArgs.assign(result->begin(), result->end());
          }
          return currentIterArgs;
        });

    if (processingFailed) {
      return rewriter.notifyMatchFailure(
          op, "copy_tile index computation failed (mismatched rank/IVs)");
    }

    rewriter.replaceOp(op, loopNest.results);
    return success();
  }
};

struct TTLLowerToLoopsPass
    : public tt::ttl::impl::TTLLowerToLoopsBase<TTLLowerToLoopsPass> {
  using tt::ttl::impl::TTLLowerToLoopsBase<
      TTLLowerToLoopsPass>::TTLLowerToLoopsBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    RewritePatternSet patterns(func.getContext());
    patterns.add<LowerComputeToLoops>(func.getContext(), unrollCompute);
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(func, frozen))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
