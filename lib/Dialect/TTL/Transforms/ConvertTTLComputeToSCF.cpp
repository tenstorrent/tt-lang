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
/// @param linearIV Optional linear induction variable for flattened loops.
///                 When provided, used directly for ttl.linearized_index
///                 operations.
static FailureOr<scf::ValueVector>
generateTileProcessing(OpBuilder &b, Location loc, ComputeOp op,
                       ArrayRef<AffineMap> indexingMaps, ValueRange ivs,
                       ValueRange iterArgs, Value linearIV = Value()) {
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

  // Pre-pass: materialize ttl.linearized_index ops as affine.apply.
  // For flattened loops, use the linear IV directly to avoid complex composed
  // maps.
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (auto linIdx = dyn_cast<LinearizedIndexOp>(&bodyOp)) {
      Value computedLinearIdx;

      if (linearIV) {
        // Flattened loop: use the linear IV directly (already linearized)
        computedLinearIdx = linearIV;
      } else {
        // Nested loops: apply linearization map to multi-dimensional IVs
        AffineMap indexMap = linIdx.getIndexMap();
        if (static_cast<int64_t>(ivs.size()) != indexMap.getNumDims()) {
          return failure();
        }
        SmallVector<OpFoldResult> operands(ivs.begin(), ivs.end());
        OpFoldResult result =
            affine::makeComposedFoldedAffineApply(b, loc, indexMap, operands);
        computedLinearIdx = getValueOrCreateConstantIndexOp(b, loc, result);
      }

      // Add to mapping so cloning will use the computed value
      mapping.map(linIdx.getResult(), computedLinearIdx);
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
      auto outputTy = cast<RankedTensorType>(iterArgs.front().getType());
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

    // Flatten multi-dimensional iteration into a single loop following the
    // upstream MLIR affine-loop-coalescing approach. Calculate total iterations
    // as the product of all dimension sizes, then delinearize inside the loop.
    int64_t totalIterations = 1;
    SmallVector<int64_t> dimSizes;
    for (auto range : iterDomain) {
      auto sizeAttr =
          dyn_cast_if_present<IntegerAttr>(range.size.dyn_cast<Attribute>());
      if (!sizeAttr) {
        return rewriter.notifyMatchFailure(
            op,
            "dynamic iteration bounds not yet supported for flattened loops");
      }
      int64_t dimSize = sizeAttr.getInt();
      dimSizes.push_back(dimSize);
      totalIterations *= dimSize;
    }

    // Create single flattened loop: for %iv = 0 to totalIterations step 1
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, totalIterations);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> initValues(op.getOutputs());

    bool processingFailed = false;
    auto forOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, initValues,
        [&](OpBuilder &b, Location loc, Value linearIV, ValueRange iterArgs) {
          // Delinearize the linear IV into multi-dimensional indices following
          // upstream MLIR formula: iv_i = (linearIV floordiv stride_i) mod
          // range_i where stride_i = product of ranges for dimensions [i+1,
          // ..., n-1]
          SmallVector<Value> delinearizedIVs;

          for (size_t i = 0; i < dimSizes.size(); ++i) {
            // Calculate stride = product of all inner dimension sizes
            int64_t stride = 1;
            for (size_t j = i + 1; j < dimSizes.size(); ++j) {
              stride *= dimSizes[j];
            }

            // For the last dimension, stride=1, so just use: iv_i = linearIV
            // mod range_i For other dimensions: iv_i = (linearIV floordiv
            // stride) mod range_i
            AffineExpr d0 = b.getAffineDimExpr(0);
            AffineMap map;

            if (stride == 1) {
              // Innermost dimension: iv = linearIV mod range
              map = AffineMap::get(1, 0, d0 % dimSizes[i]);
            } else {
              // Outer dimensions: iv = (linearIV floordiv stride) mod range
              map = AffineMap::get(1, 0, (d0.floorDiv(stride)) % dimSizes[i]);
            }

            Value dimIndex =
                b.create<affine::AffineApplyOp>(loc, map, ValueRange{linearIV});
            delinearizedIVs.push_back(dimIndex);
          }

          auto result = generateTileProcessing(
              b, loc, op, indexingMaps, delinearizedIVs, iterArgs,
              linearIV); // Pass linear IV for flattened loop
          if (failed(result)) {
            processingFailed = true;
            b.create<scf::YieldOp>(loc, ValueRange{});
            return;
          }
          b.create<scf::YieldOp>(loc, *result);
        });

    if (processingFailed) {
      return rewriter.notifyMatchFailure(
          op, "copy_tile index computation failed (mismatched rank/IVs)");
    }

    // Transfer ttl.unroll_factor attribute from ttl.compute to the flattened
    // loop
    if (auto unrollFactor =
            op->getAttrOfType<IntegerAttr>(kUnrollFactorAttrName)) {
      forOp->setAttr(kUnrollFactorAttrName, unrollFactor);
    }

    // Transfer ttl.num_inputs attribute (needed by unroll pass for DST
    // allocation)
    if (auto numInputs = op->getAttrOfType<IntegerAttr>("ttl.num_inputs")) {
      forOp->setAttr("ttl.num_inputs", numInputs);
    }

    rewriter.replaceOp(op, forOp.getResults());
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

  /// Validate that all linearized_index ops have index_map dimensions matching
  /// their enclosing compute op's iteration domain rank.
  static LogicalResult validateLinearizedIndexRanks(ComputeOp computeOp) {
    int64_t iterDomainRank = 0;
    for (Value operand :
         llvm::concat<Value>(computeOp.getInputs(), computeOp.getOutputs())) {
      int64_t rank = cast<RankedTensorType>(operand.getType()).getRank();
      iterDomainRank = std::max(iterDomainRank, rank);
    }

    Block &bodyBlock = computeOp.getBody().front();
    for (Operation &bodyOp : bodyBlock.without_terminator()) {
      if (auto linIdx = dyn_cast<LinearizedIndexOp>(&bodyOp)) {
        AffineMap indexMap = linIdx.getIndexMap();
        if (static_cast<int64_t>(indexMap.getNumDims()) != iterDomainRank) {
          return linIdx.emitOpError()
                 << "index_map has " << indexMap.getNumDims()
                 << " dimensions but iteration domain has " << iterDomainRank;
        }
      }
    }
    return success();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Pre-validate linearized_index ranks before running patterns.
    for (auto computeOp : func.getOps<ComputeOp>()) {
      if (failed(validateLinearizedIndexRanks(computeOp))) {
        return signalPassFailure();
      }
    }

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
