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

    // Check for unroll factor (only if pass option enables unrolling).
    int64_t unrollFactor = 1;
    if (enableUnroll) {
      if (auto unrollAttr =
              op->getAttrOfType<IntegerAttr>(kUnrollFactorAttrName)) {
        unrollFactor = unrollAttr.getInt();
      }
    }

    // Collect dimension sizes for linearization/delinearization.
    SmallVector<Value> dimSizes;
    for (const Range &range : iterDomain) {
      dimSizes.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, range.size));
    }

    // Compute total number of tiles (product of all dimensions).
    Value totalTiles = dimSizes[0];
    for (size_t i = 1; i < dimSizes.size(); ++i) {
      totalTiles = rewriter.create<arith::MulIOp>(loc, totalTiles, dimSizes[i]);
    }

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Initial values for iter_args are the output tensors.
    SmallVector<Value> initValues(op.getOutputs());
    SmallVector<Value> currentResults = initValues;

    // Helper lambda to delinearize a flat index to multi-dimensional coords.
    // Uses affine.apply for constant folding when dimensions are static.
    // For dims [D0, D1, ..., Dn], linearIdx -> [i0, i1, ..., in] where
    // linearIdx = i0 * (D1*D2*...*Dn) + i1 * (D2*...*Dn) + ... + in
    auto delinearize = [&](OpBuilder &b, Location loc, Value linearIdx,
                           ArrayRef<Value> dims) -> SmallVector<Value> {
      SmallVector<Value> coords;
      if (dims.size() == 1) {
        coords.push_back(linearIdx);
        return coords;
      }

      // Compute strides: stride[i] = product of dims[i+1..n]
      SmallVector<int64_t> strides(dims.size(), 1);
      for (int i = dims.size() - 2; i >= 0; --i) {
        auto dimConst = dims[i + 1].getDefiningOp<arith::ConstantIndexOp>();
        if (dimConst) {
          strides[i] = strides[i + 1] * dimConst.value();
        } else {
          // Dynamic dimension - fallback needed but shouldn't happen
          strides[i] = 1;
        }
      }

      // Build affine maps for delinearization:
      // coord[i] = (linearIdx floordiv stride[i]) mod dim[i]
      // For outermost: coord[0] = linearIdx floordiv stride[0]
      // For innermost: coord[n] = linearIdx mod dim[n]
      MLIRContext *ctx = b.getContext();
      AffineExpr idx = getAffineDimExpr(0, ctx);

      for (size_t i = 0; i < dims.size(); ++i) {
        AffineExpr coordExpr;
        if (i == 0) {
          // Outermost: idx floordiv stride[0]
          coordExpr = idx.floorDiv(strides[0]);
        } else if (i + 1 == dims.size()) {
          // Innermost: idx mod dim[n]
          auto dimConst = dims[i].getDefiningOp<arith::ConstantIndexOp>();
          int64_t dimVal = dimConst ? dimConst.value() : 1;
          coordExpr = idx % dimVal;
        } else {
          // Middle: (idx floordiv stride[i]) mod dim[i]
          auto dimConst = dims[i].getDefiningOp<arith::ConstantIndexOp>();
          int64_t dimVal = dimConst ? dimConst.value() : 1;
          coordExpr = (idx.floorDiv(strides[i])) % dimVal;
        }
        AffineMap map = AffineMap::get(1, 0, coordExpr, ctx);
        SmallVector<OpFoldResult> operands = {linearIdx};
        OpFoldResult result =
            affine::makeComposedFoldedAffineApply(b, loc, map, operands);
        coords.push_back(getValueOrCreateConstantIndexOp(b, loc, result));
      }
      return coords;
    };

    bool processingFailed = false;

    if (unrollFactor > 1) {
      // Linearized unrolling: iterate over linearized tile indices.
      // Main loop: [0, floor(total/unroll)*unroll) step unroll
      // Remainder loop: [floor(total/unroll)*unroll, total) step 1
      Value unrollVal =
          rewriter.create<arith::ConstantIndexOp>(loc, unrollFactor);
      Value divided =
          rewriter.create<arith::DivUIOp>(loc, totalTiles, unrollVal);
      Value mainBound =
          rewriter.create<arith::MulIOp>(loc, divided, unrollVal);

      // Main loop with unrolling
      scf::LoopNest mainLoop = scf::buildLoopNest(
          rewriter, loc, {c0}, {mainBound}, {unrollVal}, currentResults,
          [&](OpBuilder &b, Location loc, ValueRange ivs,
              ValueRange iterArgs) -> scf::ValueVector {
            Value baseIdx = ivs[0];
            SmallVector<Value> loopIterArgs(iterArgs.begin(), iterArgs.end());

            for (int64_t i = 0; i < unrollFactor; ++i) {
              Value offset = b.create<arith::ConstantIndexOp>(loc, i);
              Value linearIdx = b.create<arith::AddIOp>(loc, baseIdx, offset);
              SmallVector<Value> coords = delinearize(b, loc, linearIdx, dimSizes);

              auto result = generateTileProcessing(b, loc, op, indexingMaps,
                                                   coords, loopIterArgs);
              if (failed(result)) {
                processingFailed = true;
                return scf::ValueVector(iterArgs.begin(), iterArgs.end());
              }
              loopIterArgs.assign(result->begin(), result->end());
            }
            return loopIterArgs;
          });

      if (processingFailed) {
        return rewriter.notifyMatchFailure(
            op, "copy_tile index computation failed in main loop");
      }

      currentResults.assign(mainLoop.results.begin(), mainLoop.results.end());

      // Check if remainder loop is needed: only if total % unroll != 0.
      // When total is statically known, we can skip generating the loop entirely.
      bool needsRemainder = true;
      if (auto totalConst = totalTiles.getDefiningOp<arith::ConstantIndexOp>()) {
        int64_t total = totalConst.value();
        needsRemainder = (total % unrollFactor) != 0;
      }

      if (needsRemainder) {
        // Remainder loop (step 1, no unrolling)
        scf::LoopNest remLoop = scf::buildLoopNest(
            rewriter, loc, {mainBound}, {totalTiles}, {c1}, currentResults,
            [&](OpBuilder &b, Location loc, ValueRange ivs,
                ValueRange iterArgs) -> scf::ValueVector {
              Value linearIdx = ivs[0];
              SmallVector<Value> coords =
                  delinearize(b, loc, linearIdx, dimSizes);

              auto result = generateTileProcessing(b, loc, op, indexingMaps,
                                                   coords, iterArgs);
              if (failed(result)) {
                processingFailed = true;
                return scf::ValueVector(iterArgs.begin(), iterArgs.end());
              }
              return *result;
            });

        if (processingFailed) {
          return rewriter.notifyMatchFailure(
              op, "copy_tile index computation failed in remainder loop");
        }

        currentResults.assign(remLoop.results.begin(), remLoop.results.end());
      }
    } else {
      // No unrolling: standard nested loop structure
      SmallVector<Value> lowerBounds, upperBounds, steps;
      for (size_t i = 0; i < iterDomain.size(); ++i) {
        lowerBounds.push_back(c0);
        upperBounds.push_back(dimSizes[i]);
        steps.push_back(c1);
      }

      scf::LoopNest loopNest = scf::buildLoopNest(
          rewriter, loc, lowerBounds, upperBounds, steps, currentResults,
          [&](OpBuilder &b, Location loc, ValueRange ivs,
              ValueRange iterArgs) -> scf::ValueVector {
            auto result =
                generateTileProcessing(b, loc, op, indexingMaps, ivs, iterArgs);
            if (failed(result)) {
              processingFailed = true;
              return scf::ValueVector(iterArgs.begin(), iterArgs.end());
            }
            return *result;
          });

      if (processingFailed) {
        return rewriter.notifyMatchFailure(
            op, "copy_tile index computation failed");
      }

      currentResults.assign(loopNest.results.begin(), loopNest.results.end());
    }

    rewriter.replaceOp(op, currentResults);
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
