// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
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

  // Clone body operations (skip linearized_index since it's already
  // materialized)
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (!isa<LinearizedIndexOp>(&bodyOp)) {
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
  using OpRewritePattern<ComputeOp>::OpRewritePattern;

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

    // Build loop bounds from iteration domain.
    SmallVector<Value> lowerBounds, upperBounds, steps;
    for (auto [idx, range] : llvm::enumerate(iterDomain)) {
      Value lb = getValueOrCreateConstantIndexOp(rewriter, loc, range.offset);
      Value ub = getValueOrCreateConstantIndexOp(rewriter, loc, range.size);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
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
          auto result =
              generateTileProcessing(b, loc, op, indexingMaps, ivs, iterArgs);
          if (failed(result)) {
            processingFailed = true;
            return {};
          }
          return *result;
        });

    if (processingFailed) {
      return rewriter.notifyMatchFailure(
          op, "copy_tile index computation failed (mismatched rank/IVs)");
    }

    // Mark the innermost loop for later sync insertion pass.
    // The kTileLoopAttrName attribute indicates this loop came from a ComputeOp
    // and needs DST register synchronization ops inserted.
    if (!loopNest.loops.empty()) {
      scf::ForOp innermostLoop = loopNest.loops.back();
      innermostLoop->setAttr(kTileLoopAttrName, rewriter.getUnitAttr());

      // Store CB indices for init_sfpu in the TTLInsertTileRegsSync pass.
      // TODO: Currently only stores the first input/output CB. If ComputeOp
      // has multiple inputs/outputs with different CBs, init_sfpu will only
      // be configured for the first ones. This may need to be extended to
      // handle multiple CBs if required by the hardware.
      Value icb = getAttachedCB(op.getInputs().front());
      Value ocb = getAttachedCB(op.getOutputs().front());
      if (icb) {
        if (auto bindOp = icb.getDefiningOp<BindCBOp>()) {
          innermostLoop->setAttr(
              kTileLoopInputCBAttrName,
              rewriter.getI64IntegerAttr(bindOp.getCbIndex().getSExtValue()));
        }
      }
      if (ocb) {
        if (auto bindOp = ocb.getDefiningOp<BindCBOp>()) {
          innermostLoop->setAttr(
              kTileLoopOutputCBAttrName,
              rewriter.getI64IntegerAttr(bindOp.getCbIndex().getSExtValue()));
        }
      }
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
    patterns.add<LowerComputeToLoops>(func.getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(func, frozen))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
