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

/// Compute total static elements in a tensor shape. Returns 0 for dynamic dims.
static int64_t getTotalElements(RankedTensorType type) {
  int64_t total = 1;
  for (int64_t dim : type.getShape()) {
    if (dim == ShapedType::kDynamic) {
      return 0;
    }
    total *= dim;
  }
  return total;
}

/// Get the iteration domain for a ComputeOp. The verifier ensures that the
/// maximum tensor rank equals iterator_types.size(). Use the tensor with the
/// largest shape for loop bounds (handles broadcasts where output is larger
/// than input).
static SmallVector<Range> getIterationDomain(OpBuilder &b, ComputeOp op) {
  SmallVector<Range> domain;
  Location loc = op.getLoc();

  // Find the tensor with the largest iteration domain.
  // Prefer higher rank, then larger element count for same rank.
  Value maxRankTensor;
  int64_t maxRank = 0;
  int64_t maxElements = 0;
  for (Value operand : llvm::concat<Value>(op.getInputs(), op.getOutputs())) {
    auto type = cast<RankedTensorType>(operand.getType());
    int64_t rank = type.getRank();
    int64_t elements = getTotalElements(type);
    if (rank > maxRank || (rank == maxRank && elements > maxElements)) {
      maxRank = rank;
      maxElements = elements;
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

/// Generate side-effect-only loop body. Extracts tiles from inputs, clones
/// compute body ops, and returns nothing (stores are explicit side effects).
static LogicalResult generateTileProcessing(OpBuilder &b, Location loc,
                                            ComputeOp op,
                                            ArrayRef<AffineMap> indexingMaps,
                                            ValueRange ivs) {
  // Extract tiles from inputs at current mapped indices.
  SmallVector<Value> extractedInputs;
  for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
    SmallVector<Value> indices =
        applyIndexingMap(b, loc, indexingMaps[idx], ivs);
    Value tile = b.create<tensor::ExtractOp>(loc, input, indices);
    extractedInputs.push_back(tile);
  }

  // Output block args get a dummy extract from the output tensor. These are
  // needed for SSA mapping but unused in the body (stores write via DST).
  SmallVector<Value> extractedOutputs;
  size_t numInputs = op.getInputs().size();
  for (auto [idx, output] : llvm::enumerate(op.getOutputs())) {
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

  // Pre-pass: materialize ttl.linearized_index ops as affine.apply
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (auto linIdx = dyn_cast<LinearizedIndexOp>(&bodyOp)) {
      AffineMap indexMap = linIdx.getIndexMap();

      if (static_cast<int64_t>(ivs.size()) != indexMap.getNumDims()) {
        return failure();
      }

      // TODO: Add symbol handling for dynamic dimensions using getMixedSizes()
      // to query tensor dimensions and pass as affine map symbols
      SmallVector<OpFoldResult> operands(ivs.begin(), ivs.end());
      OpFoldResult result =
          affine::makeComposedFoldedAffineApply(b, loc, indexMap, operands);
      Value linearIdx = getValueOrCreateConstantIndexOp(b, loc, result);

      mapping.map(linIdx.getResult(), linearIdx);
    }
  }

  // Clone body operations (skip linearized_index and yield)
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (!isa<LinearizedIndexOp>(&bodyOp)) {
      b.clone(bodyOp, mapping);
    }
  }

  return success();
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

    // Side-effect-only loops: no iter_args, no tensor.insert, no scf.yield
    // with tensor values. Stores are explicit side effects (tile_store).
    bool processingFailed = false;
    scf::buildLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps, ValueRange{},
        [&](OpBuilder &b, Location loc, ValueRange ivs,
            ValueRange /*iterArgs*/) -> scf::ValueVector {
          if (failed(generateTileProcessing(b, loc, op, indexingMaps, ivs))) {
            processingFailed = true;
          }
          return {};
        });

    if (processingFailed) {
      return rewriter.notifyMatchFailure(
          op, "copy_tile index computation failed (mismatched rank/IVs)");
    }

    // Replace compute op with its output operands directly.
    rewriter.replaceOp(op, op.getOutputs());
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
