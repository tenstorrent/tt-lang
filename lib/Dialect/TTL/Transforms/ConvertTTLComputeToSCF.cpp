// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

/// Categorize body operations into compute phase, sync ops, and pack phase.
/// The sync pass (TTLInsertTileRegsSyncPass) orders ops as:
///   [compute ops] -> commit -> wait -> [store ops] -> yield
/// We split these into separate phases for proper loop generation.
struct BodyPhases {
  SmallVector<Operation *> computeOps; // Ops before commit (compute phase)
  TileRegsCommitOp commitOp;
  TileRegsWaitOp waitOp;
  SmallVector<Operation *> packOps; // Ops after wait (pack phase: stores, etc.)
};

static BodyPhases categorizeBodyOps(Block &bodyBlock) {
  BodyPhases phases;
  bool seenCommit = false;
  bool seenWait = false;

  for (Operation &op : bodyBlock.without_terminator()) {
    if (auto commit = dyn_cast<TileRegsCommitOp>(&op)) {
      phases.commitOp = commit;
      seenCommit = true;
    } else if (auto wait = dyn_cast<TileRegsWaitOp>(&op)) {
      phases.waitOp = wait;
      seenWait = true;
    } else if (!seenCommit) {
      phases.computeOps.push_back(&op);
    } else if (seenWait) {
      phases.packOps.push_back(&op);
    }
    // Ops between commit and wait are ignored (shouldn't be any)
  }
  return phases;
}

/// Setup mapping from block arguments to extracted tiles.
static void setupBlockArgMapping(OpBuilder &b, Location loc, ComputeOp op,
                                 ArrayRef<AffineMap> indexingMaps,
                                 ValueRange ivs, ValueRange iterArgs,
                                 IRMapping &mapping) {
  size_t numInputs = op.getInputs().size();
  Block &bodyBlock = op.getBody().front();

  // Extract and map input tiles.
  for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
    SmallVector<Value> indices =
        applyIndexingMap(b, loc, indexingMaps[idx], ivs);
    Value tile = b.create<tensor::ExtractOp>(loc, input, indices);
    mapping.map(bodyBlock.getArgument(idx), tile);
  }

  // Extract and map output tiles.
  for (auto [idx, output] : llvm::enumerate(iterArgs)) {
    SmallVector<Value> indices =
        applyIndexingMap(b, loc, indexingMaps[numInputs + idx], ivs);
    Value tile = b.create<tensor::ExtractOp>(loc, output, indices);
    mapping.map(bodyBlock.getArgument(numInputs + idx), tile);
  }
}

/// Materialize linearized_index ops into the mapping.
static LogicalResult materializeLinearizedIndices(OpBuilder &b, Location loc,
                                                  Block &bodyBlock,
                                                  ValueRange ivs,
                                                  IRMapping &mapping) {
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (auto linIdx = dyn_cast<LinearizedIndexOp>(&bodyOp)) {
      AffineMap indexMap = linIdx.getIndexMap();
      if (static_cast<int64_t>(ivs.size()) != indexMap.getNumDims()) {
        return failure();
      }
      SmallVector<OpFoldResult> operands(ivs.begin(), ivs.end());
      OpFoldResult result =
          affine::makeComposedFoldedAffineApply(b, loc, indexMap, operands);
      Value linearIdx = getValueOrCreateConstantIndexOp(b, loc, result);
      mapping.map(linIdx.getResult(), linearIdx);
    }
  }
  return success();
}

/// Clone a subset of operations, skipping certain op types.
static void cloneOps(OpBuilder &b, ArrayRef<Operation *> ops,
                     IRMapping &mapping) {
  for (Operation *op : ops) {
    if (!isa<LinearizedIndexOp>(op)) {
      b.clone(*op, mapping);
    }
  }
}

/// Generate compute phase loop body.
/// Extracts tiles, clones compute ops, returns iter_args unchanged
/// (actual results stored in DST registers, not in tensor SSA).
static FailureOr<scf::ValueVector>
generateComputePhase(OpBuilder &b, Location loc, ComputeOp op,
                     ArrayRef<AffineMap> indexingMaps, ValueRange ivs,
                     ValueRange iterArgs, const BodyPhases &phases) {
  Block &bodyBlock = op.getBody().front();
  IRMapping mapping;

  setupBlockArgMapping(b, loc, op, indexingMaps, ivs, iterArgs, mapping);

  if (failed(materializeLinearizedIndices(b, loc, bodyBlock, ivs, mapping))) {
    return failure();
  }

  cloneOps(b, phases.computeOps, mapping);

  // Compute phase doesn't modify tensors - results go to DST registers.
  // Return iter_args unchanged; pack phase will handle tensor updates.
  return SmallVector<Value>(iterArgs.begin(), iterArgs.end());
}

/// Generate pack phase loop body.
/// Clones store ops (pack reads from DST registers based on loop indices).
/// The tensor insert is a placeholder to maintain SSA form - the actual data
/// comes from DST registers via pack_tile.
static FailureOr<scf::ValueVector>
generatePackPhase(OpBuilder &b, Location loc, ComputeOp op,
                  ArrayRef<AffineMap> indexingMaps, ValueRange ivs,
                  ValueRange iterArgs, const BodyPhases &phases) {
  Block &bodyBlock = op.getBody().front();
  size_t numInputs = op.getInputs().size();
  IRMapping mapping;

  // Setup mapping - we need block args mapped even though we're not using
  // the extracted values directly. The mapping allows cloned ops to reference
  // the correct types.
  setupBlockArgMapping(b, loc, op, indexingMaps, ivs, iterArgs, mapping);

  if (failed(materializeLinearizedIndices(b, loc, bodyBlock, ivs, mapping))) {
    return failure();
  }

  // Map compute op results to placeholder values (extracted output tiles).
  // Pack ops reference compute results via SSA, but the actual data comes from
  // DST registers. We provide placeholder values to satisfy SSA requirements.
  for (Operation *computeOp : phases.computeOps) {
    for (Value result : computeOp->getResults()) {
      if (!mapping.contains(result) && !iterArgs.empty()) {
        // Extract from first output - the type should match compute result.
        SmallVector<Value> indices =
            applyIndexingMap(b, loc, indexingMaps[numInputs], ivs);
        Value placeholder =
            b.create<tensor::ExtractOp>(loc, iterArgs[0], indices);
        mapping.map(result, placeholder);
      }
    }
  }

  // Clone pack ops - these become pack_tile which reads from DST registers.
  // The DST index is computed from loop IVs by later lowering passes.
  cloneOps(b, phases.packOps, mapping);

  // Tensor insert to maintain SSA form. The actual data comes from DST via
  // pack_tile, but we need tensor values to flow through the loop.
  auto yieldOp = cast<YieldOp>(bodyBlock.getTerminator());
  SmallVector<Value> results;
  for (auto [idx, yieldVal] : llvm::enumerate(yieldOp.getValues())) {
    Value result = mapping.lookupOrDefault(yieldVal);
    SmallVector<Value> indices =
        applyIndexingMap(b, loc, indexingMaps[numInputs + idx], ivs);
    Value updated =
        b.create<tensor::InsertOp>(loc, result, iterArgs[idx], indices);
    results.push_back(updated);
  }
  return results;
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

    // Categorize body ops into compute phase, sync ops, and pack phase.
    Block &bodyBlock = op.getBody().front();
    BodyPhases phases = categorizeBodyOps(bodyBlock);

    // If no sync ops found, fall back to original single-loop implementation.
    // This ensures backward compatibility for ops without commit/wait markers.
    if (!phases.commitOp || !phases.waitOp) {
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
            op, "tile processing failed (mismatched rank/IVs)");
      }

      rewriter.replaceOp(op, loopNest.results);
      return success();
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

    // Generate two-loop structure matching tt-metal pattern:
    //   Loop 1: All compute ops (write to DST registers)
    //   tile_regs_commit
    //   tile_regs_wait
    //   Loop 2: All pack ops (read from DST registers)
    //
    // This allows DST registers to accumulate results across all tiles
    // before packing them out to the circular buffer.

    bool computeFailed = false;
    bool packFailed = false;

    // Loop 1: Compute phase - iterate over all tiles, compute into DST regs.
    // Compute doesn't modify output tensors, so iter_args pass through
    // unchanged.
    scf::LoopNest computeLoop = scf::buildLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps, initValues,
        [&](OpBuilder &b, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          auto result = generateComputePhase(b, loc, op, indexingMaps, ivs,
                                             iterArgs, phases);
          if (failed(result)) {
            computeFailed = true;
            return {};
          }
          return *result;
        });

    if (computeFailed) {
      return rewriter.notifyMatchFailure(op, "compute phase failed");
    }

    // Propagate dst_footprint attribute to outermost compute loop for access
    // during tile op lowering (for multi-tile dynamic DST index computation).
    if (auto footprintAttr =
            op->getAttrOfType<IntegerAttr>(kDstFootprintAttrName)) {
      if (!computeLoop.loops.empty()) {
        computeLoop.loops.front()->setAttr(kDstFootprintAttrName,
                                           footprintAttr);
      }
    }

    // Insert commit/wait between compute and pack loops.
    rewriter.clone(*phases.commitOp);
    rewriter.clone(*phases.waitOp);

    // Loop 2: Pack phase - iterate over all tiles, pack from DST to CB.
    // Pack phase updates the output tensors via tensor.insert.
    scf::LoopNest packLoop = scf::buildLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps, computeLoop.results,
        [&](OpBuilder &b, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          auto result = generatePackPhase(b, loc, op, indexingMaps, ivs,
                                          iterArgs, phases);
          if (failed(result)) {
            packFailed = true;
            return {};
          }
          return *result;
        });

    if (packFailed) {
      return rewriter.notifyMatchFailure(op, "pack phase failed");
    }

    // Propagate dst_footprint attribute to outermost pack loop for access
    // during store/pack_tile lowering (for multi-tile dynamic DST index).
    if (auto footprintAttr =
            op->getAttrOfType<IntegerAttr>(kDstFootprintAttrName)) {
      if (!packLoop.loops.empty()) {
        packLoop.loops.front()->setAttr(kDstFootprintAttrName, footprintAttr);
      }
    }

    rewriter.replaceOp(op, packLoop.results);
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
