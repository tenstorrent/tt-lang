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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-lower-to-loops"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLLOWERTOLOOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"
namespace {

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
    int64_t elements = type.getNumElements();
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
      Value step = getValueOrCreateConstantIndexOp(rewriter, loc, range.stride);
      lowerBounds.push_back(lb);
      upperBounds.push_back(ub);
      steps.push_back(step);
    }

    // Side-effect-only loops: no iter_args, no tensor.insert, no scf.yield
    // with tensor values. Stores are explicit side effects (tile_store).
    bool processingFailed = false;
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps, ValueRange{},
        [&](OpBuilder &b, Location loc, ValueRange ivs,
            ValueRange /*iterArgs*/) -> scf::ValueVector {
          if (failed(generateTileProcessing(b, loc, op, indexingMaps, ivs))) {
            processingFailed = true;
          }
          return {};
        });

    // Annotate tile loops with linearization strides for CB indexing.
    // If the compute was subblocked, use the full tensor strides (which
    // differ from the subblock shape bounds). Otherwise, compute strides
    // from the iteration domain (which IS the full shape).
    auto fullStridesAttr =
        op->getAttrOfType<DenseI64ArrayAttr>(kFullLinStridesAttrName);
    SmallVector<int64_t> domainStrides;
    if (!fullStridesAttr) {
      // Extract static sizes from iteration domain.
      SmallVector<int64_t> domainSizes;
      domainSizes.reserve(iterDomain.size());
      for (auto &range : iterDomain) {
        auto size = getConstantIntValue(range.size);
        assert(size && "iteration domain must have static sizes for "
                       "linearization stride computation");
        domainSizes.push_back(*size);
      }
      domainStrides = computeStrides(domainSizes);
    }
    for (auto [idx, loop] : llvm::enumerate(loopNest.loops)) {
      int64_t stride =
          fullStridesAttr ? fullStridesAttr[idx] : domainStrides[idx];
      loop->setAttr(kTileLoopAttrName, rewriter.getIndexAttr(stride));
    }

    // Mark the outermost tile loop for unrolling if the compute was
    // subblocked (has full linearization strides). Non-subblocked computes
    // keep their tile loops for per-tile sync.
    if (fullStridesAttr && !loopNest.loops.empty()) {
      loopNest.loops.front()->setAttr("ttl.should_unroll",
                                      rewriter.getUnitAttr());
    }

    if (processingFailed) {
      return rewriter.notifyMatchFailure(
          op, "copy_tile index computation failed (mismatched rank/IVs)");
    }

    // Replace compute op with its output operands directly.
    rewriter.replaceOp(op, op.getOutputs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Post-pattern tile loop unrolling and DST index assignment
//===----------------------------------------------------------------------===//

/// Fully unroll a tile loop nest, then walk the unrolled ops to assign
/// incrementing DST indices and tile offsets. Uses loopUnrollByFactor with
/// annotateFn to tag each unrolled copy with its per-dimension iteration index.
static LogicalResult
unrollTileLoopNestAndAssignDST(SmallVector<scf::ForOp> &nest) {
  if (nest.empty()) {
    return success();
  }

  int64_t rank = nest.size();

  // Collect dim sizes (trip counts) and full strides from loop attributes.
  SmallVector<int64_t> dimSizes(rank);
  SmallVector<int64_t> fullStrides(rank);
  int64_t totalTiles = 1;
  for (int64_t d = 0; d < rank; ++d) {
    auto ub = getConstantIntValue(nest[d].getUpperBound());
    auto lb = getConstantIntValue(nest[d].getLowerBound());
    auto step = getConstantIntValue(nest[d].getStep());
    if (!ub || !lb || !step || *step == 0) {
      return failure();
    }
    dimSizes[d] = (*ub - *lb) / *step;
    totalTiles *= dimSizes[d];

    auto strideAttr = nest[d]->getAttrOfType<IntegerAttr>(kTileLoopAttrName);
    fullStrides[d] = strideAttr ? strideAttr.getInt() : 1;
  }

  if (totalTiles <= 1) {
    return success(); // Single iteration, nothing to unroll.
  }

  // Save the enclosing block before unrolling (loops will be erased).
  Block *enclosingBlock = nest.front()->getBlock();

  // Compute dstPerIteration from the innermost loop body.
  int64_t maxDstIdx = 0;
  nest.back().getBody()->walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
      maxDstIdx = std::max(maxDstIdx, static_cast<int64_t>(attr.getInt()));
    }
    if (auto copyTile = dyn_cast<CopyTileOp>(op)) {
      if (auto constIdx = getConstantIntValue(copyTile.getDstIndex())) {
        maxDstIdx = std::max(maxDstIdx, *constIdx);
      }
    }
  });
  int64_t dstPerIteration = maxDstIdx + 1;

  // Compute local (subblock-shape) strides for linearizing tile index.
  SmallVector<int64_t> localStrides = computeStrides(dimSizes);

  // Unroll from innermost to outermost. Each loop is fully unrolled
  // (factor = trip count). The annotateFn tags every cloned op with its
  // iteration index for that dimension. For trip count 1, loopUnrollByFactor
  // is a no-op, so we manually fold: replace IV with lb, inline body, erase.
  for (int64_t d = rank - 1; d >= 0; --d) {
    std::string attrName = ("_uiter_" + llvm::Twine(d)).str();
    uint64_t tripCount = static_cast<uint64_t>(dimSizes[d]);

    if (tripCount <= 1) {
      // Trip count 1: manually fold the loop. Tag all body ops with iter 0,
      // replace IV with lower bound, inline body, erase loop.
      scf::ForOp loop = nest[d];
      loop.getInductionVar().replaceAllUsesWith(loop.getLowerBound());
      // Tag body ops before moving them.
      for (Operation &bodyOp : *loop.getBody()) {
        if (!bodyOp.hasTrait<OpTrait::IsTerminator>()) {
          bodyOp.walk([&attrName](Operation *inner) {
            inner->setAttr(
                attrName,
                IntegerAttr::get(IntegerType::get(inner->getContext(), 64), 0));
          });
        }
      }
      // Move body ops to parent block (before the loop).
      Block *parentBlock = loop->getBlock();
      Block *loopBody = loop.getBody();
      loopBody->getTerminator()->erase();
      parentBlock->getOperations().splice(Block::iterator(loop),
                                          loopBody->getOperations());
      loop->erase();
      continue;
    }

    auto result =
        loopUnrollByFactor(nest[d], tripCount,
                           [&attrName](unsigned i, Operation *op, OpBuilder b) {
                             op->setAttr(attrName, b.getI64IntegerAttr(i));
                           });

    if (failed(result)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to unroll tile loop dimension " << d << "\n");
      return failure();
    }
  }

  // Walk the enclosing block to find ops tagged by the unroll annotations.
  // For each op with all dimension annotations, compute the linearized tile
  // index, assign dst_idx and tile_offset, then remove temporary annotations.
  auto walkFn = [&](Operation *op) {
    // Check if this op has all dimension annotations.
    SmallVector<int64_t> dimIndices(rank);
    for (int64_t d = 0; d < rank; ++d) {
      std::string attrName = ("_uiter_" + llvm::Twine(d)).str();
      auto attr = op->getAttrOfType<IntegerAttr>(attrName);
      if (!attr) {
        return;
      }
      dimIndices[d] = attr.getInt();
    }

    // Compute linearized tile index (for DST assignment).
    int64_t tileIdx = linearize(dimIndices, localStrides);

    // Compute tile_offset using full strides (for CB indexing).
    int64_t tileOffset = linearize(dimIndices, fullStrides);

    int64_t dstBase = tileIdx * dstPerIteration;

    // Patch dst_idx attribute.
    if (auto attr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
      if (dstBase != 0) {
        int64_t newIdx = attr.getInt() + dstBase;
        op->setAttr(kDstIdxAttrName,
                    IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                     static_cast<int32_t>(newIdx)));
      }
    }

    // Patch CopyTileOp dst_index operand.
    if (auto copyTile = dyn_cast<CopyTileOp>(op)) {
      if (dstBase != 0) {
        OpBuilder b(copyTile);
        Value offsetVal =
            b.create<arith::ConstantIndexOp>(copyTile.getLoc(), dstBase);
        Value newDstIndex = b.create<arith::AddIOp>(
            copyTile.getLoc(), copyTile.getDstIndex(), offsetVal);
        copyTile.getDstIndexMutable().assign(newDstIndex);
      }
    }

    // Set tile_offset on TTL dialect ops for CB index computation.
    if (auto *dialect = op->getDialect()) {
      if (dialect->getNamespace() == "ttl") {
        op->setAttr(
            kTileOffsetAttrName,
            IntegerAttr::get(IndexType::get(op->getContext()), tileOffset));
      }
    }

    // Remove temporary unroll annotations.
    for (int64_t d = 0; d < rank; ++d) {
      op->removeAttr(("_uiter_" + llvm::Twine(d)).str());
    }
  };

  // Walk the enclosing block (saved before unrolling) to find tagged ops.
  // This handles both the non-subblocked case (ops directly in the function)
  // and the subblocked case (ops inside subblock scf.for loops).
  for (Operation &op : *enclosingBlock) {
    op.walk(walkFn);
  }

  // Clean up any remaining temporary annotations (e.g., on arith ops created
  // by the unroller infrastructure that don't have all dimension tags).
  for (Operation &op : *enclosingBlock) {
    op.walk([rank](Operation *inner) {
      for (int64_t d = 0; d < rank; ++d) {
        inner->removeAttr(("_uiter_" + llvm::Twine(d)).str());
      }
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Unrolled tile loop nest: " << rank << "D [";
    for (int64_t d = 0; d < rank; ++d) {
      llvm::dbgs() << dimSizes[d];
      if (d < rank - 1) {
        llvm::dbgs() << "x";
      }
    }
    llvm::dbgs() << "] (" << totalTiles
                 << " tiles, dstPerIter=" << dstPerIteration << ")\n";
  });

  return success();
}

/// Reorder tile_store ops within a sync region to satisfy the hardware DST
/// protocol. Scans a block for the pattern:
///   acquire -> [stores interleaved with compute] -> commit -> wait -> release
/// and moves any tile_store ops found between acquire and commit to after
/// wait, preserving their relative order. This separates the math phase
/// (acquire->commit) from the pack phase (wait->release).
static void reorderStoresAfterSync(Block *block) {
  // Copy the ops to avoid iterator invalidation during moves.
  SmallVector<Operation *> ops = llvm::to_vector(
      llvm::map_range(*block, [](Operation &op) { return &op; }));

  SmallVector<TileStoreOp> storesToHoist;
  bool inComputeRegion = false;

  for (Operation *op : ops) {
    if (isa<TileRegsAcquireOp>(op)) {
      inComputeRegion = true;
      storesToHoist.clear();
    } else if (isa<TileRegsCommitOp>(op)) {
      inComputeRegion = false;
    } else if (auto w = dyn_cast<TileRegsWaitOp>(op)) {
      // Move all stores collected from the compute region to after wait,
      // preserving their relative order.
      Operation *insertAfter = w;
      for (TileStoreOp store : storesToHoist) {
        store->moveAfter(insertAfter);
        insertAfter = store;
      }
      storesToHoist.clear();
    } else if (inComputeRegion && isa<TileStoreOp>(op)) {
      storesToHoist.push_back(cast<TileStoreOp>(op));
    }
  }
}

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

    // Step 1: Lower compute ops to tile loops (always creates scf.for loops).
    RewritePatternSet patterns(func.getContext());
    patterns.add<LowerComputeToLoops>(func.getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(func, frozen))) {
      return signalPassFailure();
    }

    // Step 2: Fully unroll tile loop nests and assign DST indices.
    // Collect outermost tile loops first to avoid walking invalidated ops.
    SmallVector<scf::ForOp> outerTileLoops;
    func.walk([&](scf::ForOp loop) {
      if (!loop->hasAttr(kTileLoopAttrName)) {
        return;
      }
      // Check if this is the outermost tile loop (parent is not a tile loop).
      auto parent = loop->getParentOfType<scf::ForOp>();
      if (parent && parent->hasAttr(kTileLoopAttrName)) {
        return;
      }
      outerTileLoops.push_back(loop);
    });

    // Collect a tile loop nest starting from the outermost tile loop.
    // Returns loops ordered from outermost to innermost.
    auto collectTileLoopNest = [](scf::ForOp outerLoop) {
      SmallVector<scf::ForOp> nest;
      scf::ForOp current = outerLoop;
      while (current) {
        nest.push_back(current);
        scf::ForOp inner = nullptr;
        for (Operation &op : *current.getBody()) {
          if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
            if (forOp->hasAttr(kTileLoopAttrName)) {
              inner = forOp;
              break;
            }
          }
        }
        current = inner;
      }
      return nest;
    };

    for (scf::ForOp outerLoop : outerTileLoops) {
      // Only unroll tile loops from subblocked computes (marked in step 1).
      // Non-subblocked computes keep their tile loops for per-tile sync.
      if (!outerLoop->hasAttr("ttl.should_unroll")) {
        continue;
      }
      outerLoop->removeAttr("ttl.should_unroll");

      SmallVector<scf::ForOp> nest = collectTileLoopNest(outerLoop);

      // Compute total trip count.
      int64_t totalTrip = 1;
      for (scf::ForOp loop : nest) {
        auto ub = getConstantIntValue(loop.getUpperBound());
        auto lb = getConstantIntValue(loop.getLowerBound());
        auto step = getConstantIntValue(loop.getStep());
        if (!ub || !lb || !step || *step == 0) {
          continue;
        }
        totalTrip *= (*ub - *lb) / *step;
      }

      // Only unroll nests with more than 1 tile.
      if (totalTrip <= 1) {
        continue;
      }

      if (failed(unrollTileLoopNestAndAssignDST(nest))) {
        return signalPassFailure();
      }
    }

    // Step 3: Reorder tile_store ops to be after DST wait barriers.
    func.walk([](Block *block) { reorderStoresAfterSync(block); });
  }
};

} // namespace

} // namespace mlir::tt::ttl
