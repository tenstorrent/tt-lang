// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
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

  // Use the largest operand's shape for loop bounds so that broadcast
  // dimensions (size 1 in the smaller operand) still get iterated.
  // Prefer higher rank, then larger element count for same rank.
  Value maxRankTensor;
  int64_t maxRank = 0;
  int64_t maxElements = 0;
  for (Value operand : llvm::concat<Value>(op.getInputs(), op.getOutputs())) {
    auto type = cast<RankedTensorType>(operand.getType());
    int64_t rank = type.getRank();
    // ComputeOp verifier guarantees static shapes, so getNumElements is safe to
    // use here.
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
      size = tensor::DimOp::create(b, loc, maxRankTensor, i).getResult();
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
  SmallVector<Value> extractedInputs;
  for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
    SmallVector<Value> indices =
        applyIndexingMap(b, loc, indexingMaps[idx], ivs);
    Value tile = tensor::ExtractOp::create(b, loc, input, indices);
    extractedInputs.push_back(tile);
  }

  // Output block args get a dummy extract from the output tensor. These are
  // needed for SSA mapping but unused in the body (stores write via DST).
  SmallVector<Value> extractedOutputs;
  size_t numInputs = op.getInputs().size();
  for (auto [idx, output] : llvm::enumerate(op.getOutputs())) {
    SmallVector<Value> indices =
        applyIndexingMap(b, loc, indexingMaps[numInputs + idx], ivs);
    Value tile = tensor::ExtractOp::create(b, loc, output, indices);
    extractedOutputs.push_back(tile);
  }

  Block &bodyBlock = op.getBody().front();
  IRMapping mapping;
  for (auto [idx, arg] : llvm::enumerate(op.getInputs())) {
    mapping.map(bodyBlock.getArgument(idx), extractedInputs[idx]);
  }
  for (auto [idx, arg] : llvm::enumerate(op.getOutputs())) {
    mapping.map(bodyBlock.getArgument(numInputs + idx), extractedOutputs[idx]);
  }

  // Resolve iter_index ops to loop IVs via the IRMapping.
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (auto iterIdx = dyn_cast<IterIndexOp>(&bodyOp)) {
      int64_t dim = iterIdx.getDim();
      // IterIndexOp verifier guarantees dim < iteration domain rank,
      // which equals ivs.size() (loops from the same domain).
      assert(dim < static_cast<int64_t>(ivs.size()) &&
             "iter_index dim out of range for loop IVs");
      mapping.map(iterIdx.getResult(), ivs[dim]);
    }
  }

  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    // iter_index ops are resolved via the mapping -- skip the original ops.
    if (isa<IterIndexOp>(&bodyOp)) {
      continue;
    }

    b.clone(bodyOp, mapping);
  }

  return success();
}

struct LowerComputeToLoops : OpRewritePattern<ComputeOp> {
  using OpRewritePattern<ComputeOp>::OpRewritePattern;

  /// Outermost tile loops from subblocked computes that need unrolling.
  /// Populated during pattern application, consumed by runOnOperation.
  SmallVector<scf::ForOp> &loopsToUnroll;

  LowerComputeToLoops(MLIRContext *ctx, SmallVector<scf::ForOp> &loopsToUnroll)
      : OpRewritePattern<ComputeOp>(ctx), loopsToUnroll(loopsToUnroll) {}

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
    // If the compute was subblocked, use the full block strides (which
    // differ from the subblock iteration bounds). Otherwise, compute strides
    // from the iteration domain (which IS the full block shape).
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
      loop->setAttr(kTileLoopStrideAttrName, rewriter.getIndexAttr(stride));
    }

    // Record the outermost tile loop for unrolling if the compute was
    // subblocked (has full linearization strides). Non-subblocked computes
    // keep their tile loops for per-tile sync.
    if (fullStridesAttr && !loopNest.loops.empty()) {
      loopsToUnroll.push_back(loopNest.loops.front());
    }

    if (processingFailed) {
      return rewriter.notifyMatchFailure(
          op, "copy_tile index computation failed (mismatched rank/IVs)");
    }

    rewriter.replaceOp(op, op.getOutputs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Post-pattern tile loop unrolling and DST index assignment
//===----------------------------------------------------------------------===//

/// Prefix for temporary per-dimension iteration index attributes. Op
/// attributes are the only viable mechanism here: loopUnrollByFactor's
/// annotateFn callback is the sole per-clone hook, and the IRMapping in
/// generateUnrolledLoop is not exposed through loopUnrollByFactor (nor
/// would it span sequential multi-dim unrolling). Each clone gets
/// `ttl._uiter_<dim> = i`; after all loops are unrolled the indices are
/// linearized into DST/CB offsets and the attributes are removed.
static constexpr llvm::StringLiteral kUnrollIterPrefix("ttl._uiter_");

/// Build the discardable attribute name for dimension `d`.
static std::string unrollIterAttrName(int64_t d) {
  return (kUnrollIterPrefix + llvm::Twine(d)).str();
}

/// Fully unroll a tile loop nest, then assign DST indices and CB tile offsets
/// to the unrolled ops.
///
/// loopUnrollByFactor eliminates loop structure but doesn't know about DST
/// registers or CB tile grids. We need to recover each clone's position in
/// the original iteration space so we can assign unique DST indices (to avoid
/// register collisions) and tile offsets (for CB indexing). Temporary
/// `ttl._uiter_<dim>` attributes carry this information across the unroll,
/// then are removed after linearization.
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

    auto strideAttr =
        nest[d]->getAttrOfType<IntegerAttr>(kTileLoopStrideAttrName);
    fullStrides[d] = strideAttr ? strideAttr.getInt() : 1;
  }

  if (totalTiles <= 1) {
    return success(); // Single iteration, nothing to unroll.
  }

  // Loops will be erased by unrolling; save the block for post-unroll walk.
  Block *enclosingBlock = nest.front()->getBlock();

  // Find the highest DST index used in a single iteration of the innermost
  // loop body. Each unrolled copy gets DST indices offset by
  // tileIdx * dstPerIteration to avoid register collisions.
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

  SmallVector<int64_t> localStrides = computeStrides(dimSizes);

  // Innermost-first: inner loops must be gone before outer loops can unroll.
  // loopUnrollByFactor is a no-op for trip count 1, so we fold those manually.
  for (int64_t d = rank - 1; d >= 0; --d) {
    std::string attrName = unrollIterAttrName(d);
    uint64_t tripCount = static_cast<uint64_t>(dimSizes[d]);

    if (tripCount <= 1) {
      // Trip count 1: manually fold the loop. Tag all body ops with iter 0,
      // replace IV with lower bound, inline body, erase loop.
      scf::ForOp loop = nest[d];
      loop.getInductionVar().replaceAllUsesWith(loop.getLowerBound());
      for (Operation &bodyOp : *loop.getBody()) {
        if (!bodyOp.hasTrait<OpTrait::IsTerminator>()) {
          bodyOp.walk([&attrName](Operation *inner) {
            inner->setAttr(
                attrName,
                IntegerAttr::get(IntegerType::get(inner->getContext(), 64), 0));
          });
        }
      }
      Block *parentBlock = loop->getBlock();
      Block *loopBody = loop.getBody();
      loopBody->getTerminator()->erase();
      parentBlock->getOperations().splice(Block::iterator(loop),
                                          loopBody->getOperations());
      loop->erase();
      continue;
    }

    // Full unroll: factor == tripCount, so no remainder loop is generated.
    // Trip counts are static and exact (guaranteed by the subblock pass).
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

  // Recover each unrolled op's position in the original iteration space
  // and linearize to assign DST indices and CB tile offsets.
  auto walkFn = [&](Operation *op) {
    SmallVector<int64_t> dimIndices(rank);
    for (int64_t d = 0; d < rank; ++d) {
      auto attr = op->getAttrOfType<IntegerAttr>(unrollIterAttrName(d));
      if (!attr) {
        return;
      }
      dimIndices[d] = attr.getInt();
    }

    // tileIdx: linearized using local (subblock) strides — determines DST
    // register position within the subblock.
    int64_t tileIdx = linearize(dimIndices, localStrides);

    // tileOffset: linearized using full block strides — determines CB tile
    // position within the entire block, used by computeCBTileIndex.
    int64_t tileOffset = linearize(dimIndices, fullStrides);

    int64_t dstBase = tileIdx * dstPerIteration;

    // Offset dst_idx so each unrolled tile occupies a unique DST register.
    if (auto attr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
      if (dstBase != 0) {
        int64_t newIdx = attr.getInt() + dstBase;
        op->setAttr(kDstIdxAttrName,
                    IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                     static_cast<int32_t>(newIdx)));
      }
    }

    // dst_idx attribute (above) covers tile compute ops. CopyTileOp's
    // dst index is an SSA value, so we must emit an op to compute the offset.
    if (auto copyTile = dyn_cast<CopyTileOp>(op)) {
      if (dstBase != 0) {
        OpBuilder b(copyTile);
        Value offsetVal =
            arith::ConstantIndexOp::create(b, copyTile.getLoc(), dstBase);
        Value newDstIndex = arith::AddIOp::create(
            b, copyTile.getLoc(), copyTile.getDstIndex(), offsetVal);
        copyTile.getDstIndexMutable().assign(newDstIndex);
      }
    }

    // Set tile_offset on TTL ops for CB index computation. The attribute is
    // consumed by computeCBTileIndex during TTL-to-TTKernel
    // conversion.
    if (auto *dialect = op->getDialect()) {
      if (dialect->getNamespace() == "ttl") {
        op->setAttr(
            kTileOffsetAttrName,
            IntegerAttr::get(IndexType::get(op->getContext()), tileOffset));
      }
    }

    for (int64_t d = 0; d < rank; ++d) {
      op->removeAttr(unrollIterAttrName(d));
    }
  };

  // Walk the enclosing block (saved before unrolling) to find tagged ops.
  // This handles both the non-subblocked case (ops directly in the function)
  // and the subblocked case (ops inside subblock scf.for loops).
  for (Operation &op : *enclosingBlock) {
    op.walk(walkFn);
  }

  // Remove stale annotations from ops that didn't have all dimensions
  // (e.g., arith constants duplicated by the unroller).
  for (Operation &op : *enclosingBlock) {
    op.walk([rank](Operation *inner) {
      for (int64_t d = 0; d < rank; ++d) {
        inner->removeAttr(unrollIterAttrName(d));
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

    // Step 1: Lower compute ops to scf.for tile loops.
    // The pattern collects outermost tile loops from subblocked computes
    // into loopsToUnroll for step 2.
    SmallVector<scf::ForOp> loopsToUnroll;
    RewritePatternSet patterns(func.getContext());
    patterns.add<LowerComputeToLoops>(func.getContext(), loopsToUnroll);
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(func, frozen))) {
      return signalPassFailure();
    }

    // Returns loops outermost-to-innermost.
    auto collectTileLoopNest = [](scf::ForOp outerLoop) {
      SmallVector<scf::ForOp> nest;
      scf::ForOp current = outerLoop;
      while (current) {
        nest.push_back(current);
        scf::ForOp inner = nullptr;
        for (Operation &op : *current.getBody()) {
          if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
            if (forOp->hasAttr(kTileLoopStrideAttrName)) {
              inner = forOp;
              break;
            }
          }
        }
        current = inner;
      }
      return nest;
    };

    for (scf::ForOp outerLoop : loopsToUnroll) {
      SmallVector<scf::ForOp> nest = collectTileLoopNest(outerLoop);

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

      if (totalTrip <= 1) {
        continue;
      }

      if (failed(unrollTileLoopNestAndAssignDST(nest))) {
        return signalPassFailure();
      }
    }

    // Verify no temporary unroll iteration attributes leaked past this pass.
    LLVM_DEBUG(func.walk([](Operation *op) {
      for (auto attr : op->getAttrs()) {
        assert(!attr.getName().getValue().starts_with(kUnrollIterPrefix) &&
               "temporary _uiter_ attribute not cleaned up after unrolling");
      }
    }));

    // Step 3: Reorder tile_store ops to be after DST wait barriers.
    func.walk([](Block *block) { reorderStoresAfterSync(block); });
  }
};

} // namespace

} // namespace mlir::tt::ttl
