// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/LowerMatmulCompute.h"

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
#include "mlir/IR/BuiltinOps.h"

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
/// Compute the iteration domain from the ComputeOp's TilingInterface.
/// This correctly handles matmul's 3D iteration space (M, N, K) where the
/// iteration domain exceeds the operand rank due to reduction dimensions.
static SmallVector<Range> getIterationDomain(OpBuilder &b, ComputeOp op) {
  return op.getIterationDomain(b);
}

/// Generate side-effect-only loop body. Extracts tiles from inputs, clones
/// compute body ops, and returns nothing (stores are explicit side effects).
static LogicalResult generateTileProcessing(OpBuilder &b, Location loc,
                                            ComputeOp op,
                                            ArrayRef<AffineMap> indexingMaps,
                                            ValueRange ivs) {
  size_t numInputs = op.getInputs().size();
  auto extractedInputs =
      extractTilesAtIndices(b, loc, op.getInputs(), indexingMaps, ivs);
  auto extractedOutputs = extractTilesAtIndices(b, loc, op.getOutputs(),
                                                indexingMaps, ivs, numInputs);

  Block &bodyBlock = op.getBody().front();
  IRMapping mapping;
  mapComputeBodyArgs(mapping, op, extractedInputs, extractedOutputs, ivs);

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

/// Generate parallel-outer / reduction-inner loop structure for accumulating
/// computes. DstSectionOp wraps the reduction loop + stores so DST persists
/// across reduction iterations.
///
/// Structure:
///   for each parallel dim:
///     dst_section {
///       for each reduction dim:
///         <tile ops from body>
///       <stores with placeholder tile + explicit dst_index>
///     }
static scf::LoopNest generateAccumulatingLoops(
    PatternRewriter &rewriter, Location loc, ComputeOp op,
    ArrayRef<Range> iterDomain, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<StringAttr> iterTypes, ArrayRef<Value> lowerBounds,
    ArrayRef<Value> upperBounds, ArrayRef<Value> steps) {

  // Separate parallel and reduction dim indices.
  SmallVector<unsigned> parallelDims, reductionDims;
  for (auto [idx, iterType] : llvm::enumerate(iterTypes)) {
    if (iterType.getValue() == "reduction") {
      reductionDims.push_back(idx);
    } else {
      parallelDims.push_back(idx);
    }
  }
  assert(!reductionDims.empty() && "accumulating compute must have reductions");

  // Build bounds for parallel and reduction loops separately.
  auto gatherBounds = [&](ArrayRef<unsigned> dims) {
    SmallVector<Value> lbs, ubs, sts;
    for (unsigned dim : dims) {
      lbs.push_back(lowerBounds[dim]);
      ubs.push_back(upperBounds[dim]);
      sts.push_back(steps[dim]);
    }
    return std::make_tuple(lbs, ubs, sts);
  };

  SmallVector<Value> parLBs, parUBs, parSteps;
  std::tie(parLBs, parUBs, parSteps) = gatherBounds(parallelDims);
  SmallVector<Value> redLBs, redUBs, redSteps;
  std::tie(redLBs, redUBs, redSteps) = gatherBounds(reductionDims);

  // Compute linearization strides from the full iteration domain.
  SmallVector<int64_t> domainSizes;
  for (auto &range : iterDomain) {
    auto size = getConstantIntValue(range.size);
    assert(size && "iteration domain must have static sizes");
    domainSizes.push_back(*size);
  }
  SmallVector<int64_t> domainStrides = computeStrides(domainSizes);

  // Collect store ops and their dst_index from the compute body.
  Block &bodyBlock = op.getBody().front();
  SmallVector<std::pair<TileStoreOp, int32_t>> storeInfos;
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (auto store = dyn_cast<TileStoreOp>(&bodyOp)) {
      auto dstIdx = getConstantIntValue(store.getDstIndex());
      storeInfos.emplace_back(store,
                              dstIdx ? static_cast<int32_t>(*dstIdx) : 0);
    }
  }

  size_t numDims = iterTypes.size();

  // Build the full IV vector from parallel + reduction IVs, placed at
  // their original dimension positions.
  auto buildFullIVs = [&](ValueRange parallelIVs,
                          ValueRange reductionIVs) -> SmallVector<Value> {
    SmallVector<Value> fullIVs(numDims);
    for (auto [idx, dim] : llvm::enumerate(parallelDims)) {
      fullIVs[dim] = parallelIVs[idx];
    }
    for (auto [idx, dim] : llvm::enumerate(reductionDims)) {
      fullIVs[dim] = reductionIVs[idx];
    }
    return fullIVs;
  };

  // Generate tile ops (excluding stores) inside the reduction loop body.
  auto generateTileOpsOnly = [&](OpBuilder &builder, Location bodyLoc,
                                 ValueRange fullIVs) {
    size_t numInputs = op.getInputs().size();
    auto extractedInputs = extractTilesAtIndices(
        builder, bodyLoc, op.getInputs(), indexingMaps, fullIVs);
    auto extractedOutputs = extractTilesAtIndices(
        builder, bodyLoc, op.getOutputs(), indexingMaps, fullIVs, numInputs);

    IRMapping mapping;
    mapComputeBodyArgs(mapping, op, extractedInputs, extractedOutputs, fullIVs);

    for (Operation &bodyOp : bodyBlock.without_terminator()) {
      if (isa<IterIndexOp, TileStoreOp>(&bodyOp)) {
        continue;
      }
      builder.clone(bodyOp, mapping);
    }
  };

  // Outer: parallel loops.
  scf::LoopNest outerNest = scf::buildLoopNest(
      rewriter, loc, parLBs, parUBs, parSteps, ValueRange{},
      [&](OpBuilder &parBuilder, Location parLoc, ValueRange parallelIVs,
          ValueRange) -> scf::ValueVector {
        // DstSectionOp wraps reduction loop + stores.
        auto dstSection = DstSectionOp::create(parBuilder, parLoc);
        Block &sectionBody = dstSection.getBody().front();
        OpBuilder secBuilder(&sectionBody,
                             Block::iterator(sectionBody.getTerminator()));

        // Inner: reduction loops.
        scf::LoopNest redNest = scf::buildLoopNest(
            secBuilder, parLoc, redLBs, redUBs, redSteps, ValueRange{},
            [&](OpBuilder &redBuilder, Location redLoc, ValueRange reductionIVs,
                ValueRange) -> scf::ValueVector {
              SmallVector<Value> fullIVs =
                  buildFullIVs(parallelIVs, reductionIVs);
              generateTileOpsOnly(redBuilder, redLoc, fullIVs);
              return {};
            });

        // Annotate reduction loops.
        for (auto [idx, loop] : llvm::enumerate(redNest.loops)) {
          unsigned origDim = reductionDims[idx];
          loop->setAttr(kTileLoopStrideAttrName,
                        parBuilder.getIndexAttr(domainStrides[origDim]));
          loop->setAttr(kReductionLoopAttrName, parBuilder.getUnitAttr());
        }

        // Stores after the reduction loop, inside the DstSectionOp.
        // Use placeholder tile value + explicit dst_index (same as matmul).
        OpBuilder storeBuilder(&sectionBody,
                               Block::iterator(sectionBody.getTerminator()));
        for (auto &[origStore, dstIdx] : storeInfos) {
          // Get the output tile type for the placeholder.
          Type tileType = origStore.getTile().getType();
          Value placeholder = UnrealizedConversionCastOp::create(
                                  storeBuilder, parLoc, tileType, ValueRange{})
                                  .getResult(0);

          // Compute store indices from parallel IVs using the output map.
          // Reduction dims in the output map are constants (e.g., 0),
          // so we need the full IV vector. Use constant 0 for reduction IVs.
          SmallVector<Value> fullIVs(numDims);
          for (auto [idx, dim] : llvm::enumerate(parallelDims)) {
            fullIVs[dim] = parallelIVs[idx];
          }
          Value zeroIdx =
              arith::ConstantIndexOp::create(storeBuilder, parLoc, 0);
          for (unsigned dim : reductionDims) {
            fullIVs[dim] = zeroIdx;
          }

          size_t numInputs = op.getInputs().size();
          assert(op.getOutputs().size() == 1 &&
                 "multi-output accumulating computes not yet supported");
          size_t outputIdx = 0;
          SmallVector<Value> storeIndices =
              applyIndexingMap(storeBuilder, parLoc,
                               indexingMaps[numInputs + outputIdx], fullIVs);

          Value dstIdxVal =
              arith::ConstantIndexOp::create(storeBuilder, parLoc, dstIdx);
          TileStoreOp::create(storeBuilder, parLoc, placeholder,
                              origStore.getView(), storeIndices, dstIdxVal);
        }

        return {};
      });

  // Annotate parallel loops with strides.
  for (auto [idx, loop] : llvm::enumerate(outerNest.loops)) {
    unsigned origDim = parallelDims[idx];
    loop->setAttr(kTileLoopStrideAttrName,
                  rewriter.getIndexAttr(domainStrides[origDim]));
  }

  return outerNest;
}

struct LowerComputeToLoops : OpRewritePattern<ComputeOp> {
  using OpRewritePattern<ComputeOp>::OpRewritePattern;

  /// Outermost tile loops from subblocked computes that need unrolling.
  /// Populated during pattern application, consumed by runOnOperation.
  SmallVector<scf::ForOp> &loopsToUnroll;
  bool dstAccumulation;
  bool useBlockMatmul;

  LowerComputeToLoops(MLIRContext *ctx, SmallVector<scf::ForOp> &loopsToUnroll,
                      bool dstAccumulation, bool useBlockMatmul)
      : OpRewritePattern<ComputeOp>(ctx), loopsToUnroll(loopsToUnroll),
        dstAccumulation(dstAccumulation), useBlockMatmul(useBlockMatmul) {}

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

    bool isSubblocked = op->hasAttr(kFullLinStridesAttrName);
    bool isAccumulating = op.getBody()
                              .walk([](Operation *inner) {
                                return inner->hasTrait<TTLAccumulatingOpTrait>()
                                           ? WalkResult::interrupt()
                                           : WalkResult::advance();
                              })
                              .wasInterrupted();

    SmallVector<StringAttr> iterTypes;
    for (Attribute attr : op.getIteratorTypes()) {
      iterTypes.push_back(mlir::cast<StringAttr>(attr));
    }

    // Block-level matmul: a single DstSectionOp with the matmul_block call,
    // per-tile post-ops, and per-tile stores. When useBlockMatmul is false,
    // the compute falls through to per-tile loop lowering (matmul_tile).
    if (useBlockMatmul && op.containsOp<TileMatmulBlockOp>()) {
      return generateMatmulCompute(rewriter, loc, op, indexingMaps, iterTypes);
    }

    // Side-effect-only loops: no iter_args, no tensor.insert, no scf.yield
    // with tensor values. Stores are explicit side effects (tile_store).
    bool processingFailed = false;
    bool usedDstAccumulation = false;
    scf::LoopNest loopNest = [&]() {
      if (isSubblocked) {
        // Subblocked: DstSectionOp wraps the entire loop nest. After
        // unrolling, all tile ops share one DST register section.
        auto dstSection = DstSectionOp::create(rewriter, loc);
        Block &sectionBody = dstSection.getBody().front();
        OpBuilder sectionBuilder(&sectionBody,
                                 Block::iterator(sectionBody.getTerminator()));
        return scf::buildLoopNest(
            sectionBuilder, loc, lowerBounds, upperBounds, steps, ValueRange{},
            [&](OpBuilder &nested, Location nestedLoc, ValueRange ivs,
                ValueRange) -> scf::ValueVector {
              if (failed(generateTileProcessing(nested, nestedLoc, op,
                                                indexingMaps, ivs))) {
                processingFailed = true;
              }
              return {};
            });
      }

      // DST accumulation: reorder loops (parallel-outer, reduction-inner)
      // so DST persists across reduction iterations. Required for
      // reduce_max because L1 accumulation (pack_reconfig_l1_acc)
      // accumulates via addition, which is only correct for sum.
      // TODO: reduce_max without dst-accumulation could use a compiler-
      // introduced intermediate DFB for L1-based max accumulation.
      if (isAccumulating) {
        bool hasReduceMax =
            op.getBody()
                .walk([](TileReduceOp reduce) {
                  return reduce.getReduceType() == ReduceType::Max
                             ? WalkResult::interrupt()
                             : WalkResult::advance();
                })
                .wasInterrupted();
        if (dstAccumulation || hasReduceMax) {
          usedDstAccumulation = true;
          return generateAccumulatingLoops(rewriter, loc, op, iterDomain,
                                           indexingMaps, iterTypes, lowerBounds,
                                           upperBounds, steps);
        }
      }

      // Non-subblocked: DstSectionOp inside each loop iteration. Each
      // iteration gets its own DST register section.
      return scf::buildLoopNest(
          rewriter, loc, lowerBounds, upperBounds, steps, ValueRange{},
          [&](OpBuilder &nested, Location nestedLoc, ValueRange ivs,
              ValueRange) -> scf::ValueVector {
            auto dstSection = DstSectionOp::create(nested, nestedLoc);
            Block &sectionBody = dstSection.getBody().front();
            OpBuilder bodyBuilder(&sectionBody,
                                  Block::iterator(sectionBody.getTerminator()));
            if (failed(generateTileProcessing(bodyBuilder, nestedLoc, op,
                                              indexingMaps, ivs))) {
              processingFailed = true;
            }
            return {};
          });
    }();

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
    if (!usedDstAccumulation) {
      // Loops are in declaration order, matching iterTypes.
      for (auto [idx, loop] : llvm::enumerate(loopNest.loops)) {
        int64_t stride =
            fullStridesAttr ? fullStridesAttr[idx] : domainStrides[idx];
        loop->setAttr(kTileLoopStrideAttrName, rewriter.getIndexAttr(stride));
        if (iterTypes[idx].getValue() == "reduction") {
          loop->setAttr(kReductionLoopAttrName, rewriter.getUnitAttr());
        }
      }
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
    if (auto dstVal = getTileOpDstIndex(op)) {
      if (auto constIdx = foldIndexToConstant(*dstVal)) {
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

    int64_t dstBase = tileIdx * dstPerIteration;

    // Offset dst_index SSA operand so each unrolled tile occupies a
    // unique DST register.
    if (dstBase != 0) {
      if (auto oldDst = getTileOpDstIndex(op)) {
        OpBuilder b(op);
        Value offsetVal =
            arith::ConstantIndexOp::create(b, op->getLoc(), dstBase);
        Value newDst =
            arith::AddIOp::create(b, op->getLoc(), *oldDst, offsetVal);
        setTileOpDstIndex(op, newDst);
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
    patterns.add<LowerComputeToLoops>(func.getContext(), loopsToUnroll,
                                      dstAccumulation, useBlockMatmul);
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

    // Step 3: After subblock unrolling, group pack-phase ops at the end
    // of each DstSectionOp body. Only needed when unrolling produced
    // interleaved tile ops and stores. Safe because DST allocation
    // assigns distinct registers to each output tile.
    if (!loopsToUnroll.empty()) {
      bool dstCheckFailed = false;
      func.walk([&](DstSectionOp dstSection) {
        if (dstCheckFailed) {
          return;
        }
        Block &body = dstSection.getBody().front();
        SmallVector<Operation *> packOps;
        for (Operation &op : body.without_terminator()) {
          if (isa<TileStoreOp, CBPushOp>(&op)) {
            packOps.push_back(&op);
          }
        }
        // Skip bodies with 0-1 stores (no interleaving to fix).
        int64_t storeCount = llvm::count_if(
            packOps, [](Operation *op) { return isa<TileStoreOp>(op); });
        if (storeCount <= 1) {
          return;
        }

        Operation *yield = body.getTerminator();
        for (Operation *packOp : packOps) {
          packOp->moveBefore(yield);
        }
      });
      if (dstCheckFailed) {
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
  }
};

} // namespace

} // namespace mlir::tt::ttl
