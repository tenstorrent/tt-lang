// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <limits>
#include <optional>

namespace mlir::tt::ttl {

namespace {

/// Return a constant integer through index casts produced by frontend code.
/// `scf::ForOp::getStaticTripCount` does not fold these casts, but the tensor
/// recurrence rewrite requires the exact contribution window size.
static std::optional<int64_t> getConstantIntThroughIndexCast(Value value) {
  if (auto constantValue = getConstantIntValue(value)) {
    return *constantValue;
  }
  if (auto indexCast = value.getDefiningOp<arith::IndexCastOp>()) {
    return getConstantIntThroughIndexCast(indexCast.getIn());
  }
  return std::nullopt;
}

/// Return the dataflow buffer wait that provides the loop-local contribution.
/// DST lowering consumes all contributions with one coalesced wait; accepting
/// any derived tensor here would delete work that must execute per iteration.
static CBWaitOp getLoopLocalContributionWait(TensorAccumulationMatch &match,
                                             scf::ForOp loop,
                                             AttachCBOp &attachedContribution) {
  Value contribution = match.contribution;
  if (auto attach = contribution.getDefiningOp<AttachCBOp>()) {
    if (attach->getParentOp() != loop) {
      return {};
    }
    attachedContribution = attach;
    contribution = attach.getTensor();
  }

  auto wait = contribution.getDefiningOp<CBWaitOp>();
  if (!wait || wait->getParentOp() != loop) {
    return {};
  }
  return wait;
}

/// Check that deleting the source loop will not drop work other than the
/// additive recurrence itself. The only permitted side effect is the pop that
/// releases the matched contribution window after the add consumes it.
static bool onlyContainsDstReductionOps(scf::ForOp loop,
                                        TensorAccumulationMatch &match,
                                        CBWaitOp contributionWait,
                                        AttachCBOp attachedContribution) {
  DenseSet<Operation *> allowedOps;
  allowedOps.insert(match.add.getOperation());
  allowedOps.insert(contributionWait.getOperation());
  if (attachedContribution) {
    allowedOps.insert(attachedContribution.getOperation());
  }

  bool foundContributionPop = false;
  for (Operation &bodyOp : loop.getBody()->without_terminator()) {
    if (allowedOps.contains(&bodyOp)) {
      continue;
    }
    auto contributionPop = dyn_cast<CBPopOp>(&bodyOp);
    if (!contributionPop ||
        contributionPop.getCb() != contributionWait.getCb() ||
        contributionPop.getNumTiles() || foundContributionPop ||
        contributionPop->isBeforeInBlock(match.add)) {
      return false;
    }
    foundContributionPop = true;
  }
  return true;
}

/// Add the folded loop trip count as the leading tensor dimension. The
/// reduction compute indexes this dimension with the reduction IV and indexes
/// the original tensor dimensions with the parallel IVs.
static RankedTensorType
buildCoalescedContributionType(RankedTensorType unitType, int64_t tripCount) {
  SmallVector<int64_t> coalescedShape;
  coalescedShape.push_back(tripCount);
  llvm::append_range(coalescedShape, unitType.getShape());
  return RankedTensorType::get(coalescedShape, unitType.getElementType());
}

/// Build maps for operands `[initial, coalescedContribution]` and the single
/// output. The initial value and output ignore the reduction dimension; the
/// contribution uses it as the leading coordinate.
static SmallVector<Attribute>
buildDstReductionIndexingMaps(MLIRContext *context, int64_t outputRank) {
  int64_t domainRank = outputRank + 1;
  SmallVector<AffineExpr> parallelExprs;
  for (int64_t dim = 0; dim < outputRank; ++dim) {
    parallelExprs.push_back(getAffineDimExpr(dim, context));
  }

  SmallVector<AffineExpr> contributionExprs;
  contributionExprs.push_back(getAffineDimExpr(outputRank, context));
  llvm::append_range(contributionExprs, parallelExprs);

  AffineMap outputMap = AffineMap::get(domainRank, 0, parallelExprs, context);
  AffineMap contributionMap =
      AffineMap::get(domainRank, 0, contributionExprs, context);
  return {AffineMapAttr::get(outputMap), AffineMapAttr::get(contributionMap),
          AffineMapAttr::get(outputMap)};
}

/// Use one trailing reduction iterator so the later loop lowering can keep DST
/// live across all contribution tiles for the same output coordinate.
static SmallVector<Attribute>
buildDstReductionIteratorTypes(RewriterBase &rewriter, int64_t outputRank) {
  SmallVector<Attribute> iteratorTypes;
  for (int64_t dim = 0; dim < outputRank; ++dim) {
    iteratorTypes.push_back(rewriter.getStringAttr("parallel"));
  }
  iteratorTypes.push_back(rewriter.getStringAttr("reduction"));
  return iteratorTypes;
}

} // namespace

std::optional<int64_t> getStaticAccumulationTripCount(scf::ForOp loop) {
  if (std::optional<llvm::APInt> tripCount = loop.getStaticTripCount()) {
    if (tripCount->getActiveBits() > 63) {
      return std::nullopt;
    }
    return static_cast<int64_t>(tripCount->getZExtValue());
  }

  std::optional<int64_t> lowerBound =
      getConstantIntThroughIndexCast(loop.getLowerBound());
  std::optional<int64_t> upperBound =
      getConstantIntThroughIndexCast(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntThroughIndexCast(loop.getStep());
  if (!lowerBound || !upperBound || !step || *step <= 0) {
    return std::nullopt;
  }

  if (*upperBound <= *lowerBound) {
    return 0;
  }

  uint64_t distance =
      static_cast<uint64_t>(*upperBound) - static_cast<uint64_t>(*lowerBound);
  uint64_t stepValue = static_cast<uint64_t>(*step);
  uint64_t tripCount = 1 + (distance - 1) / stepValue;
  if (tripCount > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  return static_cast<int64_t>(tripCount);
}

FailureOr<int64_t> getStaticTensorTileCount(RankedTensorType tensorType) {
  if (!tensorType.hasStaticShape()) {
    return failure();
  }

  int64_t tileCount = 1;
  for (int64_t dim : tensorType.getShape()) {
    if (dim < 0 ||
        (dim != 0 && tileCount > std::numeric_limits<int64_t>::max() / dim)) {
      return failure();
    }
    tileCount *= dim;
  }
  return tileCount;
}

FailureOr<TensorAccumulationMatch> matchAdditiveTensorAccumulation(
    scf::ForOp loop, unsigned resultIndex,
    TensorAccumulationReservePlacement reservePlacement,
    ArrayRef<Operation *> allowedReserveUsers,
    ArrayRef<Operation *> allowedLoopResultUsers) {
  if (resultIndex >= loop.getNumResults()) {
    return failure();
  }

  Value loopResult = loop.getResult(resultIndex);
  llvm::SmallPtrSet<Operation *, 2> permittedLoopResultUsers;
  for (Operation *operation : allowedLoopResultUsers) {
    permittedLoopResultUsers.insert(operation);
  }

  StoreOp finalStore;
  for (Operation *user : loopResult.getUsers()) {
    if (permittedLoopResultUsers.contains(user)) {
      continue;
    }
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store.getAccumulate() || finalStore) {
      return failure();
    }
    finalStore = store;
  }
  if (!finalStore) {
    return failure();
  }

  // The yielded add must be the only producer of the carried state so the
  // rewrite can replace the loop with a single reduction compute.
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto add = yield.getOperand(resultIndex).getDefiningOp<AddOp>();
  if (!add || add->getBlock() != loop.getBody() ||
      !add.getResult().hasOneUse()) {
    return failure();
  }

  BlockArgument iterArg = loop.getRegionIterArgs()[resultIndex];
  Value contribution;
  if (add.getLhs() == iterArg) {
    contribution = add.getRhs();
  } else if (add.getRhs() == iterArg) {
    contribution = add.getLhs();
  } else {
    return failure();
  }
  if (contribution == iterArg) {
    return failure();
  }

  // Additional uses of the carried value would require preserving a second
  // per-iteration data dependence after the loop has been deleted.
  for (OpOperand &use : iterArg.getUses()) {
    if (use.getOwner() != add.getOperation()) {
      return failure();
    }
  }

  auto reserve = finalStore.getView().getDefiningOp<CBReserveOp>();
  if (!reserve) {
    return failure();
  }

  llvm::SmallPtrSet<Operation *, 4> permittedReserveUsers;
  for (Operation *operation : allowedReserveUsers) {
    permittedReserveUsers.insert(operation);
  }

  SmallVector<AttachCBOp> deadReserveAttachOps;
  for (OpOperand &reserveUse : reserve.getResult().getUses()) {
    Operation *owner = reserveUse.getOwner();
    if (owner == finalStore.getOperation() ||
        permittedReserveUsers.contains(owner)) {
      continue;
    }

    auto attach = dyn_cast<AttachCBOp>(owner);
    if (!attach || !attach.getResult().use_empty()) {
      return failure();
    }
    deadReserveAttachOps.push_back(attach);
  }

  if (finalStore->getBlock() != loop->getBlock() ||
      (reservePlacement == TensorAccumulationReservePlacement::SameBlock &&
       reserve->getBlock() != loop->getBlock())) {
    return failure();
  }

  Value initialValue = loop.getInitArgs()[resultIndex];
  auto tensorType = cast<RankedTensorType>(initialValue.getType());
  return TensorAccumulationMatch{
      resultIndex, tensorType, initialValue, finalStore,
      reserve,     add,        contribution, deadReserveAttachOps};
}

FailureOr<TensorDstAccumulationInfo>
analyzeTensorAccumulationForDst(TensorAccumulationMatch &match,
                                scf::ForOp loop) {
  if (match.contribution.getType() != match.tensorType) {
    return failure();
  }

  if (!getAttachedCB(match.initialValue)) {
    return failure();
  }

  std::optional<int64_t> tripCount = getStaticAccumulationTripCount(loop);
  if (!tripCount || *tripCount == 0) {
    return failure();
  }
  int64_t tripCountValue = *tripCount;

  AttachCBOp attachedContribution;
  CBWaitOp contributionWait =
      getLoopLocalContributionWait(match, loop, attachedContribution);
  if (!contributionWait || contributionWait.getNumTiles().has_value()) {
    return failure();
  }
  if (!onlyContainsDstReductionOps(loop, match, contributionWait,
                                   attachedContribution)) {
    return failure();
  }

  auto contributionType =
      dyn_cast<RankedTensorType>(contributionWait.getResult().getType());
  if (!contributionType || contributionType != match.tensorType ||
      !contributionType.hasStaticShape()) {
    return failure();
  }

  FailureOr<int64_t> unitTileCount = getStaticTensorTileCount(contributionType);
  if (failed(unitTileCount) || *unitTileCount <= 0 ||
      tripCountValue > std::numeric_limits<int64_t>::max() / *unitTileCount) {
    return failure();
  }
  int64_t totalContributionTiles = tripCountValue * *unitTileCount;
  auto contributionCBType =
      cast<CircularBufferType>(contributionWait.getCb().getType());
  // The coalesced wait spans the full recurrence, so the dataflow buffer must
  // already contain the complete contribution window.
  if (totalContributionTiles > contributionCBType.getTotalElements()) {
    return failure();
  }

  return TensorDstAccumulationInfo{tripCountValue,         *unitTileCount,
                                   totalContributionTiles, contributionWait,
                                   attachedContribution,   contributionType};
}

LogicalResult lowerTensorAccumulationToDst(TensorAccumulationMatch &match,
                                           scf::ForOp loop,
                                           RewriterBase &rewriter) {
  FailureOr<TensorDstAccumulationInfo> info =
      analyzeTensorAccumulationForDst(match, loop);
  if (failed(info)) {
    return failure();
  }

  Location loc = loop.getLoc();
  CBReserveOp outputReserve = match.reserve;
  if (outputReserve->getBlock() == loop->getBlock() &&
      !outputReserve->isBeforeInBlock(loop)) {
    rewriter.moveOpBefore(outputReserve, loop);
  }

  rewriter.setInsertionPoint(loop);
  IntegerAttr totalTilesAttr =
      rewriter.getI64IntegerAttr(info->totalContributionTiles);
  RankedTensorType coalescedType =
      buildCoalescedContributionType(info->contributionType, info->tripCount);
  CBWaitOp coalescedWait =
      CBWaitOp::create(rewriter, loc, coalescedType,
                       info->contributionWait.getCb(), totalTilesAttr);
  AttachCBOp coalescedContribution = AttachCBOp::create(
      rewriter, loc, coalescedType, coalescedWait.getResult(),
      info->contributionWait.getCb());

  // The output value exists only to carry the dataflow buffer attachment into
  // ttl.compute. Tile stores use the reserved dataflow buffer directly.
  Value outputInit =
      tensor::EmptyOp::create(rewriter, loc, match.tensorType.getShape(),
                              match.tensorType.getElementType());
  Value output = AttachCBOp::create(rewriter, loc, match.tensorType, outputInit,
                                    outputReserve.getCb());

  SmallVector<Attribute> indexingMaps = buildDstReductionIndexingMaps(
      rewriter.getContext(), match.tensorType.getRank());
  SmallVector<Attribute> iteratorTypes =
      buildDstReductionIteratorTypes(rewriter, match.tensorType.getRank());

  auto compute = ComputeOp::create(
      rewriter, loc, TypeRange{match.tensorType},
      ValueRange{match.initialValue, coalescedContribution.getResult()},
      ValueRange{output}, rewriter.getArrayAttr(indexingMaps),
      rewriter.getArrayAttr(iteratorTypes));

  Block *body = rewriter.createBlock(&compute.getBody());
  Type tileType = match.tensorType.getElementType();
  body->addArgument(tileType, loc);
  body->addArgument(tileType, loc);
  body->addArgument(tileType, loc);

  rewriter.setInsertionPointToStart(body);
  SmallVector<Value> outputIndices;
  for (int64_t dim = 0; dim < match.tensorType.getRank(); ++dim) {
    outputIndices.push_back(
        IterIndexOp::create(rewriter, loc, rewriter.getI64IntegerAttr(dim)));
  }

  // tile_accumulate models the in-place DST recurrence. The contribution tile
  // remains dataflow-buffer backed and is consumed directly during TTKernel
  // lowering.
  auto accumulated = createTileOpWithPlaceholderDstIndex<TileAccumulateOp>(
      rewriter, loc, body->getArgument(0), body->getArgument(1),
      AccumulationCombinerAttr::get(rewriter.getContext(),
                                    AccumulationCombiner::Add));
  createTileOpWithPlaceholderDstIndex<TileStoreOp>(
      rewriter, loc, accumulated.getResult(), outputReserve.getResult(),
      outputIndices);
  YieldOp::create(rewriter, loc);

  rewriter.setInsertionPointAfter(compute);
  CBPopOp::create(rewriter, loc, info->contributionWait.getCb(),
                  totalTilesAttr);

  rewriter.eraseOp(match.finalStore);
  for (AttachCBOp attach : match.deadReserveAttachOps) {
    rewriter.eraseOp(attach);
  }
  rewriter.eraseOp(loop);
  return success();
}

} // namespace mlir::tt::ttl
