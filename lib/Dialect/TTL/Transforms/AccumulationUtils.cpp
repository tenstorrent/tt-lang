// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <limits>
#include <optional>

namespace mlir::tt::ttl {

/// Return the next scope id by scanning existing explicit L1 accumulation ids.
///
/// Scope ids are local compiler metadata. Reusing the maximum existing id plus
/// one preserves independence when separate lowering passes annotate loops in
/// the same function.
int64_t getNextL1AccScopeId(Operation *root) {
  int64_t nextScopeId = 0;
  root->walk([&](Operation *operation) {
    auto attr = operation->getAttrOfType<IntegerAttr>(kL1AccScopeIdAttrName);
    if (!attr) {
      return;
    }
    int64_t candidateScopeId = attr.getInt() + 1;
    if (candidateScopeId > nextScopeId) {
      nextScopeId = candidateScopeId;
    }
  });
  return nextScopeId;
}

namespace {

/// Returns the number of tiles represented by a statically ranked tensor.
static int64_t getTileCount(RankedTensorType tensorType) {
  assert(tensorType.hasStaticShape() && "expected static tensor shape");
  int64_t tileCount = 1;
  for (int64_t dim : tensorType.getShape()) {
    tileCount *= dim;
  }
  return tileCount;
}

/// Finds the contribution wait only when it is immediately owned by the loop.
/// Ancestor containment would admit nested-region effects that cannot be
/// coalesced into one pre-loop wait without changing execution order.
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

/// Ensures replacing the whole loop with one reduction compute does not
/// discard side effects other than the matched wait/attach/add recurrence.
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

  for (Operation &bodyOp : loop.getBody()->without_terminator()) {
    if (!allowedOps.contains(&bodyOp)) {
      return false;
    }
  }
  return true;
}

/// Prepends the reduction dimension so the coalesced wait exposes every
/// per-iteration contribution as one compute input tensor.
static RankedTensorType
buildCoalescedContributionType(RankedTensorType unitType, int64_t tripCount) {
  SmallVector<int64_t> shape;
  shape.push_back(tripCount);
  llvm::append_range(shape, unitType.getShape());
  return RankedTensorType::get(shape, unitType.getElementType());
}

/// Builds maps for a reduction domain whose final dimension indexes the
/// coalesced contribution while the output and initial accumulator ignore it.
static SmallVector<Attribute>
buildDstReductionIndexingMaps(MLIRContext *ctx, int64_t outputRank) {
  int64_t domainRank = outputRank + 1;
  SmallVector<AffineExpr> parallelExprs;
  for (int64_t dim = 0; dim < outputRank; ++dim) {
    parallelExprs.push_back(getAffineDimExpr(dim, ctx));
  }

  SmallVector<AffineExpr> contributionExprs;
  contributionExprs.push_back(getAffineDimExpr(outputRank, ctx));
  llvm::append_range(contributionExprs, parallelExprs);

  AffineMap outputMap = AffineMap::get(domainRank, 0, parallelExprs, ctx);
  AffineMap contributionMap =
      AffineMap::get(domainRank, 0, contributionExprs, ctx);
  return {AffineMapAttr::get(outputMap), AffineMapAttr::get(contributionMap),
          AffineMapAttr::get(outputMap)};
}

/// Marks output dimensions parallel and appends the single reduction dimension
/// that drives repeated tile accumulation into DST.
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

bool isTensorLoopState(scf::ForOp loop, unsigned resultIndex) {
  return isa<RankedTensorType>(loop.getInitArgs()[resultIndex].getType());
}

FailureOr<TensorAccumulationMatch> matchAdditiveTensorAccumulation(
    scf::ForOp loop, unsigned resultIndex,
    TensorAccumulationReservePlacement reservePlacement,
    ArrayRef<Operation *> allowedReserveUsers) {
  if (resultIndex >= loop.getNumResults()) {
    return failure();
  }

  auto loopResult = loop.getResult(resultIndex);
  if (!loopResult.hasOneUse()) {
    return failure();
  }

  // The final non-accumulating store identifies the externally visible
  // destination. Accumulating stores already represent user-written DFB += and
  // are handled by a separate formation rule.
  auto finalStore = dyn_cast<StoreOp>(*loopResult.getUsers().begin());
  if (!finalStore || finalStore.getAccumulate()) {
    return failure();
  }

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

  // The iter_arg may only feed the additive recurrence. Additional uses would
  // require preserving the old loop-carried tensor value independently of the
  // selected accumulation strategy.
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
    if (owner == finalStore.getOperation()) {
      continue;
    }
    if (permittedReserveUsers.contains(owner)) {
      continue;
    }

    // Dead attach views are artifacts of earlier lowering. A live view means
    // the reservation participates in another dataflow use and cannot be owned
    // solely by the accumulation scope.
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

  auto tensorType =
      cast<RankedTensorType>(loop.getInitArgs()[resultIndex].getType());
  return TensorAccumulationMatch{resultIndex,
                                 tensorType,
                                 loop.getInitArgs()[resultIndex],
                                 iterArg,
                                 yield.getOperand(resultIndex),
                                 finalStore,
                                 reserve,
                                 add,
                                 contribution,
                                 deadReserveAttachOps};
}

LogicalResult lowerTensorAccumulationToDst(TensorAccumulationMatch &match,
                                           scf::ForOp loop,
                                           RewriterBase &rewriter) {
  if (match.contribution.getType() != match.tensorType) {
    return failure();
  }

  // The initial accumulator must already be dataflow-buffer backed because the
  // generated compute reads it as the first input before reusing DST.
  if (!getAttachedCB(match.initialValue)) {
    return failure();
  }

  // Coalescing replaces one wait per iteration with one pre-compute wait, so
  // the total tile count must be known at compile time.
  std::optional<llvm::APInt> tripCount = loop.getStaticTripCount();
  if (!tripCount || tripCount->isZero() || tripCount->getActiveBits() > 63) {
    return failure();
  }
  int64_t tripCountValue = static_cast<int64_t>(tripCount->getZExtValue());

  // The contribution wait must be the canonical one-tensor wait immediately in
  // the loop body. Explicit num_tiles would need separate accounting, and
  // nested waits cannot be hoisted without moving region-local effects.
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

  // The reduction compute preserves the original output tensor domain and adds
  // only a leading reduction dimension for the coalesced contributions.
  auto contributionType =
      dyn_cast<RankedTensorType>(contributionWait.getResult().getType());
  if (!contributionType || contributionType != match.tensorType ||
      !contributionType.hasStaticShape()) {
    return failure();
  }

  // The single coalesced wait must fit in the producer dataflow buffer. This is
  // a compile-time strategy selection, not a runtime capacity check.
  int64_t unitTileCount = getTileCount(contributionType);
  if (unitTileCount <= 0 ||
      tripCountValue > std::numeric_limits<int64_t>::max() / unitTileCount) {
    return failure();
  }
  int64_t totalContributionTiles = tripCountValue * unitTileCount;
  auto contributionCBType =
      cast<CircularBufferType>(contributionWait.getCb().getType());
  if (totalContributionTiles > contributionCBType.getTotalElements()) {
    return failure();
  }

  Location loc = loop.getLoc();
  CBReserveOp outputReserve = match.reserve;
  // The output reserve must dominate both the generated compute output view and
  // the tile stores that reuse its explicit DST indices.
  if (outputReserve->getBlock() == loop->getBlock() &&
      !outputReserve->isBeforeInBlock(loop)) {
    rewriter.moveOpBefore(outputReserve, loop);
  }

  rewriter.setInsertionPoint(loop);
  IntegerAttr totalTilesAttr =
      rewriter.getI64IntegerAttr(totalContributionTiles);
  RankedTensorType coalescedType =
      buildCoalescedContributionType(contributionType, tripCountValue);
  CBWaitOp coalescedWait = CBWaitOp::create(
      rewriter, loc, coalescedType, contributionWait.getCb(), totalTilesAttr);
  AttachCBOp coalescedContribution =
      AttachCBOp::create(rewriter, loc, coalescedType,
                         coalescedWait.getResult(), contributionWait.getCb());

  // The compute output operand is a tensor view of the reserved output
  // dataflow buffer. The placeholder tensor has no data dependence; it only
  // supplies the tensor type needed by AttachCBOp.
  Value outputInit =
      tensor::EmptyOp::create(rewriter, loc, match.tensorType.getShape(),
                              match.tensorType.getElementType());
  Value output = AttachCBOp::create(rewriter, loc, match.tensorType, outputInit,
                                    outputReserve.getCb());

  SmallVector<Attribute> indexingMaps = buildDstReductionIndexingMaps(
      rewriter.getContext(), match.tensorType.getRank());
  SmallVector<Attribute> iteratorTypes =
      buildDstReductionIteratorTypes(rewriter, match.tensorType.getRank());

  // Input 0 is the initial accumulator tile, input 1 is the contribution tile
  // for the current reduction iteration, and output 0 is the reserved DST slot.
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

  // TileAccumulateAddOp leaves the result in DST; the store records the final
  // output location without releasing DST between reduction iterations.
  auto accumulated = createTileOpWithPlaceholderDstIndex<TileAccumulateAddOp>(
      rewriter, loc, body->getArgument(0), body->getArgument(1));
  createTileOpWithPlaceholderDstIndex<TileStoreOp>(
      rewriter, loc, accumulated.getResult(), outputReserve.getResult(),
      outputIndices);
  YieldOp::create(rewriter, loc);

  rewriter.setInsertionPointAfter(compute);
  // The original per-iteration pops disappear with the loop, so the coalesced
  // wait needs one matching pop after compute consumes all contribution tiles.
  CBPopOp::create(rewriter, loc, contributionWait.getCb(), totalTilesAttr);

  // The new compute writes the original final store target, and the loop body
  // with its local wait/attach/add has been replaced completely.
  rewriter.eraseOp(match.finalStore);
  for (AttachCBOp attach : match.deadReserveAttachOps) {
    rewriter.eraseOp(attach);
  }
  rewriter.eraseOp(loop);
  return success();
}

LogicalResult lowerTensorAccumulationToL1Pack(TensorAccumulationMatch &match,
                                              scf::ForOp loop, int64_t scopeId,
                                              RewriterBase &rewriter) {
  if (match.contribution.getType() != match.tensorType ||
      loop.getNumResults() != 1 || match.resultIndex != 0) {
    return failure();
  }

  CBReserveOp outputReserve = match.reserve;
  if (outputReserve->getBlock() == loop->getBlock() &&
      !outputReserve->isBeforeInBlock(loop)) {
    rewriter.moveOpBefore(outputReserve, loop);
  }

  rewriter.setInsertionPoint(loop);
  StoreOp::create(rewriter, match.finalStore.getLoc(), match.initialValue,
                  outputReserve.getResult(), /*accumulate=*/nullptr);
  auto newLoop =
      scf::ForOp::create(rewriter, loop.getLoc(), loop.getLowerBound(),
                         loop.getUpperBound(), loop.getStep(), ValueRange{});
  for (NamedAttribute attr : loop->getAttrs()) {
    newLoop->setAttr(attr.getName(), attr.getValue());
  }
  newLoop->setAttr(kL1AccLoopAttrName, rewriter.getUnitAttr());
  newLoop->setAttr(
      kL1AccInitialAttrName,
      AccumulationInitialModeAttr::get(
          rewriter.getContext(), AccumulationInitialMode::AccumulateExisting));
  newLoop->setAttr(kL1AccScopeIdAttrName, rewriter.getI64IntegerAttr(scopeId));

  Block *newBody = newLoop.getBody();
  if (!newBody->empty() && isa<scf::YieldOp>(newBody->back())) {
    rewriter.eraseOp(&newBody->back());
  }

  IRMapping mapper;
  mapper.map(loop.getInductionVar(), newLoop.getInductionVar());
  rewriter.setInsertionPointToEnd(newBody);
  bool emittedAccumulatingStore = false;
  for (Operation &bodyOp : loop.getBody()->without_terminator()) {
    if (&bodyOp == match.add.getOperation()) {
      Value contribution = mapper.lookupOrDefault(match.contribution);
      StoreOp::create(rewriter, bodyOp.getLoc(), contribution,
                      outputReserve.getResult(), rewriter.getUnitAttr());
      emittedAccumulatingStore = true;
      continue;
    }
    rewriter.clone(bodyOp, mapper);
  }
  assert(emittedAccumulatingStore && "match must contain a loop-local add");
  scf::YieldOp::create(rewriter, loop.getBody()->getTerminator()->getLoc());

  rewriter.eraseOp(match.finalStore);
  for (AttachCBOp attach : match.deadReserveAttachOps) {
    rewriter.eraseOp(attach);
  }
  rewriter.eraseOp(loop);
  return success();
}

} // namespace mlir::tt::ttl
