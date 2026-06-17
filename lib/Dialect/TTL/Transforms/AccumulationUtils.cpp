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

/// Accept frontend-emitted index casts around integer literals when computing
/// static loop bounds before canonicalization.
static std::optional<int64_t> getConstantIntThroughIndexCast(Value value) {
  if (auto constantValue = getConstantIntValue(value)) {
    return *constantValue;
  }
  if (auto indexCast = value.getDefiningOp<arith::IndexCastOp>()) {
    return getConstantIntThroughIndexCast(indexCast.getIn());
  }
  return std::nullopt;
}

/// Require the contribution wait to be a direct loop body operation before it
/// is coalesced into a pre-loop wait.
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

/// Reject loop bodies whose unmatched operations would be lost when replacing
/// the loop with one reduction compute.
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
    if (!allowedOps.contains(&bodyOp)) {
      auto contributionPop = dyn_cast<CBPopOp>(&bodyOp);
      if (!contributionPop ||
          contributionPop.getCb() != contributionWait.getCb() ||
          contributionPop.getNumTiles() || foundContributionPop ||
          contributionPop->isBeforeInBlock(match.add)) {
        return false;
      }
      foundContributionPop = true;
    }
  }
  return true;
}

/// Return the contribution tensor type after the loop iteration space has been
/// represented as a leading reduction dimension.
static RankedTensorType
buildCoalescedContributionType(RankedTensorType unitType, int64_t tripCount) {
  SmallVector<int64_t> shape;
  shape.push_back(tripCount);
  llvm::append_range(shape, unitType.getShape());
  return RankedTensorType::get(shape, unitType.getElementType());
}

/// Return indexing maps where the contribution operand reads the reduction
/// dimension and accumulator/output operands project it out.
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

/// Return iterator types for output-parallel dimensions plus one reduction
/// dimension used for repeated DST accumulation.
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

  auto loopResult = loop.getResult(resultIndex);
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
    // The final non-accumulating store identifies the externally visible
    // destination. Accumulating stores already represent user-written DFB +=
    // and are handled by a separate formation rule.
    if (!store || store.getAccumulate() || finalStore) {
      return failure();
    }
    finalStore = store;
  }
  if (!finalStore) {
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

  // The initial accumulator must already be dataflow-buffer backed because the
  // generated compute reads it as the first input before reusing DST.
  if (!getAttachedCB(match.initialValue)) {
    return failure();
  }

  // Coalescing replaces one wait per iteration with one pre-compute wait, so
  // the total tile count must be known at compile time.
  std::optional<int64_t> tripCount = getStaticAccumulationTripCount(loop);
  if (!tripCount || *tripCount == 0) {
    return failure();
  }
  int64_t tripCountValue = *tripCount;

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
  FailureOr<int64_t> unitTileCount = getStaticTensorTileCount(contributionType);
  if (failed(unitTileCount) || *unitTileCount <= 0 ||
      tripCountValue > std::numeric_limits<int64_t>::max() / *unitTileCount) {
    return failure();
  }
  int64_t totalContributionTiles = tripCountValue * *unitTileCount;
  auto contributionCBType =
      cast<CircularBufferType>(contributionWait.getCb().getType());
  if (totalContributionTiles > contributionCBType.getTotalElements()) {
    return failure();
  }

  return TensorDstAccumulationInfo{tripCountValue,         *unitTileCount,
                                   totalContributionTiles, contributionWait,
                                   attachedContribution,   contributionType};
}

FailureOr<TensorL1PackAccumulationInfo>
analyzeTensorAccumulationForL1Pack(TensorAccumulationMatch &match,
                                   scf::ForOp loop) {
  if (match.contribution.getType() != match.tensorType ||
      loop.getNumResults() != 1 || match.resultIndex != 0) {
    return failure();
  }

  // The generated metadata configures packer L1 accumulation for every pack in
  // the loop. Additional stores would produce packs that are not part of the
  // additive recurrence.
  bool hasLoopLocalStore = false;
  loop->walk([&](StoreOp) {
    hasLoopLocalStore = true;
    return WalkResult::interrupt();
  });
  if (hasLoopLocalStore) {
    return failure();
  }

  std::optional<int64_t> unitTileCount;
  if (auto contributionType =
          dyn_cast<RankedTensorType>(match.contribution.getType())) {
    FailureOr<int64_t> tileCount = getStaticTensorTileCount(contributionType);
    if (succeeded(tileCount)) {
      unitTileCount = *tileCount;
    }
  }

  return TensorL1PackAccumulationInfo{getStaticAccumulationTripCount(loop),
                                      unitTileCount};
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
  // The output reserve must dominate both the generated compute output view and
  // the tile stores that reuse its explicit DST indices.
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
  CBPopOp::create(rewriter, loc, info->contributionWait.getCb(),
                  totalTilesAttr);

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
  if (failed(analyzeTensorAccumulationForL1Pack(match, loop))) {
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
