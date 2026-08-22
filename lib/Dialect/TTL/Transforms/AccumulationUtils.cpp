// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::tt::ttl {

namespace {

/// Return the wait that produces the matched contribution tensor. When the
/// recurrence uses an attached tensor view, report the attach separately so DST
/// lowering can recreate the same view after moving the wait into the new loop.
static CBWaitOp getContributionWait(const TensorAccumulationMatch &match,
                                    AttachCBOp &attachedContribution) {
  Value contribution = match.contribution;
  if (auto attach = contribution.getDefiningOp<AttachCBOp>()) {
    attachedContribution = attach;
    contribution = attach.getTensor();
  }

  return contribution.getDefiningOp<CBWaitOp>();
}

/// Check that deleting the source loop will not drop work other than the
/// additive recurrence itself. The only permitted side effect is the pop that
/// releases the matched contribution block after the add consumes it.
static bool onlyContainsDstReductionOps(scf::ForOp loop,
                                        const TensorAccumulationMatch &match,
                                        CBWaitOp contributionWait,
                                        AttachCBOp attachedContribution) {
  DenseSet<Operation *> allowedOps;
  AddOp add = match.add;
  allowedOps.insert(add.getOperation());
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
        contributionPop->isBeforeInBlock(add)) {
      return false;
    }
    foundContributionPop = true;
  }
  return foundContributionPop;
}

/// Resident contributions have no per-iteration dataflow buffer protocol
/// operations. The loop body can be removed only when the additive recurrence
/// is the whole body aside from the terminator.
static bool
onlyContainsResidentDstReductionOps(scf::ForOp loop,
                                    const TensorAccumulationMatch &match) {
  AddOp add = match.add;
  for (Operation &bodyOp : loop.getBody()->without_terminator()) {
    if (&bodyOp != add.getOperation()) {
      return false;
    }
  }
  return true;
}

/// Accept resident contribution setup only when it precedes the loop in the
/// same linear sequence, either directly or through the scope op that contains
/// the loop during lowering.
static bool dominatesLoopInLinearSequence(Operation *operation,
                                          scf::ForOp loop) {
  Block *block = operation->getBlock();
  Operation *projectedLoop = block == loop->getBlock()
                                 ? loop.getOperation()
                                 : block->findAncestorOpInBlock(*loop);
  return projectedLoop && operation->isBeforeInBlock(projectedLoop);
}

struct ResidentContributionReleaseInfo {
  CBPopOp existingPop;
  Operation *lastOwnedUse;
};

/// Classify the resident wait's release with the same ownership computation
/// used by auto-sync so lowering and sync insertion agree on explicit pops.
static FailureOr<ResidentContributionReleaseInfo>
analyzeResidentContributionRelease(CBWaitOp contributionWait,
                                   const DFBAcquireReleaseIndex &dfbIndex) {
  SmallVector<Operation *> consumerAcquisitions =
      dfbIndex.getAcquisitions(DFBAcquireReleaseKind::Consumer);
  SmallVector<Operation *> consumerReleases =
      dfbIndex.getReleases(DFBAcquireReleaseKind::Consumer);
  DFBAcquireInterval interval = makeDFBAcquireInterval(
      contributionWait.getOperation(), consumerAcquisitions);
  Operation *lastOwnedUse = findLastDFBAcquireOwnedUse(interval);
  DFBReleaseSearch releaseSearch =
      findOwnedDFBReleases(interval, lastOwnedUse, consumerReleases);

  if (!releaseSearch.nestedReleases.empty()) {
    return failure();
  }

  CBPopOp existingPop;
  for (Operation *release : releaseSearch.sameLevelReleases) {
    auto pop = cast<CBPopOp>(release);
    if (pop.getNumTiles() || pop->isBeforeInBlock(lastOwnedUse) ||
        existingPop) {
      return failure();
    }
    existingPop = pop;
  }

  return ResidentContributionReleaseInfo{existingPop, lastOwnedUse};
}

/// Return the logical DST capacity for a resident accumulator tensor. Scope
/// recurrence lowering uses the default double-buffered mode to match the
/// surrounding DST allocation model.
static FailureOr<std::uint32_t>
getDefaultDstCapacityForTensor(RankedTensorType tensorType) {
  std::optional<Type> elementType =
      getTileElementType(tensorType.getElementType());
  if (!elementType) {
    return failure();
  }
  return getDstCapacity(elementType->isF32(), /*fullSyncEn=*/false);
}

static FailureOr<int64_t>
getStaticTensorTileCount(RankedTensorType tensorType) {
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

static SmallVector<SmallVector<int64_t>>
enumerateTileCoordinates(RankedTensorType tensorType) {
  FailureOr<int64_t> tileCount = getStaticTensorTileCount(tensorType);
  assert(succeeded(tileCount) && *tileCount > 0 &&
         "coordinate enumeration requires a nonempty static tensor");

  SmallVector<SmallVector<int64_t>> coordinates;
  coordinates.reserve(*tileCount);
  SmallVector<int64_t> strides = computeStrides(tensorType.getShape());
  for (int64_t linearIndex = 0; linearIndex < *tileCount; ++linearIndex) {
    coordinates.push_back(delinearize(linearIndex, strides));
  }
  return coordinates;
}

static SmallVector<Value> createIndexConstants(RewriterBase &rewriter,
                                               Location loc,
                                               ArrayRef<int64_t> coordinates) {
  SmallVector<Value> values;
  values.reserve(coordinates.size());
  for (int64_t coordinate : coordinates) {
    values.push_back(arith::ConstantIndexOp::create(rewriter, loc, coordinate));
  }
  return values;
}

static Value createTilePlaceholder(RewriterBase &rewriter, Location loc,
                                   Type tileType) {
  return UnrealizedConversionCastOp::create(rewriter, loc, tileType,
                                            ValueRange{})
      .getResult(0);
}

} // namespace

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
  // rewrite can replace the loop-carried tensor with DST state.
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
      loop,    resultIndex, tensorType,   initialValue,        finalStore,
      reserve, add,         contribution, deadReserveAttachOps};
}

FailureOr<TensorDstAccumulationInfo>
analyzeTensorAccumulationForDst(const TensorAccumulationMatch &match,
                                const DFBAcquireReleaseIndex &dfbIndex) {
  return analyzeTensorAccumulationForDst(match, match.initialValue, dfbIndex);
}

FailureOr<TensorDstAccumulationInfo>
analyzeTensorAccumulationForDst(const TensorAccumulationMatch &match,
                                Value initialValue,
                                const DFBAcquireReleaseIndex &dfbIndex) {
  scf::ForOp loop = match.loop;
  if (initialValue.getType() != match.tensorType) {
    return failure();
  }

  if (match.contribution.getType() != match.tensorType) {
    return failure();
  }

  if (!getAttachedCB(initialValue)) {
    return failure();
  }

  AttachCBOp attachedContribution;
  CBWaitOp contributionWait = getContributionWait(match, attachedContribution);
  if (!contributionWait || contributionWait.getNumTiles().has_value()) {
    return failure();
  }

  TensorAccumulationContributionResidency contributionResidency;
  CBPopOp residentContributionPop;
  Operation *residentContributionLastUse = nullptr;
  if (contributionWait->getParentOp() == loop) {
    if (attachedContribution && attachedContribution->getParentOp() != loop) {
      return failure();
    }
    if (!onlyContainsDstReductionOps(loop, match, contributionWait,
                                     attachedContribution)) {
      return failure();
    }
    contributionResidency = TensorAccumulationContributionResidency::Streamed;
  } else {
    if (!dominatesLoopInLinearSequence(contributionWait.getOperation(), loop) ||
        (attachedContribution &&
         !dominatesLoopInLinearSequence(attachedContribution.getOperation(),
                                        loop)) ||
        !onlyContainsResidentDstReductionOps(loop, match)) {
      return failure();
    }

    FailureOr<ResidentContributionReleaseInfo> releaseInfo =
        analyzeResidentContributionRelease(contributionWait, dfbIndex);
    if (failed(releaseInfo)) {
      return failure();
    }
    residentContributionPop = releaseInfo->existingPop;
    residentContributionLastUse = releaseInfo->lastOwnedUse;
    contributionResidency = TensorAccumulationContributionResidency::Resident;
  }

  auto contributionType =
      dyn_cast<RankedTensorType>(contributionWait.getResult().getType());
  if (!contributionType || contributionType != match.tensorType ||
      !contributionType.hasStaticShape()) {
    return failure();
  }

  if (match.contribution.getType() != contributionType) {
    return failure();
  }

  FailureOr<int64_t> unitTileCount = getStaticTensorTileCount(contributionType);
  if (failed(unitTileCount) || *unitTileCount <= 0) {
    return failure();
  }

  FailureOr<std::uint32_t> dstCapacity =
      getDefaultDstCapacityForTensor(contributionType);
  if (failed(dstCapacity) ||
      *unitTileCount > static_cast<int64_t>(*dstCapacity)) {
    return failure();
  }

  return TensorDstAccumulationInfo{
      *unitTileCount,          initialValue,
      contributionResidency,   contributionWait,
      attachedContribution,    contributionType,
      residentContributionPop, residentContributionLastUse,
  };
}

void lowerTensorAccumulationToDst(const TensorAccumulationMatch &match,
                                  const TensorDstAccumulationInfo &info,
                                  bool synthesizeResidentContributionPop,
                                  RewriterBase &rewriter) {
  scf::ForOp loop = match.loop;
  Location loc = loop.getLoc();
  CBReserveOp outputReserve = match.reserve;
  if (outputReserve->getBlock() == loop->getBlock() &&
      !outputReserve->isBeforeInBlock(loop)) {
    rewriter.moveOpBefore(outputReserve, loop);
  }

  OpBuilder::InsertionGuard sectionGuard(rewriter);
  rewriter.setInsertionPoint(loop);
  auto dstSection = DstSectionOp::create(rewriter, loc);
  Block &sectionBody = dstSection.getBody().front();

  rewriter.setInsertionPoint(sectionBody.getTerminator());

  Type tileType = match.tensorType.getElementType();
  MLIRContext *context = rewriter.getContext();
  AccumulationCombinerAttr addCombiner =
      AccumulationCombinerAttr::get(context, AccumulationCombiner::Add);
  SmallVector<SmallVector<int64_t>> tileCoordinates =
      enumerateTileCoordinates(match.tensorType);
  assert(static_cast<int64_t>(tileCoordinates.size()) == info.unitTileCount &&
         "analysis tile count must match coordinate enumeration");

  CBWaitOp contributionWait = info.contributionWait;
  AttachCBOp attachedContribution = info.attachedContribution;

  SmallVector<Value> accumulatorTiles;
  SmallVector<Value> dstIndices;
  accumulatorTiles.reserve(tileCoordinates.size());
  dstIndices.reserve(tileCoordinates.size());
  for (auto [linearIndex, coordinates] : llvm::enumerate(tileCoordinates)) {
    Value dstIndex = arith::ConstantIndexOp::create(rewriter, loc, linearIndex);
    SmallVector<Value> coordinateValues =
        createIndexConstants(rewriter, loc, coordinates);
    Value initTile = tensor::ExtractOp::create(rewriter, loc, info.initialValue,
                                               coordinateValues);
    auto copy = CopyTileOp::create(
        rewriter, loc, TypeRange{DSTRegisterType::get(context), tileType},
        initTile, coordinateValues, dstIndex);
    accumulatorTiles.push_back(copy.getDstTile());
    dstIndices.push_back(dstIndex);
  }

  auto newLoop = scf::ForOp::create(rewriter, loc, loop.getLowerBound(),
                                    loop.getUpperBound(), loop.getStep());
  for (NamedAttribute attr : loop->getAttrs()) {
    newLoop->setAttr(attr.getName(), attr.getValue());
  }

  Value residentContributionTensor;
  if (info.contributionResidency ==
      TensorAccumulationContributionResidency::Resident) {
    residentContributionTensor = attachedContribution
                                     ? attachedContribution.getResult()
                                     : contributionWait.getResult();
  }

  {
    OpBuilder::InsertionGuard loopGuard(rewriter);
    rewriter.setInsertionPointToStart(newLoop.getBody());
    Value contributionTensor = residentContributionTensor;
    if (info.contributionResidency ==
        TensorAccumulationContributionResidency::Streamed) {
      CBWaitOp newContributionWait = CBWaitOp::create(
          rewriter, loc, info.contributionType, contributionWait.getCb(),
          /*num_tiles=*/IntegerAttr{});
      contributionTensor = newContributionWait.getResult();
      if (attachedContribution) {
        contributionTensor =
            AttachCBOp::create(rewriter, loc, info.contributionType,
                               contributionTensor, contributionWait.getCb())
                .getResult();
      }
    }

    for (auto [linearIndex, coordinates] : llvm::enumerate(tileCoordinates)) {
      SmallVector<Value> coordinateValues =
          createIndexConstants(rewriter, loc, coordinates);
      Value contributionTile = tensor::ExtractOp::create(
          rewriter, loc, contributionTensor, coordinateValues);
      TileAccumulateOp::create(rewriter, loc, tileType,
                               accumulatorTiles[linearIndex], contributionTile,
                               addCombiner, dstIndices[linearIndex]);
    }
    if (info.contributionResidency ==
        TensorAccumulationContributionResidency::Streamed) {
      CBPopOp::create(rewriter, loc, contributionWait.getCb(),
                      /*num_tiles=*/IntegerAttr{});
    }
  }

  rewriter.setInsertionPoint(sectionBody.getTerminator());
  for (auto [linearIndex, coordinates] : llvm::enumerate(tileCoordinates)) {
    SmallVector<Value> coordinateValues =
        createIndexConstants(rewriter, loc, coordinates);
    Value placeholder = createTilePlaceholder(rewriter, loc, tileType);
    TileStoreOp::create(rewriter, loc, placeholder, outputReserve.getResult(),
                        coordinateValues, dstIndices[linearIndex]);
  }

  if (info.contributionResidency ==
          TensorAccumulationContributionResidency::Resident &&
      !info.residentContributionPop && synthesizeResidentContributionPop) {
    Operation *lastUse = info.residentContributionLastUse;
    assert(lastUse && "resident contribution must have an owned use");
    if (lastUse == loop.getOperation()) {
      lastUse = dstSection.getOperation();
    }
    rewriter.setInsertionPointAfter(lastUse);
    CBPopOp::create(rewriter, loc, contributionWait.getCb(),
                    /*num_tiles=*/IntegerAttr{});
  }

  rewriter.eraseOp(match.finalStore);
  for (AttachCBOp attach : match.deadReserveAttachOps) {
    rewriter.eraseOp(attach);
  }
  rewriter.eraseOp(loop);
}

} // namespace mlir::tt::ttl
