// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cstdint>
#include <limits>

namespace mlir::tt::ttl {

namespace {

/// Return the dataflow buffer wait that provides the loop-local contribution.
/// The streaming DST lowering preserves one wait/pop pair per source
/// iteration, so the contribution must be acquired inside the source loop.
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
/// releases the matched contribution block after the add consumes it.
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
  return foundContributionPop;
}

/// Return the logical DST capacity for a resident accumulator tensor. Scope
/// lowering uses the default double-buffered DST mode because no pass option in
/// PR4 changes the recurrence section to full-sync mode.
static FailureOr<std::uint32_t>
getDefaultDstCapacityForTensor(RankedTensorType tensorType) {
  std::optional<Type> elementType =
      getTileElementType(tensorType.getElementType());
  if (!elementType) {
    return failure();
  }
  return getDstCapacity(elementType->isF32(), /*fullSyncEn=*/false);
}

static void
enumerateTileCoordinates(ArrayRef<int64_t> shape,
                         SmallVectorImpl<int64_t> &current,
                         SmallVectorImpl<SmallVector<int64_t>> &coordinates) {
  if (current.size() == shape.size()) {
    SmallVector<int64_t> coordinate;
    llvm::append_range(coordinate, current);
    coordinates.push_back(std::move(coordinate));
    return;
  }

  int64_t dim = current.size();
  for (int64_t index = 0; index < shape[dim]; ++index) {
    current.push_back(index);
    enumerateTileCoordinates(shape, current, coordinates);
    current.pop_back();
  }
}

static SmallVector<SmallVector<int64_t>>
enumerateTileCoordinates(RankedTensorType tensorType) {
  SmallVector<SmallVector<int64_t>> coordinates;
  SmallVector<int64_t> current;
  enumerateTileCoordinates(tensorType.getShape(), current, coordinates);
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
  if (failed(unitTileCount) || *unitTileCount <= 0) {
    return failure();
  }

  FailureOr<std::uint32_t> dstCapacity =
      getDefaultDstCapacityForTensor(contributionType);
  if (failed(dstCapacity) ||
      *unitTileCount > static_cast<int64_t>(*dstCapacity)) {
    return failure();
  }

  return TensorDstAccumulationInfo{*unitTileCount, contributionWait,
                                   attachedContribution, contributionType};
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
  auto dstSection = DstSectionOp::create(rewriter, loc);
  Block &sectionBody = dstSection.getBody().front();

  OpBuilder::InsertionGuard sectionGuard(rewriter);
  rewriter.setInsertionPoint(sectionBody.getTerminator());

  Type tileType = match.tensorType.getElementType();
  MLIRContext *context = rewriter.getContext();
  AccumulationCombinerAttr addCombiner =
      AccumulationCombinerAttr::get(context, AccumulationCombiner::Add);
  SmallVector<SmallVector<int64_t>> tileCoordinates =
      enumerateTileCoordinates(match.tensorType);
  assert(static_cast<int64_t>(tileCoordinates.size()) == info->unitTileCount &&
         "analysis tile count must match coordinate enumeration");

  SmallVector<Value> accumulatorTiles;
  SmallVector<Value> dstIndices;
  accumulatorTiles.reserve(tileCoordinates.size());
  dstIndices.reserve(tileCoordinates.size());
  for (auto [linearIndex, coordinates] : llvm::enumerate(tileCoordinates)) {
    Value dstIndex = arith::ConstantIndexOp::create(rewriter, loc, linearIndex);
    SmallVector<Value> coordinateValues =
        createIndexConstants(rewriter, loc, coordinates);
    Value initTile = tensor::ExtractOp::create(
        rewriter, loc, match.initialValue, coordinateValues);
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

  {
    OpBuilder::InsertionGuard loopGuard(rewriter);
    rewriter.setInsertionPointToStart(newLoop.getBody());
    CBWaitOp contributionWait = CBWaitOp::create(
        rewriter, loc, info->contributionType, info->contributionWait.getCb(),
        /*num_tiles=*/IntegerAttr{});
    Value contributionTensor = contributionWait.getResult();
    if (info->attachedContribution) {
      contributionTensor =
          AttachCBOp::create(rewriter, loc, info->contributionType,
                             contributionTensor, info->contributionWait.getCb())
              .getResult();
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
    CBPopOp::create(rewriter, loc, info->contributionWait.getCb(),
                    /*num_tiles=*/IntegerAttr{});
  }

  rewriter.setInsertionPoint(sectionBody.getTerminator());
  for (auto [linearIndex, coordinates] : llvm::enumerate(tileCoordinates)) {
    SmallVector<Value> coordinateValues =
        createIndexConstants(rewriter, loc, coordinates);
    Value placeholder = createTilePlaceholder(rewriter, loc, tileType);
    TileStoreOp::create(rewriter, loc, placeholder, outputReserve.getResult(),
                        coordinateValues, dstIndices[linearIndex]);
  }

  rewriter.eraseOp(match.finalStore);
  for (AttachCBOp attach : match.deadReserveAttachOps) {
    rewriter.eraseOp(attach);
  }
  rewriter.eraseOp(loop);
  return success();
}

} // namespace mlir::tt::ttl
