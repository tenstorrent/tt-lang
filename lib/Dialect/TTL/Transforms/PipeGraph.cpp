// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeGraph.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::tt::ttl {

static PipeKey getPipeKey(PipeType pipeType) {
  return {pipeType.getSrcX(),      pipeType.getSrcY(),
          pipeType.getDstStartX(), pipeType.getDstStartY(),
          pipeType.getDstEndX(),   pipeType.getDstEndY(),
          pipeType.getPipeNetId()};
}

namespace {

struct PipeGraphAnalysisState : LaunchNodeDomainState {
  llvm::DenseMap<Operation *, LaunchNodeDomain> operationLaunchDomains;
};

static LogicalResult collectLaunchNodeDomains(ModuleOp mod,
                                              PipeGraphAnalysisState &state) {
  state.initialize(mod);
  if (!state.hasLaunchGrid) {
    return success();
  }

  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  LaunchNodeDomainAnalysisOptions options;
  options.narrowPipeNetScopes = true;
  options.operationCallback = [&](Operation *op, const LaunchNodeDomain &domain,
                                  Operation * /*unanalyzableOp*/) {
    state.operationLaunchDomains[op] = domain;
  };
  solver.load<LaunchNodeDomainAnalysis>(state, options);
  return solver.initializeAndRun(mod);
}

static LaunchNodeDomain
getOperationLaunchDomain(Operation *op, PipeGraphAnalysisState &state) {
  if (!state.hasLaunchGrid) {
    return LaunchNodeDomain::unknown();
  }
  auto it = state.operationLaunchDomains.find(op);
  if (it == state.operationLaunchDomains.end()) {
    return LaunchNodeDomain::unknown();
  }
  return it->second;
}

static bool domainContainsReceiver(const LaunchNodeDomain &domain,
                                   PipeReceiverCoord receiver) {
  if (!domain.known) {
    return true;
  }
  return domain.nodes.find(LaunchNodeCoord{receiver.x, receiver.y}) !=
         domain.nodes.end();
}

static std::optional<int64_t> getReleasedBlockCount(CBPopOp popOp) {
  auto cbType = mlir::dyn_cast<CircularBufferType>(popOp.getCb().getType());
  if (!cbType) {
    return std::nullopt;
  }
  int64_t elementsPerBlock = cbType.getElementsPerBlock();
  int64_t releasedTiles = elementsPerBlock;
  if (auto attr = popOp.getNumTilesAttr()) {
    releasedTiles = attr.getInt();
  }
  if (releasedTiles <= 0 || releasedTiles % elementsPerBlock != 0) {
    return std::nullopt;
  }
  return releasedTiles / elementsPerBlock;
}

template <typename Fn>
static WalkResult walkNestedOpsInOrder(Operation *op, Fn &&callback) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (callback(&nestedOp).wasInterrupted()) {
          return WalkResult::interrupt();
        }
        if (walkNestedOpsInOrder(&nestedOp, callback).wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
    }
  }
  return WalkResult::advance();
}

struct LiveReceiverSlot {
  int64_t slot = 0;
  int64_t span = 0;
};

struct ReceiverSlotState {
  int64_t nextSlot = 0;
  SmallVector<LiveReceiverSlot> liveSlots;
};

static bool slotRangesOverlap(int64_t lhsSlot, int64_t lhsSpan, int64_t rhsSlot,
                              int64_t rhsSpan) {
  return lhsSlot < rhsSlot + rhsSpan && rhsSlot < lhsSlot + lhsSpan;
}

static void releaseReceiverSlots(ReceiverSlotState &state,
                                 int64_t releasedBlocks) {
  while (releasedBlocks > 0 && !state.liveSlots.empty()) {
    LiveReceiverSlot &slot = state.liveSlots.front();
    int64_t releasedFromSlot = std::min(releasedBlocks, slot.span);
    slot.slot += releasedFromSlot;
    slot.span -= releasedFromSlot;
    releasedBlocks -= releasedFromSlot;
    if (slot.span == 0) {
      state.liveSlots.erase(state.liveSlots.begin());
    }
  }
}

} // namespace

LogicalResult PipeGraph::addReceiverDFB(
    int64_t srcX, int64_t srcY, int64_t dstStartX, int64_t dstStartY,
    int64_t dstEndX, int64_t dstEndY, int64_t pipeNetId, int64_t dfbIndex,
    CircularBufferType dfbType, bool hasStaticTileOffset,
    int64_t staticTileOffset, int64_t receiverSlotSpanBlocks,
    Operation *receiverReserveOp, PipeTransferContract transferContract,
    ArrayRef<Operation *> transferCreateOps, int64_t blockCount, Location loc) {
  PipeKey key{srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY, pipeNetId};
  auto existing = receiverDFBs.find(key);
  bool hasMultipleReceivers = dstStartX != dstEndX || dstStartY != dstEndY;
  if (existing != receiverDFBs.end()) {
    if (hasMultipleReceivers &&
        (existing->second.dfbIndex != dfbIndex ||
         existing->second.dfbType != dfbType ||
         !existing->second.hasStaticTileOffset ||
         existing->second.staticTileOffset != staticTileOffset)) {
      auto diag = emitError(loc)
                  << "collective pipe receive posts publish different "
                     "destination addresses; TT-Metal NoC multicast requires "
                     "one destination SRAM address for all receivers";
      diag.attachNote(existing->second.loc)
          << "previous collective receive post for this pipe was here";
      return failure();
    }

    if (existing->second.dfbIndex != dfbIndex ||
        existing->second.dfbType != dfbType ||
        existing->second.blockCount != blockCount) {
      auto diag = emitError(loc)
                  << "conflicting receiver DFBs for the same pipe";
      diag.attachNote(existing->second.loc)
          << "previous receiver DFB for this pipe was here";
      return failure();
    }
    existing->second.hasStaticTileOffset =
        existing->second.hasStaticTileOffset && hasStaticTileOffset &&
        existing->second.staticTileOffset == staticTileOffset;
    return success();
  }
  receiverDFBs.insert({key,
                       {dfbIndex, dfbType, hasStaticTileOffset,
                        staticTileOffset, std::nullopt, receiverSlotSpanBlocks,
                        blockCount, loc, receiverReserveOp, transferContract,
                        SmallVector<Operation *>(transferCreateOps.begin(),
                                                 transferCreateOps.end())}});
  return success();
}

static FailureOr<int64_t>
assignReceiverPhysicalSlot(const PipeKey &pipeKey,
                           ReceiverDFBInfo &receiverInfo,
                           ReceiverSlotState &slotState) {
  int64_t span = receiverInfo.receiverSlotSpanBlocks;
  if (span <= 0 || span > receiverInfo.blockCount) {
    return emitError(receiverInfo.loc)
           << (pipeKey.hasSingleReceiver() ? "gather" : "collective overlap")
           << " pipe receiver DFB reserves " << span
           << " block(s) but block_count=" << receiverInfo.blockCount;
  }

  int64_t slot = slotState.nextSlot;
  if (slot + span > receiverInfo.blockCount) {
    return emitError(receiverInfo.loc)
           << (pipeKey.hasSingleReceiver() ? "gather" : "collective overlap")
           << " pipe receiver DFB reserve at slot " << slot << " spans " << span
           << " block(s), which would wrap block_count="
           << receiverInfo.blockCount;
  }

  for (const LiveReceiverSlot &liveSlot : slotState.liveSlots) {
    if (slotRangesOverlap(slot, span, liveSlot.slot, liveSlot.span)) {
      return emitError(receiverInfo.loc)
             << (pipeKey.hasSingleReceiver() ? "gather" : "collective overlap")
             << " pipe receiver DFB reuses slot " << slot
             << " before a receiver pop releases it; add a receiver pop before "
                "reusing the DFB slot or increase block_count";
    }
  }

  slotState.liveSlots.push_back(LiveReceiverSlot{slot, span});
  slotState.nextSlot = (slot + span) % receiverInfo.blockCount;
  return slot;
}

LogicalResult PipeGraph::assignReceiverSlotIndices(ModuleOp mod) {
  PipeGraphAnalysisState analysisState;
  if (failed(collectLaunchNodeDomains(mod, analysisState))) {
    return failure();
  }

  llvm::DenseMap<PipeReceiverDFBKey, ReceiverSlotState> slotStateByReceiverDFB;
  llvm::DenseMap<PipeReceiverDFBKey, llvm::DenseMap<Operation *, int64_t>>
      slotByReceiverReserve;
  auto processPost = [&](PipeTransferPostOp postOp) -> LogicalResult {
    auto createOp = findPipeTransferCreateForTransfer(postOp.getTransfer());
    if (!createOp) {
      return success();
    }
    PipeKey pipeKey =
        getPipeKey(mlir::cast<PipeType>(createOp.getPipe().getType()));
    auto receiverIt = receiverDFBs.find(pipeKey);
    if (receiverIt == receiverDFBs.end()) {
      return success();
    }

    ReceiverDFBInfo &receiverInfo = receiverIt->second;
    if (receiverInfo.receiverSlotIndex.has_value()) {
      return success();
    }

    LaunchNodeDomain postDomain =
        getOperationLaunchDomain(postOp.getOperation(), analysisState);
    std::optional<int64_t> uniformSlot;
    bool hasReceiver = false;
    bool nonUniformSlot = false;
    LogicalResult result = success();
    pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
      if (failed(result) || !domainContainsReceiver(postDomain, receiver)) {
        return;
      }
      hasReceiver = true;
      PipeReceiverDFBKey receiverDFB{receiver, receiverInfo.dfbIndex};
      auto &slotByReserve = slotByReceiverReserve[receiverDFB];
      auto reserveIt = slotByReserve.find(receiverInfo.receiverReserveOp);
      int64_t slot = 0;
      if (reserveIt == slotByReserve.end()) {
        FailureOr<int64_t> assignedSlot = assignReceiverPhysicalSlot(
            pipeKey, receiverInfo, slotStateByReceiverDFB[receiverDFB]);
        if (failed(assignedSlot)) {
          result = failure();
          return;
        }
        slot = *assignedSlot;
        slotByReserve[receiverInfo.receiverReserveOp] = slot;
      } else {
        slot = reserveIt->second;
      }
      if (!uniformSlot) {
        uniformSlot = slot;
      } else if (*uniformSlot != slot) {
        nonUniformSlot = true;
      }
    });
    if (failed(result)) {
      return failure();
    }
    if (!hasReceiver || nonUniformSlot) {
      receiverInfo.receiverSlotIndex = std::nullopt;
      return success();
    }
    receiverInfo.receiverSlotIndex = *uniformSlot;
    return success();
  };

  auto processPop = [&](CBPopOp popOp) {
    std::optional<int64_t> dfbIndex = getCBIndex(popOp.getCb());
    if (!dfbIndex) {
      return;
    }
    std::optional<int64_t> releasedBlocks = getReleasedBlockCount(popOp);
    if (!releasedBlocks) {
      return;
    }
    LaunchNodeDomain popDomain =
        getOperationLaunchDomain(popOp.getOperation(), analysisState);
    if (!popDomain.known) {
      return;
    }
    for (LaunchNodeCoord coord : popDomain.nodes) {
      PipeReceiverDFBKey receiverDFB{PipeReceiverCoord{coord.x, coord.y},
                                     *dfbIndex};
      auto stateIt = slotStateByReceiverDFB.find(receiverDFB);
      if (stateIt == slotStateByReceiverDFB.end()) {
        continue;
      }
      releaseReceiverSlots(stateIt->second, *releasedBlocks);
    }
  };

  WalkResult walkResult =
      walkNestedOpsInOrder(mod.getOperation(), [&](Operation *op) {
        if (auto postOp = dyn_cast<PipeTransferPostOp>(op)) {
          return failed(processPost(postOp)) ? WalkResult::interrupt()
                                             : WalkResult::advance();
        }
        if (auto popOp = dyn_cast<CBPopOp>(op)) {
          processPop(popOp);
        }
        return WalkResult::advance();
      });
  return walkResult.wasInterrupted() ? failure() : success();
}

LogicalResult PipeGraph::verifyReceiverDFBBlockCounts() const {
  for (auto &[pk, info] : receiverDFBs) {
    if (!info.receiverSlotIndex.has_value()) {
      return emitError(info.loc)
             << "collective pipe receiver posts reserve different DFB slots; "
                "TT-Metal NoC multicast requires one destination SRAM address "
                "for all receivers";
    }
    int64_t requiredBlocks =
        *info.receiverSlotIndex + info.receiverSlotSpanBlocks;
    if (info.blockCount < requiredBlocks) {
      return emitError(info.loc)
             << (pk.hasSingleReceiver() ? "gather" : "collective overlap")
             << " pipe receiver DFB has block_count=" << info.blockCount
             << " but slot " << *info.receiverSlotIndex
             << " is assigned to this pipe; "
             << "block_count must be >= " << requiredBlocks;
    }
  }
  return success();
}

const ReceiverDFBInfo *PipeGraph::lookupReceiverDFB(const PipeKey &key) const {
  auto it = receiverDFBs.find(key);
  if (it == receiverDFBs.end()) {
    return nullptr;
  }
  return &it->second;
}

void PipeGraph::rebuildEndpointGraph() {
  pipeEdges.clear();
  pipeReceiverEndpoints.clear();
  receiverDFBNodes.clear();

  llvm::DenseMap<PipeReceiverDFBKey, PipeReceiverDFBNodeId> nodeIdByReceiverDFB;
  for (const auto &entry : receiverDFBs) {
    const PipeKey &pipeKey = entry.first;
    const ReceiverDFBInfo &receiverInfo = entry.second;
    PipeEdgeId pipeEdgeId = pipeEdges.size();
    pipeEdges.push_back(PipeEdge{pipeEdgeId,
                                 pipeKey,
                                 receiverInfo.transferContract,
                                 receiverInfo,
                                 receiverInfo.transferCreateOps,
                                 {}});
    PipeEdge &pipeEdge = pipeEdges.back();

    pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
      PipeReceiverDFBKey receiverDFB{receiver, receiverInfo.dfbIndex};
      auto nodeIt = nodeIdByReceiverDFB.find(receiverDFB);
      PipeReceiverDFBNodeId receiverDFBNodeId = 0;
      if (nodeIt == nodeIdByReceiverDFB.end()) {
        receiverDFBNodeId = receiverDFBNodes.size();
        nodeIdByReceiverDFB.insert({receiverDFB, receiverDFBNodeId});
        receiverDFBNodes.push_back(
            PipeReceiverDFBNode{receiverDFBNodeId, receiverDFB, {}, 1});
      } else {
        receiverDFBNodeId = nodeIt->second;
      }

      PipeReceiverEndpointId endpointId = pipeReceiverEndpoints.size();
      pipeReceiverEndpoints.push_back(PipeReceiverEndpoint{
          endpointId, pipeEdgeId, receiverDFBNodeId, receiver, receiverDFB,
          *receiverInfo.receiverSlotIndex});
      pipeEdge.receiverEndpoints.push_back(endpointId);
      PipeReceiverDFBNode &receiverDFBNode =
          receiverDFBNodes[receiverDFBNodeId];
      receiverDFBNode.writerEndpoints.push_back(endpointId);
      receiverDFBNode.receiverBatchSize =
          std::max(receiverDFBNode.receiverBatchSize,
                   *receiverInfo.receiverSlotIndex +
                       receiverInfo.receiverSlotSpanBlocks);
    });
  }
}

struct PipeTransferInfo {
  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;
  SmallVector<Operation *> transferCreateOps;
};

static llvm::MapVector<PipeKey, PipeTransferInfo>
collectPipeTransferInfo(ModuleOp mod) {
  llvm::MapVector<PipeKey, PipeTransferInfo> pipeInfo;
  auto addTransfer = [&](PipeTransferCreateOp op) {
    if (!op) {
      return;
    }
    auto pipeType = mlir::cast<PipeType>(op.getPipe().getType());
    PipeTransferContract contract = getPipeTransferContract(op);
    PipeKey key = getPipeKey(pipeType);
    auto existing = pipeInfo.find(key);
    if (existing == pipeInfo.end()) {
      pipeInfo.insert({key, PipeTransferInfo{contract, {op.getOperation()}}});
      return;
    }
    if (!llvm::is_contained(existing->second.transferCreateOps,
                            op.getOperation())) {
      existing->second.transferCreateOps.push_back(op.getOperation());
    }
    // Duplicate transfers for the same PipeKey can arise from cloned regions.
    // Collective is the stronger contract and must be preserved.
    if (isCollectiveTransfer(contract)) {
      existing->second.transferContract = PipeTransferContract::Collective;
    }
  };
  mod.walk([&](Operation *op) {
    if (auto postOp = dyn_cast<PipeTransferPostOp>(op)) {
      addTransfer(findPipeTransferCreateForTransfer(postOp.getTransfer()));
      return;
    }
    if (auto sendOp = dyn_cast<PipeTransferSendOp>(op)) {
      addTransfer(findPipeTransferCreateForTransfer(sendOp.getTransfer()));
    }
  });
  return pipeInfo;
}

static LogicalResult
emitUntraceableCollectiveDestinationAddress(Operation *op) {
  return op->emitError()
         << "collective pipe destination address could not be "
            "determined statically; TT-Metal NoC multicast requires one "
            "statically proven destination SRAM address for all receivers";
}

static LogicalResult addStaticCoordinates(ArrayRef<OpFoldResult> mixedOffsets,
                                          SmallVectorImpl<int64_t> &coordinates,
                                          unsigned rank) {
  if (coordinates.empty()) {
    coordinates.assign(rank, 0);
  }
  if (coordinates.size() != rank || mixedOffsets.size() != rank) {
    return failure();
  }

  for (auto [coordinate, mixedOffset] :
       llvm::zip_equal(coordinates, mixedOffsets)) {
    std::optional<int64_t> offset = getConstantIntValue(mixedOffset);
    if (!offset.has_value()) {
      return failure();
    }
    coordinate += *offset;
  }
  return success();
}

/// Return the static tile offset within the receiver DFB for a receive
/// destination. Collective lowering has one sender-visible address-table entry
/// per pipe because NoC multicast writes one destination SRAM address to every
/// receiver.
static FailureOr<int64_t> getStaticDestinationTileOffset(Value dst) {
  Value view = traceUnrealizedCasts(dst);
  SmallVector<int64_t> coordinates;
  RankedTensorType rootType;
  bool sawOffset = false;

  while (true) {
    view = traceUnrealizedCasts(view);
    if (auto extract = view.getDefiningOp<tensor::ExtractOp>()) {
      auto tensorType =
          mlir::dyn_cast<RankedTensorType>(extract.getTensor().getType());
      if (!tensorType) {
        return failure();
      }
      SmallVector<OpFoldResult> mixedIndices;
      for (Value index : extract.getIndices()) {
        mixedIndices.push_back(index);
      }
      if (failed(addStaticCoordinates(mixedIndices, coordinates,
                                      tensorType.getRank()))) {
        return failure();
      }
      sawOffset = true;
      view = extract.getTensor();
      continue;
    }
    if (auto attach = view.getDefiningOp<AttachCBOp>()) {
      view = attach.getTensor();
      continue;
    }

    auto slice = view.getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice) {
      rootType = mlir::dyn_cast<RankedTensorType>(view.getType());
      break;
    }

    auto sourceType =
        mlir::dyn_cast<RankedTensorType>(slice.getSource().getType());
    if (!sourceType) {
      return failure();
    }

    if (failed(addStaticCoordinates(slice.getMixedOffsets(), coordinates,
                                    sourceType.getRank()))) {
      return failure();
    }
    sawOffset = true;
    view = slice.getSource();
  }

  if (!sawOffset) {
    return 0;
  }
  if (!rootType ||
      rootType.getRank() != static_cast<int64_t>(coordinates.size())) {
    return failure();
  }

  int64_t linearOffset = 0;
  for (auto [coordinate, dim] :
       llvm::zip_equal(coordinates, rootType.getShape())) {
    if (dim == ShapedType::kDynamic) {
      return failure();
    }
    linearOffset = linearOffset * dim + coordinate;
  }
  return linearOffset;
}

static FailureOr<int64_t> getTensorTileCount(RankedTensorType tensorType) {
  int64_t tileCount = 1;
  for (int64_t dimension : tensorType.getShape()) {
    if (dimension == ShapedType::kDynamic) {
      return failure();
    }
    tileCount *= dimension;
  }
  return tileCount;
}

static FailureOr<int64_t>
getReceiverSlotSpanBlocks(Value dst, CircularBufferType dfbType) {
  auto reserveOp = findCBReserveForPipeReceive(dst);
  if (!reserveOp) {
    return failure();
  }
  auto reserveType =
      mlir::dyn_cast<RankedTensorType>(reserveOp.getResult().getType());
  if (!reserveType) {
    return failure();
  }

  auto dfbBlockType =
      RankedTensorType::get(dfbType.getShape(), dfbType.getElementType());
  FailureOr<int64_t> reserveTileCount = getTensorTileCount(reserveType);
  FailureOr<int64_t> dfbBlockTileCount = getTensorTileCount(dfbBlockType);
  if (failed(reserveTileCount) || failed(dfbBlockTileCount) ||
      *dfbBlockTileCount <= 0) {
    return failure();
  }
  return (*reserveTileCount + *dfbBlockTileCount - 1) / *dfbBlockTileCount;
}

static LogicalResult addPipeReceiver(PipeGraph &graph, Operation *op,
                                     PipeType pipeType,
                                     PipeTransferContract transferContract,
                                     ArrayRef<Operation *> transferCreateOps,
                                     Value dst) {
  Value dstDFB = getAttachedCB(dst);
  if (!dstDFB) {
    return op->emitError("pipe receive destination is not attached to a DFB");
  }
  auto dfbType = mlir::dyn_cast<CircularBufferType>(dstDFB.getType());
  if (!dfbType) {
    return op->emitError("pipe receive destination is not attached to a DFB");
  }

  std::optional<int64_t> dfbIndex = getCBIndex(dstDFB);
  if (!dfbIndex.has_value()) {
    return op->emitError("could not trace pipe receiver to a DFB binding");
  }

  bool hasStaticTileOffset = true;
  int64_t staticTileOffset = 0;
  FailureOr<int64_t> offset = getStaticDestinationTileOffset(dst);
  if (failed(offset)) {
    if (isCollectiveTransfer(transferContract)) {
      return emitUntraceableCollectiveDestinationAddress(op);
    }
    hasStaticTileOffset = false;
  } else {
    staticTileOffset = *offset;
  }

  FailureOr<int64_t> slotSpanBlocks = getReceiverSlotSpanBlocks(dst, dfbType);
  if (failed(slotSpanBlocks)) {
    return op->emitError("could not determine receiver DFB reserve span");
  }
  auto reserveOp = findCBReserveForPipeReceive(dst);
  assert(reserveOp && "reserve span computation already traced reserve op");

  return graph.addReceiverDFB(
      pipeType.getSrcX(), pipeType.getSrcY(), pipeType.getDstStartX(),
      pipeType.getDstStartY(), pipeType.getDstEndX(), pipeType.getDstEndY(),
      pipeType.getPipeNetId(), *dfbIndex, dfbType, hasStaticTileOffset,
      staticTileOffset, *slotSpanBlocks, reserveOp.getOperation(),
      transferContract, transferCreateOps, dfbType.getBlockCount(),
      op->getLoc());
}

FailureOr<PipeGraph> PipeGraph::build(ModuleOp mod) {
  PipeGraph graph;
  llvm::MapVector<PipeKey, PipeTransferInfo> transferInfos =
      collectPipeTransferInfo(mod);

  WalkResult walkResult = mod.walk([&](PipeTransferPostOp postOp) {
    auto createOp = findPipeTransferCreateForTransfer(postOp.getTransfer());
    if (!createOp) {
      postOp.emitError(
          "pipe transfer post must reference a transfer derived from "
          "ttl.pipe_transfer.create");
      return WalkResult::interrupt();
    }
    auto pipeType = mlir::cast<PipeType>(createOp.getPipe().getType());
    PipeKey key = getPipeKey(pipeType);
    auto pipeInfoIt = transferInfos.find(key);
    if (pipeInfoIt == transferInfos.end()) {
      postOp.emitError(
          "pipe transfer post must reference a known pipe transfer");
      return WalkResult::interrupt();
    }
    const PipeTransferInfo &pipeInfo = pipeInfoIt->second;
    if (failed(addPipeReceiver(graph, postOp, pipeType,
                               pipeInfo.transferContract,
                               pipeInfo.transferCreateOps, postOp.getDst()))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return failure();
  }

  if (failed(graph.assignReceiverSlotIndices(mod))) {
    return failure();
  }

  if (failed(graph.verifyReceiverDFBBlockCounts())) {
    return failure();
  }

  graph.rebuildEndpointGraph();
  return std::move(graph);
}

} // namespace mlir::tt::ttl
