// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeGraph.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::tt::ttl {

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
                        staticTileOffset, 0, receiverSlotSpanBlocks, blockCount,
                        loc, receiverReserveOp, transferContract,
                        SmallVector<Operation *>(transferCreateOps.begin(),
                                                 transferCreateOps.end())}});
  return success();
}

void PipeGraph::assignReceiverSlotIndices() {
  // (receiver, DFB index) -> next reserved slot at that receiver.
  struct ReceiverKey {
    PipeReceiverCoord receiver;
    int64_t dfbIndex;
    bool operator==(const ReceiverKey &other) const {
      return receiver == other.receiver && dfbIndex == other.dfbIndex;
    }
  };
  struct ReceiverKeyInfo {
    static unsigned getHashValue(const ReceiverKey &key) {
      return llvm::hash_combine(key.receiver.x, key.receiver.y, key.dfbIndex);
    }
    static bool isEqual(const ReceiverKey &lhs, const ReceiverKey &rhs) {
      return lhs == rhs;
    }
  };
  struct ReceiverReserveKey {
    ReceiverKey receiverKey;
    Operation *reserveOp;
    bool operator==(const ReceiverReserveKey &other) const {
      return receiverKey == other.receiverKey && reserveOp == other.reserveOp;
    }
  };
  struct ReceiverReserveKeyInfo {
    static unsigned getHashValue(const ReceiverReserveKey &key) {
      return llvm::hash_combine(ReceiverKeyInfo::getHashValue(key.receiverKey),
                                key.reserveOp);
    }
    static bool isEqual(const ReceiverReserveKey &lhs,
                        const ReceiverReserveKey &rhs) {
      return lhs == rhs;
    }
  };
  llvm::DenseMap<ReceiverKey, int64_t, ReceiverKeyInfo> nextSlotAtReceiver;
  llvm::DenseMap<ReceiverReserveKey, int64_t, ReceiverReserveKeyInfo>
      slotByReserve;

  for (auto &entry : receiverDFBs) {
    const PipeKey &pipeKey = entry.first;
    ReceiverDFBInfo &receiverInfo = entry.second;
    const int64_t dfbIndex = receiverInfo.dfbIndex;
    std::optional<int64_t> slotIndex;
    bool hasNonUniformSlotIndex = false;
    pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
      ReceiverKey receiverKey{receiver, dfbIndex};
      ReceiverReserveKey reserveKey{receiverKey,
                                    receiverInfo.receiverReserveOp};
      auto reserveIt = slotByReserve.find(reserveKey);
      int64_t receiverSlot = 0;
      if (reserveIt == slotByReserve.end()) {
        receiverSlot = nextSlotAtReceiver.lookup(receiverKey);
        slotByReserve.insert({reserveKey, receiverSlot});
        nextSlotAtReceiver[receiverKey] =
            receiverSlot + receiverInfo.receiverSlotSpanBlocks;
      } else {
        receiverSlot = reserveIt->second;
      }
      if (!slotIndex.has_value()) {
        slotIndex = receiverSlot;
      } else if (*slotIndex != receiverSlot) {
        hasNonUniformSlotIndex = true;
      }
    });

    if (hasNonUniformSlotIndex || !slotIndex.has_value()) {
      receiverInfo.receiverSlotIndex = std::nullopt;
      continue;
    }

    receiverInfo.receiverSlotIndex = *slotIndex;
    pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
      int64_t &nextSlot = nextSlotAtReceiver[ReceiverKey{receiver, dfbIndex}];
      nextSlot =
          std::max(nextSlot, *slotIndex + receiverInfo.receiverSlotSpanBlocks);
    });
  }
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
      receiverDFBNode.receiverSlotPeriod =
          std::max(receiverDFBNode.receiverSlotPeriod,
                   *receiverInfo.receiverSlotIndex +
                       receiverInfo.receiverSlotSpanBlocks);
    });
  }
}

static PipeKey getPipeKey(PipeType pipeType) {
  return {pipeType.getSrcX(),      pipeType.getSrcY(),
          pipeType.getDstStartX(), pipeType.getDstStartY(),
          pipeType.getDstEndX(),   pipeType.getDstEndY(),
          pipeType.getPipeNetId()};
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

  graph.assignReceiverSlotIndices();

  if (failed(graph.verifyReceiverDFBBlockCounts())) {
    return failure();
  }

  graph.rebuildEndpointGraph();
  return std::move(graph);
}

} // namespace mlir::tt::ttl
