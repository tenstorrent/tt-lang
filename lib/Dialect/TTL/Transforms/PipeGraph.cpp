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
#include "llvm/ADT/SmallSet.h"

namespace mlir::tt::ttl {

LogicalResult PipeGraph::addReceiverDFB(int64_t srcX, int64_t srcY,
                                        int64_t dstStartX, int64_t dstStartY,
                                        int64_t dstEndX, int64_t dstEndY,
                                        int64_t pipeNetId, int64_t dfbIndex,
                                        CircularBufferType dfbType,
                                        int64_t staticTileOffset,
                                        int64_t blockCount, Location loc) {
  PipeKey key{srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY, pipeNetId};
  auto existing = receiverDFBs.find(key);
  bool isMulticast = dstStartX != dstEndX || dstStartY != dstEndY;
  if (existing != receiverDFBs.end()) {
    if (isMulticast &&
        (existing->second.dfbIndex != dfbIndex ||
         existing->second.dfbType != dfbType ||
         existing->second.staticTileOffset != staticTileOffset)) {
      auto diag = emitError(loc)
                  << "multicast pipe receive posts publish non-uniform "
                     "destination addresses; per-destination multicast "
                     "receive addresses are tracked by issue #617";
      diag.attachNote(existing->second.loc)
          << "previous multicast receive post for this pipe was here";
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
    return success();
  }
  receiverDFBs.insert(
      {key, {dfbIndex, dfbType, staticTileOffset, 0, blockCount, loc}});
  return success();
}

void PipeGraph::assignGatherSlotIndices() {
  // (receiver, DFB index) -> slots already taken at that receiver.
  struct ReceiverKey {
    int64_t recvX, recvY, dfbIndex;
    bool operator==(const ReceiverKey &other) const {
      return recvX == other.recvX && recvY == other.recvY &&
             dfbIndex == other.dfbIndex;
    }
  };
  struct ReceiverKeyInfo {
    static ReceiverKey getEmptyKey() {
      int64_t sentinel = llvm::DenseMapInfo<int64_t>::getEmptyKey();
      return {sentinel, sentinel, sentinel};
    }
    static ReceiverKey getTombstoneKey() {
      int64_t sentinel = llvm::DenseMapInfo<int64_t>::getTombstoneKey();
      return {sentinel, sentinel, sentinel};
    }
    static unsigned getHashValue(const ReceiverKey &key) {
      return llvm::hash_combine(key.recvX, key.recvY, key.dfbIndex);
    }
    static bool isEqual(const ReceiverKey &lhs, const ReceiverKey &rhs) {
      return lhs == rhs;
    }
  };
  llvm::DenseMap<ReceiverKey, llvm::SmallSet<int64_t, 4>, ReceiverKeyInfo>
      usedAtReceiver;

  // Order by the complete PipeKey so the greedy coloring is independent of
  // DenseMap iteration order.
  SmallVector<PipeKey> orderedPipes;
  orderedPipes.reserve(receiverDFBs.size());
  for (auto &[key, info] : receiverDFBs) {
    orderedPipes.push_back(key);
  }
  llvm::sort(orderedPipes, [](const PipeKey &lhs, const PipeKey &rhs) {
    return std::tie(lhs.srcX, lhs.srcY, lhs.dstStartX, lhs.dstStartY,
                    lhs.dstEndX, lhs.dstEndY, lhs.pipeNetId) <
           std::tie(rhs.srcX, rhs.srcY, rhs.dstStartX, rhs.dstStartY,
                    rhs.dstEndX, rhs.dstEndY, rhs.pipeNetId);
  });

  for (const PipeKey &pk : orderedPipes) {
    auto it = receiverDFBs.find(pk);
    const int64_t dfbIndex = it->second.dfbIndex;

    // Slots taken by earlier pipes at any of this pipe's receivers
    // (destination range is inclusive on both ends).
    llvm::SmallSet<int64_t, 4> taken;
    for (int64_t dstY = pk.dstStartY; dstY <= pk.dstEndY; ++dstY) {
      for (int64_t dstX = pk.dstStartX; dstX <= pk.dstEndX; ++dstX) {
        auto receiverIt =
            usedAtReceiver.find(ReceiverKey{dstX, dstY, dfbIndex});
        if (receiverIt == usedAtReceiver.end()) {
          continue;
        }
        for (int64_t slotIndex : receiverIt->second) {
          taken.insert(slotIndex);
        }
      }
    }

    // Lowest free slot.
    int64_t slot = 0;
    while (taken.count(slot)) {
      ++slot;
    }
    it->second.gatherSlotIdx = slot;

    // Reserve this slot at every receiver.
    for (int64_t dstY = pk.dstStartY; dstY <= pk.dstEndY; ++dstY) {
      for (int64_t dstX = pk.dstStartX; dstX <= pk.dstEndX; ++dstX) {
        usedAtReceiver[ReceiverKey{dstX, dstY, dfbIndex}].insert(slot);
      }
    }
  }
}

LogicalResult PipeGraph::verifyReceiverDFBBlockCounts() const {
  for (auto &[pk, info] : receiverDFBs) {
    int64_t requiredBlocks = info.gatherSlotIdx + 1;
    if (info.blockCount < requiredBlocks) {
      bool isUnicast = pk.dstStartX == pk.dstEndX && pk.dstStartY == pk.dstEndY;
      return emitError(info.loc)
             << (isUnicast ? "gather" : "multicast overlap")
             << " pipe receiver DFB has block_count=" << info.blockCount
             << " but slot " << info.gatherSlotIdx
             << " is assigned to this pipe; "
             << "block_count must be >= " << requiredBlocks;
    }
  }
  return success();
}

static LogicalResult emitNonUniformMulticastReceiveAddress(Operation *op) {
  return op->emitError()
         << "multicast pipe receive posts publish non-uniform destination "
            "addresses; per-destination multicast receive addresses are "
            "tracked by issue #617";
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
/// destination. Multicast lowering has one sender-visible mailbox address per
/// pipe, so each destination must publish the same static DFB address until
/// issue #617 adds explicit per-destination addresses.
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

static LogicalResult addPipeReceiver(PipeGraph &graph, Operation *op,
                                     PipeType pipeType, Value dst) {
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

  int64_t staticTileOffset = 0;
  if (pipeType.isMulticast()) {
    FailureOr<int64_t> offset = getStaticDestinationTileOffset(dst);
    if (failed(offset)) {
      return emitNonUniformMulticastReceiveAddress(op);
    }
    staticTileOffset = *offset;
  }

  return graph.addReceiverDFB(
      pipeType.getSrcX(), pipeType.getSrcY(), pipeType.getDstStartX(),
      pipeType.getDstStartY(), pipeType.getDstEndX(), pipeType.getDstEndY(),
      pipeType.getPipeNetId(), *dfbIndex, dfbType, staticTileOffset,
      dfbType.getBlockCount(), op->getLoc());
}

FailureOr<PipeGraph> PipeGraph::build(ModuleOp mod) {
  PipeGraph graph;

  LogicalResult walkResult = success();
  mod.walk([&](Operation *op) {
    if (failed(walkResult)) {
      return;
    }
    if (auto postOp = mlir::dyn_cast<PipeRecvPostOp>(op)) {
      auto pipeType = mlir::cast<PipeType>(postOp.getPipe().getType());
      walkResult = addPipeReceiver(graph, op, pipeType, postOp.getDst());
      return;
    }
  });

  if (failed(walkResult)) {
    return failure();
  }

  graph.assignGatherSlotIndices();

  if (failed(graph.verifyReceiverDFBBlockCounts())) {
    return failure();
  }

  return graph;
}

} // namespace mlir::tt::ttl
