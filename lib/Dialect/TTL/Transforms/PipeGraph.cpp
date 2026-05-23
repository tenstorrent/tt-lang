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

LogicalResult PipeGraph::addReceiverCB(int64_t srcX, int64_t srcY,
                                       int64_t dstStartX, int64_t dstStartY,
                                       int64_t dstEndX, int64_t dstEndY,
                                       int64_t pipeNetId, int64_t cbIndex,
                                       CircularBufferType cbType,
                                       int64_t staticTileOffset,
                                       int64_t blockCount, Location loc) {
  PipeKey key{srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY, pipeNetId};
  auto existing = receiverCBs.find(key);
  bool isMulticast = dstStartX != dstEndX || dstStartY != dstEndY;
  if (existing != receiverCBs.end()) {
    if (isMulticast &&
        (existing->second.cbIndex != cbIndex ||
         existing->second.cbType != cbType ||
         existing->second.staticTileOffset != staticTileOffset)) {
      auto diag = emitError(loc)
                  << "multicast pipe receive posts publish non-uniform "
                     "destination addresses; per-destination multicast "
                     "receive addresses are tracked by issue #617";
      diag.attachNote(existing->second.loc)
          << "previous multicast receive post for this pipe was here";
      return failure();
    }

    if (existing->second.cbIndex != cbIndex ||
        existing->second.cbType != cbType ||
        existing->second.blockCount != blockCount) {
      auto diag = emitError(loc)
                  << "conflicting receiver DFBs for the same pipe";
      diag.attachNote(existing->second.loc)
          << "previous receiver DFB for this pipe was here";
      return failure();
    }
    return success();
  }
  receiverCBs.insert(
      {key, {cbIndex, cbType, staticTileOffset, 0, blockCount, loc}});
  return success();
}

void PipeGraph::assignGatherSlotIndices() {
  // (receiver, cbIndex) -> slots already taken at that receiver.
  struct ReceiverKey {
    int64_t recvX, recvY, cbIndex;
    bool operator==(const ReceiverKey &o) const {
      return recvX == o.recvX && recvY == o.recvY && cbIndex == o.cbIndex;
    }
  };
  struct ReceiverKeyInfo {
    static ReceiverKey getEmptyKey() {
      int64_t s = llvm::DenseMapInfo<int64_t>::getEmptyKey();
      return {s, s, s};
    }
    static ReceiverKey getTombstoneKey() {
      int64_t s = llvm::DenseMapInfo<int64_t>::getTombstoneKey();
      return {s, s, s};
    }
    static unsigned getHashValue(const ReceiverKey &k) {
      return llvm::hash_combine(k.recvX, k.recvY, k.cbIndex);
    }
    static bool isEqual(const ReceiverKey &a, const ReceiverKey &b) {
      return a == b;
    }
  };
  llvm::DenseMap<ReceiverKey, llvm::SmallSet<int64_t, 4>, ReceiverKeyInfo>
      usedAtReceiver;

  // Order by the complete PipeKey so the greedy coloring is independent of
  // DenseMap iteration order.
  SmallVector<PipeKey> orderedPipes;
  orderedPipes.reserve(receiverCBs.size());
  for (auto &[key, info] : receiverCBs) {
    orderedPipes.push_back(key);
  }
  llvm::sort(orderedPipes, [](const PipeKey &a, const PipeKey &b) {
    return std::tie(a.srcX, a.srcY, a.dstStartX, a.dstStartY, a.dstEndX,
                    a.dstEndY, a.pipeNetId) <
           std::tie(b.srcX, b.srcY, b.dstStartX, b.dstStartY, b.dstEndX,
                    b.dstEndY, b.pipeNetId);
  });

  for (const PipeKey &pk : orderedPipes) {
    auto it = receiverCBs.find(pk);
    const int64_t cbIndex = it->second.cbIndex;

    // Slots taken by earlier pipes at any of this pipe's receivers
    // (destination range is inclusive on both ends).
    llvm::SmallSet<int64_t, 4> taken;
    for (int64_t y = pk.dstStartY; y <= pk.dstEndY; ++y) {
      for (int64_t x = pk.dstStartX; x <= pk.dstEndX; ++x) {
        auto rIt = usedAtReceiver.find(ReceiverKey{x, y, cbIndex});
        if (rIt == usedAtReceiver.end()) {
          continue;
        }
        for (int64_t s : rIt->second) {
          taken.insert(s);
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
    for (int64_t y = pk.dstStartY; y <= pk.dstEndY; ++y) {
      for (int64_t x = pk.dstStartX; x <= pk.dstEndX; ++x) {
        usedAtReceiver[ReceiverKey{x, y, cbIndex}].insert(slot);
      }
    }
  }
}

LogicalResult PipeGraph::verifyGatherBlockCounts() const {
  for (auto &[pk, info] : receiverCBs) {
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
  Value dstCB = getAttachedCB(dst);
  if (!dstCB) {
    return op->emitError("pipe receive destination is not attached to a DFB");
  }
  auto cbType = mlir::dyn_cast<CircularBufferType>(dstCB.getType());
  if (!cbType) {
    return op->emitError("pipe receive destination is not attached to a DFB");
  }

  std::optional<int64_t> cbIndex = getCBIndex(dstCB);
  if (!cbIndex.has_value()) {
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

  return graph.addReceiverCB(
      pipeType.getSrcX(), pipeType.getSrcY(), pipeType.getDstStartX(),
      pipeType.getDstStartY(), pipeType.getDstEndX(), pipeType.getDstEndY(),
      pipeType.getPipeNetId(), *cbIndex, cbType, staticTileOffset,
      cbType.getBlockCount(), op->getLoc());
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

  if (failed(graph.verifyGatherBlockCounts())) {
    return failure();
  }

  return graph;
}

} // namespace mlir::tt::ttl
