// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeGraph.h"

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
                                       int64_t blockCount, Location loc,
                                       Operation *receiverCopyOp) {
  PipeKey key{srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY, pipeNetId};
  if (receiverCBs.count(key) != 0) {
    return emitError(loc) << "duplicate receiver CB for the same pipe";
  }
  receiverCBs.insert({key, {cbIndex, 0, blockCount, loc}});
  receiverCopyToKey[receiverCopyOp] = key;
  receiverCopyOrder.push_back({receiverCopyOp, key});
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

  // (srcX, srcY) order is stable and reproducible across runs.
  SmallVector<PipeKey> orderedPipes;
  orderedPipes.reserve(receiverCBs.size());
  for (auto &[key, info] : receiverCBs) {
    orderedPipes.push_back(key);
  }
  llvm::sort(orderedPipes, [](const PipeKey &a, const PipeKey &b) {
    return std::tie(a.srcX, a.srcY, a.dstStartX, a.dstStartY, a.pipeNetId) <
           std::tie(b.srcX, b.srcY, b.dstStartX, b.dstStartY, b.pipeNetId);
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

  // Count senders per unicast destination.
  for (auto &[key, info] : receiverCBs) {
    bool isUnicast =
        key.dstStartX == key.dstEndX && key.dstStartY == key.dstEndY;
    if (!isUnicast) {
      continue;
    }
    detail::GatherDstKey dk{key.dstStartX, key.dstStartY, key.pipeNetId};
    gatherDstCounts[dk]++;
  }

  // Assign 1-based receive indices per destination. receiver CopyOps
  // targeting the same gather destination get sequential indices based
  // on the program order they were discovered during build().
  // Uses receiverCopyOrder (insertion-ordered) instead of the DenseMap
  // receiverCopyToKey, because the cumulative wait protocol requires
  // the last CopyOp in program order to reset the semaphore.
  llvm::DenseMap<detail::GatherDstKey, int64_t, detail::GatherDstKeyInfo>
      dstCounters;
  for (auto &[copyOp, key] : receiverCopyOrder) {
    detail::GatherDstKey dk{key.dstStartX, key.dstStartY, key.pipeNetId};
    if (gatherDstCounts.count(dk) == 0) {
      continue;
    }
    gatherRecvProgress[copyOp] = ++dstCounters[dk];
  }
}

LogicalResult PipeGraph::verifyGatherBlockCounts() const {
  for (auto &[pk, info] : receiverCBs) {
    int64_t requiredBlocks = info.gatherSlotIdx + 1;
    if (info.blockCount < requiredBlocks) {
      bool isUnicast = pk.dstStartX == pk.dstEndX && pk.dstStartY == pk.dstEndY;
      return emitError(info.loc)
             << (isUnicast ? "gather" : "multicast overlap")
             << " pipe receiver CB has block_count=" << info.blockCount
             << " but slot " << info.gatherSlotIdx
             << " is assigned to this pipe; "
             << "block_count must be >= " << requiredBlocks;
    }
  }
  return success();
}

std::pair<int64_t, int64_t>
PipeGraph::getGatherRecvProgress(Operation *receiverCopyOp) const {
  auto keyIt = receiverCopyToKey.find(receiverCopyOp);
  if (keyIt == receiverCopyToKey.end()) {
    return {1, 1};
  }
  const PipeKey &pk = keyIt->second;
  detail::GatherDstKey dk{pk.dstStartX, pk.dstStartY, pk.pipeNetId};
  auto it = gatherDstCounts.find(dk);
  if (it == gatherDstCounts.end()) {
    return {1, 1};
  }
  auto progIt = gatherRecvProgress.find(receiverCopyOp);
  if (progIt == gatherRecvProgress.end()) {
    return {1, 1};
  }
  return {progIt->second, it->second};
}

FailureOr<PipeGraph> PipeGraph::build(ModuleOp mod) {
  PipeGraph graph;

  // Find all Pipe->CB copies (receiver side) and extract CB index.
  LogicalResult walkResult = success();
  mod.walk([&](CopyOp copyOp) {
    if (failed(walkResult)) {
      return;
    }
    auto srcPipeType = dyn_cast<PipeType>(copyOp.getSrc().getType());
    if (!srcPipeType) {
      return;
    }

    // Found Pipe->CB copy: this is the receiver side. Either failure here
    // would let the sender silently target its own write_ptr instead of the
    // receiver's, so fail the pass loudly rather than warn-and-skip.
    Value dstCB = copyOp.getDst();
    auto cbType = dyn_cast<CircularBufferType>(dstCB.getType());
    if (!cbType) {
      copyOp.emitError("pipe copy destination is not a circular buffer");
      walkResult = failure();
      return;
    }

    Value cbVal = traceUnrealizedCasts(dstCB);
    auto bindOp = cbVal.getDefiningOp<BindCBOp>();
    if (!bindOp) {
      copyOp.emitError("could not trace pipe receiver to a BindCBOp");
      walkResult = failure();
      return;
    }

    int64_t cbIndex = bindOp.getCbIndex().getSExtValue();
    walkResult = graph.addReceiverCB(
        srcPipeType.getSrcX(), srcPipeType.getSrcY(),
        srcPipeType.getDstStartX(), srcPipeType.getDstStartY(),
        srcPipeType.getDstEndX(), srcPipeType.getDstEndY(),
        srcPipeType.getPipeNetId(), cbIndex, cbType.getBlockCount(),
        copyOp.getLoc(), copyOp);
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
