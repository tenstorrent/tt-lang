// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

//===----------------------------------------------------------------------===//
// Pipe Graph: Tracks sender->receiver CB associations for pipe copies.
//
// For gather patterns, senders must write to the receiver's CB address, not
// their own. The PipeGraph identifies receiver CBs for each pipe and manages
// gather slot/semaphore assignments.
//===----------------------------------------------------------------------===//

/// Key for identifying a pipe by its source, destination, and PipeNet ID.
struct PipeKey {
  int64_t srcX, srcY;
  int64_t dstStartX, dstStartY, dstEndX, dstEndY;
  int64_t pipeNetId;

  bool operator==(const PipeKey &other) const {
    return srcX == other.srcX && srcY == other.srcY &&
           dstStartX == other.dstStartX && dstStartY == other.dstStartY &&
           dstEndX == other.dstEndX && dstEndY == other.dstEndY &&
           pipeNetId == other.pipeNetId;
  }
};

} // namespace mlir::tt::ttl

namespace llvm {
template <>
struct DenseMapInfo<mlir::tt::ttl::PipeKey> {
  using Key = mlir::tt::ttl::PipeKey;
  static Key getEmptyKey() {
    return {DenseMapInfo<int64_t>::getEmptyKey(), 0, 0, 0, 0, 0, 0};
  }
  static Key getTombstoneKey() {
    return {DenseMapInfo<int64_t>::getTombstoneKey(), 0, 0, 0, 0, 0, 0};
  }
  static unsigned getHashValue(const Key &k) {
    return hash_combine(k.srcX, k.srcY, k.dstStartX, k.dstStartY, k.dstEndX,
                        k.dstEndY, k.pipeNetId);
  }
  static bool isEqual(const Key &a, const Key &b) { return a == b; }
};
} // namespace llvm

namespace mlir::tt::ttl {

/// Receiver CB information for a pipe.
struct ReceiverCBInfo {
  int64_t cbIndex;       // CB index (0-31) used by receiver
  int64_t gatherSlotIdx; // Slot index for gather patterns (0 if not gather)
  int64_t blockCount;    // CB block_count (for gather validation)
  Location loc;          // Source location for error reporting
};

/// Graph tracking pipe connections and receiver CB assignments.
/// Built before lowering by analyzing Pipe->CB copy operations.
class PipeGraph {
public:
  /// Analyze a module to find all pipe receivers and build the graph.
  /// Returns failure if validation detects an error (e.g., gather CB too
  /// small).
  static FailureOr<PipeGraph> build(ModuleOp mod);

  /// Get receiver CB info for a pipe identified by its coordinates.
  /// Returns nullptr if not found.
  const ReceiverCBInfo *getReceiverInfo(int64_t srcX, int64_t srcY,
                                        int64_t dstStartX, int64_t dstStartY,
                                        int64_t dstEndX, int64_t dstEndY,
                                        int64_t pipeNetId) const {
    PipeKey key{srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY, pipeNetId};
    auto it = receiverCBs.find(key);
    if (it == receiverCBs.end()) {
      return nullptr;
    }
    return &it->second;
  }

  /// Check if any pipes were found.
  bool hasPipes() const { return !receiverCBs.empty(); }

  /// Add a receiver CB mapping for a pipe.
  LogicalResult addReceiverCB(int64_t srcX, int64_t srcY, int64_t dstStartX,
                              int64_t dstStartY, int64_t dstEndX,
                              int64_t dstEndY, int64_t pipeNetId,
                              int64_t cbIndex, int64_t blockCount, Location loc,
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

  /// Assign gather slot indices for pipes sharing a destination.
  /// When multiple sources send to the same unicast destination, each source
  /// needs a different slot to avoid overwrites. Slot indices are assigned
  /// sequentially (0-based) per destination group. Groups are keyed by
  /// (destination coordinates, receiver CB index) so that separate PipeNets
  /// sharing a destination get independent slot numbering.
  ///
  /// Also populates gatherDstCounts for receiver-side cumulative semaphore
  /// waits: the count tells the receiver how many total senders target it.
  void assignGatherSlotIndices() {
    struct DstCBKey {
      int64_t dstStartX, dstStartY, dstEndX, dstEndY, cbIndex;
      bool operator==(const DstCBKey &o) const {
        return dstStartX == o.dstStartX && dstStartY == o.dstStartY &&
               dstEndX == o.dstEndX && dstEndY == o.dstEndY &&
               cbIndex == o.cbIndex;
      }
    };
    struct DstCBKeyInfo {
      static DstCBKey getEmptyKey() {
        return {llvm::DenseMapInfo<int64_t>::getEmptyKey(), 0, 0, 0, 0};
      }
      static DstCBKey getTombstoneKey() {
        return {llvm::DenseMapInfo<int64_t>::getTombstoneKey(), 0, 0, 0, 0};
      }
      static unsigned getHashValue(const DstCBKey &k) {
        return llvm::hash_combine(k.dstStartX, k.dstStartY, k.dstEndX,
                                  k.dstEndY, k.cbIndex);
      }
      static bool isEqual(const DstCBKey &a, const DstCBKey &b) {
        return a == b;
      }
    };
    llvm::DenseMap<DstCBKey, SmallVector<PipeKey>, DstCBKeyInfo> groups;
    for (auto &[key, info] : receiverCBs) {
      DstCBKey dk{key.dstStartX, key.dstStartY, key.dstEndX, key.dstEndY,
                  info.cbIndex};
      groups[dk].push_back(key);
    }
    for (auto &[dk, pipeKeys] : groups) {
      if (pipeKeys.size() <= 1) {
        continue;
      }
      llvm::sort(pipeKeys, [](const PipeKey &a, const PipeKey &b) {
        return std::tie(a.srcX, a.srcY) < std::tie(b.srcX, b.srcY);
      });
      for (int64_t i = 0; i < static_cast<int64_t>(pipeKeys.size()); ++i) {
        receiverCBs.find(pipeKeys[i])->second.gatherSlotIdx = i;
      }
    }

    // Count total senders per unicast destination for gather receive protocol.
    // Keyed by (dstX, dstY, pipeNetId) since all unicast pipes to the same
    // destination share a semaphore.
    for (auto &[key, info] : receiverCBs) {
      bool isUnicast =
          key.dstStartX == key.dstEndX && key.dstStartY == key.dstEndY;
      if (!isUnicast) {
        continue;
      }
      GatherDstKey dk{key.dstStartX, key.dstStartY, key.pipeNetId};
      gatherDstCounts[dk]++;
    }

    // Assign 1-based receive indices per destination. receiver CopyOps
    // targeting the same gather destination get sequential indices based
    // on the program order they were discovered during build().
    // Uses receiverCopyOrder (insertion-ordered) instead of the DenseMap
    // receiverCopyToKey, because the cumulative wait protocol requires
    // the last CopyOp in program order to reset the semaphore.
    llvm::DenseMap<GatherDstKey, int64_t, GatherDstKeyInfo> dstCounters;
    for (auto &[copyOp, key] : receiverCopyOrder) {
      GatherDstKey dk{key.dstStartX, key.dstStartY, key.pipeNetId};
      if (gatherDstCounts.count(dk) == 0) {
        continue;
      }
      gatherRecvProgress[copyOp] = ++dstCounters[dk];
    }
  }

  /// Verify that gather receiver CBs have enough blocks for all senders.
  /// Each sender writes to a different slot, so block_count must be >= the
  /// number of senders targeting that CB.
  LogicalResult verifyGatherBlockCounts() const {
    for (auto &[dk, numSenders] : gatherDstCounts) {
      if (numSenders <= 1) {
        continue;
      }
      // Check all receiver entries matching this destination.
      for (auto &[pk, info] : receiverCBs) {
        if (pk.dstStartX != dk.dstX || pk.dstStartY != dk.dstY ||
            pk.pipeNetId != dk.pipeNetId) {
          continue;
        }
        if (info.blockCount < numSenders) {
          return emitError(info.loc)
                 << "gather pipe receiver CB has block_count="
                 << info.blockCount << " but " << numSenders
                 << " senders target it; "
                 << "block_count must be >= number of senders";
        }
      }
    }
    return success();
  }

  /// For unicast gather receivers: returns {recvIndex, totalSenders}.
  /// recvIndex is 1-based (1st sender, 2nd sender, ...).
  /// Non-gather unicast returns {1, 1}.
  /// Keyed on the receiver CopyOp, so call order doesn't matter.
  std::pair<int64_t, int64_t>
  getGatherRecvProgress(Operation *receiverCopyOp) const {
    auto keyIt = receiverCopyToKey.find(receiverCopyOp);
    if (keyIt == receiverCopyToKey.end()) {
      return {1, 1};
    }
    const PipeKey &pk = keyIt->second;
    GatherDstKey dk{pk.dstStartX, pk.dstStartY, pk.pipeNetId};
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

private:
  llvm::DenseMap<PipeKey, ReceiverCBInfo> receiverCBs;

  // Gather receive tracking: count senders per unicast destination.
  struct GatherDstKey {
    int64_t dstX, dstY, pipeNetId;
    bool operator==(const GatherDstKey &o) const {
      return dstX == o.dstX && dstY == o.dstY && pipeNetId == o.pipeNetId;
    }
  };
  struct GatherDstKeyInfo {
    static GatherDstKey getEmptyKey() {
      return {llvm::DenseMapInfo<int64_t>::getEmptyKey(), 0, 0};
    }
    static GatherDstKey getTombstoneKey() {
      return {llvm::DenseMapInfo<int64_t>::getTombstoneKey(), 0, 0};
    }
    static unsigned getHashValue(const GatherDstKey &k) {
      return llvm::hash_combine(k.dstX, k.dstY, k.pipeNetId);
    }
    static bool isEqual(const GatherDstKey &a, const GatherDstKey &b) {
      return a == b;
    }
  };
  llvm::DenseMap<GatherDstKey, int64_t, GatherDstKeyInfo> gatherDstCounts;

  // Maps receiver CopyOp -> PipeKey for CopyOp-keyed lookups.
  llvm::DenseMap<Operation *, PipeKey> receiverCopyToKey;

  // Insertion-ordered record of receiver CopyOps. Used by
  // assignGatherSlotIndices to assign receive indices in program order
  // (DenseMap iteration order is hash-based, not insertion-ordered).
  SmallVector<std::pair<Operation *, PipeKey>> receiverCopyOrder;

  // Maps receiver CopyOp -> 1-based receive index (assigned at build time).
  llvm::DenseMap<Operation *, int64_t> gatherRecvProgress;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
