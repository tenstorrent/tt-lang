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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

namespace mlir::tt::ttl {

//===----------------------------------------------------------------------===//
// Pipe Graph: Tracks sender->receiver CB associations for pipe copies.
//
// For gather patterns, senders must write to the receiver's CB address, not
// their own. The PipeGraph identifies receiver CBs for each pipe and assigns
// runtime arg slots for passing receiver CB addresses to senders.
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
template <> struct DenseMapInfo<mlir::tt::ttl::PipeKey> {
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
  int64_t runtimeArgIdx; // Index in runtime args for receiver's CB address
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

  /// Get the number of runtime args needed for pipe receiver addresses.
  int64_t getNumPipeRuntimeArgs() const { return numPipeRuntimeArgs; }

  /// Check if any pipes were found.
  bool hasPipes() const { return !receiverCBs.empty(); }

  /// Add a receiver CB mapping for a pipe.
  void addReceiverCB(int64_t srcX, int64_t srcY, int64_t dstStartX,
                     int64_t dstStartY, int64_t dstEndX, int64_t dstEndY,
                     int64_t pipeNetId, int64_t cbIndex, int64_t blockCount,
                     Location loc, Operation *receiverCopyOp) {
    PipeKey key{srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY, pipeNetId};
    assert(receiverCBs.count(key) == 0 &&
           "duplicate receiver CB for the same pipe");
    receiverCBs[key] = {cbIndex, -1, 0, blockCount, loc};
    receiverCopyToKey[receiverCopyOp] = key;
  }

  /// Assign runtime arg indices for all receiver CB addresses.
  void assignRuntimeArgIndices() {
    int64_t nextArgIdx = 0;
    for (auto &[key, info] : receiverCBs) {
      info.runtimeArgIdx = nextArgIdx++;
    }
    numPipeRuntimeArgs = nextArgIdx;
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
    struct DstCBKeyHash {
      std::size_t operator()(const DstCBKey &k) const {
        return llvm::hash_combine(k.dstStartX, k.dstStartY, k.dstEndX,
                                  k.dstEndY, k.cbIndex);
      }
    };
    std::unordered_map<DstCBKey, SmallVector<PipeKey>, DstCBKeyHash> groups;
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
        receiverCBs[pipeKeys[i]].gatherSlotIdx = i;
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
    std::unordered_map<GatherDstKey, int64_t, GatherDstKeyHash> dstCounters;
    for (auto &[copyOp, key] : receiverCopyToKey) {
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
      // Find a receiver entry matching this destination to get block_count.
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
        break;
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

  /// Emit pipe graph as JSON for Python to read and populate runtime args.
  /// Controlled by TTLANG_PIPE_GRAPH_JSON environment variable.
  void emitJSON() const {
    const char *path = std::getenv("TTLANG_PIPE_GRAPH_JSON");
    if (!path || receiverCBs.empty()) {
      return;
    }

    llvm::json::Object root;
    llvm::json::Array pipesArray;

    for (const auto &[key, info] : receiverCBs) {
      llvm::json::Object pipeObj;
      pipeObj["srcX"] = key.srcX;
      pipeObj["srcY"] = key.srcY;
      pipeObj["dstStartX"] = key.dstStartX;
      pipeObj["dstStartY"] = key.dstStartY;
      pipeObj["dstEndX"] = key.dstEndX;
      pipeObj["dstEndY"] = key.dstEndY;
      pipeObj["pipeNetId"] = key.pipeNetId;
      pipeObj["receiverCBIndex"] = info.cbIndex;
      pipeObj["runtimeArgSlot"] = info.runtimeArgIdx;
      pipesArray.push_back(std::move(pipeObj));
    }

    root["pipes"] = std::move(pipesArray);
    root["numPipeRuntimeArgs"] = numPipeRuntimeArgs;

    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      llvm::errs() << "Error writing pipe graph JSON to " << path << ": "
                   << ec.message() << "\n";
      return;
    }

    os << llvm::json::Value(std::move(root));
  }

private:
  llvm::DenseMap<PipeKey, ReceiverCBInfo> receiverCBs;
  int64_t numPipeRuntimeArgs = 0;

  // Gather receive tracking: count senders per unicast destination.
  struct GatherDstKey {
    int64_t dstX, dstY, pipeNetId;
    bool operator==(const GatherDstKey &o) const {
      return dstX == o.dstX && dstY == o.dstY && pipeNetId == o.pipeNetId;
    }
  };
  struct GatherDstKeyHash {
    std::size_t operator()(const GatherDstKey &k) const {
      return llvm::hash_combine(k.dstX, k.dstY, k.pipeNetId);
    }
  };
  std::unordered_map<GatherDstKey, int64_t, GatherDstKeyHash> gatherDstCounts;

  // Maps receiver CopyOp -> PipeKey for CopyOp-keyed lookups.
  llvm::DenseMap<Operation *, PipeKey> receiverCopyToKey;

  // Maps receiver CopyOp -> 1-based receive index (assigned at build time).
  llvm::DenseMap<Operation *, int64_t> gatherRecvProgress;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
