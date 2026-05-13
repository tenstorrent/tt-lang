// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

//===----------------------------------------------------------------------===//
// Pipe Graph: Tracks sender->receiver CB associations for pipe copies.
//
// For gather patterns, senders must write to the receiver's CB address, not
// their own. The PipeGraph identifies receiver CBs for each pipe and
// manages gather slot/semaphore assignments.
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
    int64_t s = DenseMapInfo<int64_t>::getEmptyKey();
    return {s, s, s, s, s, s, s};
  }
  static Key getTombstoneKey() {
    int64_t s = DenseMapInfo<int64_t>::getTombstoneKey();
    return {s, s, s, s, s, s, s};
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

namespace detail {
// Per-unicast-gather-destination key. Visible in the header because
// PipeGraph::gatherDstCounts uses it as the DenseMap key type.
struct GatherDstKey {
  int64_t dstX, dstY, pipeNetId;
  bool operator==(const GatherDstKey &o) const {
    return dstX == o.dstX && dstY == o.dstY && pipeNetId == o.pipeNetId;
  }
};
struct GatherDstKeyInfo {
  static GatherDstKey getEmptyKey() {
    int64_t s = llvm::DenseMapInfo<int64_t>::getEmptyKey();
    return {s, s, s};
  }
  static GatherDstKey getTombstoneKey() {
    int64_t s = llvm::DenseMapInfo<int64_t>::getTombstoneKey();
    return {s, s, s};
  }
  static unsigned getHashValue(const GatherDstKey &k) {
    return llvm::hash_combine(k.dstX, k.dstY, k.pipeNetId);
  }
  static bool isEqual(const GatherDstKey &a, const GatherDstKey &b) {
    return a == b;
  }
};
} // namespace detail

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
                              Operation *receiverCopyOp);

  /// Assign per-pipe slot indices via greedy coloring keyed by
  /// (receiver, cbIndex). Pipes sharing a receiver+cbIndex get distinct
  /// slots so their writes do not overwrite each other in that receiver's
  /// CB. Pipes ordered by (srcX, srcY) for reproducibility. Also populates
  /// gatherDstCounts for unicast receivers' cumulative wait_min.
  void assignGatherSlotIndices();

  /// Each pipe needs `block_count >= gatherSlotIdx + 1` in its receiver
  /// CB. Covers unicast gather and multicast overlap uniformly.
  LogicalResult verifyGatherBlockCounts() const;

  /// For unicast gather receivers: returns {recvIndex, totalSenders}.
  /// recvIndex is 1-based (1st sender, 2nd sender, ...).
  /// Non-gather unicast returns {1, 1}.
  /// Keyed on the receiver CopyOp, so call order doesn't matter.
  std::pair<int64_t, int64_t>
  getGatherRecvProgress(Operation *receiverCopyOp) const;

private:
  llvm::DenseMap<PipeKey, ReceiverCBInfo> receiverCBs;

  // Per-unicast-destination sender count, keyed by (dstX, dstY, pipeNetId).
  // Multicast pipes use the runtime counter from
  // allocatePipeNetCountersForMulticast / lowerPipeToCB.
  llvm::DenseMap<detail::GatherDstKey, int64_t, detail::GatherDstKeyInfo>
      gatherDstCounts;

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
