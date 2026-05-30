// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

//===----------------------------------------------------------------------===//
// Pipe Graph: Tracks receiver dataflow buffer associations for pipe copies.
// The graph validates that each logical pipe has a consistent destination DFB
// and enough DFB slots for overlapping writes.
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

/// Receiver DFB information for a pipe.
struct ReceiverDFBInfo {
  int64_t dfbIndex;           // DFB index (0-31) used by receiver
  CircularBufferType dfbType; // Receiver DFB type
  int64_t staticTileOffset;   // Static destination tile offset within the DFB
  int64_t gatherSlotIdx;      // Slot index for overlap patterns (0 if none)
  int64_t blockCount;         // DFB block_count
  Location loc;               // Source location for error reporting
};

/// Graph tracking pipe connections and receiver DFB assignments.
/// Built after pipe receive copies have been expanded to receive-post ops.
class PipeGraph {
public:
  /// Analyze a module to find all pipe receivers and build the graph.
  /// Returns failure if validation detects an error (e.g., gather DFB too
  /// small).
  static FailureOr<PipeGraph> build(ModuleOp mod);

  /// Check if any pipes were found.
  bool hasPipes() const { return !receiverDFBs.empty(); }

  /// Add a receiver DFB mapping for a pipe.
  LogicalResult addReceiverDFB(int64_t srcX, int64_t srcY, int64_t dstStartX,
                               int64_t dstStartY, int64_t dstEndX,
                               int64_t dstEndY, int64_t pipeNetId,
                               int64_t dfbIndex, CircularBufferType dfbType,
                               int64_t staticTileOffset, int64_t blockCount,
                               Location loc);

  /// Assign per-pipe slot indices via greedy coloring keyed by
  /// (receiver, DFB index). Pipes sharing a receiver DFB get distinct
  /// slots so their writes do not overwrite each other in that receiver's
  /// DFB. Pipes ordered by (srcX, srcY) for reproducibility.
  void assignGatherSlotIndices();

  /// Each pipe needs `block_count >= gatherSlotIdx + 1` in its receiver
  /// DFB. Covers unicast gather and multicast overlap uniformly.
  LogicalResult verifyReceiverDFBBlockCounts() const;

private:
  llvm::DenseMap<PipeKey, ReceiverDFBInfo> receiverDFBs;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
