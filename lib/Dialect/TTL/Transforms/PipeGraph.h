// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <optional>

namespace mlir::tt::ttl {

//===----------------------------------------------------------------------===//
// Pipe Graph: Tracks receiver dataflow buffer associations for pipe copies.
// The graph validates that each logical pipe has a consistent destination DFB
// and enough DFB slots for overlapping writes.
//===----------------------------------------------------------------------===//

/// Physical receiver identity for a pipe destination. The current pipe type
/// stores local core coordinates; keeping that representation behind this type
/// confines future mesh/device coordinate changes to the graph interface.
struct PipeReceiverCoord {
  int64_t x = 0;
  int64_t y = 0;

  bool operator==(const PipeReceiverCoord &other) const {
    return x == other.x && y == other.y;
  }
};

/// Receiver-local DFB identity. The receiver coordinate is kept abstract from
/// the current rectangular pipe encoding so future mesh/device coordinates can
/// be localized behind PipeGraph.
struct PipeReceiverDFBKey {
  PipeReceiverCoord receiver;
  int64_t dfbIndex = 0;

  bool operator==(const PipeReceiverDFBKey &other) const {
    return receiver == other.receiver && dfbIndex == other.dfbIndex;
  }
};

/// Key for identifying a pipe by its source, destination, and PipeNet ID.
struct PipeKey {
  int64_t srcX = 0, srcY = 0;
  int64_t dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0;
  int64_t pipeNetId = 0;

  bool operator==(const PipeKey &other) const {
    return srcX == other.srcX && srcY == other.srcY &&
           dstStartX == other.dstStartX && dstStartY == other.dstStartY &&
           dstEndX == other.dstEndX && dstEndY == other.dstEndY &&
           pipeNetId == other.pipeNetId;
  }

  bool hasSingleReceiver() const {
    return dstStartX == dstEndX && dstStartY == dstEndY;
  }

  PipeReceiverCoord getSingleReceiver() const {
    assert(hasSingleReceiver() && "expected a single receiver");
    return PipeReceiverCoord{dstStartX, dstStartY};
  }

  template <typename Fn>
  void forEachReceiver(Fn &&callback) const {
    for (int64_t receiverY = dstStartY; receiverY <= dstEndY; ++receiverY) {
      for (int64_t receiverX = dstStartX; receiverX <= dstEndX; ++receiverX) {
        callback(PipeReceiverCoord{receiverX, receiverY});
      }
    }
  }
};

} // namespace mlir::tt::ttl

namespace llvm {
template <>
struct DenseMapInfo<mlir::tt::ttl::PipeReceiverCoord> {
  using Key = mlir::tt::ttl::PipeReceiverCoord;
  static unsigned getHashValue(const Key &receiver) {
    return hash_combine(receiver.x, receiver.y);
  }
  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
};

template <>
struct DenseMapInfo<mlir::tt::ttl::PipeReceiverDFBKey> {
  using Key = mlir::tt::ttl::PipeReceiverDFBKey;
  static unsigned getHashValue(const Key &receiverDFB) {
    return hash_combine(receiverDFB.receiver.x, receiverDFB.receiver.y,
                        receiverDFB.dfbIndex);
  }
  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
};

template <>
struct DenseMapInfo<mlir::tt::ttl::PipeKey> {
  using Key = mlir::tt::ttl::PipeKey;
  static unsigned getHashValue(const Key &k) {
    return hash_combine(k.srcX, k.srcY, k.dstStartX, k.dstStartY, k.dstEndX,
                        k.dstEndY, k.pipeNetId);
  }
  static bool isEqual(const Key &a, const Key &b) { return a == b; }
};
} // namespace llvm

namespace mlir::tt::ttl {

enum class PipeTransferContract {
  PointToPoint,
  Collective,
};

inline bool isCollectiveTransfer(PipeTransferContract contract) {
  return contract == PipeTransferContract::Collective;
}

/// Receiver DFB information for a pipe.
struct ReceiverDFBInfo {
  int64_t dfbIndex;           // DFB index (0-31) used by receiver
  CircularBufferType dfbType; // Receiver DFB type
  bool hasStaticTileOffset;   // Whether staticTileOffset is known.
  int64_t staticTileOffset;   // Static destination tile offset within the DFB
  std::optional<int64_t> receiverSlotIndex;
  int64_t receiverSlotSpanBlocks;
  int64_t blockCount; // DFB block_count
  Location loc;       // Source location for error reporting
  Operation *receiverReserveOp;
  PipeTransferContract transferContract;
  SmallVector<Operation *> transferCreateOps;
};

using PipeEdgeId = unsigned;
using PipeReceiverEndpointId = unsigned;
using PipeReceiverDFBNodeId = unsigned;

/// One logical PipeNet edge from a source core to a receiver set.
struct PipeEdge {
  PipeEdgeId id = 0;
  PipeKey pipe;
  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;
  ReceiverDFBInfo receiverDFBInfo;
  SmallVector<Operation *> transferCreateOps;
  SmallVector<PipeReceiverEndpointId> receiverEndpoints;
};

/// One receiver endpoint written by a logical PipeNet edge.
struct PipeReceiverEndpoint {
  PipeReceiverEndpointId id = 0;
  PipeEdgeId pipeEdge = 0;
  PipeReceiverDFBNodeId receiverDFBNode = 0;
  PipeReceiverCoord receiver;
  PipeReceiverDFBKey receiverDFB;
  int64_t receiverSlotIndex = 0;
};

/// One receiver-local dataflow buffer node in the PipeNet graph.
struct PipeReceiverDFBNode {
  PipeReceiverDFBNodeId id = 0;
  PipeReceiverDFBKey receiverDFB;
  SmallVector<PipeReceiverEndpointId> writerEndpoints;
  int64_t receiverBatchSize = 1;
  bool hasProvenPipeOnlyStream = false;
};

/// Return the semantic transfer contract used by pipe synchronization. The
/// frontend `isCollective` attr is authoritative when present because a
/// degenerate one-receiver collective still requires collective pipe
/// synchronization. This does not choose the physical NOC write instruction.
inline PipeTransferContract getPipeTransferContract(CreatePipeOp op) {
  if (auto attr = op.getIsCollectiveAttr()) {
    return attr.getValue() ? PipeTransferContract::Collective
                           : PipeTransferContract::PointToPoint;
  }
  return mlir::cast<PipeType>(op.getResult().getType()).hasMultipleReceivers()
             ? PipeTransferContract::Collective
             : PipeTransferContract::PointToPoint;
}

inline PipeTransferContract getPipeTransferContract(PipeTransferCreateOp op) {
  return op.getKind().getValue() == PipeTransferKind::Collective
             ? PipeTransferContract::Collective
             : PipeTransferContract::PointToPoint;
}

/// Graph tracking pipe connections and receiver DFB assignments.
/// Built after pipe receive copies have been expanded to pipe transfer ops.
class PipeGraph {
public:
  /// Analyze a module to find all pipe receivers and build the graph.
  /// Returns failure if validation detects an error (e.g., gather DFB too
  /// small).
  static FailureOr<PipeGraph> build(ModuleOp mod);

  /// Check if any pipes were found.
  bool hasPipes() const { return !receiverDFBs.empty(); }

  /// Add a receiver DFB mapping for a pipe.
  LogicalResult addReceiverDFB(
      int64_t srcX, int64_t srcY, int64_t dstStartX, int64_t dstStartY,
      int64_t dstEndX, int64_t dstEndY, int64_t pipeNetId, int64_t dfbIndex,
      CircularBufferType dfbType, bool hasStaticTileOffset,
      int64_t staticTileOffset, int64_t receiverSlotSpanBlocks,
      Operation *receiverReserveOp, PipeTransferContract transferContract,
      ArrayRef<Operation *> transferCreateOps, int64_t blockCount,
      Location loc);

  /// Verify that each receiver post was assigned a non-wrapping physical DFB
  /// slot.
  LogicalResult verifyReceiverDFBBlockCounts() const;

  const ReceiverDFBInfo *lookupReceiverDFB(const PipeKey &key) const;

  ArrayRef<PipeEdge> getPipeEdges() const { return pipeEdges; }

  const PipeEdge &getPipeEdge(PipeEdgeId id) const {
    assert(id < pipeEdges.size() && "invalid pipe edge id");
    return pipeEdges[id];
  }

  ArrayRef<PipeReceiverEndpointId>
  getPipeReceiverEndpoints(PipeEdgeId pipeEdge) const {
    return getPipeEdge(pipeEdge).receiverEndpoints;
  }

  ArrayRef<PipeReceiverEndpoint> getPipeReceiverEndpoints() const {
    return pipeReceiverEndpoints;
  }

  const PipeReceiverEndpoint &
  getPipeReceiverEndpoint(PipeReceiverEndpointId id) const {
    assert(id < pipeReceiverEndpoints.size() &&
           "invalid pipe receiver endpoint id");
    return pipeReceiverEndpoints[id];
  }

  ArrayRef<PipeReceiverDFBNode> getReceiverDFBNodes() const {
    return receiverDFBNodes;
  }

  const PipeReceiverDFBNode &
  getReceiverDFBNode(PipeReceiverDFBNodeId id) const {
    assert(id < receiverDFBNodes.size() && "invalid receiver DFB node id");
    return receiverDFBNodes[id];
  }

  ArrayRef<PipeReceiverEndpointId>
  getReceiverDFBWriterEndpoints(PipeReceiverDFBNodeId receiverDFBNode) const {
    return getReceiverDFBNode(receiverDFBNode).writerEndpoints;
  }

  const llvm::MapVector<PipeKey, ReceiverDFBInfo> &getReceiverDFBs() const {
    return receiverDFBs;
  }

private:
  /// Assign each pipe the physical DFB slot reserved by the corresponding
  /// receiver post. Multicast pipes require the same receiver slot at every
  /// receiver because TT-Metal NoC multicast carries one destination address.
  LogicalResult assignReceiverSlotIndices(ModuleOp mod);

  void rebuildEndpointGraph();

  LogicalResult provePipeOnlyReceiverStreams(ModuleOp mod);

  llvm::MapVector<PipeKey, ReceiverDFBInfo> receiverDFBs;
  SmallVector<PipeEdge, 0> pipeEdges;
  SmallVector<PipeReceiverEndpoint> pipeReceiverEndpoints;
  SmallVector<PipeReceiverDFBNode> receiverDFBNodes;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
