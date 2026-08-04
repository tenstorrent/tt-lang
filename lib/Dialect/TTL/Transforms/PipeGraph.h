// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H

#include "DFBAcquireReleaseAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

class PipeTransferIndex;

struct PipeGraphAnalysisState;

//===----------------------------------------------------------------------===//
// Pipe Graph: Tracks static transfers, receiver endpoints, physical receiver
// DFBs, and the address sequence selected by each endpoint.
//===----------------------------------------------------------------------===//

/// Receiver node coordinate within one device. Cross-device analyses must also
/// qualify the physical DFB identity with the logical receiver device.
struct PipeReceiverCoord {
  int64_t x = 0;
  int64_t y = 0;

  bool operator==(const PipeReceiverCoord &other) const {
    return x == other.x && y == other.y;
  }
};

/// Physical receiver DFB identity for the current single-device module.
struct PipeReceiverDFBKey {
  PipeReceiverCoord receiver;
  int64_t dfbIndex = 0;

  bool operator==(const PipeReceiverDFBKey &other) const {
    return receiver == other.receiver && dfbIndex == other.dfbIndex;
  }
};

inline void printReceiverDFB(llvm::raw_ostream &os,
                             const PipeReceiverDFBKey &receiverDFB) {
  os << "receiver(" << receiverDFB.receiver.x << ", " << receiverDFB.receiver.y
     << ") DFB " << receiverDFB.dfbIndex;
}

/// Logical source-to-receiver relation. Individual transfers and logical device
/// edges, when present, have separate identities.
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

  template <typename Fn>
  void forEachReceiver(Fn &&callback) const {
    for (int64_t receiverY = dstStartY; receiverY <= dstEndY; ++receiverY) {
      for (int64_t receiverX = dstStartX; receiverX <= dstEndX; ++receiverX) {
        callback(PipeReceiverCoord{receiverX, receiverY});
      }
    }
  }
};

inline PipeKey getPipeKey(PipeType pipeType) {
  return {pipeType.getSrcX(),      pipeType.getSrcY(),
          pipeType.getDstStartX(), pipeType.getDstStartY(),
          pipeType.getDstEndX(),   pipeType.getDstEndY(),
          pipeType.getPipeNetId()};
}

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
  static unsigned getHashValue(const Key &pipeKey) {
    return hash_combine(pipeKey.srcX, pipeKey.srcY, pipeKey.dstStartX,
                        pipeKey.dstStartY, pipeKey.dstEndX, pipeKey.dstEndY,
                        pipeKey.pipeNetId);
  }
  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
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

/// Receiver DFB geometry for one transfer definition.
struct ReceiverDFBInfo {
  int64_t dfbIndex;
  CircularBufferType dfbType;
  bool hasStaticTileOffset;
  int64_t staticTileOffset;
  int64_t receiverSlotSpanBlocks;
  int64_t blockCount;
  Location loc;
};

using PipeTransferNodeId = std::size_t;
using PipeReceiverEndpointId = std::size_t;
using PipeReceiverDFBNodeId = std::size_t;

/// Physical DFB slot for occurrence `i`:
/// `slot(i) = (initialSlot + i * repeatStride) % blockCount`.
struct ReceiverAddressRecurrence {
  int64_t initialSlot = 0;
  int64_t repeatStride = 0;
  int64_t blockCount = 1;
};

/// Compile-time model for a receiver address sequence.
enum class ReceiverAddressSequenceProofKind {
  KnownCount,
  PeriodicUnknownCount,
  FullyDynamic
};

/// Proven receiver slots for one transfer endpoint.
///
/// `KnownCount` has a recurrence and an exact execution count.
/// `PeriodicUnknownCount` has a recurrence that holds for every `i >= 0`.
/// `FullyDynamic` has neither because no recurrence was proven.
struct ReceiverAddressSequenceProof {
  std::optional<std::uint64_t> executionCount;
  std::optional<ReceiverAddressRecurrence> recurrence;

  ReceiverAddressSequenceProofKind getKind() const {
    assert((recurrence || !executionCount) &&
           "an execution count requires a receiver address recurrence");
    if (!recurrence) {
      return ReceiverAddressSequenceProofKind::FullyDynamic;
    }
    return executionCount
               ? ReceiverAddressSequenceProofKind::KnownCount
               : ReceiverAddressSequenceProofKind::PeriodicUnknownCount;
  }
};

/// One transfer definition: one send and its corresponding receiver posts.
/// The sender and receiver operations may reference distinct
/// `ttl.pipe_transfer.create` declarations; those declarations do not define
/// transfer identity.
struct PipeTransferNode {
  PipeTransferNodeId id = 0;
  PipeKey pipe;
  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;
  Operation *sendOp = nullptr;
  SmallVector<Operation *> receiverPostOps;
  SmallVector<PipeReceiverEndpointId> receiverEndpoints;
};

/// One receiver connection for a transfer definition.
struct PipeReceiverEndpoint {
  PipeReceiverEndpointId id = 0;
  PipeTransferNodeId transferNode = 0;
  PipeReceiverDFBNodeId receiverDFBNode = 0;
  PipeReceiverCoord receiver;
  PipeReceiverDFBKey receiverDFB;
  ReceiverDFBInfo receiverDFBInfo;
  Operation *postOp = nullptr;
  ReceiverAddressSequenceProof addressSequence;
};

/// One receiver-local dataflow buffer node in the PipeNet graph.
struct PipeReceiverDFBNode {
  PipeReceiverDFBNodeId id = 0;
  PipeReceiverDFBKey receiverDFB;
  SmallVector<PipeReceiverEndpointId> writerEndpoints;
  /// Every producer-side DFB advance belongs to a pipe receive. Consumer
  /// releases are validated separately because they do not move the write
  /// pointer.
  bool hasProvenPipeOnlyProducerStream = false;
  /// Reason the producer stream was not proven. Empty after a successful proof.
  std::string pipeOnlyProducerStreamFailureReason;
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

/// Graph of transfer definitions, receiver endpoints, physical receiver DFBs,
/// and proven receiver address sequences.
/// Built after pipe receive copies have been expanded to pipe transfer ops.
class PipeGraph {
public:
  /// Analyze a module to find all pipe receivers and build the graph.
  /// Returns failure if validation detects invalid transfer correspondence or
  /// receiver DFB address geometry.
  static FailureOr<PipeGraph> build(ModuleOp mod,
                                    const PipeTransferIndex &transferIndex);

  /// Check if any pipes were found.
  bool hasPipes() const { return !pipeTransferNodes.empty(); }

  /// Verify that every multicast occurrence uses one address at all receivers.
  /// Point-to-point transfers may publish an otherwise unproven address.
  LogicalResult verifyCollectiveReceiverAddresses() const;

  ArrayRef<PipeTransferNode> getPipeTransferNodes() const {
    return pipeTransferNodes;
  }

  const PipeTransferNode &getPipeTransferNode(PipeTransferNodeId id) const {
    assert(id < pipeTransferNodes.size() && "invalid pipe transfer node id");
    return pipeTransferNodes[id];
  }

  const PipeTransferNode *
  getPipeTransferNodeForProtocolOp(Operation *op) const {
    auto it = transferNodeIdByProtocolOp.find(op);
    return it == transferNodeIdByProtocolOp.end()
               ? nullptr
               : &pipeTransferNodes[it->second];
  }

  ArrayRef<PipeReceiverEndpointId>
  getPipeReceiverEndpoints(PipeTransferNodeId transferNode) const {
    return getPipeTransferNode(transferNode).receiverEndpoints;
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

  /// Return one representative endpoint when every DFB producer reservation is
  /// represented in the graph and every endpoint selects the same byte address
  /// at each reachable transfer occurrence. Return null otherwise.
  const PipeReceiverEndpoint *
  getProvenReceiverAddressEndpoint(PipeTransferNodeId transferNode) const;

  ArrayRef<PipeReceiverEndpointId>
  getReceiverDFBWriterEndpoints(PipeReceiverDFBNodeId receiverDFBNode) const {
    return getReceiverDFBNode(receiverDFBNode).writerEndpoints;
  }

  bool hasLaunchGrid() const { return hasAnalyzedLaunchGrid; }

  LaunchNodeDomain getOperationLaunchDomain(Operation *op) const;

private:
  /// Record the DFB geometry and destination offset for one receive post.
  LogicalResult addPipeReceiver(Operation *op,
                                PipeTransferCreateOp transferCreateOp,
                                Value dst);

  /// Build endpoint slot sequences when receiver DFB posts have a proven
  /// sequential order. Unproven point-to-point sequences use
  /// receiver-published addresses.
  LogicalResult
  assignReceiverAddressSequences(const PipeTransferIndex &transferIndex,
                                 PipeGraphAnalysisState &state);

  LogicalResult rebuildEndpointGraph(const PipeTransferIndex &transferIndex,
                                     PipeGraphAnalysisState &state);

  LogicalResult
  provePipeOnlyReceiverProducerStreams(PipeGraphAnalysisState &state);

  llvm::MapVector<Operation *, ReceiverDFBInfo> receiverDFBByPost;
  SmallVector<PipeTransferNode, 0> pipeTransferNodes;
  llvm::DenseMap<Operation *, PipeTransferNodeId> transferNodeIdByProtocolOp;
  SmallVector<PipeReceiverEndpoint> pipeReceiverEndpoints;
  SmallVector<PipeReceiverDFBNode> receiverDFBNodes;
  bool hasAnalyzedLaunchGrid = false;
  /// Cached operation-keyed analysis facts are valid only before lowering
  /// starts erasing or replacing IR operations.
  llvm::DenseMap<Operation *, LaunchNodeDomain> operationLaunchDomains;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
