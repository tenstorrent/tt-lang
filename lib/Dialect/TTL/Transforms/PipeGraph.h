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
#include <optional>
#include <utility>

namespace mlir::tt::ttl {

struct PipeGraphAnalysisState;

//===----------------------------------------------------------------------===//
// Pipe Graph: Tracks receiver dataflow buffer associations for pipe copies.
// The graph validates that each logical pipe has a consistent destination DFB
// and enough DFB slots for overlapping writes.
//===----------------------------------------------------------------------===//

/// Node coordinate for a pipe destination within one logical device.
struct PipeReceiverCoord {
  int64_t x = 0;
  int64_t y = 0;

  bool operator==(const PipeReceiverCoord &other) const {
    return x == other.x && y == other.y;
  }
};

/// Receiver-local DFB identity. `receiverDevice` is null for intra-device
/// pipes and identifies the logical destination device for fabric pipes.
struct PipeReceiverDFBKey {
  DeviceRefAttr receiverDevice;
  PipeReceiverCoord receiver;
  int64_t dfbIndex = 0;

  bool operator==(const PipeReceiverDFBKey &other) const {
    return receiverDevice == other.receiverDevice &&
           receiver == other.receiver && dfbIndex == other.dfbIndex;
  }
};

using PipeReceiverDFBStreamKey = std::pair<DeviceRefAttr, int64_t>;

inline void printReceiverDFB(llvm::raw_ostream &os,
                             const PipeReceiverDFBKey &receiverDFB) {
  if (receiverDFB.receiverDevice) {
    os << "device " << receiverDFB.receiverDevice << " ";
  }
  os << "receiver(" << receiverDFB.receiver.x << ", " << receiverDFB.receiver.y
     << ") DFB " << receiverDFB.dfbIndex;
}

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

  bool containsReceiver(PipeReceiverCoord receiver) const {
    return receiver.x >= dstStartX && receiver.x <= dstEndX &&
           receiver.y >= dstStartY && receiver.y <= dstEndY;
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
    return hash_combine(receiverDFB.receiverDevice, receiverDFB.receiver.x,
                        receiverDFB.receiver.y, receiverDFB.dfbIndex);
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
  DeviceRefAttr receiverDevice; // Null for intra-device pipes.
  int64_t dfbIndex;             // DFB index (0-31) used by receiver
  CircularBufferType dfbType;   // Receiver DFB type
  bool hasStaticTileOffset;     // Whether staticTileOffset is known.
  int64_t staticTileOffset;     // Static destination tile offset within the DFB
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

/// One logical PipeNet edge from a source node to a receiver set.
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

inline PipeTransferContract getPipeTransferContract(PipeRecordAttr record) {
  return record.getIsMulticast() ? PipeTransferContract::Collective
                                 : PipeTransferContract::PointToPoint;
}

inline PipeKey getPipeKey(PipeRecordAttr record, int64_t pipeNetId) {
  return {record.getSrcX(),
          record.getSrcY(),
          record.getDstStartX(),
          record.getDstStartY(),
          record.getDstEndX(),
          record.getDstEndY(),
          pipeNetId};
}

inline PipeType getPipeTypeFromRecord(MLIRContext *context,
                                      PipeRecordAttr record,
                                      int64_t pipeNetId) {
  return PipeType::get(context, record.getSrcX(), record.getSrcY(),
                       record.getDstStartX(), record.getDstStartY(),
                       record.getDstEndX(), record.getDstEndY(), pipeNetId);
}

struct PipeReference {
  enum class Kind {
    Static,
    SelectedSrc,
    SelectedDst,
  };

  Kind kind = Kind::Static;
  Value value;
  PipeType pipeType;
  SelectPipeSrcOp selectedSrc;
  SelectPipeDstOp selectedDst;

  bool isStatic() const { return kind == Kind::Static; }
  bool isSelected() const { return !isStatic(); }
  bool isSelectedSrc() const { return kind == Kind::SelectedSrc; }
  bool isSelectedDst() const { return kind == Kind::SelectedDst; }

  PipeNetRecordsAttr getRecords() const {
    assert(isSelected() && "static pipe has no records attr");
    if (selectedSrc) {
      SelectPipeSrcOp op = selectedSrc;
      return op.getRecords();
    }
    SelectPipeDstOp op = selectedDst;
    return op.getRecords();
  }

  int64_t getPipeNetId() const {
    if (pipeType) {
      return pipeType.getPipeNetId();
    }
    if (selectedSrc) {
      SelectPipeSrcOp op = selectedSrc;
      return static_cast<int64_t>(op.getPipeNetId());
    }
    SelectPipeDstOp op = selectedDst;
    return static_cast<int64_t>(op.getPipeNetId());
  }
};

/// Return a static or selected pipe reference after tracing only unrealized
/// conversion casts. Selected references must be direct select op results.
FailureOr<PipeReference> getPipeReference(Operation *op, Value pipe);

/// Enumerate the static pipe types represented by a pipe reference.
SmallVector<PipeType> getPipeTypesFromReference(MLIRContext *context,
                                                const PipeReference &ref);

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
  LogicalResult
  addReceiverDFB(int64_t srcX, int64_t srcY, int64_t dstStartX,
                 int64_t dstStartY, int64_t dstEndX, int64_t dstEndY,
                 int64_t pipeNetId, int64_t dfbIndex,
                 DeviceRefAttr receiverDevice, CircularBufferType dfbType,
                 bool hasStaticTileOffset, int64_t staticTileOffset,
                 int64_t receiverSlotSpanBlocks, Operation *receiverReserveOp,
                 PipeTransferContract transferContract,
                 ArrayRef<Operation *> transferCreateOps, int64_t blockCount,
                 Location loc);

  /// Verify that each receiver post was assigned a non-wrapping physical DFB
  /// slot.
  LogicalResult verifyReceiverDFBBlockCounts() const;

  const ReceiverDFBInfo *lookupReceiverDFB(const PipeKey &key) const;

  ArrayRef<PipeTransferPostOp> getPipeReceiverPosts(const PipeKey &pipe) const;

  ArrayRef<PipeTransferSendOp> getPipeSends(const PipeKey &pipe) const;

  void appendReceiverDFBPops(const PipeReceiverDFBKey &receiverDFB,
                             SmallVectorImpl<CBPopOp> &pops) const;

  ArrayRef<PipeEdge> getPipeEdges() const { return pipeEdges; }

  const PipeEdge &getPipeEdge(PipeEdgeId id) const {
    assert(id < pipeEdges.size() && "invalid pipe edge id");
    return pipeEdges[id];
  }

  const PipeEdge *getPipeEdgeForPipe(const PipeKey &pipe) const {
    auto it = pipeEdgeIdByPipe.find(pipe);
    return it == pipeEdgeIdByPipe.end() ? nullptr : &pipeEdges[it->second];
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

  bool hasLaunchGrid() const { return hasAnalyzedLaunchGrid; }

  LaunchNodeDomain getOperationLaunchDomain(Operation *op) const;

  const DFBReleaseOwnerMaps &getDFBReleaseOwnerMaps() const {
    return dfbReleaseOwners;
  }

private:
  /// Assign each pipe the physical DFB slot reserved by the corresponding
  /// receiver post. Multicast pipes require the same receiver slot at every
  /// receiver because TT-Metal NoC multicast carries one destination address.
  LogicalResult assignReceiverSlotIndices(PipeGraphAnalysisState &state);

  void rebuildEndpointGraph();

  LogicalResult provePipeOnlyReceiverStreams(PipeGraphAnalysisState &state);

  llvm::MapVector<PipeKey, ReceiverDFBInfo> receiverDFBs;
  SmallVector<PipeEdge, 0> pipeEdges;
  llvm::DenseMap<PipeKey, PipeEdgeId> pipeEdgeIdByPipe;
  llvm::DenseMap<PipeKey, SmallVector<PipeTransferPostOp>> receiverPostsByPipe;
  llvm::DenseMap<PipeKey, SmallVector<PipeTransferSendOp>> sendsByPipe;
  llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<CBPopOp>>
      receiverPopsByStream;
  SmallVector<PipeReceiverEndpoint> pipeReceiverEndpoints;
  SmallVector<PipeReceiverDFBNode> receiverDFBNodes;
  bool hasAnalyzedLaunchGrid = false;
  /// Cached operation-keyed analysis facts are valid only before lowering
  /// starts erasing or replacing IR operations.
  llvm::DenseMap<Operation *, LaunchNodeDomain> operationLaunchDomains;
  DFBReleaseOwnerMaps dfbReleaseOwners;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
