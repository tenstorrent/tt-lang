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
#include "ttlang/Dialect/TTL/Transforms/PipeNetExecutionUtils.h"
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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace mlir::tt::ttl {

class PipeTransferIndex;
class DFBLogicalIdentityAnalysis;

struct PipeGraphAnalysisState;

/// Compiler-generated control flow that implements PipeNet foreach callbacks.
///
/// `controlOps` identifies loops and conditions that select records rather than
/// independent user control. `ifThenDomains` records the launch nodes that may
/// enter each generated `scf.if` across all record-loop iterations.
/// `recordLoops` identifies table-driven loops whose bodies execute once per
/// matching PipeNet record.
struct PipeForeachLoweringInfo {
  SmallVector<Operation *> controlOps;
  llvm::DenseMap<Operation *, LaunchNodeDomain> ifThenDomains;
  llvm::DenseMap<Operation *, PipeNetRecordLoop> recordLoops;
};

/// Meaning of `cb_index` while PipeGraph analyzes physical stream aliases.
enum class PipeDFBIndexMode {
  /// Indices may change during later physical allocation.
  Provisional,
  /// Indices are emitted as written without allocator proof.
  DeclaredPhysical,
  /// Allocation metadata proves the finalized physical assignment.
  Finalized,
};

/// Controls whether launch-node domains are required when the module contains
/// no pipe-transfer protocol operations.
enum class PipeGraphLaunchDomainMode {
  WhenPipesPresent,
  Required,
};

//===----------------------------------------------------------------------===//
// Pipe Graph: Tracks static transfers, receiver endpoints, logical receiver
// DFB lifecycles, physical allocation, and endpoint address sequences.
//===----------------------------------------------------------------------===//

/// Receiver node coordinate within one device. Cross-device analyses also
/// qualify DFB identity with the logical receiver device.
struct PipeReceiverCoord {
  int64_t x = 0;
  int64_t y = 0;

  bool operator==(const PipeReceiverCoord &other) const {
    return x == other.x && y == other.y;
  }
};

/// Logical receiver DFB lifecycle and physical allocation within one device.
struct PipeReceiverDFBKey {
  DeviceRefAttr receiverDevice;
  PipeReceiverCoord receiver;
  int64_t dfbIndex = 0;
  int64_t dfbId = 0;

  bool operator==(const PipeReceiverDFBKey &other) const {
    return receiverDevice == other.receiverDevice &&
           receiver == other.receiver && dfbIndex == other.dfbIndex &&
           dfbId == other.dfbId;
  }
};

/// Logical DFB stream at one physical index within a logical device.
struct PipeReceiverDFBStreamKey {
  DeviceRefAttr receiverDevice;
  int64_t dfbIndex = 0;
  int64_t dfbId = 0;

  bool operator==(const PipeReceiverDFBStreamKey &other) const {
    return receiverDevice == other.receiverDevice &&
           dfbIndex == other.dfbIndex && dfbId == other.dfbId;
  }
};

using PipeReceiverDFBPhysicalStreamKey = std::pair<DeviceRefAttr, int64_t>;

inline void printReceiverDFB(llvm::raw_ostream &os,
                             const PipeReceiverDFBKey &receiverDFB) {
  if (receiverDFB.receiverDevice) {
    os << "device " << receiverDFB.receiverDevice << " ";
  }
  os << "receiver(" << receiverDFB.receiver.x << ", " << receiverDFB.receiver.y
     << ") DFB ";
  if (receiverDFB.dfbId != receiverDFB.dfbIndex) {
    os << receiverDFB.dfbId << " at physical index ";
  }
  os << receiverDFB.dfbIndex;
}

inline std::string
getReceiverDFBIdentityString(const PipeReceiverDFBKey &receiverDFB) {
  std::string identity = "receiver DFB " + std::to_string(receiverDFB.dfbId);
  if (receiverDFB.dfbId != receiverDFB.dfbIndex) {
    identity += " at physical index " + std::to_string(receiverDFB.dfbIndex);
  }
  return identity;
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

  bool operator!=(const PipeKey &other) const { return !(*this == other); }

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
                        receiverDFB.receiver.y, receiverDFB.dfbIndex,
                        receiverDFB.dfbId);
  }
  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
};

template <>
struct DenseMapInfo<mlir::tt::ttl::PipeReceiverDFBStreamKey> {
  using Key = mlir::tt::ttl::PipeReceiverDFBStreamKey;
  static unsigned getHashValue(const Key &stream) {
    return hash_combine(stream.receiverDevice, stream.dfbIndex, stream.dfbId);
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
  int64_t dfbId;
  CircularBufferType dfbType;
  bool isTensorBacked;
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
/// `ttl.pipe_transfer.create` declarations. The graph verifies and records
/// their common transfer contract and logical-device transfer.
struct PipeTransferNode {
  PipeTransferNodeId id = 0;
  PipeKey pipe;
  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;
  DeviceTransferAttr deviceTransfer;
  std::optional<std::uint64_t> sendRecordIndex;
  int64_t blockSpan = 1;
  int64_t destinationGroupDepth = 1;
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
  std::optional<std::uint64_t> postRecordIndex;
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
  /// Every physical write-pointer advance not represented by this receiver's
  /// pipe endpoints returns the pointer to the same DFB slot.
  bool hasProvenComputedAddressProducerPhase = false;
  /// Reason the physical write-pointer phase was not proven.
  std::string computedAddressProducerPhaseFailureReason;
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
  return record.getIsCollective() ? PipeTransferContract::Collective
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

/// A pipe operand represented either by a static type or by one selected
/// source or destination record.
class PipeReference {
public:
  PipeReference() = delete;
  explicit PipeReference(PipeType pipeType) : value(pipeType) {
    assert(pipeType && "static pipe reference requires a pipe type");
  }
  explicit PipeReference(SelectPipeSrcOp selectedSrc) : value(selectedSrc) {
    assert(selectedSrc && "selected source reference requires an operation");
  }
  explicit PipeReference(SelectPipeDstOp selectedDst) : value(selectedDst) {
    assert(selectedDst &&
           "selected destination reference requires an operation");
  }

  bool isStatic() const { return std::holds_alternative<PipeType>(value); }
  bool isSelected() const { return !isStatic(); }
  bool isSelectedSrc() const {
    return std::holds_alternative<SelectPipeSrcOp>(value);
  }
  bool isSelectedDst() const {
    return std::holds_alternative<SelectPipeDstOp>(value);
  }

  PipeType getStaticPipeType() const {
    assert(isStatic() && "selected pipe reference has no static pipe type");
    return std::get<PipeType>(value);
  }

  SelectPipeSrcOp getSelectedSrc() const {
    assert(isSelectedSrc() && "pipe reference is not a selected source");
    return std::get<SelectPipeSrcOp>(value);
  }

  SelectPipeDstOp getSelectedDst() const {
    assert(isSelectedDst() && "pipe reference is not a selected destination");
    return std::get<SelectPipeDstOp>(value);
  }

  PipeNetRecordsAttr getRecords() const {
    assert(isSelected() && "static pipe has no records attr");
    if (isSelectedSrc()) {
      return getSelectedSrc().getRecords();
    }
    return getSelectedDst().getRecords();
  }

  int64_t getPipeNetId() const {
    if (isStatic()) {
      return getStaticPipeType().getPipeNetId();
    }
    return getRecords().getPipeNetId();
  }

  Operation *getSelectedOperation() const {
    assert(isSelected() && "static pipe reference has no selection operation");
    if (isSelectedSrc()) {
      return getSelectedSrc().getOperation();
    }
    return getSelectedDst().getOperation();
  }

private:
  std::variant<PipeType, SelectPipeSrcOp, SelectPipeDstOp> value;
};

/// Return a static or selected pipe reference after tracing only unrealized
/// conversion casts. Requiring direct select results keeps each selected value
/// tied to the record table that defines it.
FailureOr<PipeReference> getPipeReference(Operation *op, Value pipe);

/// Return the pipe referenced by a send, receiver post, or receiver wait.
FailureOr<PipeReference>
getPipeReferenceForProtocolOp(Operation *protocolOp,
                              const PipeTransferIndex &transferIndex);

/// Enumerate the static pipe types represented by a pipe reference.
SmallVector<PipeType> getPipeTypesFromReference(MLIRContext *context,
                                                const PipeReference &ref);

/// Return the logical-device transfer associated with one pipe record.
DeviceTransferAttr
getPipeRecordDeviceTransfer(const PipeReference &ref, std::size_t recordIndex,
                            DeviceTransferAttr staticDeviceTransfer);

/// Return the number of original DFB blocks delivered by one transfer.
inline int64_t getPipeTransferBlockSpan(PipeTransferCreateOp op) {
  return static_cast<int64_t>(op.getBlockSpan());
}

/// Return the maximum number of resident transfers planned per receiver DFB.
inline int64_t getPipeTransferDestinationGroupDepth(PipeTransferCreateOp op) {
  return static_cast<int64_t>(op.getDestinationGroupDepth());
}

/// Graph of transfer definitions, receiver endpoints, physical receiver DFBs,
/// and proven receiver address sequences.
/// Built after pipe receive copies have been expanded to pipe transfer ops.
class PipeGraph {
public:
  /// Analyze a module to find all pipe receivers and build the graph.
  /// Returns failure if validation detects invalid transfer correspondence or
  /// receiver DFB address geometry.
  /// `foreachLoweringInfo` identifies compiler-generated record control so it
  /// is not interpreted as independent user control.
  static FailureOr<PipeGraph>
  build(ModuleOp mod, const PipeTransferIndex &transferIndex,
        const PipeForeachLoweringInfo &foreachLoweringInfo,
        PipeDFBIndexMode dfbIndexMode,
        PipeGraphLaunchDomainMode launchDomainMode);

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

  ArrayRef<PipeTransferNodeId>
  getPipeTransferNodeIdsForProtocolOp(Operation *op) const {
    auto it = transferNodeIdsByProtocolOp.find(op);
    return it == transferNodeIdsByProtocolOp.end()
               ? ArrayRef<PipeTransferNodeId>{}
               : ArrayRef<PipeTransferNodeId>{it->second};
  }

  bool hasPipeTransferNodeForProtocolOp(Operation *op) const {
    return !getPipeTransferNodeIdsForProtocolOp(op).empty();
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

  /// Return one representative endpoint when non-pipe producer advances
  /// preserve the physical write-pointer phase and every endpoint selects the
  /// same byte address at each reachable transfer occurrence.
  const PipeReceiverEndpoint *
  getProvenReceiverAddressEndpoint(PipeTransferNodeId transferNode) const;

  ArrayRef<PipeReceiverEndpointId>
  getReceiverDFBWriterEndpoints(PipeReceiverDFBNodeId receiverDFBNode) const {
    return getReceiverDFBNode(receiverDFBNode).writerEndpoints;
  }

  bool hasLaunchGrid() const { return hasAnalyzedLaunchGrid; }

  LaunchNodeDomain getOperationLaunchDomain(Operation *op) const;

  /// Returns DFB acquisition and release relations for the enclosing kernel.
  const DFBAcquireReleaseIndex &
  getDFBAcquireReleaseIndex(Operation *operation) const;

  /// Append DFB pops that may execute for this logical receiver stream.
  void appendReceiverDFBPops(const PipeReceiverDFBKey &receiverDFB,
                             SmallVectorImpl<CBPopOp> &pops) const;

private:
  /// Associate a selected protocol operation and record with one transfer.
  void
  recordTransferNodeForProtocolRecord(Operation *op,
                                      std::optional<std::uint64_t> recordIndex,
                                      PipeTransferNodeId transferNodeId);

  /// Record the DFB geometry and destination offset for one receive post.
  LogicalResult addPipeReceiver(Operation *op,
                                PipeTransferCreateOp transferCreateOp,
                                Value dst,
                                const DFBLogicalIdentityAnalysis &dfbIds);

  /// Build endpoint slot sequences when receiver DFB posts have a proven
  /// sequential order. Unproven point-to-point sequences use
  /// receiver-published addresses.
  LogicalResult
  assignReceiverAddressSequences(ModuleOp mod,
                                 const PipeTransferIndex &transferIndex,
                                 PipeGraphAnalysisState &state);

  LogicalResult rebuildEndpointGraph(const PipeTransferIndex &transferIndex,
                                     PipeGraphAnalysisState &state);

  LogicalResult proveReceiverProducerStreams(PipeGraphAnalysisState &state);

  llvm::MapVector<Operation *, ReceiverDFBInfo> receiverDFBByPost;
  SmallVector<PipeTransferNode, 0> pipeTransferNodes;
  // A table-driven protocol operation represents one transfer node per record.
  llvm::DenseMap<Operation *, SmallVector<PipeTransferNodeId>>
      transferNodeIdsByProtocolOp;
  // Selected protocol operations execute once for each matching record. This
  // map preserves that identity when equal PipeKeys occur more than once.
  llvm::DenseMap<std::pair<Operation *, std::uint64_t>, PipeTransferNodeId>
      transferNodeIdByProtocolOpAndRecord;
  SmallVector<PipeReceiverEndpoint> pipeReceiverEndpoints;
  SmallVector<PipeReceiverDFBNode> receiverDFBNodes;
  llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<CBPopOp>>
      receiverPopsByStream;
  bool hasAnalyzedLaunchGrid = false;
  /// Cached operation-keyed analysis facts are valid only before lowering
  /// starts erasing or replacing IR operations.
  llvm::DenseMap<Operation *, LaunchNodeDomain> operationLaunchDomains;
  llvm::DenseMap<Operation *, std::unique_ptr<DFBAcquireReleaseIndex>>
      dfbLifecycles;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPEGRAPH_H
