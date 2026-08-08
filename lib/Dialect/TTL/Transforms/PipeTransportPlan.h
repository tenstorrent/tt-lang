// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Pipe Transport Plan
//===----------------------------------------------------------------------===//
//
// This file declares backend-independent scheduling and storage decisions for
// PipeNet transfer streams.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSPORTPLAN_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSPORTPLAN_H

#include "PipeGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace llvm {
class raw_ostream;
}

namespace mlir::tt::ttl {

class PipeCapacityPlan;

using PipeTransportStreamId = std::size_t;

/// Synchronization performed before the sender writes a transfer payload.
enum class PipeSynchronizationProtocol {
  ReceiverPost,
  Capacity,
  Fabric,
};

/// Backend-independent schedule selected for one transfer stream.
enum class PipeTransportSchedule {
  Scalar,
  Grouped,
  Overlapped,
};

/// Component responsible for advancing one transport storage allocation.
enum class PipeTransportStorageOwnership {
  DFB,
  Transport,
};

/// Transport role associated with one direct scratch storage access.
enum class PipeTransportStorageRole {
  Source,
  Destination,
};

/// Required completion point for transport-owned credit updates.
enum class PipeTransportCreditCompletion {
  /// Complete the update before continuing past its operation.
  Immediate,
  /// Complete updates after the innermost loop in their iteration domain.
  IterationDomain,
};

/// Condition that permits source storage to be reused.
enum class PipeTransportSourceReuse {
  AfterCompletionGroup,
};

/// Enclosing loops ordered from outermost to innermost.
struct PipeTransportIterationDomain {
  SmallVector<Operation *, 2> enclosingLoops;
};

/// Backend-independent source storage selected for one stream.
struct PipeTransportSourceStorage {
  int64_t blockCount = 1;
  int64_t blocksPerTransfer = 1;
  int64_t stageDepth = 1;
  int64_t scratchByteOffset = 0;
  int64_t scratchBytes = 0;
  PipeTransportStorageOwnership ownership = PipeTransportStorageOwnership::DFB;
};

/// Direct address calculation for one transport-owned tensor copy.
struct PipeTransportStorageAccess {
  PipeTransportStorageRole role = PipeTransportStorageRole::Source;
  int64_t blockCount = 1;
  int64_t blockStrideBytes = 0;
  int64_t scratchByteOffset = 0;
  std::optional<int64_t> dynamicSlotCounterIndex;
};

/// Initial value for one receiver-local transport storage slot counter.
struct PipeTransportSlotCounterInitInfo {
  int64_t counterIndex = 0;
  int64_t initialSlot = 0;
};

/// Logical payload pages copied by one original transfer.
struct PipeTransportPacketization {
  int64_t pageCount = 0;
  int64_t pageSizeBytes = 0;
  int64_t payloadSizeBytes = 0;

  /// Return the complete payload size in bytes.
  int64_t getPayloadSizeBytes() const { return payloadSizeBytes; }
};

/// Destination storage and address sequence for one receiver endpoint.
struct PipeTransportEndpoint {
  PipeReceiverEndpointId endpoint = 0;
  PipeReceiverCoord destination;
  PipeReceiverDFBKey receiverDFB;
  int64_t slotSpanBlocks = 1;
  int64_t blockCount = 1;
  int64_t groupDepth = 1;
  int64_t scratchByteOffset = 0;
  int64_t scratchBytes = 0;
  PipeTransportStorageOwnership ownership = PipeTransportStorageOwnership::DFB;
  PipeTransportIterationDomain iterationDomain;
  ReceiverAddressSequenceProof addressSequence;
};

/// Transfers that must complete before source storage can be reused.
struct PipeTransportCompletionGroup {
  SmallVector<PipeReceiverEndpointId, 1> endpoints;
};

/// Backend-independent execution plan for one PipeGraph transfer node.
class PipeTransportStream {
public:
  /// Return the stream's stable index in its containing plan.
  PipeTransportStreamId getId() const { return id; }

  /// Return the PipeGraph transfer represented by this stream.
  PipeTransferNodeId getTransferNode() const { return transferNode; }

  /// Return the stream's logical source-to-destination relation.
  const PipeKey &getPipe() const { return pipe; }

  /// Return the transfer's point-to-point or collective contract.
  PipeTransferContract getTransferContract() const { return transferContract; }

  /// Return the synchronization protocol selected for this stream.
  PipeSynchronizationProtocol getSynchronizationProtocol() const {
    return synchronizationProtocol;
  }

  /// Return the selected backend-independent schedule.
  PipeTransportSchedule getSchedule() const { return schedule; }

  /// Return when transport-owned credit updates must complete.
  PipeTransportCreditCompletion getCreditCompletion() const {
    return creditCompletion;
  }

  /// Return the number of original transfers represented by one group.
  int64_t getLogicalTransfersPerGroup() const {
    return logicalTransfersPerGroup;
  }

  /// Return the source execution domain.
  const PipeTransportIterationDomain &getSourceIterationDomain() const {
    return sourceIterationDomain;
  }

  /// Return the source storage decision.
  const PipeTransportSourceStorage &getSourceStorage() const {
    return sourceStorage;
  }

  /// Return the payload page decomposition.
  const PipeTransportPacketization &getPacketization() const {
    return packetization;
  }

  /// Return receiver endpoint decisions in PipeGraph order.
  ArrayRef<PipeTransportEndpoint> getEndpoints() const { return endpoints; }

  /// Return the iteration domains containing receiver capacity releases.
  ArrayRef<PipeTransportIterationDomain>
  getCapacityReleaseIterationDomains() const {
    return capacityReleaseIterationDomains;
  }

  /// Return the transfers that define source completion.
  const PipeTransportCompletionGroup &getCompletionGroup() const {
    return completionGroup;
  }

  /// Return the condition that permits source storage reuse.
  PipeTransportSourceReuse getSourceReuse() const { return sourceReuse; }

  /// Print a deterministic representation suitable for debug tests.
  void print(llvm::raw_ostream &os) const;

private:
  friend FailureOr<class PipeTransportPlan> buildPipeTransportPlan(
      const PipeGraph &, const PipeCapacityPlan &,
      function_ref<PipeSynchronizationProtocol(PipeTransferNodeId)>);

  PipeTransportStreamId id = 0;
  PipeTransferNodeId transferNode = 0;
  PipeKey pipe;
  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;
  PipeSynchronizationProtocol synchronizationProtocol =
      PipeSynchronizationProtocol::ReceiverPost;
  PipeTransportSchedule schedule = PipeTransportSchedule::Scalar;
  PipeTransportCreditCompletion creditCompletion =
      PipeTransportCreditCompletion::Immediate;
  int64_t logicalTransfersPerGroup = 1;
  PipeTransportIterationDomain sourceIterationDomain;
  PipeTransportSourceStorage sourceStorage;
  PipeTransportPacketization packetization;
  SmallVector<PipeTransportEndpoint, 1> endpoints;
  SmallVector<PipeTransportIterationDomain, 1> capacityReleaseIterationDomains;
  PipeTransportCompletionGroup completionGroup;
  PipeTransportSourceReuse sourceReuse =
      PipeTransportSourceReuse::AfterCompletionGroup;
};

/// Backend-independent transport decisions for all PipeGraph transfer streams.
class PipeTransportPlan {
public:
  /// Return streams in deterministic PipeGraph order.
  ArrayRef<PipeTransportStream> getStreams() const { return streams; }

  /// Return the per-core scratch bytes required by transport-owned storage.
  int64_t getSramScratchBytes() const { return sramScratchBytes; }

  /// Return the stream with `id`.
  const PipeTransportStream &getStream(PipeTransportStreamId id) const;

  /// Return the stream for the PipeGraph transfer `transferNode`.
  const PipeTransportStream &
  getStreamForTransfer(PipeTransferNodeId transferNode) const;

  /// Return the unique stream that owns `operation`.
  const PipeTransportStream &getStreamForOperation(Operation *operation) const;

  /// Return whether transport synchronization replaces this DFB operation.
  bool ownsDFBLifecycle(Operation *operation) const;

  /// Return the direct storage decision for `operation`, if present.
  const PipeTransportStorageAccess *
  lookupStorageAccess(Operation *operation) const;

  /// Return receiver-local slot counters grouped by kernel function.
  const llvm::MapVector<func::FuncOp,
                        SmallVector<PipeTransportSlotCounterInitInfo>> &
  getSlotCounterInitializations() const {
    return slotCounterInitializations;
  }

  /// Print all stream decisions deterministically.
  void print(llvm::raw_ostream &os) const;

private:
  friend FailureOr<PipeTransportPlan> buildPipeTransportPlan(
      const PipeGraph &, const PipeCapacityPlan &,
      function_ref<PipeSynchronizationProtocol(PipeTransferNodeId)>);

  SmallVector<PipeTransportStream, 0> streams;
  llvm::DenseMap<PipeTransferNodeId, PipeTransportStreamId> streamByTransfer;
  llvm::DenseMap<Operation *, PipeTransportStreamId> streamByOperation;
  llvm::DenseSet<Operation *> ownedDFBLifecycleOperations;
  llvm::DenseMap<Operation *, PipeTransportStorageAccess>
      storageAccessByOperation;
  llvm::MapVector<func::FuncOp, SmallVector<PipeTransportSlotCounterInitInfo>>
      slotCounterInitializations;
  int64_t sramScratchBytes = 0;
};

/// Construct transport streams from proven PipeGraph and capacity facts.
FailureOr<PipeTransportPlan> buildPipeTransportPlan(
    const PipeGraph &pipeGraph, const PipeCapacityPlan &capacityPlan,
    function_ref<PipeSynchronizationProtocol(PipeTransferNodeId)>
        selectSynchronizationProtocol);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSPORTPLAN_H
