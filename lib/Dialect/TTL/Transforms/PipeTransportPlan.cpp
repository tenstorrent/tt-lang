// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeTransportPlan.h"

#include "PipePlanning.h"
#include "PipeTransportDFBAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Transforms/PipeConstants.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>

#define DEBUG_TYPE "ttl-pipe-transport-plan"

namespace mlir::tt::ttl {

const PipeTransportStream &
PipeTransportPlan::getStream(PipeTransportStreamId id) const {
  assert(id < streams.size() && "invalid pipe transport stream id");
  return streams[id];
}

const PipeTransportStream &
PipeTransportPlan::getStreamForTransfer(PipeTransferNodeId transferNode) const {
  auto streamIt = streamByTransfer.find(transferNode);
  assert(streamIt != streamByTransfer.end() &&
         "pipe transfer has no transport stream");
  return getStream(streamIt->second);
}

const PipeTransportStream &
PipeTransportPlan::getStreamForOperation(Operation *operation) const {
  auto streamIt = streamByOperation.find(operation);
  assert(streamIt != streamByOperation.end() &&
         "pipe protocol operation has no transport stream");
  return getStream(streamIt->second);
}

bool PipeTransportPlan::ownsDFBLifecycle(Operation *operation) const {
  return ownedDFBLifecycleOperations.contains(operation);
}

const PipeTransportStorageAccess *
PipeTransportPlan::lookupStorageAccess(Operation *operation) const {
  auto accessIt = storageAccessByOperation.find(operation);
  return accessIt == storageAccessByOperation.end() ? nullptr
                                                    : &accessIt->second;
}

/// Return the stable debug spelling for a transfer contract.
static StringRef stringifyTransferContract(PipeTransferContract contract) {
  switch (contract) {
  case PipeTransferContract::PointToPoint:
    return "point_to_point";
  case PipeTransferContract::Collective:
    return "collective";
  }
  llvm_unreachable("unknown pipe transfer contract");
}

/// Return the stable debug spelling for a synchronization protocol.
static StringRef
stringifySynchronizationProtocol(PipeSynchronizationProtocol protocol) {
  switch (protocol) {
  case PipeSynchronizationProtocol::ReceiverPost:
    return "receiver_post";
  case PipeSynchronizationProtocol::Capacity:
    return "capacity";
  }
  llvm_unreachable("unknown pipe synchronization protocol");
}

/// Return the stable debug spelling for a transport schedule.
static StringRef stringifySchedule(PipeTransportSchedule schedule) {
  switch (schedule) {
  case PipeTransportSchedule::Scalar:
    return "scalar";
  case PipeTransportSchedule::Grouped:
    return "grouped";
  case PipeTransportSchedule::Overlapped:
    return "overlapped";
  }
  llvm_unreachable("unknown pipe transport schedule");
}

/// Return the stable debug spelling for storage ownership.
static StringRef
stringifyStorageOwnership(PipeTransportStorageOwnership ownership) {
  switch (ownership) {
  case PipeTransportStorageOwnership::DFB:
    return "dfb";
  case PipeTransportStorageOwnership::Transport:
    return "transport";
  }
  llvm_unreachable("unknown pipe transport storage ownership");
}

/// Return the stable debug spelling for a credit completion point.
static StringRef
stringifyCreditCompletion(PipeTransportCreditCompletion completion) {
  switch (completion) {
  case PipeTransportCreditCompletion::Immediate:
    return "immediate";
  case PipeTransportCreditCompletion::IterationDomain:
    return "iteration_domain";
  }
  llvm_unreachable("unknown pipe transport credit completion");
}

/// Return whether storage, address, and loop proofs permit bounded overlap.
static bool supportsOverlappedSchedule(const PipeTransportStream &stream) {
  if (stream.getSourceIterationDomain().enclosingLoops.empty() ||
      stream.getCapacityReleaseIterationDomains().empty() ||
      llvm::any_of(stream.getCapacityReleaseIterationDomains(),
                   [](const PipeTransportIterationDomain &domain) {
                     return domain.enclosingLoops.empty();
                   })) {
    return false;
  }
  return llvm::all_of(
      stream.getEndpoints(), [](const PipeTransportEndpoint &endpoint) {
        if (endpoint.groupDepth < 2 || !endpoint.addressSequence.recurrence ||
            !endpoint.addressSequence.executionCount ||
            *endpoint.addressSequence.executionCount < 2) {
          return false;
        }
        const ReceiverAddressRecurrence &recurrence =
            *endpoint.addressSequence.recurrence;
        return recurrence.repeatStride != 0 &&
               recurrence.blockCount >= 2 * endpoint.slotSpanBlocks;
      });
}

/// Return the stable debug spelling for a source-reuse condition.
static StringRef stringifySourceReuse(PipeTransportSourceReuse sourceReuse) {
  switch (sourceReuse) {
  case PipeTransportSourceReuse::AfterCompletionGroup:
    return "after_completion_group";
  }
  llvm_unreachable("unknown pipe transport source-reuse condition");
}

/// Print a receiver sequence without exposing backend address calculations.
static void printAddressSequence(llvm::raw_ostream &os,
                                 const ReceiverAddressSequenceProof &sequence) {
  if (!sequence.recurrence) {
    os << "unproven";
    return;
  }

  const ReceiverAddressRecurrence &recurrence = *sequence.recurrence;
  os << "recurrence(initial=" << recurrence.initialSlot
     << ", stride=" << recurrence.repeatStride
     << ", modulus=" << recurrence.blockCount;
  if (sequence.executionCount) {
    os << ", executions=" << *sequence.executionCount;
  } else {
    os << ", executions=unbounded";
  }
  os << ")";
}

void PipeTransportStream::print(llvm::raw_ostream &os) const {
  os << "PipeTransport: stream " << id << " transfer " << transferNode
     << " src(" << pipe.srcX << ", " << pipe.srcY << ") -> dst("
     << pipe.dstStartX << ", " << pipe.dstStartY << ") to (" << pipe.dstEndX
     << ", " << pipe.dstEndY << ") net " << pipe.pipeNetId
     << " contract=" << stringifyTransferContract(transferContract)
     << " synchronization="
     << stringifySynchronizationProtocol(synchronizationProtocol)
     << " schedule=" << stringifySchedule(schedule)
     << " credit_completion=" << stringifyCreditCompletion(creditCompletion)
     << " group=" << logicalTransfersPerGroup << "\n";
  os << "PipeTransport:   source blocks=" << sourceStorage.blockCount
     << " block_span=" << sourceStorage.blocksPerTransfer
     << " stage_depth=" << sourceStorage.stageDepth
     << " ownership=" << stringifyStorageOwnership(sourceStorage.ownership)
     << " scratch_offset=" << sourceStorage.scratchByteOffset
     << " scratch_bytes=" << sourceStorage.scratchBytes
     << " pages=" << packetization.pageCount
     << " page_bytes=" << packetization.pageSizeBytes
     << " loops=" << sourceIterationDomain.enclosingLoops.size() << "\n";

  for (const PipeTransportEndpoint &endpoint : endpoints) {
    os << "PipeTransport:   endpoint " << endpoint.endpoint << " dst("
       << endpoint.destination.x << ", " << endpoint.destination.y << ") DFB "
       << endpoint.receiverDFB.dfbIndex
       << " block_count=" << endpoint.blockCount
       << " slot_span=" << endpoint.slotSpanBlocks
       << " group_depth=" << endpoint.groupDepth
       << " ownership=" << stringifyStorageOwnership(endpoint.ownership)
       << " scratch_offset=" << endpoint.scratchByteOffset
       << " scratch_bytes=" << endpoint.scratchBytes
       << " loops=" << endpoint.iterationDomain.enclosingLoops.size()
       << " address=";
    printAddressSequence(os, endpoint.addressSequence);
    os << "\n";
  }

  os << "PipeTransport:   completion endpoints=[";
  llvm::interleaveComma(completionGroup.endpoints, os);
  os << "] source_reuse=" << stringifySourceReuse(sourceReuse) << "\n";
}

void PipeTransportPlan::print(llvm::raw_ostream &os) const {
  for (const PipeTransportStream &stream : streams) {
    stream.print(os);
  }
}

/// Collect enclosing loops from outermost to innermost.
static PipeTransportIterationDomain getIterationDomain(Operation *operation) {
  PipeTransportIterationDomain domain;
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<LoopLikeOpInterface>(parent)) {
      domain.enclosingLoops.push_back(parent);
    }
  }
  std::reverse(domain.enclosingLoops.begin(), domain.enclosingLoops.end());
  return domain;
}

/// Proven transport-owned DFB lifecycle and direct storage access.
struct PipeTransportStorageSelection {
  PipeTransportDFBUse dfbUse;
  PipeTransportStorageAccess access;
};

/// Select fixed-base source storage for one grouped payload.
static PipeTransportStorageSelection
selectSourceStorage(PipeTransportDFBUse dfbUse, int64_t payloadSizeBytes) {
  PipeTransportStorageAccess access;
  access.role = PipeTransportStorageRole::Source;
  access.blockCount = 1;
  access.blockStrideBytes = payloadSizeBytes;
  return PipeTransportStorageSelection{std::move(dfbUse), access};
}

/// Select fixed or bounded-ring receiver storage for a grouped stream.
static PipeTransportStorageSelection
selectDestinationStorage(PipeTransportDFBUse dfbUse,
                         const PipeTransportEndpoint &endpoint,
                         int64_t payloadSizeBytes) {
  PipeTransportStorageAccess access;
  access.role = PipeTransportStorageRole::Destination;
  access.blockCount = endpoint.groupDepth;
  access.blockStrideBytes = payloadSizeBytes;
  return PipeTransportStorageSelection{std::move(dfbUse), access};
}

/// Return `value` aligned for PipeNet scratch, or no value on overflow.
static std::optional<int64_t> alignPipeScratchBytes(int64_t value) {
  if (value < 0 || value > std::numeric_limits<int64_t>::max() -
                               (kPipeSramScratchAlignmentBytes - 1)) {
    return std::nullopt;
  }
  return llvm::alignTo(value, kPipeSramScratchAlignmentBytes);
}

FailureOr<PipeTransportPlan> buildPipeTransportPlan(
    const PipeGraph &pipeGraph, const PipeCapacityPlan &capacityPlan,
    function_ref<PipeSynchronizationProtocol(PipeTransferNodeId)>
        selectSynchronizationProtocol) {
  PipeTransportPlan plan;
  llvm::MapVector<func::FuncOp, int64_t> nextSlotCounterIndex;
  auto recordStorageSelection = [&](PipeTransportStorageSelection &selection) {
    auto recordLifecycleOperation = [&](Operation *operation) {
      bool inserted = plan.ownedDFBLifecycleOperations.insert(operation).second;
      assert(inserted &&
             "DFB lifecycle operation belongs to multiple transports");
    };
    for (CBReserveOp reserveOp : selection.dfbUse.reserves) {
      recordLifecycleOperation(reserveOp);
    }
    for (CBPushOp pushOp : selection.dfbUse.pushes) {
      recordLifecycleOperation(pushOp);
    }
    for (CBWaitOp waitOp : selection.dfbUse.waits) {
      recordLifecycleOperation(waitOp);
    }
    for (CBPopOp popOp : selection.dfbUse.pops) {
      recordLifecycleOperation(popOp);
    }

    auto [accessIt, inserted] = plan.storageAccessByOperation.try_emplace(
        selection.dfbUse.tensorCopy.getOperation(), selection.access);
    (void)accessIt;
    assert(inserted && "DFB copy belongs to multiple transports");
    if (selection.access.dynamicSlotCounterIndex) {
      for (CBPopOp popOp : selection.dfbUse.pops) {
        auto [popAccessIt, popInserted] =
            plan.storageAccessByOperation.try_emplace(popOp, selection.access);
        (void)popAccessIt;
        assert(popInserted &&
               "DFB pop belongs to multiple transport storage plans");
      }
    }
  };

  for (const PipeTransferNode &transferNode :
       pipeGraph.getPipeTransferNodes()) {
    auto sendOp = cast<PipeTransferSendOp>(transferNode.sendOp);
    // A transport stream requires one statically identified transfer per
    // sender operation. Record-selected senders retain record-table lowering.
    ArrayRef<PipeTransferNodeId> sendTransferNodes =
        pipeGraph.getPipeTransferNodeIdsForProtocolOp(sendOp);
    assert(!sendTransferNodes.empty() &&
           "PipeGraph sender must identify at least one transfer");
    if (sendTransferNodes.size() != 1) {
      assert(transferNode.blockSpan == 1 &&
             "record-selected transfers must remain scalar");
      continue;
    }
    std::string ownershipFailure;
    FailureOr<PipeTransportDFBOwnership> storageOwnership =
        analyzePipeTransportDFBOwnership(transferNode, pipeGraph,
                                         ownershipFailure);
    auto sourceDFBType = cast<CircularBufferType>(sendOp.getSrc().getType());

    PipeTransportStream stream;
    stream.id = plan.streams.size();
    stream.transferNode = transferNode.id;
    stream.pipe = transferNode.pipe;
    stream.transferContract = transferNode.transferContract;
    stream.synchronizationProtocol =
        selectSynchronizationProtocol(transferNode.id);
    stream.schedule = transferNode.blockSpan == 1
                          ? PipeTransportSchedule::Scalar
                          : PipeTransportSchedule::Grouped;
    stream.logicalTransfersPerGroup = transferNode.blockSpan;
    stream.sourceIterationDomain = getIterationDomain(sendOp.getOperation());
    stream.sourceStorage = PipeTransportSourceStorage{
        sourceDFBType.getBlockCount(), transferNode.blockSpan,
        sourceDFBType.getBlockCount() / transferNode.blockSpan};
    FailureOr<PipeTransferPayload> maybePayload =
        getPipeTransferPayload(sendOp, transferNode.blockSpan);
    if (failed(maybePayload)) {
      return failure();
    }
    stream.packetization = PipeTransportPacketization{
        maybePayload->elementCount, maybePayload->elementSizeBytes,
        maybePayload->sizeBytes};
    stream.sourceReuse = PipeTransportSourceReuse::AfterCompletionGroup;

    for (PipeReceiverEndpointId endpointId : transferNode.receiverEndpoints) {
      const PipeReceiverEndpoint &graphEndpoint =
          pipeGraph.getPipeReceiverEndpoint(endpointId);
      PipeTransportEndpoint endpoint;
      endpoint.endpoint = graphEndpoint.id;
      endpoint.destination = graphEndpoint.receiver;
      endpoint.receiverDFB = graphEndpoint.receiverDFB;
      endpoint.slotSpanBlocks =
          graphEndpoint.receiverDFBInfo.receiverSlotSpanBlocks;
      bool transportOwnsEndpoint = succeeded(storageOwnership) &&
                                   storageOwnership->endpoint == endpointId;
      int64_t selectedGroupDepth =
          transportOwnsEndpoint && stream.synchronizationProtocol ==
                                       PipeSynchronizationProtocol::ReceiverPost
              ? 1
              : transferNode.destinationGroupDepth;
      std::optional<int64_t> requiredReceiverBlocks =
          llvm::checkedMul(endpoint.slotSpanBlocks, selectedGroupDepth);
      if (!requiredReceiverBlocks) {
        sendOp.emitError("receiver storage size exceeds int64_t");
        return failure();
      }
      endpoint.blockCount = transportOwnsEndpoint
                                ? *requiredReceiverBlocks
                                : graphEndpoint.receiverDFBInfo.blockCount;
      if (endpoint.blockCount < *requiredReceiverBlocks) {
        sendOp.emitError(
            "receiver DFB cannot store every destination transfer group");
        return failure();
      }
      endpoint.groupDepth = selectedGroupDepth;
      endpoint.iterationDomain = getIterationDomain(graphEndpoint.postOp);
      if (transportOwnsEndpoint) {
        std::optional<APInt> tripCount =
            storageOwnership->loop.getStaticTripCount();
        assert(tripCount && "transport ownership requires a static loop");
        endpoint.addressSequence = ReceiverAddressSequenceProof{
            tripCount->getZExtValue(),
            ReceiverAddressRecurrence{/*initialSlot=*/0,
                                      endpoint.slotSpanBlocks,
                                      endpoint.blockCount}};
      } else {
        endpoint.addressSequence = graphEndpoint.addressSequence;
      }
      stream.endpoints.push_back(std::move(endpoint));
      stream.completionGroup.endpoints.push_back(endpointId);
    }

    for (CBPopOp releaseOp : capacityPlan.findReleaseOps(transferNode.id)) {
      stream.capacityReleaseIterationDomains.push_back(
          getIterationDomain(releaseOp));
    }

    if (stream.schedule == PipeTransportSchedule::Grouped &&
        stream.synchronizationProtocol ==
            PipeSynchronizationProtocol::Capacity &&
        stream.transferContract == PipeTransferContract::PointToPoint &&
        supportsOverlappedSchedule(stream)) {
      stream.schedule = PipeTransportSchedule::Overlapped;
      stream.creditCompletion = PipeTransportCreditCompletion::IterationDomain;
    }

    if (succeeded(storageOwnership)) {
      PipeTransportStorageSelection sourceStorage =
          selectSourceStorage(std::move(storageOwnership->source),
                              stream.packetization.payloadSizeBytes);
      PipeTransportStorageSelection destinationStorage =
          selectDestinationStorage(std::move(storageOwnership->destination),
                                   stream.endpoints.front(),
                                   stream.packetization.payloadSizeBytes);
      {
        int64_t destinationGroups = destinationStorage.access.blockCount;
        std::optional<int64_t> destinationBytes = llvm::checkedMul(
            destinationGroups, stream.packetization.payloadSizeBytes);
        std::optional<int64_t> scratchOffset =
            alignPipeScratchBytes(plan.sramScratchBytes);
        if (!destinationBytes || !scratchOffset) {
          sendOp.emitError("pipe transport scratch allocation exceeds int64_t");
          return failure();
        }
        std::optional<int64_t> scratchEnd =
            llvm::checkedAdd(*scratchOffset, *destinationBytes);
        std::optional<int64_t> alignedScratchEnd =
            scratchEnd ? alignPipeScratchBytes(*scratchEnd) : std::nullopt;
        if (!alignedScratchEnd) {
          sendOp.emitError("pipe transport scratch allocation exceeds int64_t");
          return failure();
        }

        sourceStorage.access.scratchByteOffset = *scratchOffset;
        destinationStorage.access.scratchByteOffset = *scratchOffset;
        if (destinationStorage.access.blockCount > 1) {
          func::FuncOp receiverFunc = destinationStorage.dfbUse.tensorCopy
                                          ->getParentOfType<func::FuncOp>();
          int64_t counterIndex = nextSlotCounterIndex[receiverFunc]++;
          destinationStorage.access.dynamicSlotCounterIndex = counterIndex;
          plan.slotCounterInitializations[receiverFunc].push_back(
              PipeTransportSlotCounterInitInfo{counterIndex,
                                               /*initialSlot=*/0});
        }
        stream.sourceStorage.ownership =
            PipeTransportStorageOwnership::Transport;
        stream.sourceStorage.blockCount = transferNode.blockSpan;
        stream.sourceStorage.stageDepth = 1;
        stream.sourceStorage.scratchByteOffset = *scratchOffset;
        stream.sourceStorage.scratchBytes =
            stream.packetization.payloadSizeBytes;
        stream.endpoints.front().ownership =
            PipeTransportStorageOwnership::Transport;
        stream.endpoints.front().scratchByteOffset = *scratchOffset;
        stream.endpoints.front().scratchBytes = *destinationBytes;
        plan.sramScratchBytes = *alignedScratchEnd;
        recordStorageSelection(sourceStorage);
        recordStorageSelection(destinationStorage);
      }
    }

    auto [transferIt, transferInserted] =
        plan.streamByTransfer.try_emplace(transferNode.id, stream.id);
    (void)transferIt;
    assert(transferInserted &&
           "PipeGraph transfer belongs to multiple transport streams");

    auto recordStreamOperation = [&](Operation *operation) {
      auto [streamIt, inserted] =
          plan.streamByOperation.try_emplace(operation, stream.id);
      (void)streamIt;
      assert(inserted && "pipe protocol operation belongs to multiple streams");
    };
    recordStreamOperation(transferNode.sendOp);
    for (Operation *postOperation : transferNode.receiverPostOps) {
      recordStreamOperation(postOperation);
    }
    plan.streams.push_back(std::move(stream));
  }

  LLVM_DEBUG(plan.print(llvm::dbgs()));
  return plan;
}

} // namespace mlir::tt::ttl
