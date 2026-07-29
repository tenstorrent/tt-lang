// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeTransportPlan.h"

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

#define DEBUG_TYPE "ttl-pipe-transport-plan"

namespace mlir::tt::ttl {

const PipeTransportStream &
PipeTransportPlan::getStream(PipeTransportStreamId id) const {
  assert(id < streams.size() && "invalid pipe transport stream id");
  return streams[id];
}

const PipeTransportStream &
PipeTransportPlan::getStreamForOperation(Operation *operation) const {
  auto streamIt = streamByOperation.find(operation);
  assert(streamIt != streamByOperation.end() &&
         "pipe protocol operation has no transport stream");
  return getStream(streamIt->second);
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
  }
  llvm_unreachable("unknown pipe transport schedule");
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
     << " group=" << logicalTransfersPerGroup
     << " residual=" << residualTransferCount << "\n";
  os << "PipeTransport:   source blocks=" << sourceStorage.blockCount
     << " block_span=" << sourceStorage.blocksPerTransfer
     << " stage_depth=" << sourceStorage.stageDepth
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

FailureOr<PipeTransportPlan> buildPipeTransportPlan(
    const PipeGraph &pipeGraph,
    function_ref<PipeSynchronizationProtocol(PipeTransferNodeId)>
        selectSynchronizationProtocol) {
  PipeTransportPlan plan;
  for (const PipeTransferNode &transferNode :
       pipeGraph.getPipeTransferNodes()) {
    auto sendOp = cast<PipeTransferSendOp>(transferNode.sendOp);
    auto sourceDFBType = cast<CircularBufferType>(sendOp.getSrc().getType());
    auto tileType = dyn_cast<ttcore::TileType>(sourceDFBType.getElementType());
    if (!tileType) {
      sendOp.emitError("pipe transfer source DFB element type must be tile");
      return failure();
    }
    if (transferNode.blockSpan != 1) {
      sendOp.emitError("grouped pipe transfer lowering is not implemented");
      return failure();
    }

    PipeTransportStream stream;
    stream.id = plan.streams.size();
    stream.transferNode = transferNode.id;
    stream.pipe = transferNode.pipe;
    stream.transferContract = transferNode.transferContract;
    stream.synchronizationProtocol =
        selectSynchronizationProtocol(transferNode.id);
    stream.schedule = PipeTransportSchedule::Scalar;
    stream.logicalTransfersPerGroup = transferNode.blockSpan;
    stream.residualTransferCount = 0;
    stream.sourceIterationDomain = getIterationDomain(sendOp.getOperation());
    stream.sourceStorage = PipeTransportSourceStorage{
        sourceDFBType.getBlockCount(), transferNode.blockSpan, 1};
    std::optional<int64_t> maybePageCount = llvm::checkedMul(
        sourceDFBType.getElementsPerBlock(), transferNode.blockSpan);
    if (!maybePageCount) {
      sendOp.emitError("pipe transfer page count exceeds int64_t");
      return failure();
    }
    int64_t pageCount = *maybePageCount;
    int64_t pageSizeBytes = tileType.getSizeBytes();
    std::optional<int64_t> maybePayloadSizeBytes =
        llvm::checkedMul(pageCount, pageSizeBytes);
    if (!maybePayloadSizeBytes) {
      sendOp.emitError("pipe transfer payload size exceeds int64_t");
      return failure();
    }
    stream.packetization = PipeTransportPacketization{pageCount, pageSizeBytes,
                                                      *maybePayloadSizeBytes};
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
      endpoint.blockCount = graphEndpoint.receiverDFBInfo.blockCount;
      endpoint.groupDepth = 1;
      endpoint.iterationDomain = getIterationDomain(graphEndpoint.postOp);
      endpoint.addressSequence = graphEndpoint.addressSequence;
      stream.endpoints.push_back(std::move(endpoint));
      stream.completionGroup.endpoints.push_back(endpointId);
    }

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
