// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBAllocationLimits.h"
#include "PipeGraph.h"
#include "PipeLowering.h"
#include "PipeTransferExpansion.h"
#include "PipeTransportDFBAnalysis.h"
#include "ttlang/Analysis/LoopIterationUtils.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/PipeConstants.h"
#include "ttlang/Dialect/TTL/Transforms/PipeTransferAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#define DEBUG_TYPE "ttl-form-pipe-transports"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFORMPIPETRANSPORTS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct PipeTransportLoopCandidate {
  scf::ForOp loop;
  int64_t lowerBound = 0;
  int64_t transferCount = 0;
  SmallVector<const PipeTransferNode *, 1> transfers;
  SmallVector<PipeTransferCreateOp, 1> transferCreates;
  SmallVector<PipeTransportDFBUse, 0> dfbUses;
};

struct PipeTransportGrouping {
  int64_t groupSize = 1;
  int64_t destinationDepth = 1;
  int64_t fullGroupCount = 0;
  int64_t residualCount = 0;
  bool overlapsReceiver = false;
  uint64_t allocationBytes = 0;
  llvm::DenseMap<Value, int64_t> blockCounts;

  int64_t getCompletionGroupCount() const {
    return fullGroupCount + residualCount;
  }

  int64_t getCapacityWaitCount() const {
    return std::max<int64_t>(0, fullGroupCount - destinationDepth);
  }
};

/// Return the physical destination depth required for one group size.
///
/// Two destination groups let the sender fill one group while the receiver
/// drains the other. Existing receiver storage establishes a larger minimum
/// because grouping must preserve every declared DFB block.
static int64_t
getMinimumDestinationDepth(const PipeTransportLoopCandidate &candidate,
                           int64_t groupSize, bool requireOverlap) {
  int64_t minimumDepth = requireOverlap ? 2 : 1;
  for (const PipeTransportDFBUse &dfbUse : candidate.dfbUses) {
    if (dfbUse.role != PipeTransportDFBRole::Destination) {
      continue;
    }
    auto dfbType = cast<CircularBufferType>(dfbUse.dfb.getType());
    int64_t existingDepth =
        dfbType.getBlockCount() / groupSize +
        static_cast<int64_t>(dfbType.getBlockCount() % groupSize != 0);
    minimumDepth = std::max(minimumDepth, existingDepth);
  }
  return minimumDepth;
}

/// Return the closest enclosing `scf.for`, if one exists.
static scf::ForOp getEnclosingFor(Operation *operation) {
  return operation->getParentOfType<scf::ForOp>();
}

/// Return whether `operation` is nested in `loop`.
static bool isInsideLoop(scf::ForOp loop, Operation *operation) {
  return loop->isProperAncestor(operation);
}

/// Emit a debug-only explanation for retaining scalar transfer IR.
static void debugReject(scf::ForOp loop, StringRef reason) {
  LLVM_DEBUG(llvm::dbgs() << "PipeTransportFormation: reject " << loop.getLoc()
                          << ": " << reason << "\n");
}

/// Hoist pure loop-invariant setup so grouped transfers execute once per group
/// rather than recreating static pipe values for every logical transfer.
static void hoistPipeLoopInvariantCode(ModuleOp module) {
  llvm::DenseSet<Operation *> seenLoops;
  SmallVector<scf::ForOp> loops;
  module.walk([&](PipeTransferSendOp send) {
    scf::ForOp loop = getEnclosingFor(send);
    if (loop && seenLoops.insert(loop).second) {
      loops.push_back(loop);
    }
  });
  for (scf::ForOp loop : loops) {
    moveLoopInvariantCode(cast<LoopLikeOpInterface>(loop.getOperation()));
  }
}

/// Return the tensor type for `blockSpan` consecutive DFB blocks.
static FailureOr<RankedTensorType>
getGroupedTensorType(RankedTensorType blockType, int64_t blockSpan) {
  if (blockSpan <= 0 || blockType.getRank() <= 0) {
    return failure();
  }
  SmallVector<int64_t> groupedShape(blockType.getShape());
  std::optional<int64_t> groupedInnermost =
      llvm::checkedMul(groupedShape.back(), blockSpan);
  if (!groupedInnermost) {
    return failure();
  }
  groupedShape.back() = *groupedInnermost;
  return RankedTensorType::get(groupedShape, blockType.getElementType(),
                               blockType.getEncoding());
}

/// Return or create the DFB record for one transfer role.
static FailureOr<PipeTransportDFBUse *>
getOrAddDFBUse(PipeTransportLoopCandidate &candidate, Value dfb,
               PipeTransportDFBRole role, PipeTransferNodeId transferNode,
               std::string &reason) {
  for (PipeTransportDFBUse &existing : candidate.dfbUses) {
    if (existing.dfb != dfb) {
      continue;
    }
    if (existing.role != role || existing.transferNode != transferNode) {
      reason = "one DFB participates in multiple transport roles";
      return failure();
    }
    return &existing;
  }

  auto bind = dfb.getDefiningOp<BindCBOp>();
  if (!bind) {
    reason = "transport DFB is not defined by ttl.bind_cb";
    return failure();
  }
  PipeTransportDFBUse dfbUse;
  dfbUse.dfb = dfb;
  dfbUse.bind = bind;
  dfbUse.role = role;
  dfbUse.transferNode = transferNode;
  candidate.dfbUses.push_back(std::move(dfbUse));
  return &candidate.dfbUses.back();
}

/// Validate that changing the loop step does not remove unrelated effects.
static LogicalResult validateLoopEffects(PipeTransportLoopCandidate &candidate,
                                         const PipeTransferIndex &transferIndex,
                                         ValueOriginAnalysis &analysis,
                                         std::string &reason) {
  llvm::DenseSet<Operation *> asyncOperations;
  llvm::DenseSet<Operation *> allowedOperations;
  for (PipeTransportDFBUse &dfbUse : candidate.dfbUses) {
    llvm::for_each(dfbUse.reserves, [&](CBReserveOp reserve) {
      allowedOperations.insert(reserve);
    });
    llvm::for_each(dfbUse.pushes,
                   [&](CBPushOp push) { allowedOperations.insert(push); });
    llvm::for_each(dfbUse.waits,
                   [&](CBWaitOp wait) { allowedOperations.insert(wait); });
    llvm::for_each(dfbUse.pops,
                   [&](CBPopOp pop) { allowedOperations.insert(pop); });
    llvm::for_each(dfbUse.attaches, [&](AttachCBOp attach) {
      allowedOperations.insert(attach);
    });
    allowedOperations.insert(dfbUse.tensorCopy);
    asyncOperations.insert(dfbUse.tensorCopy);
  }
  for (const PipeTransferNode *transfer : candidate.transfers) {
    allowedOperations.insert(transfer->sendOp);
    asyncOperations.insert(transfer->sendOp);
    for (Operation *post : transfer->receiverPostOps) {
      allowedOperations.insert(post);
    }
  }

  WalkResult walkResult = candidate.loop.walk([&](Operation *operation) {
    if (operation == candidate.loop.getOperation() ||
        operation->hasTrait<OpTrait::IsTerminator>() ||
        isMemoryEffectFree(operation)) {
      return WalkResult::advance();
    }
    if (allowedOperations.contains(operation)) {
      return WalkResult::advance();
    }
    if (isa<IfSrcOp, IfDstOp>(operation)) {
      return WalkResult::advance();
    }
    if (auto wait = dyn_cast<WaitOp>(operation)) {
      const OriginSet &origins = analysis.getOrigins(wait.getXf());
      if (!origins.empty() && origins.allMatch([&](Value origin) {
            return asyncOperations.contains(origin.getDefiningOp());
          })) {
        return WalkResult::advance();
      }
    }
    if (auto pipeWait = dyn_cast<PipeTransferWaitOp>(operation)) {
      ArrayRef<Operation *> posts =
          transferIndex.getPossibleReceivePosts(pipeWait);
      if (llvm::all_of(posts, [&](Operation *post) {
            return allowedOperations.contains(post);
          })) {
        return WalkResult::advance();
      }
    }
    if (isa<scf::ForOp>(operation)) {
      reason = "candidate loop contains a nested loop";
    } else {
      reason = ("candidate loop contains unrelated side effect " +
                operation->getName().getStringRef())
                   .str();
    }
    return WalkResult::interrupt();
  });
  return walkResult.wasInterrupted() ? failure() : success();
}

/// Build one complete loop candidate from PipeGraph transfer nodes.
static FailureOr<PipeTransportLoopCandidate> buildLoopCandidate(
    scf::ForOp loop, ArrayRef<const PipeTransferNode *> transfers,
    const PipeTransferIndex &transferIndex, const PipeGraph &pipeGraph,
    ValueOriginAnalysis &analysis, std::string &reason) {
  PipeTransportLoopCandidate candidate;
  candidate.loop = loop;
  candidate.transfers.append(transfers.begin(), transfers.end());

  if (!loop.getInitArgs().empty() || loop.getNumResults() != 0) {
    reason = "candidate loop has loop-carried values";
    return failure();
  }
  std::optional<int64_t> lowerBound =
      evaluateIndexExpression(loop.getLowerBound());
  std::optional<int64_t> upperBound =
      evaluateIndexExpression(loop.getUpperBound());
  std::optional<int64_t> step = evaluateIndexExpression(loop.getStep());
  if (!lowerBound || !upperBound || !step || *step != 1 ||
      *upperBound <= *lowerBound) {
    reason = "candidate loop requires constant bounds and unit step";
    return failure();
  }
  std::optional<int64_t> transferCount =
      llvm::checkedSub(*upperBound, *lowerBound);
  if (!transferCount) {
    reason = "candidate loop transfer count exceeds int64_t";
    return failure();
  }
  candidate.lowerBound = *lowerBound;
  candidate.transferCount = *transferCount;

  llvm::DenseSet<Operation *> seenCreates;
  for (const PipeTransferNode *transfer : transfers) {
    if (transfer->blockSpan != 1 || getEnclosingFor(transfer->sendOp) != loop) {
      reason =
          "transfer is not scalar or does not execute in the candidate loop";
      return failure();
    }
    if (transfer->transferContract != PipeTransferContract::PointToPoint ||
        transfer->receiverEndpoints.size() != 1) {
      reason = "grouped scratch storage requires one point-to-point receiver";
      return failure();
    }
    const PipeReceiverEndpoint &receiverEndpoint =
        pipeGraph.getPipeReceiverEndpoint(transfer->receiverEndpoints.front());
    if (transfer->pipe.srcX == receiverEndpoint.receiver.x &&
        transfer->pipe.srcY == receiverEndpoint.receiver.y) {
      reason = "grouped scratch storage cannot alias source and destination";
      return failure();
    }
    auto send = cast<PipeTransferSendOp>(transfer->sendOp);
    FailureOr<PipeTransportDFBUse *> source =
        getOrAddDFBUse(candidate, send.getSrc(), PipeTransportDFBRole::Source,
                       transfer->id, reason);
    if (failed(source)) {
      return failure();
    }
    PipeTransferCreateOp sendCreate =
        transferIndex.getTransferCreate(send.getOperation());
    if (isInsideLoop(loop, sendCreate.getOperation())) {
      reason = "transfer creation must be outside the candidate loop";
      return failure();
    }
    if (seenCreates.insert(sendCreate.getOperation()).second) {
      candidate.transferCreates.push_back(sendCreate);
    }

    for (Operation *postOperation : transfer->receiverPostOps) {
      if (getEnclosingFor(postOperation) != loop) {
        reason = "sender and receiver do not execute in the same loop";
        return failure();
      }
      auto post = cast<PipeTransferPostOp>(postOperation);
      Value receiverDFB = getAttachedCB(post.getDst());
      if (!receiverDFB) {
        reason = "receiver post has no attached DFB";
        return failure();
      }
      if (failed(getOrAddDFBUse(candidate, receiverDFB,
                                PipeTransportDFBRole::Destination, transfer->id,
                                reason))) {
        return failure();
      }
      PipeTransferCreateOp postCreate =
          transferIndex.getTransferCreate(post.getOperation());
      if (isInsideLoop(loop, postCreate.getOperation())) {
        reason = "transfer creation must be outside the candidate loop";
        return failure();
      }
      if (seenCreates.insert(postCreate.getOperation()).second) {
        candidate.transferCreates.push_back(postCreate);
      }
    }
  }

  for (PipeTransportDFBUse &dfbUse : candidate.dfbUses) {
    FailureOr<PipeTransportDFBUse> maybeDFBUse =
        analyzePipeTransportDFBUse(candidate.loop, dfbUse.dfb, dfbUse.role,
                                   dfbUse.transferNode, pipeGraph, reason);
    if (failed(maybeDFBUse)) {
      return failure();
    }
    dfbUse = std::move(*maybeDFBUse);
    if (!hasOnlyPipeTransportLoopUses(candidate.loop, dfbUse.dfb)) {
      reason = "transport DFB has a use outside the candidate loop";
      return failure();
    }
    if (!hasPrivatePipeTransportDFBViews(dfbUse, pipeGraph)) {
      reason = "transport DFB acquired views escape their transport role";
      return failure();
    }
  }
  if (failed(validateLoopEffects(candidate, transferIndex, analysis, reason))) {
    return failure();
  }
  return std::move(candidate);
}

/// Compute scratch storage allocated for one grouped transport choice.
static std::optional<uint64_t>
getTransportScratchBytes(const PipeTransportLoopCandidate &candidate,
                         int64_t groupSize, int64_t destinationDepth) {
  uint64_t totalBytes = 0;
  for (const PipeTransferNode *transfer : candidate.transfers) {
    auto sendOp = cast<PipeTransferSendOp>(transfer->sendOp);
    auto sourceType = cast<CircularBufferType>(sendOp.getSrc().getType());
    std::string failureReason;
    FailureOr<uint64_t> sourceBlockBytes = getDFBAllocationSizeBytes(
        CircularBufferType::get(sourceType.getContext(), sourceType.getShape(),
                                sourceType.getElementType(), 1),
        failureReason);
    if (failed(sourceBlockBytes)) {
      return std::nullopt;
    }
    std::optional<uint64_t> payloadBytes = llvm::checkedMulUnsigned(
        *sourceBlockBytes, static_cast<uint64_t>(groupSize));
    std::optional<uint64_t> destinationBytes =
        payloadBytes
            ? llvm::checkedMulUnsigned(*payloadBytes,
                                       static_cast<uint64_t>(destinationDepth))
            : std::nullopt;
    if (!destinationBytes ||
        *destinationBytes > std::numeric_limits<uint64_t>::max() -
                                (kPipeSramScratchAlignmentBytes - 1)) {
      return std::nullopt;
    }
    uint64_t alignedBytes =
        llvm::alignTo(*destinationBytes,
                      static_cast<uint64_t>(kPipeSramScratchAlignmentBytes));
    std::optional<uint64_t> nextTotal =
        llvm::checkedAddUnsigned(totalBytes, alignedBytes);
    if (!nextTotal) {
      return std::nullopt;
    }
    totalBytes = *nextTotal;
  }
  return totalBytes;
}

struct ConservativePipeResources {
  uint64_t scratchBytes = 0;
  int64_t globalSemaphoreCount = 0;
};

template <typename ForeachOp>
static LogicalResult
addCallbackResourceUpperBound(ForeachOp foreachOp,
                              ConservativePipeResources &resources) {
  uint64_t endpointCount = 0;
  for (PipeRecordAttr record : foreachOp.getRecords().getPipes()) {
    std::optional<int64_t> width = llvm::checkedAdd(
        llvm::checkedSub(record.getDstEndX(), record.getDstStartX())
            .value_or(-1),
        int64_t{1});
    std::optional<int64_t> height = llvm::checkedAdd(
        llvm::checkedSub(record.getDstEndY(), record.getDstStartY())
            .value_or(-1),
        int64_t{1});
    if (!width || !height || *width <= 0 || *height <= 0) {
      return failure();
    }
    std::optional<uint64_t> recordEndpoints = llvm::checkedMulUnsigned(
        static_cast<uint64_t>(*width), static_cast<uint64_t>(*height));
    std::optional<uint64_t> updatedEndpoints =
        recordEndpoints
            ? llvm::checkedAddUnsigned(endpointCount, *recordEndpoints)
            : std::nullopt;
    if (!updatedEndpoints) {
      return failure();
    }
    endpointCount = *updatedEndpoints;
  }

  uint64_t protocolOperationCount = 0;
  WalkResult walkResult = foreachOp->walk([&](Operation *operation) {
    if (!isa<CopyOp, PipeTransferSendOp, PipeTransferPostOp>(operation)) {
      return WalkResult::advance();
    }
    if (protocolOperationCount == std::numeric_limits<uint64_t>::max()) {
      return WalkResult::interrupt();
    }
    ++protocolOperationCount;
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }
  std::optional<uint64_t> resourceUnits =
      llvm::checkedMulUnsigned(endpointCount, protocolOperationCount);
  std::optional<uint64_t> scratchBytes =
      resourceUnits ? llvm::checkedMulUnsigned(
                          *resourceUnits,
                          static_cast<uint64_t>(kPipeSramScratchAlignmentBytes))
                    : std::nullopt;
  std::optional<uint64_t> updatedScratch =
      scratchBytes
          ? llvm::checkedAddUnsigned(resources.scratchBytes, *scratchBytes)
          : std::nullopt;
  // Capacity, completion, and ready counters are independent upper bounds;
  // the exact PipeNet plan may share them.
  std::optional<uint64_t> semaphoreUnits =
      resourceUnits ? llvm::checkedMulUnsigned(*resourceUnits, uint64_t{3})
                    : std::nullopt;
  std::optional<int64_t> semaphoreCount =
      semaphoreUnits &&
              *semaphoreUnits <=
                  static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
          ? std::optional<int64_t>(static_cast<int64_t>(*semaphoreUnits))
          : std::nullopt;
  std::optional<int64_t> updatedSemaphores =
      semaphoreCount
          ? llvm::checkedAdd(resources.globalSemaphoreCount, *semaphoreCount)
          : std::nullopt;
  if (!updatedScratch || !updatedSemaphores) {
    return failure();
  }
  resources.scratchBytes = *updatedScratch;
  resources.globalSemaphoreCount = *updatedSemaphores;
  return success();
}

static FailureOr<ConservativePipeResources>
getConservativePipeResources(ModuleOp sourceModule) {
  OwningOpRef<ModuleOp> planningModule(sourceModule.clone());
  ModuleOp module = *planningModule;
  ValueOriginAnalysis preExpansionAnalysis(module);
  if (failed(verifyTransferProvenance(module, preExpansionAnalysis)) ||
      failed(expandStaticPipeTransfers(module, preExpansionAnalysis))) {
    return failure();
  }

  ValueOriginAnalysis analysis(module);
  if (failed(verifyTransferProvenance(module, analysis))) {
    return failure();
  }
  FailureOr<std::unique_ptr<PipeTransferIndex>> maybeTransferIndex =
      PipeTransferIndex::create(module, analysis);
  if (failed(maybeTransferIndex)) {
    return failure();
  }
  const PipeTransferIndex &transferIndex = **maybeTransferIndex;
  PipeForeachLoweringInfo foreachLoweringInfo;
  FailureOr<PipeGraph> maybePipeGraph =
      PipeGraph::build(module, transferIndex, foreachLoweringInfo);
  if (failed(maybePipeGraph)) {
    return failure();
  }

  PipeResourcePlan resourcePlan;
  if (failed(buildPipeResourcePlan(module, transferIndex, *maybePipeGraph,
                                   resourcePlan,
                                   /*enableComputedAddresses=*/false,
                                   PipeCounterAllocationPolicy::GlobalOnly,
                                   /*synchronizationSelection=*/nullptr))) {
    return failure();
  }
  assert(resourcePlan.sramScratch.bytes >= 0 &&
         "pipe scratch allocation must be non-negative");
  PipeResourceRequirements requirements =
      getPipeResourceRequirements(resourcePlan);
  if (maybePipeGraph->getPipeReceiverEndpoints().size() >
      static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return failure();
  }
  std::optional<int64_t> globalSemaphoreCount = llvm::checkedAdd(
      requirements.globalSemaphoreCount,
      static_cast<int64_t>(maybePipeGraph->getPipeReceiverEndpoints().size()));
  if (!globalSemaphoreCount) {
    return failure();
  }
  ConservativePipeResources resources{
      static_cast<uint64_t>(resourcePlan.sramScratch.bytes),
      *globalSemaphoreCount};
  WalkResult callbackWalk =
      module.walk([&](Operation *operation) -> WalkResult {
        if (auto foreachSrc = dyn_cast<PipeNetForeachSrcOp>(operation)) {
          return failed(addCallbackResourceUpperBound(foreachSrc, resources))
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        }
        if (auto foreachDst = dyn_cast<PipeNetForeachDstOp>(operation)) {
          return failed(addCallbackResourceUpperBound(foreachDst, resources))
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        }
        return WalkResult::advance();
      });
  if (callbackWalk.wasInterrupted()) {
    return failure();
  }
  return resources;
}

static FailureOr<uint64_t>
getResidualGlobalSemaphoreBytes(ModuleOp module,
                                const PipeTransportLoopCandidate &candidate,
                                const PipeTransportGrouping &grouping) {
  if (grouping.residualCount == 0) {
    return 0;
  }
  if (candidate.transfers.size() >
      static_cast<size_t>(std::numeric_limits<int64_t>::max() / 2)) {
    return failure();
  }
  int64_t additionalSemaphoreCount =
      static_cast<int64_t>(candidate.transfers.size()) * 2;
  return getGlobalSemaphoreL1Bytes(module, additionalSemaphoreCount);
}

/// Compute storage and synchronization facts for one `(R, K)` choice.
static std::optional<PipeTransportGrouping>
evaluateGrouping(ModuleOp module, PipeTransportLoopCandidate &candidate,
                 int64_t groupSize, int64_t destinationDepth,
                 const DFBAllocationFootprint &allocationFootprint,
                 const DFBLogicalIdentityAnalysis &identities,
                 uint64_t existingScratchBytes, uint64_t globalSemaphoreBytes,
                 uint64_t resetStateBytes, uint64_t budgetBytes) {
  if (groupSize <= 1 || groupSize > candidate.transferCount) {
    return std::nullopt;
  }
  int64_t fullGroupCount = candidate.transferCount / groupSize;
  if (fullGroupCount <= 0 || destinationDepth <= 0) {
    return std::nullopt;
  }

  PipeTransportGrouping grouping;
  grouping.groupSize = groupSize;
  grouping.destinationDepth = destinationDepth;
  grouping.fullGroupCount = fullGroupCount;
  grouping.residualCount = candidate.transferCount % groupSize;
  grouping.overlapsReceiver = fullGroupCount >= 2 && destinationDepth >= 2;

  llvm::DenseMap<int64_t, uint64_t> allocationBytesByLogicalId;
  for (PipeTransportDFBUse &dfbUse : candidate.dfbUses) {
    if (!llvm::checkedMul(cast<CircularBufferType>(dfbUse.dfb.getType())
                              .getElementsPerBlock(),
                          groupSize) ||
        failed(getGroupedTensorType(
            cast<RankedTensorType>(
                dfbUse.reserves.front().getResult().getType()),
            groupSize)) ||
        failed(getGroupedTensorType(
            cast<RankedTensorType>(dfbUse.waits.front().getResult().getType()),
            groupSize)) ||
        failed(getGroupedTensorType(
            cast<RankedTensorType>(dfbUse.tensorSlice.getResult().getType()),
            groupSize))) {
      return std::nullopt;
    }

    int64_t depth =
        dfbUse.role == PipeTransportDFBRole::Source ? 1 : destinationDepth;
    auto oldType = cast<CircularBufferType>(dfbUse.dfb.getType());
    int64_t oldGroupDepth =
        oldType.getBlockCount() / groupSize +
        static_cast<int64_t>(oldType.getBlockCount() % groupSize != 0);
    std::optional<int64_t> alignedBlockCount =
        llvm::checkedMul(groupSize, std::max(depth, oldGroupDepth));
    if (!alignedBlockCount) {
      return std::nullopt;
    }
    int64_t blockCount = *alignedBlockCount;
    if (dfbUse.bind.getTensorBackingAttr() &&
        blockCount != oldType.getBlockCount()) {
      return std::nullopt;
    }
    auto resizedType =
        CircularBufferType::get(oldType.getContext(), oldType.getShape(),
                                oldType.getElementType(), blockCount);
    std::string failureReason;
    FailureOr<uint64_t> allocationBytes =
        getDFBL1AllocationSizeBytes(module, resizedType, failureReason);
    if (failed(allocationBytes)) {
      return std::nullopt;
    }
    if (!dfbUse.bind.getTensorBackingAttr()) {
      int64_t logicalId = identities.getLogicalId(dfbUse.bind);
      uint64_t &minimumBytes = allocationBytesByLogicalId[logicalId];
      minimumBytes = std::max(minimumBytes, *allocationBytes);
    }
    grouping.blockCounts[dfbUse.dfb] = blockCount;
  }

  FailureOr<uint64_t> totalBytes =
      allocationFootprint.getTotalBytesWithMinimumAllocations(
          allocationBytesByLogicalId);
  std::optional<uint64_t> candidateScratchBytes =
      getTransportScratchBytes(candidate, groupSize, destinationDepth);
  std::optional<uint64_t> allScratchBytes =
      candidateScratchBytes ? llvm::checkedAddUnsigned(existingScratchBytes,
                                                       *candidateScratchBytes)
                            : std::nullopt;
  if (allScratchBytes) {
    allScratchBytes =
        llvm::checkedAddUnsigned(*allScratchBytes, resetStateBytes);
  }
  FailureOr<uint64_t> scratchAllocationBytes = failure();
  if (allScratchBytes) {
    scratchAllocationBytes = getL1AllocationSizeBytes(module, *allScratchBytes);
  }
  std::optional<uint64_t> allocationAndScratchBytes =
      succeeded(totalBytes) && succeeded(scratchAllocationBytes)
          ? llvm::checkedAddUnsigned(*totalBytes, *scratchAllocationBytes)
          : std::nullopt;
  FailureOr<uint64_t> residualSemaphoreBytes =
      getResidualGlobalSemaphoreBytes(module, candidate, grouping);
  std::optional<uint64_t> allSemaphoreBytes =
      succeeded(residualSemaphoreBytes)
          ? llvm::checkedAddUnsigned(globalSemaphoreBytes,
                                     *residualSemaphoreBytes)
          : std::nullopt;
  std::optional<uint64_t> requiredBytes =
      allocationAndScratchBytes && allSemaphoreBytes
          ? llvm::checkedAddUnsigned(*allocationAndScratchBytes,
                                     *allSemaphoreBytes)
          : std::nullopt;
  if (!requiredBytes || *requiredBytes > budgetBytes) {
    return std::nullopt;
  }
  grouping.allocationBytes = *requiredBytes;
  return grouping;
}

/// Bound exhaustive group-size checks by the storage required for one group.
static std::optional<int64_t> getGroupSizeUpperBound(
    ModuleOp module, PipeTransportLoopCandidate &candidate, int64_t upperBound,
    const DFBLogicalIdentityAnalysis &identities, uint64_t budgetBytes) {
  llvm::DenseMap<int64_t, uint64_t> bytesPerBlockByLogicalId;
  std::optional<int64_t> tensorBackedCapacityUpperBound;
  for (PipeTransportDFBUse &dfbUse : candidate.dfbUses) {
    auto oldType = cast<CircularBufferType>(dfbUse.dfb.getType());
    auto oneBlockType = CircularBufferType::get(
        oldType.getContext(), oldType.getShape(), oldType.getElementType(), 1);
    std::string failureReason;
    FailureOr<uint64_t> bytesPerBlock =
        getDFBL1AllocationSizeBytes(module, oneBlockType, failureReason);
    if (failed(bytesPerBlock) || *bytesPerBlock == 0) {
      return std::nullopt;
    }
    if (dfbUse.bind.getTensorBackingAttr()) {
      tensorBackedCapacityUpperBound = std::min(
          tensorBackedCapacityUpperBound.value_or(oldType.getBlockCount()),
          oldType.getBlockCount());
      continue;
    }
    int64_t logicalId = identities.getLogicalId(dfbUse.bind);
    uint64_t &minimumBytes = bytesPerBlockByLogicalId[logicalId];
    minimumBytes = std::max(minimumBytes, *bytesPerBlock);
  }

  uint64_t bytesPerGroup = 0;
  for (const auto &entry : bytesPerBlockByLogicalId) {
    std::optional<uint64_t> total =
        llvm::checkedAddUnsigned(bytesPerGroup, entry.second);
    if (!total) {
      return std::nullopt;
    }
    bytesPerGroup = *total;
  }
  uint64_t requestedUpperBound =
      static_cast<uint64_t>(std::min(upperBound, candidate.transferCount));
  if (tensorBackedCapacityUpperBound) {
    requestedUpperBound =
        std::min(requestedUpperBound,
                 static_cast<uint64_t>(*tensorBackedCapacityUpperBound));
  }
  uint64_t budgetUpperBound =
      bytesPerGroup == 0 ? requestedUpperBound : budgetBytes / bytesPerGroup;
  return static_cast<int64_t>(std::min(requestedUpperBound, budgetUpperBound));
}

/// Return whether `candidate` improves on `selected`.
static bool isBetterGrouping(const PipeTransportGrouping &candidate,
                             const PipeTransportGrouping &selected) {
  if (candidate.overlapsReceiver != selected.overlapsReceiver) {
    return candidate.overlapsReceiver;
  }
  if (candidate.getCompletionGroupCount() !=
      selected.getCompletionGroupCount()) {
    return candidate.getCompletionGroupCount() <
           selected.getCompletionGroupCount();
  }
  if (candidate.getCapacityWaitCount() != selected.getCapacityWaitCount()) {
    return candidate.getCapacityWaitCount() < selected.getCapacityWaitCount();
  }
  if (candidate.allocationBytes != selected.allocationBytes) {
    return candidate.allocationBytes < selected.allocationBytes;
  }
  return candidate.groupSize > selected.groupSize;
}

/// Select an explicit upper-bound or automatic group size.
static std::optional<PipeTransportGrouping>
selectGrouping(ModuleOp module, PipeTransportLoopCandidate &candidate,
               int64_t requestedGroupSize,
               const DFBAllocationFootprint &allocationFootprint,
               const DFBLogicalIdentityAnalysis &identities,
               uint64_t existingScratchBytes, uint64_t globalSemaphoreBytes,
               uint64_t resetStateBytes, uint64_t budgetBytes) {
  int64_t upperBound =
      requestedGroupSize > 1 ? requestedGroupSize : candidate.transferCount;
  std::optional<int64_t> maybeUpperBound = getGroupSizeUpperBound(
      module, candidate, upperBound, identities, budgetBytes);
  if (!maybeUpperBound || *maybeUpperBound <= 1) {
    return std::nullopt;
  }
  upperBound = *maybeUpperBound;

  std::optional<PipeTransportGrouping> selected;
  for (int64_t groupSize = 2; groupSize <= upperBound; ++groupSize) {
    bool requireOverlap = candidate.transferCount / groupSize >= 2;
    int64_t destinationDepth =
        getMinimumDestinationDepth(candidate, groupSize, requireOverlap);
    std::optional<PipeTransportGrouping> grouping =
        evaluateGrouping(module, candidate, groupSize, destinationDepth,
                         allocationFootprint, identities, existingScratchBytes,
                         globalSemaphoreBytes, resetStateBytes, budgetBytes);
    if (grouping && (!selected || isBetterGrouping(*grouping, *selected))) {
      selected = std::move(grouping);
    }
  }
  return selected;
}

/// Clone the scalar residual before mutating the grouped loop body.
static void createResidualLoop(scf::ForOp loop, Value residualLowerBound,
                               IRRewriter &rewriter) {
  rewriter.setInsertionPointAfter(loop);
  auto residualLoop =
      scf::ForOp::create(rewriter, loop.getLoc(), residualLowerBound,
                         loop.getUpperBound(), loop.getStep());
  IRMapping mapping;
  mapping.map(loop.getInductionVar(), residualLoop.getInductionVar());
  rewriter.setInsertionPoint(residualLoop.getBody()->getTerminator());
  for (Operation &operation : loop.getBody()->without_terminator()) {
    rewriter.clone(operation, mapping);
  }
}

/// Give the grouped loop transfer values that do not affect scalar residuals.
static void createGroupedTransfers(PipeTransportLoopCandidate &candidate,
                                   int64_t groupSize,
                                   int64_t destinationGroupDepth,
                                   IRRewriter &rewriter) {
  llvm::DenseMap<Value, Value> groupedTransferByScalarTransfer;
  rewriter.setInsertionPoint(candidate.loop);
  for (PipeTransferCreateOp create : candidate.transferCreates) {
    auto groupedCreate =
        cast<PipeTransferCreateOp>(rewriter.clone(*create.getOperation()));
    groupedCreate.setBlockSpan(groupSize);
    groupedCreate.setDestinationGroupDepth(destinationGroupDepth);
    groupedTransferByScalarTransfer[create.getTransfer()] =
        groupedCreate.getTransfer();
  }

  candidate.loop.walk([&](Operation *operation) {
    for (OpOperand &operand : operation->getOpOperands()) {
      auto groupedTransfer =
          groupedTransferByScalarTransfer.find(operand.get());
      if (groupedTransfer == groupedTransferByScalarTransfer.end()) {
        continue;
      }
      rewriter.modifyOpInPlace(operation,
                               [&] { operand.set(groupedTransfer->second); });
    }
  });
}

/// Resize one DFB declaration without changing its block shape.
static void resizeDFB(PipeTransportDFBUse &dfbUse, int64_t blockCount,
                      IRRewriter &rewriter) {
  auto oldType = cast<CircularBufferType>(dfbUse.dfb.getType());
  auto newType =
      CircularBufferType::get(oldType.getContext(), oldType.getShape(),
                              oldType.getElementType(), blockCount);
  rewriter.modifyOpInPlace(dfbUse.bind, [&] {
    dfbUse.bind.setBlockCount(blockCount);
    dfbUse.bind.getResult().setType(newType);
  });
}

/// Widen one scalar DFB lifecycle to a complete transfer group.
static void groupDFBLifecycle(PipeTransportDFBUse &dfbUse, int64_t groupSize,
                              IRRewriter &rewriter) {
  auto dfbType = cast<CircularBufferType>(dfbUse.dfb.getType());
  std::optional<int64_t> pageCount =
      llvm::checkedMul(dfbType.getElementsPerBlock(), groupSize);
  assert(pageCount && "selected grouping has an invalid page count");
  IntegerAttr pageCountAttr = rewriter.getI64IntegerAttr(*pageCount);

  auto groupAcquire = [&](auto acquire) {
    auto resultType = cast<RankedTensorType>(acquire.getResult().getType());
    FailureOr<RankedTensorType> groupedType =
        getGroupedTensorType(resultType, groupSize);
    assert(succeeded(groupedType) &&
           "selected grouping has an invalid acquire type");
    rewriter.modifyOpInPlace(acquire, [&] {
      acquire.setNumTilesAttr(pageCountAttr);
      acquire.getResult().setType(*groupedType);
    });
  };
  for (CBReserveOp reserve : dfbUse.reserves) {
    groupAcquire(reserve);
  }
  for (CBWaitOp wait : dfbUse.waits) {
    groupAcquire(wait);
  }

  auto groupRelease = [&](auto release) {
    rewriter.modifyOpInPlace(release,
                             [&] { release.setNumTilesAttr(pageCountAttr); });
  };
  for (CBPushOp push : dfbUse.pushes) {
    groupRelease(push);
  }
  for (CBPopOp pop : dfbUse.pops) {
    groupRelease(pop);
  }
  for (AttachCBOp attach : dfbUse.attaches) {
    rewriter.modifyOpInPlace(attach, [&] {
      attach.getResult().setType(attach.getTensor().getType());
    });
  }

  auto sliceType =
      cast<RankedTensorType>(dfbUse.tensorSlice.getResult().getType());
  FailureOr<RankedTensorType> groupedSliceType =
      getGroupedTensorType(sliceType, groupSize);
  assert(succeeded(groupedSliceType) &&
         "selected grouping has an invalid tensor slice type");
  rewriter.modifyOpInPlace(dfbUse.tensorSlice, [&] {
    dfbUse.tensorSlice.getResult().setType(*groupedSliceType);
  });
}

/// Materialize one selected grouping and its scalar residual.
static void applyGrouping(PipeTransportLoopCandidate &candidate,
                          const PipeTransportGrouping &grouping,
                          IRRewriter &rewriter) {
  Location location = candidate.loop.getLoc();
  rewriter.setInsertionPoint(candidate.loop);
  int64_t groupedUpperBound =
      candidate.lowerBound + grouping.fullGroupCount * grouping.groupSize;
  Value groupEnd =
      arith::ConstantIndexOp::create(rewriter, location, groupedUpperBound);
  Value groupStep =
      arith::ConstantIndexOp::create(rewriter, location, grouping.groupSize);

  if (grouping.residualCount > 0) {
    createResidualLoop(candidate.loop, groupEnd, rewriter);
  }
  createGroupedTransfers(candidate, grouping.groupSize,
                         grouping.destinationDepth, rewriter);

  rewriter.modifyOpInPlace(candidate.loop, [&] {
    candidate.loop.getUpperBoundMutable().assign(groupEnd);
    candidate.loop.getStepMutable().assign(groupStep);
  });

  for (PipeTransportDFBUse &dfbUse : candidate.dfbUses) {
    int64_t blockCount = grouping.blockCounts.lookup(dfbUse.dfb);
    assert(blockCount > 0 && "selected grouping has no DFB allocation");
    resizeDFB(dfbUse, blockCount, rewriter);
    groupDFBLifecycle(dfbUse, grouping.groupSize, rewriter);
  }
}

struct TTLFormPipeTransportsPass
    : public impl::TTLFormPipeTransportsBase<TTLFormPipeTransportsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (groupSize < 0) {
      module.emitOpError("pipe transport group size must be non-negative");
      signalPassFailure();
      return;
    }
    FailureOr<uint64_t> resetStateBytes =
        getSynchronizedDFBResetStateBytes(module);
    if (failed(resetStateBytes)) {
      module.emitOpError("DFB reset state size is not representable");
      signalPassFailure();
      return;
    }
    auto setConservativePipeBytes = [&](uint64_t scratchBytes,
                                        uint64_t semaphoreBytes) {
      FailureOr<uint64_t> scratchAllocationBytes =
          getL1AllocationSizeBytes(module, scratchBytes);
      if (failed(scratchAllocationBytes)) {
        return failure();
      }
      std::optional<uint64_t> reservation =
          llvm::checkedAddUnsigned(*scratchAllocationBytes, semaphoreBytes);
      if (!reservation) {
        return failure();
      }
      if (*reservation == 0) {
        module->removeAttr(kPipeConservativeL1BytesAttrName);
        return success();
      }
      module->setAttr(
          kPipeConservativeL1BytesAttrName,
          IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                           *reservation));
      return success();
    };
    auto planConservativeResources =
        [&]() -> FailureOr<std::pair<ConservativePipeResources, uint64_t>> {
      FailureOr<ConservativePipeResources> resources =
          getConservativePipeResources(module);
      if (failed(resources)) {
        return failure();
      }
      FailureOr<uint64_t> semaphoreBytes =
          getGlobalSemaphoreL1Bytes(module, resources->globalSemaphoreCount);
      if (failed(semaphoreBytes)) {
        return failure();
      }
      return std::make_pair(*resources, *semaphoreBytes);
    };
    if (groupSize == 1) {
      FailureOr<std::pair<ConservativePipeResources, uint64_t>> plan =
          planConservativeResources();
      if (failed(plan) || failed(setConservativePipeBytes(
                              plan->first.scratchBytes, plan->second))) {
        module.emitOpError("conservative PipeNet L1 size is not representable");
        signalPassFailure();
      }
      return;
    }

    ValueOriginAnalysis preExpansionAnalysis(module);
    if (failed(verifyTransferProvenance(module, preExpansionAnalysis)) ||
        failed(expandStaticPipeTransfers(module, preExpansionAnalysis))) {
      signalPassFailure();
      return;
    }
    hoistPipeLoopInvariantCode(module);

    ValueOriginAnalysis analysis(module);
    if (failed(verifyTransferProvenance(module, analysis))) {
      signalPassFailure();
      return;
    }
    FailureOr<std::unique_ptr<PipeTransferIndex>> maybeTransferIndex =
        PipeTransferIndex::create(module, analysis);
    if (failed(maybeTransferIndex)) {
      signalPassFailure();
      return;
    }
    const PipeTransferIndex &transferIndex = **maybeTransferIndex;
    // Selected record-table callbacks remain high-level IR until conversion;
    // the graph therefore contains only the static transfers expanded above.
    PipeForeachLoweringInfo foreachLoweringInfo;
    FailureOr<PipeGraph> maybePipeGraph =
        PipeGraph::build(module, transferIndex, foreachLoweringInfo);
    if (failed(maybePipeGraph)) {
      signalPassFailure();
      return;
    }
    PipeGraph &pipeGraph = *maybePipeGraph;

    FailureOr<std::pair<ConservativePipeResources, uint64_t>> conservativePlan =
        planConservativeResources();
    if (failed(conservativePlan) ||
        failed(setConservativePipeBytes(conservativePlan->first.scratchBytes,
                                        conservativePlan->second))) {
      module.emitOpError("conservative PipeNet L1 size is not representable");
      signalPassFailure();
      return;
    }
    ConservativePipeResources conservativePipeResources =
        conservativePlan->first;
    uint64_t globalSemaphoreBytes = conservativePlan->second;
    std::optional<uint64_t> overrideBytes =
        l1BudgetOverride == 0 ? std::nullopt
                              : std::optional<uint64_t>(l1BudgetOverride);
    uint64_t budgetBytes = getUsableDFBL1Bytes(module, overrideBytes);
    DFBLogicalIdentityAnalysis identities(module);
    if (!identities.succeeded()) {
      Operation *errorOperation = identities.getErrorOperation();
      (errorOperation ? errorOperation : module.getOperation())
          ->emitOpError(identities.getErrorMessage());
      signalPassFailure();
      return;
    }

    llvm::MapVector<Operation *, SmallVector<const PipeTransferNode *>>
        transfersByLoop;
    for (const PipeTransferNode &transfer : pipeGraph.getPipeTransferNodes()) {
      scf::ForOp loop = getEnclosingFor(transfer.sendOp);
      if (!loop) {
        continue;
      }
      transfersByLoop[loop.getOperation()].push_back(&transfer);
    }

    SmallVector<PipeTransportLoopCandidate, 0> candidates;
    for (auto &[loopOperation, transfers] : transfersByLoop) {
      auto loop = cast<scf::ForOp>(loopOperation);
      std::string reason;
      FailureOr<PipeTransportLoopCandidate> maybeCandidate = buildLoopCandidate(
          loop, transfers, transferIndex, pipeGraph, analysis, reason);
      if (failed(maybeCandidate)) {
        debugReject(loop, reason);
        continue;
      }
      candidates.push_back(std::move(maybeCandidate.value()));
    }

    IRRewriter rewriter(module.getContext());
    uint64_t selectedScratchBytes = conservativePipeResources.scratchBytes;
    uint64_t selectedGlobalSemaphoreBytes = globalSemaphoreBytes;
    for (PipeTransportLoopCandidate &candidate : candidates) {
      FailureOr<DFBAllocationFootprint> allocationFootprint =
          getLogicalDFBAllocationFootprint(module, identities);
      if (failed(allocationFootprint)) {
        module.emitOpError("failed to compute DFB allocation sizes");
        signalPassFailure();
        return;
      }
      std::optional<PipeTransportGrouping> grouping = selectGrouping(
          module, candidate, groupSize, *allocationFootprint, identities,
          selectedScratchBytes, selectedGlobalSemaphoreBytes, *resetStateBytes,
          budgetBytes);
      if (!grouping) {
        debugReject(candidate.loop,
                    "no group with R > 1 fits the combined L1 budget");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "PipeTransportFormation: select " << candidate.loop.getLoc()
                 << " R=" << grouping->groupSize
                 << " K=" << grouping->destinationDepth
                 << " groups=" << grouping->fullGroupCount
                 << " residual=" << grouping->residualCount << " l1="
                 << grouping->allocationBytes << "/" << budgetBytes << "\n");
      std::optional<uint64_t> candidateScratchBytes = getTransportScratchBytes(
          candidate, grouping->groupSize, grouping->destinationDepth);
      assert(candidateScratchBytes &&
             "selected grouping has no scratch allocation");
      std::optional<uint64_t> nextScratchBytes = llvm::checkedAddUnsigned(
          selectedScratchBytes, *candidateScratchBytes);
      assert(nextScratchBytes &&
             "selected scratch allocation exceeds uint64_t");
      selectedScratchBytes = *nextScratchBytes;
      FailureOr<uint64_t> residualSemaphoreBytes =
          getResidualGlobalSemaphoreBytes(module, candidate, *grouping);
      assert(succeeded(residualSemaphoreBytes) &&
             "selected residual allocation must be representable");
      std::optional<uint64_t> nextGlobalSemaphoreBytes =
          llvm::checkedAddUnsigned(selectedGlobalSemaphoreBytes,
                                   *residualSemaphoreBytes);
      assert(nextGlobalSemaphoreBytes &&
             "selected semaphore allocation exceeds uint64_t");
      selectedGlobalSemaphoreBytes = *nextGlobalSemaphoreBytes;
      applyGrouping(candidate, *grouping, rewriter);
    }
    if (failed(setConservativePipeBytes(selectedScratchBytes,
                                        selectedGlobalSemaphoreBytes))) {
      module.emitOpError("conservative PipeNet L1 size is not representable");
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
