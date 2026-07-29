// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBAcquireReleaseAnalysis.h"
#include "PipeGraph.h"
#include "PipeTransferExpansion.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "DFBAllocationLimits.h"
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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#define DEBUG_TYPE "ttl-form-pipe-transports"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFORMPIPETRANSPORTS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

enum class TransportDFBRole {
  Source,
  Destination,
};

struct TransportDFBUse {
  Value dfb;
  BindCBOp bind;
  TransportDFBRole role = TransportDFBRole::Source;
  PipeTransferNodeId transferNode = 0;
  SmallVector<CBReserveOp> reserves;
  SmallVector<CBPushOp> pushes;
  SmallVector<CBWaitOp> waits;
  SmallVector<CBPopOp> pops;
  SmallVector<AttachCBOp> attaches;
  CopyOp tensorCopy;
  TensorSliceOp tensorSlice;
};

struct PipeTransportLoopCandidate {
  scf::ForOp loop;
  int64_t lowerBound = 0;
  int64_t transferCount = 0;
  SmallVector<const PipeTransferNode *, 1> transfers;
  SmallVector<PipeTransferCreateOp, 1> transferCreates;
  SmallVector<TransportDFBUse, 0> dfbUses;
};

struct PipeTransportGrouping {
  int64_t groupSize = 1;
  int64_t destinationDepth = 1;
  int64_t fullGroupCount = 0;
  int64_t residualCount = 0;
  uint64_t allocationBytes = 0;
  llvm::DenseMap<Value, int64_t> blockCounts;

  int64_t getCompletionGroupCount() const {
    return fullGroupCount + residualCount;
  }

  int64_t getCapacityWaitCount() const {
    return std::max<int64_t>(0, fullGroupCount - destinationDepth);
  }
};

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

/// Return whether a value is independent of the candidate loop induction.
static bool isLoopInvariant(Value value, scf::ForOp loop) {
  if (loop.isDefinedOutsideOfLoop(value)) {
    return true;
  }
  auto result = dyn_cast<OpResult>(value);
  if (!result || !isPure(result.getOwner())) {
    return false;
  }
  return llvm::all_of(result.getOwner()->getOperands(), [&](Value operand) {
    return isLoopInvariant(operand, loop);
  });
}

/// Return whether `slice` enumerates consecutive transfers from `loop`.
static bool isContiguousLoopSlice(TensorSliceOp slice, scf::ForOp loop,
                                  CircularBufferType dfbType) {
  auto sliceType = cast<RankedTensorType>(slice.getResult().getType());
  if (sliceType.getShape() != dfbType.getShape()) {
    return false;
  }
  if (llvm::any_of(dfbType.getShape().drop_back(),
                   [](int64_t dimension) { return dimension != 1; })) {
    return false;
  }
  ValueRange indices = slice.getIndices();
  if (indices.empty() || indices.back() != loop.getInductionVar()) {
    return false;
  }
  return llvm::all_of(indices.drop_back(), [&](Value index) {
    return isLoopInvariant(index, loop);
  });
}

/// Collect one DFB's complete scalar transport lifecycle.
static LogicalResult collectDFBUsePattern(PipeTransportLoopCandidate &candidate,
                                          TransportDFBUse &dfbUse,
                                          const PipeGraph &pipeGraph,
                                          std::string &reason) {
  scf::ForOp loop = candidate.loop;
  for (OpOperand &use : dfbUse.dfb.getUses()) {
    if (!isInsideLoop(loop, use.getOwner())) {
      reason = "transport DFB has a use outside the candidate loop";
      return failure();
    }
  }

  loop.walk([&](Operation *operation) {
    if (auto reserve = dyn_cast<CBReserveOp>(operation);
        reserve && reserve.getCb() == dfbUse.dfb) {
      dfbUse.reserves.push_back(reserve);
    } else if (auto push = dyn_cast<CBPushOp>(operation);
               push && push.getCb() == dfbUse.dfb) {
      dfbUse.pushes.push_back(push);
    } else if (auto wait = dyn_cast<CBWaitOp>(operation);
               wait && wait.getCb() == dfbUse.dfb) {
      dfbUse.waits.push_back(wait);
    } else if (auto pop = dyn_cast<CBPopOp>(operation);
               pop && pop.getCb() == dfbUse.dfb) {
      dfbUse.pops.push_back(pop);
    } else if (auto attach = dyn_cast<AttachCBOp>(operation);
               attach && attach.getCb() == dfbUse.dfb) {
      dfbUse.attaches.push_back(attach);
    }
  });

  if (dfbUse.reserves.size() != 1 || dfbUse.pushes.size() != 1 ||
      dfbUse.waits.size() != 1 || dfbUse.pops.size() != 1) {
    reason = "transport DFB requires one reserve/push and one wait/pop";
    return failure();
  }

  const DFBReleaseOwnerMaps &owners = pipeGraph.getDFBReleaseOwnerMaps();
  if (lookupOwner<CBReserveOp>(owners.reserveByPush,
                               dfbUse.pushes.front().getOperation()) !=
          dfbUse.reserves.front() ||
      lookupOwner<CBWaitOp>(owners.waitByPop,
                            dfbUse.pops.front().getOperation()) !=
          dfbUse.waits.front()) {
    reason = "transport DFB releases do not have unique acquire owners";
    return failure();
  }

  SmallVector<CopyOp> tensorCopies;
  for (OpOperand &use : dfbUse.dfb.getUses()) {
    Operation *operation = use.getOwner();
    if (auto copy = dyn_cast<CopyOp>(operation)) {
      bool expectedDirection = dfbUse.role == TransportDFBRole::Source
                                   ? copy.getDst() == dfbUse.dfb
                                   : copy.getSrc() == dfbUse.dfb;
      if (expectedDirection) {
        tensorCopies.push_back(copy);
        continue;
      }
    }
    if (isa<CBReserveOp, CBPushOp, CBWaitOp, CBPopOp, AttachCBOp>(operation)) {
      continue;
    }
    if (dfbUse.role == TransportDFBRole::Source) {
      const PipeTransferNode *transfer =
          pipeGraph.getPipeTransferNodeForProtocolOp(operation);
      if (transfer && transfer->id == dfbUse.transferNode &&
          isa<PipeTransferSendOp>(operation)) {
        continue;
      }
    }
    reason = "transport DFB has an unsupported direct use";
    return failure();
  }

  if (tensorCopies.size() != 1) {
    reason = "transport DFB requires one tensor copy";
    return failure();
  }
  dfbUse.tensorCopy = tensorCopies.front();
  Value tensorValue = dfbUse.role == TransportDFBRole::Source
                          ? dfbUse.tensorCopy.getSrc()
                          : dfbUse.tensorCopy.getDst();
  dfbUse.tensorSlice = tensorValue.getDefiningOp<TensorSliceOp>();
  auto dfbType = cast<CircularBufferType>(dfbUse.dfb.getType());
  if (!dfbUse.tensorSlice ||
      !isContiguousLoopSlice(dfbUse.tensorSlice, loop, dfbType)) {
    reason = "tensor copy is not a contiguous loop-indexed DFB block";
    return failure();
  }
  if (!dfbUse.tensorCopy.getXf().hasOneUse() ||
      !isa<WaitOp>(*dfbUse.tensorCopy.getXf().getUsers().begin())) {
    reason = "tensor copy completion is observed outside one direct wait";
    return failure();
  }

  return success();
}

/// Return or create the DFB record for one transfer role.
static FailureOr<TransportDFBUse *>
getOrAddDFBUse(PipeTransportLoopCandidate &candidate, Value dfb,
               TransportDFBRole role, PipeTransferNodeId transferNode,
               std::string &reason) {
  for (TransportDFBUse &existing : candidate.dfbUses) {
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
  TransportDFBUse dfbUse;
  dfbUse.dfb = dfb;
  dfbUse.bind = bind;
  dfbUse.role = role;
  dfbUse.transferNode = transferNode;
  candidate.dfbUses.push_back(std::move(dfbUse));
  return &candidate.dfbUses.back();
}

/// Validate that changing the loop step does not remove unrelated effects.
static LogicalResult validateLoopEffects(PipeTransportLoopCandidate &candidate,
                                         ValueOriginAnalysis &analysis,
                                         std::string &reason) {
  llvm::DenseSet<Operation *> asyncOperations;
  llvm::DenseSet<Operation *> allowedOperations;
  for (TransportDFBUse &dfbUse : candidate.dfbUses) {
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
      FailureOr<PipeTransferPostOp> post =
          findPipeTransferPostForToken(analysis, pipeWait.getToken());
      if (succeeded(post) && allowedOperations.contains(*post)) {
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
static FailureOr<PipeTransportLoopCandidate>
buildLoopCandidate(scf::ForOp loop,
                   ArrayRef<const PipeTransferNode *> transfers,
                   const PipeGraph &pipeGraph, ValueOriginAnalysis &analysis,
                   std::string &reason) {
  PipeTransportLoopCandidate candidate;
  candidate.loop = loop;
  candidate.transfers.append(transfers.begin(), transfers.end());

  if (!loop.getInitArgs().empty() || loop.getNumResults() != 0) {
    reason = "candidate loop has loop-carried values";
    return failure();
  }
  std::optional<int64_t> lowerBound = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upperBound = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lowerBound || !upperBound || !step || *step != 1 ||
      *upperBound <= *lowerBound) {
    reason = "candidate loop requires constant bounds and unit step";
    return failure();
  }
  candidate.lowerBound = *lowerBound;
  candidate.transferCount = *upperBound - *lowerBound;

  llvm::DenseSet<Operation *> seenCreates;
  for (const PipeTransferNode *transfer : transfers) {
    if (transfer->blockSpan != 1 || getEnclosingFor(transfer->sendOp) != loop) {
      reason =
          "transfer is not scalar or does not execute in the candidate loop";
      return failure();
    }
    auto send = cast<PipeTransferSendOp>(transfer->sendOp);
    FailureOr<TransportDFBUse *> source =
        getOrAddDFBUse(candidate, send.getSrc(), TransportDFBRole::Source,
                       transfer->id, reason);
    if (failed(source)) {
      return failure();
    }
    FailureOr<PipeTransferCreateOp> sendCreate =
        findPipeTransferCreateForTransfer(analysis, send.getTransfer());
    if (failed(sendCreate)) {
      reason = "sender transfer has no unique creation operation";
      return failure();
    }
    if (isInsideLoop(loop, sendCreate->getOperation())) {
      reason = "transfer creation must be outside the candidate loop";
      return failure();
    }
    if (seenCreates.insert(sendCreate->getOperation()).second) {
      candidate.transferCreates.push_back(*sendCreate);
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
                                TransportDFBRole::Destination, transfer->id,
                                reason))) {
        return failure();
      }
      FailureOr<PipeTransferCreateOp> postCreate =
          findPipeTransferCreateForTransfer(analysis, post.getTransfer());
      if (failed(postCreate)) {
        reason = "receiver transfer has no unique creation operation";
        return failure();
      }
      if (isInsideLoop(loop, postCreate->getOperation())) {
        reason = "transfer creation must be outside the candidate loop";
        return failure();
      }
      if (seenCreates.insert(postCreate->getOperation()).second) {
        candidate.transferCreates.push_back(*postCreate);
      }
    }
  }

  for (TransportDFBUse &dfbUse : candidate.dfbUses) {
    if (failed(collectDFBUsePattern(candidate, dfbUse, pipeGraph, reason))) {
      return failure();
    }
  }
  if (failed(validateLoopEffects(candidate, analysis, reason))) {
    return failure();
  }
  return std::move(candidate);
}

/// Compute storage and synchronization facts for one `(R, K)` choice.
static std::optional<PipeTransportGrouping> evaluateGrouping(
    PipeTransportLoopCandidate &candidate, int64_t groupSize,
    int64_t destinationDepth,
    const DFBAllocationFootprint &allocationFootprint,
    uint64_t budgetBytes) {
  if (groupSize <= 1 || groupSize > candidate.transferCount) {
    return std::nullopt;
  }
  int64_t fullGroupCount = candidate.transferCount / groupSize;
  if (fullGroupCount <= 0 || destinationDepth <= 0 ||
      destinationDepth > fullGroupCount) {
    return std::nullopt;
  }

  PipeTransportGrouping grouping;
  grouping.groupSize = groupSize;
  grouping.destinationDepth = destinationDepth;
  grouping.fullGroupCount = fullGroupCount;
  grouping.residualCount = candidate.transferCount % groupSize;

  llvm::DenseMap<int64_t, uint64_t> allocationBytesByIndex;
  for (TransportDFBUse &dfbUse : candidate.dfbUses) {
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
        dfbUse.role == TransportDFBRole::Source ? 1 : destinationDepth;
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
    auto resizedType =
        CircularBufferType::get(oldType.getContext(), oldType.getShape(),
                                oldType.getElementType(), blockCount);
    FailureOr<uint64_t> allocationBytes =
        getDFBAllocationSizeBytes(resizedType);
    if (failed(allocationBytes)) {
      return std::nullopt;
    }
    int64_t dfbIndex = dfbUse.bind.getCbIndex().getSExtValue();
    uint64_t &minimumBytes = allocationBytesByIndex[dfbIndex];
    minimumBytes = std::max(minimumBytes, *allocationBytes);
    grouping.blockCounts[dfbUse.dfb] = blockCount;
  }

  FailureOr<uint64_t> totalBytes =
      allocationFootprint.getTotalBytesWithMinimumAllocations(
          allocationBytesByIndex);
  if (failed(totalBytes) || *totalBytes > budgetBytes) {
    return std::nullopt;
  }
  grouping.allocationBytes = *totalBytes;
  return grouping;
}

/// Return the largest legal destination depth for `groupSize`.
static std::optional<PipeTransportGrouping> selectDestinationDepth(
    PipeTransportLoopCandidate &candidate, int64_t groupSize,
    const DFBAllocationFootprint &allocationFootprint, uint64_t budgetBytes) {
  int64_t upperDepth = candidate.transferCount / groupSize;
  int64_t lower = 1;
  int64_t upper = upperDepth;
  std::optional<PipeTransportGrouping> selected;
  while (lower <= upper) {
    int64_t middle = lower + (upper - lower) / 2;
    std::optional<PipeTransportGrouping> grouping = evaluateGrouping(
        candidate, groupSize, middle, allocationFootprint, budgetBytes);
    if (grouping) {
      selected = std::move(grouping);
      lower = middle + 1;
    } else {
      upper = middle - 1;
    }
  }
  return selected;
}

/// Bound exhaustive group-size checks by the storage required for one group.
static std::optional<int64_t>
getGroupSizeUpperBound(PipeTransportLoopCandidate &candidate,
                       int64_t upperBound, uint64_t budgetBytes) {
  llvm::DenseMap<int64_t, uint64_t> bytesPerBlockByIndex;
  for (TransportDFBUse &dfbUse : candidate.dfbUses) {
    auto oldType = cast<CircularBufferType>(dfbUse.dfb.getType());
    auto oneBlockType = CircularBufferType::get(
        oldType.getContext(), oldType.getShape(), oldType.getElementType(), 1);
    FailureOr<uint64_t> bytesPerBlock = getDFBAllocationSizeBytes(oneBlockType);
    if (failed(bytesPerBlock) || *bytesPerBlock == 0) {
      return std::nullopt;
    }
    int64_t dfbIndex = dfbUse.bind.getCbIndex().getSExtValue();
    uint64_t &minimumBytes = bytesPerBlockByIndex[dfbIndex];
    minimumBytes = std::max(minimumBytes, *bytesPerBlock);
  }

  uint64_t bytesPerGroup = 0;
  for (const auto &entry : bytesPerBlockByIndex) {
    std::optional<uint64_t> total =
        llvm::checkedAddUnsigned(bytesPerGroup, entry.second);
    if (!total) {
      return std::nullopt;
    }
    bytesPerGroup = *total;
  }
  if (bytesPerGroup == 0) {
    return std::nullopt;
  }

  uint64_t requestedUpperBound =
      static_cast<uint64_t>(std::min(upperBound, candidate.transferCount));
  uint64_t budgetUpperBound = budgetBytes / bytesPerGroup;
  return static_cast<int64_t>(std::min(requestedUpperBound, budgetUpperBound));
}

/// Return whether `candidate` improves on `selected`.
static bool isBetterGrouping(const PipeTransportGrouping &candidate,
                             const PipeTransportGrouping &selected) {
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
static std::optional<PipeTransportGrouping> selectGrouping(
    PipeTransportLoopCandidate &candidate, int64_t requestedGroupSize,
    const DFBAllocationFootprint &allocationFootprint, uint64_t budgetBytes) {
  int64_t upperBound =
      requestedGroupSize > 1 ? requestedGroupSize : candidate.transferCount;
  std::optional<int64_t> maybeUpperBound =
      getGroupSizeUpperBound(candidate, upperBound, budgetBytes);
  if (!maybeUpperBound || *maybeUpperBound <= 1) {
    return std::nullopt;
  }
  upperBound = *maybeUpperBound;

  if (requestedGroupSize > 1) {
    for (int64_t groupSize = upperBound; groupSize >= 2; --groupSize) {
      std::optional<PipeTransportGrouping> grouping = selectDestinationDepth(
          candidate, groupSize, allocationFootprint, budgetBytes);
      if (grouping) {
        return grouping;
      }
    }
    return std::nullopt;
  }

  std::optional<PipeTransportGrouping> selected;
  for (int64_t groupSize = 2; groupSize <= upperBound; ++groupSize) {
    std::optional<PipeTransportGrouping> grouping = selectDestinationDepth(
        candidate, groupSize, allocationFootprint, budgetBytes);
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
static void resizeDFB(TransportDFBUse &dfbUse, int64_t blockCount,
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
static void groupDFBLifecycle(TransportDFBUse &dfbUse, int64_t groupSize,
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

  for (TransportDFBUse &dfbUse : candidate.dfbUses) {
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
    if (groupSize == 1) {
      return;
    }

    ValueOriginAnalysis preExpansionAnalysis(module);
    if (failed(verifyTransferProvenance(module, preExpansionAnalysis)) ||
        failed(expandPipeTransfers(module, preExpansionAnalysis))) {
      signalPassFailure();
      return;
    }
    hoistPipeLoopInvariantCode(module);

    ValueOriginAnalysis analysis(module);
    if (failed(verifyTransferProvenance(module, analysis))) {
      signalPassFailure();
      return;
    }
    FailureOr<PipeGraph> maybePipeGraph = PipeGraph::build(module, analysis);
    if (failed(maybePipeGraph)) {
      signalPassFailure();
      return;
    }
    PipeGraph &pipeGraph = *maybePipeGraph;

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
      FailureOr<PipeTransportLoopCandidate> maybeCandidate =
          buildLoopCandidate(loop, transfers, pipeGraph, analysis, reason);
      if (failed(maybeCandidate)) {
        debugReject(loop, reason);
        continue;
      }
      candidates.push_back(std::move(maybeCandidate.value()));
    }

    IRRewriter rewriter(module.getContext());
    for (PipeTransportLoopCandidate &candidate : candidates) {
      FailureOr<DFBAllocationFootprint> allocationFootprint =
          getDFBAllocationFootprint(module);
      if (failed(allocationFootprint)) {
        module.emitOpError("failed to compute DFB allocation sizes");
        signalPassFailure();
        return;
      }
      std::optional<uint64_t> overrideBytes =
          l1BudgetOverride == 0
              ? std::nullopt
              : std::optional<uint64_t>(l1BudgetOverride);
      uint64_t budgetBytes = getUsableDFBL1Bytes(module, overrideBytes);
      std::optional<PipeTransportGrouping> grouping =
          selectGrouping(candidate, groupSize, *allocationFootprint,
                         budgetBytes);
      if (!grouping) {
        debugReject(candidate.loop,
                    "no group with R > 1 fits the L1 DFB budget");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "PipeTransportFormation: select " << candidate.loop.getLoc()
                 << " R=" << grouping->groupSize
                 << " K=" << grouping->destinationDepth
                 << " groups=" << grouping->fullGroupCount
                 << " residual=" << grouping->residualCount << " l1="
                 << grouping->allocationBytes << "/" << budgetBytes << "\n");
      applyGrouping(candidate, *grouping, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
