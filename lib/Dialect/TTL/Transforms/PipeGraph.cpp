// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeGraph.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <utility>

#define DEBUG_TYPE "ttl-pipe-graph"

namespace mlir::tt::ttl {

static PipeKey getPipeKey(PipeType pipeType) {
  return {pipeType.getSrcX(),      pipeType.getSrcY(),
          pipeType.getDstStartX(), pipeType.getDstStartY(),
          pipeType.getDstEndX(),   pipeType.getDstEndY(),
          pipeType.getPipeNetId()};
}

struct PipeGraphAnalysisState : LaunchNodeDomainState {
  llvm::DenseMap<Operation *, LaunchNodeDomain> operationLaunchDomains;
  DFBReleaseOwnerMaps dfbReleaseOwners;
};

namespace {

static LogicalResult collectLaunchNodeDomains(ModuleOp mod,
                                              PipeGraphAnalysisState &state) {
  state.initialize(mod);
  if (!state.hasLaunchGrid) {
    return success();
  }

  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  LaunchNodeDomainAnalysisOptions options;
  options.narrowPipeNetScopes = true;
  options.operationCallback = [&](Operation *op, const LaunchNodeDomain &domain,
                                  Operation * /*unanalyzableOp*/) {
    state.operationLaunchDomains[op] = domain;
  };
  solver.load<LaunchNodeDomainAnalysis>(state, options);
  if (failed(solver.initializeAndRun(mod))) {
    return failure();
  }
  buildDFBReleaseOwnerMaps(mod, state.dfbReleaseOwners);
  return success();
}

static LaunchNodeDomain
lookupOperationLaunchDomain(Operation *op, PipeGraphAnalysisState &state) {
  if (!state.hasLaunchGrid) {
    return LaunchNodeDomain::unknown();
  }
  auto it = state.operationLaunchDomains.find(op);
  if (it == state.operationLaunchDomains.end()) {
    return LaunchNodeDomain::unknown();
  }
  return it->second;
}

static LaunchNodeCoord getLaunchNodeCoord(PipeReceiverCoord receiver) {
  return {receiver.x, receiver.y};
}

static bool isReceiverDFB(Value cb, const PipeReceiverDFBKey &receiverDFB) {
  std::optional<int64_t> dfbIndex = getCBIndex(cb);
  return dfbIndex && *dfbIndex == receiverDFB.dfbIndex;
}

static bool isPostForReceiverDFB(
    PipeTransferPostOp postOp, const PipeReceiverDFBKey &receiverDFB,
    const llvm::MapVector<PipeKey, ReceiverDFBInfo> &receiverDFBs,
    PipeGraphAnalysisState &state) {
  auto createOp = findPipeTransferCreateForTransfer(postOp.getTransfer());
  if (!createOp) {
    return false;
  }
  PipeKey pipeKey =
      getPipeKey(mlir::cast<PipeType>(createOp.getPipe().getType()));
  auto receiverIt = receiverDFBs.find(pipeKey);
  if (receiverIt == receiverDFBs.end() ||
      receiverIt->second.dfbIndex != receiverDFB.dfbIndex) {
    return false;
  }
  LaunchNodeDomain postDomain =
      lookupOperationLaunchDomain(postOp.getOperation(), state);
  return knownLaunchNodeDomainContains(
      postDomain, getLaunchNodeCoord(receiverDFB.receiver));
}

static SmallVector<PipeTransferPostOp>
getPostsOwnedByReserve(CBReserveOp reserveOp,
                       ArrayRef<PipeTransferPostOp> posts) {
  SmallVector<PipeTransferPostOp> ownedPosts;
  for (PipeTransferPostOp postOp : posts) {
    if (findCBReserveForPipeReceive(postOp.getDst()) == reserveOp) {
      ownedPosts.push_back(postOp);
    }
  }
  return ownedPosts;
}

static std::optional<int64_t> getReceiverSlotSpanBlocksForPost(
    PipeTransferPostOp postOp,
    const llvm::MapVector<PipeKey, ReceiverDFBInfo> &receiverDFBs) {
  auto createOp = findPipeTransferCreateForTransfer(postOp.getTransfer());
  if (!createOp) {
    return std::nullopt;
  }
  PipeKey pipeKey =
      getPipeKey(mlir::cast<PipeType>(createOp.getPipe().getType()));
  auto receiverIt = receiverDFBs.find(pipeKey);
  if (receiverIt == receiverDFBs.end()) {
    return std::nullopt;
  }
  return receiverIt->second.receiverSlotSpanBlocks;
}

static bool isBeforeInSameBlock(Operation *before, Operation *after) {
  return before->getBlock() == after->getBlock() &&
         before->isBeforeInBlock(after);
}

static void collectReceiveWaitsByPost(
    ModuleOp mod,
    llvm::DenseMap<Operation *, SmallVector<PipeTransferWaitOp>> &waitsByPost) {
  mod.walk([&](PipeTransferWaitOp waitOp) {
    PipeTransferPostOp postOp = findPipeTransferPostForToken(waitOp.getToken());
    if (!postOp) {
      return;
    }
    waitsByPost[postOp.getOperation()].push_back(waitOp);
  });
}

static bool hasMatchingReceiveWaitBeforePush(
    PipeTransferPostOp postOp, CBPushOp pushOp,
    const llvm::DenseMap<Operation *, SmallVector<PipeTransferWaitOp>>
        &waitsByPost) {
  auto waitIt = waitsByPost.find(postOp.getOperation());
  if (waitIt == waitsByPost.end()) {
    return false;
  }
  for (PipeTransferWaitOp waitOp : waitIt->second) {
    if (isBeforeInSameBlock(postOp, waitOp) &&
        isBeforeInSameBlock(waitOp, pushOp)) {
      return true;
    }
  }
  return false;
}

static void printReceiverDFB(llvm::raw_ostream &os,
                             const PipeReceiverDFBKey &receiverDFB) {
  os << "receiver(" << receiverDFB.receiver.x << ", " << receiverDFB.receiver.y
     << ") DFB " << receiverDFB.dfbIndex;
}

static void debugRejectPipeOnlyStream(const PipeReceiverDFBKey &receiverDFB,
                                      llvm::StringRef reason) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeGraph: reject pipe-only stream for ";
    printReceiverDFB(llvm::dbgs(), receiverDFB);
    llvm::dbgs() << ": " << reason << "\n";
  });
}

static void debugAcceptPipeOnlyStream(const PipeReceiverDFBKey &receiverDFB) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeGraph: accept pipe-only stream for ";
    printReceiverDFB(llvm::dbgs(), receiverDFB);
    llvm::dbgs() << "\n";
  });
}

template <typename Fn>
static WalkResult walkNestedOpsInOrder(Operation *op, Fn &&callback) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (callback(&nestedOp).wasInterrupted()) {
          return WalkResult::interrupt();
        }
        if (walkNestedOpsInOrder(&nestedOp, callback).wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
    }
  }
  return WalkResult::advance();
}

struct LiveReceiverSlot {
  int64_t slot = 0;
  int64_t span = 0;
};

struct ReceiverSlotState {
  int64_t nextSlot = 0;
  SmallVector<LiveReceiverSlot> liveSlots;
};

static bool slotRangesOverlap(int64_t lhsSlot, int64_t lhsSpan, int64_t rhsSlot,
                              int64_t rhsSpan) {
  return lhsSlot < rhsSlot + rhsSpan && rhsSlot < lhsSlot + lhsSpan;
}

// Pops release whole live slots in FIFO order, matching the hardware ring's
// in-order consumption. A pop that freed only part of the oldest slot would
// leave the slot's high blocks live while the overlap check treats its low
// blocks as free, so require each release to drain a whole live slot.
static LogicalResult releaseReceiverSlots(CBPopOp popOp,
                                          ReceiverSlotState &state,
                                          int64_t releasedBlocks) {
  unsigned slotsToRelease = 0;
  int64_t remainingBlocks = releasedBlocks;
  for (const LiveReceiverSlot &slot : state.liveSlots) {
    if (remainingBlocks == 0) {
      break;
    }
    if (remainingBlocks < slot.span) {
      return popOp.emitError()
             << "pipe receiver DFB pop releases " << remainingBlocks
             << " block(s), but oldest live receive slot spans " << slot.span
             << " block(s); receiver pops must release whole DFB slots";
    }
    remainingBlocks -= slot.span;
    ++slotsToRelease;
  }
  state.liveSlots.erase(state.liveSlots.begin(),
                        state.liveSlots.begin() + slotsToRelease);
  return success();
}

} // namespace

LogicalResult PipeGraph::addReceiverDFB(
    int64_t srcX, int64_t srcY, int64_t dstStartX, int64_t dstStartY,
    int64_t dstEndX, int64_t dstEndY, int64_t pipeNetId, int64_t dfbIndex,
    CircularBufferType dfbType, bool hasStaticTileOffset,
    int64_t staticTileOffset, int64_t receiverSlotSpanBlocks,
    Operation *receiverReserveOp, PipeTransferContract transferContract,
    ArrayRef<Operation *> transferCreateOps, int64_t blockCount, Location loc) {
  PipeKey key{srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY, pipeNetId};
  auto existing = receiverDFBs.find(key);
  bool hasMultipleReceivers = dstStartX != dstEndX || dstStartY != dstEndY;
  if (existing != receiverDFBs.end()) {
    if (hasMultipleReceivers &&
        (existing->second.dfbIndex != dfbIndex ||
         existing->second.dfbType != dfbType ||
         !existing->second.hasStaticTileOffset ||
         existing->second.staticTileOffset != staticTileOffset)) {
      auto diag = emitError(loc)
                  << "collective pipe receive posts publish different "
                     "destination addresses; TT-Metal NoC multicast requires "
                     "one destination SRAM address for all receivers";
      diag.attachNote(existing->second.loc)
          << "previous collective receive post for this pipe was here";
      return failure();
    }

    if (existing->second.dfbIndex != dfbIndex ||
        existing->second.dfbType != dfbType ||
        existing->second.blockCount != blockCount) {
      auto diag = emitError(loc)
                  << "conflicting receiver DFBs for the same pipe";
      diag.attachNote(existing->second.loc)
          << "previous receiver DFB for this pipe was here";
      return failure();
    }
    existing->second.hasStaticTileOffset =
        existing->second.hasStaticTileOffset && hasStaticTileOffset &&
        existing->second.staticTileOffset == staticTileOffset;
    return success();
  }
  receiverDFBs.insert({key,
                       {dfbIndex, dfbType, hasStaticTileOffset,
                        staticTileOffset, std::nullopt, receiverSlotSpanBlocks,
                        blockCount, loc, receiverReserveOp, transferContract,
                        SmallVector<Operation *>(transferCreateOps.begin(),
                                                 transferCreateOps.end())}});
  return success();
}

static FailureOr<int64_t>
assignReceiverPhysicalSlot(const PipeKey &pipeKey,
                           ReceiverDFBInfo &receiverInfo,
                           ReceiverSlotState &slotState) {
  int64_t span = receiverInfo.receiverSlotSpanBlocks;
  if (span <= 0 || span > receiverInfo.blockCount) {
    return emitError(receiverInfo.loc)
           << (pipeKey.hasSingleReceiver() ? "gather" : "collective overlap")
           << " pipe receiver DFB reserves " << span
           << " block(s) but block_count=" << receiverInfo.blockCount;
  }

  int64_t slot = slotState.nextSlot;
  if (slot + span > receiverInfo.blockCount) {
    return emitError(receiverInfo.loc)
           << (pipeKey.hasSingleReceiver() ? "gather" : "collective overlap")
           << " pipe receiver DFB reserve at slot " << slot << " spans " << span
           << " block(s), which would wrap block_count="
           << receiverInfo.blockCount;
  }

  for (const LiveReceiverSlot &liveSlot : slotState.liveSlots) {
    if (slotRangesOverlap(slot, span, liveSlot.slot, liveSlot.span)) {
      return emitError(receiverInfo.loc)
             << (pipeKey.hasSingleReceiver() ? "gather" : "collective overlap")
             << " pipe receiver DFB reuses slot " << slot
             << " before a receiver pop releases it; add a receiver pop before "
                "reusing the DFB slot or increase block_count";
    }
  }

  slotState.liveSlots.push_back(LiveReceiverSlot{slot, span});
  slotState.nextSlot = (slot + span) % receiverInfo.blockCount;
  return slot;
}

LogicalResult
PipeGraph::assignReceiverSlotIndices(ModuleOp mod,
                                     PipeGraphAnalysisState &analysisState) {
  llvm::DenseMap<PipeReceiverDFBKey, ReceiverSlotState> slotStateByReceiverDFB;
  llvm::DenseMap<PipeReceiverDFBKey, llvm::DenseMap<Operation *, int64_t>>
      slotByReceiverReserve;
  auto processPost = [&](PipeTransferPostOp postOp) -> LogicalResult {
    auto createOp = findPipeTransferCreateForTransfer(postOp.getTransfer());
    if (!createOp) {
      return success();
    }
    PipeKey pipeKey =
        getPipeKey(mlir::cast<PipeType>(createOp.getPipe().getType()));
    auto receiverIt = receiverDFBs.find(pipeKey);
    if (receiverIt == receiverDFBs.end()) {
      return success();
    }

    ReceiverDFBInfo &receiverInfo = receiverIt->second;
    if (receiverInfo.receiverSlotIndex.has_value()) {
      return success();
    }

    LaunchNodeDomain postDomain =
        lookupOperationLaunchDomain(postOp.getOperation(), analysisState);
    std::optional<int64_t> uniformSlot;
    bool hasReceiver = false;
    bool nonUniformSlot = false;
    LogicalResult result = success();
    pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
      if (failed(result) ||
          !launchNodeDomainContains(postDomain, getLaunchNodeCoord(receiver))) {
        return;
      }
      hasReceiver = true;
      PipeReceiverDFBKey receiverDFB{receiver, receiverInfo.dfbIndex};
      auto &slotByReserve = slotByReceiverReserve[receiverDFB];
      auto reserveIt = slotByReserve.find(receiverInfo.receiverReserveOp);
      int64_t slot = 0;
      if (reserveIt == slotByReserve.end()) {
        FailureOr<int64_t> assignedSlot = assignReceiverPhysicalSlot(
            pipeKey, receiverInfo, slotStateByReceiverDFB[receiverDFB]);
        if (failed(assignedSlot)) {
          result = failure();
          return;
        }
        slot = *assignedSlot;
        slotByReserve[receiverInfo.receiverReserveOp] = slot;
      } else {
        slot = reserveIt->second;
      }
      if (!uniformSlot) {
        uniformSlot = slot;
      } else if (*uniformSlot != slot) {
        nonUniformSlot = true;
      }
    });
    if (failed(result)) {
      return failure();
    }
    if (!hasReceiver || nonUniformSlot) {
      receiverInfo.receiverSlotIndex = std::nullopt;
      return success();
    }
    receiverInfo.receiverSlotIndex = *uniformSlot;
    return success();
  };

  auto processPop = [&](CBPopOp popOp) -> LogicalResult {
    std::optional<int64_t> dfbIndex = getCBIndex(popOp.getCb());
    if (!dfbIndex) {
      return success();
    }
    std::optional<int64_t> releasedBlocks = getReleasedBlockCount(popOp);
    if (!releasedBlocks) {
      return success();
    }
    LaunchNodeDomain popDomain =
        lookupOperationLaunchDomain(popOp.getOperation(), analysisState);
    if (!popDomain.known) {
      return success();
    }
    for (LaunchNodeCoord coord : popDomain.nodes) {
      PipeReceiverDFBKey receiverDFB{PipeReceiverCoord{coord.x, coord.y},
                                     *dfbIndex};
      auto stateIt = slotStateByReceiverDFB.find(receiverDFB);
      if (stateIt == slotStateByReceiverDFB.end()) {
        continue;
      }
      if (failed(
              releaseReceiverSlots(popOp, stateIt->second, *releasedBlocks))) {
        return failure();
      }
    }
    return success();
  };

  WalkResult walkResult =
      walkNestedOpsInOrder(mod.getOperation(), [&](Operation *op) {
        if (auto postOp = dyn_cast<PipeTransferPostOp>(op)) {
          return failed(processPost(postOp)) ? WalkResult::interrupt()
                                             : WalkResult::advance();
        }
        if (auto popOp = dyn_cast<CBPopOp>(op)) {
          return failed(processPop(popOp)) ? WalkResult::interrupt()
                                           : WalkResult::advance();
        }
        return WalkResult::advance();
      });
  return walkResult.wasInterrupted() ? failure() : success();
}

LogicalResult PipeGraph::verifyReceiverDFBBlockCounts() const {
  for (auto &[pk, info] : receiverDFBs) {
    if (!info.receiverSlotIndex.has_value()) {
      if (pk.hasSingleReceiver()) {
        return emitError(info.loc)
               << "point-to-point pipe receiver post is not proven to execute "
                  "on the receiver node; cannot assign a receiver DFB slot";
      }
      return emitError(info.loc)
             << "collective pipe receiver posts reserve different DFB slots; "
                "TT-Metal NoC multicast requires one destination SRAM address "
                "for all receivers";
    }
    int64_t requiredBlocks =
        *info.receiverSlotIndex + info.receiverSlotSpanBlocks;
    if (info.blockCount < requiredBlocks) {
      return emitError(info.loc)
             << (pk.hasSingleReceiver() ? "gather" : "collective overlap")
             << " pipe receiver DFB has block_count=" << info.blockCount
             << " but slot " << *info.receiverSlotIndex
             << " is assigned to this pipe; "
             << "block_count must be >= " << requiredBlocks;
    }
  }
  return success();
}

LogicalResult
PipeGraph::provePipeOnlyReceiverStreams(ModuleOp mod,
                                        PipeGraphAnalysisState &analysisState) {
  llvm::DenseMap<Operation *, SmallVector<PipeTransferWaitOp>> waitsByPost;
  collectReceiveWaitsByPost(mod, waitsByPost);

  for (PipeReceiverDFBNode &node : receiverDFBNodes) {
    PipeReceiverDFBKey receiverDFB = node.receiverDFB;
    LaunchNodeDomain receiverDomain =
        getSingleLaunchNodeDomain(getLaunchNodeCoord(receiverDFB.receiver));

    SmallVector<PipeTransferPostOp> posts;
    mod.walk([&](PipeTransferPostOp postOp) {
      if (isPostForReceiverDFB(postOp, receiverDFB, receiverDFBs,
                               analysisState)) {
        posts.push_back(postOp);
      }
    });
    if (posts.empty()) {
      debugRejectPipeOnlyStream(receiverDFB, "no matching receiver posts");
      continue;
    }

    bool valid = true;
    auto reject = [&](llvm::StringRef reason) {
      debugRejectPipeOnlyStream(receiverDFB, reason);
      valid = false;
    };
    llvm::DenseSet<Operation *> postsWithPush;
    llvm::DenseSet<Operation *> waitsWithPop;

    mod.walk([&](CBPushOp pushOp) {
      if (!valid || !isReceiverDFB(pushOp.getCb(), receiverDFB)) {
        return;
      }
      LaunchNodeDomain pushDomain =
          lookupOperationLaunchDomain(pushOp.getOperation(), analysisState);
      if (!launchNodeDomainsOverlap(pushDomain, receiverDomain)) {
        return;
      }
      if (!knownLaunchNodeDomainContains(
              pushDomain, getLaunchNodeCoord(receiverDFB.receiver)) ||
          !isNocKernelThread(pushOp)) {
        reject("push is not in the receiver NOC domain");
        return;
      }
      std::optional<int64_t> pushedBlocks = getPushedBlockCount(pushOp);
      if (!pushedBlocks) {
        reject("push block count is not a whole DFB block count");
        return;
      }
      auto ownerIt = analysisState.dfbReleaseOwners.reserveByPush.find(
          pushOp.getOperation());
      auto reserveOp =
          ownerIt == analysisState.dfbReleaseOwners.reserveByPush.end()
              ? CBReserveOp()
              : dyn_cast_or_null<CBReserveOp>(ownerIt->second);
      if (!reserveOp) {
        reject("push has no unique receiver reserve owner");
        return;
      }
      SmallVector<PipeTransferPostOp> ownedPosts =
          getPostsOwnedByReserve(reserveOp, posts);
      if (ownedPosts.empty()) {
        reject("push reserve owns no matching receiver post");
        return;
      }
      int64_t postedBlocks = 0;
      for (PipeTransferPostOp postOp : ownedPosts) {
        if (!hasMatchingReceiveWaitBeforePush(postOp, pushOp, waitsByPost)) {
          reject("post has no receive wait before push");
          return;
        }
        std::optional<int64_t> span =
            getReceiverSlotSpanBlocksForPost(postOp, receiverDFBs);
        if (!span) {
          reject("post has no receiver slot span");
          return;
        }
        postedBlocks += *span;
        if (!postsWithPush.insert(postOp.getOperation()).second) {
          reject("post is consumed by more than one push");
          return;
        }
      }
      if (*pushedBlocks != postedBlocks) {
        reject("push block count does not match posted receiver slot span");
      }
    });
    if (!valid) {
      continue;
    }

    for (PipeTransferPostOp postOp : posts) {
      if (!postsWithPush.contains(postOp.getOperation())) {
        reject("post is not consumed by a receiver push");
        break;
      }
    }
    if (!valid) {
      continue;
    }

    mod.walk([&](CBPopOp popOp) {
      if (!valid || !isReceiverDFB(popOp.getCb(), receiverDFB)) {
        return;
      }
      LaunchNodeDomain popDomain =
          lookupOperationLaunchDomain(popOp.getOperation(), analysisState);
      if (!launchNodeDomainsOverlap(popDomain, receiverDomain)) {
        return;
      }
      if (!knownLaunchNodeDomainContains(
              popDomain, getLaunchNodeCoord(receiverDFB.receiver)) ||
          !isNocKernelThread(popOp)) {
        reject("pop is not in the receiver NOC domain");
        return;
      }
      std::optional<int64_t> releasedBlocks = getReleasedBlockCount(popOp);
      if (!releasedBlocks) {
        reject("pop block count is not a whole DFB block count");
        return;
      }
      auto ownerIt =
          analysisState.dfbReleaseOwners.waitByPop.find(popOp.getOperation());
      auto waitOp = ownerIt == analysisState.dfbReleaseOwners.waitByPop.end()
                        ? CBWaitOp()
                        : dyn_cast_or_null<CBWaitOp>(ownerIt->second);
      if (!waitOp) {
        reject("pop has no unique receiver wait owner");
        return;
      }
      LaunchNodeDomain waitDomain =
          lookupOperationLaunchDomain(waitOp.getOperation(), analysisState);
      if (!knownLaunchNodeDomainContains(
              waitDomain, getLaunchNodeCoord(receiverDFB.receiver)) ||
          !isNocKernelThread(waitOp)) {
        reject("wait is not in the receiver NOC domain");
        return;
      }
      std::optional<int64_t> waitedBlocks = getWaitedBlockCount(waitOp);
      if (!waitedBlocks || *waitedBlocks != *releasedBlocks) {
        reject("wait and pop use different block counts");
        return;
      }
      if (!waitsWithPop.insert(waitOp.getOperation()).second) {
        reject("wait owns more than one pop");
      }
    });

    node.hasProvenPipeOnlyStream = valid;
    if (valid) {
      debugAcceptPipeOnlyStream(receiverDFB);
    }
  }
  return success();
}

const ReceiverDFBInfo *PipeGraph::lookupReceiverDFB(const PipeKey &key) const {
  auto it = receiverDFBs.find(key);
  if (it == receiverDFBs.end()) {
    return nullptr;
  }
  return &it->second;
}

LaunchNodeDomain PipeGraph::getOperationLaunchDomain(Operation *op) const {
  if (!hasAnalyzedLaunchGrid) {
    return LaunchNodeDomain::unknown();
  }
  auto it = operationLaunchDomains.find(op);
  if (it == operationLaunchDomains.end()) {
    return LaunchNodeDomain::unknown();
  }
  return it->second;
}

void PipeGraph::rebuildEndpointGraph() {
  pipeEdges.clear();
  pipeReceiverEndpoints.clear();
  receiverDFBNodes.clear();

  llvm::DenseMap<PipeReceiverDFBKey, PipeReceiverDFBNodeId> nodeIdByReceiverDFB;
  for (const auto &entry : receiverDFBs) {
    const PipeKey &pipeKey = entry.first;
    const ReceiverDFBInfo &receiverInfo = entry.second;
    PipeEdgeId pipeEdgeId = pipeEdges.size();
    pipeEdges.push_back(PipeEdge{pipeEdgeId,
                                 pipeKey,
                                 receiverInfo.transferContract,
                                 receiverInfo,
                                 receiverInfo.transferCreateOps,
                                 {}});
    PipeEdge &pipeEdge = pipeEdges.back();

    pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
      PipeReceiverDFBKey receiverDFB{receiver, receiverInfo.dfbIndex};
      auto nodeIt = nodeIdByReceiverDFB.find(receiverDFB);
      PipeReceiverDFBNodeId receiverDFBNodeId = 0;
      if (nodeIt == nodeIdByReceiverDFB.end()) {
        receiverDFBNodeId = receiverDFBNodes.size();
        nodeIdByReceiverDFB.insert({receiverDFB, receiverDFBNodeId});
        receiverDFBNodes.push_back(
            PipeReceiverDFBNode{receiverDFBNodeId, receiverDFB, {}, 1});
      } else {
        receiverDFBNodeId = nodeIt->second;
      }

      PipeReceiverEndpointId endpointId = pipeReceiverEndpoints.size();
      pipeReceiverEndpoints.push_back(PipeReceiverEndpoint{
          endpointId, pipeEdgeId, receiverDFBNodeId, receiver, receiverDFB,
          *receiverInfo.receiverSlotIndex});
      pipeEdge.receiverEndpoints.push_back(endpointId);
      PipeReceiverDFBNode &receiverDFBNode =
          receiverDFBNodes[receiverDFBNodeId];
      receiverDFBNode.writerEndpoints.push_back(endpointId);
      receiverDFBNode.receiverBatchSize =
          std::max(receiverDFBNode.receiverBatchSize,
                   *receiverInfo.receiverSlotIndex +
                       receiverInfo.receiverSlotSpanBlocks);
    });
  }
}

struct PipeTransferInfo {
  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;
  SmallVector<Operation *> transferCreateOps;
};

static llvm::MapVector<PipeKey, PipeTransferInfo>
collectPipeTransferInfo(ModuleOp mod) {
  llvm::MapVector<PipeKey, PipeTransferInfo> pipeInfo;
  auto addTransfer = [&](PipeTransferCreateOp op) {
    if (!op) {
      return;
    }
    auto pipeType = mlir::cast<PipeType>(op.getPipe().getType());
    PipeTransferContract contract = getPipeTransferContract(op);
    PipeKey key = getPipeKey(pipeType);
    auto existing = pipeInfo.find(key);
    if (existing == pipeInfo.end()) {
      pipeInfo.insert({key, PipeTransferInfo{contract, {op.getOperation()}}});
      return;
    }
    if (!llvm::is_contained(existing->second.transferCreateOps,
                            op.getOperation())) {
      existing->second.transferCreateOps.push_back(op.getOperation());
    }
    // Duplicate transfers for the same PipeKey can arise from cloned regions.
    // Collective is the stronger contract and must be preserved.
    if (isCollectiveTransfer(contract)) {
      existing->second.transferContract = PipeTransferContract::Collective;
    }
  };
  mod.walk([&](Operation *op) {
    if (auto postOp = dyn_cast<PipeTransferPostOp>(op)) {
      addTransfer(findPipeTransferCreateForTransfer(postOp.getTransfer()));
      return;
    }
    if (auto sendOp = dyn_cast<PipeTransferSendOp>(op)) {
      addTransfer(findPipeTransferCreateForTransfer(sendOp.getTransfer()));
    }
  });
  return pipeInfo;
}

static LogicalResult
emitUntraceableCollectiveDestinationAddress(Operation *op) {
  return op->emitError()
         << "collective pipe destination address could not be "
            "determined statically; TT-Metal NoC multicast requires one "
            "statically proven destination SRAM address for all receivers";
}

static LogicalResult addStaticCoordinates(ArrayRef<OpFoldResult> mixedOffsets,
                                          SmallVectorImpl<int64_t> &coordinates,
                                          unsigned rank) {
  if (coordinates.empty()) {
    coordinates.assign(rank, 0);
  }
  if (coordinates.size() != rank || mixedOffsets.size() != rank) {
    return failure();
  }

  for (auto [coordinate, mixedOffset] :
       llvm::zip_equal(coordinates, mixedOffsets)) {
    std::optional<int64_t> offset = getConstantIntValue(mixedOffset);
    if (!offset.has_value()) {
      return failure();
    }
    coordinate += *offset;
  }
  return success();
}

/// Return the static tile offset within the receiver DFB for a receive
/// destination. Collective lowering has one sender-visible address-table entry
/// per pipe because NoC multicast writes one destination SRAM address to every
/// receiver.
static FailureOr<int64_t> getStaticDestinationTileOffset(Value dst) {
  Value view = traceUnrealizedCasts(dst);
  SmallVector<int64_t> coordinates;
  RankedTensorType rootType;
  bool sawOffset = false;

  while (true) {
    view = traceUnrealizedCasts(view);
    if (auto extract = view.getDefiningOp<tensor::ExtractOp>()) {
      auto tensorType =
          mlir::dyn_cast<RankedTensorType>(extract.getTensor().getType());
      if (!tensorType) {
        return failure();
      }
      SmallVector<OpFoldResult> mixedIndices;
      for (Value index : extract.getIndices()) {
        mixedIndices.push_back(index);
      }
      if (failed(addStaticCoordinates(mixedIndices, coordinates,
                                      tensorType.getRank()))) {
        return failure();
      }
      sawOffset = true;
      view = extract.getTensor();
      continue;
    }
    if (auto attach = view.getDefiningOp<AttachCBOp>()) {
      view = attach.getTensor();
      continue;
    }

    auto slice = view.getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice) {
      rootType = mlir::dyn_cast<RankedTensorType>(view.getType());
      break;
    }

    auto sourceType =
        mlir::dyn_cast<RankedTensorType>(slice.getSource().getType());
    if (!sourceType) {
      return failure();
    }

    if (failed(addStaticCoordinates(slice.getMixedOffsets(), coordinates,
                                    sourceType.getRank()))) {
      return failure();
    }
    sawOffset = true;
    view = slice.getSource();
  }

  if (!sawOffset) {
    return 0;
  }
  if (!rootType ||
      rootType.getRank() != static_cast<int64_t>(coordinates.size())) {
    return failure();
  }

  int64_t linearOffset = 0;
  for (auto [coordinate, dim] :
       llvm::zip_equal(coordinates, rootType.getShape())) {
    if (dim == ShapedType::kDynamic) {
      return failure();
    }
    linearOffset = linearOffset * dim + coordinate;
  }
  return linearOffset;
}

static FailureOr<int64_t> getTensorTileCount(RankedTensorType tensorType) {
  int64_t tileCount = 1;
  for (int64_t dimension : tensorType.getShape()) {
    if (dimension == ShapedType::kDynamic) {
      return failure();
    }
    tileCount *= dimension;
  }
  return tileCount;
}

static FailureOr<int64_t>
getReceiverSlotSpanBlocks(Value dst, CircularBufferType dfbType) {
  auto reserveOp = findCBReserveForPipeReceive(dst);
  if (!reserveOp) {
    return failure();
  }
  auto reserveType =
      mlir::dyn_cast<RankedTensorType>(reserveOp.getResult().getType());
  if (!reserveType) {
    return failure();
  }

  auto dfbBlockType =
      RankedTensorType::get(dfbType.getShape(), dfbType.getElementType());
  FailureOr<int64_t> reserveTileCount = getTensorTileCount(reserveType);
  FailureOr<int64_t> dfbBlockTileCount = getTensorTileCount(dfbBlockType);
  if (failed(reserveTileCount) || failed(dfbBlockTileCount) ||
      *dfbBlockTileCount <= 0) {
    return failure();
  }
  return (*reserveTileCount + *dfbBlockTileCount - 1) / *dfbBlockTileCount;
}

static LogicalResult addPipeReceiver(PipeGraph &graph, Operation *op,
                                     PipeType pipeType,
                                     PipeTransferContract transferContract,
                                     ArrayRef<Operation *> transferCreateOps,
                                     Value dst) {
  Value dstDFB = getAttachedCB(dst);
  if (!dstDFB) {
    return op->emitError("pipe receive destination is not attached to a DFB");
  }
  auto dfbType = mlir::dyn_cast<CircularBufferType>(dstDFB.getType());
  if (!dfbType) {
    return op->emitError("pipe receive destination is not attached to a DFB");
  }

  std::optional<int64_t> dfbIndex = getCBIndex(dstDFB);
  if (!dfbIndex.has_value()) {
    return op->emitError("could not trace pipe receiver to a DFB binding");
  }

  bool hasStaticTileOffset = true;
  int64_t staticTileOffset = 0;
  FailureOr<int64_t> offset = getStaticDestinationTileOffset(dst);
  if (failed(offset)) {
    if (isCollectiveTransfer(transferContract)) {
      return emitUntraceableCollectiveDestinationAddress(op);
    }
    hasStaticTileOffset = false;
  } else {
    staticTileOffset = *offset;
  }

  FailureOr<int64_t> slotSpanBlocks = getReceiverSlotSpanBlocks(dst, dfbType);
  if (failed(slotSpanBlocks)) {
    return op->emitError("could not determine receiver DFB reserve span");
  }
  auto reserveOp = findCBReserveForPipeReceive(dst);
  assert(reserveOp && "reserve span computation already traced reserve op");

  return graph.addReceiverDFB(
      pipeType.getSrcX(), pipeType.getSrcY(), pipeType.getDstStartX(),
      pipeType.getDstStartY(), pipeType.getDstEndX(), pipeType.getDstEndY(),
      pipeType.getPipeNetId(), *dfbIndex, dfbType, hasStaticTileOffset,
      staticTileOffset, *slotSpanBlocks, reserveOp.getOperation(),
      transferContract, transferCreateOps, dfbType.getBlockCount(),
      op->getLoc());
}

FailureOr<PipeGraph> PipeGraph::build(ModuleOp mod) {
  PipeGraph graph;
  llvm::MapVector<PipeKey, PipeTransferInfo> transferInfos =
      collectPipeTransferInfo(mod);

  WalkResult walkResult = mod.walk([&](PipeTransferPostOp postOp) {
    auto createOp = findPipeTransferCreateForTransfer(postOp.getTransfer());
    if (!createOp) {
      postOp.emitError(
          "pipe transfer post must reference a transfer derived from "
          "ttl.pipe_transfer.create");
      return WalkResult::interrupt();
    }
    auto pipeType = mlir::cast<PipeType>(createOp.getPipe().getType());
    PipeKey key = getPipeKey(pipeType);
    auto pipeInfoIt = transferInfos.find(key);
    if (pipeInfoIt == transferInfos.end()) {
      postOp.emitError(
          "pipe transfer post must reference a known pipe transfer");
      return WalkResult::interrupt();
    }
    const PipeTransferInfo &pipeInfo = pipeInfoIt->second;
    if (failed(addPipeReceiver(graph, postOp, pipeType,
                               pipeInfo.transferContract,
                               pipeInfo.transferCreateOps, postOp.getDst()))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return failure();
  }

  PipeGraphAnalysisState analysisState;
  if (failed(collectLaunchNodeDomains(mod, analysisState))) {
    return failure();
  }

  if (failed(graph.assignReceiverSlotIndices(mod, analysisState))) {
    return failure();
  }

  if (failed(graph.verifyReceiverDFBBlockCounts())) {
    return failure();
  }

  graph.rebuildEndpointGraph();
  if (failed(graph.provePipeOnlyReceiverStreams(mod, analysisState))) {
    return failure();
  }
  graph.hasAnalyzedLaunchGrid = analysisState.hasLaunchGrid;
  graph.operationLaunchDomains =
      std::move(analysisState.operationLaunchDomains);
  graph.dfbReleaseOwners = std::move(analysisState.dfbReleaseOwners);
  return std::move(graph);
}

} // namespace mlir::tt::ttl
