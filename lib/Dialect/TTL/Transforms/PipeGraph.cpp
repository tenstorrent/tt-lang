// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeGraph.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/PipeTransferAnalysis.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>

#define DEBUG_TYPE "ttl-pipe-graph"

namespace mlir::tt::ttl {

/// Analysis facts and operation indexes used while constructing PipeGraph.
struct PipeGraphAnalysisState : LaunchNodeDomainState {
  llvm::DenseMap<Operation *, LaunchNodeDomain> operationLaunchDomains;
  llvm::DenseMap<Operation *, std::unique_ptr<DFBAcquireReleaseIndex>>
      dfbLifecycles;
  SmallVector<Operation *> transferProtocolOps;
  SmallVector<PipeTransferPostOp> receiverPosts;
  llvm::DenseMap<Operation *, SmallVector<PipeTransferWaitOp>>
      receiveWaitsByPost;
  llvm::DenseMap<int64_t, SmallVector<PipeTransferPostOp>>
      receiverPostsByStream;
  llvm::DenseMap<int64_t, SmallVector<CBPushOp>> pushesByStream;
  llvm::SmallPtrSet<Operation *, 16> pipeRecordControlOps;
  llvm::DenseMap<Operation *, LaunchNodeDomain> pipeRecordIfThenDomains;
  llvm::DenseMap<Operation *, PipeNetRecordLoop> pipeRecordLoops;
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
  options.computeRegionDomain =
      [&](Operation *op,
          unsigned regionNumber) -> std::optional<LaunchNodeDomain> {
    if (regionNumber != 0) {
      return std::nullopt;
    }
    auto domainIt = state.pipeRecordIfThenDomains.find(op);
    if (domainIt == state.pipeRecordIfThenDomains.end()) {
      return std::nullopt;
    }
    return domainIt->second;
  };
  solver.load<LaunchNodeDomainAnalysis>(state, options);
  if (failed(solver.initializeAndRun(mod))) {
    return failure();
  }
  for (func::FuncOp function : mod.getOps<func::FuncOp>()) {
    PlanningResult<std::unique_ptr<DFBAcquireReleaseIndex>> lifecycleResult =
        DFBAcquireReleaseIndex::create(function);
    if (lifecycleResult.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = lifecycleResult.getInvalidIR();
      diagnostic.operation->emitError(diagnostic.message);
      return failure();
    }
    assert(lifecycleResult.isPlanned() &&
           "DFB lifecycle indexing has no recoverable rejection");
    state.dfbLifecycles.try_emplace(function.getOperation(),
                                    std::move(lifecycleResult).takePlan());
  }
  return success();
}

template <typename AcquireOp>
static AcquireOp
findUniqueDFBReleaseIntervalOwner(Operation *release,
                                  PipeGraphAnalysisState &state) {
  func::FuncOp function = release->getParentOfType<func::FuncOp>();
  auto lifecycle = state.dfbLifecycles.find(function.getOperation());
  assert(lifecycle != state.dfbLifecycles.end() &&
         "every function has a DFB lifecycle index");
  ArrayRef<Operation *> owners =
      lifecycle->second->getReleaseIntervalOwners(release);
  if (owners.size() != 1) {
    return AcquireOp();
  }
  return dyn_cast<AcquireOp>(owners.front());
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

/// Return the physical DFB stream used to index receiver operations. Launch
/// domains distinguish receiver nodes that use the same DFB index.
static std::optional<int64_t> getReceiverDFBStreamKey(Value dfb) {
  return getCBIndex(dfb);
}

/// Record one receiver post in definition order and under its physical DFB
/// stream.
static void recordReceiverPost(PipeTransferPostOp postOp,
                               PipeGraphAnalysisState &state) {
  state.transferProtocolOps.push_back(postOp.getOperation());
  state.receiverPosts.push_back(postOp);
  std::optional<int64_t> maybeStreamKey =
      getReceiverDFBStreamKey(getAttachedCB(postOp.getDst()));
  if (maybeStreamKey) {
    state.receiverPostsByStream[*maybeStreamKey].push_back(postOp);
  }
}

/// Associate one receive wait with its unique possible receiver post.
static LogicalResult recordReceiveWait(PipeTransferWaitOp waitOp,
                                       PipeGraphAnalysisState &state,
                                       const PipeTransferIndex &transferIndex) {
  ArrayRef<Operation *> possiblePosts =
      transferIndex.getPossibleReceivePosts(waitOp);
  if (possiblePosts.size() != 1) {
    waitOp.emitError() << "requires exactly one possible receiver post; found "
                       << possiblePosts.size();
    return failure();
  }
  state.receiveWaitsByPost[possiblePosts.front()].push_back(waitOp);
  return success();
}

/// Collect protocol and receiver DFB operations once so graph analyses do not
/// rescan the module for every receiver.
static LogicalResult
collectPipeGraphOperations(ModuleOp mod, const PipeTransferIndex &transferIndex,
                           PipeGraphAnalysisState &state) {
  WalkResult walkResult =
      mod.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        LogicalResult recordResult = success();
        llvm::TypeSwitch<Operation *>(op)
            .Case<PipeTransferPostOp>([&](PipeTransferPostOp postOp) {
              recordReceiverPost(postOp, state);
            })
            .Case<PipeTransferSendOp>([&](PipeTransferSendOp sendOp) {
              state.transferProtocolOps.push_back(sendOp.getOperation());
            })
            .Case<PipeTransferWaitOp>([&](PipeTransferWaitOp waitOp) {
              recordResult = recordReceiveWait(waitOp, state, transferIndex);
            })
            .Case<CBPushOp>([&](CBPushOp pushOp) {
              std::optional<int64_t> maybeStreamKey =
                  getReceiverDFBStreamKey(pushOp.getCb());
              if (maybeStreamKey) {
                state.pushesByStream[*maybeStreamKey].push_back(pushOp);
              }
            });
        return failed(recordResult) ? WalkResult::interrupt()
                                    : WalkResult::advance();
      });
  return success(!walkResult.wasInterrupted());
}

/// Visit the operations associated with one receiver DFB stream.
template <typename OpTy, typename Callback>
static void forEachReceiverDFBStreamEvent(
    const llvm::DenseMap<int64_t, SmallVector<OpTy>> &eventsByStream,
    const PipeReceiverDFBKey &receiverDFB, Callback &&callback) {
  auto eventsIt = eventsByStream.find(receiverDFB.dfbIndex);
  if (eventsIt == eventsByStream.end()) {
    return;
  }
  for (OpTy event : eventsIt->second) {
    callback(event);
  }
}

static bool isPostForReceiverDFB(
    PipeTransferPostOp postOp, const PipeReceiverDFBKey &receiverDFB,
    const llvm::MapVector<Operation *, ReceiverDFBInfo> &receiverDFBByPost,
    PipeGraphAnalysisState &state) {
  auto receiverIt = receiverDFBByPost.find(postOp.getOperation());
  if (receiverIt == receiverDFBByPost.end() ||
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
    const llvm::MapVector<Operation *, ReceiverDFBInfo> &receiverDFBByPost) {
  auto receiverIt = receiverDFBByPost.find(postOp.getOperation());
  if (receiverIt == receiverDFBByPost.end()) {
    return std::nullopt;
  }
  return receiverIt->second.receiverSlotSpanBlocks;
}

static void
debugRejectPipeOnlyProducerStream(const PipeReceiverDFBKey &receiverDFB,
                                  llvm::StringRef reason) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeGraph: reject pipe-only producer stream for ";
    printReceiverDFB(llvm::dbgs(), receiverDFB);
    llvm::dbgs() << ": " << reason << "\n";
  });
}

static void
debugAcceptPipeOnlyProducerStream(const PipeReceiverDFBKey &receiverDFB) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeGraph: accept pipe-only producer stream for ";
    printReceiverDFB(llvm::dbgs(), receiverDFB);
    llvm::dbgs() << "\n";
  });
}

using ReceiverPostsByDFB =
    llvm::DenseMap<PipeReceiverDFBKey, SmallVector<PipeTransferPostOp>>;

/// Structured control that orders one receiver DFB reservation sequence.
struct ReceiverControlContext {
  Operation *function = nullptr;
  SmallVector<Block *> dynamicRegionBlocks;

  bool operator==(const ReceiverControlContext &rhs) const {
    return function == rhs.function &&
           dynamicRegionBlocks == rhs.dynamicRegionBlocks;
  }

  bool operator!=(const ReceiverControlContext &rhs) const {
    return !(*this == rhs);
  }
};

/// Return whether `op` adds no runtime branch at this receiver. PipeNet scopes
/// are structural; source and destination predicates are evaluated at the
/// receiver coordinate.
static bool isTransparentReceiverScope(Operation *op,
                                       PipeReceiverCoord receiver,
                                       const PipeGraphAnalysisState &state) {
  if (mlir::isa<PipeNetScopeOp>(op)) {
    return true;
  }
  if (auto ifDstOp = mlir::dyn_cast<IfDstOp>(op)) {
    auto pipeType = mlir::cast<PipeType>(ifDstOp.getPipe().getType());
    return knownLaunchNodeDomainContains(
        getPipeDestinationLaunchNodeDomain(pipeType, state.baseDomain),
        getLaunchNodeCoord(receiver));
  }
  if (auto ifSrcOp = mlir::dyn_cast<IfSrcOp>(op)) {
    auto pipeType = mlir::cast<PipeType>(ifSrcOp.getPipe().getType());
    return knownLaunchNodeDomainContains(
        getPipeSourceLaunchNodeDomain(pipeType), getLaunchNodeCoord(receiver));
  }
  return false;
}

/// Return the function and enclosing region blocks whose selection can vary at
/// runtime and therefore constrains DFB reservation order at this receiver.
static std::optional<ReceiverControlContext>
getReceiverControlContext(Operation *op, PipeReceiverCoord receiver,
                          const PipeGraphAnalysisState &analysisState) {
  ReceiverControlContext context;
  Operation *current = op;
  while (Block *block = current->getBlock()) {
    Operation *parent = block->getParentOp();
    if (auto function = mlir::dyn_cast_if_present<func::FuncOp>(parent)) {
      if (block != &function.getBody().front()) {
        return std::nullopt;
      }
      context.function = function;
      break;
    }
    // The graph already represents each PipeNet record separately. Ignoring
    // generated foreach control preserves the callback's order relative to
    // operations before and after it.
    if (analysisState.pipeRecordControlOps.contains(parent)) {
      current = parent;
      continue;
    }
    if (parent && isTransparentReceiverScope(parent, receiver, analysisState)) {
      current = parent;
      continue;
    }
    if (auto ifOp = mlir::dyn_cast_if_present<scf::IfOp>(parent)) {
      if (std::optional<bool> maybeSelected = evaluatePredicateAtLaunchNode(
              ifOp.getCondition(), getLaunchNodeCoord(receiver),
              analysisState)) {
        std::size_t selectedRegion = *maybeSelected ? 0 : 1;
        if (block->getParent()->getRegionNumber() != selectedRegion) {
          return std::nullopt;
        }
        current = parent;
        continue;
      }
    }
    context.dynamicRegionBlocks.push_back(block);
    if (!parent) {
      return std::nullopt;
    }
    current = parent;
  }
  if (!context.function) {
    return std::nullopt;
  }
  std::reverse(context.dynamicRegionBlocks.begin(),
               context.dynamicRegionBlocks.end());
  return context;
}

/// Return true when `before` precedes `after` in one receiver control context.
/// Projecting a node-selected wrapper into the enclosing block preserves the
/// per-node order without treating unrelated blocks as sequential.
static bool
isBeforeInReceiverControlContext(Operation *before, Operation *after,
                                 PipeReceiverCoord receiver,
                                 const PipeGraphAnalysisState &analysisState) {
  std::optional<ReceiverControlContext> maybeBeforeContext =
      getReceiverControlContext(before, receiver, analysisState);
  std::optional<ReceiverControlContext> maybeAfterContext =
      getReceiverControlContext(after, receiver, analysisState);
  if (!maybeBeforeContext || maybeBeforeContext != maybeAfterContext) {
    return false;
  }
  Operation *projectedBefore = before;
  for (Block *block = before->getBlock(); block;) {
    Operation *projectedAfter = after->getBlock() == block
                                    ? after
                                    : block->findAncestorOpInBlock(*after);
    if (projectedAfter) {
      return projectedBefore != projectedAfter &&
             projectedBefore->isBeforeInBlock(projectedAfter);
    }
    Operation *parent = block->getParentOp();
    if (!parent) {
      break;
    }
    projectedBefore = parent;
    block = parent->getBlock();
  }
  return false;
}

static bool hasMatchingReceiveWaitBeforePush(
    PipeTransferPostOp postOp, CBPushOp pushOp,
    const llvm::DenseMap<Operation *, SmallVector<PipeTransferWaitOp>>
        &waitsByPost,
    PipeReceiverCoord receiver, const PipeGraphAnalysisState &analysisState) {
  auto waitIt = waitsByPost.find(postOp.getOperation());
  if (waitIt == waitsByPost.end()) {
    return false;
  }
  return llvm::any_of(waitIt->second, [&](PipeTransferWaitOp waitOp) {
    return isBeforeInReceiverControlContext(postOp, waitOp, receiver,
                                            analysisState) &&
           isBeforeInReceiverControlContext(waitOp, pushOp, receiver,
                                            analysisState);
  });
}

/// Group posts by physical DFB because its writers share one reservation ring.
static ReceiverPostsByDFB collectReceiverPostsByDFB(
    const PipeTransferIndex &transferIndex,
    const llvm::MapVector<Operation *, ReceiverDFBInfo> &receiverDFBByPost,
    PipeGraphAnalysisState &analysisState) {
  ReceiverPostsByDFB postsByReceiverDFB;
  for (PipeTransferPostOp postOp : analysisState.receiverPosts) {
    PipeTransferCreateOp createOp =
        transferIndex.getTransferCreate(postOp.getOperation());
    FailureOr<PipeReference> pipeRef =
        getPipeReference(postOp, createOp.getPipe());
    assert(succeeded(pipeRef) &&
           "pipe transfer graph validated pipe references");
    auto receiverIt = receiverDFBByPost.find(postOp.getOperation());
    if (receiverIt == receiverDFBByPost.end()) {
      continue;
    }
    LaunchNodeDomain postDomain =
        lookupOperationLaunchDomain(postOp.getOperation(), analysisState);
    for (PipeType pipeType :
         getPipeTypesFromReference(postOp.getContext(), *pipeRef)) {
      PipeKey pipe = getPipeKey(pipeType);
      pipe.forEachReceiver([&](PipeReceiverCoord receiver) {
        if (postDomain.known && !knownLaunchNodeDomainContains(
                                    postDomain, getLaunchNodeCoord(receiver))) {
          return;
        }
        PipeReceiverDFBKey receiverDFB{receiver, receiverIt->second.dfbIndex};
        postsByReceiverDFB[receiverDFB].push_back(postOp);
      });
    }
  }
  return postsByReceiverDFB;
}

static void debugRejectReceiverSchedule(const PipeReceiverDFBKey &receiverDFB,
                                        llvm::StringRef reason) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeGraph: reject computed receiver schedule for ";
    printReceiverDFB(llvm::dbgs(), receiverDFB);
    llvm::dbgs() << ": " << reason << "\n";
  });
}

/// Lexical order proves reservation order only when all posts are in the same
/// function and the same runtime-varying enclosing regions. PipeNet scopes and
/// source or destination conditions known true at this receiver are ignored.
static std::optional<ReceiverControlContext>
getProvenReceiverScheduleContext(const PipeReceiverDFBKey &receiverDFB,
                                 ArrayRef<PipeTransferPostOp> posts,
                                 const PipeGraphAnalysisState &analysisState) {
  if (posts.empty()) {
    debugRejectReceiverSchedule(receiverDFB, "no receiver posts");
    return std::nullopt;
  }

  std::optional<ReceiverControlContext> maybeControlContext =
      getReceiverControlContext(posts.front(), receiverDFB.receiver,
                                analysisState);
  if (!maybeControlContext) {
    debugRejectReceiverSchedule(receiverDFB,
                                "receiver control cannot be evaluated");
    return std::nullopt;
  }
  for (PipeTransferPostOp postOp : posts) {
    if (getReceiverControlContext(postOp, receiverDFB.receiver,
                                  analysisState) != maybeControlContext) {
      debugRejectReceiverSchedule(
          receiverDFB,
          "receiver posts do not share one sequential control context");
      return std::nullopt;
    }
  }
  return maybeControlContext;
}

/// Next physical DFB block selected by producer reservation order.
struct ReceiverProducerState {
  int64_t nextSlot = 0;
};

} // namespace

static std::optional<std::uint64_t>
getConcreteTransferExecutionCount(Operation *op, LaunchNodeCoord coord,
                                  const PipeReference &pipeRef,
                                  PipeGraphAnalysisState &analysisState);

static InFlightDiagnostic
emitReceiverReservationPastDFBEnd(const ReceiverDFBInfo &receiverInfo,
                                  int64_t slot) {
  return emitError(receiverInfo.loc)
         << "pipe receiver DFB reservation sequence reaches slot " << slot
         << " with a span of " << receiverInfo.receiverSlotSpanBlocks
         << " blocks, which advances the DFB producer write pointer past "
            "block_count="
         << receiverInfo.blockCount
         << "; increase block_count or change the reservation sizes";
}

static FailureOr<int64_t>
assignReceiverPhysicalSlot(const ReceiverDFBInfo &receiverInfo,
                           ReceiverProducerState &producerState) {
  int64_t span = receiverInfo.receiverSlotSpanBlocks;
  assert(span > 0 && span <= receiverInfo.blockCount &&
         "verified receiver reserve span must fit the DFB");

  int64_t slot = producerState.nextSlot;
  if (span > receiverInfo.blockCount - slot) {
    return emitReceiverReservationPastDFBEnd(receiverInfo, slot);
  }

  producerState.nextSlot =
      slot + span == receiverInfo.blockCount ? 0 : slot + span;
  return slot;
}

static int64_t advanceReceiverSlot(int64_t slot, int64_t stride,
                                   int64_t blockCount) {
  assert(slot >= 0 && slot < blockCount && stride >= 0 && stride < blockCount &&
         "invalid receiver slot recurrence");
  return stride >= blockCount - slot ? stride - (blockCount - slot)
                                     : slot + stride;
}

/// Verify that every reachable reservation advances to or before the physical
/// DFB end. TT-Metal permits the write pointer to return to the first block
/// only when an advance reaches the end exactly.
static LogicalResult
verifyReceiverReservationSequence(const ReceiverAddressSequenceProof &sequence,
                                  const ReceiverDFBInfo &receiverInfo) {
  assert(sequence.recurrence && "sequence validation requires a recurrence");
  const ReceiverAddressRecurrence &recurrence = *sequence.recurrence;
  int64_t period = recurrence.blockCount /
                   std::gcd(recurrence.blockCount, recurrence.repeatStride);
  std::uint64_t occurrenceCount = static_cast<std::uint64_t>(period);
  if (sequence.executionCount) {
    occurrenceCount = std::min(occurrenceCount, *sequence.executionCount);
  }

  int64_t slot = recurrence.initialSlot;
  for (std::uint64_t occurrence = 0; occurrence < occurrenceCount;
       ++occurrence) {
    if (receiverInfo.receiverSlotSpanBlocks > recurrence.blockCount - slot) {
      emitReceiverReservationPastDFBEnd(receiverInfo, slot);
      return failure();
    }
    slot = advanceReceiverSlot(slot, recurrence.repeatStride,
                               recurrence.blockCount);
  }
  return success();
}

LogicalResult PipeGraph::assignReceiverAddressSequences(
    ModuleOp mod, const PipeTransferIndex &transferIndex,
    PipeGraphAnalysisState &analysisState) {
  ReceiverPostsByDFB postsByReceiverDFB = collectReceiverPostsByDFB(
      transferIndex, receiverDFBByPost, analysisState);
  llvm::DenseMap<PipeReceiverDFBKey, std::optional<ReceiverControlContext>>
      scheduleContextByReceiverDFB;
  for (const auto &receiverEntry : receiverDFBByPost) {
    Operation *postOperation = receiverEntry.first;
    const ReceiverDFBInfo &receiverInfo = receiverEntry.second;
    int64_t receiverDFBIndex = receiverInfo.dfbIndex;
    auto postOp = llvm::cast<PipeTransferPostOp>(postOperation);
    PipeTransferCreateOp createOp =
        transferIndex.getTransferCreate(postOp.getOperation());
    FailureOr<PipeReference> pipeRef =
        getPipeReference(postOp, createOp.getPipe());
    assert(succeeded(pipeRef) &&
           "pipe transfer graph validated pipe references");
    for (PipeType pipeType :
         getPipeTypesFromReference(postOp.getContext(), *pipeRef)) {
      PipeKey pipe = getPipeKey(pipeType);
      pipe.forEachReceiver([&](PipeReceiverCoord receiver) {
        PipeReceiverDFBKey receiverDFB{receiver, receiverDFBIndex};
        auto postIt = postsByReceiverDFB.find(receiverDFB);
        ArrayRef<PipeTransferPostOp> posts;
        if (postIt != postsByReceiverDFB.end()) {
          posts = postIt->second;
        }
        scheduleContextByReceiverDFB.try_emplace(
            receiverDFB, getProvenReceiverScheduleContext(receiverDFB, posts,
                                                          analysisState));
      });
    }
  }

  // Recurrence facts accumulated for one endpoint during the producer walk.
  struct EndpointSlotAssignment {
    bool valid = true;
    std::optional<int64_t> initialSlot;
    std::optional<std::uint64_t> executionCount;
  };

  llvm::DenseMap<PipeReceiverDFBKey, ReceiverProducerState>
      producerStateByReceiverDFB;
  llvm::DenseMap<PipeReceiverDFBKey, llvm::DenseMap<Operation *, int64_t>>
      slotByReceiverReserve;
  llvm::DenseMap<PipeReceiverEndpointId, EndpointSlotAssignment>
      assignmentByEndpoint;
  auto processPost =
      [&](PipeTransferPostOp postOp, LaunchNodeCoord coord,
          ArrayRef<ActivePipeNetRecord> activeRecords) -> LogicalResult {
    LaunchNodeDomain postDomain =
        lookupOperationLaunchDomain(postOp.getOperation(), analysisState);
    if (postDomain.known && !knownLaunchNodeDomainContains(postDomain, coord)) {
      return success();
    }
    auto receiverReserveOp = findCBReserveForPipeReceive(postOp.getDst());
    assert(receiverReserveOp &&
           "receiver post must trace to the reserve recorded by PipeGraph");
    PipeTransferCreateOp createOp =
        transferIndex.getTransferCreate(postOp.getOperation());
    FailureOr<PipeReference> pipeRef =
        getPipeReference(postOp, createOp.getPipe());
    assert(succeeded(pipeRef) &&
           "pipe transfer graph validated pipe references");

    SmallVector<PipeTransferNodeId, 1> selectedTransferNodeIds;
    ArrayRef<PipeTransferNodeId> transferNodeIds;
    Operation *selectedRecordLoop = nullptr;
    if (pipeRef->isStatic()) {
      transferNodeIds =
          getPipeTransferNodeIdsForProtocolOp(postOp.getOperation());
    } else {
      selectedRecordLoop = pipeRef->getSelectedOperation()->getParentOp();
      while (selectedRecordLoop &&
             !analysisState.pipeRecordLoops.contains(selectedRecordLoop)) {
        selectedRecordLoop = selectedRecordLoop->getParentOp();
      }
      assert(selectedRecordLoop &&
             "selected pipe operation must be nested in its record loop");
      std::optional<std::uint64_t> activeRecordIndex =
          getActivePipeNetRecordIndex(activeRecords, selectedRecordLoop);
      assert(activeRecordIndex &&
             "selected pipe operation must execute in an active record");
      auto transferIt = transferNodeIdByProtocolOpAndRecord.find(
          std::make_pair(postOp.getOperation(), *activeRecordIndex));
      assert(transferIt != transferNodeIdByProtocolOpAndRecord.end() &&
             "selected receiver record must have a transfer node");
      selectedTransferNodeIds.push_back(transferIt->second);
      transferNodeIds = selectedTransferNodeIds;
    }
    if (transferNodeIds.empty()) {
      return success();
    }

    PipeReceiverCoord receiver{coord.x, coord.y};
    for (PipeTransferNodeId transferNodeId : transferNodeIds) {
      const PipeTransferNode &transferNode =
          getPipeTransferNode(transferNodeId);
      const PipeKey &pipeKey = transferNode.pipe;
      if (!pipeKey.containsReceiver(receiver)) {
        continue;
      }
      auto endpointIt = llvm::find_if(
          transferNode.receiverEndpoints,
          [&](PipeReceiverEndpointId endpointId) {
            return getPipeReceiverEndpoint(endpointId).receiver == receiver;
          });
      assert(endpointIt != transferNode.receiverEndpoints.end() &&
             "pipe transfer node is missing a receiver endpoint");
      const PipeReceiverEndpoint &endpoint =
          getPipeReceiverEndpoint(*endpointIt);
      const ReceiverDFBInfo &receiverInfo = endpoint.receiverDFBInfo;
      PipeReceiverDFBKey receiverDFB{receiver, receiverInfo.dfbIndex};
      EndpointSlotAssignment &endpointAssignment =
          assignmentByEndpoint[*endpointIt];
      if (!scheduleContextByReceiverDFB.lookup(receiverDFB)) {
        endpointAssignment.valid = false;
        continue;
      }
      auto &slotByReserve = slotByReceiverReserve[receiverDFB];
      auto reserveIt = slotByReserve.find(receiverReserveOp.getOperation());
      int64_t slot = 0;
      bool reserveRepeatsPerRecord =
          selectedRecordLoop && selectedRecordLoop->isProperAncestor(
                                    receiverReserveOp.getOperation());
      // A reserve inside the selected record loop executes once per matching
      // record. A reserve outside it retains one slot across those callbacks.
      if (reserveRepeatsPerRecord || reserveIt == slotByReserve.end()) {
        FailureOr<int64_t> assignedSlot = assignReceiverPhysicalSlot(
            receiverInfo, producerStateByReceiverDFB[receiverDFB]);
        if (failed(assignedSlot)) {
          return failure();
        }
        slot = *assignedSlot;
        if (!reserveRepeatsPerRecord) {
          slotByReserve[receiverReserveOp.getOperation()] = slot;
        }
      } else {
        slot = reserveIt->second;
      }
      if (!postDomain.known) {
        endpointAssignment.valid = false;
        continue;
      }
      std::optional<std::uint64_t> maybeExecutionCount =
          getConcreteTransferExecutionCount(postOp.getOperation(), coord,
                                            *pipeRef, analysisState);
      if (endpointAssignment.initialSlot) {
        endpointAssignment.valid = false;
        continue;
      }
      endpointAssignment.initialSlot = slot;
      endpointAssignment.executionCount = maybeExecutionCount;
    }
    return success();
  };

  llvm::SmallSetVector<PipeReceiverCoord, 8> receiverCoords;
  for (const PipeReceiverEndpoint &endpoint : pipeReceiverEndpoints) {
    receiverCoords.insert(endpoint.receiver);
  }
  for (func::FuncOp funcOp : mod.getOps<func::FuncOp>()) {
    for (PipeReceiverCoord receiver : receiverCoords) {
      LaunchNodeCoord coord = getLaunchNodeCoord(receiver);
      WalkResult walkResult = walkPipeNetOpsInProgramOrder(
          funcOp, coord,
          [&](Operation *op) -> std::optional<PipeNetRecordLoop> {
            auto recordLoopIt = analysisState.pipeRecordLoops.find(op);
            return recordLoopIt == analysisState.pipeRecordLoops.end()
                       ? std::nullopt
                       : std::optional<PipeNetRecordLoop>(recordLoopIt->second);
          },
          [&](Operation *op, ArrayRef<ActivePipeNetRecord> activeRecords,
              std::optional<std::uint64_t>) {
            if (auto postOp = dyn_cast<PipeTransferPostOp>(op)) {
              return failed(processPost(postOp, coord, activeRecords))
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            }
            return WalkResult::advance();
          });
      if (walkResult.wasInterrupted()) {
        return failure();
      }
    }
  }

  for (PipeReceiverEndpoint &endpoint : pipeReceiverEndpoints) {
    auto assignmentIt = assignmentByEndpoint.find(endpoint.id);
    if (assignmentIt == assignmentByEndpoint.end() ||
        !assignmentIt->second.valid || !assignmentIt->second.initialSlot) {
      continue;
    }
    auto producerStateIt =
        producerStateByReceiverDFB.find(endpoint.receiverDFB);
    if (producerStateIt == producerStateByReceiverDFB.end()) {
      continue;
    }
    const ReceiverDFBInfo &receiverInfo = endpoint.receiverDFBInfo;
    ReceiverAddressRecurrence recurrence{
        *assignmentIt->second.initialSlot,
        producerStateIt->second.nextSlot,
        receiverInfo.blockCount,
    };
    ReceiverAddressSequenceProof sequence;
    sequence.executionCount = assignmentIt->second.executionCount;
    sequence.recurrence = recurrence;
    if (failed(verifyReceiverReservationSequence(sequence, receiverInfo))) {
      return failure();
    }
    endpoint.addressSequence = std::move(sequence);
  }
  return success();
}

LogicalResult PipeGraph::verifyCollectiveReceiverAddresses() const {
  for (const PipeTransferNode &transferNode : pipeTransferNodes) {
    const PipeKey &pipe = transferNode.pipe;
    // Receiver-published lowering stores one address for a multicast send, so
    // every receiver must use that address when computed lowering declines.
    if (!pipe.hasSingleReceiver() &&
        !getProvenReceiverAddressEndpoint(transferNode.id)) {
      auto diag = emitError(transferNode.sendOp->getLoc())
                  << "collective pipe receiver address sequences are not "
                     "proven equal for every transfer occurrence; TT-Metal "
                     "NoC multicast requires one destination SRAM address for "
                     "all receivers";
      for (PipeReceiverEndpointId endpointId :
           getPipeReceiverEndpoints(transferNode.id)) {
        const PipeReceiverEndpoint &endpoint =
            getPipeReceiverEndpoint(endpointId);
        const PipeReceiverDFBNode &receiverDFBNode =
            getReceiverDFBNode(endpoint.receiverDFBNode);
        if (!receiverDFBNode.hasProvenPipeOnlyProducerStream) {
          diag.attachNote(endpoint.receiverDFBInfo.loc)
              << "receiver core_x=" << endpoint.receiver.x
              << ", core_y=" << endpoint.receiver.y << " uses DFB "
              << endpoint.receiverDFBInfo.dfbIndex << ": "
              << receiverDFBNode.pipeOnlyProducerStreamFailureReason;
          break;
        }
        if (endpoint.addressSequence.getKind() ==
            ReceiverAddressSequenceProofKind::FullyDynamic) {
          diag.attachNote(endpoint.receiverDFBInfo.loc)
              << "receiver core_x=" << endpoint.receiver.x
              << ", core_y=" << endpoint.receiver.y
              << " has no proven receiver address sequence";
          break;
        }
      }
      return failure();
    }
  }
  return success();
}

/// Verify that every receiver destination can hold the sender's DFB block.
static LogicalResult
verifyTransferPayloadCompatibility(const PipeTransferNode &transferNode) {
  auto sendOp = llvm::cast<PipeTransferSendOp>(transferNode.sendOp);
  auto sourceDFBType =
      mlir::cast<CircularBufferType>(sendOp.getSrc().getType());
  int64_t sourceElementCount = sourceDFBType.getElementsPerBlock();

  for (Operation *postOperation : transferNode.receiverPostOps) {
    auto postOp = llvm::cast<PipeTransferPostOp>(postOperation);
    auto destinationType =
        mlir::dyn_cast<RankedTensorType>(postOp.getDst().getType());
    if (!destinationType || !destinationType.hasStaticShape()) {
      auto diag = postOp.emitError(
          "cannot prove that the pipe receiver destination holds the sender "
          "DFB block");
      diag.attachNote(sendOp.getLoc()) << "corresponding pipe send is here";
      return failure();
    }
    if (destinationType.getElementType() != sourceDFBType.getElementType() ||
        destinationType.getNumElements() < sourceElementCount) {
      auto diag = postOp.emitError()
                  << "pipe receiver destination " << destinationType
                  << " cannot hold sender DFB block with " << sourceElementCount
                  << " element(s) of type " << sourceDFBType.getElementType();
      diag.attachNote(sendOp.getLoc()) << "corresponding pipe send is here";
      return failure();
    }
  }
  return success();
}

LogicalResult PipeGraph::provePipeOnlyReceiverProducerStreams(
    PipeGraphAnalysisState &analysisState) {
  for (PipeReceiverDFBNode &node : receiverDFBNodes) {
    PipeReceiverDFBKey receiverDFB = node.receiverDFB;
    LaunchNodeDomain receiverDomain =
        getSingleLaunchNodeDomain(getLaunchNodeCoord(receiverDFB.receiver));

    SmallVector<PipeTransferPostOp> posts;
    forEachReceiverDFBStreamEvent(
        analysisState.receiverPostsByStream, receiverDFB,
        [&](PipeTransferPostOp postOp) {
          if (isPostForReceiverDFB(postOp, receiverDFB, receiverDFBByPost,
                                   analysisState)) {
            posts.push_back(postOp);
          }
        });
    if (posts.empty()) {
      debugRejectPipeOnlyProducerStream(receiverDFB,
                                        "no matching receiver posts");
      node.pipeOnlyProducerStreamFailureReason = "no matching receiver posts";
      continue;
    }

    bool valid = true;
    auto reject = [&](llvm::StringRef reason) {
      debugRejectPipeOnlyProducerStream(receiverDFB, reason);
      node.pipeOnlyProducerStreamFailureReason = reason;
      valid = false;
    };
    llvm::DenseSet<Operation *> postsWithPush;

    forEachReceiverDFBStreamEvent(
        analysisState.pushesByStream, receiverDFB, [&](CBPushOp pushOp) {
          if (!valid) {
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
          std::optional<int64_t> maybePushedBlocks =
              getDFBTransactionBlockCount(pushOp);
          if (!maybePushedBlocks) {
            reject("push block count is not a whole DFB block count");
            return;
          }
          CBReserveOp reserveOp =
              findUniqueDFBReleaseIntervalOwner<CBReserveOp>(
                  pushOp.getOperation(), analysisState);
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
            if (!hasMatchingReceiveWaitBeforePush(
                    postOp, pushOp, analysisState.receiveWaitsByPost,
                    receiverDFB.receiver, analysisState)) {
              reject("post has no receive wait before push");
              return;
            }
            std::optional<int64_t> maybeSpan =
                getReceiverSlotSpanBlocksForPost(postOp, receiverDFBByPost);
            if (!maybeSpan) {
              reject("post has no receiver slot span");
              return;
            }
            postedBlocks += *maybeSpan;
            if (!postsWithPush.insert(postOp.getOperation()).second) {
              reject("post is consumed by more than one push");
              return;
            }
          }
          if (*maybePushedBlocks != postedBlocks) {
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

    node.hasProvenPipeOnlyProducerStream = true;
    node.pipeOnlyProducerStreamFailureReason.clear();
    debugAcceptPipeOnlyProducerStream(receiverDFB);
  }
  return success();
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

const DFBAcquireReleaseIndex &
PipeGraph::getDFBAcquireReleaseIndex(Operation *operation) const {
  func::FuncOp function = operation->getParentOfType<func::FuncOp>();
  assert(function && "DFB lifecycle operation must be inside a function");
  auto lifecycle = dfbLifecycles.find(function.getOperation());
  assert(lifecycle != dfbLifecycles.end() &&
         "every function must have a DFB lifecycle index");
  return *lifecycle->second;
}

static std::optional<int64_t>
getReceiverAddressByteOffset(const PipeReceiverEndpoint &endpoint,
                             int64_t slot) {
  const ReceiverDFBInfo &info = endpoint.receiverDFBInfo;
  if (!info.hasStaticTileOffset || slot < 0 || slot >= info.blockCount) {
    return std::nullopt;
  }
  auto tileType = dyn_cast<ttcore::TileType>(info.dfbType.getElementType());
  if (!tileType) {
    return std::nullopt;
  }
  std::optional<int64_t> maybeBlockStrideBytes =
      llvm::checkedMul(info.dfbType.getElementsPerBlock(),
                       static_cast<int64_t>(tileType.getSizeBytes()));
  std::optional<int64_t> maybeStaticByteOffset = llvm::checkedMul(
      info.staticTileOffset, static_cast<int64_t>(tileType.getSizeBytes()));
  if (!maybeBlockStrideBytes || !maybeStaticByteOffset) {
    return std::nullopt;
  }
  return llvm::checkedMulAdd(slot, *maybeBlockStrideBytes,
                             *maybeStaticByteOffset);
}

static std::optional<std::uint64_t>
getReceiverSequencePeriod(const ReceiverAddressSequenceProof &sequence) {
  if (sequence.getKind() == ReceiverAddressSequenceProofKind::FullyDynamic ||
      sequence.recurrence->blockCount <= 0 ||
      sequence.recurrence->initialSlot < 0 ||
      sequence.recurrence->initialSlot >= sequence.recurrence->blockCount ||
      sequence.recurrence->repeatStride < 0 ||
      sequence.recurrence->repeatStride >= sequence.recurrence->blockCount) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(
      sequence.recurrence->blockCount /
      std::gcd(sequence.recurrence->blockCount,
               sequence.recurrence->repeatStride));
}

static std::optional<std::uint64_t>
getCombinedReceiverSequencePeriod(const ReceiverAddressSequenceProof &lhs,
                                  const ReceiverAddressSequenceProof &rhs) {
  std::optional<std::uint64_t> maybeLhsPeriod = getReceiverSequencePeriod(lhs);
  std::optional<std::uint64_t> maybeRhsPeriod = getReceiverSequencePeriod(rhs);
  if (!maybeLhsPeriod || !maybeRhsPeriod) {
    return std::nullopt;
  }
  std::uint64_t commonDivisor = std::gcd(*maybeLhsPeriod, *maybeRhsPeriod);
  return llvm::checkedMulUnsigned(*maybeLhsPeriod / commonDivisor,
                                  *maybeRhsPeriod);
}

static bool
havePointwiseEqualReceiverAddressSequences(const PipeReceiverEndpoint &lhs,
                                           const PipeReceiverEndpoint &rhs) {
  const ReceiverAddressSequenceProof &lhsSequence = lhs.addressSequence;
  const ReceiverAddressSequenceProof &rhsSequence = rhs.addressSequence;
  if (lhsSequence.getKind() == ReceiverAddressSequenceProofKind::FullyDynamic ||
      rhsSequence.getKind() == ReceiverAddressSequenceProofKind::FullyDynamic ||
      lhsSequence.executionCount != rhsSequence.executionCount) {
    return false;
  }
  if (lhsSequence.executionCount && *lhsSequence.executionCount == 0) {
    return true;
  }
  // Each finalized DFB index is bound to one runtime base common argument.
  if (lhs.receiverDFBInfo.dfbIndex != rhs.receiverDFBInfo.dfbIndex) {
    return false;
  }

  std::optional<std::uint64_t> maybeCombinedPeriod =
      getCombinedReceiverSequencePeriod(lhsSequence, rhsSequence);
  std::uint64_t comparisonCount = 0;
  if (lhsSequence.executionCount) {
    comparisonCount = *lhsSequence.executionCount;
    if (maybeCombinedPeriod) {
      comparisonCount = std::min(comparisonCount, *maybeCombinedPeriod);
    }
  } else {
    if (!maybeCombinedPeriod) {
      return false;
    }
    comparisonCount = *maybeCombinedPeriod;
  }

  int64_t lhsSlot = lhsSequence.recurrence->initialSlot;
  int64_t rhsSlot = rhsSequence.recurrence->initialSlot;
  for (std::uint64_t occurrence = 0; occurrence < comparisonCount;
       ++occurrence) {
    std::optional<int64_t> maybeLhsOffset =
        getReceiverAddressByteOffset(lhs, lhsSlot);
    std::optional<int64_t> maybeRhsOffset =
        getReceiverAddressByteOffset(rhs, rhsSlot);
    if (!maybeLhsOffset || !maybeRhsOffset ||
        maybeLhsOffset != maybeRhsOffset) {
      return false;
    }
    lhsSlot = advanceReceiverSlot(lhsSlot, lhsSequence.recurrence->repeatStride,
                                  lhsSequence.recurrence->blockCount);
    rhsSlot = advanceReceiverSlot(rhsSlot, rhsSequence.recurrence->repeatStride,
                                  rhsSequence.recurrence->blockCount);
  }
  return true;
}

const PipeReceiverEndpoint *PipeGraph::getProvenReceiverAddressEndpoint(
    PipeTransferNodeId transferNodeId) const {
  const PipeReceiverEndpoint *representative = nullptr;
  for (PipeReceiverEndpointId endpointId :
       getPipeReceiverEndpoints(transferNodeId)) {
    const PipeReceiverEndpoint &endpoint = getPipeReceiverEndpoint(endpointId);
    const PipeReceiverDFBNode &receiverDFBNode =
        getReceiverDFBNode(endpoint.receiverDFBNode);
    if (!receiverDFBNode.hasProvenPipeOnlyProducerStream ||
        endpoint.addressSequence.getKind() ==
            ReceiverAddressSequenceProofKind::FullyDynamic) {
      return nullptr;
    }
    if (representative && !havePointwiseEqualReceiverAddressSequences(
                              *representative, endpoint)) {
      return nullptr;
    }
    representative = &endpoint;
  }
  return representative;
}

static std::optional<std::uint64_t>
getConcreteTransferExecutionCount(Operation *op, LaunchNodeCoord coord,
                                  const PipeReference &pipeRef,
                                  PipeGraphAnalysisState &analysisState) {
  std::optional<std::uint64_t> maybeTotalCount =
      getExactExecutionCountAtLaunchNode(op, coord, analysisState);
  if (!maybeTotalCount || pipeRef.isStatic()) {
    return maybeTotalCount;
  }

  std::uint64_t matchingRecordCount = llvm::count_if(
      pipeRef.getRecords().getPipes(), [&](PipeRecordAttr record) {
        if (pipeRef.isSelectedSrc()) {
          return record.getSrcX() == coord.x && record.getSrcY() == coord.y;
        }
        return coord.x >= record.getDstStartX() &&
               coord.x <= record.getDstEndX() &&
               coord.y >= record.getDstStartY() &&
               coord.y <= record.getDstEndY();
      });
  if (matchingRecordCount == 0 || *maybeTotalCount % matchingRecordCount != 0) {
    return std::nullopt;
  }
  // The shared operation count includes every matching record. Graph nodes
  // represent individual records and therefore use the per-record count.
  return *maybeTotalCount / matchingRecordCount;
}

/// A protocol operation and the selected record it represents.
template <typename ProtocolOp>
struct PipeProtocolCandidate {
  ProtocolOp op;
  std::optional<std::uint64_t> recordIndex;
};

void PipeGraph::recordTransferNodeForProtocolRecord(
    Operation *op, std::optional<std::uint64_t> recordIndex,
    PipeTransferNodeId transferNodeId) {
  if (!recordIndex) {
    return;
  }
  auto [recordIt, inserted] = transferNodeIdByProtocolOpAndRecord.try_emplace(
      std::make_pair(op, *recordIndex), transferNodeId);
  assert((inserted || recordIt->second == transferNodeId) &&
         "selected protocol record maps to different transfers");
}

LogicalResult
PipeGraph::rebuildEndpointGraph(const PipeTransferIndex &transferIndex,
                                PipeGraphAnalysisState &analysisState) {
  pipeTransferNodes.clear();
  transferNodeIdsByProtocolOp.clear();
  transferNodeIdByProtocolOpAndRecord.clear();
  pipeReceiverEndpoints.clear();
  receiverDFBNodes.clear();

  // Sends and receiver posts that declare the same logical pipe relation.
  // Correspondence analysis matches them into individual transfers.
  using PipeSendCandidate = PipeProtocolCandidate<PipeTransferSendOp>;
  using PipePostCandidate = PipeProtocolCandidate<PipeTransferPostOp>;
  struct PipeTransferCandidates {
    SmallVector<PipeSendCandidate> sends;
    llvm::DenseMap<PipeReceiverCoord, SmallVector<PipePostCandidate>>
        postsByReceiver;
  };

  llvm::MapVector<PipeKey, PipeTransferCandidates> candidatesByPipe;
  for (Operation *op : analysisState.transferProtocolOps) {
    if (auto sendOp = dyn_cast<PipeTransferSendOp>(op)) {
      PipeTransferCreateOp createOp =
          transferIndex.getTransferCreate(sendOp.getOperation());
      FailureOr<PipeReference> pipeRef =
          getPipeReference(sendOp, createOp.getPipe());
      if (failed(pipeRef)) {
        return failure();
      }
      LaunchNodeDomain sendDomain =
          lookupOperationLaunchDomain(sendOp.getOperation(), analysisState);
      if (!sendDomain.known && analysisState.hasLaunchGrid) {
        sendOp.emitError("cannot determine whether the pipe source executes "
                         "this send");
        return failure();
      }
      SmallVector<PipeType> pipeTypes =
          getPipeTypesFromReference(sendOp.getContext(), *pipeRef);
      for (auto [recordIndex, pipeType] : llvm::enumerate(pipeTypes)) {
        PipeKey pipeKey = getPipeKey(pipeType);
        LaunchNodeCoord source{pipeKey.srcX, pipeKey.srcY};
        if (sendDomain.known &&
            !knownLaunchNodeDomainContains(sendDomain, source)) {
          continue;
        }
        std::optional<std::uint64_t> maybeExecutionCount =
            getConcreteTransferExecutionCount(sendOp.getOperation(), source,
                                              *pipeRef, analysisState);
        if (maybeExecutionCount && *maybeExecutionCount == 0) {
          continue;
        }
        candidatesByPipe[pipeKey].sends.push_back(
            {sendOp, pipeRef->isSelected()
                         ? std::optional<std::uint64_t>(recordIndex)
                         : std::nullopt});
      }
      continue;
    }

    auto postOp = dyn_cast<PipeTransferPostOp>(op);
    if (!postOp) {
      continue;
    }
    PipeTransferCreateOp createOp =
        transferIndex.getTransferCreate(postOp.getOperation());
    FailureOr<PipeReference> pipeRef =
        getPipeReference(postOp, createOp.getPipe());
    if (failed(pipeRef)) {
      return failure();
    }
    LaunchNodeDomain postDomain =
        lookupOperationLaunchDomain(postOp.getOperation(), analysisState);
    if (!postDomain.known && analysisState.hasLaunchGrid) {
      postOp.emitError(
          "cannot determine which pipe receivers execute this post");
      return failure();
    }
    SmallVector<PipeType> pipeTypes =
        getPipeTypesFromReference(postOp.getContext(), *pipeRef);
    for (auto [recordIndex, pipeType] : llvm::enumerate(pipeTypes)) {
      std::optional<std::uint64_t> selectedRecordIndex =
          pipeRef->isSelected() ? std::optional<std::uint64_t>(recordIndex)
                                : std::nullopt;
      PipeKey pipeKey = getPipeKey(pipeType);
      PipeTransferCandidates &candidates = candidatesByPipe[pipeKey];
      pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
        LaunchNodeCoord receiverCoord = getLaunchNodeCoord(receiver);
        if (postDomain.known &&
            !knownLaunchNodeDomainContains(postDomain, receiverCoord)) {
          return;
        }
        std::optional<std::uint64_t> maybeExecutionCount =
            getConcreteTransferExecutionCount(
                postOp.getOperation(), receiverCoord, *pipeRef, analysisState);
        if (!maybeExecutionCount || *maybeExecutionCount != 0) {
          candidates.postsByReceiver[receiver].push_back(
              {postOp, selectedRecordIndex});
        }
      });
      if (candidates.postsByReceiver.empty() && candidates.sends.empty()) {
        candidatesByPipe.erase(pipeKey);
      }
    }
  }

  llvm::DenseMap<PipeReceiverDFBKey, PipeReceiverDFBNodeId> nodeIdByReceiverDFB;
  for (auto &candidateEntry : candidatesByPipe) {
    PipeKey pipeKey = candidateEntry.first;
    PipeTransferCandidates &candidates = candidateEntry.second;
    if (candidates.sends.empty()) {
      Operation *postOp =
          candidates.postsByReceiver.begin()->second.front().op.getOperation();
      postOp->emitError("pipe receiver post has no corresponding send");
      return failure();
    }

    SmallVector<SmallVector<std::pair<PipeReceiverCoord, PipePostCandidate>>>
        endpointsBySend(candidates.sends.size());
    llvm::DenseMap<Operation *, std::size_t> sendIndexByPost;
    LogicalResult correspondenceResult = success();
    pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
      if (failed(correspondenceResult)) {
        return;
      }
      auto postsIt = candidates.postsByReceiver.find(receiver);
      std::size_t postCount = postsIt == candidates.postsByReceiver.end()
                                  ? 0
                                  : postsIt->second.size();
      if (postCount != candidates.sends.size()) {
        Operation *diagnosticOp =
            postCount == 0 ? candidates.sends.front().op.getOperation()
                           : postsIt->second.front().op.getOperation();
        diagnosticOp->emitError()
            << "cannot prove one receiver post per pipe transfer for receiver "
            << "(" << receiver.x << ", " << receiver.y << "); found "
            << candidates.sends.size() << " send definition(s) and "
            << postCount << " receiver post definition(s)";
        correspondenceResult = failure();
        return;
      }

      for (std::size_t sendIndex = 0; sendIndex < candidates.sends.size();
           ++sendIndex) {
        PipeTransferSendOp sendOp = candidates.sends[sendIndex].op;
        PipeTransferPostOp postOp = postsIt->second[sendIndex].op;
        PipeTransferCreateOp sendCreate =
            transferIndex.getTransferCreate(sendOp.getOperation());
        PipeTransferCreateOp postCreate =
            transferIndex.getTransferCreate(postOp.getOperation());
        FailureOr<PipeReference> sendPipeRef =
            getPipeReference(sendOp, sendCreate.getPipe());
        FailureOr<PipeReference> postPipeRef =
            getPipeReference(postOp, postCreate.getPipe());
        assert(succeeded(sendPipeRef) && succeeded(postPipeRef) &&
               "pipe transfer graph validated pipe references");
        bool haveEqualExecutionCounts = false;
        std::optional<std::uint64_t> maybeSendCount;
        std::optional<std::uint64_t> maybePostCount;
        if ((*sendPipeRef).isStatic() && (*postPipeRef).isStatic()) {
          // Static pipe operations can have matching loop and branch structure
          // even when an exact execution count cannot be evaluated.
          haveEqualExecutionCounts = proveEqualExecutionCountAtLaunchNodes(
              sendOp.getOperation(), {pipeKey.srcX, pipeKey.srcY},
              postOp.getOperation(), getLaunchNodeCoord(receiver),
              analysisState);
        } else {
          // Selected operations represent several records, so compare their
          // per-record counts rather than their shared operation counts.
          maybeSendCount = getConcreteTransferExecutionCount(
              sendOp.getOperation(), {pipeKey.srcX, pipeKey.srcY}, *sendPipeRef,
              analysisState);
          maybePostCount = getConcreteTransferExecutionCount(
              postOp.getOperation(), getLaunchNodeCoord(receiver), *postPipeRef,
              analysisState);
          haveEqualExecutionCounts = maybeSendCount && maybePostCount &&
                                     *maybeSendCount == *maybePostCount;
        }
        if (!haveEqualExecutionCounts) {
          auto diag = postOp.emitError()
                      << "cannot prove matching execution counts for this "
                         "receiver post and its pipe send";
          if ((*sendPipeRef).isSelected() || (*postPipeRef).isSelected()) {
            diag << "; receiver post count is ";
            if (maybePostCount) {
              diag << *maybePostCount;
            } else {
              diag << "unknown";
            }
            diag << " and pipe send count is ";
            if (maybeSendCount) {
              diag << *maybeSendCount;
            } else {
              diag << "unknown";
            }
          }
          diag.attachNote(sendOp.getLoc()) << "corresponding pipe send is here";
          correspondenceResult = failure();
          return;
        }
        auto [existing, inserted] =
            sendIndexByPost.try_emplace(postOp.getOperation(), sendIndex);
        if (!inserted && existing->second != sendIndex &&
            (*postPipeRef).isStatic()) {
          postOp.emitError(
              "one receiver post cannot correspond to two pipe transfers");
          correspondenceResult = failure();
          return;
        }
        endpointsBySend[sendIndex].push_back(
            {receiver, postsIt->second[sendIndex]});
      }
    });
    if (failed(correspondenceResult)) {
      return failure();
    }

    for (auto [sendIndex, sendCandidate] : llvm::enumerate(candidates.sends)) {
      PipeTransferSendOp sendOp = sendCandidate.op;
      PipeTransferCreateOp sendCreate =
          transferIndex.getTransferCreate(sendOp.getOperation());
      PipeTransferContract transferContract =
          getPipeTransferContract(sendCreate);
      int64_t blockSpan = getPipeTransferBlockSpan(sendCreate);
      int64_t destinationGroupDepth =
          getPipeTransferDestinationGroupDepth(sendCreate);
      for (const auto &endpoint : endpointsBySend[sendIndex]) {
        PipeTransferPostOp postOp = endpoint.second.op;
        PipeTransferCreateOp postCreate =
            transferIndex.getTransferCreate(postOp.getOperation());
        if (getPipeTransferContract(postCreate) != transferContract) {
          postOp.emitError(
              "pipe send and receiver post use different transfer contracts");
          return failure();
        }
        if (getPipeTransferBlockSpan(postCreate) != blockSpan) {
          auto diagnostic =
              postOp.emitError("pipe send and receiver post use different "
                               "transfer block spans");
          diagnostic.attachNote(sendOp.getLoc())
              << "corresponding pipe send uses block_span=" << blockSpan;
          return failure();
        }
        if (getPipeTransferDestinationGroupDepth(postCreate) !=
            destinationGroupDepth) {
          auto diagnostic = postOp.emitError(
              "pipe send and receiver post use different destination group "
              "depths");
          diagnostic.attachNote(sendOp.getLoc())
              << "corresponding pipe send uses destination_group_depth="
              << destinationGroupDepth;
          return failure();
        }
      }

      PipeTransferNodeId transferNodeId = pipeTransferNodes.size();
      pipeTransferNodes.push_back(PipeTransferNode{transferNodeId,
                                                   pipeKey,
                                                   transferContract,
                                                   blockSpan,
                                                   destinationGroupDepth,
                                                   sendOp.getOperation(),
                                                   {},
                                                   {}});
      PipeTransferNode &transferNode = pipeTransferNodes.back();
      transferNodeIdsByProtocolOp[sendOp.getOperation()].push_back(
          transferNodeId);
      recordTransferNodeForProtocolRecord(
          sendOp.getOperation(), sendCandidate.recordIndex, transferNodeId);
      llvm::SmallSetVector<Operation *, 4> uniquePostOps;

      for (auto [receiver, postCandidate] : endpointsBySend[sendIndex]) {
        PipeTransferPostOp postOp = postCandidate.op;
        auto infoIt = receiverDFBByPost.find(postOp.getOperation());
        assert(infoIt != receiverDFBByPost.end() &&
               "receiver post must have DFB geometry");
        const ReceiverDFBInfo &receiverInfo = infoIt->second;
        uniquePostOps.insert(postOp.getOperation());
        SmallVector<PipeTransferNodeId> &postTransferNodeIds =
            transferNodeIdsByProtocolOp[postOp.getOperation()];
        if (!llvm::is_contained(postTransferNodeIds, transferNodeId)) {
          postTransferNodeIds.push_back(transferNodeId);
        }
        recordTransferNodeForProtocolRecord(
            postOp.getOperation(), postCandidate.recordIndex, transferNodeId);
        PipeReceiverDFBKey receiverDFB{receiver, receiverInfo.dfbIndex};
        auto nodeIt = nodeIdByReceiverDFB.find(receiverDFB);
        PipeReceiverDFBNodeId receiverDFBNodeId = 0;
        if (nodeIt == nodeIdByReceiverDFB.end()) {
          receiverDFBNodeId = receiverDFBNodes.size();
          nodeIdByReceiverDFB.insert({receiverDFB, receiverDFBNodeId});
          receiverDFBNodes.push_back(PipeReceiverDFBNode{
              receiverDFBNodeId, receiverDFB, {}, false, {}});
        } else {
          receiverDFBNodeId = nodeIt->second;
        }

        PipeReceiverEndpointId endpointId = pipeReceiverEndpoints.size();
        pipeReceiverEndpoints.push_back(
            PipeReceiverEndpoint{endpointId,
                                 transferNodeId,
                                 receiverDFBNodeId,
                                 receiver,
                                 receiverDFB,
                                 receiverInfo,
                                 postOp.getOperation(),
                                 {}});
        transferNode.receiverEndpoints.push_back(endpointId);
        receiverDFBNodes[receiverDFBNodeId].writerEndpoints.push_back(
            endpointId);
      }
      transferNode.receiverPostOps.assign(uniquePostOps.begin(),
                                          uniquePostOps.end());
      if (failed(verifyTransferPayloadCompatibility(transferNode))) {
        return failure();
      }
    }
  }
  return success();
}

FailureOr<PipeReference> getPipeReference(Operation *op, Value pipe) {
  Value tracedPipe = traceUnrealizedCasts(pipe);
  if (auto pipeType = mlir::dyn_cast<PipeType>(tracedPipe.getType())) {
    return PipeReference(pipeType);
  }
  if (auto selectedSrc = tracedPipe.getDefiningOp<SelectPipeSrcOp>()) {
    return PipeReference(selectedSrc);
  }
  if (auto selectedDst = tracedPipe.getDefiningOp<SelectPipeDstOp>()) {
    return PipeReference(selectedDst);
  }
  return op->emitError() << "selected pipe operand must be a direct result of "
                            "ttl.select_pipe_src or ttl.select_pipe_dst";
}

FailureOr<PipeReference>
getPipeReferenceForProtocolOp(Operation *protocolOp,
                              const PipeTransferIndex &transferIndex) {
  return getPipeReference(
      protocolOp, transferIndex.getTransferCreate(protocolOp).getPipe());
}

SmallVector<PipeType> getPipeTypesFromReference(MLIRContext *context,
                                                const PipeReference &ref) {
  if (ref.isStatic()) {
    return SmallVector<PipeType>{ref.getStaticPipeType()};
  }
  SmallVector<PipeType> pipeTypes;
  PipeNetRecordsAttr records = ref.getRecords();
  pipeTypes.reserve(records.getPipes().size());
  for (PipeRecordAttr record : records.getPipes()) {
    pipeTypes.push_back(
        getPipeTypeFromRecord(context, record, records.getPipeNetId()));
  }
  return pipeTypes;
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
    std::optional<int64_t> maybeOffset = getConstantIntValue(mixedOffset);
    if (!maybeOffset.has_value()) {
      return failure();
    }
    coordinate += *maybeOffset;
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
  if (!tensorType.hasStaticShape()) {
    return failure();
  }
  return tensorType.getNumElements();
}

static FailureOr<int64_t>
getReceiverSlotSpanBlocks(Operation *postOp, Value dst,
                          CircularBufferType dfbType) {
  auto reserveOp = findCBReserveForPipeReceive(dst);
  if (!reserveOp) {
    return postOp->emitError("could not determine receiver DFB reserve span");
  }
  auto reserveType =
      mlir::dyn_cast<RankedTensorType>(reserveOp.getResult().getType());
  if (!reserveType) {
    return postOp->emitError("could not determine receiver DFB reserve span");
  }

  auto dfbBlockType =
      RankedTensorType::get(dfbType.getShape(), dfbType.getElementType());
  FailureOr<int64_t> reserveTileCount = getTensorTileCount(reserveType);
  FailureOr<int64_t> dfbBlockTileCount = getTensorTileCount(dfbBlockType);
  if (failed(reserveTileCount) || failed(dfbBlockTileCount) ||
      *dfbBlockTileCount <= 0) {
    return postOp->emitError("could not determine receiver DFB reserve span");
  }
  if (*reserveTileCount <= 0 || *reserveTileCount % *dfbBlockTileCount != 0) {
    return postOp->emitError()
           << "PipeNet receiver DFB reserve must contain a whole number of "
              "DFB blocks; reserve contains "
           << *reserveTileCount << " tile(s), but each DFB block contains "
           << *dfbBlockTileCount << " tile(s)";
  }
  return *reserveTileCount / *dfbBlockTileCount;
}

LogicalResult PipeGraph::addPipeReceiver(Operation *op,
                                         PipeTransferCreateOp transferCreateOp,
                                         Value dst) {
  PipeTransferContract transferContract =
      getPipeTransferContract(transferCreateOp);
  Value dstDFB = getAttachedCB(dst);
  if (!dstDFB) {
    return op->emitError("pipe receive destination is not attached to a DFB");
  }
  auto dfbType = mlir::dyn_cast<CircularBufferType>(dstDFB.getType());
  if (!dfbType) {
    return op->emitError("pipe receive destination is not attached to a DFB");
  }

  std::optional<int64_t> maybeDFBIndex = getCBIndex(dstDFB);
  if (!maybeDFBIndex.has_value()) {
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

  FailureOr<int64_t> slotSpanBlocks =
      getReceiverSlotSpanBlocks(op, dst, dfbType);
  if (failed(slotSpanBlocks)) {
    return failure();
  }
  ReceiverDFBInfo receiverInfo{*maybeDFBIndex,      dfbType,
                               hasStaticTileOffset, staticTileOffset,
                               *slotSpanBlocks,     dfbType.getBlockCount(),
                               op->getLoc()};
  bool inserted = receiverDFBByPost.insert({op, receiverInfo}).second;
  assert(inserted && "receiver post visited more than once");
  return success();
}

FailureOr<PipeGraph>
PipeGraph::build(ModuleOp mod, const PipeTransferIndex &transferIndex,
                 const PipeForeachLoweringInfo &foreachLoweringInfo) {
  PipeGraph graph;
  PipeGraphAnalysisState analysisState;
  if (failed(collectPipeGraphOperations(mod, transferIndex, analysisState))) {
    return failure();
  }

  analysisState.pipeRecordControlOps.insert(
      foreachLoweringInfo.controlOps.begin(),
      foreachLoweringInfo.controlOps.end());
  analysisState.pipeRecordIfThenDomains = foreachLoweringInfo.ifThenDomains;
  analysisState.pipeRecordLoops = foreachLoweringInfo.recordLoops;

  for (PipeTransferPostOp postOp : analysisState.receiverPosts) {
    PipeTransferCreateOp createOp =
        transferIndex.getTransferCreate(postOp.getOperation());
    if (failed(graph.addPipeReceiver(postOp, createOp, postOp.getDst()))) {
      return failure();
    }
  }
  if (failed(collectLaunchNodeDomains(mod, analysisState))) {
    return failure();
  }

  if (failed(graph.rebuildEndpointGraph(transferIndex, analysisState))) {
    return failure();
  }
  if (failed(graph.assignReceiverAddressSequences(mod, transferIndex,
                                                  analysisState))) {
    return failure();
  }
  if (failed(graph.provePipeOnlyReceiverProducerStreams(analysisState))) {
    return failure();
  }
  if (failed(graph.verifyCollectiveReceiverAddresses())) {
    return failure();
  }
  graph.hasAnalyzedLaunchGrid = analysisState.hasLaunchGrid;
  graph.operationLaunchDomains =
      std::move(analysisState.operationLaunchDomains);
  graph.dfbLifecycles = std::move(analysisState.dfbLifecycles);
  return std::move(graph);
}

} // namespace mlir::tt::ttl
