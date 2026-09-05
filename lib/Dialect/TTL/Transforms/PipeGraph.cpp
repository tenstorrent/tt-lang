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
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
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

struct RecordExecutionCountAnalysisCache {
  std::unique_ptr<ExecutionCountAnalysisSharedState> sharedState;
  ExecutionCountAnalysisQueryCache<
      std::pair<LaunchExecutionLocation, std::uint64_t>>
      analysesByContext;
};

/// Analysis facts and operation indexes used while constructing PipeGraph.
struct PipeGraphAnalysisState : LaunchNodeDomainState {
  struct ReceiveWaitAnyUse {
    PipeTransferWaitAnyOp wait;
    unsigned candidateIndex;
  };

  std::unique_ptr<DFBLogicalIdentityAnalysis> dfbLogicalIdentities;
  PipeDFBIndexMode dfbIndexMode = PipeDFBIndexMode::Provisional;
  llvm::DenseMap<Operation *, LaunchNodeDomain> operationLaunchDomains;
  llvm::DenseMap<Operation *, std::unique_ptr<DFBAcquireReleaseIndex>>
      dfbLifecycles;
  SmallVector<Operation *> transferProtocolOps;
  SmallVector<PipeTransferPostOp> receiverPosts;
  llvm::DenseMap<Operation *, SmallVector<PipeTransferWaitOp>>
      receiveWaitsByPost;
  llvm::DenseMap<Operation *, SmallVector<ReceiveWaitAnyUse>>
      receiveWaitAnysByPost;
  llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<PipeTransferPostOp>>
      receiverPostsByStream;
  llvm::DenseMap<PipeReceiverDFBPhysicalStreamKey, SmallVector<CBPushOp>>
      pushesByPhysicalStream;
  llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<CBPopOp>> popsByStream;
  llvm::SmallPtrSet<Operation *, 16> pipeRecordControlOps;
  llvm::DenseMap<Operation *, LaunchNodeDomain> pipeRecordIfThenDomains;
  llvm::DenseMap<Operation *, PipeNetRecordLoop> pipeRecordLoops;
  llvm::DenseMap<Operation *, RecordExecutionCountAnalysisCache>
      recordExecutionCountAnalyses;
};

namespace {

static LogicalResult collectLaunchNodeDomains(ModuleOp mod,
                                              PipeGraphAnalysisState &state) {
  state.initialize(mod);
  if (state.hasLaunchGrid) {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    LaunchNodeDomainAnalysisOptions options;
    options.narrowPipeNetScopes = true;
    options.operationCallback = [&](Operation *op,
                                    const LaunchNodeDomain &domain,
                                    Operation * /*unanalyzableOp*/) {
      state.operationLaunchDomains[op] = domain;
    };
    options.computeRegionDomain =
        [&](Operation *op,
            unsigned regionNumber) -> std::optional<LaunchNodeDomain> {
      if (auto ifOp = dyn_cast<scf::IfOp>(op);
          ifOp && getReadyReceiveSelection(ifOp.getCondition())) {
        return state.baseDomain;
      }
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
  }
  return success();
}

static LogicalResult collectDFBLifecycles(ModuleOp mod,
                                          PipeGraphAnalysisState &state) {
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

static bool
isProvenInactiveAtAllLocations(Operation *op,
                               ArrayRef<LaunchExecutionLocation> locations,
                               PipeGraphAnalysisState &state) {
  return !locations.empty() &&
         llvm::all_of(locations, [&](const LaunchExecutionLocation &location) {
           std::optional<std::uint64_t> executionCount =
               getExactExecutionCountAtLaunchLocation(op, location, state);
           return executionCount && *executionCount == 0;
         });
}

static LaunchNodeCoord getLaunchNodeCoord(PipeReceiverCoord receiver) {
  return {receiver.x, receiver.y};
}

static FailureOr<LaunchExecutionLocation>
getPipeGraphExecutionLocation(Operation *op, LaunchNodeCoord node,
                              DeviceTransferAttr transfer, PipeRole role) {
  FailureOr<LaunchExecutionLocation> maybeLocation =
      getPipeExecutionLocation(node, transfer, role);
  if (failed(maybeLocation)) {
    op->emitError(
        "device-range fabric transfers require scatter target lowering");
  }
  return maybeLocation;
}

static DeviceRefAttr getReceiverDevice(PipeTransferCreateOp transferCreate) {
  DeviceTransferAttr transfer = transferCreate.getDeviceTransferAttr();
  return transfer ? transfer.getEdge().getDestination() : DeviceRefAttr();
}

static DeviceRefAttr getEnclosingReceiverDevice(Operation *op) {
  for (Operation *ancestor = op->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    auto ifDst = dyn_cast<IfDstOp>(ancestor);
    if (!ifDst) {
      continue;
    }
    auto createPipe = ifDst.getPipe().getDefiningOp<CreatePipeOp>();
    if (!createPipe) {
      return {};
    }
    DeviceTransferAttr transfer = createPipe.getDeviceTransferAttr();
    return transfer ? transfer.getEdge().getDestination() : DeviceRefAttr();
  }
  return {};
}

static FailureOr<PipeReceiverDFBStreamKey>
getReceiverDFBStreamKey(Value dfb, DeviceRefAttr receiverDevice,
                        const DFBLogicalIdentityAnalysis &dfbIds) {
  std::optional<int64_t> maybeDFBIndex = getCBIndex(dfb);
  if (!maybeDFBIndex) {
    return failure();
  }
  FailureOr<int64_t> maybeDFBId = dfbIds.getLogicalId(dfb);
  if (failed(maybeDFBId)) {
    return failure();
  }
  return PipeReceiverDFBStreamKey{receiverDevice, *maybeDFBIndex, *maybeDFBId};
}

static Value getReceiverDFB(CBPushOp pushOp) { return pushOp.getCb(); }

static LogicalResult
recordReceiverPost(PipeTransferPostOp postOp, PipeGraphAnalysisState &state,
                   const PipeTransferIndex &transferIndex) {
  PipeTransferCreateOp createOp =
      transferIndex.getTransferCreate(postOp.getOperation());

  state.transferProtocolOps.push_back(postOp.getOperation());
  state.receiverPosts.push_back(postOp);
  FailureOr<PipeReceiverDFBStreamKey> maybeStreamKey = getReceiverDFBStreamKey(
      getAttachedCB(postOp.getDst()), getReceiverDevice(createOp),
      *state.dfbLogicalIdentities);
  if (failed(maybeStreamKey)) {
    return postOp.emitError(
        "could not resolve receiver DFB logical and physical identity");
  }
  state.receiverPostsByStream[*maybeStreamKey].push_back(postOp);
  return success();
}

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

/// Associate each wait-any candidate with its possible receiver posts.
static void recordReceiveWaitAny(PipeTransferWaitAnyOp waitOp,
                                 PipeGraphAnalysisState &state,
                                 const PipeTransferIndex &transferIndex) {
  for (auto [candidateIndex, possiblePosts] :
       llvm::enumerate(transferIndex.getWaitAnyCandidatePosts(waitOp))) {
    for (Operation *post : possiblePosts) {
      state.receiveWaitAnysByPost[post].push_back(
          {waitOp, static_cast<unsigned>(candidateIndex)});
    }
  }
}

/// Collect protocol and receiver DFB operations once so graph analyses do not
/// rescan the module for every receiver.
static LogicalResult
collectPipeGraphOperations(ModuleOp mod, const PipeTransferIndex &transferIndex,
                           PipeGraphAnalysisState &state) {
  bool failedToResolveDFB = false;
  WalkResult walkResult =
      mod.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        if (auto waitOp = dyn_cast<PipeTransferWaitOp>(op)) {
          return failed(recordReceiveWait(waitOp, state, transferIndex))
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        }
        llvm::TypeSwitch<Operation *>(op)
            .Case<PipeTransferPostOp>([&](PipeTransferPostOp postOp) {
              if (failed(recordReceiverPost(postOp, state, transferIndex))) {
                failedToResolveDFB = true;
              }
            })
            .Case<PipeTransferSendOp>([&](PipeTransferSendOp sendOp) {
              state.transferProtocolOps.push_back(sendOp.getOperation());
            })
            .Case<PipeTransferWaitAnyOp>([&](PipeTransferWaitAnyOp waitOp) {
              recordReceiveWaitAny(waitOp, state, transferIndex);
            })
            .Case<CBPushOp>([&](CBPushOp pushOp) {
              FailureOr<PipeReceiverDFBStreamKey> maybeStreamKey =
                  getReceiverDFBStreamKey(pushOp.getCb(),
                                          getEnclosingReceiverDevice(pushOp),
                                          *state.dfbLogicalIdentities);
              if (failed(maybeStreamKey)) {
                pushOp.emitError(
                    "could not resolve DFB logical and physical identity");
                failedToResolveDFB = true;
              } else {
                state
                    .pushesByPhysicalStream[{maybeStreamKey->receiverDevice,
                                             maybeStreamKey->dfbIndex}]
                    .push_back(pushOp);
              }
            })
            .Case<CBPopOp>([&](CBPopOp popOp) {
              FailureOr<PipeReceiverDFBStreamKey> maybeStreamKey =
                  getReceiverDFBStreamKey(popOp.getCb(),
                                          getEnclosingReceiverDevice(popOp),
                                          *state.dfbLogicalIdentities);
              if (failed(maybeStreamKey)) {
                popOp.emitError(
                    "could not resolve DFB logical and physical identity");
                failedToResolveDFB = true;
              } else {
                state.popsByStream[*maybeStreamKey].push_back(popOp);
              }
            });
        if (failedToResolveDFB) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  return success(!walkResult.wasInterrupted());
}

/// Visit operations for this logical-device receiver and operations shared by
/// every logical device using the same physical DFB index.
template <typename Callback>
static void forEachReceiverDFBPhysicalStreamEvent(
    const llvm::DenseMap<PipeReceiverDFBPhysicalStreamKey,
                         SmallVector<CBPushOp>> &eventsByStream,
    const PipeReceiverDFBKey &receiverDFB, Callback &&callback) {
  auto visit = [&](DeviceRefAttr receiverDevice) {
    auto eventsIt = eventsByStream.find({receiverDevice, receiverDFB.dfbIndex});
    if (eventsIt == eventsByStream.end()) {
      return;
    }
    for (CBPushOp event : eventsIt->second) {
      callback(event);
    }
  };

  visit(receiverDFB.receiverDevice);
  if (receiverDFB.receiverDevice) {
    visit({});
  }
}

template <typename OpTy, typename Callback>
static void forEachReceiverDFBLogicalStreamEvent(
    const llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<OpTy>>
        &eventsByStream,
    const PipeReceiverDFBKey &receiverDFB, Callback &&callback) {
  auto visit = [&](DeviceRefAttr receiverDevice) {
    auto eventsIt = eventsByStream.find(
        {receiverDevice, receiverDFB.dfbIndex, receiverDFB.dfbId});
    if (eventsIt == eventsByStream.end()) {
      return;
    }
    for (OpTy event : eventsIt->second) {
      callback(event);
    }
  };
  visit(receiverDFB.receiverDevice);
  if (receiverDFB.receiverDevice) {
    visit({});
  }
}

static const PipeReceiverEndpoint *
findPostReceiverEndpoint(PipeTransferPostOp postOp,
                         const PipeReceiverDFBKey &receiverDFB,
                         const PipeGraph &pipeGraph) {
  for (PipeTransferNodeId transferNodeId :
       pipeGraph.getPipeTransferNodeIdsForProtocolOp(postOp.getOperation())) {
    for (PipeReceiverEndpointId endpointId :
         pipeGraph.getPipeReceiverEndpoints(transferNodeId)) {
      const PipeReceiverEndpoint &endpoint =
          pipeGraph.getPipeReceiverEndpoint(endpointId);
      if (endpoint.postOp == postOp.getOperation() &&
          endpoint.receiverDFB == receiverDFB) {
        return &endpoint;
      }
    }
  }
  return nullptr;
}

static bool isPostForReceiverDFB(PipeTransferPostOp postOp,
                                 const PipeReceiverDFBKey &receiverDFB,
                                 const PipeGraph &pipeGraph) {
  return findPostReceiverEndpoint(postOp, receiverDFB, pipeGraph) != nullptr;
}

static SmallVector<PipeTransferPostOp>
getPostsOwnedByReserve(CBReserveOp reserveOp,
                       ArrayRef<PipeTransferPostOp> posts) {
  SmallVector<PipeTransferPostOp> ownedPosts;
  for (PipeTransferPostOp postOp : posts) {
    if (findCBReserveForPipeReceive(postOp.getDst(), postOp.getOperation()) ==
        reserveOp) {
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

static void
debugRejectComputedAddressProducerPhase(const PipeReceiverDFBKey &receiverDFB,
                                        llvm::StringRef reason) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeGraph: reject computed-address producer phase for ";
    printReceiverDFB(llvm::dbgs(), receiverDFB);
    llvm::dbgs() << ": " << reason << "\n";
  });
}

static void
debugAcceptComputedAddressProducerPhase(const PipeReceiverDFBKey &receiverDFB) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeGraph: accept computed-address producer phase for ";
    printReceiverDFB(llvm::dbgs(), receiverDFB);
    llvm::dbgs() << "\n";
  });
}

using ReceiverEndpointsByDFB =
    llvm::DenseMap<PipeReceiverDFBKey, SmallVector<PipeReceiverEndpointId>>;

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
/// the receiver execution location and therefore constrains reservation order.
static std::optional<ReceiverControlContext>
getReceiverControlContext(Operation *op,
                          const LaunchExecutionLocation &location,
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
    if (parent &&
        isTransparentReceiverScope(
            parent, PipeReceiverCoord{location.node.x, location.node.y},
            analysisState)) {
      current = parent;
      continue;
    }
    if (auto ifOp = mlir::dyn_cast_if_present<scf::IfOp>(parent)) {
      if (std::optional<bool> maybeSelected = evaluatePredicateAtLaunchLocation(
              ifOp.getCondition(), location, analysisState)) {
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
/// Projecting a location-selected wrapper into the enclosing block preserves
/// the per-location order without treating unrelated blocks as sequential.
static bool
isBeforeInReceiverControlContext(Operation *before, Operation *after,
                                 const LaunchExecutionLocation &location,
                                 const PipeGraphAnalysisState &analysisState) {
  std::optional<ReceiverControlContext> maybeBeforeContext =
      getReceiverControlContext(before, location, analysisState);
  std::optional<ReceiverControlContext> maybeAfterContext =
      getReceiverControlContext(after, location, analysisState);
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

/// Return true when `before` precedes `after` directly or before an enclosing
/// runtime region containing `after`.
static bool
isBeforeInReceiverExecution(Operation *before, Operation *after,
                            const LaunchExecutionLocation &location,
                            const PipeGraphAnalysisState &analysisState) {
  Operation *enclosing = after;
  while (enclosing) {
    if (isBeforeInReceiverControlContext(before, enclosing, location,
                                         analysisState)) {
      return true;
    }
    Block *block = enclosing->getBlock();
    enclosing = block ? block->getParentOp() : nullptr;
  }
  return false;
}

static bool hasMatchingReceiveWaitBeforePush(
    PipeTransferPostOp postOp, CBPushOp pushOp,
    const llvm::DenseMap<Operation *, SmallVector<PipeTransferWaitOp>>
        &waitsByPost,
    const llvm::DenseMap<Operation *,
                         SmallVector<PipeGraphAnalysisState::ReceiveWaitAnyUse>>
        &waitAnysByPost,
    const LaunchExecutionLocation &location,
    const PipeGraphAnalysisState &analysisState) {
  auto waitIt = waitsByPost.find(postOp.getOperation());
  if (waitIt != waitsByPost.end() &&
      llvm::any_of(waitIt->second, [&](PipeTransferWaitOp waitOp) {
        return isBeforeInReceiverExecution(postOp, waitOp, location,
                                           analysisState) &&
               isBeforeInReceiverExecution(waitOp, pushOp, location,
                                           analysisState);
      })) {
    return true;
  }

  auto waitAnyIt = waitAnysByPost.find(postOp.getOperation());
  if (waitAnyIt == waitAnysByPost.end()) {
    return false;
  }
  for (PipeGraphAnalysisState::ReceiveWaitAnyUse use : waitAnyIt->second) {
    if (!isBeforeInReceiverControlContext(postOp, use.wait, location,
                                          analysisState)) {
      continue;
    }
    auto isOrderedBefore = [&](Operation *before, Operation *after) {
      return isBeforeInReceiverControlContext(before, after, location,
                                              analysisState);
    };
    if (isInReadyReceiveSelectionRegion(
            pushOp, use.wait, static_cast<int64_t>(use.candidateIndex),
            isOrderedBefore)) {
      return true;
    }
  }
  return false;
}

/// Group endpoints by logical DFB lifecycle. Physical aliases from other
/// lifecycles affect pointer phase but not producer ownership.
static ReceiverEndpointsByDFB
collectReceiverEndpointsByDFB(ArrayRef<PipeReceiverEndpoint> endpoints) {
  ReceiverEndpointsByDFB endpointsByReceiverDFB;
  for (const PipeReceiverEndpoint &endpoint : endpoints) {
    endpointsByReceiverDFB[endpoint.receiverDFB].push_back(endpoint.id);
  }
  return endpointsByReceiverDFB;
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
static FailureOr<std::optional<ReceiverControlContext>>
getProvenReceiverScheduleContext(const PipeReceiverDFBKey &receiverDFB,
                                 ArrayRef<PipeReceiverEndpointId> endpoints,
                                 const PipeGraph &pipeGraph,
                                 const PipeGraphAnalysisState &analysisState) {
  if (endpoints.empty()) {
    debugRejectReceiverSchedule(receiverDFB, "no receiver posts");
    return std::optional<ReceiverControlContext>();
  }

  auto getEndpointContext = [&](PipeReceiverEndpointId endpointId)
      -> FailureOr<std::optional<ReceiverControlContext>> {
    const PipeReceiverEndpoint &endpoint =
        pipeGraph.getPipeReceiverEndpoint(endpointId);
    auto postOp = mlir::cast<PipeTransferPostOp>(endpoint.postOp);
    const PipeTransferNode &transferNode =
        pipeGraph.getPipeTransferNode(endpoint.transferNode);
    FailureOr<LaunchExecutionLocation> maybeLocation =
        getPipeGraphExecutionLocation(
            postOp.getOperation(), getLaunchNodeCoord(receiverDFB.receiver),
            transferNode.deviceTransfer, PipeRole::Destination);
    if (failed(maybeLocation)) {
      return failure();
    }
    return getReceiverControlContext(postOp, *maybeLocation, analysisState);
  };

  FailureOr<std::optional<ReceiverControlContext>> maybeFirstContext =
      getEndpointContext(endpoints.front());
  if (failed(maybeFirstContext)) {
    return failure();
  }
  if (!*maybeFirstContext) {
    debugRejectReceiverSchedule(receiverDFB,
                                "receiver control cannot be evaluated");
    return std::optional<ReceiverControlContext>();
  }
  std::optional<ReceiverControlContext> controlContext = *maybeFirstContext;
  for (PipeReceiverEndpointId endpointId : endpoints.drop_front()) {
    FailureOr<std::optional<ReceiverControlContext>> maybePostContext =
        getEndpointContext(endpointId);
    if (failed(maybePostContext)) {
      return failure();
    }
    if (*maybePostContext != controlContext) {
      debugRejectReceiverSchedule(
          receiverDFB,
          "receiver posts do not share one sequential control context");
      return std::optional<ReceiverControlContext>();
    }
  }
  return controlContext;
}

/// Next physical DFB block selected by producer reservation order.
struct ReceiverProducerState {
  int64_t nextSlot = 0;
};

} // namespace

static std::optional<std::uint64_t> getConcreteTransferExecutionCount(
    Operation *op, const LaunchExecutionLocation &location,
    const PipeReference &pipeRef, std::optional<std::uint64_t> recordIndex,
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
  ReceiverEndpointsByDFB endpointsByReceiverDFB =
      collectReceiverEndpointsByDFB(pipeReceiverEndpoints);
  llvm::DenseSet<PipeReceiverDFBKey> invariantAddressReceiverDFBs;
  llvm::DenseMap<PipeReceiverDFBKey, std::optional<ReceiverControlContext>>
      scheduleContextByReceiverDFB;
  for (const auto &[receiverDFB, endpoints] : endpointsByReceiverDFB) {
    bool everyReservationSpansFullBuffer =
        llvm::all_of(endpoints, [&](PipeReceiverEndpointId endpointId) {
          const ReceiverDFBInfo &receiverInfo =
              getPipeReceiverEndpoint(endpointId).receiverDFBInfo;
          return receiverInfo.receiverSlotSpanBlocks == receiverInfo.blockCount;
        });
    if (everyReservationSpansFullBuffer) {
      invariantAddressReceiverDFBs.insert(receiverDFB);
      for (PipeReceiverEndpointId endpointId : endpoints) {
        PipeReceiverEndpoint &endpoint = pipeReceiverEndpoints[endpointId];
        ReceiverAddressSequenceProof sequence;
        sequence.recurrence = ReceiverAddressRecurrence{
            /*initialSlot=*/0,
            /*repeatStride=*/0,
            endpoint.receiverDFBInfo.blockCount,
        };
        if (failed(verifyReceiverReservationSequence(
                sequence, endpoint.receiverDFBInfo))) {
          return failure();
        }
        endpoint.addressSequence = std::move(sequence);
      }
      continue;
    }
    FailureOr<std::optional<ReceiverControlContext>> maybeContext =
        getProvenReceiverScheduleContext(receiverDFB, endpoints, *this,
                                         analysisState);
    if (failed(maybeContext)) {
      return failure();
    }
    scheduleContextByReceiverDFB.try_emplace(receiverDFB,
                                             std::move(*maybeContext));
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
  auto resolveRecordLoop =
      [&](Operation *op) -> std::optional<PipeNetRecordLoop> {
    auto recordLoopIt = analysisState.pipeRecordLoops.find(op);
    return recordLoopIt == analysisState.pipeRecordLoops.end()
               ? std::nullopt
               : std::optional<PipeNetRecordLoop>(recordLoopIt->second);
  };
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
    std::optional<std::uint64_t> activeRecordIndex;
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
      activeRecordIndex =
          getActivePipeNetRecordIndex(activeRecords, selectedRecordLoop);
      assert(activeRecordIndex &&
             "selected pipe operation must execute in an active record");
      auto transferIt = transferNodeIdByProtocolOpAndRecord.find(
          std::make_pair(postOp.getOperation(), *activeRecordIndex));
      if (transferIt == transferNodeIdByProtocolOpAndRecord.end()) {
        PipeRecordAttr record =
            pipeRef->getRecords().getPipes()[*activeRecordIndex];
        FailureOr<LaunchExecutionLocation> maybeLocation =
            getPipeGraphExecutionLocation(postOp.getOperation(), coord,
                                          record.getDeviceTransfer(),
                                          PipeRole::Destination);
        assert(succeeded(maybeLocation) &&
               "selected receiver record must have an execution location");
        std::optional<std::uint64_t> maybeExecutionCount =
            getConcreteTransferExecutionCount(postOp.getOperation(),
                                              *maybeLocation, *pipeRef,
                                              activeRecordIndex, analysisState);
        assert(maybeExecutionCount && *maybeExecutionCount == 0 &&
               "executing selected receiver record must have a transfer node");
        return success();
      }
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
      const PipeReceiverDFBKey &receiverDFB = endpoint.receiverDFB;
      if (invariantAddressReceiverDFBs.contains(receiverDFB)) {
        continue;
      }
      FailureOr<LaunchExecutionLocation> maybeLocation =
          getPipeGraphExecutionLocation(
              postOp.getOperation(), getLaunchNodeCoord(receiver),
              transferNode.deviceTransfer, PipeRole::Destination);
      if (failed(maybeLocation)) {
        return failure();
      }
      ActivePipeNetExecution activeExecution = evaluateActivePipeNetExecution(
          activeRecords, *maybeLocation, resolveRecordLoop);
      if (!activeExecution.mayExecute) {
        continue;
      }
      EndpointSlotAssignment &endpointAssignment =
          assignmentByEndpoint[*endpointIt];
      if (!scheduleContextByReceiverDFB.lookup(receiverDFB)) {
        endpointAssignment.valid = false;
        continue;
      }
      if (endpointAssignment.initialSlot) {
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
      std::optional<std::uint64_t> maybeExecutionCount =
          getConcreteTransferExecutionCount(postOp.getOperation(),
                                            *maybeLocation, *pipeRef,
                                            activeRecordIndex, analysisState);
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
          funcOp, coord, resolveRecordLoop,
          [&](Operation *op, ArrayRef<ActivePipeNetRecord> activeRecords) {
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
    if (endpoint.addressSequence.getKind() !=
        ReceiverAddressSequenceProofKind::FullyDynamic) {
      continue;
    }
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
        if (!receiverDFBNode.hasProvenComputedAddressProducerPhase) {
          diag.attachNote(endpoint.receiverDFBInfo.loc)
              << "receiver core_x=" << endpoint.receiver.x
              << ", core_y=" << endpoint.receiver.y << " uses "
              << getReceiverDFBIdentityString(endpoint.receiverDFB) << ": "
              << receiverDFBNode.computedAddressProducerPhaseFailureReason;
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

// A transfer node pairs one send with every receiver, allowing this check to
// compare byte counts, formats, and capacities across all endpoints.
static LogicalResult
verifyTransferPayloadCompatibility(const PipeTransferNode &transferNode) {
  auto sendOp = llvm::cast<PipeTransferSendOp>(transferNode.sendOp);
  auto sourceDFBType =
      mlir::cast<CircularBufferType>(sendOp.getSrc().getType());
  int64_t sourceElementCount = sourceDFBType.getElementsPerBlock();
  IntegerAttr sendByteCount = sendOp.getByteCountAttr();

  for (Operation *postOperation : transferNode.receiverPostOps) {
    auto postOp = llvm::cast<PipeTransferPostOp>(postOperation);
    IntegerAttr postByteCount = postOp.getByteCountAttr();
    if (static_cast<bool>(sendByteCount) != static_cast<bool>(postByteCount) ||
        (sendByteCount && sendByteCount.getInt() != postByteCount.getInt())) {
      auto diag = postOp.emitError(
          "pipe sender and receiver must use the same byte_count");
      diag.attachNote(sendOp.getLoc()) << "corresponding pipe send is here";
      return failure();
    }

    if (sendByteCount) {
      Value destinationDFB = getAttachedCB(postOp.getDst());
      assert(destinationDFB &&
             "pipe transfer verifier requires an attached receiver DFB");
      auto destinationDFBType =
          cast<CircularBufferType>(destinationDFB.getType());
      auto sourceTile =
          dyn_cast<ttcore::TileType>(sourceDFBType.getElementType());
      auto destinationTile =
          dyn_cast<ttcore::TileType>(destinationDFBType.getElementType());
      FailureOr<uint64_t> sourceCapacity =
          getDFBTransferCapacityBytes(sendOp.getSrc());
      FailureOr<uint64_t> destinationCapacity =
          getDFBTransferCapacityBytes(postOp.getDst());
      uint64_t byteCount = static_cast<uint64_t>(sendByteCount.getInt());
      if (!sourceTile || !destinationTile ||
          sourceTile.getDataType() != destinationTile.getDataType() ||
          failed(sourceCapacity) || failed(destinationCapacity) ||
          byteCount > *sourceCapacity || byteCount > *destinationCapacity) {
        auto diag = postOp.emitError()
                    << "pipe receiver cannot accept the sender's byte-counted "
                       "payload of "
                    << byteCount << " byte(s)";
        diag.attachNote(sendOp.getLoc()) << "corresponding pipe send is here";
        return failure();
      }
      continue;
    }

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

LogicalResult
PipeGraph::proveReceiverProducerStreams(PipeGraphAnalysisState &analysisState) {
  for (PipeReceiverDFBNode &node : receiverDFBNodes) {
    PipeReceiverDFBKey receiverDFB = node.receiverDFB;
    LaunchNodeDomain receiverDomain =
        getSingleLaunchNodeDomain(getLaunchNodeCoord(receiverDFB.receiver));

    SmallVector<PipeTransferPostOp> posts;
    forEachReceiverDFBLogicalStreamEvent(
        analysisState.receiverPostsByStream, receiverDFB,
        [&](PipeTransferPostOp postOp) {
          if (isPostForReceiverDFB(postOp, receiverDFB, *this)) {
            posts.push_back(postOp);
          }
        });
    if (posts.empty()) {
      constexpr llvm::StringLiteral reason = "no matching receiver posts";
      debugRejectPipeOnlyProducerStream(receiverDFB, reason);
      debugRejectComputedAddressProducerPhase(receiverDFB, reason);
      node.pipeOnlyProducerStreamFailureReason = reason;
      node.computedAddressProducerPhaseFailureReason = reason;
      continue;
    }
    SmallVector<LaunchExecutionLocation> receiverLocations;
    for (PipeTransferPostOp postOp : posts) {
      const PipeReceiverEndpoint *endpoint =
          findPostReceiverEndpoint(postOp, receiverDFB, *this);
      assert(endpoint && "matching receiver post must have an endpoint");
      DeviceTransferAttr deviceTransfer =
          getPipeTransferNode(endpoint->transferNode).deviceTransfer;
      FailureOr<LaunchExecutionLocation> maybeLocation =
          getPipeGraphExecutionLocation(
              postOp.getOperation(), getLaunchNodeCoord(receiverDFB.receiver),
              deviceTransfer, PipeRole::Destination);
      if (failed(maybeLocation)) {
        return failure();
      }
      if (!llvm::is_contained(receiverLocations, *maybeLocation)) {
        receiverLocations.push_back(*maybeLocation);
      }
    }
    assert(!receiverLocations.empty() &&
           "receiver DFB must have a matching post");
    auto receiverInfoIt = receiverDFBByPost.find(posts.front().getOperation());
    assert(receiverInfoIt != receiverDFBByPost.end() &&
           "receiver post must have DFB geometry");
    int64_t physicalBlockCount = receiverInfoIt->second.blockCount;

    bool pipeOnlyValid = true;
    bool computedAddressPhaseValid = true;
    auto rejectPipeOnly = [&](llvm::StringRef reason) {
      if (!pipeOnlyValid) {
        return;
      }
      debugRejectPipeOnlyProducerStream(receiverDFB, reason);
      node.pipeOnlyProducerStreamFailureReason = reason;
      pipeOnlyValid = false;
    };
    auto rejectComputedAddressPhase = [&](llvm::StringRef reason) {
      if (!computedAddressPhaseValid) {
        return;
      }
      debugRejectComputedAddressProducerPhase(receiverDFB, reason);
      node.computedAddressProducerPhaseFailureReason = reason;
      computedAddressPhaseValid = false;
    };
    auto rejectBoth = [&](llvm::StringRef reason) {
      rejectPipeOnly(reason);
      rejectComputedAddressPhase(reason);
    };
    LogicalResult result = success();
    llvm::DenseMap<Operation *, SmallVector<Operation *>> pushesByPost;

    forEachReceiverDFBPhysicalStreamEvent(
        analysisState.pushesByPhysicalStream, receiverDFB,
        [&](CBPushOp pushOp) {
          if (failed(result)) {
            return;
          }
          LaunchNodeDomain pushDomain =
              lookupOperationLaunchDomain(pushOp.getOperation(), analysisState);
          if (!launchNodeDomainsOverlap(pushDomain, receiverDomain)) {
            return;
          }
          if (isProvenInactiveAtAllLocations(
                  pushOp.getOperation(), receiverLocations, analysisState)) {
            return;
          }
          std::optional<int64_t> maybePushedBlocks =
              getDFBTransactionBlockCount(pushOp);
          FailureOr<int64_t> maybePushDFBId =
              analysisState.dfbLogicalIdentities->getLogicalId(
                  getReceiverDFB(pushOp));
          assert(succeeded(maybePushDFBId) &&
                 "collected push must have a logical DFB identity");
          bool sameLogicalStream = *maybePushDFBId == receiverDFB.dfbId;
          CBReserveOp reserveOp;
          SmallVector<PipeTransferPostOp> ownedPosts;
          if (sameLogicalStream) {
            reserveOp = findUniqueDFBReleaseIntervalOwner<CBReserveOp>(
                pushOp.getOperation(), analysisState);
            if (reserveOp) {
              ownedPosts = getPostsOwnedByReserve(reserveOp, posts);
            }
          }

          if (ownedPosts.empty()) {
            if (sameLogicalStream) {
              rejectPipeOnly(reserveOp
                                 ? "push reserve owns no matching receiver post"
                                 : "push has no unique receiver reserve owner");
            } else if (analysisState.dfbIndexMode ==
                       PipeDFBIndexMode::Provisional) {
              return;
            } else if (analysisState.dfbIndexMode ==
                       PipeDFBIndexMode::DeclaredPhysical) {
              rejectBoth(
                  "physical DFB index aliases another logical DFB without "
                  "finalized allocation metadata");
              return;
            }
            if (!maybePushedBlocks) {
              rejectComputedAddressPhase(
                  "non-pipe push block count is not a whole DFB block count");
            } else if (*maybePushedBlocks != physicalBlockCount) {
              rejectComputedAddressPhase(
                  "non-pipe push does not advance one full physical DFB");
            }
            return;
          }
          // Unresolved outer control is safe only when the ownership and
          // receiver-context checks below prove the complete protocol.
          if (!isNocKernelThread(pushOp)) {
            rejectBoth("push is not in a receiver NOC thread");
            return;
          }
          if (!maybePushedBlocks) {
            rejectBoth("push block count is not a whole DFB block count");
            return;
          }
          int64_t postedBlocks = 0;
          for (PipeTransferPostOp postOp : ownedPosts) {
            const PipeReceiverEndpoint *endpoint =
                findPostReceiverEndpoint(postOp, receiverDFB, *this);
            assert(endpoint &&
                   "matching receiver post must have a receiver endpoint");
            DeviceTransferAttr deviceTransfer =
                getPipeTransferNode(endpoint->transferNode).deviceTransfer;
            FailureOr<LaunchExecutionLocation> maybeLocation =
                getPipeGraphExecutionLocation(
                    postOp.getOperation(),
                    getLaunchNodeCoord(receiverDFB.receiver), deviceTransfer,
                    PipeRole::Destination);
            if (failed(maybeLocation)) {
              result = failure();
              return;
            }
            if (!hasMatchingReceiveWaitBeforePush(
                    postOp, pushOp, analysisState.receiveWaitsByPost,
                    analysisState.receiveWaitAnysByPost, *maybeLocation,
                    analysisState)) {
              auto diag =
                  pushOp.emitOpError()
                  << "publishes a pipe receiver DFB reservation without "
                     "a preceding receive wait in the same control "
                     "context";
              diag.attachNote(postOp.getLoc())
                  << "matching receiver post occurrence is here";
              result = failure();
              return;
            }
            std::optional<int64_t> maybeSpan =
                getReceiverSlotSpanBlocksForPost(postOp, receiverDFBByPost);
            if (!maybeSpan) {
              rejectBoth("post has no receiver slot span");
              return;
            }
            postedBlocks += *maybeSpan;
            SmallVectorImpl<Operation *> &existingPushes =
                pushesByPost[postOp.getOperation()];
            if (llvm::any_of(existingPushes, [&](Operation *existingPush) {
                  return !mlir::insideMutuallyExclusiveRegions(
                      existingPush, pushOp.getOperation());
                })) {
              rejectBoth("post is consumed by multiple co-executing pushes");
              return;
            }
            existingPushes.push_back(pushOp.getOperation());
          }
          if (*maybePushedBlocks != postedBlocks) {
            rejectBoth(
                "push block count does not match posted receiver slot span");
          }
        });
    if (failed(result)) {
      return failure();
    }
    for (PipeTransferPostOp postOp : posts) {
      if (!pushesByPost.contains(postOp.getOperation())) {
        rejectBoth("post is not consumed by a receiver push");
        break;
      }
    }

    if (pipeOnlyValid) {
      node.hasProvenPipeOnlyProducerStream = true;
      node.pipeOnlyProducerStreamFailureReason.clear();
      debugAcceptPipeOnlyProducerStream(receiverDFB);
    }
    if (computedAddressPhaseValid) {
      node.hasProvenComputedAddressProducerPhase = true;
      node.computedAddressProducerPhaseFailureReason.clear();
      debugAcceptComputedAddressProducerPhase(receiverDFB);
    }
  }
  return success();
}

void PipeGraph::appendReceiverDFBPops(const PipeReceiverDFBKey &receiverDFB,
                                      SmallVectorImpl<CBPopOp> &pops) const {
  forEachReceiverDFBLogicalStreamEvent(
      receiverPopsByStream, receiverDFB,
      [&](CBPopOp popOp) { pops.push_back(popOp); });
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
    if (!receiverDFBNode.hasProvenComputedAddressProducerPhase ||
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

static Operation *
getSelectedRecordLoop(const PipeReference &pipeRef,
                      const PipeGraphAnalysisState &analysisState) {
  assert(pipeRef.isSelected() && "record loops require a selected pipe");
  Operation *recordLoop = pipeRef.getSelectedOperation()->getParentOp();
  while (recordLoop && !analysisState.pipeRecordLoops.contains(recordLoop)) {
    recordLoop = recordLoop->getParentOp();
  }
  assert(recordLoop &&
         "selected pipe operation must be nested in its record loop");
  return recordLoop;
}

static std::optional<std::uint64_t> getSelectedRecordExecutionCount(
    Operation *op, const LaunchExecutionLocation &location,
    const PipeReference &pipeRef, std::uint64_t recordIndex,
    PipeGraphAnalysisState &analysisState) {
  assert(pipeRef.isSelected() &&
         recordIndex < pipeRef.getRecords().getPipes().size() &&
         "selected pipe execution requires a valid record index");
  Operation *recordLoop = getSelectedRecordLoop(pipeRef, analysisState);
  auto forOp = cast<scf::ForOp>(recordLoop);
  const PipeNetRecordLoop &recordLoopInfo =
      analysisState.pipeRecordLoops.at(recordLoop);
  std::optional<std::uint64_t> maybeInductionValue =
      getPipeNetRecordLoopInductionValue(recordLoopInfo, location, recordIndex);
  if (!maybeInductionValue) {
    return std::nullopt;
  }

  auto &recordCache = analysisState.recordExecutionCountAnalyses[recordLoop];
  if (!recordCache.sharedState) {
    recordCache.sharedState =
        std::make_unique<ExecutionCountAnalysisSharedState>(forOp.getRegion());
  }
  auto context = std::make_pair(location, recordIndex);
  ExecutionCountAnalysis &analysis =
      recordCache.analysesByContext.getOrCreate(context, [&] {
        Value inductionVariable = forOp.getInductionVar();
        llvm::APInt inductionValue(IndexType::kInternalStorageBitWidth,
                                   *maybeInductionValue);
        PipeRecordAttr record = pipeRef.getRecords().getPipes()[recordIndex];
        return std::make_unique<ExecutionCountAnalysis>(
            *recordCache.sharedState,
            [inductionVariable, inductionValue, record, location,
             &analysisState](Value value) -> std::optional<llvm::APInt> {
              if (std::optional<llvm::APInt> recordValue =
                      evaluateSelectedPipeRecordValue(value, record)) {
                return recordValue;
              }
              if (value == inductionVariable) {
                return inductionValue;
              }
              return evaluateIntegerAtLaunchLocation(value, location,
                                                     analysisState);
            },
            [location, &analysisState](Region &region) {
              return getRegionInvocationCountAtLaunchLocation(region, location,
                                                              analysisState);
            });
      });
  return analysis.getExecutionCount(op);
}

static std::optional<std::uint64_t> getConcreteTransferExecutionCount(
    Operation *op, const LaunchExecutionLocation &location,
    const PipeReference &pipeRef, std::optional<std::uint64_t> recordIndex,
    PipeGraphAnalysisState &analysisState) {
  if (pipeRef.isStatic()) {
    assert(!recordIndex && "static pipe execution has no record index");
    return getExactExecutionCountAtLaunchLocation(op, location, analysisState);
  }

  assert(recordIndex && "selected pipe execution requires a record index");
  Operation *recordLoop = getSelectedRecordLoop(pipeRef, analysisState);
  std::optional<std::uint64_t> maybeLoopInvocationCount =
      getExactExecutionCountAtLaunchLocation(recordLoop, location,
                                             analysisState);
  std::optional<std::uint64_t> maybeRecordCount =
      getSelectedRecordExecutionCount(op, location, pipeRef, *recordIndex,
                                      analysisState);
  return maybeLoopInvocationCount && maybeRecordCount
             ? llvm::checkedMulUnsigned(*maybeLoopInvocationCount,
                                        *maybeRecordCount)
             : std::nullopt;
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
    DeviceTransferAttr deviceTransfer;
    SmallVector<PipeSendCandidate> sends;
    llvm::DenseMap<PipeReceiverCoord, SmallVector<PipePostCandidate>>
        postsByReceiver;
  };

  using PipeTransferIdentity = std::pair<PipeKey, DeviceTransferAttr>;
  llvm::MapVector<PipeTransferIdentity, PipeTransferCandidates>
      candidatesByPipe;
  // Static operations require a conservative role bound. Selected operations
  // are restricted per record by their enclosing PipeNet foreach semantics.
  for (Operation *op : analysisState.transferProtocolOps) {
    if (auto sendOp = dyn_cast<PipeTransferSendOp>(op)) {
      PipeTransferCreateOp createOp =
          transferIndex.getTransferCreate(sendOp.getOperation());
      FailureOr<PipeReference> pipeRef =
          getPipeReference(sendOp, createOp.getPipe());
      if (failed(pipeRef)) {
        return failure();
      }
      DeviceTransferAttr staticDeviceTransfer =
          createOp.getDeviceTransferAttr();
      LaunchNodeDomain sendDomain =
          lookupOperationLaunchDomain(sendOp.getOperation(), analysisState);
      SmallVector<PipeType> pipeTypes =
          getPipeTypesFromReference(sendOp.getContext(), *pipeRef);
      for (auto [recordIndex, pipeType] : llvm::enumerate(pipeTypes)) {
        std::optional<std::uint64_t> selectedRecordIndex =
            pipeRef->isSelected() ? std::optional<std::uint64_t>(recordIndex)
                                  : std::nullopt;
        DeviceTransferAttr deviceTransfer = getPipeRecordDeviceTransfer(
            *pipeRef, recordIndex, staticDeviceTransfer);
        PipeKey pipeKey = getPipeKey(pipeType);
        LaunchNodeCoord source{pipeKey.srcX, pipeKey.srcY};
        if (pipeRef->isStatic() && analysisState.hasLaunchGrid &&
            !sendDomain.isUpperBoundSubsetOf(
                getSingleLaunchNodeDomain(source))) {
          sendOp.emitError(
              "cannot prove that this pipe send executes only on its source "
              "node");
          return failure();
        }
        if (sendDomain.known &&
            !knownLaunchNodeDomainContains(sendDomain, source)) {
          continue;
        }
        FailureOr<LaunchExecutionLocation> maybeLocation =
            getPipeGraphExecutionLocation(sendOp.getOperation(), source,
                                          deviceTransfer, PipeRole::Source);
        if (failed(maybeLocation)) {
          return failure();
        }
        std::optional<std::uint64_t> maybeExecutionCount =
            getConcreteTransferExecutionCount(
                sendOp.getOperation(), *maybeLocation, *pipeRef,
                selectedRecordIndex, analysisState);
        if (maybeExecutionCount && *maybeExecutionCount == 0) {
          continue;
        }
        PipeTransferCandidates &candidates =
            candidatesByPipe[{pipeKey, deviceTransfer}];
        candidates.deviceTransfer = deviceTransfer;
        candidates.sends.push_back({sendOp, selectedRecordIndex});
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
    DeviceTransferAttr staticDeviceTransfer = createOp.getDeviceTransferAttr();
    LaunchNodeDomain postDomain =
        lookupOperationLaunchDomain(postOp.getOperation(), analysisState);
    SmallVector<PipeType> pipeTypes =
        getPipeTypesFromReference(postOp.getContext(), *pipeRef);
    for (auto [recordIndex, pipeType] : llvm::enumerate(pipeTypes)) {
      DeviceTransferAttr deviceTransfer = getPipeRecordDeviceTransfer(
          *pipeRef, recordIndex, staticDeviceTransfer);
      std::optional<std::uint64_t> selectedRecordIndex =
          pipeRef->isSelected() ? std::optional<std::uint64_t>(recordIndex)
                                : std::nullopt;
      PipeKey pipeKey = getPipeKey(pipeType);
      if (pipeRef->isStatic() && analysisState.hasLaunchGrid &&
          !postDomain.isUpperBoundSubsetOf(getPipeDestinationLaunchNodeDomain(
              pipeType, analysisState.baseDomain))) {
        postOp.emitError(
            "cannot prove that this pipe receiver post executes only on its "
            "destination nodes");
        return failure();
      }
      PipeTransferCandidates &candidates =
          candidatesByPipe[{pipeKey, deviceTransfer}];
      candidates.deviceTransfer = deviceTransfer;
      LogicalResult receiverResult = success();
      pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
        if (failed(receiverResult)) {
          return;
        }
        LaunchNodeCoord receiverCoord = getLaunchNodeCoord(receiver);
        if (postDomain.known &&
            !knownLaunchNodeDomainContains(postDomain, receiverCoord)) {
          return;
        }
        FailureOr<LaunchExecutionLocation> maybeLocation =
            getPipeGraphExecutionLocation(postOp.getOperation(), receiverCoord,
                                          deviceTransfer,
                                          PipeRole::Destination);
        if (failed(maybeLocation)) {
          receiverResult = failure();
          return;
        }
        std::optional<std::uint64_t> maybeExecutionCount =
            getConcreteTransferExecutionCount(
                postOp.getOperation(), *maybeLocation, *pipeRef,
                selectedRecordIndex, analysisState);
        if (!maybeExecutionCount || *maybeExecutionCount != 0) {
          candidates.postsByReceiver[receiver].push_back(
              {postOp, selectedRecordIndex});
        }
      });
      if (failed(receiverResult)) {
        return failure();
      }
      if (candidates.postsByReceiver.empty() && candidates.sends.empty()) {
        candidatesByPipe.erase({pipeKey, deviceTransfer});
      }
    }
  }

  llvm::DenseMap<PipeReceiverDFBKey, PipeReceiverDFBNodeId> nodeIdByReceiverDFB;
  for (auto &candidateEntry : candidatesByPipe) {
    PipeKey pipeKey = candidateEntry.first.first;
    PipeTransferCandidates &candidates = candidateEntry.second;
    if (candidates.sends.empty()) {
      Operation *postOp =
          candidates.postsByReceiver.begin()->second.front().op.getOperation();
      std::optional<PipeTransferSendOp> sendWithDifferentDeviceTransfer;
      for (auto &otherEntry : candidatesByPipe) {
        if (otherEntry.first.first == pipeKey &&
            otherEntry.first.second != candidates.deviceTransfer &&
            !otherEntry.second.sends.empty()) {
          sendWithDifferentDeviceTransfer = otherEntry.second.sends.front().op;
          break;
        }
      }
      if (sendWithDifferentDeviceTransfer) {
        auto diagnostic = postOp->emitError(
            "pipe receiver post has no corresponding send for its device "
            "transfer");
        diagnostic.attachNote(sendWithDifferentDeviceTransfer->getLoc())
            << "this send has the same PipeKey but a different device "
               "transfer";
        return failure();
      }
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
        FailureOr<LaunchExecutionLocation> maybeSendLocation =
            getPipeGraphExecutionLocation(
                sendOp.getOperation(), {pipeKey.srcX, pipeKey.srcY},
                candidates.deviceTransfer, PipeRole::Source);
        FailureOr<LaunchExecutionLocation> maybePostLocation =
            getPipeGraphExecutionLocation(
                postOp.getOperation(), getLaunchNodeCoord(receiver),
                candidates.deviceTransfer, PipeRole::Destination);
        if (failed(maybeSendLocation) || failed(maybePostLocation)) {
          correspondenceResult = failure();
          return;
        }
        bool haveEqualExecutionCounts = false;
        std::optional<std::uint64_t> maybeSendCount =
            getConcreteTransferExecutionCount(
                sendOp.getOperation(), *maybeSendLocation, *sendPipeRef,
                candidates.sends[sendIndex].recordIndex, analysisState);
        std::optional<std::uint64_t> maybePostCount =
            getConcreteTransferExecutionCount(
                postOp.getOperation(), *maybePostLocation, *postPipeRef,
                postsIt->second[sendIndex].recordIndex, analysisState);
        if (maybeSendCount && maybePostCount) {
          haveEqualExecutionCounts = *maybeSendCount == *maybePostCount;
        }
        if ((*sendPipeRef).isStatic() && (*postPipeRef).isStatic()) {
          // Static pipe operations can have matching loop and branch structure
          // even when an exact execution count cannot be evaluated.
          haveEqualExecutionCounts =
              haveEqualExecutionCounts ||
              proveEqualExecutionCountAtLaunchLocations(
                  sendOp.getOperation(), *maybeSendLocation,
                  postOp.getOperation(), *maybePostLocation, analysisState);
        } else if ((*sendPipeRef).isSelected() && (*postPipeRef).isSelected() &&
                   !haveEqualExecutionCounts) {
          std::optional<std::uint64_t> maybeSendRecordCount =
              getSelectedRecordExecutionCount(
                  sendOp.getOperation(), *maybeSendLocation, *sendPipeRef,
                  *candidates.sends[sendIndex].recordIndex, analysisState);
          std::optional<std::uint64_t> maybePostRecordCount =
              getSelectedRecordExecutionCount(
                  postOp.getOperation(), *maybePostLocation, *postPipeRef,
                  *postsIt->second[sendIndex].recordIndex, analysisState);
          Operation *sendRecordLoop =
              getSelectedRecordLoop(*sendPipeRef, analysisState);
          Operation *postRecordLoop =
              getSelectedRecordLoop(*postPipeRef, analysisState);
          auto resolveNoFunctionArguments =
              [](BlockArgument) -> std::optional<Value> {
            return std::nullopt;
          };
          auto sendForOp = cast<scf::ForOp>(sendRecordLoop);
          auto postForOp = cast<scf::ForOp>(postRecordLoop);
          PipeRecordAttr sendRecord =
              (*sendPipeRef)
                  .getRecords()
                  .getPipes()[*candidates.sends[sendIndex].recordIndex];
          PipeRecordAttr postRecord =
              (*postPipeRef)
                  .getRecords()
                  .getPipes()[*postsIt->second[sendIndex].recordIndex];
          auto evaluateSendContextValue = [&](Value value) {
            if (value == sendForOp.getInductionVar()) {
              return std::optional<llvm::APInt>(
                  llvm::APInt(IndexType::kInternalStorageBitWidth,
                              *candidates.sends[sendIndex].recordIndex));
            }
            return evaluateSelectedPipeRecordValue(value, sendRecord);
          };
          auto evaluatePostContextValue = [&](Value value) {
            if (value == postForOp.getInductionVar()) {
              return std::optional<llvm::APInt>(
                  llvm::APInt(IndexType::kInternalStorageBitWidth,
                              *postsIt->second[sendIndex].recordIndex));
            }
            return evaluateSelectedPipeRecordValue(value, postRecord);
          };
          bool haveEqualRecordCounts =
              maybeSendRecordCount && maybePostRecordCount &&
              *maybeSendRecordCount == *maybePostRecordCount;
          if (!haveEqualRecordCounts) {
            haveEqualRecordCounts =
                proveEqualUnresolvedExecutionCountWithinScopesAtLaunchLocations(
                    sendOp.getOperation(), sendRecordLoop, *maybeSendLocation,
                    postOp.getOperation(), postRecordLoop, *maybePostLocation,
                    analysisState, evaluateSendContextValue,
                    evaluatePostContextValue, resolveNoFunctionArguments,
                    resolveNoFunctionArguments);
          }
          haveEqualExecutionCounts =
              haveEqualRecordCounts &&
              proveEqualExecutionCountAtLaunchLocations(
                  sendRecordLoop, *maybeSendLocation, postRecordLoop,
                  *maybePostLocation, analysisState);
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
      DeviceTransferAttr deviceTransfer = candidates.deviceTransfer;
      if (!sendCandidate.recordIndex) {
        assert(sendCreate.getDeviceTransferAttr() == deviceTransfer &&
               "static send must retain its device transfer");
      }
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
        if (!endpoint.second.recordIndex) {
          assert(postCreate.getDeviceTransferAttr() == deviceTransfer &&
                 "static post must retain its device transfer");
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
                                                   deviceTransfer,
                                                   sendCandidate.recordIndex,
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
        DeviceRefAttr receiverDevice =
            deviceTransfer ? deviceTransfer.getEdge().getDestination()
                           : DeviceRefAttr();
        PipeReceiverDFBKey receiverDFB{receiverDevice, receiver,
                                       receiverInfo.dfbIndex,
                                       receiverInfo.dfbId};
        auto nodeIt = nodeIdByReceiverDFB.find(receiverDFB);
        PipeReceiverDFBNodeId receiverDFBNodeId = 0;
        if (nodeIt == nodeIdByReceiverDFB.end()) {
          receiverDFBNodeId = receiverDFBNodes.size();
          nodeIdByReceiverDFB.insert({receiverDFB, receiverDFBNodeId});
          receiverDFBNodes.push_back(PipeReceiverDFBNode{
              receiverDFBNodeId, receiverDFB, {}, false, {}, false, {}});
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
                                 postCandidate.recordIndex,
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

DeviceTransferAttr
getPipeRecordDeviceTransfer(const PipeReference &ref, std::size_t recordIndex,
                            DeviceTransferAttr staticDeviceTransfer) {
  if (ref.isStatic()) {
    assert(recordIndex == 0 && "static pipe has exactly one record");
    return staticDeviceTransfer;
  }
  ArrayRef<PipeRecordAttr> records = ref.getRecords().getPipes();
  assert(recordIndex < records.size() && "selected record index out of range");
  return records[recordIndex].getDeviceTransfer();
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

LogicalResult
PipeGraph::addPipeReceiver(Operation *op, PipeTransferCreateOp transferCreateOp,
                           Value dst,
                           const DFBLogicalIdentityAnalysis &dfbIds) {
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
  FailureOr<int64_t> maybeDFBId = dfbIds.getLogicalId(dstDFB);
  if (failed(maybeDFBId)) {
    return op->emitError(
        "could not resolve pipe receiver logical DFB identity");
  }
  BindCBOp receiverDeclaration = getDFBDeclaration(dstDFB);
  if (!receiverDeclaration) {
    return op->emitError("could not trace pipe receiver to a DFB declaration");
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
  ReceiverDFBInfo receiverInfo{
      *maybeDFBIndex,
      *maybeDFBId,
      dfbType,
      static_cast<bool>(receiverDeclaration.getTensorBackingAttr()),
      hasStaticTileOffset,
      staticTileOffset,
      *slotSpanBlocks,
      dfbType.getBlockCount(),
      op->getLoc()};
  bool inserted = receiverDFBByPost.insert({op, receiverInfo}).second;
  assert(inserted && "receiver post visited more than once");
  return success();
}

FailureOr<PipeGraph>
PipeGraph::build(ModuleOp mod, const PipeTransferIndex &transferIndex,
                 const PipeForeachLoweringInfo &foreachLoweringInfo,
                 PipeDFBIndexMode dfbIndexMode,
                 PipeGraphLaunchDomainMode launchDomainMode) {
  PipeGraph graph;
  PipeGraphAnalysisState analysisState;
  WalkResult protocolSearch = mod.walk([&](Operation *operation) {
    return isa<PipeTransferCreateOp, PipeTransferPostOp, PipeTransferSendOp,
               PipeTransferWaitOp>(operation)
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  if (!protocolSearch.wasInterrupted() &&
      launchDomainMode == PipeGraphLaunchDomainMode::WhenPipesPresent) {
    return std::move(graph);
  }
  analysisState.pipeRecordIfThenDomains = foreachLoweringInfo.ifThenDomains;
  if (!protocolSearch.wasInterrupted()) {
    if (failed(collectLaunchNodeDomains(mod, analysisState))) {
      return failure();
    }
    graph.hasAnalyzedLaunchGrid = analysisState.hasLaunchGrid;
    graph.operationLaunchDomains =
        std::move(analysisState.operationLaunchDomains);
    return std::move(graph);
  }
  analysisState.dfbLogicalIdentities =
      std::make_unique<DFBLogicalIdentityAnalysis>(mod.getOperation());
  if (!analysisState.dfbLogicalIdentities->succeeded()) {
    analysisState.dfbLogicalIdentities->getErrorOperation()->emitError(
        analysisState.dfbLogicalIdentities->getErrorMessage());
    return failure();
  }
  analysisState.dfbIndexMode = dfbIndexMode;
  if (failed(collectPipeGraphOperations(mod, transferIndex, analysisState))) {
    return failure();
  }

  analysisState.pipeRecordControlOps.insert(
      foreachLoweringInfo.controlOps.begin(),
      foreachLoweringInfo.controlOps.end());
  analysisState.pipeRecordLoops = foreachLoweringInfo.recordLoops;

  for (PipeTransferPostOp postOp : analysisState.receiverPosts) {
    PipeTransferCreateOp createOp =
        transferIndex.getTransferCreate(postOp.getOperation());
    if (failed(graph.addPipeReceiver(postOp, createOp, postOp.getDst(),
                                     *analysisState.dfbLogicalIdentities))) {
      return failure();
    }
  }
  if (failed(collectLaunchNodeDomains(mod, analysisState))) {
    return failure();
  }
  if (failed(collectDFBLifecycles(mod, analysisState))) {
    return failure();
  }

  if (failed(graph.rebuildEndpointGraph(transferIndex, analysisState))) {
    return failure();
  }
  if (failed(graph.assignReceiverAddressSequences(mod, transferIndex,
                                                  analysisState))) {
    return failure();
  }
  if (failed(graph.proveReceiverProducerStreams(analysisState))) {
    return failure();
  }
  if (failed(graph.verifyCollectiveReceiverAddresses())) {
    return failure();
  }
  graph.hasAnalyzedLaunchGrid = analysisState.hasLaunchGrid;
  graph.operationLaunchDomains =
      std::move(analysisState.operationLaunchDomains);
  graph.dfbLifecycles = std::move(analysisState.dfbLifecycles);
  graph.receiverPopsByStream = std::move(analysisState.popsByStream);
  return std::move(graph);
}

} // namespace mlir::tt::ttl
