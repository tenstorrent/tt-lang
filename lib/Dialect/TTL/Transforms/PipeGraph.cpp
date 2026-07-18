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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <utility>

#define DEBUG_TYPE "ttl-pipe-graph"

namespace mlir::tt::ttl {

struct PipeGraphAnalysisState : LaunchNodeDomainState {
  struct PipeTransferInfo {
    PipeTransferContract transferContract = PipeTransferContract::PointToPoint;
    SmallVector<Operation *> transferCreateOps;
  };

  struct DFBLifecycleOperations {
    SmallVector<Operation *> reserves;
    SmallVector<Operation *> waits;
    SmallVector<Operation *> pushes;
    SmallVector<Operation *> pops;
  };

  llvm::DenseMap<Operation *, LaunchNodeDomain> operationLaunchDomains;
  llvm::DenseMap<Operation *, llvm::DenseMap<DeviceRefAttr, LaunchNodeDomain>>
      operationExecutionDomains;
  DFBReleaseOwnerMaps dfbReleaseOwners;
  llvm::MapVector<PipeKey, PipeTransferInfo> transferInfos;
  SmallVector<PipeTransferPostOp> receiverPosts;
  SmallVector<Operation *> receiverSlotEvents;
  llvm::DenseMap<Operation *, SmallVector<PipeTransferWaitOp>>
      receiveWaitsByPost;
  llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<PipeTransferPostOp>>
      receiverPostsByStream;
  llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<CBPushOp>>
      pushesByStream;
  llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<CBPopOp>> popsByStream;
  llvm::DenseMap<PipeKey, SmallVector<PipeTransferPostOp>> receiverPostsByPipe;
  llvm::DenseMap<PipeKey, SmallVector<PipeTransferSendOp>> sendsByPipe;
  llvm::MapVector<func::FuncOp, DFBLifecycleOperations> dfbLifecycleByFunction;
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
    auto func = op->getParentOfType<func::FuncOp>();
    if (func && !func->hasAttr(kDeviceDomainAttrName)) {
      state.operationExecutionDomains[op][{}] = domain;
    }
  };
  solver.load<LaunchNodeDomainAnalysis>(state, options);
  if (failed(solver.initializeAndRun(mod))) {
    return failure();
  }

  llvm::SetVector<DeviceRefAttr> devices;
  mod.walk([&](func::FuncOp func) {
    auto domain = func->getAttrOfType<DeviceDomainAttr>(kDeviceDomainAttrName);
    if (!domain) {
      return;
    }
    for (DeviceRefAttr device : enumerateDeviceDomain(domain)) {
      devices.insert(device);
    }
  });

  for (DeviceRefAttr device : devices) {
    DataFlowSolver deviceSolver;
    dataflow::loadBaselineAnalyses(deviceSolver);
    LaunchNodeDomainAnalysisOptions deviceOptions;
    deviceOptions.currentDevice = device;
    deviceOptions.narrowPipeNetScopes = true;
    deviceOptions.operationCallback = [&](Operation *op,
                                          const LaunchNodeDomain &domain,
                                          Operation * /*unanalyzableOp*/) {
      auto func = op->getParentOfType<func::FuncOp>();
      if (!func) {
        return;
      }
      auto functionDomain =
          func->getAttrOfType<DeviceDomainAttr>(kDeviceDomainAttrName);
      if (!functionDomain ||
          !getDeviceLinearIndex(functionDomain, device).has_value()) {
        return;
      }
      state.operationExecutionDomains[op][device] = domain;
    };
    deviceSolver.load<LaunchNodeDomainAnalysis>(state, deviceOptions);
    if (failed(deviceSolver.initializeAndRun(mod))) {
      return failure();
    }
  }
  return success();
}

static LaunchNodeDomain
lookupOperationLaunchDomain(Operation *op, DeviceRefAttr device,
                            PipeGraphAnalysisState &state) {
  if (!state.hasLaunchGrid) {
    return LaunchNodeDomain::unknown();
  }
  auto operationIt = state.operationExecutionDomains.find(op);
  if (operationIt == state.operationExecutionDomains.end()) {
    return LaunchNodeDomain::unknown();
  }
  auto deviceIt = operationIt->second.find(device);
  if (deviceIt == operationIt->second.end()) {
    return LaunchNodeDomain{};
  }
  return deviceIt->second;
}

template <typename Callback>
static void forEachOperationExecutionDomain(Operation *op,
                                            PipeGraphAnalysisState &state,
                                            Callback &&callback) {
  if (!state.hasLaunchGrid) {
    callback(DeviceRefAttr(), LaunchNodeDomain::unknown());
    return;
  }
  auto operationIt = state.operationExecutionDomains.find(op);
  if (operationIt == state.operationExecutionDomains.end()) {
    callback(DeviceRefAttr(), LaunchNodeDomain::unknown());
    return;
  }
  for (const auto &[device, domain] : operationIt->second) {
    callback(device, domain);
  }
}

static LaunchNodeCoord getLaunchNodeCoord(PipeReceiverCoord receiver) {
  return {receiver.x, receiver.y};
}

static DeviceRefAttr getReceiverDevice(PipeTransferCreateOp transferCreate) {
  auto createPipe = transferCreate.getPipe().getDefiningOp<CreatePipeOp>();
  if (!createPipe) {
    return {};
  }
  DeviceTransferAttr transfer = createPipe.getDeviceTransferAttr();
  return transfer ? transfer.getEdge().getDestination() : DeviceRefAttr();
}

static std::optional<PipeReceiverDFBStreamKey>
getReceiverDFBStreamKey(Value dfb, DeviceRefAttr receiverDevice) {
  std::optional<int64_t> dfbIndex = getCBIndex(dfb);
  if (!dfbIndex) {
    return std::nullopt;
  }
  return std::make_pair(receiverDevice, *dfbIndex);
}

static void recordPipeTransfer(PipeTransferCreateOp transferCreate,
                               PipeGraphAnalysisState &state) {
  if (!transferCreate) {
    return;
  }
  auto addPipe = [&](PipeKey pipeKey, PipeTransferContract contract) {
    auto transferIt = state.transferInfos.find(pipeKey);
    if (transferIt == state.transferInfos.end()) {
      state.transferInfos.insert(
          {pipeKey, {contract, {transferCreate.getOperation()}}});
      return;
    }
    if (!llvm::is_contained(transferIt->second.transferCreateOps,
                            transferCreate.getOperation())) {
      transferIt->second.transferCreateOps.push_back(
          transferCreate.getOperation());
    }
    // Cloned regions may contribute different contracts for one pipe. Preserve
    // the stronger contract so all receiver address checks remain valid.
    if (isCollectiveTransfer(contract)) {
      transferIt->second.transferContract = PipeTransferContract::Collective;
    }
  };

  FailureOr<PipeReference> pipeRef =
      getPipeReference(transferCreate, transferCreate.getPipe());
  assert(succeeded(pipeRef) && "pipe transfer create verifier failed");
  if ((*pipeRef).isStatic()) {
    addPipe(getPipeKey((*pipeRef).pipeType),
            getPipeTransferContract(transferCreate));
    return;
  }
  PipeNetRecordsAttr records = (*pipeRef).getRecords();
  for (PipeRecordAttr record : records.getPipes()) {
    addPipe(getPipeKey(record, records.getPipeNetId()),
            getPipeTransferContract(record));
  }
}

static PipeGraphAnalysisState::DFBLifecycleOperations &
getDFBLifecycleOperations(Operation *op, PipeGraphAnalysisState &state) {
  auto func = op->getParentOfType<func::FuncOp>();
  assert(func && "DFB lifecycle operation must be nested in a function");
  return state.dfbLifecycleByFunction[func];
}

static void recordReceiverPost(PipeTransferPostOp postOp,
                               PipeGraphAnalysisState &state) {
  PipeTransferCreateOp transferCreate =
      findPipeTransferCreateForTransfer(postOp.getTransfer());
  recordPipeTransfer(transferCreate, state);
  state.receiverPosts.push_back(postOp);
  state.receiverSlotEvents.push_back(postOp.getOperation());
  if (!transferCreate) {
    return;
  }

  FailureOr<PipeReference> pipeRef =
      getPipeReference(postOp, transferCreate.getPipe());
  assert(succeeded(pipeRef) && "pipe transfer post verifier failed");
  for (PipeType pipeType :
       getPipeTypesFromReference(postOp.getContext(), *pipeRef)) {
    state.receiverPostsByPipe[getPipeKey(pipeType)].push_back(postOp);
  }
}

static void recordReceiveWait(PipeTransferWaitOp waitOp,
                              PipeGraphAnalysisState &state) {
  PipeTransferPostOp postOp = findPipeTransferPostForToken(waitOp.getToken());
  if (!postOp) {
    return;
  }
  state.receiveWaitsByPost[postOp.getOperation()].push_back(waitOp);
}

static void recordPipeSend(PipeTransferSendOp sendOp,
                           PipeGraphAnalysisState &state) {
  PipeTransferCreateOp transferCreate =
      findPipeTransferCreateForTransfer(sendOp.getTransfer());
  recordPipeTransfer(transferCreate, state);
  if (!transferCreate) {
    return;
  }
  FailureOr<PipeReference> pipeRef =
      getPipeReference(sendOp, transferCreate.getPipe());
  assert(succeeded(pipeRef) && "pipe transfer send verifier failed");
  for (PipeType pipeType :
       getPipeTypesFromReference(sendOp.getContext(), *pipeRef)) {
    state.sendsByPipe[getPipeKey(pipeType)].push_back(sendOp);
  }
}

static void recordDFBPush(CBPushOp pushOp, PipeGraphAnalysisState &state) {
  getDFBLifecycleOperations(pushOp, state)
      .pushes.push_back(pushOp.getOperation());
}

static void recordDFBPop(CBPopOp popOp, PipeGraphAnalysisState &state) {
  state.receiverSlotEvents.push_back(popOp.getOperation());
  getDFBLifecycleOperations(popOp, state).pops.push_back(popOp.getOperation());
}

static void collectPipeGraphOperations(ModuleOp mod,
                                       PipeGraphAnalysisState &state) {
  mod.walk<WalkOrder::PreOrder>([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<PipeTransferPostOp>([&](PipeTransferPostOp postOp) {
          recordReceiverPost(postOp, state);
        })
        .Case<PipeTransferSendOp>(
            [&](PipeTransferSendOp sendOp) { recordPipeSend(sendOp, state); })
        .Case<PipeTransferWaitOp>([&](PipeTransferWaitOp waitOp) {
          recordReceiveWait(waitOp, state);
        })
        .Case<CBReserveOp>([&](CBReserveOp reserveOp) {
          getDFBLifecycleOperations(reserveOp, state)
              .reserves.push_back(reserveOp.getOperation());
        })
        .Case<CBWaitOp>([&](CBWaitOp waitOp) {
          getDFBLifecycleOperations(waitOp, state)
              .waits.push_back(waitOp.getOperation());
        })
        .Case<CBPushOp>([&](CBPushOp pushOp) { recordDFBPush(pushOp, state); })
        .Case<CBPopOp>([&](CBPopOp popOp) { recordDFBPop(popOp, state); });
  });

  for (auto &entry : state.dfbLifecycleByFunction) {
    PipeGraphAnalysisState::DFBLifecycleOperations &lifecycle = entry.second;
    buildDFBReleaseOwnerMaps(lifecycle.reserves, lifecycle.waits,
                             lifecycle.pushes, lifecycle.pops,
                             state.dfbReleaseOwners);
  }
}

template <typename OpTy>
static void appendStreamEvent(
    llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<OpTy>> &eventsByStream,
    PipeReceiverDFBStreamKey stream, OpTy op) {
  SmallVector<OpTy> &events = eventsByStream[stream];
  if (!llvm::is_contained(events, op)) {
    events.push_back(op);
  }
}

template <typename OpTy>
static void indexDFBOperationByExecutionDevice(
    OpTy op, Value dfb,
    llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<OpTy>> &eventsByStream,
    PipeGraphAnalysisState &state) {
  std::optional<int64_t> dfbIndex = getCBIndex(dfb);
  if (!dfbIndex) {
    return;
  }
  bool hasDeviceInstance = false;
  forEachOperationExecutionDomain(
      op.getOperation(), state,
      [&](DeviceRefAttr device, const LaunchNodeDomain &domain) {
        if (domain.known && domain.nodes.empty()) {
          return;
        }
        hasDeviceInstance = hasDeviceInstance || static_cast<bool>(device);
        appendStreamEvent(eventsByStream, {device, *dfbIndex}, op);
      });
  if (hasDeviceInstance) {
    appendStreamEvent(eventsByStream, {DeviceRefAttr(), *dfbIndex}, op);
  }
}

static void indexReceiverDFBStreamEvents(PipeGraphAnalysisState &state) {
  for (PipeTransferPostOp postOp : state.receiverPosts) {
    PipeTransferCreateOp transferCreate =
        findPipeTransferCreateForTransfer(postOp.getTransfer());
    if (!transferCreate) {
      continue;
    }
    Value dfb = getAttachedCB(postOp.getDst());
    if (DeviceRefAttr receiverDevice = getReceiverDevice(transferCreate)) {
      auto streamKey = getReceiverDFBStreamKey(dfb, receiverDevice);
      if (streamKey) {
        appendStreamEvent(state.receiverPostsByStream, *streamKey, postOp);
      }
      continue;
    }
    indexDFBOperationByExecutionDevice(postOp, dfb, state.receiverPostsByStream,
                                       state);
  }

  for (auto &entry : state.dfbLifecycleByFunction) {
    PipeGraphAnalysisState::DFBLifecycleOperations &lifecycle = entry.second;
    for (Operation *operation : lifecycle.pushes) {
      auto pushOp = cast<CBPushOp>(operation);
      indexDFBOperationByExecutionDevice(pushOp, pushOp.getCb(),
                                         state.pushesByStream, state);
    }
    for (Operation *operation : lifecycle.pops) {
      auto popOp = cast<CBPopOp>(operation);
      indexDFBOperationByExecutionDevice(popOp, popOp.getCb(),
                                         state.popsByStream, state);
    }
  }
}

template <typename OpTy, typename Callback>
static void forEachReceiverDFBStreamEvent(
    const llvm::DenseMap<PipeReceiverDFBStreamKey, SmallVector<OpTy>>
        &eventsByStream,
    const PipeReceiverDFBKey &receiverDFB, Callback &&callback) {
  PipeReceiverDFBStreamKey streamKey{receiverDFB.receiverDevice,
                                     receiverDFB.dfbIndex};
  auto eventsIt = eventsByStream.find(streamKey);
  if (eventsIt == eventsByStream.end()) {
    return;
  }
  for (OpTy event : eventsIt->second) {
    callback(event);
  }
}

static bool isPostForReceiverDFB(
    PipeTransferPostOp postOp, const PipeReceiverDFBKey &receiverDFB,
    const llvm::MapVector<PipeKey, ReceiverDFBInfo> &receiverDFBs,
    PipeGraphAnalysisState &state) {
  LaunchNodeDomain postDomain = lookupOperationLaunchDomain(
      postOp.getOperation(), receiverDFB.receiverDevice, state);
  if (!knownLaunchNodeDomainContains(
          postDomain, getLaunchNodeCoord(receiverDFB.receiver))) {
    return false;
  }

  auto createOp = findPipeTransferCreateForTransfer(postOp.getTransfer());
  assert(createOp && "pipe transfer graph validated transfer creators");
  FailureOr<PipeReference> pipeRef =
      getPipeReference(postOp, createOp.getPipe());
  assert(succeeded(pipeRef) && "pipe transfer graph validated pipe references");
  for (PipeType pipeType :
       getPipeTypesFromReference(postOp.getContext(), *pipeRef)) {
    PipeKey pipeKey = getPipeKey(pipeType);
    if (!pipeKey.containsReceiver(receiverDFB.receiver)) {
      continue;
    }
    auto receiverIt = receiverDFBs.find(pipeKey);
    if (receiverIt != receiverDFBs.end() &&
        receiverIt->second.dfbIndex == receiverDFB.dfbIndex &&
        (!receiverIt->second.receiverDevice ||
         receiverIt->second.receiverDevice == receiverDFB.receiverDevice)) {
      return true;
    }
  }
  return false;
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
    PipeTransferPostOp postOp, const PipeReceiverDFBKey &receiverDFB,
    const llvm::MapVector<PipeKey, ReceiverDFBInfo> &receiverDFBs) {
  auto createOp = findPipeTransferCreateForTransfer(postOp.getTransfer());
  assert(createOp && "pipe transfer graph validated transfer creators");
  FailureOr<PipeReference> pipeRef =
      getPipeReference(postOp, createOp.getPipe());
  assert(succeeded(pipeRef) && "pipe transfer graph validated pipe references");
  std::optional<int64_t> span;
  for (PipeType pipeType :
       getPipeTypesFromReference(postOp.getContext(), *pipeRef)) {
    PipeKey pipeKey = getPipeKey(pipeType);
    if (!pipeKey.containsReceiver(receiverDFB.receiver)) {
      continue;
    }
    auto receiverIt = receiverDFBs.find(pipeKey);
    if (receiverIt == receiverDFBs.end() ||
        receiverIt->second.dfbIndex != receiverDFB.dfbIndex ||
        (receiverIt->second.receiverDevice &&
         receiverIt->second.receiverDevice != receiverDFB.receiverDevice)) {
      continue;
    }
    int64_t candidateSpan = receiverIt->second.receiverSlotSpanBlocks;
    if (span && *span != candidateSpan) {
      return std::nullopt;
    }
    span = candidateSpan;
  }
  return span;
}

static bool isBeforeInSameBlock(Operation *before, Operation *after) {
  return before->getBlock() == after->getBlock() &&
         before->isBeforeInBlock(after);
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
// in-order consumption. A pop that does not exactly drain tracked pipe receive
// slots would leave the static slot model out of sync with the DFB ring.
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
  if (remainingBlocks != 0) {
    return popOp.emitError()
           << "pipe receiver DFB pop releases " << releasedBlocks
           << " block(s), but only " << (releasedBlocks - remainingBlocks)
           << " live pipe receive block(s) are tracked; receiver pops must "
              "release only live pipe receive slots";
  }
  state.liveSlots.erase(state.liveSlots.begin(),
                        state.liveSlots.begin() + slotsToRelease);
  return success();
}

} // namespace

LogicalResult PipeGraph::addReceiverDFB(
    int64_t srcX, int64_t srcY, int64_t dstStartX, int64_t dstStartY,
    int64_t dstEndX, int64_t dstEndY, int64_t pipeNetId, int64_t dfbIndex,
    DeviceRefAttr receiverDevice, CircularBufferType dfbType,
    bool hasStaticTileOffset, int64_t staticTileOffset,
    int64_t receiverSlotSpanBlocks, Operation *receiverReserveOp,
    PipeTransferContract transferContract,
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
                       {receiverDevice, dfbIndex, dfbType, hasStaticTileOffset,
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
PipeGraph::assignReceiverSlotIndices(PipeGraphAnalysisState &analysisState) {
  llvm::DenseMap<PipeReceiverDFBKey, ReceiverSlotState> slotStateByReceiverDFB;
  llvm::DenseMap<PipeReceiverDFBKey, llvm::DenseMap<Operation *, int64_t>>
      slotByReceiverReserve;
  auto processPost = [&](PipeTransferPostOp postOp) -> LogicalResult {
    auto createOp = findPipeTransferCreateForTransfer(postOp.getTransfer());
    if (!createOp) {
      return postOp.emitError(
          "pipe transfer post must reference a transfer derived from "
          "ttl.pipe_transfer.create");
    }
    FailureOr<PipeReference> pipeRef =
        getPipeReference(postOp, createOp.getPipe());
    if (failed(pipeRef)) {
      return failure();
    }
    for (PipeType pipeType :
         getPipeTypesFromReference(postOp.getContext(), *pipeRef)) {
      PipeKey pipeKey = getPipeKey(pipeType);
      auto receiverIt = receiverDFBs.find(pipeKey);
      if (receiverIt == receiverDFBs.end()) {
        continue;
      }

      ReceiverDFBInfo &receiverInfo = receiverIt->second;
      if (receiverInfo.receiverSlotIndex.has_value()) {
        continue;
      }

      std::optional<int64_t> uniformSlot;
      bool hasReceiver = false;
      bool nonUniformSlot = false;
      LogicalResult result = success();
      auto processDevice = [&](DeviceRefAttr receiverDevice,
                               const LaunchNodeDomain &postDomain) {
        if (postDomain.known && postDomain.nodes.empty()) {
          return;
        }
        pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
          if (failed(result) ||
              (postDomain.known &&
               !knownLaunchNodeDomainContains(postDomain,
                                              getLaunchNodeCoord(receiver)))) {
            return;
          }
          hasReceiver = true;
          PipeReceiverDFBKey receiverDFB{receiverDevice, receiver,
                                         receiverInfo.dfbIndex};
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
      };
      if (receiverInfo.receiverDevice) {
        processDevice(receiverInfo.receiverDevice,
                      lookupOperationLaunchDomain(postOp.getOperation(),
                                                  receiverInfo.receiverDevice,
                                                  analysisState));
      } else {
        forEachOperationExecutionDomain(postOp.getOperation(), analysisState,
                                        processDevice);
      }
      if (failed(result)) {
        return failure();
      }
      if (!hasReceiver || nonUniformSlot) {
        receiverInfo.receiverSlotIndex = std::nullopt;
        continue;
      }
      receiverInfo.receiverSlotIndex = *uniformSlot;
    }
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
    LogicalResult result = success();
    forEachOperationExecutionDomain(
        popOp.getOperation(), analysisState,
        [&](DeviceRefAttr receiverDevice, const LaunchNodeDomain &popDomain) {
          if (failed(result) || !popDomain.known) {
            return;
          }
          for (LaunchNodeCoord coord : popDomain.nodes) {
            PipeReceiverDFBKey receiverDFB{
                receiverDevice, PipeReceiverCoord{coord.x, coord.y}, *dfbIndex};
            auto stateIt = slotStateByReceiverDFB.find(receiverDFB);
            if (stateIt == slotStateByReceiverDFB.end()) {
              continue;
            }
            if (failed(releaseReceiverSlots(popOp, stateIt->second,
                                            *releasedBlocks))) {
              result = failure();
              return;
            }
          }
        });
    return result;
  };

  for (Operation *op : analysisState.receiverSlotEvents) {
    if (auto postOp = dyn_cast<PipeTransferPostOp>(op)) {
      if (failed(processPost(postOp))) {
        return failure();
      }
      continue;
    }
    if (auto popOp = dyn_cast<CBPopOp>(op)) {
      if (failed(processPop(popOp))) {
        return failure();
      }
    }
  }
  return success();
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
    // assignReceiverPhysicalSlot only returns a slot with slot + span <=
    // block_count, and a reused slot shares the reserve op and receiver DFB
    // (hence the same span and block_count), so this always holds.
    assert(*info.receiverSlotIndex + info.receiverSlotSpanBlocks <=
               info.blockCount &&
           "receiver slot assignment exceeds block_count");
  }
  return success();
}

static bool provePipeOnlyReceiverStream(
    const PipeReceiverDFBKey &receiverDFB,
    const llvm::MapVector<PipeKey, ReceiverDFBInfo> &receiverDFBs,
    PipeGraphAnalysisState &analysisState) {
  LaunchNodeDomain receiverDomain =
      getSingleLaunchNodeDomain(getLaunchNodeCoord(receiverDFB.receiver));

  SmallVector<PipeTransferPostOp> posts;
  forEachReceiverDFBStreamEvent(analysisState.receiverPostsByStream,
                                receiverDFB, [&](PipeTransferPostOp postOp) {
                                  if (isPostForReceiverDFB(postOp, receiverDFB,
                                                           receiverDFBs,
                                                           analysisState)) {
                                    posts.push_back(postOp);
                                  }
                                });
  if (posts.empty()) {
    debugRejectPipeOnlyStream(receiverDFB, "no matching receiver posts");
    return false;
  }

  bool valid = true;
  auto reject = [&](llvm::StringRef reason) {
    debugRejectPipeOnlyStream(receiverDFB, reason);
    valid = false;
  };
  llvm::DenseSet<Operation *> postsWithPush;
  llvm::DenseSet<Operation *> waitsWithPop;

  forEachReceiverDFBStreamEvent(
      analysisState.pushesByStream, receiverDFB, [&](CBPushOp pushOp) {
        if (!valid) {
          return;
        }
        LaunchNodeDomain pushDomain = lookupOperationLaunchDomain(
            pushOp.getOperation(), receiverDFB.receiverDevice, analysisState);
        if (!launchNodeDomainsOverlap(pushDomain, receiverDomain)) {
          return;
        }
        if (!knownLaunchNodeDomainContains(
                pushDomain, getLaunchNodeCoord(receiverDFB.receiver))) {
          reject("push is not in the receiver NOC domain");
          return;
        }
        if (!isNocKernelThread(pushOp)) {
          reject("push is not in the receiver NOC domain");
          return;
        }
        std::optional<int64_t> pushedBlocks = getPushedBlockCount(pushOp);
        if (!pushedBlocks) {
          reject("push block count is not a whole DFB block count");
          return;
        }
        auto reserveOp = lookupOwner<CBReserveOp>(
            analysisState.dfbReleaseOwners.reserveByPush,
            pushOp.getOperation());
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
                  postOp, pushOp, analysisState.receiveWaitsByPost)) {
            reject("post has no receive wait before push");
            return;
          }
          std::optional<int64_t> span = getReceiverSlotSpanBlocksForPost(
              postOp, receiverDFB, receiverDFBs);
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
    return false;
  }

  for (PipeTransferPostOp postOp : posts) {
    if (!postsWithPush.contains(postOp.getOperation())) {
      reject("post is not consumed by a receiver push");
      break;
    }
  }
  if (!valid) {
    return false;
  }

  forEachReceiverDFBStreamEvent(
      analysisState.popsByStream, receiverDFB, [&](CBPopOp popOp) {
        if (!valid) {
          return;
        }
        LaunchNodeDomain popDomain = lookupOperationLaunchDomain(
            popOp.getOperation(), receiverDFB.receiverDevice, analysisState);
        if (!launchNodeDomainsOverlap(popDomain, receiverDomain)) {
          return;
        }
        if (!knownLaunchNodeDomainContains(
                popDomain, getLaunchNodeCoord(receiverDFB.receiver))) {
          reject("pop is not in the receiver NOC domain");
          return;
        }
        if (!isNocKernelThread(popOp)) {
          reject("pop is not in the receiver NOC domain");
          return;
        }
        std::optional<int64_t> releasedBlocks = getReleasedBlockCount(popOp);
        if (!releasedBlocks) {
          reject("pop block count is not a whole DFB block count");
          return;
        }
        auto waitOp = lookupOwner<CBWaitOp>(
            analysisState.dfbReleaseOwners.waitByPop, popOp.getOperation());
        if (!waitOp) {
          reject("pop has no unique receiver wait owner");
          return;
        }
        LaunchNodeDomain waitDomain = lookupOperationLaunchDomain(
            waitOp.getOperation(), receiverDFB.receiverDevice, analysisState);
        if (!knownLaunchNodeDomainContains(
                waitDomain, getLaunchNodeCoord(receiverDFB.receiver))) {
          reject("wait is not in the receiver NOC domain");
          return;
        }
        if (!isNocKernelThread(waitOp)) {
          reject("wait is not in the receiver NOC domain");
          return;
        }
        std::optional<int64_t> waitedBlocks = getWaitedBlockCount(waitOp);
        if (!waitedBlocks) {
          reject("wait and pop use different block counts");
          return;
        }
        if (*waitedBlocks != *releasedBlocks) {
          reject("wait and pop use different block counts");
          return;
        }
        if (!waitsWithPop.insert(waitOp.getOperation()).second) {
          reject("wait owns more than one pop");
        }
      });

  if (valid) {
    debugAcceptPipeOnlyStream(receiverDFB);
  }
  return valid;
}

LogicalResult
PipeGraph::provePipeOnlyReceiverStreams(PipeGraphAnalysisState &analysisState) {
  for (PipeReceiverDFBNode &node : receiverDFBNodes) {
    llvm::SetVector<DeviceRefAttr> receiverDevices;
    if (node.receiverDFB.receiverDevice) {
      receiverDevices.insert(node.receiverDFB.receiverDevice);
    } else {
      for (PipeReceiverEndpointId endpointId : node.writerEndpoints) {
        const PipeReceiverEndpoint &endpoint =
            pipeReceiverEndpoints[endpointId];
        const PipeEdge &pipeEdge = pipeEdges[endpoint.pipeEdge];
        auto postsIt = analysisState.receiverPostsByPipe.find(pipeEdge.pipe);
        if (postsIt == analysisState.receiverPostsByPipe.end()) {
          continue;
        }
        for (PipeTransferPostOp postOp : postsIt->second) {
          forEachOperationExecutionDomain(
              postOp.getOperation(), analysisState,
              [&](DeviceRefAttr device, const LaunchNodeDomain &domain) {
                if (!domain.known ||
                    knownLaunchNodeDomainContains(
                        domain,
                        getLaunchNodeCoord(node.receiverDFB.receiver))) {
                  receiverDevices.insert(device);
                }
              });
        }
      }
    }

    if (receiverDevices.empty()) {
      debugRejectPipeOnlyStream(node.receiverDFB,
                                "no matching receiver execution instance");
      continue;
    }

    node.hasProvenPipeOnlyStream = true;
    for (DeviceRefAttr receiverDevice : receiverDevices) {
      PipeReceiverDFBKey receiverDFB = node.receiverDFB;
      receiverDFB.receiverDevice = receiverDevice;
      if (!provePipeOnlyReceiverStream(receiverDFB, receiverDFBs,
                                       analysisState)) {
        node.hasProvenPipeOnlyStream = false;
      }
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

ArrayRef<PipeTransferPostOp>
PipeGraph::getPipeReceiverPosts(const PipeKey &pipe) const {
  auto postsIt = receiverPostsByPipe.find(pipe);
  if (postsIt == receiverPostsByPipe.end()) {
    return {};
  }
  return postsIt->second;
}

ArrayRef<PipeTransferSendOp>
PipeGraph::getPipeSends(const PipeKey &pipe) const {
  auto sendsIt = sendsByPipe.find(pipe);
  if (sendsIt == sendsByPipe.end()) {
    return {};
  }
  return sendsIt->second;
}

void PipeGraph::appendReceiverDFBPops(const PipeReceiverDFBKey &receiverDFB,
                                      SmallVectorImpl<CBPopOp> &pops) const {
  forEachReceiverDFBStreamEvent(receiverPopsByStream, receiverDFB,
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

void PipeGraph::rebuildEndpointGraph() {
  pipeEdges.clear();
  pipeEdgeIdByPipe.clear();
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
    pipeEdgeIdByPipe[pipeKey] = pipeEdgeId;
    PipeEdge &pipeEdge = pipeEdges.back();

    pipeKey.forEachReceiver([&](PipeReceiverCoord receiver) {
      PipeReceiverDFBKey receiverDFB{receiverInfo.receiverDevice, receiver,
                                     receiverInfo.dfbIndex};
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

FailureOr<PipeReference> getPipeReference(Operation *op, Value pipe) {
  Value tracedPipe = traceUnrealizedCasts(pipe);
  if (auto pipeType = mlir::dyn_cast<PipeType>(tracedPipe.getType())) {
    return PipeReference{PipeReference::Kind::Static, tracedPipe, pipeType,
                         SelectPipeSrcOp(), SelectPipeDstOp()};
  }
  if (auto selectedSrc = tracedPipe.getDefiningOp<SelectPipeSrcOp>()) {
    return PipeReference{PipeReference::Kind::SelectedSrc, tracedPipe,
                         PipeType(), selectedSrc, SelectPipeDstOp()};
  }
  if (auto selectedDst = tracedPipe.getDefiningOp<SelectPipeDstOp>()) {
    return PipeReference{PipeReference::Kind::SelectedDst, tracedPipe,
                         PipeType(), SelectPipeSrcOp(), selectedDst};
  }
  return op->emitError() << "selected pipe operand must be a direct result of "
                            "ttl.select_pipe_src or ttl.select_pipe_dst";
}

SmallVector<PipeType> getPipeTypesFromReference(MLIRContext *context,
                                                const PipeReference &ref) {
  if (ref.isStatic()) {
    return SmallVector<PipeType>{ref.pipeType};
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
  if (!tensorType.hasStaticShape()) {
    return failure();
  }
  return tensorType.getNumElements();
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
                                     const PipeKey &pipe,
                                     DeviceRefAttr receiverDevice,
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
      pipe.srcX, pipe.srcY, pipe.dstStartX, pipe.dstStartY, pipe.dstEndX,
      pipe.dstEndY, pipe.pipeNetId, *dfbIndex, receiverDevice, dfbType,
      hasStaticTileOffset, staticTileOffset, *slotSpanBlocks,
      reserveOp.getOperation(), transferContract, transferCreateOps,
      dfbType.getBlockCount(), op->getLoc());
}

FailureOr<PipeGraph> PipeGraph::build(ModuleOp mod) {
  PipeGraphAnalysisState analysisState;
  collectPipeGraphOperations(mod, analysisState);

  PipeGraph graph;
  for (PipeTransferPostOp postOp : analysisState.receiverPosts) {
    auto createOp = findPipeTransferCreateForTransfer(postOp.getTransfer());
    if (!createOp) {
      postOp.emitError(
          "pipe transfer post must reference a transfer derived from "
          "ttl.pipe_transfer.create");
      return failure();
    }
    FailureOr<PipeReference> pipeRef =
        getPipeReference(postOp, createOp.getPipe());
    if (failed(pipeRef)) {
      return failure();
    }
    for (PipeType pipeType :
         getPipeTypesFromReference(postOp.getContext(), *pipeRef)) {
      PipeKey key = getPipeKey(pipeType);
      auto pipeInfoIt = analysisState.transferInfos.find(key);
      if (pipeInfoIt == analysisState.transferInfos.end()) {
        postOp.emitError(
            "pipe transfer post must reference a known pipe transfer");
        return failure();
      }
      const PipeGraphAnalysisState::PipeTransferInfo &pipeInfo =
          pipeInfoIt->second;
      if (failed(
              addPipeReceiver(graph, postOp, key, getReceiverDevice(createOp),
                              pipeInfo.transferContract,
                              pipeInfo.transferCreateOps, postOp.getDst()))) {
        return failure();
      }
    }
  }

  if (failed(collectLaunchNodeDomains(mod, analysisState))) {
    return failure();
  }
  indexReceiverDFBStreamEvents(analysisState);

  if (failed(graph.assignReceiverSlotIndices(analysisState))) {
    return failure();
  }

  if (failed(graph.verifyReceiverDFBBlockCounts())) {
    return failure();
  }

  graph.rebuildEndpointGraph();
  if (failed(graph.provePipeOnlyReceiverStreams(analysisState))) {
    return failure();
  }
  graph.hasAnalyzedLaunchGrid = analysisState.hasLaunchGrid;
  graph.operationLaunchDomains =
      std::move(analysisState.operationLaunchDomains);
  graph.dfbReleaseOwners = std::move(analysisState.dfbReleaseOwners);
  graph.receiverPostsByPipe = std::move(analysisState.receiverPostsByPipe);
  graph.sendsByPipe = std::move(analysisState.sendsByPipe);
  graph.receiverPopsByStream = std::move(analysisState.popsByStream);
  return std::move(graph);
}

} // namespace mlir::tt::ttl
