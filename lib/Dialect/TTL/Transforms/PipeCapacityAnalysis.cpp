// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeCapacityAnalysis.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "PipeLowering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <cstddef>
#include <optional>
#include <string>

#define DEBUG_TYPE "ttl-pipe-capacity-analysis"

namespace mlir::tt::ttl {

using mlir::func::FuncOp;

ArrayRef<PipeCapacityAcquireInfo>
PipeCapacityPlan::lookupAcquires(PipeTransferSendOp op) const {
  auto it = acquires.find(op.getOperation());
  if (it == acquires.end()) {
    return {};
  }
  return it->second;
}

ArrayRef<PipeCapacityReleaseInfo>
PipeCapacityPlan::lookupReleases(CBPopOp op) const {
  auto it = releases.find(op.getOperation());
  if (it == releases.end()) {
    return {};
  }
  return it->second;
}

bool PipeCapacityPlan::usesCapacityProtocol(PipeTransferSendOp op) const {
  return capacityTransferOps.contains(op.getOperation());
}

bool PipeCapacityPlan::usesCapacityProtocol(PipeTransferPostOp op) const {
  return capacityTransferOps.contains(op.getOperation());
}

bool PipeCapacityPlan::hasSameSelectedTransfers(
    const PipeCapacityPlan &other) const {
  return capacityTransferOps.size() == other.capacityTransferOps.size() &&
         llvm::all_of(capacityTransferOps, [&](Operation *op) {
           return other.capacityTransferOps.contains(op);
         });
}

void PipeCapacityPlan::addAcquire(PipeTransferSendOp op,
                                  PipeCapacityAcquireInfo info) {
  acquires[op.getOperation()].push_back(info);
}

void PipeCapacityPlan::addRelease(CBPopOp op, PipeCapacityReleaseInfo info) {
  releases[op.getOperation()].push_back(info);
}

void PipeCapacityPlan::addInitialization(FuncOp func,
                                         PipeCapacityInitInfo info) {
  SmallVector<PipeCapacityInitInfo> &funcInitializations =
      initializations[func];
  for (const PipeCapacityInitInfo &existing : funcInitializations) {
    if (existing.counter == info.counter) {
      assert(existing.initialCapacity == info.initialCapacity &&
             "same capacity counter initialized with two different counts");
      return;
    }
  }
  funcInitializations.push_back(info);
}

void PipeCapacityPlan::markCapacityTransfer(PipeTransferSendOp op) {
  capacityTransferOps.insert(op.getOperation());
}

void PipeCapacityPlan::markCapacityTransfer(PipeTransferPostOp op) {
  capacityTransferOps.insert(op.getOperation());
}

void PipeCapacityPlan::initializeCounterAllocation(
    PipeCounterAllocationCounts counts) {
  assert(empty() && "capacity counter allocation must be initialized first");
  counterAllocator = PipeCounterAllocator(counts);
}

PipeCounterInfo PipeCapacityPlan::allocateCounter() {
  return counterAllocator.allocate();
}

namespace {

/// Transfer endpoint considered for sender-side capacity synchronization.
struct PipeCapacityEndpoint {
  const PipeTransferNode *transferNode = nullptr;
  const ReceiverDFBInfo *receiverDFBInfo = nullptr;
  PipeReceiverEndpointId endpointId = 0;
  PipeReceiverDFBNodeId receiverDFBNode = 0;
  PipeReceiverDFBKey receiverDFB;
  PipeCapacityReleaseTarget releaseTarget;
  int64_t initialCapacity = 0;
};

/// Operations that consume and release one endpoint's capacity units.
struct PipeCapacityEndpointFacts {
  PipeCapacityEndpoint endpoint;
  PipeTransferSendOp send;
  SmallVector<CBPopOp> pops;
};

/// Capacity counters may share storage across source cores only when the
/// unconditional sender-function initialization writes the same value.
struct PipeCapacityCounterColor {
  int64_t initialCapacity = 0;
  SmallVector<PipeCapacityReleaseTarget> sourceNodes;
  PipeCounterInfo counter;
};

static void printPipe(llvm::raw_ostream &os, const PipeKey &pipe) {
  os << "src(" << pipe.srcX << ", " << pipe.srcY << ") -> dst("
     << pipe.dstStartX << ", " << pipe.dstStartY << ") to (" << pipe.dstEndX
     << ", " << pipe.dstEndY << ") net " << pipe.pipeNetId;
}

static void printEndpoint(llvm::raw_ostream &os,
                          const PipeCapacityEndpoint &endpoint) {
  const PipeCapacityReleaseTarget &target = endpoint.releaseTarget;
  os << "src(" << target.logicalX << ", " << target.logicalY << ") -> ";
  printReceiverDFB(os, endpoint.receiverDFB);
  os << " capacity " << endpoint.initialCapacity;
}

static void debugSkipResource(const PipeResourceInfo &resource,
                              const llvm::Twine &reason) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: skip ";
    printPipe(llvm::dbgs(), resource.pipe);
    llvm::dbgs() << ": " << reason << "\n";
  });
}

static void debugCandidateEndpoint(const PipeCapacityEndpoint &endpoint) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: candidate ";
    printEndpoint(llvm::dbgs(), endpoint);
    llvm::dbgs() << "\n";
  });
}

static void debugRejectEndpoint(const PipeCapacityEndpoint &endpoint,
                                const llvm::Twine &reason) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: reject ";
    printEndpoint(llvm::dbgs(), endpoint);
    llvm::dbgs() << ": " << reason << "\n";
  });
}

static void debugAcceptEndpoint(const PipeCapacityEndpoint &endpoint,
                                int64_t popCount) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: accept ";
    printEndpoint(llvm::dbgs(), endpoint);
    llvm::dbgs() << ": sends=1 pops=" << popCount << "\n";
  });
}

static bool isExactlyDomain(Operation *op, const LaunchNodeDomain &expected,
                            const PipeGraph &pipeGraph) {
  return pipeGraph.getOperationLaunchDomain(op) == expected;
}

static std::optional<int64_t> getDFBIndexFromView(Value view) {
  Value cb = getAttachedCB(view);
  if (!cb) {
    return std::nullopt;
  }
  return getCBIndex(cb);
}

static bool isReceiverDFBView(Value view,
                              const PipeReceiverDFBKey &receiverDFB) {
  std::optional<int64_t> maybeDFBIndex = getDFBIndexFromView(view);
  return maybeDFBIndex && *maybeDFBIndex == receiverDFB.dfbIndex;
}

static PipeCapacityEndpoint
getCapacityEndpoint(const PipeGraph &pipeGraph,
                    const PipeReceiverEndpoint &receiverEndpoint) {
  const PipeTransferNode &transferNode =
      pipeGraph.getPipeTransferNode(receiverEndpoint.transferNode);
  return PipeCapacityEndpoint{
      &transferNode,
      &receiverEndpoint.receiverDFBInfo,
      receiverEndpoint.id,
      receiverEndpoint.receiverDFBNode,
      receiverEndpoint.receiverDFB,
      PipeCapacityReleaseTarget{transferNode.pipe.srcX, transferNode.pipe.srcY},
      receiverEndpoint.receiverDFBInfo.blockCount,
  };
}

static bool checkPosts(const PipeCapacityEndpoint &endpoint,
                       const PipeGraph &pipeGraph) {
  LaunchNodeDomain receiverDomain = getSingleLaunchNodeDomain(
      {endpoint.receiverDFB.receiver.x, endpoint.receiverDFB.receiver.y});
  bool sawPost = false;
  bool valid = true;
  for (Operation *postOperation : endpoint.transferNode->receiverPostOps) {
    auto postOp = llvm::cast<PipeTransferPostOp>(postOperation);
    LaunchNodeDomain postDomain =
        pipeGraph.getOperationLaunchDomain(postOp.getOperation());
    if (!launchNodeDomainsOverlap(postDomain, receiverDomain)) {
      continue;
    }
    if (!isExactlyDomain(postOp, receiverDomain, pipeGraph) ||
        !isNocKernelThread(postOp)) {
      debugRejectEndpoint(endpoint, "post is not in the receiver NOC domain");
      valid = false;
      continue;
    }
    if (!isReceiverDFBView(postOp.getDst(), endpoint.receiverDFB)) {
      debugRejectEndpoint(endpoint, "post destination is not the receiver DFB");
      valid = false;
      continue;
    }
    sawPost = true;
  }
  if (valid && !sawPost) {
    debugRejectEndpoint(endpoint, "no matching receiver posts");
    valid = false;
  }
  return valid;
}

static FailureOr<PipeTransferSendOp>
checkSend(const PipeCapacityEndpoint &endpoint, const PipeGraph &pipeGraph) {
  const PipeCapacityReleaseTarget &target = endpoint.releaseTarget;
  LaunchNodeDomain sourceDomain =
      getSingleLaunchNodeDomain({target.logicalX, target.logicalY});
  auto sendOp =
      llvm::dyn_cast<PipeTransferSendOp>(endpoint.transferNode->sendOp);
  if (!sendOp) {
    debugRejectEndpoint(endpoint, "no matching sends");
    return failure();
  }
  if (!isExactlyDomain(sendOp, sourceDomain, pipeGraph) ||
      !isNocKernelThread(sendOp)) {
    debugRejectEndpoint(endpoint, "send is not in the sender NOC domain");
    return failure();
  }
  return sendOp;
}

static bool collectAndCheckPops(ArrayRef<CBPopOp> candidatePops,
                                const PipeCapacityEndpoint &endpoint,
                                const PipeGraph &pipeGraph,
                                SmallVectorImpl<CBPopOp> &pops) {
  LaunchNodeDomain receiverDomain = getSingleLaunchNodeDomain(
      {endpoint.receiverDFB.receiver.x, endpoint.receiverDFB.receiver.y});
  bool valid = true;
  for (CBPopOp popOp : candidatePops) {
    LaunchNodeDomain popDomain =
        pipeGraph.getOperationLaunchDomain(popOp.getOperation());
    if (!launchNodeDomainsOverlap(popDomain, receiverDomain)) {
      continue;
    }
    if (!isExactlyDomain(popOp, receiverDomain, pipeGraph) ||
        !isNocKernelThread(popOp)) {
      debugRejectEndpoint(endpoint, "pop is not in the receiver NOC domain");
      valid = false;
      continue;
    }
    std::optional<int64_t> maybeReleasedBlocks = getReleasedBlockCount(popOp);
    if (!maybeReleasedBlocks || *maybeReleasedBlocks != 1) {
      debugRejectEndpoint(endpoint, "pop does not release one DFB block");
      valid = false;
      continue;
    }
    const DFBReleaseOwnerMaps &owners = pipeGraph.getDFBReleaseOwnerMaps();
    auto waitOp = lookupOwner<CBWaitOp>(owners.waitByPop, popOp.getOperation());
    if (!waitOp) {
      debugRejectEndpoint(endpoint,
                          "pop is not owned by a matching receiver wait");
      valid = false;
      continue;
    }
    LaunchNodeDomain waitDomain =
        pipeGraph.getOperationLaunchDomain(waitOp.getOperation());
    if (!isExactlyDomain(waitOp, receiverDomain, pipeGraph) ||
        !isNocKernelThread(waitOp)) {
      debugRejectEndpoint(endpoint, "wait is not in the receiver NOC domain");
      valid = false;
      continue;
    }
    std::optional<int64_t> maybeWaitedBlocks = getWaitedBlockCount(waitOp);
    if (!maybeWaitedBlocks || *maybeWaitedBlocks != *maybeReleasedBlocks) {
      debugRejectEndpoint(endpoint,
                          "wait and pop release different DFB block counts");
      valid = false;
      continue;
    }
    pops.push_back(popOp);
  }
  if (valid && pops.empty()) {
    debugRejectEndpoint(endpoint, "no matching receiver pops");
    valid = false;
  }
  return valid;
}

static void recordEndpointCapacityFacts(const PipeCapacityEndpointFacts &facts,
                                        PipeCounterInfo capacityCounter,
                                        PipeCapacityPlan &plan) {
  const PipeCapacityEndpoint &endpoint = facts.endpoint;
  plan.addAcquire(facts.send, PipeCapacityAcquireInfo{capacityCounter, 1});
  plan.addInitialization(
      facts.send->getParentOfType<FuncOp>(),
      PipeCapacityInitInfo{capacityCounter, endpoint.initialCapacity});
  for (CBPopOp popOp : facts.pops) {
    plan.addRelease(popOp, PipeCapacityReleaseInfo{endpoint.releaseTarget,
                                                   capacityCounter, 1});
  }
}

static bool containsSourceNode(ArrayRef<PipeCapacityReleaseTarget> sourceNodes,
                               const PipeCapacityReleaseTarget &sourceNode) {
  return llvm::any_of(sourceNodes,
                      [&](const PipeCapacityReleaseTarget &candidate) {
                        return candidate.logicalX == sourceNode.logicalX &&
                               candidate.logicalY == sourceNode.logicalY;
                      });
}

static PipeCounterInfo allocateCapacityCounter(
    const PipeCapacityEndpoint &endpoint,
    SmallVectorImpl<PipeCapacityCounterColor> &counterColors,
    PipeCapacityPlan &plan) {
  for (PipeCapacityCounterColor &color : counterColors) {
    if (color.initialCapacity == endpoint.initialCapacity &&
        !containsSourceNode(color.sourceNodes, endpoint.releaseTarget)) {
      color.sourceNodes.push_back(endpoint.releaseTarget);
      return color.counter;
    }
  }

  PipeCounterInfo counter = plan.allocateCounter();
  counterColors.push_back(PipeCapacityCounterColor{
      endpoint.initialCapacity, {endpoint.releaseTarget}, counter});
  return counter;
}

static bool isCapacityProtocolLowerable(const PipeCapacityEndpoint &endpoint,
                                        const PipeResourcePlan &resources) {
  auto resourceIt = resources.resources.find(endpoint.transferNode->sendOp);
  if (resourceIt == resources.resources.end()) {
    debugRejectEndpoint(endpoint, "pipe resource is missing");
    return false;
  }
  const PipeResourceInfo &resource = resourceIt->second;
  if (!resource.addressStorage.usesComputedReceiverDFB()) {
    debugSkipResource(resource, "receiver address is not computed");
    return false;
  }
  return true;
}

static void markCapacityTransfer(const PipeTransferNode &transferNode,
                                 PipeCapacityPlan &plan) {
  plan.markCapacityTransfer(
      llvm::cast<PipeTransferSendOp>(transferNode.sendOp));
  for (Operation *postOp : transferNode.receiverPostOps) {
    plan.markCapacityTransfer(llvm::cast<PipeTransferPostOp>(postOp));
  }
}

} // namespace

void buildPipeCapacityPlan(ModuleOp mod, const PipeGraph &pipeGraph,
                           const PipeResourcePlan &resources,
                           PipeCapacityPlan &plan) {
  PipeResourceRequirements requirements =
      getPipeResourceRequirements(resources);
  plan.initializeCounterAllocation(PipeCounterAllocationCounts{
      requirements.syncSemaphoreCount, requirements.globalSemaphoreCount});
  if (!pipeGraph.hasLaunchGrid()) {
    LLVM_DEBUG(llvm::dbgs()
               << "PipeCapacity: skip module without ttl.launch_grid\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "PipeCapacity: "
                          << pipeGraph.getReceiverDFBNodes().size()
                          << " receiver DFB node(s), "
                          << pipeGraph.getPipeReceiverEndpoints().size()
                          << " receiver endpoint(s)\n");

  llvm::DenseMap<int64_t, SmallVector<CBPopOp>> popsByDFBIndex;
  mod.walk([&](CBPopOp popOp) {
    std::optional<int64_t> maybeDFBIndex = getCBIndex(popOp.getCb());
    if (maybeDFBIndex) {
      popsByDFBIndex[*maybeDFBIndex].push_back(popOp);
    }
  });

  SmallVector<PipeCapacityEndpointFacts> endpointFacts;
  llvm::DenseMap<PipeReceiverEndpointId, std::size_t> factsIndexByEndpoint;
  for (const PipeReceiverEndpoint &receiverEndpoint :
       pipeGraph.getPipeReceiverEndpoints()) {
    PipeCapacityEndpoint endpoint =
        getCapacityEndpoint(pipeGraph, receiverEndpoint);
    debugCandidateEndpoint(endpoint);

    // Collective transfers use receiver-post synchronization until the
    // capacity-counter protocol tracks each receiver independently. A fast
    // receiver must not release a slot still owned by a slower receiver.
    if (isCollectiveTransfer(endpoint.transferNode->transferContract)) {
      debugRejectEndpoint(
          endpoint,
          "collective capacity-counter synchronization requires per-receiver "
          "release accounting");
      continue;
    }

    ArrayRef<PipeReceiverEndpointId> writerEndpoints =
        pipeGraph.getReceiverDFBWriterEndpoints(endpoint.receiverDFBNode);
    if (writerEndpoints.size() != 1) {
      std::string reason = "receiver DFB has " +
                           std::to_string(writerEndpoints.size()) +
                           " writer endpoint(s)";
      debugRejectEndpoint(endpoint, reason);
      continue;
    }

    const PipeReceiverDFBNode &receiverDFBNode =
        pipeGraph.getReceiverDFBNode(endpoint.receiverDFBNode);
    if (!receiverDFBNode.hasProvenPipeOnlyProducerStream) {
      debugRejectEndpoint(endpoint,
                          "receiver DFB producer stream is not proven "
                          "pipe-only");
      continue;
    }

    // Capacity adds one unit per pop, so each matching reserve must also
    // consume one block. Enforce that invariant here even though pop validation
    // currently rejects wider spans.
    if (endpoint.receiverDFBInfo->receiverSlotSpanBlocks != 1) {
      debugRejectEndpoint(
          endpoint,
          "receiver reserve spans " +
              llvm::Twine(endpoint.receiverDFBInfo->receiverSlotSpanBlocks) +
              " DFB blocks; capacity accounting assumes one");
      continue;
    }

    if (!checkPosts(endpoint, pipeGraph)) {
      continue;
    }

    FailureOr<PipeTransferSendOp> maybeSendOp = checkSend(endpoint, pipeGraph);
    if (failed(maybeSendOp)) {
      continue;
    }

    SmallVector<CBPopOp> pops;
    auto candidatePopsIt = popsByDFBIndex.find(endpoint.receiverDFB.dfbIndex);
    ArrayRef<CBPopOp> candidatePops;
    if (candidatePopsIt != popsByDFBIndex.end()) {
      candidatePops = candidatePopsIt->second;
    }
    if (!collectAndCheckPops(candidatePops, endpoint, pipeGraph, pops)) {
      continue;
    }

    factsIndexByEndpoint[endpoint.endpointId] = endpointFacts.size();
    endpointFacts.push_back(
        PipeCapacityEndpointFacts{endpoint, *maybeSendOp, std::move(pops)});
    debugAcceptEndpoint(endpoint, endpointFacts.back().pops.size());
  }

  SmallVector<PipeCapacityCounterColor> counterColors;
  for (const PipeTransferNode &transferNode :
       pipeGraph.getPipeTransferNodes()) {
    bool allEndpointsProven = true;
    for (PipeReceiverEndpointId endpointId :
         pipeGraph.getPipeReceiverEndpoints(transferNode.id)) {
      auto factIt = factsIndexByEndpoint.find(endpointId);
      if (factIt == factsIndexByEndpoint.end()) {
        allEndpointsProven = false;
        break;
      }
      if (!isCapacityProtocolLowerable(endpointFacts[factIt->second].endpoint,
                                       resources)) {
        allEndpointsProven = false;
        break;
      }
    }
    if (!allEndpointsProven) {
      continue;
    }

    for (PipeReceiverEndpointId endpointId :
         pipeGraph.getPipeReceiverEndpoints(transferNode.id)) {
      auto factIt = factsIndexByEndpoint.find(endpointId);
      assert(factIt != factsIndexByEndpoint.end() &&
             "proven endpoint is missing capacity facts");
      const PipeCapacityEndpointFacts &facts = endpointFacts[factIt->second];
      PipeCounterInfo capacityCounter =
          allocateCapacityCounter(facts.endpoint, counterColors, plan);
      recordEndpointCapacityFacts(facts, capacityCounter, plan);
    }
    markCapacityTransfer(transferNode, plan);
  }
}

} // namespace mlir::tt::ttl
