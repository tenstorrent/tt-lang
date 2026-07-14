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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <optional>

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
    if (existing.semaphoreIndex == info.semaphoreIndex) {
      assert(existing.initialCapacity == info.initialCapacity &&
             "same capacity semaphore initialized with two different counts");
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

void PipeCapacityPlan::initializeSemaphoreAllocation(
    int64_t firstSemaphoreIndex) {
  assert(empty() && "capacity semaphore allocation must be initialized first");
  nextSemaphoreIndex = firstSemaphoreIndex;
}

int64_t PipeCapacityPlan::allocateSemaphoreIndex() {
  return nextSemaphoreIndex++;
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

/// Operations that consume and release one endpoint's capacity credits.
struct PipeCapacityEndpointFacts {
  PipeCapacityEndpoint endpoint;
  SmallVector<PipeTransferSendOp> sends;
  SmallVector<CBPopOp> pops;
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
                                int64_t sendCount, int64_t popCount) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: accept ";
    printEndpoint(llvm::dbgs(), endpoint);
    llvm::dbgs() << ": sends=" << sendCount << " pops=" << popCount << "\n";
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

static bool collectAndCheckPosts(const PipeCapacityEndpoint &endpoint,
                                 const PipeGraph &pipeGraph) {
  LaunchNodeDomain receiverDomain = getSingleLaunchNodeDomain(
      {endpoint.receiverDFB.receiver.x, endpoint.receiverDFB.receiver.y});
  // Posts establish endpoint validity but carry no facts needed by the caller.
  SmallVector<PipeTransferPostOp> posts;
  bool valid = true;
  for (Operation *postOperation : endpoint.transferNode->receiverPostOps) {
    auto postOp = llvm::cast<PipeTransferPostOp>(postOperation);
    LaunchNodeDomain postDomain =
        pipeGraph.getOperationLaunchDomain(postOp.getOperation());
    if (!launchNodeDomainsOverlap(postDomain, receiverDomain)) {
      continue;
    }
    if (!(postDomain == receiverDomain)) {
      debugRejectEndpoint(endpoint, "post is not in the receiver NOC domain");
      valid = false;
      continue;
    }
    if (!isNocKernelThread(postOp)) {
      debugRejectEndpoint(endpoint, "post is not in the receiver NOC domain");
      valid = false;
      continue;
    }
    if (!isReceiverDFBView(postOp.getDst(), endpoint.receiverDFB)) {
      debugRejectEndpoint(endpoint, "post destination is not the receiver DFB");
      valid = false;
      continue;
    }
    posts.push_back(postOp);
  }
  if (valid && posts.empty()) {
    debugRejectEndpoint(endpoint, "no matching receiver posts");
    valid = false;
  }
  return valid;
}

static bool collectAndCheckSends(const PipeCapacityEndpoint &endpoint,
                                 const PipeGraph &pipeGraph,
                                 SmallVectorImpl<PipeTransferSendOp> &sends) {
  const PipeCapacityReleaseTarget &target = endpoint.releaseTarget;
  LaunchNodeDomain sourceDomain =
      getSingleLaunchNodeDomain({target.logicalX, target.logicalY});
  bool valid = true;
  auto sendOp = llvm::cast<PipeTransferSendOp>(endpoint.transferNode->sendOp);
  if (!isExactlyDomain(sendOp, sourceDomain, pipeGraph) ||
      !isNocKernelThread(sendOp)) {
    debugRejectEndpoint(endpoint, "send is not in the sender NOC domain");
    valid = false;
  } else {
    sends.push_back(sendOp);
  }
  if (valid && sends.empty()) {
    debugRejectEndpoint(endpoint, "no matching sends");
    valid = false;
  }
  return valid;
}

static bool collectAndCheckPops(const PipeCapacityEndpoint &endpoint,
                                const PipeGraph &pipeGraph,
                                SmallVectorImpl<CBPopOp> &pops) {
  LaunchNodeDomain receiverDomain = getSingleLaunchNodeDomain(
      {endpoint.receiverDFB.receiver.x, endpoint.receiverDFB.receiver.y});
  bool valid = true;
  SmallVector<CBPopOp> candidatePops;
  pipeGraph.appendReceiverDFBPops(endpoint.receiverDFB, candidatePops);
  for (CBPopOp popOp : candidatePops) {
    LaunchNodeDomain popDomain =
        pipeGraph.getOperationLaunchDomain(popOp.getOperation());
    if (!launchNodeDomainsOverlap(popDomain, receiverDomain)) {
      continue;
    }
    if (!(popDomain == receiverDomain)) {
      debugRejectEndpoint(endpoint, "pop is not in the receiver NOC domain");
      valid = false;
      continue;
    }
    if (!isNocKernelThread(popOp)) {
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
    if (!(waitDomain == receiverDomain)) {
      debugRejectEndpoint(endpoint, "wait is not in the receiver NOC domain");
      valid = false;
      continue;
    }
    if (!isNocKernelThread(waitOp)) {
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
                                        int64_t capacitySemaphoreIndex,
                                        PipeCapacityPlan &plan) {
  const PipeCapacityEndpoint &endpoint = facts.endpoint;
  for (PipeTransferSendOp sendOp : facts.sends) {
    plan.addAcquire(sendOp, PipeCapacityAcquireInfo{capacitySemaphoreIndex, 1});
    plan.addInitialization(
        sendOp->getParentOfType<FuncOp>(),
        PipeCapacityInitInfo{capacitySemaphoreIndex, endpoint.initialCapacity});
  }
  for (CBPopOp popOp : facts.pops) {
    plan.addRelease(popOp, PipeCapacityReleaseInfo{endpoint.releaseTarget,
                                                   capacitySemaphoreIndex, 1});
  }
}

static bool isCapacityProtocolLowerable(const PipeCapacityEndpoint &endpoint,
                                        const PipeResourcePlan &resources) {
  auto sendOp = llvm::cast<PipeTransferSendOp>(endpoint.transferNode->sendOp);
  FailureOr<PipeTransferCreateOp> maybeCreateOp =
      findPipeTransferCreateForTransfer(sendOp.getTransfer());
  assert(succeeded(maybeCreateOp) &&
         "pipe graph validated transfer provenance");
  auto createPipe = maybeCreateOp->getPipe().getDefiningOp<CreatePipeOp>();
  if (createPipe && createPipe.getDeviceTransferAttr()) {
    debugRejectEndpoint(endpoint,
                        "device transfer uses routing-plane flow control");
    return false;
  }
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

void buildPipeCapacityPlan(const PipeGraph &pipeGraph,
                           const PipeResourcePlan &resources,
                           PipeCapacityPlan &plan) {
  plan.initializeSemaphoreAllocation(
      getPipeResourceRequirements(resources).syncSemaphoreCount);
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

  SmallVector<PipeCapacityEndpointFacts> endpointFacts;
  llvm::DenseMap<PipeReceiverEndpointId, unsigned> factsIndexByEndpoint;
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
      debugRejectEndpoint(endpoint, "receiver DFB has " +
                                        llvm::Twine(writerEndpoints.size()) +
                                        " writer endpoint(s)");
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

    if (!collectAndCheckPosts(endpoint, pipeGraph)) {
      continue;
    }

    SmallVector<PipeTransferSendOp> sends;
    if (!collectAndCheckSends(endpoint, pipeGraph, sends)) {
      continue;
    }

    SmallVector<CBPopOp> pops;
    if (!collectAndCheckPops(endpoint, pipeGraph, pops)) {
      continue;
    }

    factsIndexByEndpoint[endpoint.endpointId] = endpointFacts.size();
    endpointFacts.push_back(
        PipeCapacityEndpointFacts{endpoint, std::move(sends), std::move(pops)});
    debugAcceptEndpoint(endpoint, endpointFacts.back().sends.size(),
                        endpointFacts.back().pops.size());
  }

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
      const PipeCapacityEndpointFacts &facts =
          endpointFacts[factsIndexByEndpoint.lookup(endpointId)];
      int64_t capacitySemaphoreIndex = plan.allocateSemaphoreIndex();
      recordEndpointCapacityFacts(facts, capacitySemaphoreIndex, plan);
    }
    markCapacityTransfer(transferNode, plan);
  }
}

} // namespace mlir::tt::ttl
