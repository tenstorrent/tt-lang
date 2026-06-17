// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeCapacityAnalysis.h"

#include "PipeLowering.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "ttl-pipe-capacity-analysis"

namespace mlir::tt::ttl {

using mlir::func::FuncOp;
namespace ttk = mlir::tt::ttkernel;

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

bool PipeCapacityPlan::usesCapacityProtocol(PipeTransferCreateOp op) const {
  return capacityTransfers.contains(op.getOperation());
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

void PipeCapacityPlan::markCapacityTransfer(PipeTransferCreateOp op) {
  capacityTransfers.insert(op.getOperation());
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

struct OperationLaunchDomain {
  LaunchNodeDomain domain = LaunchNodeDomain::unknown();
  Operation *unanalyzableOp = nullptr;
};

struct PipeCapacityAnalysisState : LaunchNodeDomainState {
  llvm::DenseMap<Operation *, OperationLaunchDomain> operationLaunchDomains;
};

struct PipeSourceCoord {
  int64_t x = 0;
  int64_t y = 0;
};

struct PipeCapacityEndpoint {
  const PipeEdge *pipeEdge = nullptr;
  PipeReceiverEndpointId endpointId = 0;
  PipeReceiverDFBNodeId receiverDFBNode = 0;
  PipeReceiverDFBKey receiverDFB;
  PipeSourceCoord source;
  int64_t initialCapacity = 0;
};

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

static void printReceiverDFB(llvm::raw_ostream &os,
                             const PipeReceiverDFBKey &receiverDFB) {
  os << "receiver(" << receiverDFB.receiver.x << ", " << receiverDFB.receiver.y
     << ") DFB " << receiverDFB.dfbIndex;
}

static void printEndpoint(llvm::raw_ostream &os,
                          const PipeCapacityEndpoint &endpoint) {
  os << "src(" << endpoint.source.x << ", " << endpoint.source.y << ") -> ";
  printReceiverDFB(os, endpoint.receiverDFB);
  os << " capacity " << endpoint.initialCapacity;
}

static void debugSkipResource(const PipeResourceInfo &resource,
                              llvm::StringRef reason) {
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
                                llvm::StringRef reason) {
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

static std::optional<ttk::ThreadType> getKernelThreadType(FuncOp func) {
  if (!func) {
    return std::nullopt;
  }
  if (auto attr = func->getAttrOfType<ttk::ThreadTypeAttr>("ttkernel.thread")) {
    return attr.getValue();
  }
  if (auto attr =
          func->getAttrOfType<ttk::ThreadTypeAttr>(kKernelThreadAttrName)) {
    return attr.getValue();
  }
  return std::nullopt;
}

static bool isNocThread(Operation *op) {
  return getKernelThreadType(op->getParentOfType<FuncOp>()) ==
         ttk::ThreadType::Noc;
}

static OperationLaunchDomain
getOperationLaunchDomain(Operation *op, PipeCapacityAnalysisState &state) {
  auto it = state.operationLaunchDomains.find(op);
  if (it == state.operationLaunchDomains.end()) {
    return {LaunchNodeDomain::unknown(), op};
  }
  return it->second;
}

static LaunchNodeDomain getSingleNodeDomain(PipeReceiverCoord coord) {
  LaunchNodeDomain domain;
  domain.nodes.insert(LaunchNodeCoord{coord.x, coord.y});
  return domain;
}

static LaunchNodeDomain getSingleNodeDomain(PipeSourceCoord coord) {
  LaunchNodeDomain domain;
  domain.nodes.insert(LaunchNodeCoord{coord.x, coord.y});
  return domain;
}

static bool domainsOverlap(const LaunchNodeDomain &lhs,
                           const LaunchNodeDomain &rhs) {
  if (!lhs.known || !rhs.known) {
    return true;
  }
  return !lhs.intersectWith(rhs).nodes.empty();
}

static bool isExactlyDomain(Operation *op, const LaunchNodeDomain &expected,
                            PipeCapacityAnalysisState &state) {
  OperationLaunchDomain actual = getOperationLaunchDomain(op, state);
  return actual.domain == expected;
}

static std::optional<int64_t> getDFBIndex(Value cb) {
  std::optional<int64_t> dfbIndex = getCBIndex(cb);
  if (!dfbIndex) {
    return std::nullopt;
  }
  return *dfbIndex;
}

static std::optional<int64_t> getDFBIndexFromView(Value view) {
  Value cb = getAttachedCB(view);
  if (!cb) {
    return std::nullopt;
  }
  return getDFBIndex(cb);
}

static bool isReceiverDFB(Value cb, const PipeReceiverDFBKey &receiverDFB) {
  std::optional<int64_t> dfbIndex = getDFBIndex(cb);
  return dfbIndex && *dfbIndex == receiverDFB.dfbIndex;
}

static bool isReceiverDFBView(Value view,
                              const PipeReceiverDFBKey &receiverDFB) {
  std::optional<int64_t> dfbIndex = getDFBIndexFromView(view);
  return dfbIndex && *dfbIndex == receiverDFB.dfbIndex;
}

static bool isMatchingTransfer(const PipeCapacityEndpoint &endpoint,
                               Value transfer) {
  PipeTransferCreateOp traced = findPipeTransferCreateForTransfer(transfer);
  return traced && llvm::is_contained(endpoint.pipeEdge->transferCreateOps,
                                      traced.getOperation());
}

static std::optional<int64_t> getReleasedBlockCount(CBPopOp popOp) {
  auto cbType = mlir::dyn_cast<CircularBufferType>(popOp.getCb().getType());
  if (!cbType) {
    return std::nullopt;
  }
  int64_t elementsPerBlock = cbType.getElementsPerBlock();
  int64_t releasedTiles = elementsPerBlock;
  if (auto attr = popOp.getNumTilesAttr()) {
    releasedTiles = attr.getInt();
  }
  if (releasedTiles <= 0 || releasedTiles % elementsPerBlock != 0) {
    return std::nullopt;
  }
  return releasedTiles / elementsPerBlock;
}

static LogicalResult
collectLaunchNodeDomains(ModuleOp mod, PipeCapacityAnalysisState &state) {
  state.initialize(mod);
  if (!state.hasLaunchGrid) {
    return success();
  }

  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  LaunchNodeDomainAnalysisOptions options;
  options.narrowPipeNetScopes = true;
  options.operationCallback = [&](Operation *op, const LaunchNodeDomain &domain,
                                  Operation *unanalyzableOp) {
    state.operationLaunchDomains[op] =
        OperationLaunchDomain{domain, unanalyzableOp};
  };
  solver.load<LaunchNodeDomainAnalysis>(state, options);
  return solver.initializeAndRun(mod);
}

static PipeCapacityEndpoint
getCapacityEndpoint(const PipeGraph &pipeGraph,
                    const PipeReceiverEndpoint &receiverEndpoint) {
  const PipeEdge &pipeEdge = pipeGraph.getPipeEdge(receiverEndpoint.pipeEdge);
  return PipeCapacityEndpoint{
      &pipeEdge,
      receiverEndpoint.id,
      receiverEndpoint.receiverDFBNode,
      receiverEndpoint.receiverDFB,
      PipeSourceCoord{pipeEdge.pipe.srcX, pipeEdge.pipe.srcY},
      pipeEdge.receiverDFBInfo.blockCount,
  };
}

static bool collectAndCheckPosts(ModuleOp mod,
                                 const PipeCapacityEndpoint &endpoint,
                                 PipeCapacityAnalysisState &state,
                                 llvm::SmallPtrSetImpl<Operation *> &postFuncs,
                                 SmallVectorImpl<PipeTransferPostOp> &posts) {
  LaunchNodeDomain receiverDomain =
      getSingleNodeDomain(endpoint.receiverDFB.receiver);
  bool valid = true;
  mod.walk([&](PipeTransferPostOp postOp) {
    if (!isMatchingTransfer(endpoint, postOp.getTransfer())) {
      return;
    }
    OperationLaunchDomain postDomain = getOperationLaunchDomain(postOp, state);
    if (!domainsOverlap(postDomain.domain, receiverDomain)) {
      return;
    }
    if (!(postDomain.domain == receiverDomain) || !isNocThread(postOp)) {
      debugRejectEndpoint(endpoint, "post is not in the receiver NOC domain");
      valid = false;
      return;
    }
    if (!isReceiverDFBView(postOp.getDst(), endpoint.receiverDFB)) {
      debugRejectEndpoint(endpoint, "post destination is not the receiver DFB");
      valid = false;
      return;
    }
    posts.push_back(postOp);
    if (FuncOp func = postOp->getParentOfType<FuncOp>()) {
      postFuncs.insert(func.getOperation());
    }
  });
  if (valid && posts.empty()) {
    debugRejectEndpoint(endpoint, "no matching receiver posts");
    valid = false;
  }
  return valid;
}

static bool
checkPushStream(ModuleOp mod, const PipeCapacityEndpoint &endpoint,
                PipeCapacityAnalysisState &state,
                const llvm::SmallPtrSetImpl<Operation *> &postFuncs) {
  LaunchNodeDomain receiverDomain =
      getSingleNodeDomain(endpoint.receiverDFB.receiver);
  bool valid = true;
  mod.walk([&](CBPushOp pushOp) {
    if (!isReceiverDFB(pushOp.getCb(), endpoint.receiverDFB)) {
      return;
    }
    OperationLaunchDomain pushDomain = getOperationLaunchDomain(pushOp, state);
    if (!domainsOverlap(pushDomain.domain, receiverDomain)) {
      return;
    }
    if (!(pushDomain.domain == receiverDomain) || !isNocThread(pushOp)) {
      debugRejectEndpoint(endpoint, "push is not in the receiver NOC domain");
      valid = false;
      return;
    }
    FuncOp func = pushOp->getParentOfType<FuncOp>();
    if (!func || !postFuncs.contains(func.getOperation())) {
      debugRejectEndpoint(endpoint,
                          "push is not paired with a matching receiver post");
      valid = false;
    }
  });
  return valid;
}

static bool collectAndCheckSends(ModuleOp mod,
                                 const PipeCapacityEndpoint &endpoint,
                                 PipeCapacityAnalysisState &state,
                                 SmallVectorImpl<PipeTransferSendOp> &sends) {
  LaunchNodeDomain sourceDomain = getSingleNodeDomain(endpoint.source);
  bool valid = true;
  mod.walk([&](PipeTransferSendOp sendOp) {
    if (!isMatchingTransfer(endpoint, sendOp.getTransfer())) {
      return;
    }
    if (!isExactlyDomain(sendOp, sourceDomain, state) || !isNocThread(sendOp)) {
      debugRejectEndpoint(endpoint, "send is not in the sender NOC domain");
      valid = false;
      return;
    }
    sends.push_back(sendOp);
  });
  if (valid && sends.empty()) {
    debugRejectEndpoint(endpoint, "no matching sends");
    valid = false;
  }
  return valid;
}

static bool collectAndCheckPops(ModuleOp mod,
                                const PipeCapacityEndpoint &endpoint,
                                PipeCapacityAnalysisState &state,
                                SmallVectorImpl<CBPopOp> &pops) {
  LaunchNodeDomain receiverDomain =
      getSingleNodeDomain(endpoint.receiverDFB.receiver);
  bool valid = true;
  mod.walk([&](CBPopOp popOp) {
    if (!isReceiverDFB(popOp.getCb(), endpoint.receiverDFB)) {
      return;
    }
    OperationLaunchDomain popDomain = getOperationLaunchDomain(popOp, state);
    if (!domainsOverlap(popDomain.domain, receiverDomain)) {
      return;
    }
    if (!(popDomain.domain == receiverDomain) || !isNocThread(popOp)) {
      debugRejectEndpoint(endpoint, "pop is not in the receiver NOC domain");
      valid = false;
      return;
    }
    std::optional<int64_t> releasedBlocks = getReleasedBlockCount(popOp);
    if (!releasedBlocks || *releasedBlocks != 1) {
      debugRejectEndpoint(endpoint, "pop does not release one DFB block");
      valid = false;
      return;
    }
    pops.push_back(popOp);
  });
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
    plan.addRelease(popOp, PipeCapacityReleaseInfo{
                               PipeCapacitySenderCoord{endpoint.source.x,
                                                       endpoint.source.y},
                               capacitySemaphoreIndex, 1});
  }
}

static bool isCapacityProtocolLowerable(const PipeCapacityEndpoint &endpoint,
                                        const PipeResourcePlan &resources) {
  bool hasResource = false;
  for (Operation *createOp : endpoint.pipeEdge->transferCreateOps) {
    auto resourceIt = resources.resources.find(createOp);
    if (resourceIt == resources.resources.end()) {
      LLVM_DEBUG({
        llvm::dbgs() << "PipeCapacity: reject ";
        printEndpoint(llvm::dbgs(), endpoint);
        llvm::dbgs() << ": pipe resource is missing\n";
      });
      return false;
    }
    hasResource = true;
    const PipeResourceInfo &resource = resourceIt->second;
    if (!resource.addressStorage.usesComputedReceiverDFB()) {
      debugSkipResource(resource, "receiver address is not computed");
      return false;
    }
  }
  return hasResource;
}

static void markCapacityTransferCreates(const PipeEdge &pipeEdge,
                                        PipeCapacityPlan &plan) {
  for (Operation *createOp : pipeEdge.transferCreateOps) {
    plan.markCapacityTransfer(llvm::cast<PipeTransferCreateOp>(createOp));
  }
}

} // namespace

LogicalResult buildPipeCapacityPlan(ModuleOp mod, const PipeGraph &pipeGraph,
                                    const PipeResourcePlan &resources,
                                    PipeCapacityPlan &plan) {
  PipeCapacityAnalysisState state;
  if (failed(collectLaunchNodeDomains(mod, state))) {
    return failure();
  }
  plan.initializeSemaphoreAllocation(
      getPipeResourceRequirements(resources).syncSemaphoreCount);
  if (!state.hasLaunchGrid) {
    LLVM_DEBUG(llvm::dbgs()
               << "PipeCapacity: skip module without ttl.launch_grid\n");
    return success();
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

    ArrayRef<PipeReceiverEndpointId> writerEndpoints =
        pipeGraph.getReceiverDFBWriterEndpoints(endpoint.receiverDFBNode);
    if (writerEndpoints.size() != 1) {
      LLVM_DEBUG({
        llvm::dbgs() << "PipeCapacity: reject ";
        printEndpoint(llvm::dbgs(), endpoint);
        llvm::dbgs() << ": receiver DFB has " << writerEndpoints.size()
                     << " writer endpoint(s)\n";
      });
      continue;
    }

    llvm::SmallPtrSet<Operation *, 4> postFuncs;
    SmallVector<PipeTransferPostOp> posts;
    if (!collectAndCheckPosts(mod, endpoint, state, postFuncs, posts)) {
      continue;
    }
    if (!checkPushStream(mod, endpoint, state, postFuncs)) {
      continue;
    }

    SmallVector<PipeTransferSendOp> sends;
    if (!collectAndCheckSends(mod, endpoint, state, sends)) {
      continue;
    }

    SmallVector<CBPopOp> pops;
    if (!collectAndCheckPops(mod, endpoint, state, pops)) {
      continue;
    }

    factsIndexByEndpoint[endpoint.endpointId] = endpointFacts.size();
    endpointFacts.push_back(
        PipeCapacityEndpointFacts{endpoint, std::move(sends), std::move(pops)});
    debugAcceptEndpoint(endpoint, endpointFacts.back().sends.size(),
                        endpointFacts.back().pops.size());
  }

  for (const PipeEdge &pipeEdge : pipeGraph.getPipeEdges()) {
    bool allEndpointsProven = true;
    for (PipeReceiverEndpointId endpointId :
         pipeGraph.getPipeReceiverEndpoints(pipeEdge.id)) {
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
         pipeGraph.getPipeReceiverEndpoints(pipeEdge.id)) {
      const PipeCapacityEndpointFacts &facts =
          endpointFacts[factsIndexByEndpoint[endpointId]];
      int64_t capacitySemaphoreIndex = plan.allocateSemaphoreIndex();
      recordEndpointCapacityFacts(facts, capacitySemaphoreIndex, plan);
    }
    markCapacityTransferCreates(pipeEdge, plan);
  }

  return success();
}

} // namespace mlir::tt::ttl
