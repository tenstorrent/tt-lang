// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipePlanning.h"

#include "mlir/IR/Dominance.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-pipe-capacity-analysis"

namespace mlir::tt::ttl {

bool PipeSynchronizationSelection::usesCapacityProtocol(Operation *op) const {
  assert((isa<PipeTransferSendOp, PipeTransferPostOp>(op)) &&
         "only pipe transfer protocol operations select synchronization");
  return capacityTransferOps.contains(op);
}

ArrayRef<PipeCapacityAcquireInfo>
PipeCapacityPlan::lookupAcquires(PipeTransferSendOp op) const {
  auto acquireIt = acquires.find(op.getOperation());
  if (acquireIt == acquires.end()) {
    return {};
  }
  return acquireIt->second;
}

ArrayRef<PipeCapacityReleaseInfo>
PipeCapacityPlan::lookupReleases(CBPopOp op) const {
  auto releaseIt = releases.find(op.getOperation());
  if (releaseIt == releases.end()) {
    return {};
  }
  return releaseIt->second;
}

void PipeCapacityPlan::addAcquire(PipeTransferSendOp op,
                                  PipeCapacityAcquireInfo info) {
  acquires[op.getOperation()].push_back(info);
}

void PipeCapacityPlan::addRelease(CBPopOp op, PipeCapacityReleaseInfo info) {
  releases[op.getOperation()].push_back(info);
}

void PipeCapacityPlan::addInitialization(func::FuncOp function,
                                         PipeCapacityInitInfo info) {
  SmallVector<PipeCapacityInitInfo> &functionInitializations =
      initializations[function];
  for (const PipeCapacityInitInfo &existing : functionInitializations) {
    if (existing.counter == info.counter) {
      assert(existing.initialCapacity == info.initialCapacity &&
             "same capacity counter initialized with two different counts");
      return;
    }
  }
  functionInitializations.push_back(info);
}

void PipeCapacityPlan::initializeCounterAllocation(
    PipeCounterAllocationCounts counts, PipeCounterAllocationPolicy policy) {
  assert(empty() && "capacity counter allocation must be initialized first");
  counterAllocator = PipeCounterAllocator(counts, policy);
}

PipeCounterInfo PipeCapacityPlan::allocateCounter() {
  return counterAllocator.allocate();
}

const PipeTransferPlan &
PipeModulePlan::getTransferPlan(Operation *operation) const {
  auto planIt = transferPlans.find(operation);
  assert(planIt != transferPlans.end() &&
         "active pipe send or receiver post has no transfer plan");
  return planIt->second;
}

static PipeType getPipeType(MLIRContext *context,
                            const PipeResourceInfo &resources) {
  const PipeKey &pipe = resources.pipe;
  return PipeType::get(context, pipe.srcX, pipe.srcY, pipe.dstStartX,
                       pipe.dstStartY, pipe.dstEndX, pipe.dstEndY,
                       pipe.pipeNetId);
}

static FailureOr<PipeSendPlan>
buildPipeSendPlan(PipeTransferSendOp sendOp,
                  const DominanceInfo &dominanceInfo) {
  FailureOr<CircularBufferType> maybeDFBType =
      utils::getTTLCircularBufferType(sendOp.getSrc());
  if (failed(maybeDFBType)) {
    sendOp.emitError("pipe transfer source must have a TTL DFB type");
    return failure();
  }
  auto tileType =
      llvm::dyn_cast<ttcore::TileType>((*maybeDFBType).getElementType());
  if (!tileType) {
    sendOp.emitError("pipe transfer source DFB element type must be tile");
    return failure();
  }

  bool readFromDFB =
      llvm::any_of(sendOp.getSrc().getUsers(), [&](Operation *user) {
        return isa<CBWaitOp>(user) && user->getOperand(0) == sendOp.getSrc() &&
               dominanceInfo.dominates(user, sendOp);
      });

  int64_t elementCount = 1;
  for (int64_t dimension : (*maybeDFBType).getShape()) {
    elementCount *= dimension;
  }
  return PipeSendPlan{readFromDFB, elementCount * static_cast<int64_t>(
                                                      tileType.getSizeBytes())};
}

static FailureOr<PipePostPlan>
buildPipePostPlan(PipeTransferPostOp postOp,
                  const PipeResourceInfo &resources) {
  if (resources.addressStorage.usesComputedReceiverDFB()) {
    return PipePostPlan{};
  }

  Value receiverDFB = getAttachedCB(postOp.getDst());
  if (!receiverDFB) {
    postOp.emitError("pipe receive destination is not attached to a DFB");
    return failure();
  }
  FailureOr<CircularBufferType> maybeDFBType =
      utils::getTTLCircularBufferType(receiverDFB);
  if (failed(maybeDFBType)) {
    postOp.emitError("pipe receive destination is not attached to a TTL DFB");
    return failure();
  }
  auto tileType =
      llvm::dyn_cast<ttcore::TileType>((*maybeDFBType).getElementType());
  if (!tileType) {
    postOp.emitError("pipe receiver DFB element type must be tile");
    return failure();
  }
  return PipePostPlan{PipeReceiverAddressPublicationPlan{
      receiverDFB, static_cast<int64_t>(tileType.getSizeBytes())}};
}

static void printPipe(llvm::raw_ostream &os, const PipeKey &pipe) {
  os << "src(" << pipe.srcX << ", " << pipe.srcY << ") -> dst("
     << pipe.dstStartX << ", " << pipe.dstStartY << ") to (" << pipe.dstEndX
     << ", " << pipe.dstEndY << ") net " << pipe.pipeNetId;
}

static void debugSkipResource(const PipeResourceInfo &resource,
                              const llvm::Twine &reason) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: skip ";
    printPipe(llvm::dbgs(), resource.pipe);
    llvm::dbgs() << ": " << reason << "\n";
  });
}

static void debugRejectEndpoint(const PipeCapacityEndpointFacts &endpointFacts,
                                const llvm::Twine &reason) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: reject ";
    endpointFacts.print(llvm::dbgs());
    llvm::dbgs() << ": " << reason << "\n";
  });
}

static bool
isCapacityProtocolLowerable(const PipeCapacityEndpointFacts &endpointFacts,
                            const PipeGraph &pipeGraph,
                            const PipeResourcePlan &resources) {
  const PipeTransferNode &transferNode =
      pipeGraph.getPipeTransferNode(endpointFacts.transferNode);
  auto resourceIt = resources.resources.find(transferNode.sendOp);
  if (resourceIt == resources.resources.end()) {
    debugRejectEndpoint(endpointFacts, "pipe resource is missing");
    return false;
  }
  const PipeResourceInfo &resource = resourceIt->second;
  if (!resource.addressStorage.usesComputedReceiverDFB()) {
    debugSkipResource(resource, "receiver address is not computed");
    return false;
  }
  return true;
}

static SmallVector<PipeTransferNodeId>
selectCapacityTransfers(const PipeCapacityAnalysisResult &capacityFacts,
                        const PipeGraph &pipeGraph,
                        const PipeResourcePlan &resources) {
  SmallVector<PipeTransferNodeId> selectedTransfers;
  for (const PipeTransferNode &transferNode :
       pipeGraph.getPipeTransferNodes()) {
    bool allEndpointsProven = true;
    for (PipeReceiverEndpointId endpoint :
         pipeGraph.getPipeReceiverEndpoints(transferNode.id)) {
      if (!capacityFacts.hasEndpointFacts(endpoint)) {
        allEndpointsProven = false;
        break;
      }
      if (!isCapacityProtocolLowerable(capacityFacts.getEndpointFacts(endpoint),
                                       pipeGraph, resources)) {
        allEndpointsProven = false;
        break;
      }
    }
    if (allEndpointsProven) {
      selectedTransfers.push_back(transferNode.id);
    }
  }
  return selectedTransfers;
}

/// Capacity counters may share storage across source cores only when the
/// unconditional sender-function initialization writes the same value.
struct PipeCapacityCounterColor {
  int64_t initialCapacity = 0;
  SmallVector<PipeCapacityReleaseTarget> sourceNodes;
  PipeCounterInfo counter;
};

static bool containsSourceNode(ArrayRef<PipeCapacityReleaseTarget> sourceNodes,
                               const PipeCapacityReleaseTarget &sourceNode) {
  return llvm::any_of(sourceNodes,
                      [&](const PipeCapacityReleaseTarget &candidate) {
                        return candidate.logicalX == sourceNode.logicalX &&
                               candidate.logicalY == sourceNode.logicalY;
                      });
}

class PipeCapacityPlanBuilder {
public:
  static void buildSelectedCapacityPlan(
      ArrayRef<PipeTransferNodeId> selectedTransfers,
      const PipeCapacityAnalysisResult &capacityFacts,
      const PipeGraph &pipeGraph, const PipeResourcePlan &resources,
      PipeCounterAllocationPolicy counterPolicy, PipeCapacityPlan &plan) {
    PipeResourceRequirements requirements =
        getPipeResourceRequirements(resources);
    plan.initializeCounterAllocation(
        PipeCounterAllocationCounts{requirements.syncSemaphoreCount,
                                    requirements.globalSemaphoreCount},
        counterPolicy);
    SmallVector<PipeCapacityCounterColor> counterColors;
    for (PipeTransferNodeId transferNodeId : selectedTransfers) {
      const PipeTransferNode &transferNode =
          pipeGraph.getPipeTransferNode(transferNodeId);
      auto resourceIt = resources.resources.find(transferNode.sendOp);
      assert(resourceIt != resources.resources.end() &&
             "selected capacity transfer is missing final resources");
      assert(resourceIt->second.addressStorage.usesComputedReceiverDFB() &&
             "selected capacity transfer lost computed receiver addresses");
      for (PipeReceiverEndpointId endpoint :
           pipeGraph.getPipeReceiverEndpoints(transferNode.id)) {
        const PipeCapacityEndpointFacts &endpointFacts =
            capacityFacts.getEndpointFacts(endpoint);
        PipeCounterInfo capacityCounter =
            allocateCapacityCounter(endpointFacts, counterColors, plan);
        recordEndpointCapacityFacts(endpointFacts, capacityCounter, plan);
      }
    }
  }

private:
  static void
  recordEndpointCapacityFacts(const PipeCapacityEndpointFacts &endpointFacts,
                              PipeCounterInfo capacityCounter,
                              PipeCapacityPlan &plan) {
    plan.addAcquire(endpointFacts.send,
                    PipeCapacityAcquireInfo{capacityCounter, 1});
    plan.addInitialization(
        endpointFacts.send->getParentOfType<func::FuncOp>(),
        PipeCapacityInitInfo{capacityCounter, endpointFacts.initialCapacity});
    for (CBPopOp popOp : endpointFacts.pops) {
      plan.addRelease(popOp,
                      PipeCapacityReleaseInfo{endpointFacts.releaseTarget,
                                              capacityCounter, 1});
    }
  }

  static PipeCounterInfo allocateCapacityCounter(
      const PipeCapacityEndpointFacts &endpointFacts,
      SmallVectorImpl<PipeCapacityCounterColor> &counterColors,
      PipeCapacityPlan &plan) {
    for (PipeCapacityCounterColor &color : counterColors) {
      if (color.initialCapacity == endpointFacts.initialCapacity &&
          !containsSourceNode(color.sourceNodes, endpointFacts.releaseTarget)) {
        color.sourceNodes.push_back(endpointFacts.releaseTarget);
        return color.counter;
      }
    }

    PipeCounterInfo counter = plan.allocateCounter();
    counterColors.push_back(PipeCapacityCounterColor{
        endpointFacts.initialCapacity, {endpointFacts.releaseTarget}, counter});
    return counter;
  }
};

FailureOr<PipeModulePlan>
buildPipeModulePlan(ModuleOp module, ValueOriginAnalysis &analysis,
                    const PipeTransferIndex &transferIndex,
                    const PipeGraph &pipeGraph,
                    const PipePlanningOptions &options) {
  PipeModulePlan plan;
  PipeSynchronizationSelection synchronizationSelection;
  buildPipeNetIndex(module, plan.pipeNetIndex);

  if (options.enableCapacitySynchronization) {
    PipeCapacityAnalysisResult capacityFacts =
        analyzePipeCapacity(module, pipeGraph);
    // Preliminary resources determine which transfers have computed receiver
    // addresses. Final allocation omits sender-ready counters for transfers
    // selected for capacity synchronization.
    PipeResourcePlan preliminaryResourcePlan;
    if (failed(buildPipeResourcePlan(
            module, transferIndex, pipeGraph, preliminaryResourcePlan,
            options.enableComputedAddresses, options.counterAllocationPolicy,
            /*synchronizationSelection=*/nullptr))) {
      return failure();
    }
    SmallVector<PipeTransferNodeId> selectedCapacityTransfers =
        selectCapacityTransfers(capacityFacts, pipeGraph,
                                preliminaryResourcePlan);
    for (PipeTransferNodeId transferNode : selectedCapacityTransfers) {
      const PipeTransferNode &selectedTransfer =
          pipeGraph.getPipeTransferNode(transferNode);
      synchronizationSelection.capacityTransferOps.insert(
          selectedTransfer.sendOp);
      for (Operation *postOp : selectedTransfer.receiverPostOps) {
        synchronizationSelection.capacityTransferOps.insert(postOp);
      }
    }
    if (failed(buildPipeResourcePlan(
            module, transferIndex, pipeGraph, plan.resourcePlan,
            options.enableComputedAddresses, options.counterAllocationPolicy,
            &synchronizationSelection))) {
      return failure();
    }
    SmallVector<PipeTransferNodeId> finalSelectedCapacityTransfers =
        selectCapacityTransfers(capacityFacts, pipeGraph, plan.resourcePlan);
    if (finalSelectedCapacityTransfers != selectedCapacityTransfers) {
      module.emitError(
          "PipeNet capacity protocol selection changed after resource "
          "replanning");
      return failure();
    }
    PipeCapacityPlanBuilder::buildSelectedCapacityPlan(
        selectedCapacityTransfers, capacityFacts, pipeGraph, plan.resourcePlan,
        options.counterAllocationPolicy, plan.capacityPlan);
  } else if (failed(buildPipeResourcePlan(
                 module, transferIndex, pipeGraph, plan.resourcePlan,
                 options.enableComputedAddresses,
                 options.counterAllocationPolicy,
                 /*synchronizationSelection=*/nullptr))) {
    return failure();
  }

  const PipeCapacityPlan *maybeCapacityPlan =
      options.enableCapacitySynchronization ? &plan.capacityPlan : nullptr;
  plan.resourceRequirements =
      getPipeResourceRequirements(plan.resourcePlan, maybeCapacityPlan);

  module.walk([&](WaitOp waitOp) {
    if (analysis.getOrigins(waitOp.getXf()).allMatch([](Value origin) {
          return static_cast<bool>(origin.getDefiningOp<PipeTransferSendOp>());
        })) {
      plan.completedPipeSendWaits.insert(waitOp);
    }
  });

  DominanceInfo dominanceInfo(module);
  for (const auto &resourceEntry : plan.resourcePlan.resources) {
    Operation *operation = resourceEntry.first;
    const PipeResourceInfo &resources = resourceEntry.second;
    auto sendOp = dyn_cast<PipeTransferSendOp>(operation);
    auto postOp = dyn_cast<PipeTransferPostOp>(operation);
    if (!sendOp && !postOp) {
      assert(isa<PipeTransferWaitOp>(operation) &&
             "pipe resources assigned to an unsupported operation");
      continue;
    }
    bool usesCapacityProtocol =
        synchronizationSelection.usesCapacityProtocol(operation);
    PipeSynchronizationProtocol synchronizationProtocol =
        usesCapacityProtocol ? PipeSynchronizationProtocol::Capacity
                             : PipeSynchronizationProtocol::ReceiverPost;

    auto addTransferPlan = [&](auto operationPlan) {
      PipeTransferPlan transferPlan(getPipeType(module.getContext(), resources),
                                    resources, synchronizationProtocol,
                                    std::move(operationPlan));
      auto [planIt, inserted] =
          plan.transferPlans.insert({operation, std::move(transferPlan)});
      (void)planIt;
      assert(inserted && "pipe operation has more than one transfer plan");
    };

    if (sendOp) {
      FailureOr<PipeSendPlan> maybeSendPlan =
          buildPipeSendPlan(sendOp, dominanceInfo);
      if (failed(maybeSendPlan)) {
        return failure();
      }
      addTransferPlan(*maybeSendPlan);
    } else {
      FailureOr<PipePostPlan> maybePostPlan =
          buildPipePostPlan(postOp, resources);
      if (failed(maybePostPlan)) {
        return failure();
      }
      addTransferPlan(*maybePostPlan);
    }
  }

  return plan;
}

void applyPipeModuleAttributes(ModuleOp module, const PipeModulePlan &plan) {
  Builder builder(module.getContext());
  const PipeResourcePlan &resources = plan.getResourcePlan();
  for (const auto &[function, dfbIndices] :
       resources.computedAddressDFBIndices) {
    function->setAttr(kPipeComputedAddressDFBIndicesAttrName,
                      builder.getDenseI32ArrayAttr(dfbIndices));
  }

  const PipeResourceRequirements &requirements = plan.getResourceRequirements();
  module->setAttr(kPipeSyncSemaphoreCountAttrName,
                  builder.getI64IntegerAttr(requirements.syncSemaphoreCount));
  if (requirements.globalSemaphoreCount > 0) {
    module->setAttr(
        kPipeGlobalSemaphoreCountAttrName,
        builder.getI64IntegerAttr(requirements.globalSemaphoreCount));
  }
  if (requirements.sramScratchBytes > 0) {
    module->setAttr(kPipeSramScratchBytesAttrName,
                    builder.getI64IntegerAttr(requirements.sramScratchBytes));
  }
}

} // namespace mlir::tt::ttl
