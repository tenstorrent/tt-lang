// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipePlanning.h"

#include "mlir/IR/Dominance.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/PipeTransferAnalysis.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-pipe-capacity-analysis"

namespace mlir::tt::ttl {

bool PipeSynchronizationSelection::usesCapacityProtocol(Operation *op) const {
  assert((isa<PipeTransferSendOp, PipeTransferPostOp>(op)) &&
         "only pipe transfer protocol operations select synchronization");
  return capacityTransferOps.contains(op);
}

bool PipeSynchronizationSelection::usesFabricProtocol(Operation *op) const {
  return fabricTransferOps.contains(op);
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

SmallVector<CBPopOp, 1>
PipeCapacityPlan::findReleaseOps(PipeTransferNodeId transferNode) const {
  SmallVector<CBPopOp, 1> releaseOps;
  for (const auto &[operation, releaseInfos] : releases) {
    if (llvm::any_of(releaseInfos,
                     [&](const PipeCapacityReleaseInfo &releaseInfo) {
                       return releaseInfo.transferNode == transferNode;
                     })) {
      releaseOps.push_back(cast<CBPopOp>(operation));
    }
  }
  return releaseOps;
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

FailureOr<PipeTransferPayload> getPipeTransferPayload(PipeTransferSendOp sendOp,
                                                      int64_t blockSpan) {
  FailureOr<CircularBufferType> maybeDFBType =
      utils::getTTLCircularBufferType(sendOp.getSrc());
  if (failed(maybeDFBType)) {
    sendOp.emitError("pipe transfer source must have a TTL DFB type");
    return failure();
  }
  CircularBufferType dfbType = *maybeDFBType;
  auto tileType = llvm::dyn_cast<ttcore::TileType>(dfbType.getElementType());
  if (!tileType) {
    sendOp.emitError("pipe transfer source DFB element type must be tile");
    return failure();
  }
  if (blockSpan <= 0 || dfbType.getBlockCount() % blockSpan != 0) {
    sendOp.emitError("source DFB block count must be divisible by pipe "
                     "transfer block span");
    return failure();
  }

  std::optional<int64_t> maybeElementCount =
      llvm::checkedMul(dfbType.getElementsPerBlock(), blockSpan);
  if (!maybeElementCount) {
    sendOp.emitError("pipe transfer element count exceeds int64_t");
    return failure();
  }
  int64_t elementSizeBytes = tileType.getSizeBytes();
  std::optional<int64_t> maybeSizeBytes =
      llvm::checkedMul(*maybeElementCount, elementSizeBytes);
  if (!maybeSizeBytes) {
    sendOp.emitError("pipe transfer payload size exceeds int64_t");
    return failure();
  }
  return PipeTransferPayload{*maybeElementCount, elementSizeBytes,
                             *maybeSizeBytes};
}

static FailureOr<PipeSendPlan>
buildPipeSendPlan(PipeTransferSendOp sendOp, const DominanceInfo &dominanceInfo,
                  int64_t blockSpan, const FabricRoutePlan *fabricRoutePlan) {
  FailureOr<PipeTransferPayload> maybePayload =
      getPipeTransferPayload(sendOp, blockSpan);
  if (failed(maybePayload)) {
    return failure();
  }

  bool readFromDFB =
      llvm::any_of(sendOp.getSrc().getUsers(), [&](Operation *user) {
        return isa<CBWaitOp>(user) && user->getOperand(0) == sendOp.getSrc() &&
               dominanceInfo.dominates(user, sendOp);
      });

  ArrayRef<std::size_t> fabricRouteIndices =
      fabricRoutePlan
          ? fabricRoutePlan->lookupRouteIndices(sendOp.getOperation())
          : ArrayRef<std::size_t>();
  return PipeSendPlan{readFromDFB, maybePayload->sizeBytes,
                      SmallVector<std::size_t>(fabricRouteIndices)};
}

static FailureOr<PipePostPlan>
buildPipePostPlan(PipeTransferPostOp postOp,
                  ArrayRef<PipeResourceInfo> resources,
                  const FabricRoutePlan *fabricRoutePlan) {
  ArrayRef<std::size_t> fabricRouteIndices =
      fabricRoutePlan
          ? fabricRoutePlan->lookupRouteIndices(postOp.getOperation())
          : ArrayRef<std::size_t>();
  SmallVector<PipeAddressMode> addressModes =
      llvm::map_to_vector(resources, [](const PipeResourceInfo &resource) {
        return resource.addressStorage.mode;
      });
  if (llvm::all_of(resources, [](const PipeResourceInfo &resource) {
        return resource.addressStorage.usesComputedReceiverAddress();
      })) {
    return PipePostPlan{/*addressPublication=*/std::nullopt,
                        std::move(addressModes),
                        SmallVector<std::size_t>(fabricRouteIndices)};
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
  return PipePostPlan{
      PipeReceiverAddressPublicationPlan{
          receiverDFB, static_cast<int64_t>(tileType.getSizeBytes())},
      std::move(addressModes), SmallVector<std::size_t>(fabricRouteIndices)};
}

static bool
haveEquivalentComputedAddresses(const PipeComputedAddressInfo &lhs,
                                const PipeComputedAddressInfo &rhs) {
  return lhs.receiverDFBIndex == rhs.receiverDFBIndex &&
         lhs.baseRuntimeCommonArgIndex == rhs.baseRuntimeCommonArgIndex &&
         lhs.baseByteOffset == rhs.baseByteOffset &&
         lhs.initialSlot == rhs.initialSlot &&
         lhs.repeatStride == rhs.repeatStride &&
         lhs.blockCount == rhs.blockCount &&
         lhs.blockStrideBytes == rhs.blockStrideBytes &&
         lhs.staticTileByteOffset == rhs.staticTileByteOffset &&
         lhs.dynamicSlotCounterIndex == rhs.dynamicSlotCounterIndex;
}

static bool haveEquivalentAddressStorage(const PipeAddressStorageInfo &lhs,
                                         const PipeAddressStorageInfo &rhs,
                                         PipeRole role) {
  if (lhs.mode != rhs.mode) {
    return false;
  }
  if (lhs.mode == PipeAddressMode::ReceiverPublishedAddressTable) {
    if (role == PipeRole::Destination) {
      return true;
    }
    return lhs.sramAddressTable && rhs.sramAddressTable &&
           lhs.sramAddressTable->byteOffset == rhs.sramAddressTable->byteOffset;
  }
  if (role == PipeRole::Source) {
    return true;
  }
  return lhs.computedAddress && rhs.computedAddress &&
         haveEquivalentComputedAddresses(*lhs.computedAddress,
                                         *rhs.computedAddress);
}

static bool haveEquivalentProjectedResources(const PipeResourceInfo &lhs,
                                             const PipeResourceInfo &rhs,
                                             PipeRole role) {
  if (!(lhs.pipe == rhs.pipe) || lhs.transferContract != rhs.transferContract ||
      !haveEquivalentAddressStorage(lhs.addressStorage, rhs.addressStorage,
                                    role)) {
    return false;
  }
  return role == PipeRole::Source
             ? lhs.readyCounter == rhs.readyCounter
             : lhs.completion.counter == rhs.completion.counter;
}

static FailureOr<SmallVector<PipeResourceInfo>>
projectSelectedResources(Operation *operation, PipeNetRecordsAttr records,
                         ArrayRef<PipeResourceInfo> resources, PipeRole role) {
  FailureOr<SmallVector<PipeRecordLocalIndex>> localRecords =
      getPipeRecordLocalIndices(records, role);
  if (failed(localRecords) || localRecords->size() != resources.size()) {
    operation->emitError(
        "cannot derive device-local selected-resource indices");
    return failure();
  }
  SmallVector<std::optional<PipeResourceInfo>> projected;
  for (auto [localRecord, resource] :
       llvm::zip_equal(*localRecords, resources)) {
    if (localRecord.index >= projected.size()) {
      projected.resize(localRecord.index + 1);
    }
    std::optional<PipeResourceInfo> &slot = projected[localRecord.index];
    if (slot && !haveEquivalentProjectedResources(*slot, resource, role)) {
      operation->emitError()
          << (role == PipeRole::Source ? "source" : "destination")
          << "-incident records require inconsistent resource assignments";
      return failure();
    }
    slot = resource;
  }
  auto first = llvm::find_if(
      projected, [](const auto &resource) { return resource.has_value(); });
  assert(first != projected.end() &&
         "selected resource projection must contain one active row");
  SmallVector<PipeResourceInfo> result;
  result.reserve(projected.size());
  for (const std::optional<PipeResourceInfo> &resource : projected) {
    result.push_back(resource.value_or(**first));
  }
  return result;
}

static FailureOr<SelectedPipeResources>
buildSelectedResources(Operation *operation, PipeNetRecordsAttr records,
                       ArrayRef<PipeResourceInfo> resources) {
  FailureOr<SmallVector<PipeResourceInfo>> source =
      projectSelectedResources(operation, records, resources, PipeRole::Source);
  FailureOr<SmallVector<PipeResourceInfo>> destination =
      projectSelectedResources(operation, records, resources,
                               PipeRole::Destination);
  if (failed(source) || failed(destination)) {
    return failure();
  }
  return SelectedPipeResources{std::move(*source), std::move(*destination)};
}

template <typename Resources>
static bool allUseComputedReceiverDFB(const Resources &resources) {
  if (const auto *staticResources = std::get_if<PipeResourceInfo>(&resources)) {
    return staticResources->addressStorage.usesComputedReceiverDFB();
  }
  return llvm::all_of(
      std::get<SelectedPipeResources>(resources).destination,
      [](const PipeResourceInfo &resource) {
        return resource.addressStorage.usesComputedReceiverDFB();
      });
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

static bool isCapacityProtocolLowerable(
    const PipeCapacityEndpointFacts &endpointFacts, const PipeGraph &pipeGraph,
    const PipeResourcePlan &resources, const FabricRoutePlan *fabricRoutePlan) {
  const PipeTransferNode &transferNode =
      pipeGraph.getPipeTransferNode(endpointFacts.transferNode);
  if (fabricRoutePlan &&
      !fabricRoutePlan->lookupRouteIndices(transferNode.sendOp).empty()) {
    debugRejectEndpoint(endpointFacts,
                        "device transfer uses routing-plane flow control");
    return false;
  }
  auto resourceIt = resources.resources.find(transferNode.sendOp);
  if (resourceIt == resources.resources.end()) {
    debugRejectEndpoint(endpointFacts, "pipe resource is missing");
    return false;
  }
  const PipeResourceInfo &resource = resourceIt->second;
  if (endpointFacts.transportOwnsStorage) {
    return true;
  }
  if (!resource.addressStorage.usesComputedReceiverDFB()) {
    debugSkipResource(resource, "receiver address is not computed");
    return false;
  }
  return true;
}

static SmallVector<PipeTransferNodeId> selectCapacityTransfers(
    const PipeCapacityAnalysisResult &capacityFacts, const PipeGraph &pipeGraph,
    const PipeResourcePlan &resources, const FabricRoutePlan *fabricRoutePlan) {
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
                                       pipeGraph, resources, fabricRoutePlan)) {
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

/// Capacity counters may share storage across source nodes only when the
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
      for (PipeReceiverEndpointId endpoint :
           pipeGraph.getPipeReceiverEndpoints(transferNode.id)) {
        const PipeCapacityEndpointFacts &endpointFacts =
            capacityFacts.getEndpointFacts(endpoint);
        assert((endpointFacts.transportOwnsStorage ||
                resourceIt->second.addressStorage.usesComputedReceiverDFB()) &&
               "selected capacity transfer has no direct receiver address");
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
    plan.addAcquire(
        endpointFacts.send,
        PipeCapacityAcquireInfo{capacityCounter,
                                endpointFacts.receiverBlocksPerTransfer});
    plan.addInitialization(
        endpointFacts.send->getParentOfType<func::FuncOp>(),
        PipeCapacityInitInfo{capacityCounter, endpointFacts.initialCapacity});
    for (CBPopOp popOp : endpointFacts.pops) {
      plan.addRelease(popOp, PipeCapacityReleaseInfo{
                                 endpointFacts.transferNode,
                                 endpointFacts.releaseTarget, capacityCounter,
                                 endpointFacts.receiverBlocksPerTransfer});
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

FailureOr<PipeModulePlan> buildPipeModulePlan(
    ModuleOp module, ValueOriginAnalysis &analysis,
    const PipeTransferIndex &transferIndex, const PipeGraph &pipeGraph,
    const PipeNetIndex &pipeNetIndex, const PipePlanningOptions &options) {
  PipeModulePlan plan;
  PipeSynchronizationSelection synchronizationSelection;
  plan.pipeNetIndex = pipeNetIndex;

  const FabricRoutePlan *fabricRoutePlan = options.fabricRoutePlan;
  if (fabricRoutePlan) {
    for (const auto &[operation, routeIndices] :
         fabricRoutePlan->routeIndices) {
      assert(!routeIndices.empty() && "fabric route table must not be empty");
      synchronizationSelection.fabricTransferOps.insert(operation);
    }
  }

  if (options.enableCapacitySynchronization) {
    PipeCapacityAnalysisResult capacityFacts = analyzePipeCapacity(pipeGraph);
    // Preliminary resources determine which transfers have computed receiver
    // addresses. Final allocation omits sender-ready counters for transfers
    // selected for capacity synchronization.
    PipeResourcePlan preliminaryResourcePlan;
    if (failed(buildPipeResourcePlan(
            module, transferIndex, pipeGraph, preliminaryResourcePlan,
            options.enableComputedAddresses, options.counterAllocationPolicy,
            fabricRoutePlan ? &synchronizationSelection : nullptr))) {
      return failure();
    }
    SmallVector<PipeTransferNodeId> selectedCapacityTransfers =
        selectCapacityTransfers(capacityFacts, pipeGraph,
                                preliminaryResourcePlan, fabricRoutePlan);
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
        selectCapacityTransfers(capacityFacts, pipeGraph, plan.resourcePlan,
                                fabricRoutePlan);
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
                 fabricRoutePlan ? &synchronizationSelection : nullptr))) {
    return failure();
  }

  auto selectSynchronizationProtocol = [&](PipeTransferNodeId transferNodeId) {
    const PipeTransferNode &transferNode =
        pipeGraph.getPipeTransferNode(transferNodeId);
    auto sendOp = cast<PipeTransferSendOp>(transferNode.sendOp);
    if (synchronizationSelection.usesFabricProtocol(sendOp)) {
      return PipeSynchronizationProtocol::Fabric;
    }
    return synchronizationSelection.usesCapacityProtocol(sendOp)
               ? PipeSynchronizationProtocol::Capacity
               : PipeSynchronizationProtocol::ReceiverPost;
  };
  FailureOr<PipeTransportPlan> maybeTransportPlan = buildPipeTransportPlan(
      pipeGraph, plan.capacityPlan, selectSynchronizationProtocol);
  if (failed(maybeTransportPlan)) {
    return failure();
  }
  plan.transportPlan = std::move(*maybeTransportPlan);
  finalizePipeTransportResources(plan.transportPlan, plan.resourcePlan);
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
  auto addTransferPlan =
      [&](Operation *operation, PipeReference pipeReference,
          PipeTransferPlan::Resources resources) -> LogicalResult {
    auto sendOp = dyn_cast<PipeTransferSendOp>(operation);
    auto postOp = dyn_cast<PipeTransferPostOp>(operation);
    auto waitOp = dyn_cast<PipeTransferWaitOp>(operation);
    assert((sendOp || postOp || waitOp) &&
           "pipe resources assigned to an unsupported operation");
    bool usesFabricProtocol =
        (sendOp || postOp) &&
        synchronizationSelection.usesFabricProtocol(operation);
    bool usesCapacityProtocol =
        (sendOp || postOp) &&
        synchronizationSelection.usesCapacityProtocol(operation);
    PipeSynchronizationProtocol synchronizationProtocol =
        usesFabricProtocol     ? PipeSynchronizationProtocol::Fabric
        : usesCapacityProtocol ? PipeSynchronizationProtocol::Capacity
                               : PipeSynchronizationProtocol::ReceiverPost;
    if (usesFabricProtocol && !allUseComputedReceiverDFB(resources)) {
      auto diagnostic = operation->emitError(
          "fabric pipe transfer requires computed receiver DFB addresses");
      ArrayRef<PipeTransferNodeId> transferNodeIds =
          pipeGraph.getPipeTransferNodeIdsForProtocolOp(operation);
      bool attachedReason = false;
      for (PipeTransferNodeId transferNodeId : transferNodeIds) {
        const PipeTransferNode &transferNode =
            pipeGraph.getPipeTransferNode(transferNodeId);
        for (PipeReceiverEndpointId endpointId :
             pipeGraph.getPipeReceiverEndpoints(transferNode.id)) {
          const PipeReceiverEndpoint &endpoint =
              pipeGraph.getPipeReceiverEndpoint(endpointId);
          const PipeReceiverDFBNode &receiverDFB =
              pipeGraph.getReceiverDFBNode(endpoint.receiverDFBNode);
          if (!receiverDFB.hasProvenPipeOnlyProducerStream) {
            diagnostic.attachNote(endpoint.receiverDFBInfo.loc)
                << "receiver DFB " << endpoint.receiverDFBInfo.dfbIndex << ": "
                << receiverDFB.pipeOnlyProducerStreamFailureReason;
            attachedReason = true;
            break;
          }
          if (endpoint.addressSequence.getKind() ==
              ReceiverAddressSequenceProofKind::FullyDynamic) {
            diagnostic.attachNote(endpoint.receiverDFBInfo.loc)
                << "receiver DFB " << endpoint.receiverDFBInfo.dfbIndex
                << " has no proven receiver address sequence";
            attachedReason = true;
            break;
          }
        }
        if (attachedReason) {
          break;
        }
      }
      if (!attachedReason) {
        assert(!transferNodeIds.empty() &&
               "pipe resources require at least one transfer node");
        const PipeTransferNode &transferNode =
            pipeGraph.getPipeTransferNode(transferNodeIds.front());
        assert(!transferNode.receiverEndpoints.empty() &&
               "pipe transfer node requires at least one receiver endpoint");
        const PipeReceiverEndpoint &endpoint =
            pipeGraph.getPipeReceiverEndpoint(
                transferNode.receiverEndpoints.front());
        diagnostic.attachNote(endpoint.receiverDFBInfo.loc)
            << "receiver address sequences are not proven equal for every "
               "transfer occurrence";
      }
      return failure();
    }

    auto insertTransferPlan =
        [&](PipeTransferPlan::OperationPlan operationPlan) {
          PipeTransferPlan transferPlan(
              std::move(pipeReference), std::move(resources),
              synchronizationProtocol, std::move(operationPlan));
          auto [planIt, inserted] =
              plan.transferPlans.insert({operation, std::move(transferPlan)});
          (void)planIt;
          assert(inserted && "pipe operation has more than one transfer plan");
        };

    if (sendOp) {
      PipeTransferCreateOp transferCreate =
          transferIndex.getTransferCreate(operation);
      FailureOr<PipeSendPlan> maybeSendPlan = buildPipeSendPlan(
          sendOp, dominanceInfo, getPipeTransferBlockSpan(transferCreate),
          fabricRoutePlan);
      if (failed(maybeSendPlan)) {
        return failure();
      }
      insertTransferPlan(*maybeSendPlan);
    } else if (postOp) {
      FailureOr<PipePostPlan> maybePostPlan = [&]() {
        if (const auto *staticResource =
                std::get_if<PipeResourceInfo>(&resources)) {
          return buildPipePostPlan(
              postOp, ArrayRef<PipeResourceInfo>(staticResource, 1),
              fabricRoutePlan);
        }
        return buildPipePostPlan(
            postOp, std::get<SelectedPipeResources>(resources).destination,
            fabricRoutePlan);
      }();
      if (failed(maybePostPlan)) {
        return failure();
      }
      insertTransferPlan(*maybePostPlan);
    } else {
      insertTransferPlan(PipeWaitPlan{});
    }
    return success();
  };

  LogicalResult traversalResult = plan.resourcePlan.forEachResourceTable(
      [&](Operation *operation, ArrayRef<PipeResourceInfo> resources,
          PipeResourceTableKind tableKind) {
        FailureOr<PipeReference> maybePipeReference =
            getPipeReferenceForProtocolOp(operation, transferIndex);
        if (failed(maybePipeReference)) {
          return failure();
        }
        if (tableKind == PipeResourceTableKind::Static) {
          assert(maybePipeReference->isStatic() && resources.size() == 1 &&
                 "static resources require one static pipe reference");
          return addTransferPlan(operation, std::move(*maybePipeReference),
                                 resources.front());
        }
        assert(maybePipeReference->isSelected() &&
               "selected resources require a selected pipe reference");
        FailureOr<SelectedPipeResources> selectedResources =
            buildSelectedResources(operation, maybePipeReference->getRecords(),
                                   resources);
        if (failed(selectedResources)) {
          return failure();
        }
        return addTransferPlan(operation, std::move(*maybePipeReference),
                               std::move(*selectedResources));
      });
  if (failed(traversalResult)) {
    return failure();
  }

  return plan;
}

void applyPipeModuleAttributes(ModuleOp module, const PipeModulePlan &plan) {
  Builder builder(module.getContext());
  const PipeResourcePlan &resources = plan.getResourcePlan();
  module.walk([&](func::FuncOp function) {
    function->removeAttr(kPipeComputedAddressDFBIndicesAttrName);
  });
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
