//===- FabricForwarderPlan.cpp - Fabric forwarder planning ----------------===//
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "FabricForwarderPlan.h"

#include "PipeLowering.h"
#include "PipePlanning.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "ttlang/Analysis/LoopIterationUtils.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Transforms/PipeConstants.h"
#include "ttlang/Dialect/TTL/Transforms/PipeTransferAnalysis.h"
#include "ttlang/Target/TargetInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace mlir::tt::ttl {

namespace {

// TT-Metal can bind at most two independent Blackhole forwarding links for one
// route direction; runtime binding rejects hardware with fewer required links.
constexpr int64_t kMaximumBlackholeFabricForwardersPerRoute = 2;

// Active records assigned to one forwarder for one operation and route.
struct FabricForwarderGroupPlan {
  LaunchNodeCoord forwarder;
  SmallVector<int64_t> recordIndices;
};

// Selected-record facts retained after proving one operation can aggregate.
struct FabricOperationCandidate {
  // The kernel and protocol operation described by these record tables.
  func::FuncOp function;
  Operation *operation = nullptr;
  // Dense record-indexed tables. Records excluded by the operation domain
  // retain default entries and never appear in `recordIndicesByRoute`.
  SmallVector<LaunchNodeCoord> localNodesByRecord;
  SmallVector<LaunchNodeCoord> remoteNodesByRecord;
  SmallVector<const FabricRoute *> routesByRecord;
  // Active record indices grouped by their physical fabric route.
  llvm::MapVector<const FabricRoute *, SmallVector<int64_t>>
      recordIndicesByRoute;
  // Each selected record must execute this many times on every participating
  // device and worker.
  std::uint64_t executionsPerRecord = 0;
};

// Return the common positive execution count at every participating location.
static std::optional<std::uint64_t>
getUniformPositiveExecutionCount(Operation *operation,
                                 const PipeGraph &pipeGraph,
                                 ArrayRef<LaunchExecutionLocation> locations) {
  std::optional<std::uint64_t> uniformExecutionCount;
  for (const LaunchExecutionLocation &location : locations) {
    std::optional<std::uint64_t> executionCount =
        pipeGraph.getExactExecutionCountAtLaunchLocation(operation, location);
    if (!executionCount || *executionCount == 0 ||
        (uniformExecutionCount && *uniformExecutionCount != *executionCount)) {
      return std::nullopt;
    }
    uniformExecutionCount = *executionCount;
  }
  return uniformExecutionCount;
}

// Return whether every location reaches corresponding executions through the
// same statically bounded loops and selected conditional regions. Equal totals
// alone cannot justify the rendezvous inserted by forwarder aggregation.
static bool hasUniformForwarderControlFlow(
    Operation *operation, Operation *selectedPipeOperation,
    const PipeForeachLoweringInfo &foreachInfo,
    const llvm::SmallPtrSetImpl<Operation *> &generatedControlOps,
    const PipeGraph &pipeGraph, ArrayRef<LaunchExecutionLocation> locations) {
  Operation *selectedRecordLoop = selectedPipeOperation->getParentOp();
  while (selectedRecordLoop &&
         !foreachInfo.recordLoops.contains(selectedRecordLoop)) {
    selectedRecordLoop = selectedRecordLoop->getParentOp();
  }
  if (!selectedRecordLoop || !selectedRecordLoop->isProperAncestor(operation)) {
    return false;
  }

  Operation *nestedOperation = operation;
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto recordLoop = foreachInfo.recordLoops.find(parent);
    if (recordLoop != foreachInfo.recordLoops.end()) {
      if (parent != selectedRecordLoop) {
        std::optional<std::uint64_t> uniformMatchingCount;
        for (const LaunchExecutionLocation &location : locations) {
          std::optional<std::uint64_t> matchingCount =
              getMatchingPipeNetRecordCount(recordLoop->second, location);
          if (!matchingCount || *matchingCount == 0 ||
              (uniformMatchingCount &&
               *uniformMatchingCount != *matchingCount)) {
            return false;
          }
          uniformMatchingCount = *matchingCount;
        }
      }
      nestedOperation = parent;
      continue;
    }
    // Record-loop metadata accounts for callback multiplicity; auxiliary
    // controls generated for record selection do not add user control flow.
    if (generatedControlOps.contains(parent)) {
      nestedOperation = parent;
      continue;
    }
    if (isa<func::FuncOp>(parent)) {
      return true;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      unsigned selectedRegion =
          nestedOperation->getParentRegion()->getRegionNumber();
      if (selectedRegion > 1 ||
          llvm::any_of(locations, [&](const LaunchExecutionLocation &location) {
            std::optional<bool> condition =
                pipeGraph.evaluatePredicateAtLaunchLocation(ifOp.getCondition(),
                                                            location);
            return !condition || *condition != (selectedRegion == 0);
          })) {
        return false;
      }
      nestedOperation = parent;
      continue;
    }
    if (auto loop = dyn_cast<LoopLikeOpInterface>(parent)) {
      std::optional<std::uint64_t> tripCount = tt::getLoopTripCount(loop);
      if (!tripCount || *tripCount == 0) {
        return false;
      }
      nestedOperation = parent;
      continue;
    }
    if (auto executeRegion = dyn_cast<scf::ExecuteRegionOp>(parent)) {
      if (!executeRegion.getRegion().hasOneBlock()) {
        return false;
      }
      nestedOperation = parent;
      continue;
    }
    if (isa<PipeNetScopeOp>(parent)) {
      nestedOperation = parent;
      continue;
    }
    return false;
  }
  return false;
}

// Orient the record's device edge from the kernel executing the operation.
static std::pair<DeviceRefAttr, DeviceRefAttr>
getLocalAndRemoteDevices(PipeRecordAttr record, bool isSender) {
  DeviceTransferAttr transfer = record.getDeviceTransfer();
  assert(transfer && transfer.getEdge().getDestination() &&
         "fabric record must have a point destination device");
  DeviceRefAttr source = transfer.getEdge().getSource();
  DeviceRefAttr destination = transfer.getEdge().getDestination();
  return isSender ? std::make_pair(source, destination)
                  : std::make_pair(destination, source);
}

// Return the function route used by `record` in the operation's direction.
static const FabricRoute *findDeviceRouteForRecord(const FabricRoutePlan &plan,
                                                   func::FuncOp function,
                                                   PipeRecordAttr record,
                                                   bool isSender) {
  auto functionIt = plan.routesByFunction.find(function);
  assert(functionIt != plan.routesByFunction.end() &&
         "fabric operation has no function route plan");
  auto devices = getLocalAndRemoteDevices(record, isSender);
  auto route = llvm::find_if(functionIt->second.routes,
                             [&](const FabricRoute &candidate) {
                               return candidate.localDevice == devices.first &&
                                      candidate.remoteDevice == devices.second;
                             });
  assert(route != functionIt->second.routes.end() &&
         "fabric operation names an unknown route");
  return &*route;
}

// Return the endpoint that executes the operation represented by `record`.
static LaunchNodeCoord getLocalProtocolNode(PipeRecordAttr record,
                                            bool isSender) {
  return isSender
             ? LaunchNodeCoord{record.getSrcX(), record.getSrcY()}
             : LaunchNodeCoord{record.getDstStartX(), record.getDstStartY()};
}

// Return the endpoint reached by the operation represented by `record`.
static LaunchNodeCoord getRemoteProtocolNode(PipeRecordAttr record,
                                             bool isSender) {
  return isSender
             ? LaunchNodeCoord{record.getDstStartX(), record.getDstStartY()}
             : LaunchNodeCoord{record.getSrcX(), record.getSrcY()};
}

// Coordinate sorting makes metadata deterministic. Balanced groups minimize
// the largest serial transfer count. The first member is the forwarder so it
// is guaranteed to execute the operation.
static SmallVector<FabricForwarderGroupPlan>
partitionRecordsByNode(ArrayRef<LaunchNodeCoord> localNodesByRecord,
                       ArrayRef<int64_t> recordIndices,
                       int64_t maximumForwarders) {
  SmallVector<std::pair<LaunchNodeCoord, int64_t>> sortedRecords;
  sortedRecords.reserve(recordIndices.size());
  for (int64_t recordIndex : recordIndices) {
    sortedRecords.emplace_back(localNodesByRecord[recordIndex], recordIndex);
  }
  llvm::sort(sortedRecords);
  int64_t groupCount =
      std::min<int64_t>(maximumForwarders, sortedRecords.size());
  int64_t baseGroupSize = sortedRecords.size() / groupCount;
  int64_t extraMembers = sortedRecords.size() % groupCount;

  SmallVector<FabricForwarderGroupPlan> groups;
  std::size_t nextRecord = 0;
  for (int64_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
    int64_t groupSize = baseGroupSize + (groupIndex < extraMembers ? 1 : 0);
    FabricForwarderGroupPlan &group = groups.emplace_back();
    group.forwarder = sortedRecords[nextRecord].first;
    for (int64_t memberIndex = 0; memberIndex < groupSize; ++memberIndex) {
      group.recordIndices.push_back(
          sortedRecords[nextRecord + memberIndex].second);
    }
    nextRecord += groupSize;
  }
  return groups;
}

// Return whether `route` owns the local endpoint for `record`.
static bool routeMatchesRecord(const FabricRoute &route, PipeRecordAttr record,
                               bool isSender) {
  auto [localDevice, remoteDevice] = getLocalAndRemoteDevices(record, isSender);
  return route.localDevice == localDevice &&
         route.remoteDevice == remoteDevice &&
         llvm::is_contained(route.sourceNodes,
                            getLocalProtocolNode(record, isSender));
}

// Return whether each local device and node matches at most one record;
// repeated matches would alias its per-operation scratch state.
static bool
hasAtMostOneRecordPerLocalNode(const FabricOperationCandidate &candidate) {
  llvm::DenseMap<DeviceRefAttr, std::set<LaunchNodeCoord>> nodesByDevice;
  for (const auto &[route, recordIndices] : candidate.recordIndicesByRoute) {
    for (int64_t recordIndex : recordIndices) {
      if (!nodesByDevice[route->localDevice]
               .insert(candidate.localNodesByRecord[recordIndex])
               .second) {
        return false;
      }
    }
  }
  return true;
}

// Return whether candidate operations cover each local node for `route`
// exactly once.
static bool routeCandidatesPartitionSourceNodes(
    const FabricRoute &route, ArrayRef<std::size_t> candidateIndices,
    ArrayRef<FabricOperationCandidate> candidates) {
  SmallVector<LaunchNodeCoord> candidateLocalNodes;
  for (std::size_t candidateIndex : candidateIndices) {
    const FabricOperationCandidate &candidate = candidates[candidateIndex];
    auto recordIndicesIt = candidate.recordIndicesByRoute.find(&route);
    assert(recordIndicesIt != candidate.recordIndicesByRoute.end() &&
           "route candidate must contain records for the route");
    for (int64_t recordIndex : recordIndicesIt->second) {
      candidateLocalNodes.push_back(candidate.localNodesByRecord[recordIndex]);
    }
  }
  return launchNodeSetsEqual(candidateLocalNodes, route.sourceNodes);
}

// Reserve one forwarder for each operation because their control regions can
// execute independently; use remaining route capacity on the largest group.
static SmallVector<int64_t> allocateRouteForwarders(
    const FabricRoute &route, ArrayRef<std::size_t> candidateIndices,
    ArrayRef<FabricOperationCandidate> candidates, int64_t maximumForwarders) {
  assert(!candidateIndices.empty() &&
         candidateIndices.size() <=
             static_cast<std::size_t>(maximumForwarders) &&
         "every route candidate requires one forwarder");
  SmallVector<int64_t> forwarderCounts(candidateIndices.size(), 1);
  int64_t remainingForwarders =
      maximumForwarders - static_cast<int64_t>(candidateIndices.size());
  while (remainingForwarders > 0) {
    std::optional<std::size_t> selectedCandidate;
    std::uint64_t largestGroupSize = 0;
    for (auto [position, candidateIndex] : llvm::enumerate(candidateIndices)) {
      const FabricOperationCandidate &candidate = candidates[candidateIndex];
      auto recordIndicesIt = candidate.recordIndicesByRoute.find(&route);
      assert(recordIndicesIt != candidate.recordIndicesByRoute.end() &&
             "route candidate must contain records for the route");
      std::uint64_t recordCount = recordIndicesIt->second.size();
      if (forwarderCounts[position] >= static_cast<int64_t>(recordCount)) {
        continue;
      }
      std::uint64_t groupSize = llvm::divideCeil(
          recordCount, static_cast<std::uint64_t>(forwarderCounts[position]));
      if (!selectedCandidate || groupSize > largestGroupSize) {
        selectedCandidate = position;
        largestGroupSize = groupSize;
      }
    }
    if (!selectedCandidate) {
      break;
    }
    ++forwarderCounts[*selectedCandidate];
    --remainingForwarders;
  }
  return forwarderCounts;
}

// Return whether a group's cumulative arrivals remain representable across
// every execution of the selected record.
static bool canRepresentForwarderCounterValue(std::uint64_t executionsPerRecord,
                                              std::uint64_t groupSize) {
  constexpr std::uint64_t maximumCounterValue =
      std::numeric_limits<std::uint32_t>::max();
  return executionsPerRecord <= maximumCounterValue / groupSize;
}

} // namespace

const FabricForwarderOperationPlan *
FabricForwarderPlan::lookup(Operation *operation) const {
  auto operationIt = operations.find(operation);
  return operationIt == operations.end() ? nullptr : &operationIt->second;
}

FailureOr<FabricForwarderPlan> buildFabricForwarderPlan(
    ModuleOp module, const PipeTransferIndex &transferIndex,
    const PipeForeachLoweringInfo &foreachLoweringInfo,
    const PipeGraph &pipeGraph, const FabricRoutePlan &fabricRoutePlan) {
  FabricForwarderPlan plan;
  if (fabricRoutePlan.routesByFunction.empty()) {
    return plan;
  }

  std::string targetFailureReason;
  FailureOr<std::optional<ttcore::Arch>> targetArch =
      resolveTargetArch(module, targetFailureReason);
  if (failed(targetArch)) {
    module.emitOpError() << targetFailureReason;
    return failure();
  }
  if (!*targetArch || **targetArch != ttcore::Arch::Blackhole) {
    return plan;
  }

  llvm::SmallPtrSet<Operation *, 16> generatedControlOps(
      foreachLoweringInfo.controlOps.begin(),
      foreachLoweringInfo.controlOps.end());
  llvm::DenseMap<const FabricRoute *, unsigned> operationUseCounts;
  SmallVector<FabricOperationCandidate, 0> candidates;
  for (const auto &[operation, routeIndices] : fabricRoutePlan.routeIndices) {
    ArrayRef<std::size_t> operationRouteIndices = routeIndices;
    FailureOr<PipeReference> pipeReference =
        getPipeReferenceForProtocolOp(operation, transferIndex);
    if (failed(pipeReference)) {
      return failure();
    }
    bool isSender = isa<PipeTransferSendOp>(operation);
    bool isReceiver = isa<PipeTransferPostOp>(operation);
    assert((isSender || isReceiver) &&
           "fabric routes belong to sends or receiver posts");
    // Aggregation is valid only when every selected record sends readiness;
    // otherwise lowering handles each readiness-producing record separately.
    if (isReceiver &&
        !fabricRoutePlan.receiversWithReadinessForEveryRecord.contains(
            operation)) {
      continue;
    }
    bool selectedEndpoint = pipeReference->isSelected() &&
                            (!isSender || pipeReference->isSelectedSrc()) &&
                            (!isReceiver || pipeReference->isSelectedDst());
    func::FuncOp function = operation->getParentOfType<func::FuncOp>();
    assert(function && "fabric protocol operation must be inside a function");
    LaunchNodeDomain operationDomain =
        pipeGraph.getOperationLaunchDomain(operation);
    llvm::SmallPtrSet<const FabricRoute *, 4> operationRoutes;
    std::optional<FabricOperationCandidate> candidate;
    SmallVector<LaunchExecutionLocation> activeLocations;
    bool candidateRecordsValid = true;
    if (pipeReference->isSelected()) {
      PipeNetRecordsAttr records = pipeReference->getRecords();
      FailureOr<std::uint64_t> recordCount = getPipeRecordCount(records);
      assert(succeeded(recordCount) &&
             operationRouteIndices.size() == *recordCount &&
             "selected fabric route table must match its records");
      // Function-entry expectations cannot represent repeated helper calls or
      // control transfers that revisit a kernel block.
      if (selectedEndpoint && function->hasAttr(kKernelThreadAttrName) &&
          function.getBody().hasOneBlock()) {
        candidate.emplace();
        candidate->function = function;
        candidate->operation = operation;
        candidate->localNodesByRecord.resize(*recordCount);
        candidate->remoteNodesByRecord.resize(*recordCount);
        candidate->routesByRecord.resize(*recordCount);
      }
      forEachPipeRecord(records, [&](std::uint64_t recordIndex,
                                     PipeRecordAttr record) {
        DeviceTransferAttr deviceTransfer = record.getDeviceTransfer();
        if (!deviceTransfer || deviceTransfer.getEdge().getSource() ==
                                   deviceTransfer.getEdge().getDestination()) {
          candidateRecordsValid = false;
          return;
        }
        const FabricRoute *route = findDeviceRouteForRecord(
            fabricRoutePlan, function, record, isSender);
        bool executesRoute =
            route->routeIndex == operationRouteIndices[recordIndex] &&
            routeMatchesRecord(*route, record, isSender);
        LaunchNodeCoord localNode = getLocalProtocolNode(record, isSender);
        bool mayExecuteOperation =
            executesRoute &&
            (!operationDomain.known ||
             knownLaunchNodeDomainContains(operationDomain, localNode));
        bool executesOperation = mayExecuteOperation && operationDomain.known;
        if (mayExecuteOperation) {
          operationRoutes.insert(route);
        }
        if (!candidate) {
          return;
        }
        if (record.getDstStartX() != record.getDstEndX() ||
            record.getDstStartY() != record.getDstEndY()) {
          candidateRecordsValid = false;
          return;
        }
        // Dense route tables use valid route indices for inactive rows. Only
        // rows selected by both the route and operation domain participate.
        if (!executesOperation) {
          return;
        }
        candidate->localNodesByRecord[recordIndex] = localNode;
        candidate->remoteNodesByRecord[recordIndex] =
            getRemoteProtocolNode(record, isSender);
        candidate->routesByRecord[recordIndex] = route;
        candidate->recordIndicesByRoute[route].push_back(
            static_cast<int64_t>(recordIndex));
        activeLocations.emplace_back(localNode, deviceTransfer.getDomain(),
                                     route->localDevice);
      });
    } else {
      for (std::size_t routeIndex : routeIndices) {
        const FunctionFabricRoutePlan &functionPlan =
            fabricRoutePlan.routesByFunction.find(function)->second;
        for (const FabricRoute &route : functionPlan.routes) {
          if (route.routeIndex == routeIndex) {
            operationRoutes.insert(&route);
          }
        }
      }
    }
    for (const FabricRoute *route : operationRoutes) {
      ++operationUseCounts[route];
    }
    if (!candidate || !candidateRecordsValid ||
        candidate->recordIndicesByRoute.empty()) {
      continue;
    }

    if (!hasAtMostOneRecordPerLocalNode(*candidate)) {
      continue;
    }
    std::set<LaunchNodeCoord> activeLocalNodeSet;
    for (const auto &routeRecords : candidate->recordIndicesByRoute) {
      for (int64_t recordIndex : routeRecords.second) {
        LaunchNodeCoord node = candidate->localNodesByRecord[recordIndex];
        activeLocalNodeSet.insert(node);
      }
    }
    SmallVector<LaunchNodeCoord> activeLocalNodes(activeLocalNodeSet.begin(),
                                                  activeLocalNodeSet.end());
    SmallVector<LaunchNodeCoord> domainNodes(operationDomain.nodes.begin(),
                                             operationDomain.nodes.end());
    // Every node that can execute the operation must enter one planned group;
    // otherwise a participant can wait for a worker that never arrives.
    if (!operationDomain.known ||
        !launchNodeSetsEqual(activeLocalNodes, domainNodes)) {
      continue;
    }
    std::optional<std::uint64_t> executionsPerRecord =
        getUniformPositiveExecutionCount(operation, pipeGraph, activeLocations);
    if (!executionsPerRecord ||
        !hasUniformForwarderControlFlow(
            operation, pipeReference->getSelectedOperation(),
            foreachLoweringInfo, generatedControlOps, pipeGraph,
            activeLocations)) {
      continue;
    }
    candidate->executionsPerRecord = *executionsPerRecord;
    candidates.push_back(std::move(*candidate));
  }

  llvm::DenseMap<const FabricRoute *, SmallVector<std::size_t>>
      candidateIndicesByRoute;
  for (auto [candidateIndex, candidate] : llvm::enumerate(candidates)) {
    for (const auto &[route, recordIndices] : candidate.recordIndicesByRoute) {
      (void)recordIndices;
      candidateIndicesByRoute[route].push_back(candidateIndex);
    }
  }
  llvm::SmallPtrSet<const FabricRoute *, 16> unavailableRoutes;
  for (const auto &[route, useCount] : operationUseCounts) {
    ArrayRef<std::size_t> routeCandidates = candidateIndicesByRoute[route];
    if (routeCandidates.size() != useCount ||
        routeCandidates.size() >
            static_cast<std::size_t>(
                kMaximumBlackholeFabricForwardersPerRoute) ||
        !routeCandidatesPartitionSourceNodes(*route, routeCandidates,
                                             candidates)) {
      unavailableRoutes.insert(route);
      continue;
    }
    SmallVector<int64_t> forwarderCounts =
        allocateRouteForwarders(*route, routeCandidates, candidates,
                                kMaximumBlackholeFabricForwardersPerRoute);
    for (auto [candidatePosition, candidateIndex] :
         llvm::enumerate(routeCandidates)) {
      const FabricOperationCandidate &candidate = candidates[candidateIndex];
      std::uint64_t recordCount =
          candidate.recordIndicesByRoute.find(route)->second.size();
      std::uint64_t maximumGroupSize = llvm::divideCeil(
          recordCount,
          static_cast<std::uint64_t>(forwarderCounts[candidatePosition]));
      if (!canRepresentForwarderCounterValue(candidate.executionsPerRecord,
                                             maximumGroupSize)) {
        unavailableRoutes.insert(route);
        break;
      }
    }
  }
  // A route retains all direct workers if any operation using it cannot
  // aggregate. Every candidate sharing that route must then remain direct.
  SmallVector<bool> candidateAvailable(candidates.size(), true);
  bool availabilityChanged;
  do {
    availabilityChanged = false;
    for (auto [candidateIndex, candidate] : llvm::enumerate(candidates)) {
      if (!candidateAvailable[candidateIndex] ||
          llvm::none_of(candidate.recordIndicesByRoute, [&](const auto &entry) {
            return unavailableRoutes.contains(entry.first);
          })) {
        continue;
      }
      candidateAvailable[candidateIndex] = false;
      availabilityChanged = true;
      for (const auto &[route, recordIndices] :
           candidate.recordIndicesByRoute) {
        (void)recordIndices;
        unavailableRoutes.insert(route);
      }
    }
  } while (availabilityChanged);

  llvm::SmallPtrSet<const FabricRoute *, 16> selectedRoutes;
  SmallVector<bool> candidateSelected(candidates.size(), false);
  for (const auto &[route, routeCandidates] : candidateIndicesByRoute) {
    if (!unavailableRoutes.contains(route) &&
        route->sourceNodes.size() >
            static_cast<std::size_t>(
                kMaximumBlackholeFabricForwardersPerRoute)) {
      selectedRoutes.insert(route);
    }
  }
  // One operation uses the forwarder protocol for every record it can select.
  // Selecting any of its routes therefore selects all of them.
  do {
    availabilityChanged = false;
    for (auto [candidateIndex, candidate] : llvm::enumerate(candidates)) {
      if (!candidateAvailable[candidateIndex] ||
          candidateSelected[candidateIndex] ||
          llvm::none_of(candidate.recordIndicesByRoute, [&](const auto &entry) {
            return selectedRoutes.contains(entry.first);
          })) {
        continue;
      }
      candidateSelected[candidateIndex] = true;
      availabilityChanged = true;
      for (const auto &[route, recordIndices] :
           candidate.recordIndicesByRoute) {
        (void)recordIndices;
        selectedRoutes.insert(route);
      }
    }
  } while (availabilityChanged);

  llvm::DenseMap<const FabricRoute *, SmallVector<int64_t>>
      routeForwarderCounts;
  for (const FabricRoute *route : selectedRoutes) {
    routeForwarderCounts[route] = allocateRouteForwarders(
        *route, candidateIndicesByRoute[route], candidates,
        kMaximumBlackholeFabricForwardersPerRoute);
  }

  llvm::DenseMap<const FabricRoute *, std::size_t> routePlanIndices;
  for (auto [candidateIndex, candidate] : llvm::enumerate(candidates)) {
    if (!candidateSelected[candidateIndex]) {
      continue;
    }
    SmallVector<FabricForwarderGroupPlan> groups;
    for (const auto &[route, recordIndices] : candidate.recordIndicesByRoute) {
      ArrayRef<std::size_t> routeCandidates = candidateIndicesByRoute[route];
      auto candidatePosition = llvm::find(routeCandidates, candidateIndex);
      assert(candidatePosition != routeCandidates.end() &&
             "selected route must contain its candidate");
      std::size_t position = candidatePosition - routeCandidates.begin();
      llvm::append_range(
          groups,
          partitionRecordsByNode(candidate.localNodesByRecord, recordIndices,
                                 routeForwarderCounts[route][position]));
    }

    FabricForwarderOperationPlan &operationPlan =
        plan.operations[candidate.operation];
    std::size_t recordCount = candidate.localNodesByRecord.size();
    operationPlan.groupIndexByRecord.assign(recordCount, 0);
    operationPlan.slotByRecord.assign(recordCount, 0);
    operationPlan.forwarderXByGroup.reserve(groups.size());
    operationPlan.forwarderYByGroup.reserve(groups.size());
    operationPlan.groupSizeByGroup.reserve(groups.size());

    for (auto [groupIndex, group] : llvm::enumerate(groups)) {
      assert(!group.recordIndices.empty() &&
             "every forwarder group must contain one record");
      int64_t groupSize = static_cast<int64_t>(group.recordIndices.size());
      operationPlan.maximumGroupSize =
          std::max(operationPlan.maximumGroupSize, groupSize);
      operationPlan.forwarderXByGroup.push_back(group.forwarder.x);
      operationPlan.forwarderYByGroup.push_back(group.forwarder.y);
      operationPlan.groupSizeByGroup.push_back(groupSize);
      for (auto [slotIndex, recordIndex] :
           llvm::enumerate(group.recordIndices)) {
        std::size_t recordPosition = static_cast<std::size_t>(recordIndex);
        const FabricRoute *route = candidate.routesByRecord[recordPosition];
        operationPlan.groupIndexByRecord[recordPosition] = groupIndex;
        operationPlan.slotByRecord[recordPosition] = slotIndex;

        auto routePlanIt = routePlanIndices.find(route);
        if (routePlanIt == routePlanIndices.end()) {
          std::size_t routePlanIndex = plan.routes.size();
          routePlanIndices[route] = routePlanIndex;
          plan.routes.push_back(
              FabricForwarderPlan::RouteSourcePlan{candidate.function,
                                                   route->localDevice,
                                                   route->remoteDevice,
                                                   route->routeIndex,
                                                   {}});
          routePlanIt = routePlanIndices.find(route);
        }
        SmallVectorImpl<LaunchNodeCoord> &routeForwarders =
            plan.routes[routePlanIt->second].sourceNodes;
        if (!llvm::is_contained(routeForwarders, group.forwarder)) {
          routeForwarders.push_back(group.forwarder);
        }
      }
    }

    std::optional<int64_t> groupTableSize = llvm::checkedMul(
        static_cast<int64_t>(groups.size()), operationPlan.maximumGroupSize);
    if (!groupTableSize) {
      candidate.operation->emitError(
          "fabric forwarder record table is too large");
      return failure();
    }
    operationPlan.groupRecordsByGroupAndSlot.assign(*groupTableSize, 0);
    operationPlan.workerXByGroupAndSlot.assign(*groupTableSize, 0);
    operationPlan.workerYByGroupAndSlot.assign(*groupTableSize, 0);
    operationPlan.peerXByGroupAndSlot.assign(*groupTableSize, 0);
    operationPlan.peerYByGroupAndSlot.assign(*groupTableSize, 0);
    for (auto [groupIndex, group] : llvm::enumerate(groups)) {
      for (auto [slotIndex, memberRecord] :
           llvm::enumerate(group.recordIndices)) {
        int64_t tableIndex =
            groupIndex * operationPlan.maximumGroupSize + slotIndex;
        operationPlan.groupRecordsByGroupAndSlot[tableIndex] = memberRecord;
        operationPlan.workerXByGroupAndSlot[tableIndex] =
            candidate.localNodesByRecord[memberRecord].x;
        operationPlan.workerYByGroupAndSlot[tableIndex] =
            candidate.localNodesByRecord[memberRecord].y;
        operationPlan.peerXByGroupAndSlot[tableIndex] =
            candidate.remoteNodesByRecord[memberRecord].x;
        operationPlan.peerYByGroupAndSlot[tableIndex] =
            candidate.remoteNodesByRecord[memberRecord].y;
      }
    }

    if (auto sendOp = dyn_cast<PipeTransferSendOp>(candidate.operation)) {
      PipeTransferCreateOp transferCreate =
          transferIndex.getTransferCreate(candidate.operation);
      FailureOr<PipeTransferPayload> payload = getPipeTransferPayload(
          sendOp, getPipeTransferBlockSpan(transferCreate));
      if (failed(payload)) {
        return failure();
      }
      operationPlan.payloadSizeBytes = payload->sizeBytes;
      std::optional<int64_t> payloadStrideBytes =
          alignPipeSramScratchBytes(payload->sizeBytes);
      if (!payloadStrideBytes) {
        candidate.operation->emitError(
            "fabric forwarder payload size is too large");
        return failure();
      }
      operationPlan.payloadStrideBytes = *payloadStrideBytes;
    }
  }

  int64_t nextOperationOffset = 0;
  for (auto &[operation, operationPlan] : plan.operations) {
    std::optional<int64_t> alignedOperationOffset =
        alignPipeSramScratchBytes(nextOperationOffset);
    if (!alignedOperationOffset) {
      operation->emitError("fabric forwarder scratch size is too large");
      return failure();
    }
    nextOperationOffset = *alignedOperationOffset;
    operationPlan.scratchByteOffset = nextOperationOffset;
    std::optional<int64_t> payloadBytes = llvm::checkedMul(
        operationPlan.payloadStrideBytes, operationPlan.maximumGroupSize);
    std::optional<int64_t> arrivalCounterOffset =
        payloadBytes
            ? llvm::checkedAdd(operationPlan.scratchByteOffset, *payloadBytes)
            : std::nullopt;
    std::optional<int64_t> completionCounterOffset =
        arrivalCounterOffset
            ? llvm::checkedAdd(*arrivalCounterOffset,
                               static_cast<int64_t>(sizeof(std::uint32_t)))
            : std::nullopt;
    std::optional<int64_t> nextOffset =
        completionCounterOffset
            ? llvm::checkedAdd(*completionCounterOffset,
                               static_cast<int64_t>(sizeof(std::uint32_t)))
            : std::nullopt;
    if (!nextOffset) {
      operation->emitError("fabric forwarder scratch size is too large");
      return failure();
    }
    operationPlan.arrivalCounterByteOffset = *arrivalCounterOffset;
    operationPlan.completionCounterByteOffset = *completionCounterOffset;
    nextOperationOffset = *nextOffset;
  }
  std::optional<int64_t> alignedScratchBytes =
      alignPipeSramScratchBytes(nextOperationOffset);
  if (!alignedScratchBytes) {
    module.emitError("fabric forwarder scratch size is too large");
    return failure();
  }
  plan.sramScratchBytes = plan.operations.empty() ? 0 : *alignedScratchBytes;
  return plan;
}

void applyFabricForwarderRoutes(const FabricForwarderPlan &forwarderPlan,
                                FabricRoutePlan &fabricRoutePlan) {
  for (const FabricForwarderPlan::RouteSourcePlan &plannedRoute :
       forwarderPlan.routes) {
    auto functionIt =
        fabricRoutePlan.routesByFunction.find(plannedRoute.function);
    assert(functionIt != fabricRoutePlan.routesByFunction.end() &&
           "planned forwarder function has no fabric routes");
    auto route = llvm::find_if(
        functionIt->second.routes, [&](const FabricRoute &candidate) {
          return candidate.localDevice == plannedRoute.localDevice &&
                 candidate.remoteDevice == plannedRoute.remoteDevice &&
                 candidate.routeIndex == plannedRoute.routeIndex;
        });
    assert(route != functionIt->second.routes.end() &&
           "planned forwarder route is missing");
    route->sourceNodes = plannedRoute.sourceNodes;
  }
}

} // namespace mlir::tt::ttl
