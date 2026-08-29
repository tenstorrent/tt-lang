// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeLowering.h"

#include "CommonRuntimeArgLayout.h"
#include "FabricManagerLifetimeAnalysis.h"
#include "PipePlanning.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Analysis/LoopIterationUtils.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LiveIntervalUtils.h"
#include "ttlang/Dialect/TTL/Transforms/PipeConstants.h"
#include "ttlang/Dialect/TTL/Transforms/PipeRecordLoweringUtils.h"
#include "ttlang/Dialect/TTL/Transforms/PipeTransferAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "ttlang/Target/TargetInfo.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::tt::ttl {

using mlir::func::FuncOp;
namespace ttk = mlir::tt::ttkernel;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

template <typename ValueT>
class RecordAlignedTableBuilder {
public:
  LogicalResult
  set(Operation *operation, std::size_t recordCount, std::size_t recordIndex,
      ValueT value,
      llvm::function_ref<LogicalResult(Operation *, const ValueT &,
                                       const ValueT &)>
          handleCollision) {
    SmallVector<std::optional<ValueT>> &table = tables[operation];
    if (table.empty()) {
      table.resize(recordCount);
    }
    assert(table.size() == recordCount &&
           "one operation must use one selected record table");
    assert(recordIndex < recordCount && "selected record index out of range");
    if (table[recordIndex]) {
      return handleCollision(operation, *table[recordIndex], value);
    }
    table[recordIndex] = std::move(value);
    return success();
  }

  llvm::MapVector<Operation *, SmallVector<ValueT>> finalize() && {
    llvm::MapVector<Operation *, SmallVector<ValueT>> finalizedTables;
    for (auto &[operation, optionalValues] : tables) {
      auto firstActiveValue =
          llvm::find_if(optionalValues, [](const std::optional<ValueT> &value) {
            return value.has_value();
          });
      assert(firstActiveValue != optionalValues.end() &&
             "selected record table has no active rows");
      SmallVector<ValueT> values;
      values.reserve(optionalValues.size());
      for (const std::optional<ValueT> &value : optionalValues) {
        // Inactive rows are never read. Reusing an active value keeps every
        // table total and aligned with its selected record indices.
        values.push_back(value.value_or(**firstActiveValue));
      }
      finalizedTables.insert({operation, std::move(values)});
    }
    return finalizedTables;
  }

private:
  llvm::MapVector<Operation *, SmallVector<std::optional<ValueT>>> tables;
};

static Value makeZeroI32(Location loc, ConversionPatternRewriter &rewriter) {
  return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
}

struct PipeSourceKey {
  int64_t srcX;
  int64_t srcY;

  bool operator==(const PipeSourceKey &other) const {
    return srcX == other.srcX && srcY == other.srcY;
  }
};

/// Receiver location used to determine whether completion counters alias.
struct PipeCounterLocation {
  DeviceRefAttr device;
  int64_t nodeX;
  int64_t nodeY;

  bool operator==(const PipeCounterLocation &other) const {
    return device == other.device && nodeX == other.nodeX &&
           nodeY == other.nodeY;
  }
};

} // namespace mlir::tt::ttl

namespace llvm {
template <>
struct DenseMapInfo<mlir::tt::ttl::PipeSourceKey> {
  using Key = mlir::tt::ttl::PipeSourceKey;
  static unsigned getHashValue(const Key &sourceKey) {
    return hash_combine(sourceKey.srcX, sourceKey.srcY);
  }
  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
};

} // namespace llvm

namespace mlir::tt::ttl {

static PipeSourceKey getPipeSourceKey(PipeType pipeType) {
  return {pipeType.getSrcX(), pipeType.getSrcY()};
}

static std::size_t addFabricRoute(SmallVectorImpl<FabricRoute> &routes,
                                  DeviceRefAttr localDevice,
                                  DeviceRefAttr remoteDevice,
                                  LaunchNodeCoord localNode) {
  auto route = llvm::find_if(routes, [&](const FabricRoute &existing) {
    return existing.localDevice == localDevice &&
           existing.remoteDevice == remoteDevice;
  });
  if (route != routes.end()) {
    if (!llvm::is_contained(route->sourceNodes, localNode)) {
      route->sourceNodes.push_back(localNode);
    }
    return route->routeIndex;
  }

  std::size_t routeIndex =
      llvm::count_if(routes, [&](const FabricRoute &existing) {
        return existing.localDevice == localDevice;
      });
  routes.push_back(
      FabricRoute{localDevice, remoteDevice, {localNode}, routeIndex});
  return routeIndex;
}

static FailureOr<FunctionFabricRoutePlan *>
getFunctionFabricRoutePlan(func::FuncOp func, DeviceDomainAttr deviceDomain,
                           Operation *transferOp, FabricRoutePlan &plan) {
  FunctionFabricRoutePlan &functionPlan = plan.routesByFunction[func];
  if (functionPlan.deviceDomain && functionPlan.deviceDomain != deviceDomain) {
    transferOp->emitError(
        "all device transfers in one kernel must use the same device domain");
    return failure();
  }
  functionPlan.deviceDomain = deviceDomain;
  return &functionPlan;
}

static std::size_t getFabricRouteCount(ArrayRef<FabricRoute> routes) {
  std::size_t routeCount = 0;
  for (const FabricRoute &route : routes) {
    routeCount = std::max(routeCount, route.routeIndex + 1);
  }
  return routeCount;
}

static FabricManagerIntervalKind
getGeneratedFabricManagerKind(ArrayRef<Operation *> operations) {
  bool hasSend = llvm::any_of(operations, llvm::IsaPred<PipeTransferSendOp>);
  bool hasPost = llvm::any_of(operations, llvm::IsaPred<PipeTransferPostOp>);
  assert((hasSend || hasPost) &&
         "fabric runtime interval has no protocol operation");
  if (hasSend && hasPost) {
    return FabricManagerIntervalKind::GeneratedMixed;
  }
  return hasSend ? FabricManagerIntervalKind::GeneratedSender
                 : FabricManagerIntervalKind::GeneratedReceiver;
}

static SmallVector<std::size_t>
getIntervalRouteIndices(ArrayRef<Operation *> operations,
                        const FabricRoutePlan &plan) {
  llvm::SmallSetVector<std::size_t, 4> routeIndices;
  for (Operation *operation : operations) {
    routeIndices.insert_range(plan.lookupRouteIndices(operation));
  }
  return SmallVector<std::size_t>(routeIndices.begin(), routeIndices.end());
}

static SmallVector<PipeTransferNodeId>
getIntervalTransferNodes(ArrayRef<Operation *> operations,
                         const PipeGraph &pipeGraph) {
  llvm::SmallSetVector<PipeTransferNodeId, 4> transferNodes;
  for (Operation *operation : operations) {
    transferNodes.insert_range(
        pipeGraph.getPipeTransferNodeIdsForProtocolOp(operation));
  }
  SmallVector<PipeTransferNodeId> sortedTransferNodes(transferNodes.begin(),
                                                      transferNodes.end());
  llvm::sort(sortedTransferNodes);
  return sortedTransferNodes;
}

static bool sourceNodesEqual(ArrayRef<LaunchNodeCoord> lhs,
                             ArrayRef<LaunchNodeCoord> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(lhs, [&](LaunchNodeCoord lhsNode) {
           return llvm::is_contained(rhs, lhsNode);
         });
}

static bool fabricRoutesEqual(const FabricRoute &lhs, const FabricRoute &rhs) {
  return lhs.localDevice == rhs.localDevice &&
         lhs.remoteDevice == rhs.remoteDevice &&
         sourceNodesEqual(lhs.sourceNodes, rhs.sourceNodes);
}

struct FabricManagerExecutionLocation {
  DeviceRefAttr device;
  LaunchNodeCoord node;

  bool operator==(const FabricManagerExecutionLocation &other) const {
    return device == other.device && node == other.node;
  }
};

static SmallVector<FabricManagerExecutionLocation>
getIntervalExecutionLocations(const FabricManagerIntervalPlan &interval,
                              const PipeGraph &pipeGraph) {
  SmallVector<FabricManagerExecutionLocation> locations;
  for (PipeTransferNodeId transferNodeId : interval.transferNodes) {
    const PipeTransferNode &transferNode =
        pipeGraph.getPipeTransferNode(transferNodeId);
    assert(transferNode.deviceTransfer &&
           "fabric manager transfer must cross devices");
    if (interval.kind == FabricManagerIntervalKind::GeneratedSender) {
      if (llvm::is_contained(interval.protocolOperations,
                             transferNode.sendOp)) {
        locations.push_back({transferNode.deviceTransfer.getEdge().getSource(),
                             {transferNode.pipe.srcX, transferNode.pipe.srcY}});
      }
      continue;
    }
    assert(interval.kind == FabricManagerIntervalKind::GeneratedReceiver &&
           "only pure generated intervals can be serialized");
    for (PipeReceiverEndpointId endpointId : transferNode.receiverEndpoints) {
      const PipeReceiverEndpoint &endpoint =
          pipeGraph.getPipeReceiverEndpoint(endpointId);
      if (llvm::is_contained(interval.protocolOperations, endpoint.postOp)) {
        locations.push_back(
            {transferNode.deviceTransfer.getEdge().getDestination(),
             {endpoint.receiver.x, endpoint.receiver.y}});
      }
    }
  }
  return locations;
}

static bool
locationsAreUnique(ArrayRef<FabricManagerExecutionLocation> locations) {
  return llvm::all_of(llvm::enumerate(locations), [&](auto indexedLocation) {
    return !llvm::is_contained(locations.take_front(indexedLocation.index()),
                               indexedLocation.value());
  });
}

static bool
executionLocationsEqual(ArrayRef<FabricManagerExecutionLocation> lhs,
                        ArrayRef<FabricManagerExecutionLocation> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(lhs, [&](FabricManagerExecutionLocation lhsLocation) {
           return llvm::is_contained(rhs, lhsLocation);
         });
}

static std::optional<std::uint64_t> getIntervalInvocationUpperBound(
    const FabricRuntimeIntervalPlan &interval,
    const llvm::SmallPtrSetImpl<Operation *> &generatedControlOps) {
  if (interval.protocolOperations.size() != 1) {
    return std::nullopt;
  }
  std::uint64_t invocationUpperBound = 1;
  for (Operation *parent = interval.protocolOperations.front()->getParentOp();
       parent && !isa<FuncOp>(parent); parent = parent->getParentOp()) {
    if (generatedControlOps.contains(parent) || isa<scf::IfOp>(parent)) {
      continue;
    }
    if (auto loop = dyn_cast<LoopLikeOpInterface>(parent)) {
      std::optional<std::uint64_t> tripCount = getLoopTripCount(loop);
      if (!tripCount) {
        return std::nullopt;
      }
      std::optional<std::uint64_t> product =
          llvm::checkedMulUnsigned(invocationUpperBound, *tripCount);
      if (!product) {
        return std::nullopt;
      }
      invocationUpperBound = *product;
      continue;
    }
    if (parent->getNumRegions() != 0) {
      return std::nullopt;
    }
  }
  return invocationUpperBound;
}

static std::optional<bool> getInvocationCounterRequirement(
    ArrayRef<std::size_t> receiverRuntimeIntervals,
    ArrayRef<std::size_t> senderRuntimeIntervals, const FabricRoutePlan &plan,
    const llvm::SmallPtrSetImpl<Operation *> &generatedControlOps) {
  assert(receiverRuntimeIntervals.size() == senderRuntimeIntervals.size() &&
         "paired manager functions must have equal interval counts");
  std::uint64_t totalInvocationUpperBound = 0;
  // A runtime ordinal preserves the generation sequence when a conditional
  // skips one interval. Constants are sufficient only for one single-shot
  // interval.
  bool requiresInvocationCounter = receiverRuntimeIntervals.size() > 1;
  for (auto [receiverRuntimeIndex, senderRuntimeIndex] :
       llvm::zip_equal(receiverRuntimeIntervals, senderRuntimeIntervals)) {
    std::optional<std::uint64_t> receiverUpperBound =
        getIntervalInvocationUpperBound(
            plan.runtimeIntervals[receiverRuntimeIndex], generatedControlOps);
    std::optional<std::uint64_t> senderUpperBound =
        getIntervalInvocationUpperBound(
            plan.runtimeIntervals[senderRuntimeIndex], generatedControlOps);
    if (!receiverUpperBound || receiverUpperBound != senderUpperBound) {
      return std::nullopt;
    }
    std::optional<std::uint64_t> newTotal = llvm::checkedAddUnsigned(
        totalInvocationUpperBound, *receiverUpperBound);
    if (!newTotal) {
      return std::nullopt;
    }
    totalInvocationUpperBound = *newTotal;
    requiresInvocationCounter |= *receiverUpperBound > 1;
  }

  // Each invocation consumes two monotonically increasing generations. Leave
  // room for the sender's final release generation.
  constexpr std::uint64_t maxInvocationCount =
      (std::numeric_limits<std::uint32_t>::max() - 2) / 2;
  if (totalInvocationUpperBound > maxInvocationCount) {
    return std::nullopt;
  }
  return requiresInvocationCounter;
}

static SmallVector<const FabricRoute *>
getIntervalRoutes(const FabricManagerIntervalPlan &interval,
                  const FunctionFabricRoutePlan &functionPlan) {
  SmallVector<const FabricRoute *> routes;
  for (const FabricRoute &route : functionPlan.routes) {
    if (llvm::is_contained(interval.routeIndices, route.routeIndex)) {
      routes.push_back(&route);
    }
  }
  return routes;
}

static bool intervalRoutesEqual(const FabricManagerIntervalPlan &lhs,
                                const FabricManagerIntervalPlan &rhs,
                                const FabricRoutePlan &plan) {
  auto lhsPlan = plan.routesByFunction.find(lhs.function);
  auto rhsPlan = plan.routesByFunction.find(rhs.function);
  assert(lhsPlan != plan.routesByFunction.end() &&
         rhsPlan != plan.routesByFunction.end() &&
         "generated manager interval is missing its route plan");
  SmallVector<const FabricRoute *> lhsRoutes =
      getIntervalRoutes(lhs, lhsPlan->second);
  SmallVector<const FabricRoute *> rhsRoutes =
      getIntervalRoutes(rhs, rhsPlan->second);
  return lhsRoutes.size() == rhsRoutes.size() &&
         llvm::all_of(lhsRoutes, [&](const FabricRoute *lhsRoute) {
           return llvm::any_of(rhsRoutes, [&](const FabricRoute *rhsRoute) {
             return fabricRoutesEqual(*lhsRoute, *rhsRoute);
           });
         });
}

static bool
externalManagerIntervalsAreSequential(const FabricManagerIntervalPlan &lhs,
                                      const FabricManagerIntervalPlan &rhs) {
  if (lhs.kind != FabricManagerIntervalKind::External ||
      rhs.kind != FabricManagerIntervalKind::External ||
      lhs.function != rhs.function) {
    return false;
  }

  bool lhsIsScoped = lhs.acquireBoundary == lhs.releaseBoundary;
  bool rhsIsScoped = rhs.acquireBoundary == rhs.releaseBoundary;
  bool sameSingleLaunchNode = lhs.launchNodes && rhs.launchNodes &&
                              lhs.launchNodes->size() == 1 &&
                              lhs.launchNodes == rhs.launchNodes;
  // One RISC cannot overlap synchronous scoped calls. Multiple launch nodes
  // may progress independently, so cross-region ordering requires one node.
  if (lhsIsScoped && rhsIsScoped && sameSingleLaunchNode) {
    return true;
  }

  if (lhs.releaseBoundary->getBlock() != rhs.acquireBoundary->getBlock()) {
    return false;
  }
  return lhs.releaseBoundary->isBeforeInBlock(rhs.acquireBoundary);
}

static Operation *getFunctionLevelBoundary(Operation *operation,
                                           FuncOp function) {
  while (operation->getParentOp() != function) {
    operation = operation->getParentOp();
    assert(operation && "fabric runtime boundary must be inside its function");
  }
  return operation;
}

struct FabricRuntimeCoalescingCandidate {
  std::size_t intervalIndex;
  Operation *acquireBoundary;
  Operation *releaseBoundary;
};

static void coalesceFabricRuntimeCandidates(
    ArrayRef<FabricRuntimeCoalescingCandidate> candidates,
    FabricRoutePlan &plan, SmallVectorImpl<bool> &removed) {
  const FabricRuntimeCoalescingCandidate &representative = candidates.front();
  FabricRuntimeIntervalPlan &coalesced =
      plan.runtimeIntervals[representative.intervalIndex];
  Operation *acquireBoundary = representative.acquireBoundary;
  Operation *releaseBoundary = representative.releaseBoundary;
  for (const FabricRuntimeCoalescingCandidate &candidate :
       candidates.drop_front()) {
    if (candidate.acquireBoundary != acquireBoundary &&
        candidate.acquireBoundary->isBeforeInBlock(acquireBoundary)) {
      acquireBoundary = candidate.acquireBoundary;
    }
    if (candidate.releaseBoundary != releaseBoundary &&
        releaseBoundary->isBeforeInBlock(candidate.releaseBoundary)) {
      releaseBoundary = candidate.releaseBoundary;
    }
    FabricRuntimeIntervalPlan &merged =
        plan.runtimeIntervals[candidate.intervalIndex];
    coalesced.managerIntervalIndices.append(merged.managerIntervalIndices);
    coalesced.protocolOperations.append(merged.protocolOperations);
    removed[candidate.intervalIndex] = true;
  }
  coalesced.acquireBoundary = acquireBoundary;
  coalesced.releaseBoundary = releaseBoundary;
}

static void coalesceUnserializedFabricRuntimeIntervals(FabricRoutePlan &plan) {
  // Each function has one host-specialized connection record set. Unserialized
  // intervals in one block must share a manager or they reopen that set.
  llvm::MapVector<Block *, SmallVector<FabricRuntimeCoalescingCandidate>>
      candidatesByBlock;
  for (auto [intervalIndex, interval] :
       llvm::enumerate(plan.runtimeIntervals)) {
    if (interval.ownershipSemaphoreIndex) {
      continue;
    }
    assert(interval.managerIntervalIndices.size() == 1 &&
           "runtime intervals must be coalesced once");
    FuncOp function = interval.acquireBoundary->getParentOfType<FuncOp>();
    assert(function && "fabric runtime interval must be inside a function");
    Operation *acquireBoundary =
        getFunctionLevelBoundary(interval.acquireBoundary, function);
    Operation *releaseBoundary =
        getFunctionLevelBoundary(interval.releaseBoundary, function);
    if (acquireBoundary->getBlock() != releaseBoundary->getBlock()) {
      continue;
    }
    candidatesByBlock[acquireBoundary->getBlock()].push_back(
        {intervalIndex, acquireBoundary, releaseBoundary});
  }

  SmallVector<bool> removed(plan.runtimeIntervals.size());
  for (const auto &entry : candidatesByBlock) {
    ArrayRef<FabricRuntimeCoalescingCandidate> candidates = entry.second;
    if (candidates.size() > 1) {
      coalesceFabricRuntimeCandidates(candidates, plan, removed);
    }
  }

  SmallVector<FabricRuntimeIntervalPlan> coalescedIntervals;
  coalescedIntervals.reserve(plan.runtimeIntervals.size());
  for (std::size_t intervalIndex = 0;
       intervalIndex < plan.runtimeIntervals.size(); ++intervalIndex) {
    if (removed[intervalIndex]) {
      continue;
    }
    coalescedIntervals.push_back(
        std::move(plan.runtimeIntervals[intervalIndex]));
  }
  plan.runtimeIntervals = std::move(coalescedIntervals);
}

static void planFabricManagerOwnership(
    FabricRoutePlan &plan, const PipeGraph &pipeGraph,
    const llvm::SmallPtrSetImpl<Operation *> &generatedControlOps,
    bool enableLocalOwnership) {
  llvm::MapVector<FuncOp, SmallVector<std::size_t>> intervalsByFunction;
  for (auto [runtimeIntervalIndex, runtimeInterval] :
       llvm::enumerate(plan.runtimeIntervals)) {
    assert(runtimeInterval.managerIntervalIndices.size() == 1 &&
           "runtime intervals must be coalesced after ownership planning");
    const FabricManagerIntervalPlan &managerInterval =
        plan.managerIntervals[runtimeInterval.managerIntervalIndices.front()];
    intervalsByFunction[managerInterval.function].push_back(
        runtimeIntervalIndex);
  }

  SmallVector<std::optional<std::size_t>> ownershipGroupByManager(
      plan.managerIntervals.size());
  llvm::SmallPtrSet<Operation *, 4> pairedSenderFunctions;
  if (enableLocalOwnership) {
    for (const auto &[receiverFunction, receiverRuntimeIntervals] :
         intervalsByFunction) {
      if (receiverRuntimeIntervals.empty() ||
          llvm::any_of(receiverRuntimeIntervals,
                       [&](std::size_t runtimeIntervalIndex) {
                         return plan.managerIntervals
                                    [plan.runtimeIntervals[runtimeIntervalIndex]
                                         .managerIntervalIndices.front()]
                                        .kind !=
                                FabricManagerIntervalKind::GeneratedReceiver;
                       })) {
        continue;
      }

      SmallVector<FuncOp> candidateSenderFunctions;
      llvm::DenseMap<Operation *, bool> invocationCounterRequirements;
      for (const auto &[senderFunction, senderRuntimeIntervals] :
           intervalsByFunction) {
        if (pairedSenderFunctions.contains(senderFunction) ||
            receiverFunction == senderFunction ||
            senderRuntimeIntervals.size() != receiverRuntimeIntervals.size()) {
          continue;
        }
        bool matches = true;
        for (auto [receiverRuntimeIndex, senderRuntimeIndex] : llvm::zip_equal(
                 receiverRuntimeIntervals, senderRuntimeIntervals)) {
          const FabricRuntimeIntervalPlan &receiverRuntime =
              plan.runtimeIntervals[receiverRuntimeIndex];
          const FabricRuntimeIntervalPlan &senderRuntime =
              plan.runtimeIntervals[senderRuntimeIndex];
          const FabricManagerIntervalPlan &receiverInterval =
              plan.managerIntervals[receiverRuntime.managerIntervalIndices
                                        .front()];
          const FabricManagerIntervalPlan &senderInterval =
              plan.managerIntervals[senderRuntime.managerIntervalIndices
                                        .front()];
          SmallVector<FabricManagerExecutionLocation> receiverLocations =
              getIntervalExecutionLocations(receiverInterval, pipeGraph);
          SmallVector<FabricManagerExecutionLocation> senderLocations =
              getIntervalExecutionLocations(senderInterval, pipeGraph);
          if (senderInterval.kind !=
                  FabricManagerIntervalKind::GeneratedSender ||
              receiverInterval.transferNodes != senderInterval.transferNodes ||
              !locationsAreUnique(receiverLocations) ||
              !locationsAreUnique(senderLocations) ||
              !executionLocationsEqual(receiverLocations, senderLocations) ||
              !intervalRoutesEqual(receiverInterval, senderInterval, plan)) {
            matches = false;
            break;
          }
        }
        std::optional<bool> invocationCounterRequirement;
        if (matches) {
          invocationCounterRequirement = getInvocationCounterRequirement(
              receiverRuntimeIntervals, senderRuntimeIntervals, plan,
              generatedControlOps);
          matches = invocationCounterRequirement.has_value();
        }
        if (matches) {
          candidateSenderFunctions.push_back(senderFunction);
          FuncOp candidateSenderFunction = senderFunction;
          invocationCounterRequirements[candidateSenderFunction
                                            .getOperation()] =
              *invocationCounterRequirement;
        }
      }
      if (candidateSenderFunctions.size() != 1) {
        continue;
      }

      FuncOp senderFunction = candidateSenderFunctions.front();
      pairedSenderFunctions.insert(senderFunction);
      ArrayRef<std::size_t> senderRuntimeIntervals =
          intervalsByFunction.find(senderFunction)->second;
      bool useInvocationCounter =
          invocationCounterRequirements.lookup(senderFunction.getOperation());
      std::size_t semaphoreIndex = plan.ownershipSemaphoreCount++;
      for (std::size_t intervalPosition = 0;
           intervalPosition < receiverRuntimeIntervals.size();
           ++intervalPosition) {
        std::size_t receiverRuntimeIndex =
            receiverRuntimeIntervals[intervalPosition];
        std::size_t senderRuntimeIndex =
            senderRuntimeIntervals[intervalPosition];
        FabricRuntimeIntervalPlan &receiverRuntime =
            plan.runtimeIntervals[receiverRuntimeIndex];
        FabricRuntimeIntervalPlan &senderRuntime =
            plan.runtimeIntervals[senderRuntimeIndex];
        receiverRuntime.acquireBoundary =
            receiverRuntime.protocolOperations.front();
        receiverRuntime.releaseBoundary =
            receiverRuntime.protocolOperations.front();
        senderRuntime.acquireBoundary =
            senderRuntime.protocolOperations.front();
        senderRuntime.releaseBoundary =
            senderRuntime.protocolOperations.front();
        FabricManagerIntervalPlan &receiverManager =
            plan.managerIntervals[receiverRuntime.managerIntervalIndices
                                      .front()];
        FabricManagerIntervalPlan &senderManager =
            plan.managerIntervals[senderRuntime.managerIntervalIndices.front()];
        receiverManager.acquireBoundary = receiverRuntime.acquireBoundary;
        receiverManager.releaseBoundary = receiverRuntime.releaseBoundary;
        senderManager.acquireBoundary = senderRuntime.acquireBoundary;
        senderManager.releaseBoundary = senderRuntime.releaseBoundary;
        receiverRuntime.ownershipSemaphoreIndex = semaphoreIndex;
        receiverRuntime.useInvocationCounter = useInvocationCounter;
        receiverRuntime.acquireGeneration =
            useInvocationCounter ? 0 : 2 * intervalPosition;
        receiverRuntime.releaseGeneration =
            useInvocationCounter ? 1 : 2 * intervalPosition + 1;
        senderRuntime.ownershipSemaphoreIndex = semaphoreIndex;
        senderRuntime.useInvocationCounter = useInvocationCounter;
        senderRuntime.acquireGeneration =
            useInvocationCounter ? 1 : 2 * intervalPosition + 1;
        senderRuntime.releaseGeneration =
            useInvocationCounter ? 2 : 2 * intervalPosition + 2;
        ownershipGroupByManager[receiverRuntime.managerIntervalIndices
                                    .front()] = semaphoreIndex;
        ownershipGroupByManager[senderRuntime.managerIntervalIndices.front()] =
            semaphoreIndex;
      }
    }
  }

  for (std::size_t lhsIndex = 0; lhsIndex < plan.managerIntervals.size();
       ++lhsIndex) {
    for (std::size_t rhsIndex = lhsIndex + 1;
         rhsIndex < plan.managerIntervals.size(); ++rhsIndex) {
      if (ownershipGroupByManager[lhsIndex] &&
          ownershipGroupByManager[lhsIndex] ==
              ownershipGroupByManager[rhsIndex]) {
        continue;
      }
      if (externalManagerIntervalsAreSequential(
              plan.managerIntervals[lhsIndex],
              plan.managerIntervals[rhsIndex]) ||
          externalManagerIntervalsAreSequential(
              plan.managerIntervals[rhsIndex],
              plan.managerIntervals[lhsIndex])) {
        continue;
      }
      plan.managerIntervals[lhsIndex].interferingIntervals.push_back(rhsIndex);
      plan.managerIntervals[rhsIndex].interferingIntervals.push_back(lhsIndex);
    }
  }
  coalesceUnserializedFabricRuntimeIntervals(plan);
}

LogicalResult buildFabricRoutePlan(
    ModuleOp module, const PipeTransferIndex &transferIndex,
    const PipeGraph &pipeGraph, const PipeForeachLoweringInfo &foreachInfo,
    ArrayRef<ExternalFabricManagerInterval> externalManagerIntervals,
    bool enableLocalManagerOwnership, FabricRoutePlan &plan) {
  LogicalResult result = success();
  RecordAlignedTableBuilder<std::size_t> routeIndices;

  auto recordRouteIndex = [&](Operation *operation,
                              std::optional<std::uint64_t> recordIndex,
                              std::size_t routeIndex) -> LogicalResult {
    FailureOr<PipeReference> pipeReference =
        getPipeReferenceForProtocolOp(operation, transferIndex);
    if (failed(pipeReference)) {
      return failure();
    }
    assert(pipeReference->isSelected() == recordIndex.has_value() &&
           "fabric graph record identity must match its pipe reference");
    std::size_t recordCount =
        pipeReference->isSelected()
            ? pipeReference->getRecords().getPipes().size()
            : 1;
    std::size_t selectedIndex = recordIndex.value_or(0);
    return routeIndices.set(
        operation, recordCount, selectedIndex, routeIndex,
        [](Operation *operation, std::size_t existingRouteIndex,
           std::size_t newRouteIndex) {
          if (existingRouteIndex == newRouteIndex) {
            return success();
          }
          operation->emitError(
              "one pipe record requires two fabric connection indices");
          return failure();
        });
  };

  for (const PipeTransferNode &transferNode :
       pipeGraph.getPipeTransferNodes()) {
    DeviceTransferAttr transfer = transferNode.deviceTransfer;
    if (!transfer) {
      continue;
    }
    auto send = mlir::cast<PipeTransferSendOp>(transferNode.sendOp);
    DeviceRefAttr destination = transfer.getEdge().getDestination();
    if (!destination) {
      send.emitError(
          "device-range fabric transfers require scatter target lowering");
      result = failure();
      continue;
    }

    DeviceRefAttr source = transfer.getEdge().getSource();
    FuncOp sendFunc = send->getParentOfType<FuncOp>();
    FailureOr<FunctionFabricRoutePlan *> maybeSendFunctionPlan =
        getFunctionFabricRoutePlan(sendFunc, transfer.getDomain(), send, plan);
    if (failed(maybeSendFunctionPlan)) {
      result = failure();
      continue;
    }
    std::size_t routeIndex = addFabricRoute(
        (*maybeSendFunctionPlan)->routes, source, destination,
        LaunchNodeCoord{transferNode.pipe.srcX, transferNode.pipe.srcY});
    if (failed(
            recordRouteIndex(send, transferNode.sendRecordIndex, routeIndex))) {
      result = failure();
      continue;
    }

    for (PipeReceiverEndpointId endpointId : transferNode.receiverEndpoints) {
      const PipeReceiverEndpoint &endpoint =
          pipeGraph.getPipeReceiverEndpoint(endpointId);
      Operation *postOp = endpoint.postOp;
      FuncOp postFunc = postOp->getParentOfType<FuncOp>();
      FailureOr<FunctionFabricRoutePlan *> maybePostFunctionPlan =
          getFunctionFabricRoutePlan(postFunc, transfer.getDomain(), postOp,
                                     plan);
      if (failed(maybePostFunctionPlan)) {
        result = failure();
        continue;
      }
      std::size_t reverseRouteIndex = addFabricRoute(
          (*maybePostFunctionPlan)->routes, destination, source,
          LaunchNodeCoord{endpoint.receiver.x, endpoint.receiver.y});
      if (failed(recordRouteIndex(postOp, endpoint.postRecordIndex,
                                  reverseRouteIndex))) {
        result = failure();
        continue;
      }
    }
  }

  plan.routeIndices = std::move(routeIndices).finalize();

  llvm::SmallPtrSet<Operation *, 16> generatedControlOps(
      foreachInfo.controlOps.begin(), foreachInfo.controlOps.end());
  llvm::MapVector<Operation *, SmallVector<Operation *>> operationsByInterval;
  for (const auto &[operation, indices] : plan.routeIndices) {
    if (indices.empty()) {
      continue;
    }

    Operation *scope = operation;
    for (Operation *parent = operation->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (generatedControlOps.contains(parent)) {
        scope = parent;
      }
      if (isa<FuncOp>(parent)) {
        break;
      }
    }
    operationsByInterval[scope].push_back(operation);
  }
  for (auto &[scope, operations] : operationsByInterval) {
    FuncOp function = scope->getParentOfType<FuncOp>();
    assert(function && "fabric interval must be inside a function");
    std::size_t managerIntervalIndex = plan.managerIntervals.size();
    FabricManagerIntervalKind kind = getGeneratedFabricManagerKind(operations);
    StringAttr identity = StringAttr::get(
        module.getContext(),
        (Twine("generated.") + Twine(managerIntervalIndex)).str());
    plan.managerIntervals.push_back(FabricManagerIntervalPlan{
        identity,
        kind,
        function,
        std::nullopt,
        operations,
        getIntervalRouteIndices(operations, plan),
        getIntervalTransferNodes(operations, pipeGraph),
        scope,
        scope,
        {},
        std::nullopt});
    plan.runtimeIntervals.push_back(
        FabricRuntimeIntervalPlan{{managerIntervalIndex},
                                  scope,
                                  scope,
                                  std::move(operations),
                                  std::nullopt,
                                  false,
                                  0,
                                  0});
  }

  for (ExternalFabricManagerInterval externalInterval :
       externalManagerIntervals) {
    LaunchNodeDomain launchDomain =
        pipeGraph.getOperationLaunchDomain(externalInterval.acquire);
    if (!launchDomain.known) {
      externalInterval.acquire.emitError(
          "cannot prove the exact launch-node domain for external fabric "
          "manager claim '")
          << externalInterval.claim.getValue() << "'";
      result = failure();
      continue;
    }
    StringAttr identity = StringAttr::get(
        module.getContext(),
        (Twine("external.") + externalInterval.claim.getValue()).str());
    std::optional<SmallVector<LaunchNodeCoord>> launchNodes;
    if (externalInterval.acquire->getBlock() !=
        &externalInterval.function.getBody().front()) {
      launchNodes.emplace(launchDomain.nodes.begin(), launchDomain.nodes.end());
    }
    plan.managerIntervals.push_back(
        FabricManagerIntervalPlan{identity,
                                  FabricManagerIntervalKind::External,
                                  externalInterval.function,
                                  externalInterval.claim,
                                  {},
                                  {},
                                  {},
                                  externalInterval.acquire,
                                  externalInterval.release,
                                  {},
                                  std::move(launchNodes)});
  }

  planFabricManagerOwnership(plan, pipeGraph, generatedControlOps,
                             enableLocalManagerOwnership);
  return result;
}

void applyFabricRoutePlan(ModuleOp mod, const FabricRoutePlan &plan) {
  Builder builder(mod.getContext());
  for (const auto &[func, functionPlan] : plan.routesByFunction) {
    SmallVector<Attribute> routeAttrs;
    routeAttrs.reserve(functionPlan.routes.size());
    for (const FabricRoute &route : functionPlan.routes) {
      SmallVector<Attribute> sourceNodes;
      sourceNodes.reserve(route.sourceNodes.size());
      for (LaunchNodeCoord sourceNode : route.sourceNodes) {
        sourceNodes.push_back(
            builder.getDenseI64ArrayAttr({sourceNode.x, sourceNode.y}));
      }
      routeAttrs.push_back(DictionaryAttr::get(
          mod.getContext(),
          {builder.getNamedAttr("local", route.localDevice),
           builder.getNamedAttr("remote", route.remoteDevice),
           builder.getNamedAttr("route_index",
                                builder.getI64IntegerAttr(route.routeIndex)),
           builder.getNamedAttr("source_nodes",
                                builder.getArrayAttr(sourceNodes))}));
    }
    func->setAttr(kFabricRoutesAttrName,
                  ArrayAttr::get(mod.getContext(), routeAttrs));
    func->setAttr(kFabricDeviceDomainAttrName, functionPlan.deviceDomain);
    func->setAttr(
        kFabricRuntimeArgBaseCommonIndexAttrName,
        builder.getI64IntegerAttr(
            CommonRuntimeArgLayout(func).getFabricRuntimeArgBaseIndex()));
  }

  llvm::MapVector<FuncOp, SmallVector<Attribute>> intervalsByFunction;
  for (const FabricManagerIntervalPlan &interval : plan.managerIntervals) {
    SmallVector<StringAttr> interferingIntervals;
    interferingIntervals.reserve(interval.interferingIntervals.size());
    for (std::size_t interferingIndex : interval.interferingIntervals) {
      assert(interferingIndex < plan.managerIntervals.size() &&
             "fabric interference index out of range");
      interferingIntervals.push_back(
          plan.managerIntervals[interferingIndex].identity);
    }
    SmallVector<int64_t> routeIndices(interval.routeIndices.begin(),
                                      interval.routeIndices.end());
    DenseI64ArrayAttr launchNodes;
    if (interval.launchNodes) {
      SmallVector<int64_t> nodeCoordinates;
      nodeCoordinates.reserve(2 * interval.launchNodes->size());
      for (LaunchNodeCoord node : *interval.launchNodes) {
        nodeCoordinates.push_back(node.x);
        nodeCoordinates.push_back(node.y);
      }
      launchNodes = builder.getDenseI64ArrayAttr(nodeCoordinates);
    }
    intervalsByFunction[interval.function].push_back(
        FabricManagerIntervalAttr::get(
            mod.getContext(), interval.identity, interval.kind,
            interval.claim.value_or(StringAttr()),
            builder.getDenseI64ArrayAttr(routeIndices), interferingIntervals,
            launchNodes));
  }
  for (auto &[function, intervals] : intervalsByFunction) {
    function->setAttr(kFabricManagerIntervalsAttrName,
                      builder.getArrayAttr(intervals));
  }
}

void initializeFabricRuntime(const FabricRoutePlan &plan,
                             FabricRuntimeMap &runtime) {
  llvm::DenseMap<std::pair<Operation *, int64_t>, Value>
      ownershipInvocationCounters;
  for (const FabricRuntimeIntervalPlan &interval : plan.runtimeIntervals) {
    if (!interval.ownershipSemaphoreIndex || !interval.useInvocationCounter) {
      continue;
    }
    FuncOp func = interval.acquireBoundary->getParentOfType<FuncOp>();
    assert(func && "fabric connection interval must be inside a function");
    auto counterKey =
        std::make_pair(func.getOperation(), *interval.ownershipSemaphoreIndex);
    if (ownershipInvocationCounters.contains(counterKey)) {
      continue;
    }
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto counterType = MemRefType::get({1}, builder.getI32Type());
    Value counter = memref::AllocaOp::create(builder, loc, counterType);
    Value counterIndex = arith::ConstantIndexOp::create(builder, loc, 0);
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    memref::StoreOp::create(builder, loc, zero, counter,
                            ValueRange{counterIndex});
    ownershipInvocationCounters.insert({counterKey, counter});
  }

  for (const FabricRuntimeIntervalPlan &interval : plan.runtimeIntervals) {
    FuncOp func = interval.acquireBoundary->getParentOfType<FuncOp>();
    assert(func && "fabric connection interval must be inside a function");
    auto functionPlanIt = plan.routesByFunction.find(func);
    assert(functionPlanIt != plan.routesByFunction.end() &&
           "fabric connection interval is missing its route plan");
    const SmallVector<FabricRoute> &routes = functionPlanIt->second.routes;
    std::size_t routeCount = getFabricRouteCount(routes);
    OpBuilder builder(interval.acquireBoundary);
    Location loc = interval.acquireBoundary->getLoc();
    Value ownershipSemaphorePtr;
    Value ownershipInvocationCounter;
    Value ownershipInvocation;
    Value ownershipGenerationBase;
    if (interval.ownershipSemaphoreIndex) {
      Value semaphoreIndex = arith::ConstantIndexOp::create(
          builder, loc, *interval.ownershipSemaphoreIndex);
      Value ownershipSemaphore =
          ttk::GetSemaphoreOp::create(builder, loc, semaphoreIndex);
      auto l1PtrType = ttk::L1AddrPtrType::get(builder.getContext(), 32);
      ownershipSemaphorePtr = ttk::CastToL1PtrOp::create(
          builder, loc, l1PtrType, ownershipSemaphore);
      if (interval.useInvocationCounter) {
        auto counterKey = std::make_pair(func.getOperation(),
                                         *interval.ownershipSemaphoreIndex);
        ownershipInvocationCounter =
            ownershipInvocationCounters.lookup(counterKey);
        assert(ownershipInvocationCounter &&
               "repeated fabric ownership is missing its local counter");
        Value counterIndex = arith::ConstantIndexOp::create(builder, loc, 0);
        ownershipInvocation = memref::LoadOp::create(
            builder, loc, ownershipInvocationCounter, ValueRange{counterIndex});
        Value generationsPerInvocation =
            arith::ConstantIntOp::create(builder, loc, 2, 32);
        ownershipGenerationBase = arith::MulIOp::create(
            builder, loc, ownershipInvocation, generationsPerInvocation);
      }
      Value acquireGeneration = arith::ConstantIntOp::create(
          builder, loc, interval.acquireGeneration, 32);
      if (ownershipGenerationBase) {
        acquireGeneration = arith::AddIOp::create(
            builder, loc, ownershipGenerationBase, acquireGeneration);
      }
      ttk::SemaphoreWaitMinOp::create(builder, loc, ownershipSemaphorePtr,
                                      acquireGeneration);
    }
    Value runtimeArgBaseCommonIndex = arith::ConstantIndexOp::create(
        builder, loc,
        CommonRuntimeArgLayout(func).getFabricRuntimeArgBaseIndex());
    Value runtimeArgBaseI32 = ttk::GetCommonArgValOp::create(
        builder, loc, builder.getI32Type(), runtimeArgBaseCommonIndex);
    Value runtimeArgBase = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), runtimeArgBaseI32);
    Value connectionCount = ttk::GetArgValOp::create(
        builder, loc, builder.getI32Type(), runtimeArgBase);
    Value manager = ttk::CreateRoutingPlaneConnectionManagerOp::create(
        builder, loc,
        ttk::RoutingPlaneConnectionManagerType::get(builder.getContext()));
    Value connectionRecordsOffset =
        arith::ConstantIndexOp::create(builder, loc, 1 + 4 * routeCount);
    Value connectionRecordsBase = arith::AddIOp::create(
        builder, loc, runtimeArgBase, connectionRecordsOffset);
    Value routeId = ttk::OpenRoutingPlaneConnectionsOp::create(
        builder, loc, builder.getI32Type(), manager, connectionCount,
        connectionRecordsBase);
    FabricRuntimeInfo runtimeInfo{manager, routeId, connectionCount,
                                  runtimeArgBase, routeCount};
    for (Operation *operation : interval.protocolOperations) {
      auto [runtimeIt, inserted] = runtime.try_emplace(operation, runtimeInfo);
      (void)runtimeIt;
      assert(inserted &&
             "fabric protocol operation has multiple connection intervals");
    }

    builder.setInsertionPointAfter(interval.releaseBoundary);
    ttk::CloseRoutingPlaneConnectionsOp close =
        ttk::CloseRoutingPlaneConnectionsOp::create(builder, loc, manager,
                                                    connectionCount);
    if (ownershipSemaphorePtr) {
      builder.setInsertionPointAfter(close);
      Value releaseGeneration = arith::ConstantIntOp::create(
          builder, loc, interval.releaseGeneration, 32);
      if (ownershipGenerationBase) {
        releaseGeneration = arith::AddIOp::create(
            builder, loc, ownershipGenerationBase, releaseGeneration);
      }
      ttk::NocSemaphoreSetOp::create(builder, loc, ownershipSemaphorePtr,
                                     releaseGeneration);
      if (ownershipInvocationCounter) {
        Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
        Value nextInvocation =
            arith::AddIOp::create(builder, loc, ownershipInvocation, one);
        Value counterIndex = arith::ConstantIndexOp::create(builder, loc, 0);
        memref::StoreOp::create(builder, loc, nextInvocation,
                                ownershipInvocationCounter,
                                ValueRange{counterIndex});
      }
    }
  }
}

/// Compiler-managed pipe resources follow tensor buffer addresses and computed
/// receiver DFB bases in the common runtime argument list.
/// [Device 2.0] Keep this as a resource-plan lookup so the final device API
/// lowering can replace common-arg plumbing without changing pipe semantics.
static int64_t getPipeRuntimeCommonArgIndex(FuncOp func,
                                            int64_t pipeRuntimeArgIndex) {
  return CommonRuntimeArgLayout(func).getPipeResourceIndex(pipeRuntimeArgIndex);
}

static int64_t getPipeRuntimeCommonArgIndex(Operation *op,
                                            int64_t pipeRuntimeArgIndex) {
  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "pipe op is not inside a function");
  return getPipeRuntimeCommonArgIndex(func, pipeRuntimeArgIndex);
}

static Value buildPipeRuntimeCommonArg(Location loc, OpBuilder &builder,
                                       int64_t commonArgIndex) {
  auto argIndex = arith::ConstantIndexOp::create(builder, loc, commonArgIndex);
  return ttk::GetCommonArgValOp::create(builder, loc, builder.getI32Type(),
                                        argIndex)
      .getResult();
}

static Value buildPipeRuntimeCommonArg(Location loc, OpBuilder &builder,
                                       Value commonArgIndex) {
  return ttk::GetCommonArgValOp::create(builder, loc, builder.getI32Type(),
                                        commonArgIndex)
      .getResult();
}

static Value buildLocalSemaphoreAddress(Location loc, OpBuilder &builder,
                                        int64_t semaphoreIndex) {
  Value semaphoreIndexValue =
      arith::ConstantIndexOp::create(builder, loc, semaphoreIndex);
  return ttk::GetSemaphoreOp::create(builder, loc, semaphoreIndexValue)
      .getResult();
}

/// Return the first pipe-resource runtime arg index used for GlobalSemaphore
/// counter addresses.
static int64_t
getFirstPipeGlobalSemaphoreArgOffset(const PipeResourcePlan &info) {
  // GlobalSemaphore addresses follow the optional SRAM scratch base in the
  // common runtime args built by python/ttl/kernel_runner.py.
  return info.sramScratch.bytes > 0 ? 1 : 0;
}

/// Build the L1 address for any compiler-managed PipeNet counter.
static Value buildPipeCounterAddress(Location loc, FuncOp func,
                                     PipeCounterInfo counter,
                                     const PipeResourcePlan &pipeResourcePlan,
                                     OpBuilder &builder) {
  // [Device 2.0] This should become a typed semaphore-object lookup when the
  // device API exposes Semaphore/GlobalSemaphore objects directly.
  switch (counter.getStorage()) {
  case PipeCounterStorage::LocalSemaphore:
    return buildLocalSemaphoreAddress(loc, builder, counter.getIndex());
  case PipeCounterStorage::GlobalSemaphore: {
    int64_t argIndex = getPipeRuntimeCommonArgIndex(
        func, getFirstPipeGlobalSemaphoreArgOffset(pipeResourcePlan) +
                  counter.getIndex());
    return buildPipeRuntimeCommonArg(loc, builder, argIndex);
  }
  }
  llvm_unreachable("unknown pipe counter storage");
}

static Value buildPipeCounterPtr(Location loc, FuncOp func,
                                 PipeCounterInfo counter,
                                 const PipeResourcePlan &pipeResourcePlan,
                                 OpBuilder &builder) {
  auto l1PtrTy = ttk::L1AddrPtrType::get(builder.getContext(), 32);
  Value address =
      buildPipeCounterAddress(loc, func, counter, pipeResourcePlan, builder);
  return ttk::CastToL1PtrOp::create(builder, loc, l1PtrTy, address).getResult();
}

static Value loadIndexTableEntry(Location loc, ArrayRef<int64_t> values,
                                 Value recordIndex, OpBuilder &builder) {
  return buildConstantIndexTableLookup(builder, loc, values, recordIndex);
}

static Value buildSelectedRouteIndex(Location loc,
                                     ArrayRef<std::size_t> routeIndices,
                                     Value recordIndex,
                                     ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t> indexTable(routeIndices.begin(), routeIndices.end());
  return loadIndexTableEntry(loc, indexTable, recordIndex, rewriter);
}

static Value buildSelectedPipeCounterAddress(
    Operation *op, Location loc, ArrayRef<PipeCounterInfo> counters,
    Value recordIndex, const PipeResourcePlan &pipeResourcePlan,
    OpBuilder &builder) {
  assert(!counters.empty() && "selected pipe counter table is empty");

  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "selected pipe operation must be inside a function");
  bool hasLocalCounter = llvm::any_of(counters, [](PipeCounterInfo counter) {
    return counter.getStorage() == PipeCounterStorage::LocalSemaphore;
  });
  bool hasGlobalCounter = llvm::any_of(counters, [](PipeCounterInfo counter) {
    return counter.getStorage() == PipeCounterStorage::GlobalSemaphore;
  });
  if (!hasGlobalCounter) {
    SmallVector<int64_t> localIndices = llvm::map_to_vector(
        counters, [](PipeCounterInfo counter) { return counter.getIndex(); });
    Value semaphoreIndex =
        loadIndexTableEntry(loc, localIndices, recordIndex, builder);
    return ttk::GetSemaphoreOp::create(builder, loc, semaphoreIndex)
        .getResult();
  }
  auto getGlobalArgIndex = [&](PipeCounterInfo counter) {
    return getPipeRuntimeCommonArgIndex(
        func, getFirstPipeGlobalSemaphoreArgOffset(pipeResourcePlan) +
                  counter.getIndex());
  };
  if (!hasLocalCounter) {
    SmallVector<int64_t> globalArgIndices =
        llvm::map_to_vector(counters, getGlobalArgIndex);
    Value commonArgIndex =
        loadIndexTableEntry(loc, globalArgIndices, recordIndex, builder);
    return ttk::GetCommonArgValOp::create(builder, loc, builder.getI32Type(),
                                          commonArgIndex)
        .getResult();
  }

  PipeCounterInfo validLocalCounter =
      *llvm::find_if(counters, [](PipeCounterInfo counter) {
        return counter.getStorage() == PipeCounterStorage::LocalSemaphore;
      });
  PipeCounterInfo validGlobalCounter =
      *llvm::find_if(counters, [](PipeCounterInfo counter) {
        return counter.getStorage() == PipeCounterStorage::GlobalSemaphore;
      });
  SmallVector<int64_t> isGlobal;
  SmallVector<int64_t> localIndices;
  SmallVector<int64_t> globalArgIndices;
  for (PipeCounterInfo counter : counters) {
    bool usesGlobal =
        counter.getStorage() == PipeCounterStorage::GlobalSemaphore;
    isGlobal.push_back(usesGlobal ? 1 : 0);
    localIndices.push_back(
        (usesGlobal ? validLocalCounter : counter).getIndex());
    globalArgIndices.push_back(
        getGlobalArgIndex(usesGlobal ? counter : validGlobalCounter));
  }

  // arith.select cannot prevent either address operation from executing. Use
  // an existing index in the unused storage class so both addresses are valid.
  Value localIndex =
      loadIndexTableEntry(loc, localIndices, recordIndex, builder);
  Value localAddress =
      ttk::GetSemaphoreOp::create(builder, loc, localIndex).getResult();
  Value typedLocalAddress =
      ttk::CastToL1AddrOp::create(builder, loc, localAddress);
  Value globalArgIndex =
      loadIndexTableEntry(loc, globalArgIndices, recordIndex, builder);
  Value globalAddress = ttk::GetCommonArgValOp::create(
                            builder, loc, builder.getI32Type(), globalArgIndex)
                            .getResult();
  Value typedGlobalAddress =
      ttk::CastToL1AddrOp::create(builder, loc, globalAddress);
  Value storageKind = loadIndexTableEntry(loc, isGlobal, recordIndex, builder);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value usesGlobal = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, storageKind, zero);
  return arith::SelectOp::create(builder, loc, usesGlobal, typedGlobalAddress,
                                 typedLocalAddress);
}

static Value buildSelectedReadyCounterAddress(
    Operation *op, Location loc, ArrayRef<PipeResourceInfo> resources,
    Value recordIndex, const PipeResourcePlan &pipeResourcePlan,
    ConversionPatternRewriter &rewriter) {
  SmallVector<PipeCounterInfo> counters;
  counters.reserve(resources.size());
  for (const PipeResourceInfo &resource : resources) {
    assert(resource.readyCounter &&
           "selected pipe missing sender-ready counter");
    counters.push_back(*resource.readyCounter);
  }
  return buildSelectedPipeCounterAddress(op, loc, counters, recordIndex,
                                         pipeResourcePlan, rewriter);
}

/// Add a static byte offset to an L1 address without changing the address
/// representation.
static Value addByteOffset(Location loc, Value baseAddress, int64_t byteOffset,
                           ConversionPatternRewriter &rewriter) {
  if (byteOffset == 0) {
    return baseAddress;
  }
  auto offsetValue =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                rewriter.getI32IntegerAttr(byteOffset));
  return arith::AddIOp::create(rewriter, loc, baseAddress, offsetValue)
      .getResult();
}

static Value addByteOffset(Location loc, Value baseAddress, Value byteOffset,
                           ConversionPatternRewriter &rewriter) {
  Value byteOffsetI32 = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(), byteOffset);
  return arith::AddIOp::create(rewriter, loc, baseAddress, byteOffsetI32)
      .getResult();
}

Value buildPipeSramScratchAddress(Operation *operation, int64_t byteOffset,
                                  OpBuilder &builder) {
  int64_t scratchArgIndex = getPipeRuntimeCommonArgIndex(operation, 0);
  Value scratchBase =
      buildPipeRuntimeCommonArg(operation->getLoc(), builder, scratchArgIndex);
  if (byteOffset == 0) {
    return scratchBase;
  }
  auto offsetValue = arith::ConstantOp::create(
      builder, operation->getLoc(), builder.getI32Type(),
      builder.getI32IntegerAttr(byteOffset));
  return arith::AddIOp::create(builder, operation->getLoc(), scratchBase,
                               offsetValue)
      .getResult();
}

/// Source-node address-table entry selected for one transfer allocation unit.
/// The common arg contains the host-allocated SRAM scratch buffer address;
/// byteOffset selects this transfer's 32-bit receiver-published address slot.
struct AddressTableInfo {
  int64_t scratchRuntimeCommonArgIndex;
  int64_t byteOffset = 0;
};

/// Record the scratch common-arg index with the per-transfer SRAM offset from
/// the resource plan.
static AddressTableInfo
getAddressTableInfo(Operation *op, const PipeResourceInfo &pipeResource) {
  assert(pipeResource.addressStorage.mode ==
             PipeAddressMode::ReceiverPublishedAddressTable &&
         "address-table info requested for computed-address pipe");
  assert(pipeResource.addressStorage.sramAddressTable.has_value() &&
         "receiver-published-address pipe missing address-table storage");
  int64_t scratchArgIndex = getPipeRuntimeCommonArgIndex(op, 0);
  return AddressTableInfo{
      scratchArgIndex,
      pipeResource.addressStorage.sramAddressTable->byteOffset};
}

/// Build the L1 address of this transfer's source-core address-table slot.
static Value buildAddressTableAddress(Location loc,
                                      const AddressTableInfo &info,
                                      ConversionPatternRewriter &rewriter) {
  Value scratchBase = buildPipeRuntimeCommonArg(
      loc, rewriter, info.scratchRuntimeCommonArgIndex);
  return addByteOffset(loc, scratchBase, info.byteOffset, rewriter);
}

static Value buildSelectedAddressTableAddress(
    Operation *op, Location loc, ArrayRef<PipeResourceInfo> resources,
    Value recordIndex, ConversionPatternRewriter &rewriter) {
  auto publishedResource =
      llvm::find_if(resources, [](const PipeResourceInfo &resource) {
        return resource.addressStorage.mode ==
               PipeAddressMode::ReceiverPublishedAddressTable;
      });
  assert(publishedResource != resources.end() &&
         "selected pipe has no receiver-published address record");
  assert(publishedResource->addressStorage.sramAddressTable &&
         "receiver-published-address pipe missing address-table storage");
  int64_t fallbackByteOffset =
      publishedResource->addressStorage.sramAddressTable->byteOffset;
  SmallVector<int64_t> byteOffsets;
  byteOffsets.reserve(resources.size());
  for (const PipeResourceInfo &resource : resources) {
    byteOffsets.push_back(
        resource.addressStorage.sramAddressTable
            ? resource.addressStorage.sramAddressTable->byteOffset
            : fallbackByteOffset);
  }
  Value byteOffset =
      loadIndexTableEntry(loc, byteOffsets, recordIndex, rewriter);
  Value scratchBase = buildPipeRuntimeCommonArg(
      loc, rewriter, getPipeRuntimeCommonArgIndex(op, 0));
  return addByteOffset(loc, scratchBase, byteOffset, rewriter);
}

/// Load the receiver-published destination DFB address from this pipe's
/// source-core SRAM address-table entry.
static Value
buildAddressTableDestinationAddress(Location loc, Value tableAddress,
                                    ConversionPatternRewriter &rewriter) {
  // [Device 2.0] Address tables are compiler-managed SRAM state; only this
  // final load should depend on raw L1 pointer operations.
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  auto tablePtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, tableAddress);
  auto zeroI32 = arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                           rewriter.getI32IntegerAttr(0));
  return ttk::LoadFromL1Op::create(rewriter, loc, rewriter.getI32Type(),
                                   tablePtr, zeroI32)
      .getResult();
}

static Value
buildAddressTableDestinationAddress(Location loc, const AddressTableInfo &info,
                                    ConversionPatternRewriter &rewriter) {
  return buildAddressTableDestinationAddress(
      loc, buildAddressTableAddress(loc, info, rewriter), rewriter);
}

/// Find the slot counter allocated during resource planning. Missing state is
/// a pass-ordering bug because computed-address sends are planned before
/// conversion patterns mutate the IR.
static Value lookupComputedAddressCounter(
    PipeTransferSendOp op,
    const PipeComputedAddressCounterMap &computedAddressCounters) {
  FuncOp senderFunc = op->getParentOfType<FuncOp>();
  auto funcIt = computedAddressCounters.find(senderFunc);
  assert(funcIt != computedAddressCounters.end() &&
         "sender function missing computed-address counters");
  return funcIt->second;
}

/// Compute the receiver DFB destination address selected for this send. A
/// transfer that executes at most once uses `initialSlot` directly. A transfer
/// that can repeat with a nonzero stride uses a sender-local counter for
/// `slot(i)`.
static Value buildComputedReceiverDFBDestinationAddress(
    PipeTransferSendOp op, Location loc,
    const PipeAddressStorageInfo &addressStorage,
    const PipeComputedAddressCounterMap &computedAddressCounters,
    ConversionPatternRewriter &rewriter) {
  assert(addressStorage.computedAddress.has_value() &&
         "computed pipe missing computed-address info");
  const PipeComputedAddressInfo &info = *addressStorage.computedAddress;
  Value baseAddress =
      addressStorage.usesTransportScratch()
          ? buildPipeSramScratchAddress(op, info.baseByteOffset, rewriter)
          : addByteOffset(loc,
                          buildPipeRuntimeCommonArg(
                              loc, rewriter, info.baseRuntimeCommonArgIndex),
                          info.baseByteOffset, rewriter);
  if (!info.usesDynamicSlotCounter()) {
    int64_t byteOffset =
        info.initialSlot * info.blockStrideBytes + info.staticTileByteOffset;
    return addByteOffset(loc, baseAddress, byteOffset, rewriter);
  }

  Value counterIndex = arith::ConstantIndexOp::create(
      rewriter, loc, *info.dynamicSlotCounterIndex);
  Value slotCounters =
      lookupComputedAddressCounter(op, computedAddressCounters);
  Value currentSlot = memref::LoadOp::create(rewriter, loc, slotCounters,
                                             ValueRange{counterIndex});
  Value blockStrideBytes =
      arith::ConstantIntOp::create(rewriter, loc, info.blockStrideBytes, 32);
  Value blockByteOffset =
      arith::MulIOp::create(rewriter, loc, currentSlot, blockStrideBytes);
  Value receiverAddress =
      arith::AddIOp::create(rewriter, loc, baseAddress, blockByteOffset);
  receiverAddress =
      addByteOffset(loc, receiverAddress, info.staticTileByteOffset, rewriter);

  Value repeatStride =
      arith::ConstantIntOp::create(rewriter, loc, info.repeatStride, 32);
  Value blockCount =
      arith::ConstantIntOp::create(rewriter, loc, info.blockCount, 32);
  Value nextSlotUnwrapped =
      arith::AddIOp::create(rewriter, loc, currentSlot, repeatStride);
  Value nextSlot =
      arith::RemUIOp::create(rewriter, loc, nextSlotUnwrapped, blockCount);
  memref::StoreOp::create(rewriter, loc, nextSlot, slotCounters,
                          ValueRange{counterIndex});
  return receiverAddress;
}

/// Compute a receiver DFB address from record-indexed address formulas without
/// expanding one control-flow branch per record.
static Value buildSelectedComputedReceiverDFBDestinationAddress(
    PipeTransferSendOp op, Location loc, ArrayRef<PipeResourceInfo> resources,
    Value recordIndex,
    const PipeComputedAddressCounterMap &computedAddressCounters,
    ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t> baseArgIndices;
  SmallVector<int64_t> initialSlots;
  SmallVector<int64_t> repeatStrides;
  SmallVector<int64_t> blockCounts;
  SmallVector<int64_t> blockStrideBytes;
  SmallVector<int64_t> staticTileByteOffsets;
  SmallVector<int64_t> usesDynamicCounter;
  SmallVector<int64_t> counterIndices;
  std::optional<int64_t> validCounterIndex;

  auto computedResource =
      llvm::find_if(resources, [](const PipeResourceInfo &resource) {
        return resource.addressStorage.computedAddress.has_value();
      });
  assert(computedResource != resources.end() &&
         "selected pipe has no computed-address record");
  const PipeComputedAddressInfo &fallbackInfo =
      *computedResource->addressStorage.computedAddress;

  for (const PipeResourceInfo &resource : resources) {
    const PipeComputedAddressInfo &info =
        resource.addressStorage.computedAddress
            ? *resource.addressStorage.computedAddress
            : fallbackInfo;
    baseArgIndices.push_back(info.baseRuntimeCommonArgIndex);
    initialSlots.push_back(info.initialSlot);
    repeatStrides.push_back(info.repeatStride);
    blockCounts.push_back(info.blockCount);
    blockStrideBytes.push_back(info.blockStrideBytes);
    staticTileByteOffsets.push_back(info.staticTileByteOffset);
    usesDynamicCounter.push_back(info.usesDynamicSlotCounter() ? 1 : 0);
    if (info.dynamicSlotCounterIndex) {
      validCounterIndex = info.dynamicSlotCounterIndex;
    }
    counterIndices.push_back(info.dynamicSlotCounterIndex.value_or(0));
  }
  assert(!resources.empty() && "selected pipe resource table is empty");

  if (validCounterIndex) {
    for (std::size_t record = 0; record < counterIndices.size(); ++record) {
      if (!usesDynamicCounter[record]) {
        counterIndices[record] = *validCounterIndex;
      }
    }
  }

  Value baseArgIndex =
      loadIndexTableEntry(loc, baseArgIndices, recordIndex, rewriter);
  Value baseAddress = buildPipeRuntimeCommonArg(loc, rewriter, baseArgIndex);
  Value initialSlot =
      loadIndexTableEntry(loc, initialSlots, recordIndex, rewriter);
  Value blockStride =
      loadIndexTableEntry(loc, blockStrideBytes, recordIndex, rewriter);
  Value tileByteOffset =
      loadIndexTableEntry(loc, staticTileByteOffsets, recordIndex, rewriter);
  Value initialSlotI32 = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(), initialSlot);
  Value blockStrideI32 = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(), blockStride);
  Value tileByteOffsetI32 = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(), tileByteOffset);

  auto buildAddress = [&](Value slot) {
    Value blockByteOffset =
        arith::MulIOp::create(rewriter, loc, slot, blockStrideI32);
    Value byteOffset = arith::AddIOp::create(rewriter, loc, blockByteOffset,
                                             tileByteOffsetI32);
    return addByteOffset(loc, baseAddress, byteOffset, rewriter);
  };

  if (!validCounterIndex) {
    return buildAddress(initialSlotI32);
  }

  Value dynamicFlag =
      loadIndexTableEntry(loc, usesDynamicCounter, recordIndex, rewriter);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value hasDynamicCounter = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ne, dynamicFlag, zero);
  auto selectAddress = scf::IfOp::create(
      rewriter, loc, TypeRange{rewriter.getI32Type()}, hasDynamicCounter,
      /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&selectAddress.getThenRegion().front());
    Value counterIndex =
        loadIndexTableEntry(loc, counterIndices, recordIndex, rewriter);
    Value slotCounters =
        lookupComputedAddressCounter(op, computedAddressCounters);
    Value currentSlotI32 = memref::LoadOp::create(rewriter, loc, slotCounters,
                                                  ValueRange{counterIndex});
    Value receiverAddress = buildAddress(currentSlotI32);
    Value repeatStride =
        loadIndexTableEntry(loc, repeatStrides, recordIndex, rewriter);
    Value blockCount =
        loadIndexTableEntry(loc, blockCounts, recordIndex, rewriter);
    Value repeatStrideI32 = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(), repeatStride);
    Value blockCountI32 = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(), blockCount);
    Value nextSlotUnwrapped =
        arith::AddIOp::create(rewriter, loc, currentSlotI32, repeatStrideI32);
    Value nextSlot =
        arith::RemUIOp::create(rewriter, loc, nextSlotUnwrapped, blockCountI32);
    memref::StoreOp::create(rewriter, loc, nextSlot, slotCounters,
                            ValueRange{counterIndex});
    scf::YieldOp::create(rewriter, loc, receiverAddress);

    rewriter.setInsertionPointToStart(&selectAddress.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, buildAddress(initialSlotI32));
  }
  rewriter.setInsertionPointAfter(selectAddress);
  return selectAddress.getResult(0);
}

static Value buildSelectedUsesComputedReceiverDFB(
    Location loc, ArrayRef<PipeAddressMode> addressModes, Value recordIndex,
    ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t> usesComputedAddress =
      llvm::map_to_vector(addressModes, [](PipeAddressMode mode) {
        return static_cast<int64_t>(mode ==
                                    PipeAddressMode::ComputedReceiverDFB);
      });
  Value selectedMode =
      loadIndexTableEntry(loc, usesComputedAddress, recordIndex, rewriter);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                               selectedMode, zero);
}

static void lowerPipeCapacityRelease(Location loc, FuncOp func,
                                     const PipeCapacityReleaseInfo &release,
                                     const PipeResourcePlan &pipeResourcePlan,
                                     Value nocVal,
                                     ConversionPatternRewriter &rewriter) {
  const PipeCapacityReleaseTarget &target = release.target;
  auto indexTy = rewriter.getIndexType();
  Value counterAddress = buildPipeCounterAddress(loc, func, release.counter,
                                                 pipeResourcePlan, rewriter);
  Value sourceXLogical =
      arith::ConstantIndexOp::create(rewriter, loc, target.logicalX);
  Value sourceYLogical =
      arith::ConstantIndexOp::create(rewriter, loc, target.logicalY);
  Value sourceXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, sourceXLogical);
  Value sourceYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, sourceYLogical);
  Value releaseCount =
      arith::ConstantIntOp::create(rewriter, loc, release.count, 32);
  Value remoteCapacityNocAddr =
      ttk::GetNocAddrOp::create(rewriter, loc, sourceXTranslated,
                                sourceYTranslated, counterAddress, nocVal)
          .getResult();
  ttk::NocSemaphoreIncOp::create(rewriter, loc, remoteCapacityNocAddr,
                                 releaseCount, nocVal, /*posted=*/BoolAttr());
}

struct SelectedPipeFields {
  Value recordIndex;
  Value srcX;
  Value srcY;
  Value dstStartX;
  Value dstStartY;
  Value dstEndX;
  Value dstEndY;
  Value numDests;
  Value srcInDstRange;
  bool isCollective;
};

static SelectedPipeFields getSelectedPipeFields(const PipeReference &pipeRef) {
  assert(pipeRef.isSelected() && "expected selected pipe reference");
  if (pipeRef.isSelectedSrc()) {
    SelectPipeSrcOp op = pipeRef.getSelectedSrc();
    return SelectedPipeFields{
        op.getRecordIndex(),
        op.getSrcX(),
        op.getSrcY(),
        op.getDstStartX(),
        op.getDstStartY(),
        op.getDstEndX(),
        op.getDstEndY(),
        op.getNumDests(),
        op.getSrcInDstRange(),
        op.getRecords().getPipes().front().getIsCollective()};
  }
  SelectPipeDstOp op = pipeRef.getSelectedDst();
  return SelectedPipeFields{
      op.getRecordIndex(),
      op.getSrcX(),
      op.getSrcY(),
      op.getDstStartX(),
      op.getDstStartY(),
      op.getDstEndX(),
      op.getDstEndY(),
      op.getNumDests(),
      op.getSrcInDstRange(),
      op.getRecords().getPipes().front().getIsCollective()};
}

/// Compute the exact DFB address selected by ttl.copy(pipe, dst). Receivers
/// publish this address so senders do not have to infer receiver DFB state.
static Value
buildReceiverPublishedAddress(Value dst, Location loc,
                              const PipeReceiverAddressPublicationPlan &info,
                              ConversionPatternRewriter &rewriter) {
  auto receiverCBConverted =
      utils::convertTTLCBToTTKernel(info.receiverDFB, rewriter, loc);
  assert(succeeded(receiverCBConverted) &&
         "pipe post planning guarantees a convertible receiver DFB");

  auto receiverWritePtr =
      ttk::GetWritePtrOp::create(rewriter, loc, *receiverCBConverted);
  Value publishedAddress = receiverWritePtr;
  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value localTileIndex = zeroIdx;
  Value globalTileIndex =
      utils::addSliceOffset(dst, localTileIndex, rewriter, loc);
  if (globalTileIndex == localTileIndex) {
    return publishedAddress;
  }

  auto tileOffsetI32 = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(), globalTileIndex);
  auto pageSizeBytes =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                rewriter.getI32IntegerAttr(info.tileSizeBytes));
  auto byteOffset =
      arith::MulIOp::create(rewriter, loc, tileOffsetI32, pageSizeBytes);
  return arith::AddIOp::create(rewriter, loc, receiverWritePtr, byteOffset)
      .getResult();
}

namespace {

/// Emits transport-specific PipeNet sender operations.
class PipeSendTransportEmitter {
public:
  virtual ~PipeSendTransportEmitter() = default;

  virtual void preparePayloadWrite() = 0;
  virtual LogicalResult emitPayloadWrite(Value srcAddr, Value dstAddr,
                                         Value totalSizeBytes) = 0;
  virtual void emitPayloadWriteBarrier() = 0;
  virtual LogicalResult
  emitReceiverCompletionIncrement(Value receiverCompletionCounterAddr) = 0;
  virtual void emitCompletionSignalBarrier() = 0;
};

/// Emits NoC operations performed while a receiver posts a transfer.
class PipeReceiverPostTransportEmitter {
public:
  virtual ~PipeReceiverPostTransportEmitter() = default;

  virtual LogicalResult emitReceiverAddressPublish(Value senderTableAddress,
                                                   Value publishedAddress) = 0;
  virtual void emitAddressPublishBarrier() = 0;
  virtual LogicalResult
  emitSenderReadyIncrement(Value senderReadyCounterAddr) = 0;
};

class NocPipeTransportEmitterBase : public PipeSendTransportEmitter,
                                    public PipeReceiverPostTransportEmitter {
protected:
  struct LogicalCore {
    Value x;
    Value y;
  };

  struct TranslatedCore {
    Value x;
    Value y;
  };

  struct DestinationRange {
    Value startX;
    Value startY;
    Value endX;
    Value endY;
  };

public:
  NocPipeTransportEmitterBase(Operation *op,
                              ConversionPatternRewriter &rewriter)
      : loc(op->getLoc()), rewriter(rewriter), nocIdx(getNocIndex(op)),
        nocVal(arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(),
                                         rewriter.getI8IntegerAttr(nocIdx))) {}

  LogicalResult emitReceiverAddressPublish(Value senderTableAddress,
                                           Value publishedAddress) override {
    LogicalCore sourceCore = getSourceLogicalCore();
    // The remote publish and the following ready signal reuse the translated
    // source coordinates, so they must be created before either branch.
    getSourceCore();
    Value currentX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    Value currentY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
    Value xMatches = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, currentX, sourceCore.x);
    Value yMatches = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, currentY, sourceCore.y);
    Value receiverIsSource =
        arith::AndIOp::create(rewriter, loc, xMatches, yMatches);
    auto localPublish = scf::IfOp::create(rewriter, loc, receiverIsSource,
                                          /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&localPublish.getThenRegion().front());
      emitLocalReceiverAddressPublish(senderTableAddress, publishedAddress);
      rewriter.setInsertionPointToStart(&localPublish.getElseRegion().front());
      emitRemoteReceiverAddressPublish(senderTableAddress, publishedAddress);
    }
    rewriter.setInsertionPointAfter(localPublish);
    return success();
  }

  void emitLocalReceiverAddressPublish(Value senderTableAddress,
                                       Value publishedAddress) {
    auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
    Value tablePtr =
        ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, senderTableAddress);
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    ttk::StoreToL1Op::create(rewriter, loc, publishedAddress, tablePtr, zero);
  }

  void emitRemoteReceiverAddressPublish(Value senderTableAddress,
                                        Value publishedAddress) {
    TranslatedCore sourceCore = getSourceCore();
    auto byteEnableAll = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(0xF));
    // An inline NoC write does not update the sender's local SRAM when the
    // sender is also this receiver, so that case uses a direct L1 store.
    ttk::NocInlineDwWriteOp::create(rewriter, loc, sourceCore.x, sourceCore.y,
                                    senderTableAddress, publishedAddress,
                                    byteEnableAll, nocVal);
  }

  void emitAddressPublishBarrier() override {
    ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);
  }

  LogicalResult
  emitSenderReadyIncrement(Value senderReadyCounterAddr) override {
    TranslatedCore sourceCore = getSourceCore();
    auto senderReadyCounterNocAddr =
        ttk::GetNocAddrOp::create(rewriter, loc, sourceCore.x, sourceCore.y,
                                  senderReadyCounterAddr, nocVal);
    auto readyCounterIncrement =
        arith::ConstantIndexOp::create(rewriter, loc, 1);
    ttk::NocSemaphoreIncOp::create(
        rewriter, loc, senderReadyCounterNocAddr.getResult(),
        readyCounterIncrement, nocVal, /*posted=*/BoolAttr());
    return success();
  }

  void emitPayloadWriteBarrier() override {
    ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);
  }

  void emitCompletionSignalBarrier() override {
    ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocVal);
  }

protected:
  virtual LogicalCore getSourceLogicalCore() = 0;
  virtual TranslatedCore getSourceCore() = 0;

  TranslatedCore buildTranslatedCore(Value logicalX, Value logicalY) {
    auto translatedX = ttk::ConvertLogicalXToTranslatedOp::create(
        rewriter, loc, rewriter.getIndexType(), logicalX);
    auto translatedY = ttk::ConvertLogicalYToTranslatedOp::create(
        rewriter, loc, rewriter.getIndexType(), logicalY);
    return {translatedX, translatedY};
  }

  TranslatedCore buildTranslatedCore(int64_t logicalX, int64_t logicalY) {
    auto logicalXValue =
        arith::ConstantIndexOp::create(rewriter, loc, logicalX);
    auto logicalYValue =
        arith::ConstantIndexOp::create(rewriter, loc, logicalY);
    auto translatedX = ttk::ConvertLogicalXToTranslatedOp::create(
        rewriter, loc, rewriter.getIndexType(), logicalXValue);
    auto translatedY = ttk::ConvertLogicalYToTranslatedOp::create(
        rewriter, loc, rewriter.getIndexType(), logicalYValue);
    return {translatedX, translatedY};
  }

  Location loc;
  ConversionPatternRewriter &rewriter;
  int64_t nocIdx;
  Value nocVal;
};

class NocPipeTransportEmitter final : public NocPipeTransportEmitterBase {
public:
  NocPipeTransportEmitter(Operation *op, PipeType pipeType,
                          ConversionPatternRewriter &rewriter)
      : NocPipeTransportEmitterBase(op, rewriter), pipeType(pipeType) {}

  LogicalResult emitReceiverAddressPublish(Value senderTableAddress,
                                           Value publishedAddress) override {
    if (!pipeType.srcInDstRange()) {
      emitRemoteReceiverAddressPublish(senderTableAddress, publishedAddress);
      return success();
    }
    if (pipeType.hasSingleReceiver()) {
      emitLocalReceiverAddressPublish(senderTableAddress, publishedAddress);
      return success();
    }
    return NocPipeTransportEmitterBase::emitReceiverAddressPublish(
        senderTableAddress, publishedAddress);
  }

  void preparePayloadWrite() override {
    // Materialize destination coordinates before computing the payload address
    // so address selection does not change emitted operation order.
    if (pipeType.hasSingleReceiver()) {
      getDstStartCore();
      return;
    }
    getDestinationRange();
  }

  LogicalResult emitPayloadWrite(Value srcAddr, Value dstAddr,
                                 Value totalSizeBytes) override {
    if (pipeType.hasSingleReceiver()) {
      TranslatedCore dstStartCore = getDstStartCore();
      ttk::NocAsyncWriteOp::create(
          rewriter, loc, srcAddr, ValueRange{dstStartCore.x, dstStartCore.y},
          ValueRange{}, dstAddr, totalSizeBytes, nocVal);
      return success();
    }

    DestinationRange destinationRange = getDestinationRange();
    auto numDests = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(pipeType.getNumDests()));
    if (pipeType.srcInDstRange()) {
      ttk::NocAsyncWriteMulticastLoopbackSrcOp::create(
          rewriter, loc, srcAddr, totalSizeBytes, numDests,
          destinationRange.startX, destinationRange.startY,
          destinationRange.endX, destinationRange.endY, dstAddr, nocVal,
          /*linked=*/nullptr);
      return success();
    }
    ttk::NocAsyncWriteMulticastOp::create(
        rewriter, loc, srcAddr, totalSizeBytes, numDests,
        destinationRange.startX, destinationRange.startY, destinationRange.endX,
        destinationRange.endY, dstAddr, nocVal, /*linked=*/nullptr);
    return success();
  }

  /// Emit page-addressed unicast writes for one transport payload.
  LogicalResult emitPayloadPageWrites(Value srcAddr, Value dstAddr,
                                      int64_t pageCount,
                                      int64_t pageSizeBytes) {
    assert(pipeType.hasSingleReceiver() &&
           "page writes require a unicast transport");
    assert(pageCount > 1 && pageSizeBytes > 0 &&
           "page writes require a multi-page payload");

    Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upperBound = arith::ConstantIndexOp::create(rewriter, loc, pageCount);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value pageSize =
        arith::ConstantIntOp::create(rewriter, loc, pageSizeBytes, 32);
    auto pageLoop =
        scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);

    OpBuilder::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPointToStart(pageLoop.getBody());
    Value pageIndex = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(), pageLoop.getInductionVar());
    Value pageOffset =
        arith::MulIOp::create(rewriter, loc, pageIndex, pageSize);
    Value pageSrcAddr =
        arith::AddIOp::create(rewriter, loc, srcAddr, pageOffset);
    Value pageDstAddr =
        arith::AddIOp::create(rewriter, loc, dstAddr, pageOffset);
    TranslatedCore dstStartCore = getDstStartCore();
    ttk::NocAsyncWriteOp::create(rewriter, loc, pageSrcAddr,
                                 ValueRange{dstStartCore.x, dstStartCore.y},
                                 ValueRange{}, pageDstAddr, pageSize, nocVal);
    return success();
  }

  void emitPayloadWriteBarrier() override {
    ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);
  }

  LogicalResult emitReceiverCompletionIncrement(
      Value receiverCompletionCounterAddr) override {
    auto completionIncrement = arith::ConstantIndexOp::create(rewriter, loc, 1);

    if (pipeType.hasSingleReceiver()) {
      TranslatedCore dstStartCore = getDstStartCore();
      auto receiverCompletionNocAddr = ttk::GetNocAddrOp::create(
          rewriter, loc, dstStartCore.x, dstStartCore.y,
          receiverCompletionCounterAddr, nocVal);
      ttk::NocSemaphoreIncOp::create(
          rewriter, loc, receiverCompletionNocAddr.getResult(),
          completionIncrement, nocVal, /*posted=*/BoolAttr());
      return success();
    }

    DestinationRange destinationRange = getDestinationRange();
    int64_t numRemoteDests = pipeType.srcInDstRange()
                                 ? pipeType.getNumDests() - 1
                                 : pipeType.getNumDests();
    auto remoteReceiverCount =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(numRemoteDests));
    auto remoteReceiverCompletionMcastNocAddr =
        ttk::GetNocMulticastAddrOp::create(
            rewriter, loc, destinationRange.startX, destinationRange.startY,
            destinationRange.endX, destinationRange.endY,
            receiverCompletionCounterAddr, nocVal);
    ttk::NocSemaphoreIncMulticastOp::create(
        rewriter, loc, remoteReceiverCompletionMcastNocAddr.getResult(),
        completionIncrement, remoteReceiverCount, nocVal,
        /*posted=*/BoolAttr());

    if (pipeType.srcInDstRange()) {
      TranslatedCore sourceCore = getSourceCore();
      auto localReceiverCompletionNocAddr =
          ttk::GetNocAddrOp::create(rewriter, loc, sourceCore.x, sourceCore.y,
                                    receiverCompletionCounterAddr, nocVal);
      ttk::NocSemaphoreIncOp::create(
          rewriter, loc, localReceiverCompletionNocAddr.getResult(),
          completionIncrement, nocVal, /*posted=*/BoolAttr());
    }
    return success();
  }

private:
  LogicalCore getSourceLogicalCore() override {
    return {arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX()),
            arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY())};
  }

  TranslatedCore getSourceCore() override {
    if (!sourceCore) {
      sourceCore = buildTranslatedCore(pipeType.getSrcX(), pipeType.getSrcY());
    }
    return *sourceCore;
  }

  TranslatedCore getDstStartCore() {
    if (!dstStartCore) {
      dstStartCore =
          buildTranslatedCore(pipeType.getDstStartX(), pipeType.getDstStartY());
    }
    return *dstStartCore;
  }

  DestinationRange getDestinationRange() {
    if (destinationRange) {
      return *destinationRange;
    }
    TranslatedCore dstStartTranslatedCore = getDstStartCore();
    // Preserve the memoized start coordinate for unicast and completion uses.
    auto [dstStartX, dstStartY] = dstStartTranslatedCore;
    auto [dstEndX, dstEndY] =
        buildTranslatedCore(pipeType.getDstEndX(), pipeType.getDstEndY());
    // NoC 1 traverses the grid in reverse coordinate order, while multicast
    // operations require their endpoints in traversal order.
    if (nocIdx == 1) {
      std::swap(dstStartX, dstEndX);
      std::swap(dstStartY, dstEndY);
    }
    destinationRange = DestinationRange{dstStartX, dstStartY, dstEndX, dstEndY};
    return *destinationRange;
  }

  PipeType pipeType;
  std::optional<TranslatedCore> sourceCore;
  std::optional<TranslatedCore> dstStartCore;
  std::optional<DestinationRange> destinationRange;
};

/// Emit one protocol body for every record in a PipeNet table. Record fields
/// select the required unicast, multicast, and loopback hardware operations;
/// the conditions are transport semantics, not special cases for record
/// indices.
class SelectedNocPipeTransportEmitter final
    : public NocPipeTransportEmitterBase {
public:
  SelectedNocPipeTransportEmitter(Operation *op, SelectedPipeFields fields,
                                  ConversionPatternRewriter &rewriter)
      : NocPipeTransportEmitterBase(op, rewriter), fields(fields) {}

  void preparePayloadWrite() override {
    // Coordinate translations must dominate the conditional regions emitted
    // below, so cache the applicable coordinates before creating those regions.
    if (fields.isCollective) {
      getDestinationRange();
      return;
    }
    getDstStartCore();
  }

  LogicalResult emitPayloadWrite(Value srcAddr, Value dstAddr,
                                 Value totalSizeBytes) override {
    if (!fields.isCollective) {
      emitUnicastPayloadWrite(srcAddr, dstAddr, totalSizeBytes);
      return success();
    }

    DestinationRange destinationRange = getDestinationRange();
    Value numDests = getNumDests();
    // A one-receiver collective uses unicast hardware operations. This also
    // avoids a zero-recipient multicast completion for local loopback.
    auto singleReceiverIf = scf::IfOp::create(
        rewriter, loc, getHasSingleReceiver(), /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(
          &singleReceiverIf.getThenRegion().front());
      emitUnicastPayloadWrite(srcAddr, dstAddr, totalSizeBytes);
      rewriter.setInsertionPointToStart(
          &singleReceiverIf.getElseRegion().front());
      emitMulticastPayloadWrite(srcAddr, dstAddr, totalSizeBytes, numDests,
                                destinationRange);
    }
    rewriter.setInsertionPointAfter(singleReceiverIf);
    return success();
  }

  LogicalResult emitReceiverCompletionIncrement(
      Value receiverCompletionCounterAddr) override {
    auto completionIncrement = arith::ConstantIndexOp::create(rewriter, loc, 1);

    if (!fields.isCollective) {
      emitUnicastCompletionIncrement(receiverCompletionCounterAddr,
                                     completionIncrement);
      return success();
    }

    DestinationRange destinationRange = getDestinationRange();
    Value numDests = getNumDests();
    auto singleReceiverIf = scf::IfOp::create(
        rewriter, loc, getHasSingleReceiver(), /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(
          &singleReceiverIf.getThenRegion().front());
      emitUnicastCompletionIncrement(receiverCompletionCounterAddr,
                                     completionIncrement);
      rewriter.setInsertionPointToStart(
          &singleReceiverIf.getElseRegion().front());
      emitMulticastCompletionIncrement(receiverCompletionCounterAddr,
                                       completionIncrement, numDests,
                                       destinationRange);
    }
    rewriter.setInsertionPointAfter(singleReceiverIf);
    return success();
  }

private:
  void emitUnicastPayloadWrite(Value srcAddr, Value dstAddr,
                               Value totalSizeBytes) {
    TranslatedCore dstStartCore = getDstStartCore();
    ttk::NocAsyncWriteOp::create(rewriter, loc, srcAddr,
                                 ValueRange{dstStartCore.x, dstStartCore.y},
                                 ValueRange{}, dstAddr, totalSizeBytes, nocVal);
  }

  void emitMulticastPayloadWrite(Value srcAddr, Value dstAddr,
                                 Value totalSizeBytes, Value numDests,
                                 DestinationRange destinationRange) {
    // Standard multicast does not write the sender's local memory, so a
    // receiver range containing the sender requires the loopback operation.
    auto loopbackIf = scf::IfOp::create(rewriter, loc, fields.srcInDstRange,
                                        /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&loopbackIf.getThenRegion().front());
      ttk::NocAsyncWriteMulticastLoopbackSrcOp::create(
          rewriter, loc, srcAddr, totalSizeBytes, numDests,
          destinationRange.startX, destinationRange.startY,
          destinationRange.endX, destinationRange.endY, dstAddr, nocVal,
          /*linked=*/nullptr);
      rewriter.setInsertionPointToStart(&loopbackIf.getElseRegion().front());
      ttk::NocAsyncWriteMulticastOp::create(
          rewriter, loc, srcAddr, totalSizeBytes, numDests,
          destinationRange.startX, destinationRange.startY,
          destinationRange.endX, destinationRange.endY, dstAddr, nocVal,
          /*linked=*/nullptr);
    }
    rewriter.setInsertionPointAfter(loopbackIf);
  }

  void emitUnicastCompletionIncrement(Value receiverCompletionCounterAddr,
                                      Value completionIncrement) {
    TranslatedCore dstStartCore = getDstStartCore();
    auto receiverCompletionNocAddr =
        ttk::GetNocAddrOp::create(rewriter, loc, dstStartCore.x, dstStartCore.y,
                                  receiverCompletionCounterAddr, nocVal);
    ttk::NocSemaphoreIncOp::create(
        rewriter, loc, receiverCompletionNocAddr.getResult(),
        completionIncrement, nocVal, /*posted=*/BoolAttr());
  }

  void emitMulticastCompletionIncrement(Value receiverCompletionCounterAddr,
                                        Value completionIncrement,
                                        Value numDests,
                                        DestinationRange destinationRange) {
    // The multicast atomic updates only remote receivers. If the sender is also
    // a receiver, exclude it from that count and update it with a local atomic.
    auto one = arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                         rewriter.getI32IntegerAttr(1));
    Value numRemoteWithLoopback =
        arith::SubIOp::create(rewriter, loc, numDests, one);
    Value numRemoteDests = arith::SelectOp::create(
        rewriter, loc, fields.srcInDstRange, numRemoteWithLoopback, numDests);
    auto remoteReceiverCompletionMcastNocAddr =
        ttk::GetNocMulticastAddrOp::create(
            rewriter, loc, destinationRange.startX, destinationRange.startY,
            destinationRange.endX, destinationRange.endY,
            receiverCompletionCounterAddr, nocVal);
    ttk::NocSemaphoreIncMulticastOp::create(
        rewriter, loc, remoteReceiverCompletionMcastNocAddr.getResult(),
        completionIncrement, numRemoteDests, nocVal, /*posted=*/BoolAttr());

    auto localIncrementIf = scf::IfOp::create(
        rewriter, loc, fields.srcInDstRange, /*withElseRegion=*/false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(
          &localIncrementIf.getThenRegion().front());
      TranslatedCore sourceCore = getSourceCore();
      auto localReceiverCompletionNocAddr =
          ttk::GetNocAddrOp::create(rewriter, loc, sourceCore.x, sourceCore.y,
                                    receiverCompletionCounterAddr, nocVal);
      ttk::NocSemaphoreIncOp::create(
          rewriter, loc, localReceiverCompletionNocAddr.getResult(),
          completionIncrement, nocVal, /*posted=*/BoolAttr());
    }
    rewriter.setInsertionPointAfter(localIncrementIf);
  }

  Value getHasSingleReceiver() {
    if (!hasSingleReceiver) {
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      hasSingleReceiver = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, fields.numDests, one);
    }
    return hasSingleReceiver;
  }

  LogicalCore getSourceLogicalCore() override {
    return {fields.srcX, fields.srcY};
  }

  TranslatedCore getSourceCore() override {
    if (!sourceCore) {
      sourceCore = buildTranslatedCore(fields.srcX, fields.srcY);
    }
    return *sourceCore;
  }

  TranslatedCore getDstStartCore() {
    if (!dstStartCore) {
      dstStartCore = buildTranslatedCore(fields.dstStartX, fields.dstStartY);
    }
    return *dstStartCore;
  }

  DestinationRange getDestinationRange() {
    if (destinationRange) {
      return *destinationRange;
    }
    TranslatedCore dstStartTranslatedCore = getDstStartCore();
    auto [dstStartX, dstStartY] = dstStartTranslatedCore;
    auto [dstEndX, dstEndY] =
        buildTranslatedCore(fields.dstEndX, fields.dstEndY);
    // NoC 1 traverses the grid in reverse coordinate order, while multicast
    // operations require their endpoints in traversal order.
    if (nocIdx == 1) {
      std::swap(dstStartX, dstEndX);
      std::swap(dstStartY, dstEndY);
    }
    destinationRange = DestinationRange{dstStartX, dstStartY, dstEndX, dstEndY};
    return *destinationRange;
  }

  Value getNumDests() {
    if (!numDests) {
      numDests = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI32Type(), fields.numDests);
    }
    return numDests;
  }

  SelectedPipeFields fields;
  Value hasSingleReceiver;
  Value numDests;
  std::optional<TranslatedCore> sourceCore;
  std::optional<TranslatedCore> dstStartCore;
  std::optional<DestinationRange> destinationRange;
};

class FabricRouteEmitter {
public:
  FabricRouteEmitter(Operation *op, Value routeIndex,
                     const FabricRuntimeInfo &runtime,
                     ConversionPatternRewriter &rewriter)
      : loc(op->getLoc()), routeIndex(routeIndex), runtime(runtime),
        rewriter(rewriter), nocVal(arith::ConstantIntOp::create(
                                rewriter, loc, getNocIndex(op), 8)) {}

  void emitAtomicIncrement(Value remoteX, Value remoteY, Value semaphoreAddress,
                           Value increment) {
    FabricRouteTarget target = buildRouteTarget();
    ttk::RoutingPlaneAtomicIncOp::create(
        rewriter, loc, runtime.manager, runtime.routeId, buildConnectionIndex(),
        target.destinationDeviceId, target.destinationMeshId,
        target.destinationHopCount,
        buildRemoteNocAddress(remoteX, remoteY, semaphoreAddress), increment);
  }

  void emitFusedWriteAtomicIncrement(Value remoteX, Value remoteY,
                                     Value sourceAddress,
                                     Value destinationAddress, Value sizeBytes,
                                     Value semaphoreAddress, Value increment) {
    FabricRouteTarget target = buildRouteTarget();
    ttk::RoutingPlaneFusedWriteAtomicIncOp::create(
        rewriter, loc, runtime.manager, runtime.routeId, buildConnectionIndex(),
        target.destinationDeviceId, target.destinationMeshId,
        target.destinationHopCount, sourceAddress, sizeBytes,
        buildRemoteNocAddress(remoteX, remoteY, destinationAddress),
        buildRemoteNocAddress(remoteX, remoteY, semaphoreAddress), increment);
  }

private:
  struct FabricRouteTarget {
    Value destinationDeviceId;
    Value destinationMeshId;
    Value destinationHopCount;
  };

  struct TranslatedNode {
    Value x;
    Value y;
  };

  TranslatedNode buildTranslatedNode(Value logicalX, Value logicalY) {
    return {
        ttk::ConvertLogicalXToTranslatedOp::create(
            rewriter, loc, rewriter.getIndexType(), logicalX),
        ttk::ConvertLogicalYToTranslatedOp::create(
            rewriter, loc, rewriter.getIndexType(), logicalY),
    };
  }

  Value buildRemoteNocAddress(Value logicalX, Value logicalY, Value l1Address) {
    TranslatedNode node = buildTranslatedNode(logicalX, logicalY);
    return ttk::GetNocAddrOp::create(rewriter, loc, node.x, node.y, l1Address,
                                     nocVal)
        .getResult();
  }

  Value buildConnectionIndex() {
    Value firstConnectionIndex =
        arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value argIndex =
        arith::AddIOp::create(rewriter, loc, firstConnectionIndex, routeIndex);
    argIndex =
        arith::AddIOp::create(rewriter, loc, runtime.runtimeArgBase, argIndex);
    return ttk::GetArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                    argIndex);
  }

  FabricRouteTarget buildRouteTarget() {
    Value firstDeviceIndex =
        arith::ConstantIndexOp::create(rewriter, loc, 1 + runtime.routeCount);
    Value firstMeshIndex = arith::ConstantIndexOp::create(
        rewriter, loc, 1 + 2 * runtime.routeCount);
    Value firstHopCountIndex = arith::ConstantIndexOp::create(
        rewriter, loc, 1 + 3 * runtime.routeCount);
    Value destinationDeviceIndex =
        arith::AddIOp::create(rewriter, loc, firstDeviceIndex, routeIndex);
    Value destinationMeshIndex =
        arith::AddIOp::create(rewriter, loc, firstMeshIndex, routeIndex);
    Value destinationHopCountIndex =
        arith::AddIOp::create(rewriter, loc, firstHopCountIndex, routeIndex);
    destinationDeviceIndex = arith::AddIOp::create(
        rewriter, loc, runtime.runtimeArgBase, destinationDeviceIndex);
    destinationMeshIndex = arith::AddIOp::create(
        rewriter, loc, runtime.runtimeArgBase, destinationMeshIndex);
    destinationHopCountIndex = arith::AddIOp::create(
        rewriter, loc, runtime.runtimeArgBase, destinationHopCountIndex);
    return {
        ttk::GetArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                 destinationDeviceIndex),
        ttk::GetArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                 destinationMeshIndex),
        ttk::GetArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                 destinationHopCountIndex),
    };
  }

  Location loc;
  Value routeIndex;
  const FabricRuntimeInfo &runtime;
  ConversionPatternRewriter &rewriter;
  Value nocVal;
};

class FabricPipeTransportEmitter final : public PipeSendTransportEmitter {
public:
  FabricPipeTransportEmitter(Operation *op, Value destinationX,
                             Value destinationY, Value routeIndex,
                             const FabricRuntimeInfo &runtime,
                             ConversionPatternRewriter &rewriter)
      : loc(op->getLoc()), destinationX(destinationX),
        destinationY(destinationY), rewriter(rewriter),
        routeEmitter(op, routeIndex, runtime, rewriter) {}

  void preparePayloadWrite() override {}

  LogicalResult emitPayloadWrite(Value srcAddr, Value dstAddr,
                                 Value totalSizeBytes) override {
    sourceAddress = srcAddr;
    destinationAddress = dstAddr;
    sizeBytes = totalSizeBytes;
    return success();
  }

  void emitPayloadWriteBarrier() override {}

  LogicalResult emitReceiverCompletionIncrement(
      Value receiverCompletionCounterAddr) override {
    assert(sourceAddress && destinationAddress && sizeBytes &&
           "fabric payload must be prepared before completion signaling");
    routeEmitter.emitFusedWriteAtomicIncrement(
        destinationX, destinationY, sourceAddress, destinationAddress,
        sizeBytes, receiverCompletionCounterAddr,
        arith::ConstantIntOp::create(rewriter, loc, 1, 32));
    return success();
  }

  void emitCompletionSignalBarrier() override {}

private:
  Location loc;
  Value destinationX;
  Value destinationY;
  ConversionPatternRewriter &rewriter;
  FabricRouteEmitter routeEmitter;
  Value sourceAddress;
  Value destinationAddress;
  Value sizeBytes;
};

} // namespace

LogicalResult PipeResourcePlan::forEachResourceTable(
    llvm::function_ref<LogicalResult(Operation *, ArrayRef<PipeResourceInfo>,
                                     PipeResourceTableKind)>
        callback) const {
  for (const auto &[operation, resource] : resources) {
    if (failed(callback(operation, ArrayRef<PipeResourceInfo>(&resource, 1),
                        PipeResourceTableKind::Static))) {
      return failure();
    }
  }
  for (const auto &[operation, resourceTable] : selectedResources) {
    if (failed(callback(operation, resourceTable,
                        PipeResourceTableKind::Selected))) {
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Receiver post sequence counter initialization
//===----------------------------------------------------------------------===//

static void appendUniquePipeCounter(SmallVectorImpl<PipeCounterInfo> &counters,
                                    PipeCounterInfo counter) {
  if (!llvm::is_contained(counters, counter)) {
    counters.push_back(counter);
  }
}

static void sortPipeCounters(SmallVectorImpl<PipeCounterInfo> &counters) {
  llvm::sort(counters, [](PipeCounterInfo lhs, PipeCounterInfo rhs) {
    return std::make_pair(lhs.getStorage(), lhs.getIndex()) <
           std::make_pair(rhs.getStorage(), rhs.getIndex());
  });
}

static PipeCounterTable
buildZeroInitializedCounterTable(FuncOp func,
                                 SmallVector<PipeCounterInfo> counters) {
  assert(!counters.empty() && "counter table must not be empty");
  sortPipeCounters(counters);
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(&func.getBody().front());
  Location loc = func.getLoc();
  auto memrefType = MemRefType::get({static_cast<int64_t>(counters.size())},
                                    builder.getI32Type());
  Value values = memref::AllocaOp::create(builder, loc, memrefType);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  for (std::size_t counterIndex = 0; counterIndex < counters.size();
       ++counterIndex) {
    Value index = arith::ConstantIndexOp::create(builder, loc, counterIndex);
    memref::StoreOp::create(builder, loc, zero, values, ValueRange{index});
  }
  return PipeCounterTable{values, std::move(counters)};
}

void initializePipePostSequenceCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeCounterTableMap &postSequenceCounters) {
  llvm::MapVector<FuncOp, SmallVector<PipeCounterInfo>> countersByFunc;
  LogicalResult traversalResult = pipeResourcePlan.forEachResourceTable(
      [&](Operation *protocolOp, ArrayRef<PipeResourceInfo> resources,
          PipeResourceTableKind) {
        auto postOp = dyn_cast<PipeTransferPostOp>(protocolOp);
        if (!postOp) {
          return success();
        }
        FuncOp func = postOp->getParentOfType<FuncOp>();
        assert(func && "pipe transfer post must be inside a function");
        SmallVector<PipeCounterInfo> &counters = countersByFunc[func];
        for (const PipeResourceInfo &resource : resources) {
          appendUniquePipeCounter(counters, resource.completion.counter);
        }
        return success();
      });
  assert(succeeded(traversalResult) && "infallible resource traversal failed");

  for (auto &[func, counters] : countersByFunc) {
    postSequenceCounters[func] =
        buildZeroInitializedCounterTable(func, std::move(counters));
  }
}

void materializePipeTransportCompletionBarriers(
    const PipeTransportPlan &pipeTransportPlan) {
  llvm::SmallSetVector<Operation *, 8> completionLoops;
  auto recordCompletionLoop =
      [&](const PipeTransportIterationDomain &iterationDomain) {
        assert(!iterationDomain.enclosingLoops.empty() &&
               "iteration-domain completion requires an enclosing loop");
        completionLoops.insert(iterationDomain.enclosingLoops.back());
      };

  for (const PipeTransportStream &stream : pipeTransportPlan.getStreams()) {
    if (stream.getCreditCompletion() !=
        PipeTransportCreditCompletion::IterationDomain) {
      continue;
    }
    recordCompletionLoop(stream.getSourceIterationDomain());
    for (const PipeTransportIterationDomain &iterationDomain :
         stream.getCapacityReleaseIterationDomains()) {
      recordCompletionLoop(iterationDomain);
    }
  }

  for (Operation *loop : completionLoops) {
    OpBuilder builder(loop);
    builder.setInsertionPointAfter(loop);
    Location loc = loop->getLoc();
    int64_t nocIndex = getNocIndex(loop);
    Value noc = arith::ConstantOp::create(builder, loc, builder.getI8Type(),
                                          builder.getI8IntegerAttr(nocIndex));
    ttk::NocAsyncAtomicBarrierOp::create(builder, loc, noc);
  }
}

void initializePipeCapacityCounters(
    const PipeCapacityPlan &pipeCapacityPlan,
    const PipeResourcePlan &pipeResourcePlan,
    PipeCounterProgressMap &senderCapacityCounters) {
  for (const auto &entry : pipeCapacityPlan.getInitializations()) {
    FuncOp func = entry.first;
    const SmallVector<PipeCapacityInitInfo> &initializations = entry.second;
    SmallVector<PipeCapacityInitInfo> sortedInitializations(initializations);
    llvm::sort(sortedInitializations, [](const PipeCapacityInitInfo &lhs,
                                         const PipeCapacityInitInfo &rhs) {
      return std::make_pair(lhs.counter.getStorage(), lhs.counter.getIndex()) <
             std::make_pair(rhs.counter.getStorage(), rhs.counter.getIndex());
    });

    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto counterMemrefTy = MemRefType::get({1}, builder.getI32Type());
    Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
    Value zeroI32 = arith::ConstantIntOp::create(builder, loc, 0, 32);
    auto &perFuncCounters = senderCapacityCounters[func];
    for (const PipeCapacityInitInfo &init : sortedInitializations) {
      Value capacityCounterPtr = buildPipeCounterPtr(loc, func, init.counter,
                                                     pipeResourcePlan, builder);
      Value initialCapacity =
          arith::ConstantIntOp::create(builder, loc, init.initialCapacity, 32);
      ttk::NocSemaphoreSetOp::create(builder, loc, capacityCounterPtr,
                                     initialCapacity);
      // The sender tracks its cumulative acquired count in a kernel-local
      // counter and waits for the shared capacity counter to reach it, so the
      // receiver's remote increment stays the only writer of the shared word.
      auto counter = memref::AllocaOp::create(builder, loc, counterMemrefTy);
      memref::StoreOp::create(builder, loc, zeroI32, counter,
                              ValueRange{zeroIdx});
      perFuncCounters.push_back(
          PipeCounterProgress{init.counter, counter.getResult()});
    }
  }
}

void initializeFabricReadyCounters(const PipeModulePlan &pipeModulePlan,
                                   const PipeResourcePlan &pipeResourcePlan,
                                   PipeCounterTableMap &fabricReadyCounters) {
  llvm::MapVector<FuncOp, SmallVector<PipeCounterInfo>> countersByFunc;
  auto appendFabricReadyCounter = [&](Operation *protocolOp,
                                      const PipeResourceInfo &resource) {
    auto sendOp = dyn_cast<PipeTransferSendOp>(protocolOp);
    if (!sendOp || pipeModulePlan.getTransferPlan(protocolOp)
                           .getSynchronizationProtocol() !=
                       PipeSynchronizationProtocol::Fabric) {
      return;
    }
    assert(resource.readyCounter &&
           resource.readyCounter->getStorage() ==
               PipeCounterStorage::GlobalSemaphore &&
           "fabric readiness requires a global counter");
    FuncOp func = sendOp->getParentOfType<FuncOp>();
    assert(func && "pipe transfer send must be inside a function");
    SmallVector<PipeCounterInfo> &counters = countersByFunc[func];
    appendUniquePipeCounter(counters, *resource.readyCounter);
  };

  LogicalResult traversalResult = pipeResourcePlan.forEachResourceTable(
      [&](Operation *protocolOp, ArrayRef<PipeResourceInfo> resources,
          PipeResourceTableKind) {
        for (const PipeResourceInfo &resource : resources) {
          appendFabricReadyCounter(protocolOp, resource);
        }
        return success();
      });
  assert(succeeded(traversalResult) && "infallible resource traversal failed");

  for (auto &[func, counters] : countersByFunc) {
    fabricReadyCounters[func] =
        buildZeroInitializedCounterTable(func, std::move(counters));
  }
}

void initializePipeComputedAddressCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeComputedAddressCounterMap &computedAddressCounters) {
  for (const auto &initializationEntry :
       pipeResourcePlan.computedAddressCounterInitializations) {
    func::FuncOp func = initializationEntry.first;
    const SmallVector<PipeComputedAddressCounterInitInfo> &initializations =
        initializationEntry.second;
    SmallVector<PipeComputedAddressCounterInitInfo> sortedInitializations(
        initializations);
    llvm::sort(sortedInitializations,
               [](const PipeComputedAddressCounterInitInfo &lhs,
                  const PipeComputedAddressCounterInitInfo &rhs) {
                 return lhs.counterIndex < rhs.counterIndex;
               });

    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto counterMemrefTy =
        MemRefType::get({static_cast<int64_t>(sortedInitializations.size())},
                        builder.getI32Type());
    Value counters = memref::AllocaOp::create(builder, loc, counterMemrefTy);
    for (const PipeComputedAddressCounterInitInfo &init :
         sortedInitializations) {
      Value counterIndex =
          arith::ConstantIndexOp::create(builder, loc, init.counterIndex);
      Value initialSlot =
          arith::ConstantIntOp::create(builder, loc, init.initialSlot, 32);
      memref::StoreOp::create(builder, loc, initialSlot, counters,
                              ValueRange{counterIndex});
    }
    computedAddressCounters[func] = counters;
  }
}

void initializePipeTransportSlotCounters(
    const PipeTransportPlan &pipeTransportPlan,
    PipeTransportSlotCounterMap &slotCounters) {
  for (const auto &entry : pipeTransportPlan.getSlotCounterInitializations()) {
    FuncOp func = entry.first;
    SmallVector<PipeTransportSlotCounterInitInfo> sortedInitializations(
        entry.second);
    llvm::sort(sortedInitializations,
               [](const PipeTransportSlotCounterInitInfo &lhs,
                  const PipeTransportSlotCounterInitInfo &rhs) {
                 return lhs.counterIndex < rhs.counterIndex;
               });

    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto counterMemrefType = MemRefType::get({1}, builder.getI32Type());
    Value zeroIndex = arith::ConstantIndexOp::create(builder, loc, 0);
    auto &perFuncCounters = slotCounters[func];
    for (const PipeTransportSlotCounterInitInfo &init : sortedInitializations) {
      auto counter = memref::AllocaOp::create(builder, loc, counterMemrefType);
      Value initialSlot =
          arith::ConstantIntOp::create(builder, loc, init.initialSlot, 32);
      memref::StoreOp::create(builder, loc, initialSlot, counter,
                              ValueRange{zeroIndex});
      perFuncCounters[init.counterIndex] = counter.getResult();
    }
  }
}

Value lookupPipeTransportSlotCounter(
    Operation *operation, int64_t counterIndex,
    const PipeTransportSlotCounterMap &slotCounters) {
  FuncOp func = operation->getParentOfType<FuncOp>();
  assert(func && "transport storage operation must be inside a function");
  auto funcIt = slotCounters.find(func);
  assert(funcIt != slotCounters.end() &&
         "function is missing transport storage slot counters");
  auto counterIt = funcIt->second.find(counterIndex);
  assert(counterIt != funcIt->second.end() &&
         "transport storage slot counter is missing");
  return counterIt->second;
}

static FailureOr<PipeCounterProgress>
lookupPipeCounterProgress(const PipeCounterProgressMap &progress, FuncOp func,
                          PipeCounterInfo counter) {
  auto funcIt = progress.find(func);
  if (funcIt == progress.end()) {
    return failure();
  }
  auto progressIt =
      llvm::find_if(funcIt->second, [&](const PipeCounterProgress &entry) {
        return entry.counter == counter;
      });
  if (progressIt == funcIt->second.end()) {
    return failure();
  }
  return *progressIt;
}

struct PipeCounterTableEntry {
  Value values;
  std::size_t index = 0;
};

static FailureOr<PipeCounterTableEntry>
lookupPipeCounterTableEntry(const PipeCounterTableMap &tables, FuncOp func,
                            PipeCounterInfo counter) {
  auto tableIt = tables.find(func);
  if (tableIt == tables.end()) {
    return failure();
  }
  auto counterIt = llvm::find(tableIt->second.counters, counter);
  if (counterIt == tableIt->second.counters.end()) {
    return failure();
  }
  return PipeCounterTableEntry{
      tableIt->second.values,
      static_cast<std::size_t>(
          std::distance(tableIt->second.counters.begin(), counterIt))};
}

static Value buildSelectedCounterTableIndex(
    Location loc, ArrayRef<PipeCounterInfo> recordCounters,
    const PipeCounterTable &counterTable, Value recordIndex,
    ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t> counterIndices;
  counterIndices.reserve(recordCounters.size());
  for (PipeCounterInfo counter : recordCounters) {
    auto counterIt = llvm::find(counterTable.counters, counter);
    assert(counterIt != counterTable.counters.end() &&
           "selected counter is missing from its function table");
    counterIndices.push_back(
        std::distance(counterTable.counters.begin(), counterIt));
  }
  return loadIndexTableEntry(loc, counterIndices, recordIndex, rewriter);
}

/// Assign a completion sequence when posting the receive. Tokens may be stored
/// or reordered, so each token must retain the sequence of its own post.
static Value incrementPipePostSequence(Location loc, Value sequenceCounter,
                                       Value sequenceIndex,
                                       ConversionPatternRewriter &rewriter) {
  Value previousSequence = memref::LoadOp::create(
      rewriter, loc, sequenceCounter, ValueRange{sequenceIndex});
  Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
  Value tokenSequence =
      arith::AddIOp::create(rewriter, loc, previousSequence, one);
  memref::StoreOp::create(rewriter, loc, tokenSequence, sequenceCounter,
                          ValueRange{sequenceIndex});
  return tokenSequence;
}

static LogicalResult lowerSelectedPipeTransferSend(
    PipeTransferSendOp op, Value srcCB, const PipeTransferPlan &transferPlan,
    const PipeResourcePlan &pipeResourcePlan,
    const PipeCounterTableMap &fabricReadyCounters,
    const PipeComputedAddressCounterMap &computedAddressCounters,
    const FabricRuntimeMap &fabricRuntime,
    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  const PipeResourceAccessPlan &resourceAccessPlan =
      transferPlan.getResourceAccessPlan();
  const PipeReference &pipeRef = resourceAccessPlan.getPipeReference();
  SelectedPipeFields fields = getSelectedPipeFields(pipeRef);
  ArrayRef<PipeResourceInfo> resources =
      resourceAccessPlan.getSelectedResources();
  const PipeSendPlan &sendPlan = transferPlan.getSend();
  bool usesFabric = transferPlan.getSynchronizationProtocol() ==
                    PipeSynchronizationProtocol::Fabric;
  assert(usesFabric == !sendPlan.fabricRouteIndices.empty() &&
         "selected fabric transfer plan is missing its routes");
  assert(
      (!usesFabric || sendPlan.fabricRouteIndices.size() == resources.size()) &&
      "selected fabric route table must match the resource table");

  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  std::unique_ptr<PipeSendTransportEmitter> transport;
  if (usesFabric) {
    auto runtimeIt = fabricRuntime.find(op.getOperation());
    if (runtimeIt == fabricRuntime.end()) {
      op.emitError("fabric pipe transfer has no initialized routing-plane "
                   "runtime state");
      return failure();
    }
    if (llvm::any_of(sendPlan.fabricRouteIndices, [&](std::size_t routeIndex) {
          return routeIndex >= runtimeIt->second.routeCount;
        })) {
      op.emitError("fabric pipe transfer route index exceeds the initialized "
                   "routing-plane targets");
      return failure();
    }
    Value routeIndex = buildSelectedRouteIndex(loc, sendPlan.fabricRouteIndices,
                                               fields.recordIndex, rewriter);
    transport = std::make_unique<FabricPipeTransportEmitter>(
        op, fields.dstStartX, fields.dstStartY, routeIndex, runtimeIt->second,
        rewriter);
  } else {
    transport =
        std::make_unique<SelectedNocPipeTransportEmitter>(op, fields, rewriter);
  }

  Value senderSemAddr = buildSelectedReadyCounterAddress(
      op, loc, resources, fields.recordIndex, pipeResourcePlan, rewriter);
  Value expectedSignals;
  if (fields.isCollective) {
    expectedSignals = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(), fields.numDests);
  } else {
    expectedSignals = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
  }
  if (usesFabric) {
    FuncOp func = op->getParentOfType<FuncOp>();
    auto readyTableIt = fabricReadyCounters.find(func);
    assert(readyTableIt != fabricReadyCounters.end() &&
           "selected fabric sender is missing its readiness table");
    SmallVector<PipeCounterInfo> readyCounters;
    readyCounters.reserve(resources.size());
    for (const PipeResourceInfo &resource : resources) {
      assert(resource.readyCounter &&
             resource.readyCounter->getStorage() ==
                 PipeCounterStorage::GlobalSemaphore &&
             "selected fabric readiness requires global counters");
      readyCounters.push_back(*resource.readyCounter);
    }
    Value readyIndex = buildSelectedCounterTableIndex(
        loc, readyCounters, readyTableIt->second, fields.recordIndex, rewriter);
    Value previousReady = memref::LoadOp::create(
        rewriter, loc, readyTableIt->second.values, ValueRange{readyIndex});
    Value expectedReady =
        arith::AddIOp::create(rewriter, loc, previousReady, expectedSignals);
    memref::StoreOp::create(rewriter, loc, expectedReady,
                            readyTableIt->second.values,
                            ValueRange{readyIndex});
    auto senderSemPtr =
        ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, senderSemAddr);
    ttk::SemaphoreWaitMinOp::create(rewriter, loc, senderSemPtr, expectedReady);
  } else {
    auto senderSemPtr =
        ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, senderSemAddr);
    ttk::SemaphoreWaitOp::create(rewriter, loc, senderSemPtr, expectedSignals);
    auto zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    ttk::NocSemaphoreSetOp::create(rewriter, loc, senderSemPtr, zeroIndex);
  }

  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  assert(succeeded(cbConverted) && "preflight checked source DFB type");
  Value srcPtrIdx;
  if (sendPlan.usesReadPointer) {
    auto cbReadPtr = ttk::GetReadPtrOp::create(rewriter, loc, *cbConverted);
    srcPtrIdx = arith::IndexCastOp::create(rewriter, loc,
                                           rewriter.getIndexType(), cbReadPtr);
  } else {
    auto srcWritePtr = ttk::GetWritePtrOp::create(rewriter, loc, *cbConverted);
    srcPtrIdx = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), srcWritePtr);
  }

  Value srcAddr = arith::IndexCastOp::create(rewriter, loc,
                                             rewriter.getI32Type(), srcPtrIdx);
  auto totalSizeVal = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(sendPlan.payloadSizeBytes));
  transport->preparePayloadWrite();

  Value dstAddr;
  bool allUseComputedAddress =
      llvm::all_of(resources, [](const PipeResourceInfo &resource) {
        return resource.addressStorage.usesComputedReceiverDFB();
      });
  bool anyUseComputedAddress =
      llvm::any_of(resources, [](const PipeResourceInfo &resource) {
        return resource.addressStorage.usesComputedReceiverDFB();
      });
  if (usesFabric || allUseComputedAddress) {
    dstAddr = buildSelectedComputedReceiverDFBDestinationAddress(
        op, loc, resources, fields.recordIndex, computedAddressCounters,
        rewriter);
  } else if (!anyUseComputedAddress) {
    Value tableAddress = buildSelectedAddressTableAddress(
        op, loc, resources, fields.recordIndex, rewriter);
    dstAddr = buildAddressTableDestinationAddress(loc, tableAddress, rewriter);
  } else {
    SmallVector<PipeAddressMode> addressModes =
        llvm::map_to_vector(resources, [](const PipeResourceInfo &resource) {
          return resource.addressStorage.mode;
        });
    Value usesComputedAddress = buildSelectedUsesComputedReceiverDFB(
        loc, addressModes, fields.recordIndex, rewriter);
    auto addressSelection = scf::IfOp::create(
        rewriter, loc, TypeRange{rewriter.getI32Type()}, usesComputedAddress,
        /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(
          &addressSelection.getThenRegion().front());
      Value computedAddress =
          buildSelectedComputedReceiverDFBDestinationAddress(
              op, loc, resources, fields.recordIndex, computedAddressCounters,
              rewriter);
      scf::YieldOp::create(rewriter, loc, computedAddress);

      rewriter.setInsertionPointToStart(
          &addressSelection.getElseRegion().front());
      Value tableAddress = buildSelectedAddressTableAddress(
          op, loc, resources, fields.recordIndex, rewriter);
      Value publishedAddress =
          buildAddressTableDestinationAddress(loc, tableAddress, rewriter);
      scf::YieldOp::create(rewriter, loc, publishedAddress);
    }
    rewriter.setInsertionPointAfter(addressSelection);
    dstAddr = addressSelection.getResult(0);
  }
  if (failed(transport->emitPayloadWrite(srcAddr, dstAddr, totalSizeVal))) {
    return failure();
  }
  transport->emitPayloadWriteBarrier();

  SmallVector<PipeCounterInfo> completionCounters =
      llvm::map_to_vector(resources, [](const PipeResourceInfo &resource) {
        return resource.completion.counter;
      });
  Value completionCounterAddress = buildSelectedPipeCounterAddress(
      op, loc, completionCounters, fields.recordIndex, pipeResourcePlan,
      rewriter);
  if (failed(transport->emitReceiverCompletionIncrement(
          completionCounterAddress))) {
    return failure();
  }
  transport->emitCompletionSignalBarrier();

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

/// Return whether an overlapped payload requires page-granular NoC writes.
static bool
shouldEmitPayloadPageWrites(PipeTransferSendOp op, PipeType pipeType,
                            const PipeTransportStream &transportStream) {
  const PipeTransportPacketization &packetization =
      transportStream.getPacketization();
  int64_t maxBurstBytes = getTargetNocMaxBurstBytes(op);
  return transportStream.getSchedule() == PipeTransportSchedule::Overlapped &&
         pipeType.hasSingleReceiver() && packetization.pageCount > 1 &&
         packetization.pageSizeBytes <= maxBurstBytes &&
         packetization.getPayloadSizeBytes() > maxBurstBytes;
}

void lowerInactivePipeTransferSend(PipeTransferSendOp op,
                                   ConversionPatternRewriter &rewriter) {
  rewriter.replaceOp(op, makeZeroI32(op.getLoc(), rewriter));
}

LogicalResult lowerPipeTransferSend(
    PipeTransferSendOp op, Value srcCB, const PipeTransferPlan &transferPlan,
    const PipeTransportPlan &pipeTransportPlan,
    const PipeResourcePlan &pipeResourcePlan,
    const PipeCapacityPlan &pipeCapacityPlan,
    const PipeCounterProgressMap &senderCapacityCounters,
    const PipeCounterTableMap &fabricReadyCounters,
    const PipeComputedAddressCounterMap &computedAddressCounters,
    const FabricRuntimeMap &fabricRuntime,
    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  assert(!pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation()) &&
         "inactive sender must not use an active transfer plan");
  assert(transferPlan.isSend() && "sender operation has a non-send plan");
  const PipeResourceAccessPlan &resourceAccessPlan =
      transferPlan.getResourceAccessPlan();
  const PipeReference &pipeRef = resourceAccessPlan.getPipeReference();
  if (pipeRef.isSelected()) {
    return lowerSelectedPipeTransferSend(
        op, srcCB, transferPlan, pipeResourcePlan, fabricReadyCounters,
        computedAddressCounters, fabricRuntime, rewriter);
  }
  const PipeTransportStream &transportStream =
      pipeTransportPlan.getStreamForOperation(op);
  PipeType pipeType = pipeRef.getStaticPipeType();
  const PipeResourceInfo &pipeResource = resourceAccessPlan.getResources();
  const PipeSendPlan &sendPlan = transferPlan.getSend();
  const PipeTransportPacketization &packetization =
      transportStream.getPacketization();
  assert(sendPlan.payloadSizeBytes == packetization.getPayloadSizeBytes() &&
         "transport and send plans disagree on payload size");
  PipeCompletionInfo completionInfo = pipeResource.completion;
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);

  bool usesFabric = transferPlan.getSynchronizationProtocol() ==
                    PipeSynchronizationProtocol::Fabric;
  assert(usesFabric == !sendPlan.fabricRouteIndices.empty() &&
         "fabric transfer plan is missing its route");
  assert((!usesFabric || sendPlan.fabricRouteIndices.size() == 1) &&
         "static fabric send must have one route");
  assert(
      (!usesFabric || pipeResource.addressStorage.usesComputedReceiverDFB()) &&
      "fabric transfer plan uses a receiver-published address");

  bool usesCapacityProtocol = transferPlan.getSynchronizationProtocol() ==
                              PipeSynchronizationProtocol::Capacity;
  ArrayRef<PipeCapacityAcquireInfo> capacityAcquires =
      pipeCapacityPlan.lookupAcquires(op);
  assert(usesCapacityProtocol == !capacityAcquires.empty() &&
         "capacity-protocol send must have at least one capacity acquire");
  FuncOp senderFunc = op->getParentOfType<FuncOp>();
  assert(senderFunc && "pipe transfer send must be inside a function");
  SmallVector<Value> capacityCounters;
  if (!capacityAcquires.empty()) {
    for (const PipeCapacityAcquireInfo &capacityAcquire : capacityAcquires) {
      FailureOr<PipeCounterProgress> maybeCounter = lookupPipeCounterProgress(
          senderCapacityCounters, senderFunc, capacityAcquire.counter);
      if (failed(maybeCounter)) {
        op.emitError("pipe capacity acquire without sender counter; "
                     "initializePipeCapacityCounters must run before "
                     "convert-ttl-to-ttkernel");
        return failure();
      }
      capacityCounters.push_back(maybeCounter->value);
    }
  }

  std::unique_ptr<PipeSendTransportEmitter> transport;
  NocPipeTransportEmitter *nocTransport = nullptr;
  if (usesFabric) {
    auto runtimeIt = fabricRuntime.find(op.getOperation());
    if (runtimeIt == fabricRuntime.end()) {
      op.emitError("fabric pipe transfer has no initialized routing-plane "
                   "runtime state");
      return failure();
    }
    std::size_t routeIndex = sendPlan.fabricRouteIndices.front();
    if (routeIndex >= runtimeIt->second.routeCount) {
      op.emitError("fabric pipe transfer route index exceeds the initialized "
                   "routing-plane targets");
      return failure();
    }
    Value destinationX =
        arith::ConstantIndexOp::create(rewriter, loc, pipeType.getDstStartX());
    Value destinationY =
        arith::ConstantIndexOp::create(rewriter, loc, pipeType.getDstStartY());
    Value routeIndexValue =
        arith::ConstantIndexOp::create(rewriter, loc, routeIndex);
    transport = std::make_unique<FabricPipeTransportEmitter>(
        op, destinationX, destinationY, routeIndexValue, runtimeIt->second,
        rewriter);
  } else {
    auto emitter =
        std::make_unique<NocPipeTransportEmitter>(op, pipeType, rewriter);
    nocTransport = emitter.get();
    transport = std::move(emitter);
  }
  int64_t numDests = pipeType.getNumDests();

  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();

  if (usesCapacityProtocol) {
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (auto [capacityAcquire, senderCapacityCounter] :
         llvm::zip_equal(capacityAcquires, capacityCounters)) {
      Value capacityCounterPtr = buildPipeCounterPtr(
          loc, senderFunc, capacityAcquire.counter, pipeResourcePlan, rewriter);
      // Advance the sender's cumulative acquired count and block until the
      // shared capacity counter reaches it. The receiver's remote increment is
      // the only writer, so the acquire never writes the shared counter.
      Value previousAcquired = memref::LoadOp::create(
          rewriter, loc, senderCapacityCounter, ValueRange{zeroIdx});
      Value capacityCount = arith::ConstantIntOp::create(
          rewriter, loc, capacityAcquire.count, 32);
      Value nextAcquired =
          arith::AddIOp::create(rewriter, loc, previousAcquired, capacityCount);
      memref::StoreOp::create(rewriter, loc, nextAcquired,
                              senderCapacityCounter, ValueRange{zeroIdx});
      ttk::SemaphoreWaitMinOp::create(rewriter, loc, capacityCounterPtr,
                                      nextAcquired);
    }
  } else if (usesFabric) {
    assert(pipeResource.readyCounter &&
           "fabric sender is missing its readiness counter");
    FailureOr<PipeCounterTableEntry> maybeReadyProgress =
        lookupPipeCounterTableEntry(fabricReadyCounters, senderFunc,
                                    *pipeResource.readyCounter);
    if (failed(maybeReadyProgress)) {
      op.emitError("fabric pipe send has no cumulative readiness counter");
      return failure();
    }
    int64_t expectedReceiverPosts =
        isCollectiveTransfer(pipeResource.transferContract) ? numDests : 1;
    Value readyIndex = arith::ConstantIndexOp::create(
        rewriter, loc, maybeReadyProgress->index);
    Value previousReady = memref::LoadOp::create(
        rewriter, loc, maybeReadyProgress->values, ValueRange{readyIndex});
    Value readyIncrement =
        arith::ConstantIntOp::create(rewriter, loc, expectedReceiverPosts, 32);
    Value expectedReady =
        arith::AddIOp::create(rewriter, loc, previousReady, readyIncrement);
    memref::StoreOp::create(rewriter, loc, expectedReady,
                            maybeReadyProgress->values, ValueRange{readyIndex});
    Value readyCounterPtr =
        buildPipeCounterPtr(loc, senderFunc, *pipeResource.readyCounter,
                            pipeResourcePlan, rewriter);
    ttk::SemaphoreWaitMinOp::create(rewriter, loc, readyCounterPtr,
                                    expectedReady);
  } else {
    assert(pipeResource.readyCounter &&
           "sender-ready protocol selected without a sender-ready counter");
    int64_t expectedReceiverPosts =
        isCollectiveTransfer(pipeResource.transferContract) ? numDests : 1;
    Value senderReadyCounterAddr =
        buildPipeCounterAddress(loc, senderFunc, *pipeResource.readyCounter,
                                pipeResourcePlan, rewriter);
    auto senderReadyCounterPtr = ttk::CastToL1PtrOp::create(
        rewriter, loc, l1PtrTy, senderReadyCounterAddr);
    auto expectedReadyCount = arith::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(expectedReceiverPosts));
    ttk::SemaphoreWaitOp::create(rewriter, loc, senderReadyCounterPtr,
                                 expectedReadyCount);
    auto readyCounterResetValue =
        arith::ConstantIndexOp::create(rewriter, loc, 0);
    ttk::NocSemaphoreSetOp::create(rewriter, loc, senderReadyCounterPtr,
                                   readyCounterResetValue);
  }

  Value srcPtrIdx;
  if (transportStream.getSourceStorage().ownership ==
      PipeTransportStorageOwnership::Transport) {
    Value scratchAddress = buildPipeSramScratchAddress(
        op, transportStream.getSourceStorage().scratchByteOffset, rewriter);
    srcPtrIdx =
        arith::IndexCastOp::create(rewriter, loc, indexTy, scratchAddress);
  } else {
    auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
    assert(succeeded(cbConverted) && "preflight checked source DFB type");
    // A producer stages into the write pointer before publication. A consumer
    // sends from the read pointer after waiting for publication.
    if (sendPlan.usesReadPointer) {
      auto cbReadPtr = ttk::GetReadPtrOp::create(rewriter, loc, *cbConverted);
      srcPtrIdx = arith::IndexCastOp::create(rewriter, loc, indexTy, cbReadPtr);
    } else {
      auto srcWritePtr =
          ttk::GetWritePtrOp::create(rewriter, loc, *cbConverted);
      srcPtrIdx =
          arith::IndexCastOp::create(rewriter, loc, indexTy, srcWritePtr);
    }
  }
  transport->preparePayloadWrite();

  // Transfer the entire block in one NoC write. Tiles are contiguous in the
  // DFB, and destination DFB layout is uniform across nodes, so lowering sends
  // all tiles at once instead of one per tile.
  auto totalSizeVal = arith::ConstantOp::create(
      rewriter, loc, i32Ty,
      rewriter.getI32IntegerAttr(sendPlan.payloadSizeBytes));

  Value srcAddr = arith::IndexCastOp::create(rewriter, loc, i32Ty, srcPtrIdx);

  Value dstAddr;
  if (pipeResource.addressStorage.usesComputedReceiverAddress()) {
    dstAddr = buildComputedReceiverDFBDestinationAddress(
        op, loc, pipeResource.addressStorage, computedAddressCounters,
        rewriter);
  } else {
    AddressTableInfo addressTableInfo = getAddressTableInfo(op, pipeResource);
    dstAddr =
        buildAddressTableDestinationAddress(loc, addressTableInfo, rewriter);
  }

  bool usePageWrites =
      !usesFabric && shouldEmitPayloadPageWrites(op, pipeType, transportStream);
  LogicalResult writeResult =
      usePageWrites
          ? nocTransport->emitPayloadPageWrites(srcAddr, dstAddr,
                                                packetization.pageCount,
                                                packetization.pageSizeBytes)
          : transport->emitPayloadWrite(srcAddr, dstAddr, totalSizeVal);
  if (failed(writeResult)) {
    return failure();
  }

  // Wait for payload writes to complete before signaling receiver completion.
  // Without this barrier, the receiver may wake up before all data arrives.
  Value receiverCompletionCounterAddr = buildPipeCounterAddress(
      loc, senderFunc, completionInfo.counter, pipeResourcePlan, rewriter);
  transport->emitPayloadWriteBarrier();

  if (failed(transport->emitReceiverCompletionIncrement(
          receiverCompletionCounterAddr))) {
    return failure();
  }

  if (transportStream.getCreditCompletion() ==
      PipeTransportCreditCompletion::Immediate) {
    transport->emitCompletionSignalBarrier();
  }

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

void lowerInactivePipeTransferPost(PipeTransferPostOp op,
                                   ConversionPatternRewriter &rewriter) {
  auto token = UnrealizedConversionCastOp::create(
      rewriter, op.getLoc(), op.getToken().getType(), ValueRange{});
  rewriter.replaceOp(op, token.getResult(0));
}

static LogicalResult
lowerSelectedPipeTransferPost(PipeTransferPostOp op, Value dst,
                              const PipeTransferPlan &transferPlan,
                              const PipeCounterTableMap &postSequenceCounters,
                              const PipeResourcePlan &pipeResourcePlan,
                              const FabricRuntimeMap &fabricRuntime,
                              ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  const PipeResourceAccessPlan &resourceAccessPlan =
      transferPlan.getResourceAccessPlan();
  const PipeReference &pipeRef = resourceAccessPlan.getPipeReference();
  SelectedPipeFields fields = getSelectedPipeFields(pipeRef);
  ArrayRef<PipeResourceInfo> resources =
      resourceAccessPlan.getSelectedResources();
  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "pipe transfer post must be inside a function");
  auto sequenceIt = postSequenceCounters.find(func);
  if (sequenceIt == postSequenceCounters.end()) {
    op.emitError(
        "table-driven pipe receive has no completion sequence counters");
    return failure();
  }
  SmallVector<PipeCounterInfo> completionCounters =
      llvm::map_to_vector(resources, [](const PipeResourceInfo &resource) {
        return resource.completion.counter;
      });
  const PipePostPlan &postPlan = transferPlan.getPost();
  bool usesFabric = transferPlan.getSynchronizationProtocol() ==
                    PipeSynchronizationProtocol::Fabric;
  assert(usesFabric == !postPlan.fabricRouteIndices.empty() &&
         "selected fabric post plan is missing its routes");
  assert(
      (!usesFabric || postPlan.fabricRouteIndices.size() == resources.size()) &&
      "selected fabric route table must match the resource table");
  assert(postPlan.addressModes.size() == resources.size() &&
         "selected post address modes must match the resource table");
  bool anyUsePublishedAddress = llvm::is_contained(
      postPlan.addressModes, PipeAddressMode::ReceiverPublishedAddressTable);
  bool allUseComputedAddress =
      llvm::all_of(postPlan.addressModes, [](PipeAddressMode mode) {
        return mode == PipeAddressMode::ComputedReceiverDFB;
      });
  bool anyUseComputedAddress = llvm::is_contained(
      postPlan.addressModes, PipeAddressMode::ComputedReceiverDFB);
  assert(anyUsePublishedAddress == postPlan.addressPublication.has_value() &&
         "selected post address publication plan does not match its records");
  if (usesFabric && !allUseComputedAddress) {
    op.emitError(
        "fabric pipe transfer requires a computed receiver DFB address");
    return failure();
  }

  const FabricRuntimeInfo *fabricRuntimeInfo = nullptr;
  if (usesFabric) {
    auto runtimeIt = fabricRuntime.find(op.getOperation());
    if (runtimeIt == fabricRuntime.end()) {
      op.emitError("fabric pipe receiver has no initialized routing-plane "
                   "runtime state");
      return failure();
    }
    if (llvm::any_of(postPlan.fabricRouteIndices, [&](std::size_t routeIndex) {
          return routeIndex >= runtimeIt->second.routeCount;
        })) {
      op.emitError("fabric pipe reverse route index exceeds the initialized "
                   "routing-plane targets");
      return failure();
    }
    fabricRuntimeInfo = &runtimeIt->second;
  }

  Value senderSemAddr = buildSelectedReadyCounterAddress(
      op, loc, resources, fields.recordIndex, pipeResourcePlan, rewriter);
  if (usesFabric) {
    Value routeIndex = buildSelectedRouteIndex(loc, postPlan.fabricRouteIndices,
                                               fields.recordIndex, rewriter);
    FabricRouteEmitter routeEmitter(op, routeIndex, *fabricRuntimeInfo,
                                    rewriter);
    routeEmitter.emitAtomicIncrement(
        fields.srcX, fields.srcY, senderSemAddr,
        arith::ConstantIntOp::create(rewriter, loc, 1, 32));
  } else {
    SelectedNocPipeTransportEmitter transport(op, fields, rewriter);
    if (anyUsePublishedAddress) {
      auto emitAddressPublication = [&]() -> LogicalResult {
        Value publishedAddress = buildReceiverPublishedAddress(
            dst, loc, *postPlan.addressPublication, rewriter);
        Value tableAddress = buildSelectedAddressTableAddress(
            op, loc, resources, fields.recordIndex, rewriter);
        if (failed(transport.emitReceiverAddressPublish(tableAddress,
                                                        publishedAddress))) {
          return failure();
        }
        transport.emitAddressPublishBarrier();
        return success();
      };

      if (!anyUseComputedAddress) {
        if (failed(emitAddressPublication())) {
          return failure();
        }
      } else {
        Value usesComputedAddress = buildSelectedUsesComputedReceiverDFB(
            loc, postPlan.addressModes, fields.recordIndex, rewriter);
        Value shouldPublishAddress = arith::XOrIOp::create(
            rewriter, loc, usesComputedAddress,
            arith::ConstantIntOp::create(rewriter, loc, 1, 1));
        auto publishAddress =
            scf::IfOp::create(rewriter, loc, shouldPublishAddress,
                              /*withElseRegion=*/false);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(
              &publishAddress.getThenRegion().front());
          if (failed(emitAddressPublication())) {
            return failure();
          }
        }
        rewriter.setInsertionPointAfter(publishAddress);
      }
    }
    if (failed(transport.emitSenderReadyIncrement(senderSemAddr))) {
      return failure();
    }
  }

  Value sequenceIndex = buildSelectedCounterTableIndex(
      loc, completionCounters, sequenceIt->second, fields.recordIndex,
      rewriter);
  Value tokenSequence = incrementPipePostSequence(
      loc, sequenceIt->second.values, sequenceIndex, rewriter);
  rewriter.replaceOp(op, tokenSequence);
  return success();
}

LogicalResult
lowerPipeTransferPost(PipeTransferPostOp op, Value dst,
                      const PipeTransferPlan &transferPlan,
                      const PipeCounterTableMap &postSequenceCounters,
                      const PipeResourcePlan &pipeResourcePlan,
                      const FabricRuntimeMap &fabricRuntime,
                      ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  assert(!pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation()) &&
         "inactive receiver post must not use an active transfer plan");
  assert(transferPlan.isPost() && "receiver post has another operation plan");
  const PipeResourceAccessPlan &resourceAccessPlan =
      transferPlan.getResourceAccessPlan();
  const PipeReference &pipeRef = resourceAccessPlan.getPipeReference();
  if (pipeRef.isSelected()) {
    return lowerSelectedPipeTransferPost(op, dst, transferPlan,
                                         postSequenceCounters, pipeResourcePlan,
                                         fabricRuntime, rewriter);
  }
  PipeType pipeType = pipeRef.getStaticPipeType();
  const PipeResourceInfo &pipeResource = resourceAccessPlan.getResources();
  const PipePostPlan &postPlan = transferPlan.getPost();
  auto func = op->getParentOfType<func::FuncOp>();
  assert(func && "pipe transfer post must be inside a function");
  FailureOr<PipeCounterTableEntry> maybeSequenceCounter =
      lookupPipeCounterTableEntry(postSequenceCounters, func,
                                  pipeResource.completion.counter);
  if (failed(maybeSequenceCounter)) {
    op.emitError("pipe receive post has no sequence counter for its completion "
                 "counter");
    return failure();
  }
  Value sequenceCounter = maybeSequenceCounter->values;

  bool usesFabric = transferPlan.getSynchronizationProtocol() ==
                    PipeSynchronizationProtocol::Fabric;
  assert(usesFabric == !postPlan.fabricRouteIndices.empty() &&
         "fabric receiver post plan is missing its reverse route");
  assert((!usesFabric || postPlan.fabricRouteIndices.size() == 1) &&
         "static fabric receiver post must have one route");
  assert(postPlan.addressModes.size() == 1 &&
         "static receiver post must have one address mode");
  if (usesFabric && postPlan.addressPublication) {
    op.emitError(
        "fabric pipe transfer requires a computed receiver DFB address");
    return failure();
  }

  if (usesFabric) {
    assert(pipeResource.readyCounter &&
           pipeResource.readyCounter->getStorage() ==
               PipeCounterStorage::GlobalSemaphore &&
           "fabric receiver post requires a global readiness counter");
    auto runtimeIt = fabricRuntime.find(op.getOperation());
    if (runtimeIt == fabricRuntime.end()) {
      op.emitError("fabric pipe receiver has no initialized routing-plane "
                   "runtime state");
      return failure();
    }
    std::size_t routeIndex = postPlan.fabricRouteIndices.front();
    if (routeIndex >= runtimeIt->second.routeCount) {
      op.emitError("fabric pipe reverse route index exceeds the initialized "
                   "routing-plane targets");
      return failure();
    }
    Value senderReadyCounterAddress = buildPipeCounterAddress(
        loc, func, *pipeResource.readyCounter, pipeResourcePlan, rewriter);
    Value routeIndexValue =
        arith::ConstantIndexOp::create(rewriter, loc, routeIndex);
    Value sourceX =
        arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX());
    Value sourceY =
        arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY());
    FabricRouteEmitter routeEmitter(op, routeIndexValue, runtimeIt->second,
                                    rewriter);
    routeEmitter.emitAtomicIncrement(
        sourceX, sourceY, senderReadyCounterAddress,
        arith::ConstantIntOp::create(rewriter, loc, 1, 32));
  } else {
    NocPipeTransportEmitter transport(op, pipeType, rewriter);
    if (postPlan.addressPublication) {
      AddressTableInfo addressTableInfo = getAddressTableInfo(op, pipeResource);
      Value publishedAddress = buildReceiverPublishedAddress(
          dst, loc, *postPlan.addressPublication, rewriter);
      Value tableAddress =
          buildAddressTableAddress(loc, addressTableInfo, rewriter);
      if (failed(transport.emitReceiverAddressPublish(tableAddress,
                                                      publishedAddress))) {
        return failure();
      }
      transport.emitAddressPublishBarrier();
    }

    if (transferPlan.getSynchronizationProtocol() ==
        PipeSynchronizationProtocol::ReceiverPost) {
      assert(pipeResource.readyCounter &&
             "sender-ready protocol selected without a sender-ready counter");
      Value senderReadyCounterAddr = buildPipeCounterAddress(
          loc, func, *pipeResource.readyCounter, pipeResourcePlan, rewriter);
      if (failed(transport.emitSenderReadyIncrement(senderReadyCounterAddr))) {
        return failure();
      }
    }
  }

  Value sequenceIndex = arith::ConstantIndexOp::create(
      rewriter, loc, maybeSequenceCounter->index);
  Value tokenSequence =
      incrementPipePostSequence(loc, sequenceCounter, sequenceIndex, rewriter);
  rewriter.replaceOp(op, tokenSequence);
  return success();
}

static Value computeDFBPopNumTiles(CBPopOp op, CircularBufferType dfbType,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc) {
  if (auto attr = op.getNumTilesAttr()) {
    return arith::ConstantIntOp::create(rewriter, loc, attr.getInt(), 32);
  }
  return arith::ConstantIntOp::create(rewriter, loc,
                                      dfbType.getElementsPerBlock(), 32);
}

LogicalResult lowerCBPop(CBPopOp op, Value cb,
                         const PipeCapacityPlan &pipeCapacityPlan,
                         const PipeTransportPlan &pipeTransportPlan,
                         const PipeTransportSlotCounterMap &slotCounters,
                         const PipeResourcePlan &pipeResourcePlan,
                         ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  if (!pipeTransportPlan.ownsDFBLifecycle(op.getOperation())) {
    Value originalCb = op.getCb();
    FailureOr<CircularBufferType> maybeDFBType =
        utils::getTTLCircularBufferType(originalCb);
    if (failed(maybeDFBType)) {
      return rewriter.notifyMatchFailure(op, "failed to get TTL DFB type");
    }

    auto convertedCb = utils::convertTTLCBToTTKernel(cb, rewriter, loc);
    if (failed(convertedCb)) {
      return rewriter.notifyMatchFailure(op, "failed to convert DFB operand");
    }

    Value numTiles = computeDFBPopNumTiles(op, *maybeDFBType, rewriter, loc);
    ttk::CBPopFrontOp::create(rewriter, loc, *convertedCb, numTiles);
  }

  const PipeTransportStorageAccess *storageAccess =
      pipeTransportPlan.lookupStorageAccess(op);
  if (storageAccess && storageAccess->dynamicSlotCounterIndex) {
    Value slotCounter = lookupPipeTransportSlotCounter(
        op, *storageAccess->dynamicSlotCounterIndex, slotCounters);
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value currentSlot = memref::LoadOp::create(rewriter, loc, slotCounter,
                                               ValueRange{zeroIndex});
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value blockCount = arith::ConstantIntOp::create(
        rewriter, loc, storageAccess->blockCount, 32);
    Value nextSlotUnwrapped =
        arith::AddIOp::create(rewriter, loc, currentSlot, one);
    Value nextSlot =
        arith::RemUIOp::create(rewriter, loc, nextSlotUnwrapped, blockCount);
    memref::StoreOp::create(rewriter, loc, nextSlot, slotCounter,
                            ValueRange{zeroIndex});
  }

  // The release preserves the pop's control dependence even when transport
  // synchronization replaces the local DFB state update.
  ArrayRef<PipeCapacityReleaseInfo> releases =
      pipeCapacityPlan.lookupReleases(op);
  if (!releases.empty()) {
    FuncOp func = op->getParentOfType<FuncOp>();
    assert(func && "DFB pop must be inside a function");
    int64_t nocIdx = getNocIndex(op);
    Value nocVal = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(nocIdx));
    for (const PipeCapacityReleaseInfo &release : releases) {
      lowerPipeCapacityRelease(loc, func, release, pipeResourcePlan, nocVal,
                               rewriter);
    }
    bool requiresImmediateCompletion =
        llvm::any_of(releases, [&](const PipeCapacityReleaseInfo &release) {
          return pipeTransportPlan.getStreamForTransfer(release.transferNode)
                     .getCreditCompletion() ==
                 PipeTransportCreditCompletion::Immediate;
        });
    if (requiresImmediateCompletion) {
      ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocVal);
    }
  }

  rewriter.eraseOp(op);
  return success();
}

/// Lower the receiver completion wait using the posted token's sequence.
LogicalResult lowerPipeTransferWait(PipeTransferWaitOp op, Value tokenSequence,
                                    const PipeTransferPlan &transferPlan,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter) {
  assert(!pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation()) &&
         "inactive receiver wait must not use an active transfer plan");
  assert(transferPlan.isWait() && "receiver wait has another operation plan");
  auto loc = op.getLoc();
  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "pipe transfer wait must be inside a function");
  const PipeResourceAccessPlan &resourceAccessPlan =
      transferPlan.getResourceAccessPlan();
  Value receiverCompletionCounterAddress;
  if (resourceAccessPlan.isSelected()) {
    SelectedPipeFields fields =
        getSelectedPipeFields(resourceAccessPlan.getPipeReference());
    SmallVector<PipeCounterInfo> completionCounters =
        llvm::map_to_vector(resourceAccessPlan.getSelectedResources(),
                            [](const PipeResourceInfo &resource) {
                              return resource.completion.counter;
                            });
    receiverCompletionCounterAddress = buildSelectedPipeCounterAddress(
        op, loc, completionCounters, fields.recordIndex, pipeResourcePlan,
        rewriter);
  } else {
    PipeCompletionInfo completionInfo =
        resourceAccessPlan.getResources().completion;
    receiverCompletionCounterAddress = buildPipeCounterAddress(
        loc, func, completionInfo.counter, pipeResourcePlan, rewriter);
  }

  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  // [Device 2.0] Completion waits should consume the allocated completion
  // object directly once device APIs expose typed semaphore waits.
  Value receiverCompletionCounterPtr = ttk::CastToL1PtrOp::create(
      rewriter, loc, l1PtrTy, receiverCompletionCounterAddress);
  ttk::SemaphoreWaitMinOp::create(rewriter, loc, receiverCompletionCounterPtr,
                                  tokenSequence);

  rewriter.eraseOp(op);
  return success();
}

static Value buildWaitAnyCompletionAddress(
    PipeTransferWaitAnyOp op, const PipeResourceAccessPlan &candidate,
    const PipeResourcePlan &pipeResourcePlan, OpBuilder &builder) {
  Location loc = op.getLoc();
  FuncOp function = op->getParentOfType<FuncOp>();
  assert(function && "pipe wait-any must be inside a function");
  if (!candidate.isSelected()) {
    return buildPipeCounterAddress(loc, function,
                                   candidate.getResources().completion.counter,
                                   pipeResourcePlan, builder);
  }
  SelectedPipeFields fields =
      getSelectedPipeFields(candidate.getPipeReference());
  SmallVector<PipeCounterInfo> completionCounters = llvm::map_to_vector(
      candidate.getSelectedResources(), [](const PipeResourceInfo &resource) {
        return resource.completion.counter;
      });
  return buildSelectedPipeCounterAddress(op, loc, completionCounters,
                                         fields.recordIndex, pipeResourcePlan,
                                         builder);
}

static Value buildWaitAnyCandidateReached(
    PipeTransferWaitAnyOp op, Value candidateIndex, ValueRange tokenSequences,
    const PipeWaitAnyPlan &waitAnyPlan,
    const PipeResourcePlan &pipeResourcePlan, OpBuilder &builder) {
  Location loc = op.getLoc();
  SmallVector<int64_t> cases;
  cases.reserve(tokenSequences.size());
  for (int64_t candidate = 0;
       candidate < static_cast<int64_t>(tokenSequences.size()); ++candidate) {
    cases.push_back(candidate);
  }
  auto switchOp =
      scf::IndexSwitchOp::create(builder, loc, TypeRange{builder.getI1Type()},
                                 candidateIndex, cases, cases.size());
  ArrayRef<PipeResourceAccessPlan> candidatePlans = waitAnyPlan.getCandidates();
  for (auto [ordinal, region] : llvm::enumerate(switchOp.getCaseRegions())) {
    assert(region.empty() && "new index switch case must be empty");
    Block *block = new Block();
    region.push_back(block);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(block);
    Value address = buildWaitAnyCompletionAddress(op, candidatePlans[ordinal],
                                                  pipeResourcePlan, builder);
    auto l1PointerType = ttk::L1AddrPtrType::get(builder.getContext(), 32);
    Value pointer =
        ttk::CastToL1PtrOp::create(builder, loc, l1PointerType, address);
    Value reached = ttk::SemaphoreReachedOp::create(
        builder, loc, builder.getI1Type(), pointer, tokenSequences[ordinal]);
    scf::YieldOp::create(builder, loc, reached);
  }
  Region &defaultRegion = switchOp.getDefaultRegion();
  assert(defaultRegion.empty() && "new index switch default must be empty");
  Block *defaultBlock = new Block();
  defaultRegion.push_back(defaultBlock);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(defaultBlock);
    Value notReached = arith::ConstantIntOp::create(builder, loc, 0, 1);
    scf::YieldOp::create(builder, loc, notReached);
  }
  return switchOp.getResults().front();
}

LogicalResult lowerPipeTransferWaitAny(PipeTransferWaitAnyOp op,
                                       ValueRange tokenSequences,
                                       const PipeWaitAnyPlan &waitAnyPlan,
                                       const PipeResourcePlan &pipeResourcePlan,
                                       ConversionPatternRewriter &rewriter) {
  assert(tokenSequences.size() == waitAnyPlan.getCandidates().size() &&
         "wait-any token and candidate plan counts differ");
  Location loc = op.getLoc();
  int64_t candidateCount = static_cast<int64_t>(tokenSequences.size());
  Value countI32 =
      arith::ConstantIntOp::create(rewriter, loc, candidateCount, 32);
  Value startI32 = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(), op.getStart());
  Value signedRemainder =
      arith::RemSIOp::create(rewriter, loc, startI32, countI32);
  Value nonnegativeStart =
      arith::AddIOp::create(rewriter, loc, signedRemainder, countI32);
  Value normalizedStartI32 =
      arith::RemUIOp::create(rewriter, loc, nonnegativeStart, countI32);
  Value sentinel = countI32;

  auto whileOp = scf::WhileOp::create(
      rewriter, loc, TypeRange{rewriter.getI32Type()}, ValueRange{sentinel},
      [&](OpBuilder &builder, Location bodyLoc, ValueRange beforeValues) {
        Value continuePolling =
            arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::eq,
                                  beforeValues.front(), sentinel);
        scf::ConditionOp::create(builder, bodyLoc, continuePolling,
                                 beforeValues);
      },
      [&](OpBuilder &builder, Location bodyLoc, ValueRange afterValues) {
        Value lowerBound = arith::ConstantIndexOp::create(builder, bodyLoc, 0);
        Value upperBound =
            arith::ConstantIndexOp::create(builder, bodyLoc, candidateCount);
        Value step = arith::ConstantIndexOp::create(builder, bodyLoc, 1);
        auto scanLoop = scf::ForOp::create(
            builder, bodyLoc, lowerBound, upperBound, step, afterValues,
            [&](OpBuilder &scanBuilder, Location scanLoc, Value offset,
                ValueRange iterArgs) {
              Value selected = iterArgs.front();
              Value notSelected = arith::CmpIOp::create(
                  scanBuilder, scanLoc, arith::CmpIPredicate::eq, selected,
                  sentinel);
              auto ifOp = scf::IfOp::create(scanBuilder, scanLoc,
                                            TypeRange{scanBuilder.getI32Type()},
                                            notSelected,
                                            /*withElseRegion=*/true);
              scanBuilder.setInsertionPointToStart(
                  &ifOp.getThenRegion().front());
              Value offsetI32 = arith::IndexCastOp::create(
                  scanBuilder, scanLoc, scanBuilder.getI32Type(), offset);
              Value rotated = arith::AddIOp::create(
                  scanBuilder, scanLoc, normalizedStartI32, offsetI32);
              Value candidateI32 = arith::RemUIOp::create(scanBuilder, scanLoc,
                                                          rotated, countI32);
              Value candidateIndex = arith::IndexCastOp::create(
                  scanBuilder, scanLoc, scanBuilder.getIndexType(),
                  candidateI32);
              Value reached = buildWaitAnyCandidateReached(
                  op, candidateIndex, tokenSequences, waitAnyPlan,
                  pipeResourcePlan, scanBuilder);
              Value nextSelected = arith::SelectOp::create(
                  scanBuilder, scanLoc, reached, candidateI32, selected);
              scf::YieldOp::create(scanBuilder, scanLoc, nextSelected);
              scanBuilder.setInsertionPointToStart(
                  &ifOp.getElseRegion().front());
              scf::YieldOp::create(scanBuilder, scanLoc, selected);
              scanBuilder.setInsertionPointAfter(ifOp);
              scf::YieldOp::create(scanBuilder, scanLoc, ifOp.getResults());
            });
        scf::YieldOp::create(builder, bodyLoc, scanLoop.getResults());
      });

  rewriter.replaceOp(op, whileOp.getResults().front());
  return success();
}

//===----------------------------------------------------------------------===//
// Pipe conditional operation lowering patterns
//===----------------------------------------------------------------------===//

namespace {

/// Return the rectangular logical core range selected by `pipeType`'s source.
static ArrayAttr getSourceCoreRanges(MLIRContext *context, PipeType pipeType) {
  auto source = ttcore::CoreCoordAttr::get(context, pipeType.getSrcY(),
                                           pipeType.getSrcX());
  auto range = ttcore::CoreRangeAttr::get(context, source, source);
  return ArrayAttr::get(context, {range});
}

/// Return the rectangular logical core range selected by `pipeType`'s
/// destinations.
static ArrayAttr getDestinationCoreRanges(MLIRContext *context,
                                          PipeType pipeType) {
  int64_t minX = std::min(pipeType.getDstStartX(), pipeType.getDstEndX());
  int64_t maxX = std::max(pipeType.getDstStartX(), pipeType.getDstEndX());
  int64_t minY = std::min(pipeType.getDstStartY(), pipeType.getDstEndY());
  int64_t maxY = std::max(pipeType.getDstStartY(), pipeType.getDstEndY());
  auto start = ttcore::CoreCoordAttr::get(context, minY, minX);
  auto end = ttcore::CoreCoordAttr::get(context, maxY, maxX);
  auto range = ttcore::CoreRangeAttr::get(context, start, end);
  return ArrayAttr::get(context, {range});
}

/// Replace `op` with an `scf.if` that records its static execution domain.
///
/// Retaining the core ranges lets later TTKernel transformations distinguish
/// side effects that cannot execute on the same core without reconstructing
/// role predicates from SSA.
template <typename Op>
static void lowerToScfIf(Op op, Value cond, ArrayAttr executionCoreRanges,
                         ConversionPatternRewriter &rewriter) {
  auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), cond,
                                /*withElseRegion=*/false);
  ifOp->setAttr(ttk::kExecutionCoreRangesAttrName, executionCoreRanges);
  Block &srcBlock = op.getBody().front();
  Block &thenBlock = ifOp.getThenRegion().front();
  if (Operation *terminator = srcBlock.getTerminator();
      terminator && mlir::isa<YieldOp>(terminator)) {
    rewriter.eraseOp(terminator);
  }
  rewriter.inlineBlockBefore(&srcBlock, thenBlock.getTerminator());
  rewriter.eraseOp(op);
}

static Value buildSrcMatch(OpBuilder &builder, Location loc, Value coreX,
                           Value coreY, PipeType pipeType) {
  auto sourceX =
      arith::ConstantIndexOp::create(builder, loc, pipeType.getSrcX());
  auto sourceY =
      arith::ConstantIndexOp::create(builder, loc, pipeType.getSrcY());
  return buildNodePointMatch(builder, loc, coreX, coreY, sourceX, sourceY);
}

static Value buildDstMatch(OpBuilder &builder, Location loc, Value coreX,
                           Value coreY, PipeType pipeType) {
  Value minX = arith::ConstantIndexOp::create(
      builder, loc, std::min(pipeType.getDstStartX(), pipeType.getDstEndX()));
  Value maxX = arith::ConstantIndexOp::create(
      builder, loc, std::max(pipeType.getDstStartX(), pipeType.getDstEndX()));
  Value minY = arith::ConstantIndexOp::create(
      builder, loc, std::min(pipeType.getDstStartY(), pipeType.getDstEndY()));
  Value maxY = arith::ConstantIndexOp::create(
      builder, loc, std::max(pipeType.getDstStartY(), pipeType.getDstEndY()));
  return buildNodeRangeMatch(builder, loc, coreX, coreY, minX, minY, maxX,
                             maxY);
}

struct IfSrcLowering : OpConversionPattern<IfSrcOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfSrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pipeType = mlir::cast<PipeType>(op.getPipe().getType());

    auto coreX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    auto coreY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());

    Value isSrc = buildSrcMatch(rewriter, loc, coreX, coreY, pipeType);
    lowerToScfIf(op, isSrc,
                 getSourceCoreRanges(rewriter.getContext(), pipeType),
                 rewriter);
    return success();
  }
};

struct IfDstLowering : OpConversionPattern<IfDstOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfDstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pipeType = mlir::cast<PipeType>(op.getPipe().getType());

    auto coreX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    auto coreY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());

    Value isDst = buildDstMatch(rewriter, loc, coreX, coreY, pipeType);
    lowerToScfIf(op, isDst,
                 getDestinationCoreRanges(rewriter.getContext(), pipeType),
                 rewriter);
    return success();
  }
};

struct PipeRoleTables {
  SmallVector<int64_t> minX;
  SmallVector<int64_t> minY;
  SmallVector<int64_t> maxX;
  SmallVector<int64_t> maxY;
  SmallVector<int64_t> deviceIndex;

  std::size_t size() const { return minX.size(); }
};

static PipeRoleTables buildPipeRoleTables(PipeNetRecordsAttr records,
                                          PipeRole role) {
  using PipeRoleRecord =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;
  SmallVector<PipeRoleRecord> roleRecords;
  for (PipeRecordAttr record : records.getPipes()) {
    for (const PipeRecordRoleFacts &facts :
         getPipeRecordRoleFacts(record, role)) {
      assert(facts.device &&
             "selected device role requires device transfer records");
      roleRecords.emplace_back(
          facts.minX, facts.minY, facts.maxX, facts.maxY,
          getLogicalDeviceIndex(facts.deviceDomain, facts.device));
    }
  }
  llvm::sort(roleRecords);
  roleRecords.erase(std::unique(roleRecords.begin(), roleRecords.end()),
                    roleRecords.end());

  PipeRoleTables tables;
  for (auto [minX, minY, maxX, maxY, deviceIndex] : roleRecords) {
    tables.minX.push_back(minX);
    tables.minY.push_back(minY);
    tables.maxX.push_back(maxX);
    tables.maxY.push_back(maxY);
    tables.deviceIndex.push_back(deviceIndex);
  }
  return tables;
}

static Value lowerSelectedRolePredicate(Operation *op,
                                        PipeNetRecordsAttr records,
                                        PipeRole role,
                                        ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();
  PipeRoleTables tables = buildPipeRoleTables(records, role);
  DeviceTransferAttr transfer = records.getPipes().front().getDeviceTransfer();
  assert(transfer && "selected device role requires device transfer records");

  Value nodeX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  Value nodeY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  Value currentDevice = CurrentDeviceIndexOp::create(
      rewriter, loc, rewriter.getIndexType(), transfer.getDomain());
  Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value upper = arith::ConstantIndexOp::create(rewriter, loc, tables.size());
  Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value initialMatch = arith::ConstantIntOp::create(rewriter, loc, 0, 1);

  auto loop = scf::ForOp::create(
      rewriter, loc, lower, upper, step, ValueRange{initialMatch},
      [&](OpBuilder &builder, Location bodyLoc, Value recordIndex,
          ValueRange iterArgs) {
        Value minX = buildConstantIndexTableLookup(builder, bodyLoc,
                                                   tables.minX, recordIndex);
        Value minY = buildConstantIndexTableLookup(builder, bodyLoc,
                                                   tables.minY, recordIndex);
        Value maxX = buildConstantIndexTableLookup(builder, bodyLoc,
                                                   tables.maxX, recordIndex);
        Value maxY = buildConstantIndexTableLookup(builder, bodyLoc,
                                                   tables.maxY, recordIndex);
        Value roleDevice = buildConstantIndexTableLookup(
            builder, bodyLoc, tables.deviceIndex, recordIndex);
        Value coordinateMatches = buildNodeRangeMatch(
            builder, bodyLoc, nodeX, nodeY, minX, minY, maxX, maxY);
        Value deviceMatches =
            arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::eq,
                                  currentDevice, roleDevice);
        Value recordMatches = arith::AndIOp::create(
            builder, bodyLoc, coordinateMatches, deviceMatches);
        Value accumulatedMatch = arith::OrIOp::create(
            builder, bodyLoc, iterArgs.front(), recordMatches);
        scf::YieldOp::create(builder, bodyLoc, accumulatedMatch);
      });
  return loop.getResult(0);
}

// Lower a per-pipe-role predicate op to the OR of per-pipe matches in the
// named PipeNet. `roleBuilder` produces the i1 match for one static pipe.
template <typename Op>
static LogicalResult lowerRolePredicate(
    Op op, ConversionPatternRewriter &rewriter,
    const PipeNetIndex &pipeNetIndex, PipeRole role,
    llvm::function_ref<Value(OpBuilder &, Location, Value, Value, PipeType)>
        roleBuilder) {
  auto loc = op.getLoc();
  if (PipeNetRecordsAttr records = op.getRecordsAttr()) {
    rewriter.replaceOp(op,
                       lowerSelectedRolePredicate(op, records, role, rewriter));
    return success();
  }
  int64_t netId = op.getPipeNetId();
  auto it = pipeNetIndex.find(netId);
  assert(it != pipeNetIndex.end() && !it->second.empty() &&
         "role predicate must reference a preflighted PipeNet");
  auto coreX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  auto coreY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  Value result;
  for (const PipeInfo &pipeInfo : it->second) {
    Value match = roleBuilder(rewriter, loc, coreX, coreY, pipeInfo.pipeType);
    result = result ? Value(arith::OrIOp::create(rewriter, loc, result, match))
                    : match;
  }
  rewriter.replaceOp(op, result);
  return success();
}

template <typename Op>
struct IsRoleLoweringBase : OpConversionPattern<Op> {
  IsRoleLoweringBase(const TypeConverter &tc, MLIRContext *ctx,
                     const PipeNetIndex *index)
      : OpConversionPattern<Op>(tc, ctx), pipeNetIndex(index) {}
  const PipeNetIndex *pipeNetIndex;
};

struct IsSrcLowering : IsRoleLoweringBase<IsSrcOp> {
  using IsRoleLoweringBase::IsRoleLoweringBase;
  LogicalResult
  matchAndRewrite(IsSrcOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerRolePredicate(op, rewriter, *pipeNetIndex, PipeRole::Source,
                              buildSrcMatch);
  }
};

struct IsDstLowering : IsRoleLoweringBase<IsDstOp> {
  using IsRoleLoweringBase::IsRoleLoweringBase;
  LogicalResult
  matchAndRewrite(IsDstOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerRolePredicate(op, rewriter, *pipeNetIndex,
                              PipeRole::Destination, buildDstMatch);
  }
};

struct IsActiveLowering : IsRoleLoweringBase<IsActiveOp> {
  using IsRoleLoweringBase::IsRoleLoweringBase;
  LogicalResult
  matchAndRewrite(IsActiveOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerRolePredicate(
        op, rewriter, *pipeNetIndex, PipeRole::Active,
        [](OpBuilder &builder, Location loc, Value coreX, Value coreY,
           PipeType pipeType) {
          Value isSrc = buildSrcMatch(builder, loc, coreX, coreY, pipeType);
          Value isDst = buildDstMatch(builder, loc, coreX, coreY, pipeType);
          return Value(arith::OrIOp::create(builder, loc, isSrc, isDst));
        });
  }
};

struct CreatePipeLowering : OpConversionPattern<CreatePipeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CreatePipeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // CreatePipeOp produces a pipe type whose parameters carry the coordinate
    // info; coordinates are encoded into generated code by if_src/if_dst.
    // Replace with an unrealized cast so uses in nested regions (if_src /
    // if_dst bodies) that may be processed in a different order still resolve.
    // The unrealized cast preserves the type for downstream patterns.
    auto cast = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), ValueRange{});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

} // namespace

LogicalResult buildPipeNetIndex(ModuleOp mod, PipeNetIndex &index) {
  using PipeKey =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
  llvm::MapVector<int64_t, llvm::SmallSetVector<PipeKey, 4>> seenPerNet;
  auto addPipe = [&](PipeType pipeType, PipeTransferContract contract) {
    int64_t netId = pipeType.getPipeNetId();
    PipeKey key{pipeType.getSrcX(),      pipeType.getSrcY(),
                pipeType.getDstStartX(), pipeType.getDstStartY(),
                pipeType.getDstEndX(),   pipeType.getDstEndY()};
    if (seenPerNet[netId].insert(key)) {
      index[netId].push_back(PipeInfo{pipeType, contract});
      return;
    }
    if (!isCollectiveTransfer(contract)) {
      return;
    }
    for (PipeInfo &pipeInfo : index[netId]) {
      PipeType existingType = pipeInfo.pipeType;
      PipeKey existingKey{
          existingType.getSrcX(),      existingType.getSrcY(),
          existingType.getDstStartX(), existingType.getDstStartY(),
          existingType.getDstEndX(),   existingType.getDstEndY()};
      if (existingKey == key) {
        pipeInfo.transferContract = PipeTransferContract::Collective;
        return;
      }
    }
    llvm_unreachable("deduplicated pipe must exist in its PipeNet index");
  };

  mod.walk([&](CreatePipeOp op) {
    addPipe(mlir::cast<PipeType>(op.getResult().getType()),
            getPipeTransferContract(op));
  });
  auto addRecords = [&](PipeNetRecordsAttr records) {
    for (PipeRecordAttr record : records.getPipes()) {
      addPipe(getPipeTypeFromRecord(mod.getContext(), record,
                                    records.getPipeNetId()),
              getPipeTransferContract(record));
    }
  };
  mod.walk([&](PipeNetForeachSrcOp op) { addRecords(op.getRecords()); });
  mod.walk([&](PipeNetForeachDstOp op) { addRecords(op.getRecords()); });

  WalkResult validation = mod.walk([&](PipeNetPredicateOpInterface predicate) {
    if (predicate.getReferencedRecords()) {
      return WalkResult::advance();
    }
    int64_t pipeNetId = predicate.getReferencedPipeNetId();
    auto indexIt = index.find(pipeNetId);
    if (indexIt != index.end() && !indexIt->second.empty()) {
      return WalkResult::advance();
    }
    predicate->emitError() << "references unknown PipeNet " << pipeNetId;
    return WalkResult::interrupt();
  });
  if (validation.wasInterrupted()) {
    return failure();
  }
  return success();
}

namespace {

/// Allocation unit for all resources owned by one transfer definition.
///
/// One send and its corresponding receiver posts share an address mechanism
/// and sender-ready counter. Each receiver wait uses the completion counter
/// assigned to the same unit.
struct PipeTransferAllocationUnit {
  PipeTransferNodeId transferNodeId = 0;
  Operation *sendOp = nullptr;
  /// Send, receiver-post, and receiver-wait operations for this transfer.
  SmallVector<Operation *> protocolOps;
  /// Record indices distinguish graph nodes that share one protocol operation.
  SmallVector<std::pair<Operation *, unsigned>> selectedProtocolRecords;

  /// Logical pipe whose source owns this unit's address and ready resources.
  PipeKey pipe;

  /// Logical source device, when the transfer crosses devices.
  DeviceRefAttr sourceDevice;

  PipeType pipeType;

  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;

  /// Stable tie-breaker for deterministic allocation.
  int64_t ordinal = 0;

  /// Conservative post-to-send lifetime for sender-owned resources.
  OperationLiveInterval interval;

  /// Assigned first-fit color within the source node's allocation group.
  std::size_t resourceColor = 0;

  /// Completion-counter color; disjoint receiver sets may share one color.
  std::optional<int64_t> maybeCompletionCounterColor;

  /// Deterministic order used by first-fit interval coloring.
  bool operator<(const PipeTransferAllocationUnit &rhs) const {
    return std::make_tuple(interval.startOrdinal, pipe.srcX, pipe.srcY,
                           pipe.pipeNetId, pipe.dstStartX, pipe.dstStartY,
                           pipe.dstEndX, pipe.dstEndY, ordinal) <
           std::make_tuple(rhs.interval.startOrdinal, rhs.pipe.srcX,
                           rhs.pipe.srcY, rhs.pipe.pipeNetId,
                           rhs.pipe.dstStartX, rhs.pipe.dstStartY,
                           rhs.pipe.dstEndX, rhs.pipe.dstEndY, rhs.ordinal);
  }
};

static bool isSelectedTransferUnit(const PipeTransferAllocationUnit &unit) {
  return !unit.selectedProtocolRecords.empty();
}

} // namespace

static bool pipeTransferIntervalsOverlap(const PipeTransferAllocationUnit &lhs,
                                         const PipeTransferAllocationUnit &rhs,
                                         const DominanceInfo &dominanceInfo) {
  return intervalsOverlap(lhs.interval, rhs.interval, dominanceInfo);
}

static bool pipeResourceUnitsInterfere(const PipeTransferAllocationUnit &lhs,
                                       const PipeTransferAllocationUnit &rhs,
                                       const DominanceInfo &dominanceInfo) {
  if (lhs.sourceDevice && rhs.sourceDevice &&
      lhs.sourceDevice != rhs.sourceDevice) {
    // Equal resource indices refer to distinct physical storage on distinct
    // logical devices.
    return false;
  }
  if (isSelectedTransferUnit(lhs) || isSelectedTransferUnit(rhs)) {
    // One protocol operation executes for several records, so its
    // operation-level live interval cannot prove that two records are disjoint.
    return true;
  }
  return pipeTransferIntervalsOverlap(lhs, rhs, dominanceInfo);
}

static FailureOr<SmallVector<PipeTransferAllocationUnit, 0>>
collectPipeTransferAllocationUnits(
    ModuleOp mod, const PipeTransferIndex &transferIndex,
    const PipeGraph &pipeGraph, const DominanceInfo &dominanceInfo,
    const PostDominanceInfo &postDominanceInfo,
    llvm::SmallPtrSetImpl<Operation *> &staticallyInactiveOps) {
  SmallVector<PipeTransferAllocationUnit, 0> units;
  llvm::DenseMap<Operation *, int64_t> operationOrdinals;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> waitOpsByPost;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> waitAnyOpsByPost;
  int64_t nextOperationOrdinal = 0;
  WalkResult provenanceWalkResult = mod.walk([&](Operation *op) {
    if (isa<PipeTransferPostOp, PipeTransferSendOp>(op)) {
      operationOrdinals[op] = nextOperationOrdinal++;
      if (!pipeGraph.hasPipeTransferNodeForProtocolOp(op)) {
        staticallyInactiveOps.insert(op);
      }
      return WalkResult::advance();
    }
    if (auto waitOp = dyn_cast<PipeTransferWaitOp>(op)) {
      ArrayRef<Operation *> possiblePosts =
          transferIndex.getPossibleReceivePosts(waitOp);
      if (possiblePosts.size() != 1) {
        waitOp.emitError() << "requires exactly one possible receiver post; "
                              "found "
                           << possiblePosts.size();
        return WalkResult::interrupt();
      }
      Operation *postOp = possiblePosts.front();
      if (!pipeGraph.hasPipeTransferNodeForProtocolOp(postOp)) {
        staticallyInactiveOps.insert(op);
        return WalkResult::advance();
      }
      waitOpsByPost[postOp].push_back(op);
      return WalkResult::advance();
    }
    if (auto waitOp = dyn_cast<PipeTransferWaitAnyOp>(op)) {
      for (ArrayRef<Operation *> possiblePosts :
           transferIndex.getWaitAnyCandidatePosts(waitOp)) {
        for (Operation *postOp : possiblePosts) {
          assert(pipeGraph.hasPipeTransferNodeForProtocolOp(postOp) &&
                 "validated wait-any post must have a transfer graph node");
          waitAnyOpsByPost[postOp].push_back(op);
        }
      }
    }
    return WalkResult::advance();
  });
  if (provenanceWalkResult.wasInterrupted()) {
    return failure();
  }

  auto recordSelectedProtocolRow =
      [&](PipeTransferAllocationUnit &unit, Operation *protocolOp,
          std::optional<std::uint64_t> recordIndex) {
        if (recordIndex) {
          unit.selectedProtocolRecords.push_back(
              {protocolOp, static_cast<unsigned>(*recordIndex)});
        }
      };

  units.reserve(pipeGraph.getPipeTransferNodes().size());
  for (const PipeTransferNode &transferNode :
       pipeGraph.getPipeTransferNodes()) {
    assert(transferNode.sendOp &&
           "pipe transfer graph node must have a send operation");
    auto sendOp = cast<PipeTransferSendOp>(transferNode.sendOp);
    PipeTransferAllocationUnit unit;
    unit.transferNodeId = transferNode.id;
    unit.sendOp = sendOp.getOperation();
    unit.pipe = transferNode.pipe;
    if (transferNode.deviceTransfer) {
      unit.sourceDevice = transferNode.deviceTransfer.getEdge().getSource();
    }
    unit.pipeType = PipeType::get(mod.getContext(), unit.pipe.srcX,
                                  unit.pipe.srcY, unit.pipe.dstStartX,
                                  unit.pipe.dstStartY, unit.pipe.dstEndX,
                                  unit.pipe.dstEndY, unit.pipe.pipeNetId);
    unit.transferContract = transferNode.transferContract;
    unit.ordinal = static_cast<int64_t>(transferNode.id);
    unit.protocolOps.push_back(sendOp.getOperation());
    recordSelectedProtocolRow(unit, sendOp.getOperation(),
                              transferNode.sendRecordIndex);
    updateIntervalEnd(unit.interval, sendOp.getOperation(), dominanceInfo);
    for (Operation *postOp : transferNode.receiverPostOps) {
      auto ordinalIt = operationOrdinals.find(postOp);
      assert(ordinalIt != operationOrdinals.end() &&
             "receiver post is missing an operation ordinal");
      unit.protocolOps.push_back(postOp);
      std::optional<std::uint64_t> postRecordIndex;
      for (PipeReceiverEndpointId endpointId : transferNode.receiverEndpoints) {
        const PipeReceiverEndpoint &endpoint =
            pipeGraph.getPipeReceiverEndpoint(endpointId);
        if (endpoint.postOp != postOp) {
          continue;
        }
        assert(
            (!postRecordIndex || postRecordIndex == endpoint.postRecordIndex) &&
            "one post maps to different records in one transfer node");
        postRecordIndex = endpoint.postRecordIndex;
      }
      recordSelectedProtocolRow(unit, postOp, postRecordIndex);
      auto waitIt = waitOpsByPost.find(postOp);
      if (waitIt != waitOpsByPost.end()) {
        unit.protocolOps.append(waitIt->second.begin(), waitIt->second.end());
        for (Operation *waitOp : waitIt->second) {
          recordSelectedProtocolRow(unit, waitOp, postRecordIndex);
        }
      }
      auto waitAnyIt = waitAnyOpsByPost.find(postOp);
      if (waitAnyIt != waitAnyOpsByPost.end()) {
        for (Operation *waitAnyOp : waitAnyIt->second) {
          updateIntervalEnd(unit.interval, waitAnyOp, dominanceInfo);
        }
      }
      updateIntervalStart(unit.interval, postOp, ordinalIt->second,
                          dominanceInfo);
    }
    finalizeInterval(unit.interval, dominanceInfo, postDominanceInfo);
    units.push_back(std::move(unit));
  }
  return units;
}

using SourceColorMap =
    llvm::MapVector<PipeSourceKey, SmallVector<SmallVector<std::size_t>>>;

static SourceColorMap
assignLiveIntervalColors(MutableArrayRef<PipeTransferAllocationUnit> units,
                         const DominanceInfo &dominanceInfo) {
  llvm::MapVector<PipeSourceKey, SmallVector<std::size_t>> unitIndicesBySource;
  for (std::size_t index = 0, size = units.size(); index < size; ++index) {
    unitIndicesBySource[getPipeSourceKey(units[index].pipeType)].push_back(
        index);
  }

  SourceColorMap colorUsersBySource;
  for (auto &entry : unitIndicesBySource) {
    SmallVector<SmallVector<std::size_t>> colorUsers =
        assignGreedyIntervalColors<std::size_t>(
            entry.second,
            [&](std::size_t lhsIndex, std::size_t rhsIndex) {
              return std::less<PipeTransferAllocationUnit>()(units[lhsIndex],
                                                             units[rhsIndex]);
            },
            [&](std::size_t lhsIndex, std::size_t rhsIndex) {
              return pipeResourceUnitsInterfere(units[lhsIndex],
                                                units[rhsIndex], dominanceInfo);
            });

    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      for (std::size_t unitIndex : indexedColor.value()) {
        units[unitIndex].resourceColor = indexedColor.index();
      }
    }

    colorUsersBySource.insert({entry.first, std::move(colorUsers)});
  }

  return colorUsersBySource;
}

static bool usesFabricProtocol(
    const PipeTransferAllocationUnit &unit,
    const PipeSynchronizationSelection *synchronizationSelection) {
  if (!synchronizationSelection) {
    return false;
  }
  PipeTransferSendOp sendOp = llvm::cast<PipeTransferSendOp>(unit.sendOp);
  return synchronizationSelection->usesFabricProtocol(sendOp);
}

static FailureOr<SmallVector<std::size_t>>
buildWaitAnyCompletionGroups(ModuleOp module,
                             ArrayRef<PipeTransferAllocationUnit> units,
                             const PipeTransferIndex &transferIndex) {
  if (units.size() > std::numeric_limits<unsigned>::max()) {
    module.emitError("too many PipeNet resource allocation units");
    return failure();
  }
  llvm::IntEqClasses completionGroups(static_cast<unsigned>(units.size()));
  using RecordUnit = std::pair<unsigned, std::size_t>;
  llvm::DenseMap<Operation *, SmallVector<RecordUnit>> unitsByPostRecord;
  for (auto indexedUnit : llvm::enumerate(units)) {
    llvm::SmallPtrSet<Operation *, 4> selectedPosts;
    for (auto [operation, recordIndex] :
         indexedUnit.value().selectedProtocolRecords) {
      if (isa<PipeTransferPostOp>(operation)) {
        unitsByPostRecord[operation].push_back(
            {recordIndex, indexedUnit.index()});
        selectedPosts.insert(operation);
      }
    }
    for (Operation *operation : indexedUnit.value().protocolOps) {
      if (isa<PipeTransferPostOp>(operation) &&
          !selectedPosts.contains(operation)) {
        unitsByPostRecord[operation].push_back({0, indexedUnit.index()});
      }
    }
  }

  WalkResult walkResult = module.walk([&](PipeTransferWaitAnyOp waitOp) {
    for (ArrayRef<Operation *> possiblePosts :
         transferIndex.getWaitAnyCandidatePosts(waitOp)) {
      assert(!possiblePosts.empty() && "candidate must have a receiver post");
      Operation *firstPost = possiblePosts.front();
      PipeTransferCreateOp commonCreate =
          transferIndex.getTransferCreate(firstPost);
      std::optional<int64_t> commonDFBIndex = getCBIndex(
          getAttachedCB(cast<PipeTransferPostOp>(firstPost).getDst()));
      auto firstUnitsIt = unitsByPostRecord.find(firstPost);
      assert(commonDFBIndex && firstUnitsIt != unitsByPostRecord.end() &&
             "active wait-any post must have a destination DFB and unit");
      ArrayRef<RecordUnit> commonRecordUnits = firstUnitsIt->second;
      for (Operation *post : possiblePosts.drop_front()) {
        std::optional<int64_t> dfbIndex =
            getCBIndex(getAttachedCB(cast<PipeTransferPostOp>(post).getDst()));
        auto unitsIt = unitsByPostRecord.find(post);
        if (transferIndex.getTransferCreate(post) != commonCreate ||
            dfbIndex != commonDFBIndex || unitsIt == unitsByPostRecord.end() ||
            unitsIt->second.size() != commonRecordUnits.size()) {
          waitOp.emitError()
              << "requires each candidate's possible posts to use one logical "
                 "receive channel and destination DFB stream";
          return WalkResult::interrupt();
        }
        for (const RecordUnit &commonRecordUnit : commonRecordUnits) {
          unsigned recordIndex = commonRecordUnit.first;
          std::size_t commonUnitIndex = commonRecordUnit.second;
          auto matchingUnit =
              llvm::find_if(unitsIt->second, [&](const RecordUnit &recordUnit) {
                return recordUnit.first == recordIndex;
              });
          if (matchingUnit == unitsIt->second.end() ||
              units[commonUnitIndex].pipe != units[matchingUnit->second].pipe) {
            waitOp.emitError()
                << "requires each candidate's possible posts to use one "
                   "logical receive channel and destination DFB stream";
            return WalkResult::interrupt();
          }
          completionGroups.join(static_cast<unsigned>(commonUnitIndex),
                                static_cast<unsigned>(matchingUnit->second));
        }
      }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }
  completionGroups.compress();
  SmallVector<std::size_t> representatives;
  representatives.reserve(units.size());
  for (std::size_t unitIndex = 0; unitIndex < units.size(); ++unitIndex) {
    representatives.push_back(
        completionGroups[static_cast<unsigned>(unitIndex)]);
  }
  return representatives;
}

static bool usesSenderReadyCounter(
    const PipeTransferAllocationUnit &unit,
    const PipeSynchronizationSelection *synchronizationSelection) {
  // Selected transfers publish receiver addresses, so their sender must wait
  // until the matching table entry has been initialized.
  if (!synchronizationSelection || isSelectedTransferUnit(unit)) {
    return true;
  }
  PipeTransferSendOp sendOp = llvm::cast<PipeTransferSendOp>(unit.sendOp);
  return !synchronizationSelection->usesCapacityProtocol(sendOp);
}

static SmallVector<PipeCounterLocation>
getCompletionCounterLocations(const PipeTransferAllocationUnit &unit,
                              const PipeGraph &pipeGraph) {
  SmallVector<PipeCounterLocation> locations;
  for (PipeReceiverEndpointId endpointId :
       pipeGraph.getPipeReceiverEndpoints(unit.transferNodeId)) {
    const PipeReceiverEndpoint &endpoint =
        pipeGraph.getPipeReceiverEndpoint(endpointId);
    PipeCounterLocation location{endpoint.receiverDFB.receiverDevice,
                                 endpoint.receiver.x, endpoint.receiver.y};
    if (!llvm::is_contained(locations, location)) {
      locations.push_back(location);
    }
  }
  assert(!locations.empty() && "pipe completion counter has no destination");
  return locations;
}

static bool counterLocationsOverlap(ArrayRef<PipeCounterLocation> lhs,
                                    ArrayRef<PipeCounterLocation> rhs) {
  return llvm::any_of(lhs, [&](const PipeCounterLocation &lhsLocation) {
    return llvm::any_of(rhs, [&](const PipeCounterLocation &rhsLocation) {
      if (lhsLocation.nodeX != rhsLocation.nodeX ||
          lhsLocation.nodeY != rhsLocation.nodeY) {
        return false;
      }
      return !lhsLocation.device || !rhsLocation.device ||
             lhsLocation.device == rhsLocation.device;
    });
  });
}

/// Reuse a counter color only when every receiver location is distinct.
///
/// A receiver without a proven device may refer to any device.
static int64_t allocateCompletionCounterColor(
    ArrayRef<PipeCounterLocation> locations,
    SmallVectorImpl<SmallVector<PipeCounterLocation>> &locationsByColor) {
  for (auto indexedAllocation : llvm::enumerate(locationsByColor)) {
    if (!counterLocationsOverlap(locations, indexedAllocation.value())) {
      indexedAllocation.value().append(locations.begin(), locations.end());
      return static_cast<int64_t>(indexedAllocation.index());
    }
  }
  locationsByColor.emplace_back(locations.begin(), locations.end());
  return static_cast<int64_t>(locationsByColor.size() - 1);
}

static std::optional<FuncOp>
getSingleSenderFunc(const PipeTransferAllocationUnit &unit) {
  FuncOp senderFunc = unit.sendOp->getParentOfType<FuncOp>();
  return senderFunc ? std::optional<FuncOp>(senderFunc) : std::nullopt;
}

static int64_t getReceiverDFBBlockStrideBytes(const ReceiverDFBInfo &info) {
  auto tileType = llvm::cast<ttcore::TileType>(info.dfbType.getElementType());
  return info.dfbType.getElementsPerBlock() * tileType.getSizeBytes();
}

static int64_t getReceiverDFBStaticByteOffset(const ReceiverDFBInfo &info) {
  auto tileType = llvm::cast<ttcore::TileType>(info.dfbType.getElementType());
  return info.staticTileOffset * tileType.getSizeBytes();
}

/// Return metadata only when the sender can compute every receiver address.
/// The caller uses receiver-published addresses when this proof fails.
static std::optional<PipeComputedAddressInfo>
getComputedAddressInfo(const PipeReceiverEndpoint &receiverEndpoint) {
  const ReceiverDFBInfo &receiverInfo = receiverEndpoint.receiverDFBInfo;
  if (receiverInfo.isTensorBacked || !receiverInfo.hasStaticTileOffset) {
    return std::nullopt;
  }
  if (!llvm::isa<ttcore::TileType>(receiverInfo.dfbType.getElementType())) {
    return std::nullopt;
  }
  // Static receiver addresses are derived from the pipe graph's physical slot
  // assignment. Non-pipe DFB traffic can advance the hardware ring without a
  // pipe post, so computed addressing requires the graph to prove that the
  // receiver stream contains only pipe-delivered blocks.
  const ReceiverAddressSequenceProof &sequence =
      receiverEndpoint.addressSequence;
  if (sequence.getKind() == ReceiverAddressSequenceProofKind::FullyDynamic) {
    return std::nullopt;
  }
  const ReceiverAddressRecurrence &recurrence = *sequence.recurrence;
  if (recurrence.blockCount <= 0 || recurrence.initialSlot < 0 ||
      recurrence.initialSlot >= recurrence.blockCount ||
      recurrence.repeatStride < 0 ||
      recurrence.repeatStride >= recurrence.blockCount ||
      recurrence.blockCount != receiverInfo.blockCount) {
    return std::nullopt;
  }
  int64_t blockStrideBytes = getReceiverDFBBlockStrideBytes(receiverInfo);
  int64_t staticTileByteOffset = getReceiverDFBStaticByteOffset(receiverInfo);
  if (blockStrideBytes <= 0 || !llvm::isInt<32>(blockStrideBytes) ||
      !llvm::isInt<32>(staticTileByteOffset) ||
      !llvm::isInt<32>(recurrence.initialSlot) ||
      !llvm::isInt<32>(recurrence.repeatStride) ||
      !llvm::isInt<32>(receiverInfo.blockCount)) {
    return std::nullopt;
  }
  int64_t maxBlockByteOffset =
      (receiverInfo.blockCount - 1) * blockStrideBytes + staticTileByteOffset;
  if (!llvm::isInt<32>(maxBlockByteOffset)) {
    return std::nullopt;
  }
  return PipeComputedAddressInfo{receiverInfo.dfbIndex,
                                 /*baseRuntimeCommonArgIndex=*/0,
                                 /*baseByteOffset=*/0,
                                 recurrence.initialSlot,
                                 recurrence.repeatStride,
                                 receiverInfo.blockCount,
                                 blockStrideBytes,
                                 staticTileByteOffset,
                                 std::nullopt};
}

/// Computed-address facts indexed by transfer allocation unit before resource
/// coloring builds the final plan.
struct ComputedAddressPlan {
  llvm::DenseMap<std::size_t, PipeComputedAddressInfo> infoByUnitIndex;
  llvm::MapVector<FuncOp, SmallVector<PipeComputedAddressCounterInitInfo>>
      counterInitializations;
  llvm::MapVector<FuncOp, SmallVector<int32_t>> dfbIndices;
};

static ComputedAddressPlan
buildComputedAddressPlan(ModuleOp module,
                         MutableArrayRef<PipeTransferAllocationUnit> units,
                         const PipeGraph &pipeGraph) {
  ComputedAddressPlan plan;

  llvm::SmallSetVector<int64_t, 4> tensorBackedDFBIndices;
  module.walk([&](BindCBOp bind) {
    if (bind.getTensorBackingAttr()) {
      tensorBackedDFBIndices.insert(bind.getCbIndex().getSExtValue());
    }
  });

  /// One transfer whose recurrence can be materialized by its sender.
  struct Candidate {
    std::size_t unitIndex = 0;
    FuncOp senderFunc;
    PipeComputedAddressInfo computedAddress;
  };
  SmallVector<Candidate> candidates;
  llvm::MapVector<FuncOp, llvm::SmallSetVector<int64_t, 4>> dfbIndicesByFunc;

  for (auto indexedUnit : llvm::enumerate(units)) {
    PipeTransferAllocationUnit &unit = indexedUnit.value();
    const PipeTransferNode &transferNode =
        pipeGraph.getPipeTransferNode(unit.transferNodeId);
    // Local table-driven transfers retain receiver publication because it
    // produces substantially smaller kernels. Device transfers cannot publish
    // receiver-local addresses directly, so they require this computation.
    if (isSelectedTransferUnit(unit) && !transferNode.deviceTransfer) {
      continue;
    }
    const PipeReceiverEndpoint *receiverEndpoint =
        pipeGraph.getProvenReceiverAddressEndpoint(transferNode.id);
    if (!receiverEndpoint) {
      continue;
    }
    const ReceiverDFBInfo &receiverInfo = receiverEndpoint->receiverDFBInfo;
    // One common runtime argument supplies the physical DFB base. An index
    // reused by tensor-backed storage can require a different base by epoch.
    if (tensorBackedDFBIndices.contains(receiverInfo.dfbIndex)) {
      continue;
    }
    std::optional<PipeComputedAddressInfo> maybeComputedAddress =
        getComputedAddressInfo(*receiverEndpoint);
    if (!maybeComputedAddress) {
      continue;
    }
    std::optional<FuncOp> maybeSenderFunc = getSingleSenderFunc(unit);
    if (!maybeSenderFunc) {
      continue;
    }
    candidates.push_back(Candidate{indexedUnit.index(), *maybeSenderFunc,
                                   *maybeComputedAddress});
    dfbIndicesByFunc[*maybeSenderFunc].insert(receiverInfo.dfbIndex);
  }

  if (candidates.empty()) {
    return plan;
  }

  llvm::DenseMap<FuncOp, SmallVector<int64_t>> sortedDFBIndicesByFunc;
  for (auto &[func, dfbSet] : dfbIndicesByFunc) {
    SmallVector<int64_t> sortedDFBIndices(dfbSet.begin(), dfbSet.end());
    llvm::sort(sortedDFBIndices);
    sortedDFBIndicesByFunc[func] = sortedDFBIndices;

    plan.dfbIndices[func] =
        llvm::map_to_vector(sortedDFBIndices, [](int64_t dfbIndex) {
          return static_cast<int32_t>(dfbIndex);
        });
  }

  llvm::MapVector<FuncOp, int64_t> nextDynamicSlotCounterIndexByFunc;
  for (const Candidate &candidate : candidates) {
    FuncOp senderFunc = candidate.senderFunc;
    const SmallVector<int64_t> &dfbIndices = sortedDFBIndicesByFunc[senderFunc];
    PipeComputedAddressInfo computedAddress = candidate.computedAddress;
    auto dfbIt = llvm::find(dfbIndices, computedAddress.receiverDFBIndex);
    assert(dfbIt != dfbIndices.end() && "candidate DFB missing from func list");
    computedAddress.baseRuntimeCommonArgIndex =
        CommonRuntimeArgLayout(senderFunc,
                               static_cast<int64_t>(dfbIndices.size()))
            .getComputedReceiverDFBBaseIndex(
                std::distance(dfbIndices.begin(), dfbIt));

    const PipeTransferAllocationUnit &unit = units[candidate.unitIndex];
    const PipeTransferNode &transferNode =
        pipeGraph.getPipeTransferNode(unit.transferNodeId);
    const PipeReceiverEndpoint *receiverEndpoint =
        pipeGraph.getProvenReceiverAddressEndpoint(transferNode.id);
    assert(receiverEndpoint &&
           "computed-address unit missing receiver address proof");
    const ReceiverAddressSequenceProof &sequence =
        receiverEndpoint->addressSequence;
    bool canRepeat =
        sequence.getKind() != ReceiverAddressSequenceProofKind::KnownCount ||
        *sequence.executionCount > 1;
    if (canRepeat && computedAddress.repeatStride != 0) {
      int64_t counterIndex = nextDynamicSlotCounterIndexByFunc[senderFunc]++;
      computedAddress.dynamicSlotCounterIndex = counterIndex;
      plan.counterInitializations[senderFunc].push_back(
          PipeComputedAddressCounterInitInfo{counterIndex,
                                             computedAddress.initialSlot});
    }
    plan.infoByUnitIndex[candidate.unitIndex] = computedAddress;
  }

  return plan;
}

// Compact the per-source colors whose units need a resource into a dense
// 0..N-1 index range, keyed by original color index. Returns the compacted map
// and the maximum compacted count across sources.
template <typename PredT>
static std::pair<
    llvm::MapVector<PipeSourceKey, llvm::DenseMap<std::size_t, int64_t>>,
    int64_t>
compactColors(const SourceColorMap &colorUsersBySource,
              PredT unitNeedsResource) {
  llvm::MapVector<PipeSourceKey, llvm::DenseMap<std::size_t, int64_t>>
      compactedBySource;
  int64_t maxPerSource = 0;
  for (const auto &[sourceKey, colorUsers] : colorUsersBySource) {
    int64_t nextColor = 0;
    llvm::DenseMap<std::size_t, int64_t> &compacted =
        compactedBySource[sourceKey];
    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      if (llvm::any_of(indexedColor.value(), unitNeedsResource)) {
        compacted[indexedColor.index()] = nextColor++;
      }
    }
    maxPerSource = std::max(maxPerSource, nextColor);
  }
  return {std::move(compactedBySource), maxPerSource};
}

LogicalResult buildPipeResourcePlan(
    ModuleOp mod, const PipeTransferIndex &transferIndex,
    const PipeGraph &pipeGraph, PipeResourcePlan &info,
    bool enableComputedAddresses, PipeCounterAllocationPolicy counterPolicy,
    const PipeSynchronizationSelection *synchronizationSelection) {
  DominanceInfo dominanceInfo(mod);
  PostDominanceInfo postDominanceInfo(mod);
  FailureOr<SmallVector<PipeTransferAllocationUnit, 0>> maybeUnits =
      collectPipeTransferAllocationUnits(mod, transferIndex, pipeGraph,
                                         dominanceInfo, postDominanceInfo,
                                         info.staticallyInactiveOps);
  if (failed(maybeUnits)) {
    return failure();
  }
  SmallVector<PipeTransferAllocationUnit, 0> &units = *maybeUnits;
  SourceColorMap colorUsersBySource =
      assignLiveIntervalColors(units, dominanceInfo);
  ComputedAddressPlan computedAddressPlan;
  if (enableComputedAddresses) {
    computedAddressPlan = buildComputedAddressPlan(mod, units, pipeGraph);
  }
  info.computedAddressCounterInitializations =
      computedAddressPlan.counterInitializations;
  info.computedAddressDFBIndices = computedAddressPlan.dfbIndices;

  FailureOr<SmallVector<std::size_t>> maybeCompletionGroups =
      buildWaitAnyCompletionGroups(mod, units, transferIndex);
  if (failed(maybeCompletionGroups)) {
    return failure();
  }

  struct CompletionCounterGroup {
    bool usesFabric = false;
    SmallVector<std::size_t> unitIndices;
    SmallVector<PipeCounterLocation> locations;
  };
  SmallVector<CompletionCounterGroup> completionGroups;
  llvm::DenseMap<std::size_t, std::size_t> groupIndexByRepresentative;
  for (auto indexedUnit : llvm::enumerate(units)) {
    std::size_t representative = (*maybeCompletionGroups)[indexedUnit.index()];
    auto [groupIt, inserted] = groupIndexByRepresentative.try_emplace(
        representative, completionGroups.size());
    bool usesFabric =
        usesFabricProtocol(indexedUnit.value(), synchronizationSelection);
    if (inserted) {
      completionGroups.emplace_back();
      completionGroups.back().usesFabric = usesFabric;
    }
    CompletionCounterGroup &group = completionGroups[groupIt->second];
    if (group.usesFabric != usesFabric) {
      mod.emitError("one wait-any completion group mixes local and fabric "
                    "synchronization");
      return failure();
    }
    group.unitIndices.push_back(indexedUnit.index());
    for (PipeCounterLocation location :
         getCompletionCounterLocations(indexedUnit.value(), pipeGraph)) {
      if (!llvm::is_contained(group.locations, location)) {
        group.locations.push_back(location);
      }
    }
  }

  SmallVector<SmallVector<PipeCounterLocation>>
      nodeLocalCompletionLocationsByColor;
  SmallVector<SmallVector<PipeCounterLocation>>
      fabricCompletionLocationsByColor;
  for (const CompletionCounterGroup &group : completionGroups) {
    SmallVector<SmallVector<PipeCounterLocation>> &locationsByColor =
        group.usesFabric ? fabricCompletionLocationsByColor
                         : nodeLocalCompletionLocationsByColor;
    int64_t color =
        allocateCompletionCounterColor(group.locations, locationsByColor);
    for (std::size_t unitIndex : group.unitIndices) {
      units[unitIndex].maybeCompletionCounterColor = color;
    }
  }
  PipeCounterAllocator counterAllocator(PipeCounterAllocationCounts{},
                                        counterPolicy);
  SmallVector<PipeCounterInfo> nodeLocalCompletionCounters;
  nodeLocalCompletionCounters.reserve(
      nodeLocalCompletionLocationsByColor.size());
  for (std::size_t counterIndex = 0;
       counterIndex < nodeLocalCompletionLocationsByColor.size();
       ++counterIndex) {
    nodeLocalCompletionCounters.push_back(counterAllocator.allocate());
  }
  SmallVector<PipeCounterInfo> fabricCompletionCounters;
  fabricCompletionCounters.reserve(fabricCompletionLocationsByColor.size());
  for (std::size_t counterIndex = 0;
       counterIndex < fabricCompletionLocationsByColor.size(); ++counterIndex) {
    fabricCompletionCounters.push_back(counterAllocator.allocateGlobal());
  }

  auto [readyColorBySourceColor, maxReadyCountersPerSource] =
      compactColors(colorUsersBySource, [&](std::size_t unitIndex) {
        return usesSenderReadyCounter(units[unitIndex],
                                      synchronizationSelection);
      });

  // The same ready color is reused on different source nodes, so every source
  // must interpret that color as the same storage kind.
  PipeCounterAllocationCounts counterCounts = counterAllocator.getCounts();
  bool hasFabricReadyCounter =
      llvm::any_of(units, [&](const PipeTransferAllocationUnit &unit) {
        return usesSenderReadyCounter(unit, synchronizationSelection) &&
               usesFabricProtocol(unit, synchronizationSelection);
      });
  bool useGlobalReadyCounters =
      hasFabricReadyCounter ||
      counterPolicy == PipeCounterAllocationPolicy::GlobalOnly ||
      counterCounts.localSemaphoreCount + maxReadyCountersPerSource >
          kMaxHardwareSemaphoreIds;

  // A global semaphore index refers to distinct storage on each source node.
  // Only counters live on the same source need distinct indices.
  SmallVector<PipeCounterInfo> readyCounterByColor;
  readyCounterByColor.reserve(maxReadyCountersPerSource);
  for (int64_t color = 0; color < maxReadyCountersPerSource; ++color) {
    readyCounterByColor.push_back(useGlobalReadyCounters
                                      ? counterAllocator.allocateGlobal()
                                      : counterAllocator.allocate());
  }

  auto [addressColorBySourceColor, maxAddressColorsPerSource] =
      compactColors(colorUsersBySource, [&](std::size_t unitIndex) {
        return computedAddressPlan.infoByUnitIndex.find(unitIndex) ==
               computedAddressPlan.infoByUnitIndex.end();
      });
  int64_t maxAddressTableBytes =
      maxAddressColorsPerSource * kPipeAddressWordBytes;
  RecordAlignedTableBuilder<PipeResourceInfo> selectedResources;

  for (auto indexedUnit : llvm::enumerate(units)) {
    const PipeTransferAllocationUnit &unit = indexedUnit.value();
    assert(unit.maybeCompletionCounterColor &&
           "pipe transfer is missing a completion counter color");
    int64_t completionColor = *unit.maybeCompletionCounterColor;
    ArrayRef<PipeCounterInfo> completionCounters =
        usesFabricProtocol(unit, synchronizationSelection)
            ? ArrayRef<PipeCounterInfo>(fabricCompletionCounters)
            : ArrayRef<PipeCounterInfo>(nodeLocalCompletionCounters);
    assert(completionColor < static_cast<int64_t>(completionCounters.size()));
    PipeSourceKey sourceKey = getPipeSourceKey(unit.pipeType);
    std::optional<PipeCounterInfo> maybeReadyCounter;
    if (usesSenderReadyCounter(unit, synchronizationSelection)) {
      auto sourceIt = readyColorBySourceColor.find(sourceKey);
      assert(sourceIt != readyColorBySourceColor.end());
      auto colorIt = sourceIt->second.find(unit.resourceColor);
      assert(colorIt != sourceIt->second.end());
      int64_t readyColor = colorIt->second;
      assert(readyColor < static_cast<int64_t>(readyCounterByColor.size()));
      maybeReadyCounter = readyCounterByColor[readyColor];
    }

    auto computedIt =
        computedAddressPlan.infoByUnitIndex.find(indexedUnit.index());
    PipeAddressStorageInfo addressStorage;
    if (computedIt != computedAddressPlan.infoByUnitIndex.end()) {
      addressStorage =
          PipeAddressStorageInfo::computedReceiverDFB(computedIt->second);
    } else {
      auto sourceIt = addressColorBySourceColor.find(sourceKey);
      assert(sourceIt != addressColorBySourceColor.end());
      auto colorIt = sourceIt->second.find(unit.resourceColor);
      assert(colorIt != sourceIt->second.end());
      addressStorage = PipeAddressStorageInfo::receiverPublishedAddressTable(
          PipeSramAddressTableInfo{colorIt->second * kPipeAddressWordBytes});
    }
    PipeResourceInfo pipeResource{
        unit.transferNodeId,
        unit.pipe,
        unit.transferContract,
        PipeCompletionInfo{completionCounters[completionColor]},
        maybeReadyCounter,
        addressStorage,
    };
    llvm::SmallPtrSet<Operation *, 4> selectedProtocolOps;
    for (auto [protocolOp, recordIndex] : unit.selectedProtocolRecords) {
      selectedProtocolOps.insert(protocolOp);
      FailureOr<PipeReference> pipeRef =
          getPipeReferenceForProtocolOp(protocolOp, transferIndex);
      assert(succeeded(pipeRef) && pipeRef->isSelected() &&
             "selected protocol operation requires a selected pipe reference");
      if (failed(selectedResources.set(
              protocolOp, pipeRef->getRecords().getPipes().size(), recordIndex,
              pipeResource,
              [](Operation *operation, const PipeResourceInfo &,
                 const PipeResourceInfo &) {
                operation->emitError(
                    "one pipe record was assigned to two resource units");
                return failure();
              }))) {
        return failure();
      }
    }
    for (Operation *protocolOp : unit.protocolOps) {
      if (selectedProtocolOps.contains(protocolOp)) {
        continue;
      }
      auto [resourceIt, inserted] =
          info.resources.insert({protocolOp, pipeResource});
      assert((inserted || resourceIt->second.pipe == pipeResource.pipe) &&
             "pipe protocol operation assigned to two transfers");
    }
  }

  info.selectedResources = std::move(selectedResources).finalize();

  info.sramScratch.bytes =
      maxAddressTableBytes == 0
          ? 0
          : llvm::alignTo(maxAddressTableBytes, kPipeSramScratchAlignmentBytes);
  return success();
}

void finalizePipeTransportResources(const PipeTransportPlan &transportPlan,
                                    PipeResourcePlan &pipeResourcePlan) {
  int64_t transportScratchBytes = transportPlan.getSramScratchBytes();
  SmallVector<std::pair<Operation *, PipeResourceInfo *>> resources;
  resources.reserve(pipeResourcePlan.resources.size());
  for (auto &[operation, resource] : pipeResourcePlan.resources) {
    resources.emplace_back(operation, &resource);
  }
  for (auto &[operation, selectedResources] :
       pipeResourcePlan.selectedResources) {
    for (PipeResourceInfo &resource : selectedResources) {
      resources.emplace_back(operation, &resource);
    }
  }

  llvm::MapVector<FuncOp, int64_t> nextComputedCounterIndex;
  for (const auto &[function, initializations] :
       pipeResourcePlan.computedAddressCounterInitializations) {
    int64_t &nextIndex = nextComputedCounterIndex[function];
    for (const PipeComputedAddressCounterInitInfo &initialization :
         initializations) {
      nextIndex = std::max(nextIndex, initialization.counterIndex + 1);
    }
  }

  for (const PipeTransportStream &stream : transportPlan.getStreams()) {
    if (stream.getSourceStorage().ownership !=
            PipeTransportStorageOwnership::Transport ||
        stream.getEndpoints().size() != 1 ||
        stream.getEndpoints().front().ownership !=
            PipeTransportStorageOwnership::Transport) {
      continue;
    }

    const PipeTransportEndpoint &endpoint = stream.getEndpoints().front();
    std::optional<int64_t> dynamicSlotCounterIndex;
    if (stream.getSchedule() == PipeTransportSchedule::Overlapped) {
      auto sendResource =
          llvm::find_if(pipeResourcePlan.resources, [&](const auto &entry) {
            return isa<PipeTransferSendOp>(entry.first) &&
                   entry.second.transferNode == stream.getTransferNode();
          });
      assert(sendResource != pipeResourcePlan.resources.end() &&
             "transport stream is missing sender resources");
      if (sendResource->second.addressStorage.computedAddress) {
        const PipeComputedAddressInfo &computedAddress =
            *sendResource->second.addressStorage.computedAddress;
        assert(computedAddress.initialSlot == 0 &&
               "transport-owned storage must start at slot zero");
        dynamicSlotCounterIndex = computedAddress.dynamicSlotCounterIndex;
      }
      if (!dynamicSlotCounterIndex) {
        FuncOp senderFunc =
            sendResource->first->getParentOfType<func::FuncOp>();
        int64_t counterIndex = nextComputedCounterIndex[senderFunc]++;
        dynamicSlotCounterIndex = counterIndex;
        pipeResourcePlan.computedAddressCounterInitializations[senderFunc]
            .push_back(PipeComputedAddressCounterInitInfo{counterIndex,
                                                          /*initialSlot=*/0});
      }
    }

    int64_t destinationGroupDepth =
        stream.getSchedule() == PipeTransportSchedule::Overlapped
            ? endpoint.groupDepth
            : 1;
    PipeComputedAddressInfo computedAddress{
        endpoint.receiverDFB.dfbIndex,
        /*baseRuntimeCommonArgIndex=*/0,
        endpoint.scratchByteOffset,
        /*initialSlot=*/0,
        /*repeatStride=*/destinationGroupDepth > 1 ? 1 : 0,
        /*blockCount=*/destinationGroupDepth,
        stream.getPacketization().getPayloadSizeBytes(),
        /*staticTileByteOffset=*/0,
        dynamicSlotCounterIndex,
    };
    PipeAddressStorageInfo scratchAddress =
        PipeAddressStorageInfo::transportScratch(computedAddress);
    for (auto [operation, resource] : resources) {
      (void)operation;
      if (resource->transferNode == stream.getTransferNode()) {
        resource->addressStorage = scratchAddress;
      }
    }
  }

  llvm::MapVector<FuncOp, llvm::SmallSetVector<int64_t, 4>> dfbIndicesBySender;
  for (auto [operation, resource] : resources) {
    auto sendOp = dyn_cast<PipeTransferSendOp>(operation);
    if (!sendOp || !resource->addressStorage.usesComputedReceiverDFB()) {
      continue;
    }
    assert(resource->addressStorage.computedAddress.has_value() &&
           "computed receiver DFB is missing address information");
    dfbIndicesBySender[sendOp->getParentOfType<FuncOp>()].insert(
        resource->addressStorage.computedAddress->receiverDFBIndex);
  }

  pipeResourcePlan.computedAddressDFBIndices.clear();
  llvm::DenseMap<PipeTransferNodeId, PipeResourceInfo *>
      senderResourceByTransfer;
  for (auto &[senderFunc, dfbIndexSet] : dfbIndicesBySender) {
    SmallVector<int64_t> dfbIndices(dfbIndexSet.begin(), dfbIndexSet.end());
    llvm::sort(dfbIndices);
    pipeResourcePlan.computedAddressDFBIndices[senderFunc] =
        llvm::map_to_vector(dfbIndices, [](int64_t dfbIndex) {
          return static_cast<int32_t>(dfbIndex);
        });
  }

  for (auto [operation, resource] : resources) {
    auto sendOp = dyn_cast<PipeTransferSendOp>(operation);
    if (!sendOp) {
      continue;
    }
    if (resource->addressStorage.usesComputedReceiverDFB()) {
      PipeComputedAddressInfo &computedAddress =
          *resource->addressStorage.computedAddress;
      FuncOp senderFunc = sendOp->getParentOfType<FuncOp>();
      auto indicesIt =
          pipeResourcePlan.computedAddressDFBIndices.find(senderFunc);
      assert(indicesIt != pipeResourcePlan.computedAddressDFBIndices.end() &&
             "sender is missing computed receiver DFB indices");
      ArrayRef<int32_t> dfbIndices = indicesIt->second;
      auto dfbIndexIt =
          llvm::find(dfbIndices, computedAddress.receiverDFBIndex);
      assert(dfbIndexIt != dfbIndices.end() &&
             "computed receiver DFB is missing its runtime argument");
      computedAddress.baseRuntimeCommonArgIndex =
          CommonRuntimeArgLayout(senderFunc,
                                 static_cast<int64_t>(dfbIndices.size()))
              .getComputedReceiverDFBBaseIndex(
                  std::distance(dfbIndices.begin(), dfbIndexIt));
    }
    auto [resourceIt, inserted] =
        senderResourceByTransfer.try_emplace(resource->transferNode, resource);
    assert((inserted || resourceIt->second->pipe == resource->pipe) &&
           "pipe transfer has inconsistent sender resources");
  }

  for (auto [operation, resource] : resources) {
    (void)operation;
    auto senderIt = senderResourceByTransfer.find(resource->transferNode);
    assert(senderIt != senderResourceByTransfer.end() &&
           "pipe transfer is missing sender address storage");
    resource->addressStorage = senderIt->second->addressStorage;
  }

  llvm::DenseMap<PipeSourceKey, llvm::DenseMap<int64_t, int64_t>>
      compactAddressOffsets;
  int64_t addressTableBytes = 0;
  for (auto [operation, resource] : resources) {
    (void)operation;
    if (resource->addressStorage.mode !=
        PipeAddressMode::ReceiverPublishedAddressTable) {
      continue;
    }
    assert(resource->addressStorage.sramAddressTable.has_value() &&
           "address-table pipe is missing SRAM storage");
    int64_t oldOffset = resource->addressStorage.sramAddressTable->byteOffset;
    auto &sourceOffsets = compactAddressOffsets[PipeSourceKey{
        resource->pipe.srcX, resource->pipe.srcY}];
    auto [offsetIt, inserted] = sourceOffsets.try_emplace(
        oldOffset,
        static_cast<int64_t>(sourceOffsets.size()) * kPipeAddressWordBytes);
    (void)inserted;
    int64_t compactOffset = offsetIt->second;
    resource->addressStorage.sramAddressTable->byteOffset =
        transportScratchBytes + compactOffset;
    addressTableBytes =
        std::max(addressTableBytes, compactOffset + kPipeAddressWordBytes);
  }

  int64_t alignedAddressTableBytes =
      addressTableBytes == 0
          ? 0
          : llvm::alignTo(addressTableBytes, kPipeSramScratchAlignmentBytes);
  assert(transportScratchBytes <=
             std::numeric_limits<int64_t>::max() - alignedAddressTableBytes &&
         "combined pipe scratch allocation exceeds int64_t");
  pipeResourcePlan.sramScratch.bytes =
      transportScratchBytes + alignedAddressTableBytes;
}

PipeResourceRequirements
getPipeResourceRequirements(const PipeResourcePlan &info,
                            const PipeCapacityPlan *pipeCapacityPlan) {
  PipeCounterAllocationCounts counts;
  LogicalResult traversalResult = info.forEachResourceTable(
      [&](Operation *, ArrayRef<PipeResourceInfo> resources,
          PipeResourceTableKind) {
        for (const PipeResourceInfo &resource : resources) {
          counts.include(resource.completion.counter);
          if (resource.readyCounter) {
            counts.include(*resource.readyCounter);
          }
        }
        return success();
      });
  assert(succeeded(traversalResult) && "infallible resource traversal failed");
  if (pipeCapacityPlan) {
    PipeCounterAllocationCounts capacityCounts =
        pipeCapacityPlan->getCounterAllocationCounts();
    assert(capacityCounts.localSemaphoreCount >= counts.localSemaphoreCount &&
           capacityCounts.globalSemaphoreCount >= counts.globalSemaphoreCount &&
           "capacity allocation must continue after pipe resource allocation");
    counts = capacityCounts;
  }
  return PipeResourceRequirements{
      counts.localSemaphoreCount,
      counts.globalSemaphoreCount,
      info.sramScratch.bytes,
  };
}

void populatePipeLoweringPatterns(RewritePatternSet &patterns,
                                  const TypeConverter &typeConverter,
                                  const PipeNetIndex &pipeNetIndex) {
  patterns.add<IfSrcLowering, IfDstLowering, CreatePipeLowering>(
      typeConverter, patterns.getContext());
  patterns.add<IsSrcLowering, IsDstLowering, IsActiveLowering>(
      typeConverter, patterns.getContext(), &pipeNetIndex);
}

} // namespace mlir::tt::ttl
