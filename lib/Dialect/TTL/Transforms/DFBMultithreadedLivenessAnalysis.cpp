// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBMultithreadedLivenessAnalysis.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LiveIntervalUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

namespace mlir::tt::ttl {

namespace {

struct EventPair {
  unsigned entry = 0;
  unsigned completion = 0;
};

class HappensBeforeGraph {
public:
  EventPair addOperation() {
    unsigned entry = addEvent();
    unsigned completion = addEvent();
    addEdge(entry, completion);
    return {entry, completion};
  }

  void addEdge(unsigned source, unsigned destination) {
    assert(source < successors.size() && destination < successors.size());
    successors[source].push_back(destination);
  }

  void computeReachability() {
    unsigned eventCount = successors.size();
    reachable.assign(eventCount, llvm::BitVector(eventCount));
    for (unsigned eventIndex = 0; eventIndex < eventCount; ++eventIndex) {
      reachable[eventIndex].set(eventIndex);
      for (unsigned successor : successors[eventIndex]) {
        reachable[eventIndex].set(successor);
      }
    }

    for (unsigned intermediate = 0; intermediate < eventCount; ++intermediate) {
      for (unsigned source = 0; source < eventCount; ++source) {
        if (reachable[source].test(intermediate)) {
          reachable[source] |= reachable[intermediate];
        }
      }
    }
  }

  bool strictlyPrecedes(unsigned source, unsigned destination) const {
    assert(source < reachable.size() && destination < reachable.size());
    return source != destination && reachable[source].test(destination) &&
           !reachable[destination].test(source);
  }

private:
  unsigned addEvent() {
    unsigned eventIndex = successors.size();
    successors.emplace_back();
    return eventIndex;
  }

  SmallVector<SmallVector<unsigned>> successors;
  SmallVector<llvm::BitVector> reachable;
};

struct LogicalDFB {
  int64_t logicalId = 0;
  Type type;
  func::FuncOp producerThread;
  func::FuncOp consumerThread;
  SmallVector<BindCBOp> declarations;
  SmallVector<Operation *> reserves;
  SmallVector<Operation *> pushes;
  SmallVector<Operation *> waits;
  SmallVector<Operation *> pops;
  SmallVector<Operation *> runtimeUses;
  SmallVector<unsigned> earliestEvents;
  SmallVector<unsigned> terminalEvents;
  std::optional<int64_t> transactionTileCount;
  bool bounded = false;
};

struct AnalysisFailure {
  Operation *operation = nullptr;
  std::string message;

  void set(Operation *failureOperation, std::string failureMessage) {
    if (!message.empty()) {
      return;
    }
    operation = failureOperation;
    message = std::move(failureMessage);
  }
};

static BindCBOp getBindOp(Value dfb) {
  return traceUnrealizedCasts(dfb).getDefiningOp<BindCBOp>();
}

static bool
collectLogicalDFBs(ModuleOp moduleOp, SmallVectorImpl<LogicalDFB> &logicalDFBs,
                   DenseMap<Operation *, unsigned> &bindToLogicalDFB,
                   AnalysisFailure &analysisFailure) {
  llvm::MapVector<int64_t, unsigned> idToLogicalDFB;
  llvm::SmallDenseSet<int64_t> explicitIds;
  llvm::SmallDenseSet<int64_t> seenFallbackIndices;
  SmallVector<int64_t> fallbackIndices;
  int64_t maxReservedId = -1;
  int64_t compilerDeclarationCount = 0;

  moduleOp.walk([&](BindCBOp bindOp) {
    if (auto dfbId = bindOp.getDfbId()) {
      int64_t explicitId = dfbId->getSExtValue();
      explicitIds.insert(explicitId);
      maxReservedId = std::max(maxReservedId, explicitId);
      return;
    }
    if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
      ++compilerDeclarationCount;
      return;
    }
    int64_t fallbackIndex = bindOp.getCbIndex().getSExtValue();
    maxReservedId = std::max(maxReservedId, fallbackIndex);
    if (seenFallbackIndices.insert(fallbackIndex).second) {
      fallbackIndices.push_back(fallbackIndex);
    }
  });

  int64_t generatedIdCount = compilerDeclarationCount;
  for (int64_t fallbackIndex : fallbackIndices) {
    generatedIdCount += explicitIds.contains(fallbackIndex);
  }
  if (generatedIdCount > 0 &&
      maxReservedId > std::numeric_limits<int64_t>::max() - generatedIdCount) {
    analysisFailure.set(
        moduleOp,
        "logical DFB identifiers leave no space for generated identities");
    return false;
  }

  int64_t nextLogicalId = generatedIdCount > 0 ? maxReservedId + 1 : 0;
  DenseMap<int64_t, int64_t> fallbackLogicalIds;
  for (int64_t fallbackIndex : fallbackIndices) {
    int64_t logicalId =
        explicitIds.contains(fallbackIndex) ? nextLogicalId++ : fallbackIndex;
    fallbackLogicalIds[fallbackIndex] = logicalId;
  }

  moduleOp.walk([&](BindCBOp bindOp) {
    if (!analysisFailure.message.empty()) {
      return;
    }
    Type dfbType = bindOp.getResult().getType();
    int64_t logicalId;
    if (auto dfbId = bindOp.getDfbId()) {
      logicalId = dfbId->getSExtValue();
    } else if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
      logicalId = nextLogicalId++;
    } else {
      auto fallbackIt =
          fallbackLogicalIds.find(bindOp.getCbIndex().getSExtValue());
      assert(fallbackIt != fallbackLogicalIds.end() &&
             "every untagged user DFB must have a fallback identity");
      logicalId = fallbackIt->second;
    }

    auto [logicalIt, inserted] =
        idToLogicalDFB.insert({logicalId, logicalDFBs.size()});
    unsigned logicalIndex = logicalIt->second;
    if (inserted) {
      LogicalDFB logicalDFB;
      logicalDFB.logicalId = logicalId;
      logicalDFB.type = dfbType;
      logicalDFBs.push_back(std::move(logicalDFB));
    } else if (logicalDFBs[logicalIndex].type != dfbType) {
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream
          << "logical DFB " << logicalId
          << " has inconsistent types across thread functions: expected "
          << logicalDFBs[logicalIndex].type << " but found " << dfbType;
      analysisFailure.set(bindOp, messageStream.str());
      return;
    }

    logicalDFBs[logicalIndex].declarations.push_back(bindOp);
    bindToLogicalDFB[bindOp.getOperation()] = logicalIndex;
  });

  if (!analysisFailure.message.empty()) {
    return false;
  }

  moduleOp.walk([&](Operation *operation) {
    if (!analysisFailure.message.empty() || isa<AttachCBOp>(operation)) {
      return;
    }

    SmallVector<unsigned> operationLogicalDFBs;
    for (Value operand : operation->getOperands()) {
      if (!isa<CircularBufferType>(operand.getType())) {
        continue;
      }
      BindCBOp bindOp = getBindOp(operand);
      if (!bindOp) {
        analysisFailure.set(
            operation,
            "DFB operand must resolve to ttl.bind_cb before physical index "
            "allocation");
        continue;
      }
      auto logicalIt = bindToLogicalDFB.find(bindOp.getOperation());
      assert(logicalIt != bindToLogicalDFB.end() &&
             "every bind must have a logical DFB");
      if (!llvm::is_contained(operationLogicalDFBs, logicalIt->second)) {
        operationLogicalDFBs.push_back(logicalIt->second);
      }
    }

    for (unsigned logicalIndex : operationLogicalDFBs) {
      LogicalDFB &logicalDFB = logicalDFBs[logicalIndex];
      logicalDFB.runtimeUses.push_back(operation);
      if (isa<CBReserveOp>(operation)) {
        logicalDFB.reserves.push_back(operation);
      } else if (isa<CBPushOp>(operation)) {
        logicalDFB.pushes.push_back(operation);
      } else if (isa<CBWaitOp>(operation)) {
        logicalDFB.waits.push_back(operation);
      } else if (isa<CBPopOp>(operation)) {
        logicalDFB.pops.push_back(operation);
      }
    }
  });

  return analysisFailure.message.empty();
}

static void
buildProgramOrderGraph(ModuleOp moduleOp, HappensBeforeGraph &graph,
                       DenseMap<Operation *, EventPair> &operationEvents) {
  for (func::FuncOp funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (funcOp.getBody().empty() || !funcOp.getBody().hasOneBlock()) {
      continue;
    }

    std::optional<EventPair> previousEvents;
    for (Operation &operation : funcOp.getBody().front()) {
      EventPair currentEvents = graph.addOperation();
      operationEvents[&operation] = currentEvents;
      if (previousEvents.has_value()) {
        graph.addEdge(previousEvents->completion, currentEvents.entry);
      }
      previousEvents = currentEvents;
    }
  }
}

static std::optional<EventPair>
getProjectedEvents(Operation *operation,
                   const DenseMap<Operation *, EventPair> &operationEvents) {
  func::FuncOp funcOp = operation->getParentOfType<func::FuncOp>();
  if (!funcOp || funcOp.getBody().empty() || !funcOp.getBody().hasOneBlock()) {
    return std::nullopt;
  }

  Block &functionBody = funcOp.getBody().front();
  Operation *projected = operation->getBlock() == &functionBody
                             ? operation
                             : functionBody.findAncestorOpInBlock(*operation);
  if (!projected) {
    return std::nullopt;
  }

  auto eventIt = operationEvents.find(projected);
  if (eventIt == operationEvents.end()) {
    return std::nullopt;
  }
  return eventIt->second;
}

static bool isDirectFunctionBodyOperation(Operation *operation) {
  func::FuncOp funcOp = operation->getParentOfType<func::FuncOp>();
  return funcOp && funcOp.getBody().hasOneBlock() &&
         operation->getBlock() == &funcOp.getBody().front();
}

static bool releaseFollowsOwnedUses(Operation *acquire, Operation *release) {
  SmallVector<Operation *> acquires = {acquire};
  DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);
  Operation *lastOwnedUse = findLastDFBAcquireOwnedUse(interval);
  return lastOwnedUse == acquire || lastOwnedUse->isBeforeInBlock(release);
}

static int64_t getDefaultTileCount(Operation *operation) {
  Value dfb;
  if (auto reserveOp = dyn_cast<CBReserveOp>(operation)) {
    dfb = reserveOp.getCb();
  } else if (auto pushOp = dyn_cast<CBPushOp>(operation)) {
    dfb = pushOp.getCb();
  } else if (auto waitOp = dyn_cast<CBWaitOp>(operation)) {
    dfb = waitOp.getCb();
  } else {
    dfb = cast<CBPopOp>(operation).getCb();
  }
  return cast<CircularBufferType>(dfb.getType()).getElementsPerBlock();
}

static int64_t getTileCount(Operation *operation) {
  int64_t defaultTileCount = getDefaultTileCount(operation);
  if (auto reserveOp = dyn_cast<CBReserveOp>(operation)) {
    return static_cast<int64_t>(
        reserveOp.getNumTiles().value_or(defaultTileCount));
  }
  if (auto pushOp = dyn_cast<CBPushOp>(operation)) {
    return static_cast<int64_t>(
        pushOp.getNumTiles().value_or(defaultTileCount));
  }
  if (auto waitOp = dyn_cast<CBWaitOp>(operation)) {
    return static_cast<int64_t>(
        waitOp.getNumTiles().value_or(defaultTileCount));
  }
  auto popOp = cast<CBPopOp>(operation);
  return static_cast<int64_t>(popOp.getNumTiles().value_or(defaultTileCount));
}

static std::optional<int64_t>
getMatchedOneShotTileCount(const LogicalDFB &logicalDFB) {
  if (logicalDFB.reserves.size() != 1 || logicalDFB.pushes.size() != 1 ||
      logicalDFB.waits.size() != 1 || logicalDFB.pops.size() != 1) {
    return std::nullopt;
  }

  Operation *reserve = logicalDFB.reserves.front();
  Operation *push = logicalDFB.pushes.front();
  Operation *wait = logicalDFB.waits.front();
  Operation *pop = logicalDFB.pops.front();
  if (!isDirectFunctionBodyOperation(reserve) ||
      !isDirectFunctionBodyOperation(push) ||
      !isDirectFunctionBodyOperation(wait) ||
      !isDirectFunctionBodyOperation(pop)) {
    return std::nullopt;
  }
  if (reserve->getBlock() != push->getBlock() ||
      wait->getBlock() != pop->getBlock() || !reserve->isBeforeInBlock(push) ||
      !wait->isBeforeInBlock(pop)) {
    return std::nullopt;
  }
  if (!releaseFollowsOwnedUses(reserve, push) ||
      !releaseFollowsOwnedUses(wait, pop)) {
    return std::nullopt;
  }

  int64_t reserveTiles = getTileCount(reserve);
  if (reserveTiles <= 0 || reserveTiles != getTileCount(push) ||
      reserveTiles != getTileCount(wait) || reserveTiles != getTileCount(pop)) {
    return std::nullopt;
  }
  return reserveTiles;
}

static SmallVector<unsigned>
findMinimalEvents(ArrayRef<unsigned> events,
                  const HappensBeforeGraph &happensBeforeGraph) {
  SmallVector<unsigned> minimalEvents;
  for (unsigned candidate : events) {
    bool hasPredecessor = llvm::any_of(events, [&](unsigned otherEvent) {
      return happensBeforeGraph.strictlyPrecedes(otherEvent, candidate);
    });
    if (!hasPredecessor && !llvm::is_contained(minimalEvents, candidate)) {
      minimalEvents.push_back(candidate);
    }
  }
  return minimalEvents;
}

static void computeLogicalDFBFrontiers(
    LogicalDFB &logicalDFB, const HappensBeforeGraph &happensBeforeGraph,
    const DenseMap<Operation *, EventPair> &operationEvents) {
  if (!logicalDFB.transactionTileCount.has_value()) {
    return;
  }

  logicalDFB.producerThread =
      logicalDFB.reserves.front()->getParentOfType<func::FuncOp>();
  logicalDFB.consumerThread =
      logicalDFB.waits.front()->getParentOfType<func::FuncOp>();
  assert(logicalDFB.producerThread && logicalDFB.consumerThread &&
         "direct DFB lifecycle operations must have enclosing functions");

  SmallVector<unsigned> useEntries;
  SmallVector<unsigned> useCompletions;
  for (Operation *runtimeUse : logicalDFB.runtimeUses) {
    std::optional<EventPair> events =
        getProjectedEvents(runtimeUse, operationEvents);
    if (!events.has_value()) {
      return;
    }
    useEntries.push_back(events->entry);
    useCompletions.push_back(events->completion);
  }
  if (useEntries.empty()) {
    return;
  }

  std::optional<EventPair> popEvents =
      getProjectedEvents(logicalDFB.pops.front(), operationEvents);
  assert(popEvents.has_value() &&
         "one-shot lifecycle operations must have direct event pairs");
  for (unsigned useCompletion : useCompletions) {
    if (useCompletion != popEvents->completion &&
        !happensBeforeGraph.strictlyPrecedes(useCompletion,
                                             popEvents->completion)) {
      return;
    }
  }

  logicalDFB.earliestEvents = findMinimalEvents(useEntries, happensBeforeGraph);
  logicalDFB.terminalEvents = {popEvents->completion};
  logicalDFB.bounded = !logicalDFB.earliestEvents.empty();
}

static bool isOrderedBefore(const LogicalDFB &before, const LogicalDFB &after,
                            const HappensBeforeGraph &happensBeforeGraph) {
  if (!before.bounded || !after.bounded) {
    return false;
  }
  return llvm::all_of(before.terminalEvents, [&](unsigned terminalEvent) {
    return llvm::all_of(after.earliestEvents, [&](unsigned earliestEvent) {
      return happensBeforeGraph.strictlyPrecedes(terminalEvent, earliestEvent);
    });
  });
}

static bool logicalDFBsConflict(const LogicalDFB &lhs, const LogicalDFB &rhs,
                                const HappensBeforeGraph &happensBeforeGraph) {
  // TT-Metal keeps each physical DFB's cumulative counters and ring pointers
  // in its producer and consumer kernel threads. An empty cut does not transfer
  // that state to different threads.
  if (lhs.type != rhs.type || lhs.producerThread != rhs.producerThread ||
      lhs.consumerThread != rhs.consumerThread) {
    return true;
  }
  if (!lhs.transactionTileCount.has_value() ||
      lhs.transactionTileCount != rhs.transactionTileCount) {
    return true;
  }
  auto dfbType = cast<CircularBufferType>(lhs.type);
  if (dfbType.getTotalElements() % *lhs.transactionTileCount != 0) {
    return true;
  }
  return !isOrderedBefore(lhs, rhs, happensBeforeGraph) &&
         !isOrderedBefore(rhs, lhs, happensBeforeGraph);
}

static std::optional<SmallVector<int32_t>>
computePhysicalAssignments(ArrayRef<LogicalDFB> logicalDFBs,
                           const HappensBeforeGraph &happensBeforeGraph,
                           ModuleOp moduleOp,
                           AnalysisFailure &analysisFailure) {
  SmallVector<unsigned> logicalIndices =
      llvm::to_vector(llvm::seq<unsigned>(0, logicalDFBs.size()));
  SmallVector<SmallVector<unsigned>> colors =
      assignGreedyIntervalColors<unsigned>(
          logicalIndices,
          [&](unsigned lhsIndex, unsigned rhsIndex) {
            const LogicalDFB &lhs = logicalDFBs[lhsIndex];
            const LogicalDFB &rhs = logicalDFBs[rhsIndex];
            return lhs.logicalId != rhs.logicalId
                       ? lhs.logicalId < rhs.logicalId
                       : lhsIndex < rhsIndex;
          },
          [&](unsigned lhsIndex, unsigned rhsIndex) {
            return logicalDFBsConflict(logicalDFBs[lhsIndex],
                                       logicalDFBs[rhsIndex],
                                       happensBeforeGraph);
          });

  if (colors.size() > kMaxCircularBuffers) {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "multithreaded DFB allocation needs " << colors.size()
                  << " physical indices but hardware supports at most "
                  << kMaxCircularBuffers;
    analysisFailure.set(moduleOp, messageStream.str());
    return std::nullopt;
  }

  SmallVector<int32_t> assignments(logicalDFBs.size(), -1);
  for (auto indexedColor : llvm::enumerate(colors)) {
    int32_t physicalIndex = static_cast<int32_t>(indexedColor.index());
    for (unsigned logicalIndex : indexedColor.value()) {
      assignments[logicalIndex] = physicalIndex;
    }
  }

  for (unsigned lhsIndex = 0; lhsIndex < logicalDFBs.size(); ++lhsIndex) {
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < logicalDFBs.size();
         ++rhsIndex) {
      if (assignments[lhsIndex] != assignments[rhsIndex]) {
        continue;
      }
      if (logicalDFBsConflict(logicalDFBs[lhsIndex], logicalDFBs[rhsIndex],
                              happensBeforeGraph)) {
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "DFB allocator assigned interfering logical DFBs "
                      << logicalDFBs[lhsIndex].logicalId << " and "
                      << logicalDFBs[rhsIndex].logicalId
                      << " to physical index " << assignments[lhsIndex];
        analysisFailure.set(moduleOp, messageStream.str());
        return std::nullopt;
      }
    }
  }

  return assignments;
}

} // namespace

DFBMultithreadedLivenessAnalysis::DFBMultithreadedLivenessAnalysis(
    Operation *operation) {
  ModuleOp moduleOp = cast<ModuleOp>(operation);
  AnalysisFailure analysisFailure;
  SmallVector<LogicalDFB, 0> logicalDFBs;
  DenseMap<Operation *, unsigned> bindToLogicalDFB;
  if (!collectLogicalDFBs(moduleOp, logicalDFBs, bindToLogicalDFB,
                          analysisFailure)) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  if (logicalDFBs.empty()) {
    return;
  }

  HappensBeforeGraph happensBeforeGraph;
  DenseMap<Operation *, EventPair> operationEvents;
  buildProgramOrderGraph(moduleOp, happensBeforeGraph, operationEvents);

  for (LogicalDFB &logicalDFB : logicalDFBs) {
    std::optional<int64_t> transactionTileCount =
        getMatchedOneShotTileCount(logicalDFB);
    if (!transactionTileCount.has_value()) {
      continue;
    }
    logicalDFB.transactionTileCount = transactionTileCount;
    auto pushEventsIt = operationEvents.find(logicalDFB.pushes.front());
    auto waitEventsIt = operationEvents.find(logicalDFB.waits.front());
    assert(pushEventsIt != operationEvents.end() &&
           waitEventsIt != operationEvents.end() &&
           "one-shot lifecycle operations must have direct event pairs");
    EventPair pushEvents = pushEventsIt->second;
    EventPair waitEvents = waitEventsIt->second;
    happensBeforeGraph.addEdge(pushEvents.completion, waitEvents.completion);
  }

  happensBeforeGraph.computeReachability();
  for (LogicalDFB &logicalDFB : logicalDFBs) {
    if (!logicalDFB.transactionTileCount.has_value()) {
      continue;
    }
    computeLogicalDFBFrontiers(logicalDFB, happensBeforeGraph, operationEvents);
  }

  std::optional<SmallVector<int32_t>> physicalAssignments =
      computePhysicalAssignments(logicalDFBs, happensBeforeGraph, moduleOp,
                                 analysisFailure);
  if (!physicalAssignments.has_value()) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }

  for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
    int32_t physicalIndex = (*physicalAssignments)[indexedLogicalDFB.index()];
    physicalSlotCount = std::max(physicalSlotCount, physicalIndex + 1);
    const LogicalDFB &logicalDFB = indexedLogicalDFB.value();
    assignments.push_back({logicalDFB.logicalId, physicalIndex, logicalDFB.type,
                           logicalDFB.declarations, logicalDFB.bounded});
  }
}

} // namespace mlir::tt::ttl
