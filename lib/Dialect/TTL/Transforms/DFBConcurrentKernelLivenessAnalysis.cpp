// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBConcurrentKernelLivenessAnalysis.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "DFBAnalysisFailure.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

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
#include <optional>

//===----------------------------------------------------------------------===//
// Concurrent Kernel DFB Liveness Analysis
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttl {

namespace {

/// Entry and completion event indices for one top-level kernel operation.
///
/// Every pair has an entry-to-completion edge. Program order and DFB protocol
/// edges connect pairs from the same or different kernel functions.
struct EventPair {
  unsigned entry = 0;
  unsigned completion = 0;
};

/// Directed event graph used to answer strict happens-before queries.
///
/// `strictlyPrecedes` treats mutually reachable events as unordered because a
/// cycle does not prove that either event completes first.
class HappensBeforeGraph {
public:
  /// Creates entry and completion events and orders entry before completion.
  EventPair addOperation() {
    unsigned entry = addEvent();
    unsigned completion = addEvent();
    addEdge(entry, completion);
    return {entry, completion};
  }

  /// Adds a happens-before edge between existing events.
  void addEdge(unsigned source, unsigned destination) {
    assert(source < successors.size() && destination < successors.size());
    successors[source].push_back(destination);
  }

  /// Computes transitive reachability for all event pairs.
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

  /// Returns true only when `source` reaches `destination` without a cycle.
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

/// Returns the declaration reached through unrealized conversion casts, or a
/// null op when the value does not resolve to `ttl.bind_cb`.
static BindCBOp getBindOp(Value dfb) {
  return traceUnrealizedCasts(dfb).getDefiningOp<BindCBOp>();
}

/// Groups declarations and runtime operations by resolved logical identity.
///
/// A DFB operand that does not resolve to `ttl.bind_cb` violates the
/// finalizer's input contract and fails the analysis.
static bool
collectLogicalDFBs(ModuleOp moduleOp,
                   const DFBLogicalIdentityAnalysis &identityAnalysis,
                   SmallVectorImpl<DFBLogicalLifecycle> &logicalDFBs,
                   DenseMap<Operation *, unsigned> &bindToLogicalDFB,
                   DFBAnalysisFailure &analysisFailure) {
  llvm::MapVector<int64_t, unsigned> idToLogicalDFB;

  moduleOp.walk([&](BindCBOp bindOp) {
    Type dfbType = bindOp.getResult().getType();
    int64_t logicalId = identityAnalysis.getLogicalId(bindOp);

    auto [logicalIt, inserted] =
        idToLogicalDFB.insert({logicalId, logicalDFBs.size()});
    unsigned logicalIndex = logicalIt->second;
    if (inserted) {
      DFBLogicalLifecycle logicalDFB;
      logicalDFB.logicalId = logicalId;
      logicalDFB.type = dfbType;
      logicalDFBs.push_back(std::move(logicalDFB));
    }

    logicalDFBs[logicalIndex].compilerCreated |=
        bindOp->hasAttr(kCompilerAllocatedAttrName);
    logicalDFBs[logicalIndex].declarations.push_back(bindOp);
    bindToLogicalDFB[bindOp.getOperation()] = logicalIndex;
  });

  moduleOp.walk([&](Operation *operation) {
    // `ttl.attach_cb` associates tensor SSA with a DFB but does not access the
    // hardware buffer or change its protocol state. Acquire ownership still
    // prevents release before uses of the attached tensor complete.
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
        return;
      }
      auto logicalIt = bindToLogicalDFB.find(bindOp.getOperation());
      assert(logicalIt != bindToLogicalDFB.end() &&
             "every bind must have a logical DFB");
      if (!llvm::is_contained(operationLogicalDFBs, logicalIt->second)) {
        operationLogicalDFBs.push_back(logicalIt->second);
      }
    }

    for (unsigned logicalIndex : operationLogicalDFBs) {
      DFBLogicalLifecycle &logicalDFB = logicalDFBs[logicalIndex];
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

/// Rejects incomplete compiler-created DFB acquire pairs.
///
/// A declaration with no acquires does not participate in the runtime protocol.
/// Once either acquire exists, both are required; a missing role usually means
/// a transformation distributed one DFB without preserving its logical ID.
static bool
verifyCompilerDFBAcquirePairs(ArrayRef<DFBLogicalLifecycle> logicalDFBs,
                              DFBAnalysisFailure &analysisFailure) {
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    if (!logicalDFB.compilerCreated ||
        logicalDFB.reserves.empty() == logicalDFB.waits.empty()) {
      continue;
    }

    bool hasReserve = !logicalDFB.reserves.empty();
    Operation *acquire =
        hasReserve ? logicalDFB.reserves.front() : logicalDFB.waits.front();
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "compiler-created logical DFB " << logicalDFB.logicalId
                  << " has " << (hasReserve ? "ttl.cb_reserve" : "ttl.cb_wait")
                  << " but no "
                  << (hasReserve ? "ttl.cb_wait" : "ttl.cb_reserve");
    analysisFailure.set(acquire, messageStream.str());
    return false;
  }
  return true;
}

/// Returns the top-level operation containing `operation`.
static Operation *getTopLevelKernelOperation(Operation *operation) {
  func::FuncOp funcOp = operation->getParentOfType<func::FuncOp>();
  if (!funcOp || funcOp.getBody().empty() || !funcOp.getBody().hasOneBlock()) {
    return nullptr;
  }

  Block &functionBody = funcOp.getBody().front();
  return operation->getBlock() == &functionBody
             ? operation
             : functionBody.findAncestorOpInBlock(*operation);
}

/// Adds program-order events for DFB-accessing operations.
///
/// Functions remain unordered until DFB protocol edges connect their event
/// sequences. Contracting unrelated operations preserves reachability between
/// every event queried by the lifetime proof.
static void
buildProgramOrderGraph(ModuleOp moduleOp,
                       ArrayRef<DFBLogicalLifecycle> logicalDFBs,
                       HappensBeforeGraph &graph,
                       DenseMap<Operation *, EventPair> &operationEventMap,
                       SmallVectorImpl<DFBOperationEventPair> &operationEvents,
                       SmallVectorImpl<DFBEventEdge> &programOrderEdges) {
  llvm::DenseSet<Operation *> modeledOperations;
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    for (Operation *runtimeUse : logicalDFB.runtimeUses) {
      if (Operation *topLevelOperation =
              getTopLevelKernelOperation(runtimeUse)) {
        modeledOperations.insert(topLevelOperation);
      }
    }
  }

  for (func::FuncOp funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (funcOp.getBody().empty() || !funcOp.getBody().hasOneBlock()) {
      continue;
    }

    std::optional<EventPair> previousEvents;
    for (Operation &operation : funcOp.getBody().front()) {
      if (!modeledOperations.contains(&operation)) {
        continue;
      }
      EventPair currentEvents = graph.addOperation();
      operationEventMap[&operation] = currentEvents;
      operationEvents.push_back(
          {&operation, currentEvents.entry, currentEvents.completion});
      programOrderEdges.push_back(
          {currentEvents.entry, currentEvents.completion});
      if (previousEvents.has_value()) {
        graph.addEdge(previousEvents->completion, currentEvents.entry);
        programOrderEdges.push_back(
            {previousEvents->completion, currentEvents.entry});
      }
      previousEvents = currentEvents;
    }
  }
}

/// Projects a nested operation to the top-level operation that contains it.
///
/// Multi-block functions and operations without a top-level ancestor cannot be
/// represented by the current event model.
static std::optional<EventPair>
getProjectedEvents(Operation *operation,
                   const DenseMap<Operation *, EventPair> &operationEvents) {
  Operation *projected = getTopLevelKernelOperation(operation);
  if (!projected) {
    return std::nullopt;
  }

  auto eventIt = operationEvents.find(projected);
  if (eventIt == operationEvents.end()) {
    return std::nullopt;
  }
  return eventIt->second;
}

/// Returns true when `operation` is directly in the function's sole body block.
static bool isDirectFunctionBodyOperation(Operation *operation) {
  func::FuncOp funcOp = operation->getParentOfType<func::FuncOp>();
  return funcOp && funcOp.getBody().hasOneBlock() &&
         operation->getBlock() == &funcOp.getBody().front();
}

/// Returns true when `release` follows every use owned by `acquire`.
static bool releaseFollowsOwnedUses(Operation *acquire, Operation *release) {
  SmallVector<Operation *> acquires = {acquire};
  DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);
  Operation *lastOwnedUse = findLastDFBAcquireOwnedUse(interval);
  return lastOwnedUse == acquire || lastOwnedUse->isBeforeInBlock(release);
}

/// Returns the implicit transaction count used when `num_tiles` is absent.
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

/// Returns the explicit or implicit transaction count of a lifecycle op.
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

/// Returns the common transaction count for a matched DFB lifecycle.
///
/// The proof requires exactly one direct reserve, push, wait, and pop in
/// protocol order, with releases following all owned uses. Repeated lifecycle
/// operations require dynamic occurrence matching that this analysis does not
/// perform.
static std::optional<int64_t>
getMatchedLifecycleTileCount(const DFBLogicalLifecycle &logicalDFB) {
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

/// Returns events that have no strict predecessor in `events`.
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

/// Computes the earliest use entries and terminal pop completion used to prove
/// storage reuse.
///
/// Every runtime use must be the pop or complete strictly before the pop.
/// Unordered or later uses leave the DFB unbounded.
static void computeLogicalDFBFrontiers(
    DFBLogicalLifecycle &logicalDFB,
    const HappensBeforeGraph &happensBeforeGraph,
    const DenseMap<Operation *, EventPair> &operationEvents) {
  if (!logicalDFB.transactionTileCount.has_value()) {
    return;
  }

  logicalDFB.producerKernel =
      logicalDFB.reserves.front()->getParentOfType<func::FuncOp>();
  logicalDFB.consumerKernel =
      logicalDFB.waits.front()->getParentOfType<func::FuncOp>();
  assert(logicalDFB.producerKernel && logicalDFB.consumerKernel &&
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
         "matched lifecycle operations must have direct event pairs");
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

/// Returns true when every terminal event of `before` strictly precedes every
/// earliest event of `after`.
static bool proveOrderedBefore(const DFBLogicalLifecycle &before,
                               const DFBLogicalLifecycle &after,
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

} // namespace

DFBConcurrentKernelLivenessAnalysis::DFBConcurrentKernelLivenessAnalysis(
    Operation *operation, AnalysisManager &analysisManager) {
  const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis =
      analysisManager.getAnalysis<DFBLogicalIdentityAnalysis>();
  if (!logicalIdentityAnalysis.succeeded()) {
    errorOperation = logicalIdentityAnalysis.getErrorOperation();
    errorMessage = logicalIdentityAnalysis.getErrorMessage().str();
    return;
  }
  analyze(operation, logicalIdentityAnalysis);
}

void DFBConcurrentKernelLivenessAnalysis::analyze(
    Operation *operation,
    const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis) {
  ModuleOp moduleOp = cast<ModuleOp>(operation);
  DFBAnalysisFailure analysisFailure;
  DenseMap<Operation *, unsigned> bindToLogicalDFB;
  if (!collectLogicalDFBs(moduleOp, logicalIdentityAnalysis, logicalDFBs,
                          bindToLogicalDFB, analysisFailure)) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  if (!verifyCompilerDFBAcquirePairs(logicalDFBs, analysisFailure)) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  if (logicalDFBs.empty()) {
    return;
  }

  HappensBeforeGraph happensBeforeGraph;
  DenseMap<Operation *, EventPair> operationEventMap;
  buildProgramOrderGraph(moduleOp, logicalDFBs, happensBeforeGraph,
                         operationEventMap, operationEvents, programOrderEdges);

  for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    std::optional<int64_t> transactionTileCount =
        getMatchedLifecycleTileCount(logicalDFB);
    if (!transactionTileCount.has_value()) {
      continue;
    }
    logicalDFB.transactionTileCount = transactionTileCount;
    auto pushEventsIt = operationEventMap.find(logicalDFB.pushes.front());
    auto waitEventsIt = operationEventMap.find(logicalDFB.waits.front());
    assert(pushEventsIt != operationEventMap.end() &&
           waitEventsIt != operationEventMap.end() &&
           "matched lifecycle operations must have direct event pairs");
    EventPair pushEvents = pushEventsIt->second;
    EventPair waitEvents = waitEventsIt->second;
    // A wait may begin before its producer publishes data. Only wait
    // completion is ordered after push completion.
    happensBeforeGraph.addEdge(pushEvents.completion, waitEvents.completion);
    matchedLifecycleEdges.push_back(
        {logicalDFB.logicalId, logicalDFB.pushes.front(),
         logicalDFB.waits.front(), pushEvents.completion, waitEvents.completion,
         *transactionTileCount});
  }

  happensBeforeGraph.computeReachability();
  for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    if (!logicalDFB.transactionTileCount.has_value()) {
      continue;
    }
    computeLogicalDFBFrontiers(logicalDFB, happensBeforeGraph,
                               operationEventMap);
  }

  orderedBefore.assign(logicalDFBs.size(), llvm::BitVector(logicalDFBs.size()));
  for (unsigned beforeIndex = 0; beforeIndex < logicalDFBs.size();
       ++beforeIndex) {
    for (unsigned afterIndex = 0; afterIndex < logicalDFBs.size();
         ++afterIndex) {
      if (proveOrderedBefore(logicalDFBs[beforeIndex], logicalDFBs[afterIndex],
                             happensBeforeGraph)) {
        orderedBefore[beforeIndex].set(afterIndex);
      }
    }
  }
}

bool DFBConcurrentKernelLivenessAnalysis::isOrderedBefore(
    unsigned beforeIndex, unsigned afterIndex) const {
  assert(beforeIndex < orderedBefore.size() &&
         afterIndex < orderedBefore.size());
  return orderedBefore[beforeIndex].test(afterIndex);
}

} // namespace mlir::tt::ttl
