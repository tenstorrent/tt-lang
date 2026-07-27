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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

//===----------------------------------------------------------------------===//
// Multithreaded DFB Liveness Analysis
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
/// `strictlyPrecedes` excludes events in the same cycle because a cyclic model
/// does not prove that either event completes first.
class HappensBeforeGraph {
public:
  /// Adds an operation's entry and completion events and their required edge.
  EventPair addOperation() {
    unsigned entry = addEvent();
    unsigned completion = addEvent();
    addEdge(entry, completion);
    return {entry, completion};
  }

  /// Adds one required ordering edge between existing events.
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

/// Module-wide state collected for one logical dataflow buffer.
///
/// Declarations are grouped by `logicalId`. A bounded DFB has one matched
/// lifecycle, stable producer and consumer functions, and non-empty earliest
/// and terminal event frontiers.
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

  // These fields are populated only after the one-shot lifecycle and event
  // projection checks succeed.
  SmallVector<unsigned> earliestEvents;
  SmallVector<unsigned> terminalEvents;
  std::optional<int64_t> transactionTileCount;
  bool bounded = false;
};

/// First diagnostic discovered by an analysis walk.
///
/// Analysis helpers record failures instead of emitting diagnostics. Retaining
/// the first failure preserves the operation that violated the input contract.
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

/// Returns the declaration reached through unrealized conversion casts.
static BindCBOp getBindOp(Value dfb) {
  return traceUnrealizedCasts(dfb).getDefiningOp<BindCBOp>();
}

/// Groups declarations and runtime operations by resolved logical identity.
///
/// A DFB operand that does not resolve to `ttl.bind_cb` violates the
/// finalizer's input contract and terminates collection.
static bool
collectLogicalDFBs(ModuleOp moduleOp,
                   const DFBLogicalIdentityAnalysis &identityAnalysis,
                   SmallVectorImpl<LogicalDFB> &logicalDFBs,
                   DenseMap<Operation *, unsigned> &bindToLogicalDFB,
                   AnalysisFailure &analysisFailure) {
  llvm::MapVector<int64_t, unsigned> idToLogicalDFB;

  moduleOp.walk([&](BindCBOp bindOp) {
    Type dfbType = bindOp.getResult().getType();
    int64_t logicalId = identityAnalysis.getLogicalId(bindOp);

    auto [logicalIt, inserted] =
        idToLogicalDFB.insert({logicalId, logicalDFBs.size()});
    unsigned logicalIndex = logicalIt->second;
    if (inserted) {
      LogicalDFB logicalDFB;
      logicalDFB.logicalId = logicalId;
      logicalDFB.type = dfbType;
      logicalDFBs.push_back(std::move(logicalDFB));
    }

    logicalDFBs[logicalIndex].declarations.push_back(bindOp);
    bindToLogicalDFB[bindOp.getOperation()] = logicalIndex;
  });

  moduleOp.walk([&](Operation *operation) {
    // `ttl.attach_cb` associates tensor SSA with a DFB but does not access the
    // hardware buffer or change its protocol state.
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

/// Adds program-order events for every single-block kernel function.
///
/// Functions remain unordered until DFB protocol edges connect their event
/// sequences.
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

/// Projects a nested operation to the top-level operation that contains it.
///
/// Multi-block functions and operations without a top-level ancestor cannot be
/// represented by the current event model.
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

/// Returns true when `operation` has an event pair without projection.
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

/// Returns the common transaction count for a matched one-shot lifecycle.
///
/// A lifecycle is matched only when it has one direct reserve, push, wait, and
/// pop in protocol order, with releases following their owned uses. Returning
/// no value keeps the DFB unbounded and prevents physical reuse.
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

/// Returns the minimal antichain of `events` under happens-before.
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

/// Computes the lifetime frontiers used to prove storage reuse.
///
/// Every runtime use must have a projected event whose completion does not
/// follow the terminal pop. Otherwise the DFB remains unbounded.
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

/// Returns true when `before` ends before every earliest event of `after`.
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

/// Returns true unless two logical DFBs can safely share physical storage.
///
/// Sharing requires identical storage, transaction, and kernel-participant
/// state plus a proven lifetime order in one direction.
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

/// Colors the interference graph and validates every shared physical index.
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

DFBLogicalIdentityAnalysis::DFBLogicalIdentityAnalysis(Operation *operation) {
  ModuleOp moduleOp = cast<ModuleOp>(operation);
  int64_t maxExplicitId = -1;
  int64_t compilerDeclarationCount = 0;
  BindCBOp firstCompilerDeclaration;

  // Discover the complete explicit ID range before generating IDs. A single
  // walk that assigned IDs immediately could collide with a later declaration.
  moduleOp.walk([&](BindCBOp bindOp) {
    if (!errorMessage.empty()) {
      return;
    }
    if (auto dfbId = bindOp.getDfbId()) {
      int64_t logicalId = dfbId->getSExtValue();
      maxExplicitId = std::max(maxExplicitId, logicalId);
      return;
    }
    if (!bindOp->hasAttr(kCompilerAllocatedAttrName)) {
      errorOperation = bindOp;
      errorMessage =
          "user-declared DFB requires dfb_id before physical allocation";
      return;
    }
    if (!firstCompilerDeclaration) {
      firstCompilerDeclaration = bindOp;
    }
    if (compilerDeclarationCount == std::numeric_limits<int64_t>::max()) {
      errorOperation = bindOp;
      errorMessage = "too many compiler-created DFB declarations";
      return;
    }
    ++compilerDeclarationCount;
  });
  if (!errorMessage.empty()) {
    return;
  }

  if (compilerDeclarationCount > 0 &&
      maxExplicitId >
          std::numeric_limits<int64_t>::max() - compilerDeclarationCount) {
    errorOperation = firstCompilerDeclaration;
    errorMessage =
        "logical DFB identifiers leave no space for compiler-created DFBs";
    return;
  }

  int64_t nextCompilerId = compilerDeclarationCount > 0 ? maxExplicitId + 1 : 0;
  llvm::DenseMap<int64_t, BindCBOp> firstDeclarationById;
  // Type consistency is an identity invariant: declarations with one ID
  // describe one logical allocation even when they occur in different threads.
  moduleOp.walk([&](BindCBOp bindOp) {
    if (!errorMessage.empty()) {
      return;
    }
    auto dfbId = bindOp.getDfbId();
    int64_t logicalId = dfbId ? dfbId->getSExtValue() : nextCompilerId++;
    auto [firstIt, inserted] =
        firstDeclarationById.try_emplace(logicalId, bindOp);
    if (!inserted &&
        firstIt->second.getResult().getType() != bindOp.getResult().getType()) {
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream
          << "logical DFB " << logicalId
          << " has inconsistent types across thread functions: expected "
          << firstIt->second.getResult().getType() << " but found "
          << bindOp.getResult().getType();
      errorOperation = bindOp;
      errorMessage = messageStream.str();
      return;
    }
    assignments.push_back({bindOp, logicalId});
    identities[bindOp.getOperation()] = logicalId;
  });
}

int64_t DFBLogicalIdentityAnalysis::getLogicalId(BindCBOp bindOp) const {
  auto identityIt = identities.find(bindOp.getOperation());
  assert(identityIt != identities.end() &&
         "every DFB declaration must have a resolved logical identity");
  return identityIt->second;
}

DFBMultithreadedLivenessAnalysis::DFBMultithreadedLivenessAnalysis(
    Operation *operation) {
  ModuleOp moduleOp = cast<ModuleOp>(operation);
  DFBLogicalIdentityAnalysis identityAnalysis(operation);
  if (!identityAnalysis.succeeded()) {
    errorOperation = identityAnalysis.getErrorOperation();
    errorMessage = identityAnalysis.getErrorMessage().str();
    return;
  }

  AnalysisFailure analysisFailure;
  SmallVector<LogicalDFB, 0> logicalDFBs;
  DenseMap<Operation *, unsigned> bindToLogicalDFB;
  if (!collectLogicalDFBs(moduleOp, identityAnalysis, logicalDFBs,
                          bindToLogicalDFB, analysisFailure)) {
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
    // A wait may begin before its producer publishes data. Only wait
    // completion is ordered after push completion.
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
