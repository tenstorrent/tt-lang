// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBConcurrentKernelLivenessAnalysis.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "DFBAnalysisFailure.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

namespace {

// Entry and completion events keep operation duration distinct from source
// order and protocol completion edges.
struct EventPair {
  unsigned entry = 0;
  unsigned completion = 0;
};

// Represents only ordering proved for one launch node. Cyclic reachability is
// deliberately not treated as strict order.
class HappensBeforeGraph {
public:
  // Creates distinct entry and completion events so later constraints may
  // order either execution or completion.
  EventPair addOperation() {
    unsigned entry = successors.size();
    successors.emplace_back();
    unsigned completion = successors.size();
    successors.emplace_back();
    addEdge(entry, completion);
    return {entry, completion};
  }

  // Records one proved ordering constraint without inferring its converse.
  void addEdge(unsigned source, unsigned destination) {
    assert(source < successors.size() && destination < successors.size());
    successors[source].push_back(destination);
  }

  // Visits each edge once per source, which avoids cubic closure on sparse
  // per-node program-order graphs.
  void computeReachability() {
    unsigned eventCount = successors.size();
    reachable.assign(eventCount, llvm::BitVector(eventCount));
    for (unsigned source = 0; source < eventCount; ++source) {
      SmallVector<unsigned> pending = {source};
      while (!pending.empty()) {
        unsigned event = pending.pop_back_val();
        if (reachable[source].test(event)) {
          continue;
        }
        reachable[source].set(event);
        pending.append(successors[event].begin(), successors[event].end());
      }
    }
  }

  // Requires asymmetric reachability because mutually reachable events do not
  // establish a safe lifetime order.
  bool strictlyPrecedes(unsigned source, unsigned destination) const {
    assert(source < reachable.size() && destination < reachable.size());
    return source != destination && reachable[source].test(destination) &&
           !reachable[destination].test(source);
  }

private:
  SmallVector<SmallVector<unsigned>> successors;
  SmallVector<llvm::BitVector> reachable;
};

// Associates a storage access with its computed launch domain and the operation
// that prevented a precise result, when applicable.
struct AccessDomain {
  LaunchNodeDomain domain = LaunchNodeDomain::unknown();
  Operation *unanalyzableOperation = nullptr;
};

// Retains access-domain results from the shared launch-domain analysis.
struct LivenessDomainState : LaunchNodeDomainState {
  DenseMap<Operation *, AccessDomain> accessDomains;
};

using AccessExecutionCounts =
    DenseMap<const DFBAccessOccurrence *, std::optional<std::uint64_t>>;

static AccessDomain refineUnknownAccessDomainFromExecutionCounts(
    Operation *operation, AccessDomain accessDomain,
    const LivenessDomainState &domainState) {
  if (accessDomain.domain.known) {
    return accessDomain;
  }

  LaunchNodeDomain exactDomain;
  for (LaunchNodeCoord node : domainState.baseDomain.nodes) {
    std::optional<std::uint64_t> executionCount =
        getExactExecutionCountAtLaunchNode(operation, node, domainState);
    if (!executionCount) {
      return accessDomain;
    }
    if (*executionCount > 0) {
      exactDomain.nodes.insert(node);
    }
  }
  return {std::move(exactDomain), nullptr};
}

// Unknown membership is included only for counterfactual diagnostics. Reuse
// proofs continue to require exact launch-node membership.
static bool mayContainLaunchNode(const LaunchNodeDomain &domain,
                                 LaunchNodeCoord node,
                                 bool includeUnknownDomains) {
  return knownLaunchNodeDomainContains(domain, node) ||
         (includeUnknownDomains && !domain.known);
}

// Exact-zero accesses cannot contribute ordering edges even when their launch
// domain is otherwise unknown.
static bool mayAccessLaunchNode(const DFBAccessOccurrence &access,
                                LaunchNodeCoord node,
                                const AccessExecutionCounts &executionCounts,
                                bool includeUnknownDomains) {
  if (!mayContainLaunchNode(access.launchDomain, node, includeUnknownDomains)) {
    return false;
  }
  auto executionCountIt = executionCounts.find(&access);
  assert(executionCountIt != executionCounts.end() &&
         "every reported DFB access must have an execution-count fact");
  std::optional<std::uint64_t> executionCount = executionCountIt->second;
  return !executionCount || *executionCount != 0;
}

// Identifies attributes that copy a provisional physical DFB index and would
// become stale after allocation changes declaration indices.
static bool isDerivedDFBIndexAttribute(StringRef attributeName) {
  return attributeName == kUnpackToDestFp32AttrName ||
         attributeName.starts_with(kCBIndexAttrPrefix) ||
         attributeName == kBcastOutputCBIndexAttrName ||
         attributeName == kReduceOutputCBIndexAttrName ||
         attributeName == kTransposeOutputCBIndexAttrName;
}

// Resolves the hardware pointer owner only from explicit kernel semantics.
// Missing or invalid ownership attributes remain unknown because assuming a
// processor could permit unsafe physical-index reuse.
static std::optional<DFBPointerOwner>
getPointerOwner(Operation *operation, LaunchNodeCoord node,
                DFBProtocolEffectKind effect) {
  if (!isProducerDFBProtocolEffect(effect) &&
      !isConsumerDFBProtocolEffect(effect)) {
    return std::nullopt;
  }
  func::FuncOp kernel = operation->getParentOfType<func::FuncOp>();
  if (!kernel) {
    return std::nullopt;
  }
  auto thread =
      kernel->getAttrOfType<ttkernel::ThreadTypeAttr>(kKernelThreadAttrName);
  if (!thread) {
    return std::nullopt;
  }

  DFBPointerDirection direction = isProducerDFBProtocolEffect(effect)
                                      ? DFBPointerDirection::Write
                                      : DFBPointerDirection::Read;
  if (thread.getValue() == ttkernel::ThreadType::Compute) {
    DFBPointerProcessor processor = direction == DFBPointerDirection::Write
                                        ? DFBPointerProcessor::Pack
                                        : DFBPointerProcessor::Unpack;
    return DFBPointerOwner{node, processor, direction};
  }
  if (thread.getValue() != ttkernel::ThreadType::Noc) {
    return std::nullopt;
  }
  auto nocIndexAttr = kernel->getAttrOfType<IntegerAttr>(kNocIndexAttrName);
  if (!nocIndexAttr) {
    return std::nullopt;
  }
  int64_t nocIndex = nocIndexAttr.getInt();
  if (nocIndex != 0 && nocIndex != 1) {
    return std::nullopt;
  }
  DFBPointerProcessor processor =
      nocIndex == 0 ? DFBPointerProcessor::Noc0 : DFBPointerProcessor::Noc1;
  return DFBPointerOwner{node, processor, direction};
}

// Projects nested accesses to their containing top-level operation because the
// graph models only source order that is unconditional at that level.
static Operation *getTopLevelKernelOperation(Operation *operation) {
  func::FuncOp function = operation->getParentOfType<func::FuncOp>();
  if (!function || function.getBody().empty() ||
      !function.getBody().hasOneBlock()) {
    return nullptr;
  }
  Block &functionBody = function.getBody().front();
  return operation->getBlock() == &functionBody
             ? operation
             : functionBody.findAncestorOpInBlock(*operation);
}

// Missing projected events remain unknown because nested source order alone
// cannot establish cross-region execution order.
static std::optional<EventPair>
getProjectedEvents(Operation *operation,
                   const DenseMap<Operation *, EventPair> &operationEvents) {
  Operation *projected = getTopLevelKernelOperation(operation);
  if (!projected) {
    return std::nullopt;
  }
  auto eventIt = operationEvents.find(projected);
  return eventIt == operationEvents.end()
             ? std::nullopt
             : std::optional<EventPair>(eventIt->second);
}

// Effect summaries receive occurrence-specific events; concrete and opaque
// accesses use their projected operation events.
static std::optional<EventPair> getAccessEvents(
    const DFBAccessOccurrence &access,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, EventPair> &accessEvents) {
  auto accessEventIt = accessEvents.find(&access);
  if (accessEventIt != accessEvents.end()) {
    return accessEventIt->second;
  }
  return getProjectedEvents(access.operation, operationEvents);
}

// Requires a release to follow every use owned by its acquisition; textual
// acquire/release order alone does not prove storage quiescence.
static bool releaseFollowsOwnedUses(Operation *acquire, Operation *release) {
  if (acquire->getBlock() != release->getBlock()) {
    return false;
  }
  SmallVector<Operation *> acquires = {acquire};
  DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);
  Operation *lastOwnedUse = findLastDFBAcquireOwnedUse(interval);
  return lastOwnedUse == acquire || lastOwnedUse->isBeforeInBlock(release);
}

// Finds every access without a proved predecessor so all possible lifetime
// starts constrain reuse.
static SmallVector<unsigned> findMinimalEntryEvents(
    ArrayRef<const DFBAccessOccurrence *> accesses,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, EventPair> &accessEvents) {
  SmallVector<unsigned> minimal;
  for (const DFBAccessOccurrence *candidate : accesses) {
    std::optional<EventPair> candidateEvents =
        getAccessEvents(*candidate, operationEvents, accessEvents);
    if (!candidateEvents) {
      continue;
    }
    bool hasPredecessor = llvm::any_of(accesses, [&](const auto *other) {
      std::optional<EventPair> otherEvents =
          getAccessEvents(*other, operationEvents, accessEvents);
      return otherEvents &&
             graph.strictlyPrecedes(otherEvents->entry, candidateEvents->entry);
    });
    if (!hasPredecessor &&
        !llvm::is_contained(minimal, candidateEvents->entry)) {
      minimal.push_back(candidateEvents->entry);
    }
  }
  return minimal;
}

// Requires a custom function that consumes a physical index to name the same
// logical DFB as a direct storage dependency.
static LogicalResult verifyCustomFunctionIndexDependency(
    DFBAccessOpInterface access, int64_t logicalId,
    const DFBLogicalIdentityAnalysis &identityAnalysis,
    DFBAnalysisFailure &analysisFailure) {
  SmallVector<int64_t> dependencyIds;
  for (Value operand : access.getDFBDependencyOperands()) {
    FailureOr<int64_t> dependencyLogicalId =
        identityAnalysis.getLogicalId(operand);
    if (succeeded(dependencyLogicalId)) {
      dependencyIds.push_back(*dependencyLogicalId);
    }
  }

  if (!llvm::is_contained(dependencyIds, logicalId)) {
    analysisFailure.set(
        access.getOperation(),
        "custom function consumes the physical index for logical DFB " +
            std::to_string(logicalId) +
            " without listing that DFB as a dependency operand");
    return failure();
  }
  return success();
}

// Integer comparisons derive predicates from an index rather than another
// index value. Every other pure result remains conservative.
static void appendPhysicalIndexResults(Operation *operation,
                                       SmallVectorImpl<Value> &pending) {
  if (isa<arith::CmpIOp>(operation)) {
    return;
  }
  pending.append(operation->result_begin(), operation->result_end());
}

// Verifies every transitive use of one physical DFB index. Pure SSA operations
// propagate the dependency conservatively to index-capable results. Calls,
// terminators, region-bearing operations, resultless consumers and
// side-effecting operations are rejected because the analysis cannot prove
// where the integer is consumed.
static LogicalResult
verifyPhysicalIndexUses(GetDfbIdOp getId,
                        const DFBLogicalIdentityAnalysis &identityAnalysis,
                        DFBAnalysisFailure &analysisFailure) {
  FailureOr<int64_t> logicalId = identityAnalysis.getLogicalId(getId.getDfb());
  if (failed(logicalId)) {
    analysisFailure.set(
        getId, "ttl.get_dfb_id operand must resolve to a logical DFB before "
               "physical index allocation");
    return failure();
  }

  SmallVector<Value> pending = {getId.getResult()};
  DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second) {
      continue;
    }
    for (OpOperand &use : value.getUses()) {
      Operation *consumer = use.getOwner();
      if (auto access = dyn_cast<DFBAccessOpInterface>(consumer)) {
        if (failed(verifyCustomFunctionIndexDependency(
                access, *logicalId, identityAnalysis, analysisFailure))) {
          return failure();
        }
        appendPhysicalIndexResults(consumer, pending);
        continue;
      }
      if (isa<CallOpInterface>(consumer) ||
          consumer->hasTrait<OpTrait::IsTerminator>() ||
          consumer->getNumRegions() != 0 || consumer->getNumResults() == 0 ||
          !isPure(consumer)) {
        analysisFailure.set(consumer,
                            "physical index for logical DFB " +
                                std::to_string(*logicalId) +
                                " escapes through an unsupported operation");
        return failure();
      }
      appendPhysicalIndexResults(consumer, pending);
    }
  }
  return success();
}

// Collects logical DFB declarations and storage accesses in one module walk.
// Stale copied indices, malformed identities, and untracked physical-index
// escapes are rejected before launch-domain or lifetime analysis begins.
static LogicalResult collectLogicalDFBs(
    ModuleOp module, const DFBLogicalIdentityAnalysis &identityAnalysis,
    SmallVectorImpl<DFBLogicalLifecycle> &logicalDFBs,
    SmallVectorImpl<Operation *> &unknownAccessOperations,
    DFBAnalysisFailure &analysisFailure, bool &dependsOnLaunchNode) {
  llvm::MapVector<int64_t, unsigned> logicalIndexById;
  for (const DFBLogicalIdentityAssignment &assignment :
       identityAnalysis.getAssignments()) {
    BindCBOp declaration = assignment.declaration;
    int64_t logicalId = assignment.logicalId;
    auto [logicalIt, inserted] =
        logicalIndexById.insert({logicalId, logicalDFBs.size()});
    unsigned logicalIndex = logicalIt->second;
    if (inserted) {
      DFBLogicalLifecycle logicalDFB;
      logicalDFB.logicalId = logicalId;
      logicalDFB.type = declaration.getResult().getType();
      logicalDFB.tensorBacking = declaration.getTensorBackingAttr();
      logicalDFB.compilerCreated =
          declaration->hasAttr(kCompilerAllocatedAttrName);
      logicalDFBs.push_back(std::move(logicalDFB));
    } else {
      logicalDFBs[logicalIndex].compilerCreated &=
          declaration->hasAttr(kCompilerAllocatedAttrName);
    }
    logicalDFBs[logicalIndex].declarations.push_back(declaration);
  }

  WalkResult collectionResult = module.walk([&](Operation *operation) {
    for (NamedAttribute attribute : operation->getAttrs()) {
      StringRef attributeName = attribute.getName().getValue();
      if (!isDerivedDFBIndexAttribute(attributeName)) {
        continue;
      }
      analysisFailure.set(
          operation,
          ("contains derived DFB-index attribute '" + attributeName +
           "' before DFB index finalization; run ttl-finalize-dfb-indices "
           "before ttl-set-compute-kernel-config and "
           "ttl-annotate-cb-associations")
              .str());
      return WalkResult::interrupt();
    }
    dependsOnLaunchNode |=
        isa<CoreXOp, CoreYOp, CreatePipeOp, PipeNetPredicateOpInterface>(
            operation);
    if (auto getId = dyn_cast<GetDfbIdOp>(operation);
        getId && failed(verifyPhysicalIndexUses(getId, identityAnalysis,
                                                analysisFailure))) {
      return WalkResult::interrupt();
    }
    auto access = dyn_cast<DFBAccessOpInterface>(operation);
    if (access) {
      // Static index references have no SSA consumer for the general index-use
      // analysis to visit, so validate their declared storage dependency here.
      for (Value indexDFB : access.getDFBIndexOperands()) {
        FailureOr<int64_t> logicalId = identityAnalysis.getLogicalId(indexDFB);
        if (failed(logicalId)) {
          analysisFailure.set(
              operation,
              "DFB index template argument must resolve to a logical DFB "
              "before physical index allocation");
          return WalkResult::interrupt();
        }
        if (failed(verifyCustomFunctionIndexDependency(
                access, *logicalId, identityAnalysis, analysisFailure))) {
          return WalkResult::interrupt();
        }
      }
      if (access.hasUnknownDFBAccess()) {
        unknownAccessOperations.push_back(operation);
      }
    }
    if (!mayAccessDFBStorage(operation)) {
      return WalkResult::advance();
    }
    SmallVector<Value> dfbOperands;
    if (access) {
      dfbOperands = access.getDFBDependencyOperands();
    } else {
      llvm::append_range(dfbOperands, operation->getOperands());
    }
    SmallVector<std::optional<unsigned>> dependencyLogicalIndices;
    dependencyLogicalIndices.reserve(dfbOperands.size());
    for (Value operand : dfbOperands) {
      if (!isa<CircularBufferType>(operand.getType())) {
        dependencyLogicalIndices.push_back(std::nullopt);
        continue;
      }
      FailureOr<int64_t> logicalId = identityAnalysis.getLogicalId(operand);
      if (failed(logicalId)) {
        analysisFailure.set(
            operation,
            "DFB operand must resolve to ttl.bind_cb before physical index "
            "allocation");
        return WalkResult::interrupt();
      }
      auto logicalIt = logicalIndexById.find(*logicalId);
      assert(logicalIt != logicalIndexById.end() &&
             "resolved DFB identity must have a logical lifecycle");
      dependencyLogicalIndices.push_back(logicalIt->second);
    }
    if (!access) {
      SmallVector<unsigned> uniqueLogicalIndices;
      for (std::optional<unsigned> logicalIndex : dependencyLogicalIndices) {
        if (!logicalIndex ||
            llvm::is_contained(uniqueLogicalIndices, *logicalIndex)) {
          continue;
        }
        uniqueLogicalIndices.push_back(*logicalIndex);
      }
      for (unsigned logicalIndex : uniqueLogicalIndices) {
        logicalDFBs[logicalIndex].accesses.push_back(
            {operation, std::nullopt, 0, 0, LaunchNodeDomain::unknown(),
             nullptr});
      }
      return WalkResult::advance();
    }

    // Preserve dependency occurrences because aliased operands may have
    // different summaries. An occurrence without an effect remains opaque for
    // the operation's complete duration.
    llvm::BitVector effectedDependencies(dfbOperands.size());
    for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
      assert(effect.dependencyIndex < dependencyLogicalIndices.size() &&
             "protocol effect must reference a dependency occurrence");
      std::optional<unsigned> logicalIndex =
          dependencyLogicalIndices[effect.dependencyIndex];
      assert(logicalIndex &&
             "protocol effect dependency must have dataflow buffer type");
      assert(effect.dfb == dfbOperands[effect.dependencyIndex] &&
             "protocol effect value must match its dependency occurrence");
      logicalDFBs[*logicalIndex].accesses.push_back(
          {operation, effect.kind, effect.numTiles, effect.sequenceIndex,
           LaunchNodeDomain::unknown(), nullptr});
      effectedDependencies.set(effect.dependencyIndex);
    }
    for (auto [dependencyIndex, operand] : llvm::enumerate(dfbOperands)) {
      if (!isa<CircularBufferType>(operand.getType()) ||
          effectedDependencies.test(dependencyIndex)) {
        continue;
      }
      std::optional<unsigned> logicalIndex =
          dependencyLogicalIndices[dependencyIndex];
      assert(logicalIndex && "DFB dependencies were validated above");
      logicalDFBs[*logicalIndex].accesses.push_back(
          {operation, std::nullopt, 0, 0, LaunchNodeDomain::unknown(),
           nullptr});
    }
    return WalkResult::advance();
  });
  if (collectionResult.wasInterrupted()) {
    return failure();
  }

  for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    if (!logicalDFB.compilerCreated || logicalDFB.accesses.empty()) {
      continue;
    }
    auto hasEffect = [&](DFBProtocolEffectKind effect) {
      return llvm::any_of(logicalDFB.accesses,
                          [&](const DFBAccessOccurrence &access) {
                            return access.protocolEffect == effect;
                          });
    };
    StringRef missingOperation;
    if (!hasEffect(DFBProtocolEffectKind::Reserve)) {
      missingOperation = "ttl.cb_reserve";
    } else if (!hasEffect(DFBProtocolEffectKind::Push)) {
      missingOperation = "ttl.cb_push";
    } else if (!hasEffect(DFBProtocolEffectKind::Wait)) {
      missingOperation = "ttl.cb_wait";
    } else if (!hasEffect(DFBProtocolEffectKind::Pop)) {
      missingOperation = "ttl.cb_pop";
    } else {
      continue;
    }
    analysisFailure.set(
        logicalDFB.declarations.front(),
        ("compiler-allocated logical DFB has a partial lifecycle: missing " +
         missingOperation)
            .str());
    return failure();
  }
  return success();
}

// Collects complete per-access counts only for the enabled debug report.
// Counts are cached by operation because one call may describe several DFB
// effects or dependencies.
static AccessExecutionCounts
collectAccessExecutionCounts(ArrayRef<DFBLogicalLifecycle> logicalDFBs,
                             LaunchNodeCoord node,
                             const LaunchNodeDomainState &domainState) {
  AccessExecutionCounts executionCounts;
  DenseMap<Operation *, std::optional<std::uint64_t>> operationCounts;
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      if (!mayContainLaunchNode(access.launchDomain, node,
                                /*includeUnknownDomains=*/true)) {
        continue;
      }
      auto operationCountIt = operationCounts.find(access.operation);
      if (operationCountIt == operationCounts.end()) {
        std::optional<std::uint64_t> executionCount =
            getExactExecutionCountAtLaunchNode(access.operation, node,
                                               domainState);
        operationCountIt =
            operationCounts.try_emplace(access.operation, executionCount).first;
      }
      executionCounts.try_emplace(&access, operationCountIt->second);
    }
  }
  return executionCounts;
}

// Builds source-order events only for accesses active on `node`. Direct
// protocol effects receive separate events in their declared sequence;
// operations in different kernels remain concurrent unless protocol edges
// order them.
static void buildProgramOrderGraph(
    ModuleOp module, ArrayRef<DFBLogicalLifecycle> logicalDFBs,
    LaunchNodeCoord node, HappensBeforeGraph &graph,
    DenseMap<Operation *, EventPair> &operationEvents,
    DenseMap<const DFBAccessOccurrence *, EventPair> &accessEvents,
    const AccessExecutionCounts *executionCounts, bool includeUnknownDomains) {
  llvm::DenseSet<Operation *> modeledOperations;
  DenseMap<Operation *, SmallVector<const DFBAccessOccurrence *>>
      directProtocolAccesses;
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      bool mayAccess = executionCounts
                           ? mayAccessLaunchNode(access, node, *executionCounts,
                                                 includeUnknownDomains)
                           : mayContainLaunchNode(access.launchDomain, node,
                                                  includeUnknownDomains);
      if (!mayAccess) {
        continue;
      }
      if (Operation *projected = getTopLevelKernelOperation(access.operation)) {
        modeledOperations.insert(projected);
        if (access.protocolEffect && projected == access.operation) {
          directProtocolAccesses[projected].push_back(&access);
        }
      }
    }
  }
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.getBody().empty() || !function.getBody().hasOneBlock()) {
      continue;
    }
    std::optional<EventPair> previousEvents;
    for (Operation &operation : function.getBody().front()) {
      if (!modeledOperations.contains(&operation)) {
        continue;
      }
      EventPair events = graph.addOperation();
      operationEvents[&operation] = events;
      if (previousEvents) {
        graph.addEdge(previousEvents->completion, events.entry);
      }
      auto protocolIt = directProtocolAccesses.find(&operation);
      if (protocolIt != directProtocolAccesses.end()) {
        llvm::sort(protocolIt->second, [](const auto *lhs, const auto *rhs) {
          return lhs->sequenceIndex < rhs->sequenceIndex;
        });
        unsigned previousCompletion = events.entry;
        for (const DFBAccessOccurrence *access : protocolIt->second) {
          EventPair effectEvents = graph.addOperation();
          accessEvents[access] = effectEvents;
          graph.addEdge(previousCompletion, effectEvents.entry);
          graph.addEdge(effectEvents.completion, events.completion);
          previousCompletion = effectEvents.completion;
        }
      }
      previousEvents = events;
    }
  }
}

// Adds the completion edge implied by each matching push/wait transaction.
// Unknown-domain edges are restricted to the counterfactual debug graph.
static void addMatchedPushWaitEdges(
    ArrayRef<DFBLogicalLifecycle> logicalDFBs, LaunchNodeCoord node,
    HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, EventPair> &accessEvents,
    const AccessExecutionCounts *executionCounts, bool includeUnknownDomains) {
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    SmallVector<const DFBAccessOccurrence *> pushes;
    SmallVector<const DFBAccessOccurrence *> waits;
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      bool mayAccess = executionCounts
                           ? mayAccessLaunchNode(access, node, *executionCounts,
                                                 includeUnknownDomains)
                           : mayContainLaunchNode(access.launchDomain, node,
                                                  includeUnknownDomains);
      if (!mayAccess) {
        continue;
      }
      if (access.protocolEffect == DFBProtocolEffectKind::Push) {
        pushes.push_back(&access);
      } else if (access.protocolEffect == DFBProtocolEffectKind::Wait) {
        waits.push_back(&access);
      }
    }
    if (pushes.size() != waits.size()) {
      continue;
    }
    for (auto [push, wait] : llvm::zip_equal(pushes, waits)) {
      if (push->numTiles != wait->numTiles) {
        continue;
      }
      std::optional<EventPair> pushEvents =
          getAccessEvents(*push, operationEvents, accessEvents);
      std::optional<EventPair> waitEvents =
          getAccessEvents(*wait, operationEvents, accessEvents);
      if (pushEvents && waitEvents) {
        graph.addEdge(pushEvents->completion, waitEvents->completion);
      }
    }
  }
}

// Derives per-node lifetime facts. Exact-domain facts control reuse;
// counterfactual facts are retained only for debug reporting.
static DFBQuiescenceProof computePerNodeLifetime(
    DFBLogicalLifecycle &logicalDFB, LaunchNodeCoord node,
    SmallVectorImpl<DFBPerNodeLifetime> &lifetimes,
    SmallVectorImpl<DFBPerNodeLifetimeDiagnostics> *lifetimeDiagnostics,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, EventPair> &accessEvents,
    const LaunchNodeDomainState &domainState,
    const AccessExecutionCounts *reportedExecutionCounts,
    bool includeUnknownDomains = false) {
  DFBPerNodeLifetime &lifetime = lifetimes.emplace_back();
  lifetime.node = node;
  DFBPerNodeLifetimeDiagnostics *diagnostics = nullptr;
  if (reportedExecutionCounts) {
    assert(lifetimeDiagnostics &&
           "reported execution counts require lifetime diagnostics");
    diagnostics = &lifetimeDiagnostics->emplace_back();
  } else {
    assert(!lifetimeDiagnostics &&
           "lifetime diagnostics require reported execution counts");
  }
  SmallVector<const DFBAccessOccurrence *> reserves;
  SmallVector<const DFBAccessOccurrence *> pushes;
  SmallVector<const DFBAccessOccurrence *> waits;
  SmallVector<const DFBAccessOccurrence *> pops;
  SmallVector<const DFBAccessOccurrence *> activeAccesses;
  DenseMap<const DFBAccessOccurrence *, std::optional<std::uint64_t>>
      executionCounts;
  for (auto [accessIndex, access] : llvm::enumerate(logicalDFB.accesses)) {
    if (!mayContainLaunchNode(access.launchDomain, node,
                              includeUnknownDomains)) {
      continue;
    }
    std::optional<std::uint64_t> executionCount;
    if (reportedExecutionCounts) {
      auto executionCountIt = reportedExecutionCounts->find(&access);
      assert(executionCountIt != reportedExecutionCounts->end() &&
             "every reported DFB access must have an execution-count fact");
      executionCount = executionCountIt->second;
      diagnostics->occurrences.push_back(
          {static_cast<unsigned>(accessIndex), executionCount});
    }
    if (includeUnknownDomains && executionCount && *executionCount == 0) {
      continue;
    }
    activeAccesses.push_back(&access);
    if (!access.protocolEffect) {
      continue;
    }
    if (!reportedExecutionCounts) {
      executionCount = getExactExecutionCountAtLaunchNode(access.operation,
                                                          node, domainState);
    }
    executionCounts[&access] = executionCount;
    switch (*access.protocolEffect) {
    case DFBProtocolEffectKind::Reserve:
      reserves.push_back(&access);
      break;
    case DFBProtocolEffectKind::Push:
      pushes.push_back(&access);
      break;
    case DFBProtocolEffectKind::Wait:
      waits.push_back(&access);
      break;
    case DFBProtocolEffectKind::Pop:
      pops.push_back(&access);
      break;
    }
  }

  if (includeUnknownDomains && activeAccesses.empty()) {
    assert(diagnostics &&
           "counterfactual lifetimes require allocation-report data");
    diagnostics->mayBeActive = false;
    return {};
  }

  if (reserves.empty() || pushes.empty() || waits.empty() || pops.empty()) {
    return {DFBQuiescenceFailureReason::MissingProtocolEffect,
            activeAccesses.empty() ? logicalDFB.declarations.front()
                                   : activeAccesses.front()->operation};
  }
  if (reserves.size() != pushes.size() || reserves.size() != waits.size() ||
      reserves.size() != pops.size()) {
    return {DFBQuiescenceFailureReason::MismatchedTransaction,
            activeAccesses.front()->operation};
  }

  for (const DFBAccessOccurrence *protocolAccess :
       llvm::concat<const DFBAccessOccurrence *>(reserves, pushes, waits,
                                                 pops)) {
    auto executionCountIt = executionCounts.find(protocolAccess);
    assert(executionCountIt != executionCounts.end() &&
           "active protocol access must have an execution count fact");
    std::optional<std::uint64_t> executionCount = executionCountIt->second;
    if (!executionCount || *executionCount != 1) {
      return {DFBQuiescenceFailureReason::UnsupportedControlFlow,
              protocolAccess->operation};
    }
  }

  int64_t physicalTileCount =
      cast<CircularBufferType>(logicalDFB.type).getTotalElements();
  if (physicalTileCount <= 0) {
    return {DFBQuiescenceFailureReason::MismatchedTransaction,
            reserves.front()->operation};
  }
  std::optional<DFBPointerOwner> writeOwner;
  std::optional<DFBPointerOwner> readOwner;
  for (auto [reserve, push, wait, pop] :
       llvm::zip_equal(reserves, pushes, waits, pops)) {
    if (reserve->numTiles <= 0 || physicalTileCount % reserve->numTiles != 0 ||
        reserve->numTiles != push->numTiles ||
        reserve->numTiles != wait->numTiles ||
        reserve->numTiles != pop->numTiles) {
      return {DFBQuiescenceFailureReason::MismatchedTransaction,
              reserve->operation};
    }

    std::optional<EventPair> reserveEvents =
        getAccessEvents(*reserve, operationEvents, accessEvents);
    std::optional<EventPair> pushEvents =
        getAccessEvents(*push, operationEvents, accessEvents);
    std::optional<EventPair> waitEvents =
        getAccessEvents(*wait, operationEvents, accessEvents);
    std::optional<EventPair> popEvents =
        getAccessEvents(*pop, operationEvents, accessEvents);
    bool reservePrecedesPush =
        isa<CBReserveOp>(reserve->operation) && isa<CBPushOp>(push->operation)
            ? reserve->operation->getBlock() == push->operation->getBlock() &&
                  reserve->operation->isBeforeInBlock(push->operation)
            : reserveEvents && pushEvents &&
                  graph.strictlyPrecedes(reserveEvents->completion,
                                         pushEvents->entry);
    bool waitPrecedesPop =
        isa<CBWaitOp>(wait->operation) && isa<CBPopOp>(pop->operation)
            ? wait->operation->getBlock() == pop->operation->getBlock() &&
                  wait->operation->isBeforeInBlock(pop->operation)
            : waitEvents && popEvents &&
                  graph.strictlyPrecedes(waitEvents->completion,
                                         popEvents->entry);
    if (!reserveEvents || !pushEvents || !waitEvents || !popEvents ||
        !reservePrecedesPush || !waitPrecedesPop) {
      return {DFBQuiescenceFailureReason::IncompleteUseOrder, pop->operation};
    }
    if (isa<CBReserveOp>(reserve->operation) &&
        !releaseFollowsOwnedUses(reserve->operation, push->operation)) {
      return {DFBQuiescenceFailureReason::IncompleteUseOrder, push->operation};
    }
    if (isa<CBWaitOp>(wait->operation) &&
        !releaseFollowsOwnedUses(wait->operation, pop->operation)) {
      return {DFBQuiescenceFailureReason::IncompleteUseOrder, pop->operation};
    }

    std::optional<DFBPointerOwner> reserveOwner = getPointerOwner(
        reserve->operation, node, DFBProtocolEffectKind::Reserve);
    std::optional<DFBPointerOwner> pushOwner =
        getPointerOwner(push->operation, node, DFBProtocolEffectKind::Push);
    std::optional<DFBPointerOwner> waitOwner =
        getPointerOwner(wait->operation, node, DFBProtocolEffectKind::Wait);
    std::optional<DFBPointerOwner> popOwner =
        getPointerOwner(pop->operation, node, DFBProtocolEffectKind::Pop);
    if (!reserveOwner || !pushOwner || !waitOwner || !popOwner ||
        *reserveOwner != *pushOwner || *waitOwner != *popOwner ||
        (writeOwner && *writeOwner != *reserveOwner) ||
        (readOwner && *readOwner != *waitOwner)) {
      return {DFBQuiescenceFailureReason::UnknownPointerOwner,
              reserve->operation};
    }
    writeOwner = reserveOwner;
    readOwner = waitOwner;
    lifetime.transactionTileCounts.push_back(reserve->numTiles);
  }
  lifetime.writePointerOwner = writeOwner;
  lifetime.readPointerOwner = readOwner;

  std::optional<EventPair> terminalEvents =
      getAccessEvents(*pops.back(), operationEvents, accessEvents);
  if (!terminalEvents) {
    return {DFBQuiescenceFailureReason::UnsupportedControlFlow,
            pops.back()->operation};
  }
  for (const DFBAccessOccurrence *activeAccess : activeAccesses) {
    std::optional<EventPair> useEvents =
        getAccessEvents(*activeAccess, operationEvents, accessEvents);
    if (!useEvents || (useEvents->completion != terminalEvents->completion &&
                       !graph.strictlyPrecedes(useEvents->completion,
                                               terminalEvents->completion))) {
      return {DFBQuiescenceFailureReason::IncompleteUseOrder,
              activeAccess->operation};
    }
  }
  lifetime.earliestEntryEvents = findMinimalEntryEvents(
      activeAccesses, graph, operationEvents, accessEvents);
  lifetime.terminalCompletionEvents = {terminalEvents->completion};
  if (diagnostics) {
    for (const DFBAccessOccurrence *activeAccess : activeAccesses) {
      std::optional<EventPair> activeEvents =
          getAccessEvents(*activeAccess, operationEvents, accessEvents);
      if (activeEvents && llvm::is_contained(lifetime.earliestEntryEvents,
                                             activeEvents->entry)) {
        diagnostics->earliestAccessOccurrenceIndices.push_back(
            static_cast<unsigned>(activeAccess - logicalDFB.accesses.data()));
      }
    }
    diagnostics->terminalAccessOccurrenceIndices = {
        static_cast<unsigned>(pops.back() - logicalDFB.accesses.data())};
  }
  if (lifetime.earliestEntryEvents.empty()) {
    return {DFBQuiescenceFailureReason::IncompleteUseOrder,
            pops.back()->operation};
  }
  return {};
}

// Proves non-overlap only when every possible end of `before` strictly precedes
// every possible start of `after`.
static bool proveOrderedBefore(const DFBPerNodeLifetime &before,
                               const DFBPerNodeLifetime &after,
                               const HappensBeforeGraph &graph) {
  if (!before.quiescence.proven() || !after.quiescence.proven()) {
    return false;
  }
  return llvm::all_of(before.terminalCompletionEvents, [&](unsigned terminal) {
    return llvm::all_of(after.earliestEntryEvents, [&](unsigned earliest) {
      return graph.strictlyPrecedes(terminal, earliest);
    });
  });
}

} // namespace

// Per-node facts are stored on each logical lifecycle because the conflict
// model must compare identical logical DFBs across all shared launch nodes.
const DFBPerNodeLifetime *
DFBLogicalLifecycle::findNodeLifetime(LaunchNodeCoord node) const {
  auto lifetimeIt = llvm::find_if(nodeLifetimes, [&](const auto &lifetime) {
    return lifetime.node == node;
  });
  return lifetimeIt == nodeLifetimes.end() ? nullptr : &*lifetimeIt;
}

// Logical identity is obtained through the analysis manager so every lifetime
// uses the same declaration aggregation as physical allocation.
DFBConcurrentKernelLivenessAnalysis::DFBConcurrentKernelLivenessAnalysis(
    Operation *operation, AnalysisManager &analysisManager) {
  const DFBLogicalIdentityAnalysis &identityAnalysis =
      analysisManager.getAnalysis<DFBLogicalIdentityAnalysis>();
  if (!identityAnalysis.succeeded()) {
    errorOperation = identityAnalysis.getErrorOperation();
    errorMessage = identityAnalysis.getErrorMessage().str();
    return;
  }
  analyze(operation, identityAnalysis);
}

// Completes validation and immutable fact construction before the finalizer
// mutates any DFB index or derived attribute.
void DFBConcurrentKernelLivenessAnalysis::analyze(
    Operation *operation, const DFBLogicalIdentityAnalysis &identityAnalysis) {
  ModuleOp module = cast<ModuleOp>(operation);
  DFBAnalysisFailure analysisFailure;
  bool dependsOnLaunchNode = false;
  SmallVector<Operation *> unknownAccessOperations;
  if (failed(collectLogicalDFBs(module, identityAnalysis, logicalDFBs,
                                unknownAccessOperations, analysisFailure,
                                dependsOnLaunchNode))) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  if (logicalDFBs.empty()) {
    return;
  }

  LivenessDomainState domainState;
  domainState.initialize(module);
  if (!domainState.hasLaunchGrid) {
    if (dependsOnLaunchNode) {
      for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
        logicalDFB.launchDomain = LaunchNodeDomain::unknown();
      }
      return;
    }
    // Straight-line kernel lifetimes are identical on every launched node.
    // One representative node proves the uniform relation without assuming a
    // launch-grid extent that is not yet present in the pipeline.
    domainState.hasLaunchGrid = true;
    domainState.baseDomain.nodes.insert({0, 0});
  }

  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  LaunchNodeDomainAnalysisOptions options;
  options.narrowPipeNetScopes = true;
  options.emitInvalidPipeNetDiagnostics = false;
  options.operationCallback = [&](Operation *accessOperation,
                                  const LaunchNodeDomain &domain,
                                  Operation *unanalyzableOperation) {
    domainState.accessDomains[accessOperation] = {domain,
                                                  unanalyzableOperation};
  };
  solver.load<LaunchNodeDomainAnalysis>(domainState, options);
  if (failed(solver.initializeAndRun(module))) {
    errorOperation = module;
    errorMessage = "failed to compute DFB launch-node domains";
    return;
  }
  if (domainState.sawError) {
    errorOperation = domainState.errorOperation;
    errorMessage = domainState.errorMessage;
    return;
  }

  DenseMap<Operation *, AccessDomain> refinedAccessDomains;
  auto getRefinedAccessDomain =
      [&](Operation *accessOperation) -> const AccessDomain & {
    auto refinedDomainIt = refinedAccessDomains.find(accessOperation);
    if (refinedDomainIt != refinedAccessDomains.end()) {
      return refinedDomainIt->second;
    }
    auto domainIt = domainState.accessDomains.find(accessOperation);
    AccessDomain accessDomain =
        domainIt == domainState.accessDomains.end()
            ? AccessDomain{LaunchNodeDomain::unknown(), accessOperation}
            : domainIt->second;
    return refinedAccessDomains
        .try_emplace(accessOperation,
                     refineUnknownAccessDomainFromExecutionCounts(
                         accessOperation, accessDomain, domainState))
        .first->second;
  };

  for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    if (logicalDFB.accesses.empty()) {
      logicalDFB.launchDomain = LaunchNodeDomain::unknown();
      continue;
    }
    for (DFBAccessOccurrence &access : logicalDFB.accesses) {
      const AccessDomain &accessDomain =
          getRefinedAccessDomain(access.operation);
      access.launchDomain = accessDomain.domain;
      access.unanalyzableDomainOperation = accessDomain.unanalyzableOperation;
      logicalDFB.launchDomain =
          logicalDFB.launchDomain.unionWith(access.launchDomain);
    }
  }

  // Unknown external access may name any user-managed physical allocation,
  // including one also declared by the operation. Compiler-created DFBs cannot
  // be referenced by external code without an explicit dependency.
  for (Operation *unknownAccess : unknownAccessOperations) {
    const AccessDomain &accessDomain = getRefinedAccessDomain(unknownAccess);
    for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
      if (logicalDFB.compilerCreated) {
        continue;
      }
      logicalDFB.accesses.push_back({unknownAccess, std::nullopt, 0, 0,
                                     accessDomain.domain,
                                     accessDomain.unanalyzableOperation});
      logicalDFB.launchDomain =
          logicalDFB.launchDomain.unionWith(accessDomain.domain);
    }
  }

  launchNodes.append(domainState.baseDomain.nodes.begin(),
                     domainState.baseDomain.nodes.end());
  orderedBeforeByNode.reserve(launchNodes.size());
  bool collectAllocationDiagnostics = false;
  LLVM_DEBUG(collectAllocationDiagnostics = true);
  if (collectAllocationDiagnostics) {
    for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
      logicalDFB.allocationDiagnostics =
          std::make_unique<DFBLogicalLifecycleDiagnostics>();
    }
  }
  for (LaunchNodeCoord node : launchNodes) {
    std::optional<AccessExecutionCounts> reportedExecutionCounts;
    if (collectAllocationDiagnostics) {
      reportedExecutionCounts.emplace(
          collectAccessExecutionCounts(logicalDFBs, node, domainState));
    }
    HappensBeforeGraph graph;
    DenseMap<Operation *, EventPair> operationEvents;
    DenseMap<const DFBAccessOccurrence *, EventPair> accessEvents;
    buildProgramOrderGraph(module, logicalDFBs, node, graph, operationEvents,
                           accessEvents, /*executionCounts=*/nullptr,
                           /*includeUnknownDomains=*/false);
    addMatchedPushWaitEdges(logicalDFBs, node, graph, operationEvents,
                            accessEvents, /*executionCounts=*/nullptr,
                            /*includeUnknownDomains=*/false);
    graph.computeReachability();

    for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
      if (!knownLaunchNodeDomainContains(logicalDFB.launchDomain, node)) {
        continue;
      }
      DFBLogicalLifecycleDiagnostics *allocationDiagnostics =
          logicalDFB.allocationDiagnostics.get();
      DFBQuiescenceProof proof = computePerNodeLifetime(
          logicalDFB, node, logicalDFB.nodeLifetimes,
          allocationDiagnostics
              ? &allocationDiagnostics->nodeLifetimeDiagnostics
              : nullptr,
          graph, operationEvents, accessEvents, domainState,
          reportedExecutionCounts ? &*reportedExecutionCounts : nullptr);
      logicalDFB.nodeLifetimes.back().quiescence = proof;
    }

    SmallVector<llvm::BitVector> nodeOrdering(
        logicalDFBs.size(), llvm::BitVector(logicalDFBs.size()));
    for (unsigned beforeIndex = 0; beforeIndex < logicalDFBs.size();
         ++beforeIndex) {
      const DFBPerNodeLifetime *before =
          logicalDFBs[beforeIndex].findNodeLifetime(node);
      if (!before) {
        continue;
      }
      for (unsigned afterIndex = 0; afterIndex < logicalDFBs.size();
           ++afterIndex) {
        const DFBPerNodeLifetime *after =
            logicalDFBs[afterIndex].findNodeLifetime(node);
        if (after && proveOrderedBefore(*before, *after, graph)) {
          nodeOrdering[beforeIndex].set(afterIndex);
        }
      }
    }
    orderedBeforeByNode.push_back(std::move(nodeOrdering));

    if (!collectAllocationDiagnostics) {
      continue;
    }
    HappensBeforeGraph diagnosticGraph;
    DenseMap<Operation *, EventPair> diagnosticOperationEvents;
    DenseMap<const DFBAccessOccurrence *, EventPair> diagnosticAccessEvents;
    buildProgramOrderGraph(module, logicalDFBs, node, diagnosticGraph,
                           diagnosticOperationEvents, diagnosticAccessEvents,
                           &*reportedExecutionCounts,
                           /*includeUnknownDomains=*/true);
    addMatchedPushWaitEdges(logicalDFBs, node, diagnosticGraph,
                            diagnosticOperationEvents, diagnosticAccessEvents,
                            &*reportedExecutionCounts,
                            /*includeUnknownDomains=*/true);
    diagnosticGraph.computeReachability();
    for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
      if (logicalDFB.launchDomain.known) {
        continue;
      }
      DFBLogicalLifecycleDiagnostics &allocationDiagnostics =
          *logicalDFB.allocationDiagnostics;
      DFBQuiescenceProof proof = computePerNodeLifetime(
          logicalDFB, node, allocationDiagnostics.counterfactualNodeLifetimes,
          &allocationDiagnostics.counterfactualNodeLifetimeDiagnostics,
          diagnosticGraph, diagnosticOperationEvents, diagnosticAccessEvents,
          domainState, &*reportedExecutionCounts,
          /*includeUnknownDomains=*/true);
      allocationDiagnostics.counterfactualNodeLifetimes.back().quiescence =
          proof;
    }
  }

  for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    logicalDFB.bounded = logicalDFB.launchDomain.known &&
                         !logicalDFB.nodeLifetimes.empty() &&
                         llvm::all_of(logicalDFB.nodeLifetimes,
                                      [](const DFBPerNodeLifetime &lifetime) {
                                        return lifetime.quiescence.proven();
                                      });
  }
}

// The planner queries cached reachability so allocation cannot depend on a
// second IR traversal or on later mutation order.
bool DFBConcurrentKernelLivenessAnalysis::isOrderedBefore(
    unsigned beforeIndex, unsigned afterIndex, LaunchNodeCoord node) const {
  auto nodeIt = llvm::find(launchNodes, node);
  assert(nodeIt != launchNodes.end() && "node must be in the launch grid");
  unsigned nodeIndex = nodeIt - launchNodes.begin();
  assert(beforeIndex < orderedBeforeByNode[nodeIndex].size() &&
         afterIndex < orderedBeforeByNode[nodeIndex].size());
  return orderedBeforeByNode[nodeIndex][beforeIndex].test(afterIndex);
}

} // namespace mlir::tt::ttl
