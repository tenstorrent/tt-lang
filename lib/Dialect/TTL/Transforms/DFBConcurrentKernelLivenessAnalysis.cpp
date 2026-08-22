// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBConcurrentKernelLivenessAnalysis.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "DFBAnalysisFailure.h"
#include "ttlang/Analysis/LoopIterationUtils.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
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

// First and last dynamic executions represented by one static access.
struct AccessEventSpan {
  EventPair first;
  EventPair last;
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

// Structured loops whose Cartesian iteration space executes one access once.
// Loop operation identity, rather than a numeric count, distinguishes domains.
struct StaticIterationDomain {
  SmallVector<Operation *> loops;

  bool operator==(const StaticIterationDomain &rhs) const {
    return loops == rhs.loops;
  }
};

// One statically counted run of an access occurrence.
struct AccessRun {
  const DFBAccessOccurrence *access = nullptr;
  std::uint64_t executionCount = 0;
  StaticIterationDomain iterationDomain;
  bool conditionalExecution = false;
};

using AccessRuns = DenseMap<const DFBAccessOccurrence *, AccessRun>;

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

// Proves that an access executes once in every iteration of one immutable
// structured loop nest. Region selection between the loops and the access is
// rejected because an aggregate count cannot prove per-iteration execution.
static std::optional<StaticIterationDomain> getUniformStaticIterationDomain(
    Operation *operation, std::uint64_t executionCount, LaunchNodeCoord node,
    const LivenessDomainState &domainState) {
  func::FuncOp function = operation->getParentOfType<func::FuncOp>();
  if (!function || function.getBody().empty() ||
      !function.getBody().hasOneBlock()) {
    return std::nullopt;
  }

  // Exact count-one accesses retain the existing event-graph proof. Only
  // repeated accesses require a uniform structured iteration domain.
  if (executionCount == 1) {
    return StaticIterationDomain{};
  }

  StaticIterationDomain domain;
  std::uint64_t domainExecutionCount = 1;
  Operation *nestedOperation = operation;
  while (nestedOperation->getParentRegion() != &function.getBody()) {
    Region *region = nestedOperation->getParentRegion();
    Operation *parent = region ? region->getParentOp() : nullptr;
    auto loop = dyn_cast_or_null<LoopLikeOpInterface>(parent);
    if (!parent || !region->hasOneBlock() ||
        nestedOperation->getBlock() != &region->front()) {
      return std::nullopt;
    }
    if (!loop) {
      std::optional<std::uint64_t> parentExecutionCount =
          getExactExecutionCountAtLaunchNode(parent, node, domainState);
      if (!parentExecutionCount || *parentExecutionCount != 1) {
        return std::nullopt;
      }
      nestedOperation = parent;
      continue;
    }
    SmallVector<Region *> loopRegions = loop.getLoopRegions();
    if (loopRegions.size() != 1 || loopRegions.front() != region) {
      return std::nullopt;
    }
    std::optional<std::uint64_t> tripCount = tt::getLoopTripCount(loop);
    if (!tripCount || *tripCount == 0) {
      return std::nullopt;
    }
    std::optional<std::uint64_t> product =
        llvm::checkedMulUnsigned(domainExecutionCount, *tripCount);
    if (!product) {
      return std::nullopt;
    }
    domainExecutionCount = *product;
    domain.loops.push_back(parent);
    nestedOperation = parent;
  }

  if (nestedOperation->getBlock() != &function.getBody().front() ||
      domainExecutionCount != executionCount) {
    return std::nullopt;
  }
  std::reverse(domain.loops.begin(), domain.loops.end());
  return domain;
}

// Proves an unresolved access cannot repeat. Treating the access as present
// once preserves the conditional lifetime without assuming it executes.
static bool structurallyExecutesAtMostOnce(Operation *operation) {
  func::FuncOp function = operation->getParentOfType<func::FuncOp>();
  if (!function || function.getBody().empty() ||
      !function.getBody().hasOneBlock()) {
    return false;
  }

  Operation *nestedOperation = operation;
  while (nestedOperation->getParentRegion() != &function.getBody()) {
    Region *region = nestedOperation->getParentRegion();
    Operation *parent = region ? region->getParentOp() : nullptr;
    if (!parent || !region->hasOneBlock() ||
        nestedOperation->getBlock() != &region->front() ||
        !isa<affine::AffineIfOp, scf::IfOp, scf::IndexSwitchOp,
             scf::ExecuteRegionOp>(parent)) {
      return false;
    }
    nestedOperation = parent;
  }
  return nestedOperation->getBlock() == &function.getBody().front();
}

// Possible-domain analysis includes unknown membership while exact-domain
// analysis continues to require proven launch-node membership.
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

// Effect summaries receive occurrence-specific event spans; concrete and
// opaque accesses use their projected operation events.
static std::optional<AccessEventSpan>
getAccessEventSpan(const DFBAccessOccurrence &access,
                   const DenseMap<Operation *, EventPair> &operationEvents,
                   const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
                       &accessEvents) {
  auto accessEventIt = accessEvents.find(&access);
  if (accessEventIt != accessEvents.end()) {
    return accessEventIt->second;
  }
  std::optional<EventPair> projectedEvents =
      getProjectedEvents(access.operation, operationEvents);
  return projectedEvents ? std::optional<AccessEventSpan>(AccessEventSpan{
                               *projectedEvents, *projectedEvents})
                         : std::nullopt;
}

// Proves ordering between corresponding executions, not all-before-all
// ordering across the complete runs.
static bool runPrecedesWithinEachIteration(const AccessRun &before,
                                           const AccessRun &after) {
  if (!(before.iterationDomain == after.iterationDomain) ||
      before.executionCount != after.executionCount) {
    return false;
  }
  if (before.access->operation == after.access->operation) {
    return before.access->protocolEffect && after.access->protocolEffect &&
           before.access->sequenceIndex < after.access->sequenceIndex;
  }
  if (before.conditionalExecution && after.conditionalExecution) {
    return before.access->operation->getBlock() ==
               after.access->operation->getBlock() &&
           before.access->operation->isBeforeInBlock(after.access->operation);
  }
  if (before.executionCount <= 1) {
    return false;
  }
  return before.access->operation->getBlock() ==
             after.access->operation->getBlock() &&
         before.access->operation->isBeforeInBlock(after.access->operation);
}

// Requires a release to follow every use owned by its acquisition; textual
// acquire/release order alone does not prove storage quiescence.
// `sameKindAcquires` contains every same-DFB acquisition of the same kind.
static bool releaseFollowsOwnedUses(Operation *acquire, Operation *release,
                                    ArrayRef<Operation *> sameKindAcquires) {
  if (acquire->getBlock() != release->getBlock()) {
    return false;
  }
  DFBAcquireInterval interval =
      makeDFBAcquireInterval(acquire, sameKindAcquires);
  Operation *lastOwnedUse = findLastDFBAcquireOwnedUse(interval);
  return lastOwnedUse == acquire || lastOwnedUse->isBeforeInBlock(release);
}

// Finds every access without a proved predecessor so all possible lifetime
// starts constrain reuse.
static SmallVector<unsigned> findMinimalEntryEvents(
    ArrayRef<const DFBAccessOccurrence *> accesses,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  SmallVector<unsigned> minimal;
  for (const DFBAccessOccurrence *candidate : accesses) {
    std::optional<AccessEventSpan> candidateEvents =
        getAccessEventSpan(*candidate, operationEvents, accessEvents);
    if (!candidateEvents) {
      continue;
    }
    bool hasPredecessor = llvm::any_of(accesses, [&](const auto *other) {
      std::optional<AccessEventSpan> otherEvents =
          getAccessEventSpan(*other, operationEvents, accessEvents);
      return otherEvents &&
             graph.strictlyPrecedes(otherEvents->first.entry,
                                    candidateEvents->first.entry);
    });
    if (!hasPredecessor &&
        !llvm::is_contained(minimal, candidateEvents->first.entry)) {
      minimal.push_back(candidateEvents->first.entry);
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

static AccessRuns
collectAccessRuns(ArrayRef<DFBLogicalLifecycle> logicalDFBs,
                  LaunchNodeCoord node, const LivenessDomainState &domainState,
                  const AccessExecutionCounts &executionCounts,
                  bool includeUnknownDomains) {
  AccessRuns runs;
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      if (!mayContainLaunchNode(access.launchDomain, node,
                                includeUnknownDomains)) {
        continue;
      }
      auto countIt = executionCounts.find(&access);
      assert(countIt != executionCounts.end() &&
             "active DFB access must have an execution-count fact");
      if (countIt->second && *countIt->second == 0) {
        continue;
      }
      if (!countIt->second) {
        if (structurallyExecutesAtMostOnce(access.operation)) {
          runs.try_emplace(
              &access, AccessRun{&access, 1, StaticIterationDomain{}, true});
        }
        continue;
      }
      std::optional<StaticIterationDomain> domain =
          getUniformStaticIterationDomain(access.operation, *countIt->second,
                                          node, domainState);
      if (!domain) {
        continue;
      }
      runs.try_emplace(&access, AccessRun{&access, *countIt->second,
                                          std::move(*domain), false});
    }
  }
  return runs;
}

// Builds source-order events only for accesses active on `node`. Direct
// protocol effects receive separate events in their declared sequence;
// operations in different kernels remain concurrent unless protocol edges
// order them.
static void buildProgramOrderGraph(
    ModuleOp module, ArrayRef<DFBLogicalLifecycle> logicalDFBs,
    LaunchNodeCoord node, HappensBeforeGraph &graph,
    DenseMap<Operation *, EventPair> &operationEvents,
    DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, const AccessRuns &accessRuns,
    bool includeUnknownDomains) {
  llvm::DenseSet<Operation *> modeledOperations;
  DenseMap<Operation *, SmallVector<const DFBAccessOccurrence *>>
      directProtocolAccesses;
  DenseMap<Operation *, SmallVector<const DFBAccessOccurrence *>>
      projectedAccesses;
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      if (!mayAccessLaunchNode(access, node, executionCounts,
                               includeUnknownDomains)) {
        continue;
      }
      if (Operation *projected = getTopLevelKernelOperation(access.operation)) {
        modeledOperations.insert(projected);
        projectedAccesses[projected].push_back(&access);
        auto runIt = accessRuns.find(&access);
        if (access.protocolEffect && projected == access.operation &&
            runIt != accessRuns.end() && runIt->second.executionCount == 1) {
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
          accessEvents[access] = {effectEvents, effectEvents};
          graph.addEdge(previousCompletion, effectEvents.entry);
          graph.addEdge(effectEvents.completion, events.completion);
          previousCompletion = effectEvents.completion;
        }
      }
      previousEvents = events;
    }
  }

  for (auto &[projected, accesses] : projectedAccesses) {
    EventPair projectedEvents = operationEvents.lookup(projected);
    for (const DFBAccessOccurrence *access : accesses) {
      auto runIt = accessRuns.find(access);
      if (runIt == accessRuns.end() || runIt->second.executionCount <= 1) {
        continue;
      }
      EventPair firstEvents = graph.addOperation();
      EventPair lastEvents = graph.addOperation();
      accessEvents[access] = {firstEvents, lastEvents};
      graph.addEdge(projectedEvents.entry, firstEvents.entry);
      graph.addEdge(firstEvents.completion, lastEvents.entry);
      graph.addEdge(lastEvents.completion, projectedEvents.completion);
    }
    for (const DFBAccessOccurrence *beforeAccess : accesses) {
      auto beforeRunIt = accessRuns.find(beforeAccess);
      auto beforeEventsIt = accessEvents.find(beforeAccess);
      if (beforeRunIt == accessRuns.end() ||
          beforeRunIt->second.executionCount <= 1 ||
          beforeEventsIt == accessEvents.end()) {
        continue;
      }
      for (const DFBAccessOccurrence *afterAccess : accesses) {
        auto afterRunIt = accessRuns.find(afterAccess);
        auto afterEventsIt = accessEvents.find(afterAccess);
        if (afterRunIt == accessRuns.end() ||
            afterRunIt->second.executionCount <= 1 ||
            afterEventsIt == accessEvents.end() ||
            !runPrecedesWithinEachIteration(beforeRunIt->second,
                                            afterRunIt->second)) {
          continue;
        }
        graph.addEdge(beforeEventsIt->second.first.completion,
                      afterEventsIt->second.first.entry);
        graph.addEdge(afterEventsIt->second.first.completion,
                      beforeEventsIt->second.last.entry);
        graph.addEdge(beforeEventsIt->second.last.completion,
                      afterEventsIt->second.last.entry);
      }
    }
  }
}

// Conditional transaction effects match only when their structured execution
// conditions are equivalent.
static bool
proveEquivalentConditionalRuns(const AccessRun &lhs, const AccessRun &rhs,
                               LaunchNodeCoord node,
                               const LaunchNodeDomainState &domainState) {
  if (!lhs.conditionalExecution && !rhs.conditionalExecution) {
    return true;
  }
  return lhs.conditionalExecution && rhs.conditionalExecution &&
         proveEquivalentConditionalExecutionAtLaunchNodes(
             lhs.access->operation, node, rhs.access->operation, node,
             domainState);
}

// Adds completion edges between matched push/wait transaction instances.
static void addMatchedPushWaitEdges(
    ArrayRef<DFBLogicalLifecycle> logicalDFBs, HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessRuns &accessRuns, LaunchNodeCoord node,
    const LaunchNodeDomainState &domainState) {
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    SmallVector<const AccessRun *> pushes;
    SmallVector<const AccessRun *> waits;
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      auto runIt = accessRuns.find(&access);
      if (runIt == accessRuns.end()) {
        continue;
      }
      if (access.protocolEffect == DFBProtocolEffectKind::Push) {
        pushes.push_back(&runIt->second);
      } else if (access.protocolEffect == DFBProtocolEffectKind::Wait) {
        waits.push_back(&runIt->second);
      }
    }

    std::size_t pushIndex = 0;
    std::size_t waitIndex = 0;
    std::uint64_t pushOffset = 0;
    std::uint64_t waitOffset = 0;
    SmallVector<std::pair<unsigned, unsigned>> synchronizationEdges;
    bool matched = true;
    while (pushIndex < pushes.size() && waitIndex < waits.size()) {
      const AccessRun &push = *pushes[pushIndex];
      const AccessRun &wait = *waits[waitIndex];
      if (push.access->numTiles != wait.access->numTiles ||
          !proveEquivalentConditionalRuns(push, wait, node, domainState)) {
        matched = false;
        break;
      }
      std::optional<AccessEventSpan> pushEvents =
          getAccessEventSpan(*push.access, operationEvents, accessEvents);
      std::optional<AccessEventSpan> waitEvents =
          getAccessEventSpan(*wait.access, operationEvents, accessEvents);
      if (!pushEvents || !waitEvents) {
        matched = false;
        break;
      }

      std::uint64_t matchedCount = std::min(push.executionCount - pushOffset,
                                            wait.executionCount - waitOffset);
      if (pushOffset == 0 && waitOffset == 0) {
        synchronizationEdges.emplace_back(pushEvents->first.completion,
                                          waitEvents->first.completion);
      }
      pushOffset += matchedCount;
      waitOffset += matchedCount;
      if (pushOffset == push.executionCount &&
          waitOffset == wait.executionCount) {
        synchronizationEdges.emplace_back(pushEvents->last.completion,
                                          waitEvents->last.completion);
      }
      if (pushOffset == push.executionCount) {
        ++pushIndex;
        pushOffset = 0;
      }
      if (waitOffset == wait.executionCount) {
        ++waitIndex;
        waitOffset = 0;
      }
    }
    matched &= pushIndex == pushes.size() && waitIndex == waits.size();
    if (!matched) {
      continue;
    }
    for (auto [pushCompletion, waitCompletion] : synchronizationEdges) {
      graph.addEdge(pushCompletion, waitCompletion);
    }
  }
}

// Proves ordering between corresponding executions with equal counts and
// iteration domains, not all-before-all ordering across the complete runs.
static bool proveRunBeforeWithinEachIteration(
    const AccessRun &before, const AccessRun &after,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  if (before.executionCount != after.executionCount ||
      !(before.iterationDomain == after.iterationDomain)) {
    return false;
  }
  if (runPrecedesWithinEachIteration(before, after)) {
    return true;
  }
  if (before.executionCount != 1 || after.executionCount != 1) {
    return false;
  }
  std::optional<AccessEventSpan> beforeEvents =
      getAccessEventSpan(*before.access, operationEvents, accessEvents);
  std::optional<AccessEventSpan> afterEvents =
      getAccessEventSpan(*after.access, operationEvents, accessEvents);
  return beforeEvents && afterEvents &&
         graph.strictlyPrecedes(beforeEvents->last.completion,
                                afterEvents->first.entry);
}

// Proves that the final execution of `before` completes before the first
// execution of `after` begins.
static bool proveAllRunExecutionsBefore(
    const AccessRun &before, const AccessRun &after,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  if (before.executionCount == 1 && after.executionCount == 1 &&
      runPrecedesWithinEachIteration(before, after)) {
    return true;
  }
  std::optional<AccessEventSpan> beforeEvents =
      getAccessEventSpan(*before.access, operationEvents, accessEvents);
  std::optional<AccessEventSpan> afterEvents =
      getAccessEventSpan(*after.access, operationEvents, accessEvents);
  return beforeEvents && afterEvents &&
         graph.strictlyPrecedes(beforeEvents->last.completion,
                                afterEvents->first.entry);
}

static bool proveAlignedAcquireReleaseRuns(
    ArrayRef<const AccessRun *> acquires, ArrayRef<const AccessRun *> releases,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  if (acquires.size() != releases.size()) {
    return false;
  }
  bool pairsAreAligned =
      llvm::all_of(llvm::zip_equal(acquires, releases), [&](auto pair) {
        const AccessRun &acquire = *std::get<0>(pair);
        const AccessRun &release = *std::get<1>(pair);
        bool nativeAcquirePrecedesRelease =
            ((isa<CBReserveOp>(acquire.access->operation) &&
              isa<CBPushOp>(release.access->operation)) ||
             (isa<CBWaitOp>(acquire.access->operation) &&
              isa<CBPopOp>(release.access->operation))) &&
            acquire.access->operation->getBlock() ==
                release.access->operation->getBlock() &&
            acquire.access->operation->isBeforeInBlock(
                release.access->operation);
        return acquire.executionCount == release.executionCount &&
               acquire.iterationDomain == release.iterationDomain &&
               acquire.access->numTiles == release.access->numTiles &&
               (nativeAcquirePrecedesRelease ||
                proveRunBeforeWithinEachIteration(
                    acquire, release, graph, operationEvents, accessEvents));
      });
  if (!pairsAreAligned) {
    return false;
  }
  for (std::size_t runIndex = 1; runIndex < acquires.size(); ++runIndex) {
    if (!proveAllRunExecutionsBefore(*releases[runIndex - 1],
                                     *acquires[runIndex], graph,
                                     operationEvents, accessEvents)) {
      return false;
    }
  }
  return true;
}

static bool
runIsInsideInterval(const AccessRun &use, const AccessRun &acquire,
                    const AccessRun &release, const HappensBeforeGraph &graph,
                    const DenseMap<Operation *, EventPair> &operationEvents,
                    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
                        &accessEvents) {
  return proveRunBeforeWithinEachIteration(acquire, use, graph, operationEvents,
                                           accessEvents) &&
         proveRunBeforeWithinEachIteration(use, release, graph, operationEvents,
                                           accessEvents);
}

static void appendTransactionRun(DFBPerNodeLifetime &lifetime,
                                 std::uint64_t executionCount,
                                 int64_t tilesPerExecution) {
  if (executionCount == 0) {
    return;
  }
  if (!lifetime.transactionRuns.empty() &&
      lifetime.transactionRuns.back().tilesPerExecution == tilesPerExecution) {
    lifetime.transactionRuns.back().executionCount += executionCount;
    return;
  }
  lifetime.transactionRuns.push_back({executionCount, tilesPerExecution});
}

// Derives exact-domain or possible-domain per-node lifetime facts. Possible
// facts control reuse only after proving conditional boundedness.
static DFBQuiescenceProof computePerNodeLifetime(
    DFBLogicalLifecycle &logicalDFB, LaunchNodeCoord node,
    SmallVectorImpl<DFBPerNodeLifetime> &lifetimes,
    SmallVectorImpl<DFBPerNodeLifetimeDiagnostics> *lifetimeDiagnostics,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, const AccessRuns &accessRuns,
    const LaunchNodeDomainState &domainState,
    bool includeUnknownDomains = false) {
  DFBPerNodeLifetime &lifetime = lifetimes.emplace_back();
  lifetime.node = node;
  DFBPerNodeLifetimeDiagnostics *diagnostics = nullptr;
  if (lifetimeDiagnostics) {
    diagnostics = &lifetimeDiagnostics->emplace_back();
  }
  SmallVector<const AccessRun *> reserves;
  SmallVector<const AccessRun *> pushes;
  SmallVector<const AccessRun *> waits;
  SmallVector<const AccessRun *> pops;
  SmallVector<const DFBAccessOccurrence *> activeAccesses;
  const DFBAccessOccurrence *unsupportedAccess = nullptr;
  bool hasReserve = false;
  bool hasPush = false;
  bool hasWait = false;
  bool hasPop = false;
  for (auto [accessIndex, access] : llvm::enumerate(logicalDFB.accesses)) {
    if (!mayContainLaunchNode(access.launchDomain, node,
                              includeUnknownDomains)) {
      continue;
    }
    auto executionCountIt = executionCounts.find(&access);
    assert(executionCountIt != executionCounts.end() &&
           "every DFB access must have an execution-count fact");
    std::optional<std::uint64_t> executionCount = executionCountIt->second;
    if (diagnostics) {
      diagnostics->occurrences.push_back(
          {static_cast<unsigned>(accessIndex), executionCount});
    }
    if (executionCount && *executionCount == 0) {
      continue;
    }
    activeAccesses.push_back(&access);
    if (access.protocolEffect) {
      switch (*access.protocolEffect) {
      case DFBProtocolEffectKind::Reserve:
        hasReserve = true;
        break;
      case DFBProtocolEffectKind::Push:
        hasPush = true;
        break;
      case DFBProtocolEffectKind::Wait:
        hasWait = true;
        break;
      case DFBProtocolEffectKind::Pop:
        hasPop = true;
        break;
      }
    }
    auto runIt = accessRuns.find(&access);
    if (runIt == accessRuns.end()) {
      unsupportedAccess = &access;
      continue;
    }
    if (!access.protocolEffect) {
      continue;
    }
    switch (*access.protocolEffect) {
    case DFBProtocolEffectKind::Reserve:
      reserves.push_back(&runIt->second);
      break;
    case DFBProtocolEffectKind::Push:
      pushes.push_back(&runIt->second);
      break;
    case DFBProtocolEffectKind::Wait:
      waits.push_back(&runIt->second);
      break;
    case DFBProtocolEffectKind::Pop:
      pops.push_back(&runIt->second);
      break;
    }
  }

  if (includeUnknownDomains && activeAccesses.empty()) {
    lifetime.mayBeActive = false;
    return {};
  }

  if (!hasReserve || !hasPush || !hasWait || !hasPop) {
    return {DFBQuiescenceFailureReason::MissingProtocolEffect,
            activeAccesses.empty() ? logicalDFB.declarations.front()
                                   : activeAccesses.front()->operation};
  }

  if (unsupportedAccess) {
    return {DFBQuiescenceFailureReason::UnsupportedControlFlow,
            unsupportedAccess->operation};
  }
  assert(!reserves.empty() && !pushes.empty() && !waits.empty() &&
         !pops.empty() && "supported protocol effects must have access runs");

  SmallVector<const AccessRun *> conditionalRuns;
  for (const DFBAccessOccurrence *activeAccess : activeAccesses) {
    const AccessRun &run = accessRuns.at(activeAccess);
    if (run.conditionalExecution) {
      conditionalRuns.push_back(&run);
    }
  }
  if (!conditionalRuns.empty()) {
    bool singleConditionalTransaction =
        conditionalRuns.size() == activeAccesses.size() &&
        reserves.size() == 1 && pushes.size() == 1 && waits.size() == 1 &&
        pops.size() == 1;
    const AccessRun &reference = *conditionalRuns.front();
    bool sameCondition = singleConditionalTransaction &&
                         llvm::all_of(llvm::drop_begin(conditionalRuns),
                                      [&](const AccessRun *run) {
                                        return proveEquivalentConditionalRuns(
                                            reference, *run, node, domainState);
                                      });
    if (!sameCondition) {
      return {DFBQuiescenceFailureReason::UnsupportedControlFlow,
              reference.access->operation};
    }
    lifetime.conditionalExecutionProven = true;
  }
  auto getTransactionCount = [](ArrayRef<const AccessRun *> runs) {
    std::optional<std::uint64_t> total = 0;
    for (const AccessRun *run : runs) {
      total = llvm::checkedAddUnsigned(*total, run->executionCount);
      if (!total) {
        break;
      }
    }
    return total;
  };
  std::optional<std::uint64_t> reserveCount = getTransactionCount(reserves);
  std::optional<std::uint64_t> pushCount = getTransactionCount(pushes);
  std::optional<std::uint64_t> waitCount = getTransactionCount(waits);
  std::optional<std::uint64_t> popCount = getTransactionCount(pops);
  if (!reserveCount || !pushCount || !waitCount || !popCount ||
      *reserveCount != *pushCount || *reserveCount != *waitCount ||
      *reserveCount != *popCount) {
    return {DFBQuiescenceFailureReason::MismatchedTransaction,
            activeAccesses.front()->operation};
  }

  if (!proveAlignedAcquireReleaseRuns(reserves, pushes, graph, operationEvents,
                                      accessEvents) ||
      !proveAlignedAcquireReleaseRuns(waits, pops, graph, operationEvents,
                                      accessEvents)) {
    return {DFBQuiescenceFailureReason::IncompleteUseOrder,
            activeAccesses.front()->operation};
  }

  int64_t physicalTileCount =
      cast<CircularBufferType>(logicalDFB.type).getTotalElements();
  if (physicalTileCount <= 0) {
    return {DFBQuiescenceFailureReason::MismatchedTransaction,
            reserves.front()->access->operation};
  }
  std::optional<DFBPointerOwner> writeOwner;
  std::optional<DFBPointerOwner> readOwner;
  SmallVector<Operation *> nativeReserves;
  SmallVector<Operation *> nativeWaits;
  for (const AccessRun *reserve : reserves) {
    if (isa<CBReserveOp>(reserve->access->operation)) {
      nativeReserves.push_back(reserve->access->operation);
    }
  }
  for (const AccessRun *wait : waits) {
    if (isa<CBWaitOp>(wait->access->operation)) {
      nativeWaits.push_back(wait->access->operation);
    }
  }
  for (auto [reserve, push] : llvm::zip_equal(reserves, pushes)) {
    if (reserve->access->numTiles <= 0) {
      return {DFBQuiescenceFailureReason::MismatchedTransaction,
              reserve->access->operation};
    }
    if (isa<CBReserveOp>(reserve->access->operation) &&
        !releaseFollowsOwnedUses(reserve->access->operation,
                                 push->access->operation, nativeReserves)) {
      return {DFBQuiescenceFailureReason::IncompleteUseOrder,
              push->access->operation};
    }
    std::optional<DFBPointerOwner> reserveOwner = getPointerOwner(
        reserve->access->operation, node, DFBProtocolEffectKind::Reserve);
    std::optional<DFBPointerOwner> pushOwner = getPointerOwner(
        push->access->operation, node, DFBProtocolEffectKind::Push);
    if (!reserveOwner || !pushOwner || *reserveOwner != *pushOwner ||
        (writeOwner && *writeOwner != *reserveOwner)) {
      return {DFBQuiescenceFailureReason::UnknownPointerOwner,
              reserve->access->operation};
    }
    writeOwner = reserveOwner;
  }
  for (auto [wait, pop] : llvm::zip_equal(waits, pops)) {
    if (wait->access->numTiles <= 0) {
      return {DFBQuiescenceFailureReason::MismatchedTransaction,
              wait->access->operation};
    }
    if (isa<CBWaitOp>(wait->access->operation) &&
        !releaseFollowsOwnedUses(wait->access->operation,
                                 pop->access->operation, nativeWaits)) {
      return {DFBQuiescenceFailureReason::IncompleteUseOrder,
              pop->access->operation};
    }
    std::optional<DFBPointerOwner> waitOwner = getPointerOwner(
        wait->access->operation, node, DFBProtocolEffectKind::Wait);
    std::optional<DFBPointerOwner> popOwner = getPointerOwner(
        pop->access->operation, node, DFBProtocolEffectKind::Pop);
    if (!waitOwner || !popOwner || *waitOwner != *popOwner ||
        (readOwner && *readOwner != *waitOwner)) {
      return {DFBQuiescenceFailureReason::UnknownPointerOwner,
              wait->access->operation};
    }
    readOwner = waitOwner;
  }

  std::size_t reserveIndex = 0;
  std::size_t waitIndex = 0;
  std::uint64_t reserveOffset = 0;
  std::uint64_t waitOffset = 0;
  while (reserveIndex < reserves.size() && waitIndex < waits.size()) {
    const AccessRun &reserve = *reserves[reserveIndex];
    const AccessRun &wait = *waits[waitIndex];
    if (reserve.access->numTiles != wait.access->numTiles ||
        physicalTileCount % reserve.access->numTiles != 0) {
      return {DFBQuiescenceFailureReason::MismatchedTransaction,
              reserve.access->operation};
    }
    std::uint64_t matchedCount =
        std::min(reserve.executionCount - reserveOffset,
                 wait.executionCount - waitOffset);
    appendTransactionRun(lifetime, matchedCount, reserve.access->numTiles);
    reserveOffset += matchedCount;
    waitOffset += matchedCount;
    if (reserveOffset == reserve.executionCount) {
      ++reserveIndex;
      reserveOffset = 0;
    }
    if (waitOffset == wait.executionCount) {
      ++waitIndex;
      waitOffset = 0;
    }
  }
  if (reserveIndex != reserves.size() || waitIndex != waits.size()) {
    return {DFBQuiescenceFailureReason::MismatchedTransaction,
            reserves.front()->access->operation};
  }
  lifetime.writePointerOwner = writeOwner;
  lifetime.readPointerOwner = readOwner;

  for (const DFBAccessOccurrence *activeAccess : activeAccesses) {
    if (activeAccess->protocolEffect) {
      continue;
    }
    const AccessRun &use = accessRuns.at(activeAccess);
    bool covered = false;
    for (auto [reserve, push] : llvm::zip_equal(reserves, pushes)) {
      covered |= runIsInsideInterval(use, *reserve, *push, graph,
                                     operationEvents, accessEvents);
    }
    for (auto [wait, pop] : llvm::zip_equal(waits, pops)) {
      covered |= runIsInsideInterval(use, *wait, *pop, graph, operationEvents,
                                     accessEvents);
    }
    if (!covered) {
      return {DFBQuiescenceFailureReason::IncompleteUseOrder,
              activeAccess->operation};
    }
  }

  std::optional<AccessEventSpan> terminalEvents =
      getAccessEventSpan(*pops.back()->access, operationEvents, accessEvents);
  if (!terminalEvents) {
    return {DFBQuiescenceFailureReason::UnsupportedControlFlow,
            pops.back()->access->operation};
  }
  for (const DFBAccessOccurrence *activeAccess : activeAccesses) {
    std::optional<AccessEventSpan> useEvents =
        getAccessEventSpan(*activeAccess, operationEvents, accessEvents);
    if (!useEvents ||
        (useEvents->last.completion != terminalEvents->last.completion &&
         !graph.strictlyPrecedes(useEvents->last.completion,
                                 terminalEvents->last.completion))) {
      return {DFBQuiescenceFailureReason::IncompleteUseOrder,
              activeAccess->operation};
    }
  }
  lifetime.earliestEntryEvents = findMinimalEntryEvents(
      activeAccesses, graph, operationEvents, accessEvents);
  lifetime.terminalCompletionEvents = {terminalEvents->last.completion};
  if (diagnostics) {
    for (const DFBAccessOccurrence *activeAccess : activeAccesses) {
      std::optional<AccessEventSpan> activeEvents =
          getAccessEventSpan(*activeAccess, operationEvents, accessEvents);
      if (activeEvents && llvm::is_contained(lifetime.earliestEntryEvents,
                                             activeEvents->first.entry)) {
        diagnostics->earliestAccessOccurrenceIndices.push_back(
            static_cast<unsigned>(activeAccess - logicalDFB.accesses.data()));
      }
    }
    diagnostics->terminalAccessOccurrenceIndices = {static_cast<unsigned>(
        pops.back()->access - logicalDFB.accesses.data())};
  }
  if (lifetime.earliestEntryEvents.empty()) {
    return {DFBQuiescenceFailureReason::IncompleteUseOrder,
            pops.back()->access->operation};
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

const DFBPerNodeLifetime *
DFBLogicalLifecycle::findPossibleNodeLifetime(LaunchNodeCoord node) const {
  auto lifetimeIt =
      llvm::find_if(possibleNodeLifetimes, [&](const auto &lifetime) {
        return lifetime.node == node;
      });
  return lifetimeIt == possibleNodeLifetimes.end() ? nullptr : &*lifetimeIt;
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
  bool hasUnknownDFBLaunchDomain =
      llvm::any_of(logicalDFBs, [](const DFBLogicalLifecycle &logicalDFB) {
        return !logicalDFB.launchDomain.known;
      });

  launchNodes.append(domainState.baseDomain.nodes.begin(),
                     domainState.baseDomain.nodes.end());
  orderedBeforeByNode.reserve(launchNodes.size());
  conditionallyOrderedBeforeByNode.reserve(launchNodes.size());
  bool collectAllocationDiagnostics = false;
  LLVM_DEBUG(collectAllocationDiagnostics = true);
  if (collectAllocationDiagnostics) {
    for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
      logicalDFB.allocationDiagnostics =
          std::make_unique<DFBLogicalLifecycleDiagnostics>();
    }
  }
  for (LaunchNodeCoord node : launchNodes) {
    AccessExecutionCounts executionCounts =
        collectAccessExecutionCounts(logicalDFBs, node, domainState);
    AccessRuns accessRuns =
        collectAccessRuns(logicalDFBs, node, domainState, executionCounts,
                          /*includeUnknownDomains=*/false);
    HappensBeforeGraph graph;
    DenseMap<Operation *, EventPair> operationEvents;
    DenseMap<const DFBAccessOccurrence *, AccessEventSpan> accessEvents;
    buildProgramOrderGraph(module, logicalDFBs, node, graph, operationEvents,
                           accessEvents, executionCounts, accessRuns,
                           /*includeUnknownDomains=*/false);
    addMatchedPushWaitEdges(logicalDFBs, graph, operationEvents, accessEvents,
                            accessRuns, node, domainState);
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
          graph, operationEvents, accessEvents, executionCounts, accessRuns,
          domainState);
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

    // Possible-domain reachability cannot affect exact-domain reuse.
    if (!hasUnknownDFBLaunchDomain) {
      conditionallyOrderedBeforeByNode.emplace_back(
          logicalDFBs.size(), llvm::BitVector(logicalDFBs.size()));
      continue;
    }

    HappensBeforeGraph possibleGraph;
    DenseMap<Operation *, EventPair> possibleOperationEvents;
    DenseMap<const DFBAccessOccurrence *, AccessEventSpan> possibleAccessEvents;
    AccessRuns possibleAccessRuns =
        collectAccessRuns(logicalDFBs, node, domainState, executionCounts,
                          /*includeUnknownDomains=*/true);
    buildProgramOrderGraph(module, logicalDFBs, node, possibleGraph,
                           possibleOperationEvents, possibleAccessEvents,
                           executionCounts, possibleAccessRuns,
                           /*includeUnknownDomains=*/true);
    addMatchedPushWaitEdges(logicalDFBs, possibleGraph, possibleOperationEvents,
                            possibleAccessEvents, possibleAccessRuns, node,
                            domainState);
    possibleGraph.computeReachability();
    for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
      if (logicalDFB.launchDomain.known) {
        continue;
      }
      DFBLogicalLifecycleDiagnostics *allocationDiagnostics =
          logicalDFB.allocationDiagnostics.get();
      DFBQuiescenceProof proof = computePerNodeLifetime(
          logicalDFB, node, logicalDFB.possibleNodeLifetimes,
          allocationDiagnostics
              ? &allocationDiagnostics->possibleNodeLifetimeDiagnostics
              : nullptr,
          possibleGraph, possibleOperationEvents, possibleAccessEvents,
          executionCounts, possibleAccessRuns, domainState,
          /*includeUnknownDomains=*/true);
      logicalDFB.possibleNodeLifetimes.back().quiescence = proof;
    }

    SmallVector<llvm::BitVector> conditionalNodeOrdering(
        logicalDFBs.size(), llvm::BitVector(logicalDFBs.size()));
    for (unsigned beforeIndex = 0; beforeIndex < logicalDFBs.size();
         ++beforeIndex) {
      const DFBPerNodeLifetime *before =
          logicalDFBs[beforeIndex].findPossibleNodeLifetime(node);
      if (!before) {
        continue;
      }
      for (unsigned afterIndex = 0; afterIndex < logicalDFBs.size();
           ++afterIndex) {
        const DFBPerNodeLifetime *after =
            logicalDFBs[afterIndex].findPossibleNodeLifetime(node);
        if (after && proveOrderedBefore(*before, *after, possibleGraph)) {
          conditionalNodeOrdering[beforeIndex].set(afterIndex);
        }
      }
    }
    conditionallyOrderedBeforeByNode.push_back(
        std::move(conditionalNodeOrdering));
  }

  for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    logicalDFB.bounded = logicalDFB.launchDomain.known &&
                         !logicalDFB.nodeLifetimes.empty() &&
                         llvm::all_of(logicalDFB.nodeLifetimes,
                                      [](const DFBPerNodeLifetime &lifetime) {
                                        return lifetime.quiescence.proven();
                                      });
    bool hasProvenConditionalLifecycle =
        llvm::any_of(logicalDFB.possibleNodeLifetimes,
                     [](const DFBPerNodeLifetime &lifetime) {
                       return lifetime.mayBeActive &&
                              lifetime.conditionalExecutionProven &&
                              lifetime.quiescence.proven();
                     });
    logicalDFB.conditionallyBounded =
        !logicalDFB.launchDomain.known && hasProvenConditionalLifecycle &&
        llvm::all_of(logicalDFB.possibleNodeLifetimes,
                     [](const DFBPerNodeLifetime &lifetime) {
                       return !lifetime.mayBeActive ||
                              (lifetime.conditionalExecutionProven &&
                               lifetime.quiescence.proven());
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

bool DFBConcurrentKernelLivenessAnalysis::isConditionallyOrderedBefore(
    unsigned beforeIndex, unsigned afterIndex, LaunchNodeCoord node) const {
  auto nodeIt = llvm::find(launchNodes, node);
  assert(nodeIt != launchNodes.end() && "node must be in the launch grid");
  unsigned nodeIndex = nodeIt - launchNodes.begin();
  assert(beforeIndex < conditionallyOrderedBeforeByNode[nodeIndex].size() &&
         afterIndex < conditionallyOrderedBeforeByNode[nodeIndex].size());
  return conditionallyOrderedBeforeByNode[nodeIndex][beforeIndex].test(
      afterIndex);
}

} // namespace mlir::tt::ttl
