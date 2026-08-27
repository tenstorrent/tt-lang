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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

namespace {

// Entry and completion events keep operation duration distinct from source
// order and protocol completion edges.
struct EventPair {
  unsigned entry = 0;
  unsigned completion = 0;

  bool operator==(const EventPair &rhs) const {
    return std::tie(entry, completion) == std::tie(rhs.entry, rhs.completion);
  }

  bool operator!=(const EventPair &rhs) const { return !(*this == rhs); }
};

// First and last dynamic executions represented by one static access.
struct AccessEventSpan {
  EventPair first;
  EventPair last;

  bool operator==(const AccessEventSpan &rhs) const {
    return std::tie(first, last) == std::tie(rhs.first, rhs.last);
  }

  bool operator!=(const AccessEventSpan &rhs) const { return !(*this == rhs); }
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

  // Computes exact reachability while retaining cycles as non-strict order.
  void computeReachability() {
    unsigned eventCount = successors.size();
    reachable.assign(eventCount, llvm::BitVector(eventCount));
    cyclicEvents = llvm::BitVector(eventCount);
    rejectedCyclePartners.clear();
    rejectedCyclePartners.resize(eventCount);
    if (eventCount == 0) {
      return;
    }

    struct DFSFrame {
      unsigned event;
      unsigned nextSuccessor;
    };

    llvm::BitVector visited(eventCount);
    SmallVector<unsigned> finishOrder;
    finishOrder.reserve(eventCount);
    for (unsigned rootEvent = 0; rootEvent < eventCount; ++rootEvent) {
      if (visited.test(rootEvent)) {
        continue;
      }
      SmallVector<DFSFrame> pending = {{rootEvent, 0}};
      visited.set(rootEvent);
      while (!pending.empty()) {
        DFSFrame &frame = pending.back();
        if (frame.nextSuccessor < successors[frame.event].size()) {
          unsigned successor = successors[frame.event][frame.nextSuccessor++];
          if (!visited.test(successor)) {
            visited.set(successor);
            pending.push_back({successor, 0});
          }
          continue;
        }
        finishOrder.push_back(frame.event);
        pending.pop_back();
      }
    }
    SmallVector<SmallVector<unsigned>> predecessors(eventCount);
    for (auto [sourceEvent, eventSuccessors] : llvm::enumerate(successors)) {
      for (unsigned destinationEvent : eventSuccessors) {
        predecessors[destinationEvent].push_back(sourceEvent);
      }
    }

    constexpr unsigned unassignedComponent =
        std::numeric_limits<unsigned>::max();
    SmallVector<unsigned> componentForEvent(eventCount, unassignedComponent);
    SmallVector<SmallVector<unsigned>> componentMembers;
    for (unsigned rootEvent : llvm::reverse(finishOrder)) {
      if (componentForEvent[rootEvent] != unassignedComponent) {
        continue;
      }
      unsigned component = componentMembers.size();
      componentMembers.emplace_back();
      SmallVector<unsigned> pending = {rootEvent};
      componentForEvent[rootEvent] = component;
      while (!pending.empty()) {
        unsigned event = pending.pop_back_val();
        componentMembers.back().push_back(event);
        for (unsigned predecessor : predecessors[event]) {
          if (componentForEvent[predecessor] == unassignedComponent) {
            componentForEvent[predecessor] = component;
            pending.push_back(predecessor);
          }
        }
      }
    }

    unsigned componentCount = componentMembers.size();
    SmallVector<SmallVector<unsigned>> componentSuccessors(componentCount);
    for (auto [sourceEvent, eventSuccessors] : llvm::enumerate(successors)) {
      unsigned sourceComponent = componentForEvent[sourceEvent];
      for (unsigned destinationEvent : eventSuccessors) {
        unsigned destinationComponent = componentForEvent[destinationEvent];
        if (sourceComponent != destinationComponent) {
          componentSuccessors[sourceComponent].push_back(destinationComponent);
        }
      }
    }
    for (SmallVector<unsigned> &successorComponents : componentSuccessors) {
      llvm::sort(successorComponents);
      successorComponents.erase(
          std::unique(successorComponents.begin(), successorComponents.end()),
          successorComponents.end());
    }

    SmallVector<unsigned> incomingEdges(componentCount);
    for (const SmallVector<unsigned> &successorComponents :
         componentSuccessors) {
      for (unsigned successorComponent : successorComponents) {
        ++incomingEdges[successorComponent];
      }
    }
    SmallVector<unsigned> readyComponents;
    for (unsigned component = 0; component < componentCount; ++component) {
      if (incomingEdges[component] == 0) {
        readyComponents.push_back(component);
      }
    }
    SmallVector<unsigned> topologicalOrder;
    topologicalOrder.reserve(componentCount);
    while (!readyComponents.empty()) {
      unsigned component = readyComponents.pop_back_val();
      topologicalOrder.push_back(component);
      for (unsigned successorComponent : componentSuccessors[component]) {
        if (--incomingEdges[successorComponent] == 0) {
          readyComponents.push_back(successorComponent);
        }
      }
    }
    assert(topologicalOrder.size() == componentCount &&
           "SCC condensation must be acyclic");

    SmallVector<llvm::BitVector> componentReachability(
        componentCount, llvm::BitVector(eventCount));
    for (unsigned component : llvm::reverse(topologicalOrder)) {
      llvm::BitVector &componentEvents = componentReachability[component];
      for (unsigned event : componentMembers[component]) {
        componentEvents.set(event);
      }
      for (unsigned successorComponent : componentSuccessors[component]) {
        componentEvents |= componentReachability[successorComponent];
      }
    }
    for (unsigned event = 0; event < eventCount; ++event) {
      reachable[event] = componentReachability[componentForEvent[event]];
    }

    for (unsigned source = 0; source < eventCount; ++source) {
      for (unsigned destination : successors[source]) {
        if (source != destination && reachable[destination].test(source)) {
          cyclicEvents.set(source);
          cyclicEvents.set(destination);
        }
      }
    }

#ifndef NDEBUG
    // Small graphs retain an independent oracle without affecting model-scale
    // analysis cost.
    if (eventCount <= 128) {
      for (unsigned sourceEvent = 0; sourceEvent < eventCount; ++sourceEvent) {
        llvm::BitVector reference(eventCount);
        SmallVector<unsigned> pending = {sourceEvent};
        while (!pending.empty()) {
          unsigned event = pending.pop_back_val();
          if (reference.test(event)) {
            continue;
          }
          reference.set(event);
          pending.append(successors[event].begin(), successors[event].end());
        }
        assert(reference == reachable[sourceEvent] &&
               "SCC reachability differs from traversal reference");
      }
    }
#endif
  }

  // Requires asymmetric reachability because mutually reachable events do not
  // establish a safe lifetime order.
  bool strictlyPrecedes(unsigned source, unsigned destination) const {
    assert(source < reachable.size() && destination < reachable.size());
    return source != destination && reachable[source].test(destination) &&
           !reachable[destination].test(source);
  }

  /// Returns true when two distinct events belong to one reachability cycle.
  bool mutuallyReachable(unsigned lhs, unsigned rhs) const {
    assert(lhs < reachable.size() && rhs < reachable.size());
    return lhs != rhs && reachable[lhs].test(rhs) && reachable[rhs].test(lhs);
  }

  bool eventParticipatesInCycle(unsigned event) const {
    assert(event < cyclicEvents.size());
    return cyclicEvents.test(event);
  }

  bool hasInconsistentOrder(unsigned lhs, unsigned rhs) const {
    assert(lhs < reachable.size() && rhs < reachable.size());
    return mutuallyReachable(lhs, rhs) ||
           llvm::is_contained(rejectedCyclePartners[lhs], rhs);
  }

  bool eventParticipatesInInconsistentOrder(unsigned event) const {
    assert(event < rejectedCyclePartners.size());
    return eventParticipatesInCycle(event) ||
           !rejectedCyclePartners[event].empty();
  }

  bool operator==(const HappensBeforeGraph &rhs) const {
    return std::tie(successors, reachable, cyclicEvents,
                    rejectedCyclePartners) ==
           std::tie(rhs.successors, rhs.reachable, rhs.cyclicEvents,
                    rhs.rejectedCyclePartners);
  }

  unsigned getEventCount() const { return successors.size(); }

  /// Atomically adds acyclic edges to a graph whose transitive closure is
  /// current. Only predecessors of each source gain reachability, and they
  /// gain exactly the existing successor set of its destination.
  bool tryAddEdgesAndUpdateReachability(
      ArrayRef<std::pair<unsigned, unsigned>> edges,
      bool tolerateCandidateInternalCycle = false) {
    assert(reachable.size() == successors.size() &&
           "transitive closure must be current before incremental updates");
    SmallVector<llvm::BitVector> candidateReachability = reachable;
    for (auto [source, destination] : edges) {
      if (source == destination ||
          candidateReachability[destination].test(source)) {
        bool contradictsExistingOrder =
            source != destination && reachable[destination].test(source);
        if (contradictsExistingOrder || !tolerateCandidateInternalCycle) {
          recordRejectedCycle(source, destination);
        }
        return false;
      }
      if (candidateReachability[source].test(destination)) {
        continue;
      }
      llvm::BitVector newlyReachable = candidateReachability[destination];
      for (llvm::BitVector &sourceReachability : candidateReachability) {
        if (sourceReachability.test(source)) {
          sourceReachability |= newlyReachable;
        }
      }
    }

    for (auto [source, destination] : edges) {
      addEdge(source, destination);
    }
    reachable = std::move(candidateReachability);
    return true;
  }

private:
  void recordRejectedCycle(unsigned source, unsigned destination) {
    auto addPartner = [&](unsigned event, unsigned partner) {
      if (!llvm::is_contained(rejectedCyclePartners[event], partner)) {
        rejectedCyclePartners[event].push_back(partner);
      }
    };
    addPartner(source, destination);
    addPartner(destination, source);
  }

  SmallVector<SmallVector<unsigned>> successors;
  SmallVector<llvm::BitVector> reachable;
  llvm::BitVector cyclicEvents;
  SmallVector<SmallVector<unsigned>> rejectedCyclePartners;
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

// One static occurrence of a typed synchronized reset before launch-node
// participation is validated.
struct SynchronizedResetOccurrence {
  Operation *operation = nullptr;
  SynchronizedDFBResetAttr reset;
  LogicalKernelAttr participant;
  SmallVector<unsigned> targetLogicalIndices;
  bool allDFBs = false;
  LaunchNodeDomain launchDomain = LaunchNodeDomain::unknown();
};

// Structured loops whose Cartesian iteration space executes one operation once.
// Local access ordering uses loop identity; cross-kernel reset matching uses
// the constant trip-count sequence.
struct StaticIterationDomain {
  SmallVector<Operation *> loops;
  SmallVector<std::uint64_t> tripCounts;

  bool operator==(const StaticIterationDomain &rhs) const {
    return loops == rhs.loops && tripCounts == rhs.tripCounts;
  }
};

// One proved reset instance or uniform reset run on one launch node.
struct ValidatedSynchronizedReset {
  SynchronizedDFBResetAttr reset;
  SmallVector<Operation *> participantOperations;
  SmallVector<StaticIterationDomain, 0> participantIterationDomains;
  SmallVector<unsigned> targetLogicalIndices;
  std::uint64_t executionCount = 1;
  bool conditionalExecution = false;

  bool operator==(const ValidatedSynchronizedReset &rhs) const {
    return std::tie(reset, participantOperations, participantIterationDomains,
                    targetLogicalIndices, executionCount,
                    conditionalExecution) ==
           std::tie(rhs.reset, rhs.participantOperations,
                    rhs.participantIterationDomains, rhs.targetLogicalIndices,
                    rhs.executionCount, rhs.conditionalExecution);
  }

  // Dispatch-wide partitioning is valid only for a single reset instance.
  bool isModeledLifetimeBoundary() const { return executionCount == 1; }
};

// One static occurrence of a DFB configuration-epoch boundary before
// launch-node participation is validated.
struct DFBReconfigurationOccurrence {
  Operation *operation = nullptr;
  DFBReconfigurationAttr boundary;
  LogicalKernelAttr participant;
  LaunchNodeDomain launchDomain = LaunchNodeDomain::unknown();
};

// One proved dynamic DFB configuration-epoch boundary on one launch node.
struct ValidatedDFBReconfiguration {
  DFBReconfigurationAttr boundary;
  SmallVector<Operation *> participantOperations;
  bool conditionalExecution = false;

  bool operator==(const ValidatedDFBReconfiguration &rhs) const {
    return std::tie(boundary, participantOperations, conditionalExecution) ==
           std::tie(rhs.boundary, rhs.participantOperations,
                    rhs.conditionalExecution);
  }
};

// First and last collective instances of one reset declaration. Equal events
// represent a single reset instance.
using ResetBoundaryEvents = DenseMap<SynchronizedDFBResetAttr, AccessEventSpan>;
using ReconfigurationBoundaryEvents =
    DenseMap<DFBReconfigurationAttr, EventPair>;

using AccessExecutionCounts =
    DenseMap<const DFBAccessOccurrence *, std::optional<std::uint64_t>>;

// Compares dynamic iteration positions across different logical kernels. Equal
// constant trip counts at each nesting depth define the same ordinal iteration
// sequence without depending on unrelated loop operation identity.
static bool hasEquivalentIterationSequence(const StaticIterationDomain &lhs,
                                           const StaticIterationDomain &rhs) {
  return lhs.tripCounts == rhs.tripCounts;
}

// One statically counted run of an access occurrence.
struct AccessRun {
  const DFBAccessOccurrence *access = nullptr;
  std::uint64_t executionCount = 0;
  StaticIterationDomain iterationDomain;
  bool conditionalExecution = false;

  bool operator==(const AccessRun &rhs) const {
    return std::tie(access, executionCount, iterationDomain,
                    conditionalExecution) ==
           std::tie(rhs.access, rhs.executionCount, rhs.iterationDomain,
                    rhs.conditionalExecution);
  }
};

using AccessRuns = DenseMap<const DFBAccessOccurrence *, AccessRun>;

// The listed operations execute each nested region at most once per
// invocation, so only enclosing loops can repeat an access.
static bool executesRegionsAtMostOnce(Operation *operation) {
  return isa<affine::AffineIfOp, scf::IfOp, scf::IndexSwitchOp,
             scf::ExecuteRegionOp, IfSrcOp, IfDstOp>(operation);
}

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
// structured loop nest. At-most-once regions must be selected on every
// enclosing-loop invocation.
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
  std::uint64_t nestedExecutionCount = executionCount;
  Operation *nestedOperation = operation;
  while (nestedOperation->getParentRegion() != &function.getBody()) {
    Region *region = nestedOperation->getParentRegion();
    Operation *parent = region ? region->getParentOp() : nullptr;
    auto loop = dyn_cast_or_null<LoopLikeOpInterface>(parent);
    if (!parent || !region->hasOneBlock() ||
        nestedOperation->getBlock() != &region->front()) {
      return std::nullopt;
    }
    if (loop && !isa<affine::AffineForOp, scf::ForOp>(parent)) {
      return std::nullopt;
    }
    if (!loop) {
      std::optional<std::uint64_t> parentExecutionCount =
          getExactExecutionCountAtLaunchNode(parent, node, domainState);
      // Equality proves that an at-most-once region was selected for every
      // parent invocation, including each enclosing-loop iteration.
      if (!executesRegionsAtMostOnce(parent) || !parentExecutionCount ||
          *parentExecutionCount != nestedExecutionCount) {
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
    std::optional<std::uint64_t> parentExecutionCount =
        getExactExecutionCountAtLaunchNode(parent, node, domainState);
    if (!parentExecutionCount) {
      return std::nullopt;
    }
    std::optional<std::uint64_t> loopBodyExecutionCount =
        llvm::checkedMulUnsigned(*parentExecutionCount, *tripCount);
    if (!loopBodyExecutionCount ||
        *loopBodyExecutionCount != nestedExecutionCount) {
      return std::nullopt;
    }
    domain.loops.push_back(parent);
    domain.tripCounts.push_back(*tripCount);
    nestedExecutionCount = *parentExecutionCount;
    nestedOperation = parent;
  }

  if (nestedOperation->getBlock() != &function.getBody().front() ||
      nestedExecutionCount != 1) {
    return std::nullopt;
  }
  std::reverse(domain.loops.begin(), domain.loops.end());
  std::reverse(domain.tripCounts.begin(), domain.tripCounts.end());
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
        !executesRegionsAtMostOnce(parent)) {
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

#ifndef NDEBUG
static bool structurallyPrecedesReference(Operation *before, Operation *after) {
  if (before == after || before->getParentOfType<func::FuncOp>() !=
                             after->getParentOfType<func::FuncOp>()) {
    return false;
  }
  for (Block *commonBlock = before->getBlock(); commonBlock;) {
    Operation *projectedBefore =
        before->getBlock() == commonBlock
            ? before
            : commonBlock->findAncestorOpInBlock(*before);
    Operation *projectedAfter =
        after->getBlock() == commonBlock
            ? after
            : commonBlock->findAncestorOpInBlock(*after);
    if (projectedBefore && projectedAfter &&
        projectedBefore != projectedAfter) {
      return projectedBefore->isBeforeInBlock(projectedAfter);
    }
    Operation *parent = commonBlock->getParentOp();
    commonBlock = parent ? parent->getBlock() : nullptr;
  }
  return false;
}
#endif

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

// Caches structural order while liveness analyzes immutable IR.
class StructuralOperationOrder {
public:
  explicit StructuralOperationOrder(ModuleOp module) {
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      SmallVector<BlockPosition> enclosingPositions;
      indexRegion(function.getBody(), function, enclosingPositions);
    }
  }

  Operation *getFunction(Operation *operation) const {
    auto locationIt = locations.find(operation);
    return locationIt == locations.end() ? nullptr
                                         : locationIt->second.function;
  }

  Operation *getTopLevelOperation(Operation *operation) const {
    auto locationIt = locations.find(operation);
    return locationIt == locations.end() ? nullptr
                                         : locationIt->second.topLevelOperation;
  }

  bool precedes(Operation *before, Operation *after) const {
    std::pair<Operation *, Operation *> operations = {before, after};
    auto cachedIt = precedence.find(operations);
    if (cachedIt != precedence.end()) {
      return cachedIt->second;
    }
    bool result = false;
    auto beforeIt = locations.find(before);
    auto afterIt = locations.find(after);
    if (before != after && beforeIt != locations.end() &&
        afterIt != locations.end() &&
        beforeIt->second.function == afterIt->second.function) {
      ArrayRef<BlockPosition> beforePositions = beforeIt->second.positions;
      ArrayRef<BlockPosition> afterPositions = afterIt->second.positions;
      for (auto [beforePosition, afterPosition] :
           llvm::zip(beforePositions, afterPositions)) {
        if (beforePosition.block != afterPosition.block) {
          break;
        }
        if (beforePosition.ordinal != afterPosition.ordinal) {
          result = beforePosition.ordinal < afterPosition.ordinal;
          break;
        }
      }
    }
#ifndef NDEBUG
    if (locations.size() <= 128) {
      assert(result == structurallyPrecedesReference(before, after) &&
             "indexed structural operation order must match the reference");
    }
#endif
    precedence.try_emplace(operations, result);
    return result;
  }

private:
  struct BlockPosition {
    Block *block;
    unsigned ordinal;
    Operation *operation;
  };

  struct OperationLocation {
    Operation *function;
    Operation *topLevelOperation;
    SmallVector<BlockPosition> positions;
  };

  void indexRegion(Region &region, func::FuncOp function,
                   SmallVectorImpl<BlockPosition> &enclosingPositions) {
    for (Block &block : region) {
      for (auto [ordinal, operation] : llvm::enumerate(block)) {
        enclosingPositions.push_back(
            {&block, static_cast<unsigned>(ordinal), &operation});
        Operation *topLevelOperation =
            function.getBody().hasOneBlock()
                ? enclosingPositions.front().operation
                : nullptr;
        locations.try_emplace(&operation,
                              OperationLocation{function, topLevelOperation,
                                                SmallVector<BlockPosition>(
                                                    enclosingPositions.begin(),
                                                    enclosingPositions.end())});
        for (Region &nestedRegion : operation.getRegions()) {
          indexRegion(nestedRegion, function, enclosingPositions);
        }
        enclosingPositions.pop_back();
      }
    }
  }

  DenseMap<Operation *, OperationLocation> locations;
  mutable DenseMap<std::pair<Operation *, Operation *>, bool> precedence;
};

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

// Retains contradictory order across logical DFB access events. Strict ordering
// excludes cycles, but unsafe allocation-group policy must distinguish a
// missing relation from contradictory evidence.
static SmallVector<llvm::BitVector> collectInconsistentAccessOrder(
    ArrayRef<DFBLogicalLifecycle> logicalDFBs, LaunchNodeCoord node,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, bool includeUnknownDomains) {
  SmallVector<SmallVector<unsigned>> inconsistentEventsByLogical(
      logicalDFBs.size());
  for (auto [logicalIndex, logicalDFB] : llvm::enumerate(logicalDFBs)) {
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      if (!mayAccessLaunchNode(access, node, executionCounts,
                               includeUnknownDomains)) {
        continue;
      }
      std::optional<AccessEventSpan> span =
          getAccessEventSpan(access, operationEvents, accessEvents);
      if (!span) {
        continue;
      }
      unsigned candidates[] = {span->first.entry, span->first.completion,
                               span->last.entry, span->last.completion};
      for (unsigned event : candidates) {
        if (graph.eventParticipatesInInconsistentOrder(event) &&
            !llvm::is_contained(inconsistentEventsByLogical[logicalIndex],
                                event)) {
          inconsistentEventsByLogical[logicalIndex].push_back(event);
        }
      }
    }
  }

  SmallVector<llvm::BitVector> inconsistent(
      logicalDFBs.size(), llvm::BitVector(logicalDFBs.size()));
  for (unsigned lhsIndex = 0; lhsIndex < logicalDFBs.size(); ++lhsIndex) {
    for (unsigned rhsIndex = lhsIndex; rhsIndex < logicalDFBs.size();
         ++rhsIndex) {
      bool inconsistentOrder = llvm::any_of(
          inconsistentEventsByLogical[lhsIndex], [&](unsigned lhsEvent) {
            return llvm::any_of(
                inconsistentEventsByLogical[rhsIndex], [&](unsigned rhsEvent) {
                  return graph.hasInconsistentOrder(lhsEvent, rhsEvent);
                });
          });
      if (inconsistentOrder) {
        inconsistent[lhsIndex].set(rhsIndex);
        inconsistent[rhsIndex].set(lhsIndex);
      }
    }
  }
  return inconsistent;
}

// A summarized call orders its effects by sequence index; accesses in
// distinct operations use structural IR order.
static bool
accessOccurrencePrecedes(const DFBAccessOccurrence &before,
                         const DFBAccessOccurrence &after,
                         const StructuralOperationOrder &structuralOrder) {
  if (before.operation == after.operation) {
    return before.getProtocolEffect() && after.getProtocolEffect() &&
           before.sequenceIndex < after.sequenceIndex;
  }
  return structuralOrder.precedes(before.operation, after.operation);
}

// Proves ordering between corresponding executions, not all-before-all
// ordering across the complete runs.
static bool runPrecedesWithinEachIteration(
    const AccessRun &before, const AccessRun &after,
    const StructuralOperationOrder &structuralOrder) {
  if (!(before.iterationDomain == after.iterationDomain) ||
      before.executionCount != after.executionCount) {
    return false;
  }
  if (before.access->operation == after.access->operation) {
    return accessOccurrencePrecedes(*before.access, *after.access,
                                    structuralOrder);
  }
  if (!(before.conditionalExecution && after.conditionalExecution) &&
      before.executionCount <= 1) {
    return false;
  }
  return accessOccurrencePrecedes(*before.access, *after.access,
                                  structuralOrder);
}

// The middle edge preserves iteration-to-iteration order without one event per
// execution.
static void addPerIterationSpanOrder(HappensBeforeGraph &graph,
                                     const AccessEventSpan &before,
                                     const AccessEventSpan &after) {
  graph.addEdge(before.first.completion, after.first.entry);
  graph.addEdge(after.first.completion, before.last.entry);
  graph.addEdge(before.last.completion, after.last.entry);
}

// Requires a release to follow every use owned by its acquisition; textual
// acquire/release order alone does not prove that those uses have completed.
// `sameKindAcquires` contains every same-DFB acquisition of the same kind.
static bool
releaseFollowsOwnedUses(Operation *acquire, Operation *release,
                        ArrayRef<Operation *> sameKindAcquires,
                        const StructuralOperationOrder &structuralOrder) {
  DFBAcquireInterval interval =
      makeDFBAcquireInterval(acquire, sameKindAcquires);
  SmallVector<Operation *> ownedUses;
  collectDFBAcquireOwnedUses(interval, ownedUses);
  return llvm::all_of(ownedUses, [&](Operation *use) {
    return structuralOrder.precedes(use, release);
  });
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

static void recordEntryFrontierEvidence(
    DFBPerNodeLifetime &lifetime, DFBPerNodeLifetimeDiagnostics *diagnostics,
    ArrayRef<const DFBAccessOccurrence *> accesses,
    const DFBLogicalLifecycle &logicalDFB,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  for (const DFBAccessOccurrence *access : accesses) {
    std::optional<AccessEventSpan> events =
        getAccessEventSpan(*access, operationEvents, accessEvents);
    if (!events || !llvm::is_contained(lifetime.earliestEntryEvents,
                                       events->first.entry)) {
      continue;
    }
    if (!lifetime.entryEvidence) {
      lifetime.entryEvidence = access->operation;
    }
    if (diagnostics) {
      diagnostics->earliestAccessOccurrenceIndices.push_back(
          static_cast<unsigned>(access - logicalDFB.accesses.data()));
    }
  }
}

// Retains every possible lifetime end so each one constrains storage reuse.
static SmallVector<const DFBAccessOccurrence *> findMaximalCompletionAccesses(
    ArrayRef<const DFBAccessOccurrence *> accesses,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  SmallVector<const DFBAccessOccurrence *> maximal;
  for (const DFBAccessOccurrence *candidate : accesses) {
    std::optional<AccessEventSpan> candidateEvents =
        getAccessEventSpan(*candidate, operationEvents, accessEvents);
    if (!candidateEvents) {
      continue;
    }
    bool hasSuccessor = llvm::any_of(accesses, [&](const auto *otherAccess) {
      std::optional<AccessEventSpan> otherEvents =
          getAccessEventSpan(*otherAccess, operationEvents, accessEvents);
      return otherEvents &&
             graph.strictlyPrecedes(candidateEvents->last.completion,
                                    otherEvents->last.completion);
    });
    if (!hasSuccessor) {
      maximal.push_back(candidate);
    }
  }
  return maximal;
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

static LogicalResult expandSelectedResetAllocationGroups(
    ArrayRef<DFBLogicalLifecycle> logicalDFBs,
    MutableArrayRef<SynchronizedResetOccurrence> resetOccurrences,
    DFBAnalysisFailure &analysisFailure) {
  DenseMap<int64_t, SmallVector<unsigned>> membersByAllocationGroup;
  for (auto [logicalIndex, logicalDFB] : llvm::enumerate(logicalDFBs)) {
    if (logicalDFB.allocationGroup) {
      membersByAllocationGroup[logicalDFB.allocationGroup.getOrdinal()]
          .push_back(logicalIndex);
    }
  }

  for (SynchronizedResetOccurrence &reset : resetOccurrences) {
    if (reset.allDFBs) {
      continue;
    }
    SmallVector<unsigned> expandedTargets = reset.targetLogicalIndices;
    for (unsigned logicalIndex : reset.targetLogicalIndices) {
      DFBAllocationGroupAttr allocationGroup =
          logicalDFBs[logicalIndex].allocationGroup;
      if (!allocationGroup) {
        continue;
      }
      auto groupIt =
          membersByAllocationGroup.find(allocationGroup.getOrdinal());
      assert(groupIt != membersByAllocationGroup.end() &&
             "allocation group must contain its reset target");
      for (unsigned member : groupIt->second) {
        if (logicalDFBs[member].tensorBacking) {
          std::string message;
          llvm::raw_string_ostream messageStream(message);
          messageStream
              << "selected synchronized DFB reset targeting allocation group "
              << allocationGroup
              << " requires scratch-backed members; logical DFB "
              << logicalDFBs[member].logicalId << " is tensor-backed";
          analysisFailure.set(reset.operation, messageStream.str());
          return failure();
        }
      }
      llvm::append_range(expandedTargets, groupIt->second);
    }
    llvm::sort(expandedTargets);
    expandedTargets.erase(llvm::unique(expandedTargets), expandedTargets.end());
    reset.targetLogicalIndices = std::move(expandedTargets);
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
    SmallVectorImpl<SynchronizedResetOccurrence> &resetOccurrences,
    SmallVectorImpl<DFBReconfigurationOccurrence> &reconfigurationOccurrences,
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
      logicalDFB.allocationGroup = assignment.allocationGroup;
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
    auto selectedReset = dyn_cast<ResetDFBsOp>(operation);
    auto allDFBsReset = dyn_cast<ResetAllDFBsOp>(operation);
    if (selectedReset || allDFBsReset) {
      SynchronizedDFBResetAttr reset =
          selectedReset ? selectedReset.getReset() : allDFBsReset.getReset();
      func::FuncOp kernel = operation->getParentOfType<func::FuncOp>();
      auto logicalKernel =
          kernel
              ? kernel->getAttrOfType<LogicalKernelAttr>(kLogicalKernelAttrName)
              : LogicalKernelAttr();
      if (!logicalKernel ||
          !llvm::is_contained(reset.getParticipants(), logicalKernel)) {
        analysisFailure.set(
            operation,
            "synchronized DFB reset must execute in one of its declared "
            "logical-kernel participants");
        return WalkResult::interrupt();
      }
      SynchronizedResetOccurrence occurrence;
      occurrence.operation = operation;
      occurrence.reset = reset;
      occurrence.participant = logicalKernel;
      occurrence.allDFBs = static_cast<bool>(allDFBsReset);
      ValueRange resetDFBs =
          selectedReset ? selectedReset.getDfbs() : ValueRange();
      for (Value target : resetDFBs) {
        FailureOr<int64_t> logicalId = identityAnalysis.getLogicalId(target);
        if (failed(logicalId)) {
          analysisFailure.set(
              operation,
              "synchronized DFB reset DFB must resolve to ttl.bind_cb before "
              "physical index allocation");
          return WalkResult::interrupt();
        }
        auto logicalIt = logicalIndexById.find(*logicalId);
        assert(logicalIt != logicalIndexById.end() &&
               "resolved reset target must have a logical lifecycle");
        unsigned logicalIndex = logicalIt->second;
        occurrence.targetLogicalIndices.push_back(logicalIndex);
      }
      llvm::sort(occurrence.targetLogicalIndices);
      resetOccurrences.push_back(std::move(occurrence));
      return WalkResult::advance();
    }
    if (auto reconfiguration = dyn_cast<DFBReconfigurationOp>(operation)) {
      func::FuncOp kernel = reconfiguration->getParentOfType<func::FuncOp>();
      auto logicalKernel =
          kernel
              ? kernel->getAttrOfType<LogicalKernelAttr>(kLogicalKernelAttrName)
              : LogicalKernelAttr();
      DFBReconfigurationAttr boundary = reconfiguration.getBoundaryAttr();
      if (!logicalKernel ||
          !llvm::is_contained(boundary.getParticipants(), logicalKernel)) {
        analysisFailure.set(
            reconfiguration,
            "DFB reconfiguration must execute in one of its declared "
            "logical-kernel participants");
        return WalkResult::interrupt();
      }
      reconfigurationOccurrences.push_back({reconfiguration, boundary,
                                            logicalKernel,
                                            LaunchNodeDomain::unknown()});
      return WalkResult::advance();
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
            {operation, std::monostate{}, 0, 0, LaunchNodeDomain::unknown(),
             nullptr});
      }
      return WalkResult::advance();
    }

    // Preserve dependency occurrences because aliased operands may have
    // different summaries. An occurrence without an effect or non-transactional
    // access remains opaque until a synchronized reset proves completion.
    llvm::BitVector describedDependencies(dfbOperands.size());
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
      describedDependencies.set(effect.dependencyIndex);
    }
    for (const DFBNonTransactionalAccess &nonTransactionalAccess :
         access.getDFBNonTransactionalAccesses()) {
      assert(nonTransactionalAccess.dependencyIndex <
                 dependencyLogicalIndices.size() &&
             "non-transactional access must reference a dependency occurrence");
      std::optional<unsigned> logicalIndex =
          dependencyLogicalIndices[nonTransactionalAccess.dependencyIndex];
      assert(logicalIndex &&
             "non-transactional access dependency must have DFB type");
      assert(nonTransactionalAccess.dfb ==
                 dfbOperands[nonTransactionalAccess.dependencyIndex] &&
             "non-transactional access must match its dependency occurrence");
      assert(
          !describedDependencies.test(nonTransactionalAccess.dependencyIndex) &&
          "verified dependency occurrence must have one access contract");
      logicalDFBs[*logicalIndex].accesses.push_back(
          {operation, nonTransactionalAccess.kind, 0,
           nonTransactionalAccess.sequenceIndex, LaunchNodeDomain::unknown(),
           nullptr});
      describedDependencies.set(nonTransactionalAccess.dependencyIndex);
    }
    for (auto [dependencyIndex, operand] : llvm::enumerate(dfbOperands)) {
      if (!isa<CircularBufferType>(operand.getType()) ||
          describedDependencies.test(dependencyIndex)) {
        continue;
      }
      std::optional<unsigned> logicalIndex =
          dependencyLogicalIndices[dependencyIndex];
      assert(logicalIndex && "DFB dependencies were validated above");
      DFBLogicalLifecycle &logicalDFB = logicalDFBs[*logicalIndex];
      bool opaqueExternalAccess = isa<OpaqueCallOp>(operation);
      logicalDFB.accesses.push_back({operation, std::monostate{}, 0, 0,
                                     LaunchNodeDomain::unknown(), nullptr,
                                     opaqueExternalAccess});
      logicalDFB.hasOpaqueExternalAccess |= opaqueExternalAccess;
    }
    return WalkResult::advance();
  });
  if (collectionResult.wasInterrupted()) {
    return failure();
  }

  if (failed(expandSelectedResetAllocationGroups(logicalDFBs, resetOccurrences,
                                                 analysisFailure))) {
    return failure();
  }

  for (SynchronizedResetOccurrence &reset : resetOccurrences) {
    if (!reset.allDFBs) {
      continue;
    }
    for (unsigned logicalIndex = 0; logicalIndex < logicalDFBs.size();
         ++logicalIndex) {
      reset.targetLogicalIndices.push_back(logicalIndex);
    }
  }

  for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    if (!logicalDFB.compilerCreated || logicalDFB.accesses.empty()) {
      continue;
    }
    auto hasEffect = [&](DFBProtocolEffectKind effect) {
      return llvm::any_of(logicalDFB.accesses,
                          [&](const DFBAccessOccurrence &access) {
                            return access.isProtocolEffect(effect);
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
    unsigned logicalIndex = &logicalDFB - logicalDFBs.data();
    bool hasResetTerminator =
        llvm::any_of(resetOccurrences, [&](const auto &reset) {
          return llvm::is_contained(reset.targetLogicalIndices, logicalIndex);
        });
    if (hasResetTerminator) {
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

struct ProgramOrderTopologyAccess {
  const DFBAccessOccurrence *access = nullptr;
  AccessRun run;

  bool operator==(const ProgramOrderTopologyAccess &rhs) const {
    return std::tie(access, run) == std::tie(rhs.access, rhs.run);
  }
};

// Inputs that determine event identity and source-order topology. Reset-to-
// access and protocol synchronization edges depend on domain selection and are
// added independently after any topology reuse.
struct ProgramOrderTopologyInputs {
  DenseSet<Operation *> modeledOperations;
  SmallVector<ProgramOrderTopologyAccess> accesses;
  SmallVector<ValidatedSynchronizedReset> synchronizedResets;
  SmallVector<ValidatedDFBReconfiguration> reconfigurations;

  bool operator==(const ProgramOrderTopologyInputs &rhs) const {
    if (modeledOperations.size() != rhs.modeledOperations.size() ||
        !llvm::all_of(modeledOperations, [&](Operation *operation) {
          return rhs.modeledOperations.contains(operation);
        })) {
      return false;
    }
    return std::tie(accesses, synchronizedResets, reconfigurations) ==
           std::tie(rhs.accesses, rhs.synchronizedResets, rhs.reconfigurations);
  }
};

static ProgramOrderTopologyInputs collectProgramOrderTopologyInputs(
    ArrayRef<DFBLogicalLifecycle> logicalDFBs,
    ArrayRef<ValidatedSynchronizedReset> synchronizedResets,
    ArrayRef<ValidatedDFBReconfiguration> reconfigurations,
    LaunchNodeCoord node, const AccessExecutionCounts &executionCounts,
    const AccessRuns &accessRuns,
    const StructuralOperationOrder &structuralOrder,
    bool includeUnknownDomains) {
  ProgramOrderTopologyInputs inputs;
  for (const ValidatedSynchronizedReset &reset : synchronizedResets) {
    for (Operation *operation : reset.participantOperations) {
      if (Operation *projected =
              structuralOrder.getTopLevelOperation(operation)) {
        inputs.modeledOperations.insert(projected);
      }
    }
  }
  for (const ValidatedDFBReconfiguration &reconfiguration : reconfigurations) {
    for (Operation *operation : reconfiguration.participantOperations) {
      if (Operation *projected =
              structuralOrder.getTopLevelOperation(operation)) {
        inputs.modeledOperations.insert(projected);
      }
    }
  }
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      if (!mayAccessLaunchNode(access, node, executionCounts,
                               includeUnknownDomains)) {
        continue;
      }
      Operation *projected =
          structuralOrder.getTopLevelOperation(access.operation);
      if (!projected) {
        continue;
      }
      inputs.modeledOperations.insert(projected);
      auto runIt = accessRuns.find(&access);
      if (runIt == accessRuns.end()) {
        continue;
      }
      bool directProtocolEvent = access.getProtocolEffect() &&
                                 projected == access.operation &&
                                 runIt->second.executionCount == 1;
      bool repeatedAccessEvents = runIt->second.executionCount > 1;
      bool nestedSingleAccessEvent =
          projected != access.operation && runIt->second.executionCount == 1;
      if (directProtocolEvent || repeatedAccessEvents ||
          nestedSingleAccessEvent) {
        inputs.accesses.push_back({&access, runIt->second});
      }
    }
  }
  inputs.synchronizedResets.append(synchronizedResets.begin(),
                                   synchronizedResets.end());
  inputs.reconfigurations.append(reconfigurations.begin(),
                                 reconfigurations.end());
  return inputs;
}

static LogicalResult validateSynchronizedResetDeclarations(
    ArrayRef<SynchronizedResetOccurrence> occurrences,
    DFBAnalysisFailure &analysisFailure) {
  struct ResetDeclaration {
    SynchronizedDFBResetAttr reset;
    bool allDFBs = false;
  };
  llvm::MapVector<int64_t, ResetDeclaration> resetByOrdinal;
  for (const SynchronizedResetOccurrence &occurrence : occurrences) {
    int64_t ordinal = occurrence.reset.getOrdinal();
    auto [resetIt, inserted] = resetByOrdinal.try_emplace(
        ordinal, ResetDeclaration{occurrence.reset, occurrence.allDFBs});
    if (!inserted && (resetIt->second.reset != occurrence.reset ||
                      resetIt->second.allDFBs != occurrence.allDFBs)) {
      analysisFailure.set(
          occurrence.operation,
          "synchronized DFB reset ordinal identifies inconsistent "
          "operation or participant set");
      return failure();
    }
  }
  for (const auto &resetEntry : resetByOrdinal) {
    SynchronizedDFBResetAttr reset = resetEntry.second.reset;
    for (LogicalKernelAttr participant : reset.getParticipants()) {
      if (llvm::none_of(occurrences, [&](const auto &occurrence) {
            return occurrence.reset == reset &&
                   occurrence.participant == participant;
          })) {
        auto evidenceIt =
            llvm::find_if(occurrences, [&](const auto &occurrence) {
              return occurrence.reset == reset;
            });
        assert(evidenceIt != occurrences.end() &&
               "validated reset must have one occurrence");
        Operation *evidence = evidenceIt->operation;
        analysisFailure.set(
            evidence,
            "synchronized DFB reset is missing a declared logical-kernel "
            "participant");
        return failure();
      }
    }
  }
  return success();
}

static LogicalResult validateDFBReconfigurationDeclarations(
    ArrayRef<DFBReconfigurationOccurrence> occurrences,
    DFBAnalysisFailure &analysisFailure) {
  llvm::MapVector<int64_t, DFBReconfigurationAttr> boundaryByOrdinal;
  DFBReconfigurationAttr referenceBoundary;
  for (const DFBReconfigurationOccurrence &occurrence : occurrences) {
    int64_t ordinal = occurrence.boundary.getOrdinal();
    auto [boundaryIt, inserted] =
        boundaryByOrdinal.try_emplace(ordinal, occurrence.boundary);
    if (!inserted && boundaryIt->second != occurrence.boundary) {
      analysisFailure.set(
          occurrence.operation,
          "DFB reconfiguration ordinal identifies an inconsistent "
          "participant set");
      return failure();
    }
    if (!referenceBoundary) {
      referenceBoundary = occurrence.boundary;
    } else if (referenceBoundary.getParticipants() !=
               occurrence.boundary.getParticipants()) {
      analysisFailure.set(
          occurrence.operation,
          "all DFB reconfiguration boundaries must declare the same "
          "participant set");
      return failure();
    }
  }
  for (const auto &boundaryEntry : boundaryByOrdinal) {
    DFBReconfigurationAttr boundary = boundaryEntry.second;
    for (LogicalKernelAttr participant : boundary.getParticipants()) {
      if (llvm::none_of(occurrences, [&](const auto &occurrence) {
            return occurrence.boundary == boundary &&
                   occurrence.participant == participant;
          })) {
        auto evidenceIt =
            llvm::find_if(occurrences, [&](const auto &occurrence) {
              return occurrence.boundary == boundary;
            });
        assert(evidenceIt != occurrences.end() &&
               "validated boundary must have one occurrence");
        analysisFailure.set(
            evidenceIt->operation,
            "DFB reconfiguration is missing a declared logical-kernel "
            "participant");
        return failure();
      }
    }
  }
  return success();
}

static LogicalResult validateDFBReconfigurationsAtNode(
    ArrayRef<DFBReconfigurationOccurrence> occurrences, LaunchNodeCoord node,
    const LivenessDomainState &domainState,
    const StructuralOperationOrder &structuralOrder,
    SmallVectorImpl<ValidatedDFBReconfiguration> &validatedBoundaries,
    DFBAnalysisFailure &analysisFailure) {
  llvm::MapVector<DFBReconfigurationAttr,
                  SmallVector<const DFBReconfigurationOccurrence *>>
      occurrencesByBoundary;
  for (const DFBReconfigurationOccurrence &occurrence : occurrences) {
    occurrencesByBoundary[occurrence.boundary].push_back(&occurrence);
  }

  for (auto &[boundary, groupedOccurrences] : occurrencesByBoundary) {
    ValidatedDFBReconfiguration validated;
    validated.boundary = boundary;
    bool anyParticipantActive = false;
    bool allParticipantsExact = true;
    bool allParticipantsConditional = true;
    Operation *referenceConditionalOperation = nullptr;
    for (LogicalKernelAttr participant : boundary.getParticipants()) {
      SmallVector<const DFBReconfigurationOccurrence *> activeOccurrences;
      bool participantMayBeActive = false;
      for (const DFBReconfigurationOccurrence *occurrence :
           groupedOccurrences) {
        if (occurrence->participant != participant ||
            !mayContainLaunchNode(occurrence->launchDomain, node,
                                  /*includeUnknownDomains=*/true)) {
          continue;
        }
        std::optional<std::uint64_t> executionCount =
            getExactExecutionCountAtLaunchNode(occurrence->operation, node,
                                               domainState);
        if (executionCount && *executionCount == 0) {
          continue;
        }
        participantMayBeActive = true;
        if (executionCount && *executionCount != 1) {
          analysisFailure.set(
              occurrence->operation,
              "DFB reconfiguration must execute at most once per dispatch "
              "and launch node");
          return failure();
        }
        if (!executionCount &&
            !structurallyExecutesAtMostOnce(occurrence->operation)) {
          analysisFailure.set(
              occurrence->operation,
              "DFB reconfiguration requires an exact zero-or-one dynamic "
              "instance count");
          return failure();
        }
        activeOccurrences.push_back(occurrence);
        allParticipantsExact &= executionCount.has_value();
        allParticipantsConditional &= !executionCount.has_value();
      }
      anyParticipantActive |= participantMayBeActive;
      if (activeOccurrences.size() > 1) {
        analysisFailure.set(
            activeOccurrences.back()->operation,
            "DFB reconfiguration has multiple dynamic instance candidates "
            "for one logical-kernel participant");
        return failure();
      }
      if (activeOccurrences.empty()) {
        validated.participantOperations.push_back(nullptr);
        continue;
      }
      Operation *participantOperation = activeOccurrences.front()->operation;
      validated.participantOperations.push_back(participantOperation);
      std::optional<std::uint64_t> executionCount =
          getExactExecutionCountAtLaunchNode(participantOperation, node,
                                             domainState);
      if (!executionCount) {
        if (referenceConditionalOperation &&
            !proveEquivalentConditionalExecutionAtLaunchNodes(
                referenceConditionalOperation, node, participantOperation, node,
                domainState)) {
          analysisFailure.set(
              participantOperation,
              "DFB reconfiguration participants execute under different "
              "structured conditions");
          return failure();
        }
        referenceConditionalOperation = participantOperation;
      }
    }
    if (!anyParticipantActive) {
      continue;
    }
    if (llvm::is_contained(validated.participantOperations, nullptr)) {
      Operation *evidence = *llvm::find_if(
          validated.participantOperations, [](Operation *participantOperation) {
            return participantOperation != nullptr;
          });
      analysisFailure.set(
          evidence,
          "DFB reconfiguration has inconsistent participant execution at "
          "one launch node");
      return failure();
    }
    if (!allParticipantsExact && !allParticipantsConditional) {
      analysisFailure.set(
          validated.participantOperations.front(),
          "DFB reconfiguration participants have inconsistent dynamic "
          "instance counts");
      return failure();
    }
    validated.conditionalExecution = allParticipantsConditional;
    validatedBoundaries.push_back(std::move(validated));
  }

  for (unsigned beforeIndex = 0; beforeIndex < validatedBoundaries.size();
       ++beforeIndex) {
    for (unsigned afterIndex = beforeIndex + 1;
         afterIndex < validatedBoundaries.size(); ++afterIndex) {
      const ValidatedDFBReconfiguration &lhs = validatedBoundaries[beforeIndex];
      const ValidatedDFBReconfiguration &rhs = validatedBoundaries[afterIndex];
      std::optional<bool> lhsPrecedes;
      for (auto [lhsOperation, rhsOperation] : llvm::zip_equal(
               lhs.participantOperations, rhs.participantOperations)) {
        bool participantLhsPrecedes =
            structuralOrder.precedes(lhsOperation, rhsOperation);
        bool participantRhsPrecedes =
            structuralOrder.precedes(rhsOperation, lhsOperation);
        if (participantLhsPrecedes == participantRhsPrecedes) {
          analysisFailure.set(
              rhsOperation,
              "DFB reconfiguration boundaries must have a strict structured "
              "order in every participant");
          return failure();
        }
        if (!lhsPrecedes) {
          lhsPrecedes = participantLhsPrecedes;
        } else if (*lhsPrecedes != participantLhsPrecedes) {
          analysisFailure.set(
              rhsOperation,
              "DFB reconfiguration participants execute boundaries in "
              "different orders");
          return failure();
        }
      }
    }
  }
  return success();
}

static LogicalResult validateSynchronizedResetsAtNode(
    ArrayRef<SynchronizedResetOccurrence> occurrences, LaunchNodeCoord node,
    const LivenessDomainState &domainState,
    SmallVectorImpl<ValidatedSynchronizedReset> &validatedResets,
    DFBAnalysisFailure &analysisFailure) {
  llvm::MapVector<SynchronizedDFBResetAttr,
                  SmallVector<const SynchronizedResetOccurrence *>>
      occurrencesByReset;
  for (const SynchronizedResetOccurrence &occurrence : occurrences) {
    occurrencesByReset[occurrence.reset].push_back(&occurrence);
  }

  for (auto &[reset, groupedOccurrences] : occurrencesByReset) {
    ValidatedSynchronizedReset validated;
    validated.reset = reset;
    bool anyParticipantActive = false;
    bool allParticipantsExact = true;
    bool allParticipantsConditional = true;
    Operation *referenceConditionalOperation = nullptr;
    std::optional<std::uint64_t> referenceExecutionCount;
    std::optional<StaticIterationDomain> referenceIterationDomain;
    for (LogicalKernelAttr participant : reset.getParticipants()) {
      SmallVector<const SynchronizedResetOccurrence *> activeOccurrences;
      bool participantMayBeActive = false;
      for (const SynchronizedResetOccurrence *occurrence : groupedOccurrences) {
        if (occurrence->participant != participant ||
            !mayContainLaunchNode(occurrence->launchDomain, node,
                                  /*includeUnknownDomains=*/true)) {
          continue;
        }
        std::optional<std::uint64_t> executionCount =
            getExactExecutionCountAtLaunchNode(occurrence->operation, node,
                                               domainState);
        if (executionCount && *executionCount == 0) {
          continue;
        }
        participantMayBeActive = true;
        if (!executionCount &&
            !structurallyExecutesAtMostOnce(occurrence->operation)) {
          analysisFailure.set(
              occurrence->operation,
              "synchronized DFB reset must execute at most once per dispatch "
              "and launch node or once per iteration of an immutable "
              "sequential structured loop nest");
          return failure();
        }
        activeOccurrences.push_back(occurrence);
        allParticipantsExact &= executionCount.has_value();
        allParticipantsConditional &= !executionCount.has_value();
      }
      anyParticipantActive |= participantMayBeActive;
      if (activeOccurrences.size() > 1) {
        analysisFailure.set(
            activeOccurrences.back()->operation,
            "synchronized DFB reset has multiple dynamic instance candidates "
            "for one logical-kernel participant");
        return failure();
      }
      if (activeOccurrences.empty()) {
        validated.participantOperations.push_back(nullptr);
        validated.participantIterationDomains.emplace_back();
        continue;
      }
      const SynchronizedResetOccurrence &occurrence =
          *activeOccurrences.front();
      if (validated.targetLogicalIndices.empty()) {
        validated.targetLogicalIndices = occurrence.targetLogicalIndices;
      } else if (validated.targetLogicalIndices !=
                 occurrence.targetLogicalIndices) {
        analysisFailure.set(
            occurrence.operation,
            "synchronized DFB reset participants must declare identical "
            "target sets");
        return failure();
      }
      std::optional<std::uint64_t> executionCount =
          getExactExecutionCountAtLaunchNode(occurrence.operation, node,
                                             domainState);
      StaticIterationDomain iterationDomain;
      if (executionCount) {
        if (*executionCount > 1) {
          std::optional<StaticIterationDomain> uniformDomain =
              getUniformStaticIterationDomain(
                  occurrence.operation, *executionCount, node, domainState);
          if (!uniformDomain || uniformDomain->loops.empty()) {
            analysisFailure.set(
                occurrence.operation,
                ("repeated synchronized DFB reset with exact count " +
                 llvm::Twine(*executionCount) +
                 " must execute once in every iteration of an immutable "
                 "sequential structured loop nest")
                    .str());
            return failure();
          }
          iterationDomain = std::move(*uniformDomain);
        }
        if (!referenceExecutionCount) {
          referenceExecutionCount = executionCount;
          referenceIterationDomain = iterationDomain;
        } else if (*referenceExecutionCount != *executionCount ||
                   !hasEquivalentIterationSequence(*referenceIterationDomain,
                                                   iterationDomain)) {
          analysisFailure.set(
              occurrence.operation,
              "synchronized DFB reset participants must execute in the same "
              "structured iteration sequence");
          return failure();
        }
      } else {
        if (referenceConditionalOperation &&
            !proveEquivalentConditionalExecutionAtLaunchNodes(
                referenceConditionalOperation, node, occurrence.operation, node,
                domainState)) {
          analysisFailure.set(
              occurrence.operation,
              "synchronized DFB reset participants execute under different "
              "structured conditions");
          return failure();
        }
        referenceConditionalOperation = occurrence.operation;
      }
      validated.participantOperations.push_back(occurrence.operation);
      validated.participantIterationDomains.push_back(
          std::move(iterationDomain));
    }
    if (!anyParticipantActive) {
      continue;
    }
    if (llvm::is_contained(validated.participantOperations, nullptr)) {
      auto evidenceIt = llvm::find_if(
          validated.participantOperations,
          [](Operation *operation) { return operation != nullptr; });
      assert(evidenceIt != validated.participantOperations.end() &&
             "active reset must have one participant operation");
      Operation *evidence = *evidenceIt;
      analysisFailure.set(
          evidence,
          "synchronized DFB reset has inconsistent participant execution at "
          "one launch node");
      return failure();
    }
    if (!allParticipantsExact && !allParticipantsConditional) {
      analysisFailure.set(
          validated.participantOperations.front(),
          "synchronized DFB reset participants have inconsistent dynamic "
          "instance counts");
      return failure();
    }
    validated.conditionalExecution = allParticipantsConditional;
    validated.executionCount = referenceExecutionCount.value_or(1);
    validatedResets.push_back(std::move(validated));
  }
  return success();
}

struct ProgramOrderGraphState {
  HappensBeforeGraph graph;
  DenseMap<Operation *, EventPair> operationEvents;
  DenseMap<const DFBAccessOccurrence *, AccessEventSpan> accessEvents;
  ResetBoundaryEvents resetBoundaryEvents;
  ReconfigurationBoundaryEvents reconfigurationBoundaryEvents;

  bool operator==(const ProgramOrderGraphState &rhs) const {
    return std::tie(graph, operationEvents, accessEvents, resetBoundaryEvents,
                    reconfigurationBoundaryEvents) ==
           std::tie(rhs.graph, rhs.operationEvents, rhs.accessEvents,
                    rhs.resetBoundaryEvents, rhs.reconfigurationBoundaryEvents);
  }
};

// Builds source-order events only for active accesses. Direct protocol effects
// receive separate events in their declared sequence; operations in different
// kernels remain concurrent unless protocol edges order them.
static void buildProgramOrderTopology(
    ModuleOp module, const ProgramOrderTopologyInputs &inputs,
    const StructuralOperationOrder &structuralOrder, HappensBeforeGraph &graph,
    DenseMap<Operation *, EventPair> &operationEvents,
    DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    ResetBoundaryEvents &resetBoundaryEvents,
    ReconfigurationBoundaryEvents &reconfigurationBoundaryEvents) {
  DenseMap<Operation *, SmallVector<const DFBAccessOccurrence *>>
      directProtocolAccesses;
  DenseMap<Operation *, SmallVector<const DFBAccessOccurrence *>>
      projectedAccesses;
  DenseMap<const DFBAccessOccurrence *, const AccessRun *> accessRuns;
  for (const ProgramOrderTopologyAccess &input : inputs.accesses) {
    const DFBAccessOccurrence *access = input.access;
    accessRuns.try_emplace(access, &input.run);
    if (Operation *projected =
            structuralOrder.getTopLevelOperation(access->operation)) {
      projectedAccesses[projected].push_back(access);
      if (access->getProtocolEffect() && projected == access->operation &&
          input.run.executionCount == 1) {
        directProtocolAccesses[projected].push_back(access);
      }
    }
  }
  ArrayRef<ValidatedSynchronizedReset> synchronizedResets =
      inputs.synchronizedResets;
  ArrayRef<ValidatedDFBReconfiguration> reconfigurations =
      inputs.reconfigurations;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.getBody().empty() || !function.getBody().hasOneBlock()) {
      continue;
    }
    std::optional<EventPair> previousEvents;
    for (Operation &operation : function.getBody().front()) {
      if (!inputs.modeledOperations.contains(&operation)) {
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

  // Enclosing events provide every valid relation between projections.
  for (auto &[projected, accesses] : projectedAccesses) {
    EventPair projectedEvents = operationEvents.lookup(projected);
    for (const DFBAccessOccurrence *access : accesses) {
      auto runIt = accessRuns.find(access);
      if (runIt == accessRuns.end() || runIt->second->executionCount <= 1) {
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
          beforeRunIt->second->executionCount <= 1 ||
          beforeEventsIt == accessEvents.end()) {
        continue;
      }
      for (const DFBAccessOccurrence *afterAccess : accesses) {
        auto afterRunIt = accessRuns.find(afterAccess);
        auto afterEventsIt = accessEvents.find(afterAccess);
        if (afterRunIt == accessRuns.end() ||
            afterRunIt->second->executionCount <= 1 ||
            afterEventsIt == accessEvents.end() ||
            !runPrecedesWithinEachIteration(
                *beforeRunIt->second, *afterRunIt->second, structuralOrder)) {
          continue;
        }
        addPerIterationSpanOrder(graph, beforeEventsIt->second,
                                 afterEventsIt->second);
      }
    }
  }

  for (auto &[projected, accesses] : projectedAccesses) {
    EventPair projectedEvents = operationEvents.lookup(projected);
    SmallVector<const DFBAccessOccurrence *> nestedSingleAccesses;
    for (const DFBAccessOccurrence *access : accesses) {
      auto runIt = accessRuns.find(access);
      if (access->operation == projected || runIt == accessRuns.end() ||
          runIt->second->executionCount != 1) {
        continue;
      }
      EventPair events = graph.addOperation();
      accessEvents[access] = {events, events};
      graph.addEdge(projectedEvents.entry, events.entry);
      graph.addEdge(events.completion, projectedEvents.completion);
      nestedSingleAccesses.push_back(access);
    }
    for (const DFBAccessOccurrence *before : nestedSingleAccesses) {
      for (const DFBAccessOccurrence *after : nestedSingleAccesses) {
        if (before == after) {
          continue;
        }
        if (accessOccurrencePrecedes(*before, *after, structuralOrder)) {
          graph.addEdge(accessEvents.at(before).last.completion,
                        accessEvents.at(after).first.entry);
        }
      }
    }
  }

  for (const ValidatedSynchronizedReset &reset : synchronizedResets) {
    SmallVector<EventPair> participantEvents;
    for (Operation *operation : reset.participantOperations) {
      std::optional<EventPair> events =
          getProjectedEvents(operation, operationEvents);
      if (!events) {
        participantEvents.clear();
        break;
      }
      participantEvents.push_back(*events);
    }
    if (participantEvents.size() != reset.participantOperations.size()) {
      continue;
    }
    EventPair firstEvents = graph.addOperation();
    EventPair lastEvents = firstEvents;
    if (reset.executionCount > 1) {
      lastEvents = graph.addOperation();
      graph.addEdge(firstEvents.completion, lastEvents.entry);
    }
    resetBoundaryEvents.try_emplace(reset.reset,
                                    AccessEventSpan{firstEvents, lastEvents});
    for (const EventPair &participant : participantEvents) {
      graph.addEdge(participant.entry, firstEvents.entry);
      graph.addEdge(lastEvents.completion, participant.completion);
    }
  }

  for (const ValidatedSynchronizedReset &before : synchronizedResets) {
    for (const ValidatedSynchronizedReset &after : synchronizedResets) {
      if (before.reset == after.reset ||
          before.reset.getParticipants() != after.reset.getParticipants()) {
        continue;
      }
      bool everyParticipantOrdered =
          llvm::all_of(llvm::zip_equal(before.participantOperations,
                                       after.participantOperations),
                       [&](auto pair) {
                         return structuralOrder.precedes(std::get<0>(pair),
                                                         std::get<1>(pair));
                       });
      if (everyParticipantOrdered) {
        auto beforeEvents = resetBoundaryEvents.find(before.reset);
        auto afterEvents = resetBoundaryEvents.find(after.reset);
        if (beforeEvents != resetBoundaryEvents.end() &&
            afterEvents != resetBoundaryEvents.end()) {
          bool matchingRepeatedSequence =
              before.executionCount > 1 &&
              before.executionCount == after.executionCount &&
              llvm::equal(before.participantIterationDomains,
                          after.participantIterationDomains);
          if (matchingRepeatedSequence) {
            addPerIterationSpanOrder(graph, beforeEvents->second,
                                     afterEvents->second);
          } else if (before.executionCount == 1 && after.executionCount == 1) {
            graph.addEdge(beforeEvents->second.last.completion,
                          afterEvents->second.first.entry);
          }
        }
      }
    }
  }

  for (const ValidatedDFBReconfiguration &reconfiguration : reconfigurations) {
    SmallVector<EventPair> participantEvents;
    for (Operation *operation : reconfiguration.participantOperations) {
      std::optional<EventPair> events =
          getProjectedEvents(operation, operationEvents);
      if (!events) {
        participantEvents.clear();
        break;
      }
      participantEvents.push_back(*events);
    }
    if (participantEvents.size() !=
        reconfiguration.participantOperations.size()) {
      continue;
    }
    EventPair boundaryEvents = graph.addOperation();
    reconfigurationBoundaryEvents.try_emplace(reconfiguration.boundary,
                                              boundaryEvents);
    for (const EventPair &participant : participantEvents) {
      graph.addEdge(participant.entry, boundaryEvents.entry);
      graph.addEdge(boundaryEvents.completion, participant.completion);
    }
  }

  for (const ValidatedDFBReconfiguration &before : reconfigurations) {
    for (const ValidatedDFBReconfiguration &after : reconfigurations) {
      if (before.boundary == after.boundary) {
        continue;
      }
      bool everyParticipantOrdered =
          llvm::all_of(llvm::zip_equal(before.participantOperations,
                                       after.participantOperations),
                       [&](auto pair) {
                         return structuralOrder.precedes(std::get<0>(pair),
                                                         std::get<1>(pair));
                       });
      if (!everyParticipantOrdered) {
        continue;
      }
      auto beforeEvents = reconfigurationBoundaryEvents.find(before.boundary);
      auto afterEvents = reconfigurationBoundaryEvents.find(after.boundary);
      if (beforeEvents != reconfigurationBoundaryEvents.end() &&
          afterEvents != reconfigurationBoundaryEvents.end()) {
        graph.addEdge(beforeEvents->second.completion,
                      afterEvents->second.entry);
      }
    }
  }
}

static void addSynchronizedResetAccessEdges(
    ArrayRef<DFBLogicalLifecycle> logicalDFBs,
    ArrayRef<ValidatedSynchronizedReset> synchronizedResets,
    LaunchNodeCoord node, HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const ResetBoundaryEvents &resetBoundaryEvents,
    const AccessExecutionCounts &executionCounts, const AccessRuns &accessRuns,
    const StructuralOperationOrder &structuralOrder,
    bool includeUnknownDomains) {
  for (const ValidatedSynchronizedReset &reset : synchronizedResets) {
    auto boundaryIt = resetBoundaryEvents.find(reset.reset);
    if (boundaryIt == resetBoundaryEvents.end()) {
      continue;
    }
    AccessEventSpan boundaryEvents = boundaryIt->second;
    for (unsigned logicalIndex : reset.targetLogicalIndices) {
      for (const DFBAccessOccurrence &access :
           logicalDFBs[logicalIndex].accesses) {
        if (!mayAccessLaunchNode(access, node, executionCounts,
                                 includeUnknownDomains)) {
          continue;
        }
        std::optional<AccessEventSpan> events =
            getAccessEventSpan(access, operationEvents, accessEvents);
        if (!events) {
          continue;
        }
        Operation *accessFunction =
            structuralOrder.getFunction(access.operation);
        Operation *localReset = nullptr;
        const StaticIterationDomain *localResetDomain = nullptr;
        for (auto [participantIndex, participant] :
             llvm::enumerate(reset.participantOperations)) {
          if (structuralOrder.getFunction(participant) == accessFunction) {
            localReset = participant;
            localResetDomain =
                &reset.participantIterationDomains[participantIndex];
            break;
          }
        }
        if (!localReset) {
          continue;
        }
        bool sameProjectedOperation =
            structuralOrder.getTopLevelOperation(access.operation) ==
            structuralOrder.getTopLevelOperation(localReset);
        if (!accessEvents.contains(&access) && sameProjectedOperation) {
          // The enclosing event spans both operations and cannot order them.
          continue;
        }
        auto runIt = accessRuns.find(&access);
        bool matchingRepeatedDomain =
            reset.executionCount > 1 && runIt != accessRuns.end() &&
            runIt->second.executionCount == reset.executionCount &&
            runIt->second.iterationDomain == *localResetDomain;
        if (matchingRepeatedDomain) {
          if (structuralOrder.precedes(access.operation, localReset)) {
            addPerIterationSpanOrder(graph, *events, boundaryEvents);
          } else if (structuralOrder.precedes(localReset, access.operation)) {
            addPerIterationSpanOrder(graph, boundaryEvents, *events);
          }
          continue;
        }
        if (reset.executionCount > 1 && sameProjectedOperation) {
          continue;
        }
        if (structuralOrder.precedes(access.operation, localReset)) {
          graph.addEdge(events->last.completion, boundaryEvents.first.entry);
        } else if (structuralOrder.precedes(localReset, access.operation)) {
          graph.addEdge(boundaryEvents.last.completion, events->first.entry);
        }
      }
    }
  }
}

static void addDFBReconfigurationAccessEdges(
    ArrayRef<DFBLogicalLifecycle> logicalDFBs,
    ArrayRef<ValidatedDFBReconfiguration> reconfigurations,
    LaunchNodeCoord node, HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const ReconfigurationBoundaryEvents &reconfigurationBoundaryEvents,
    const AccessExecutionCounts &executionCounts,
    const StructuralOperationOrder &structuralOrder,
    bool includeUnknownDomains) {
  for (const ValidatedDFBReconfiguration &reconfiguration : reconfigurations) {
    auto boundaryIt =
        reconfigurationBoundaryEvents.find(reconfiguration.boundary);
    if (boundaryIt == reconfigurationBoundaryEvents.end()) {
      continue;
    }
    EventPair boundaryEvents = boundaryIt->second;
    for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
      for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
        if (!mayAccessLaunchNode(access, node, executionCounts,
                                 includeUnknownDomains)) {
          continue;
        }
        std::optional<AccessEventSpan> events =
            getAccessEventSpan(access, operationEvents, accessEvents);
        if (!events) {
          continue;
        }
        Operation *accessFunction =
            structuralOrder.getFunction(access.operation);
        Operation *localBoundary = nullptr;
        for (Operation *participant : reconfiguration.participantOperations) {
          if (structuralOrder.getFunction(participant) == accessFunction) {
            localBoundary = participant;
            break;
          }
        }
        if (!localBoundary) {
          continue;
        }
        if (structuralOrder.precedes(access.operation, localBoundary)) {
          graph.addEdge(events->last.completion, boundaryEvents.entry);
        } else if (structuralOrder.precedes(localBoundary, access.operation)) {
          graph.addEdge(boundaryEvents.completion, events->first.entry);
        }
      }
    }
  }
}

static ProgramOrderGraphState buildProgramOrderTopologyState(
    ModuleOp module, const ProgramOrderTopologyInputs &inputs,
    const StructuralOperationOrder &structuralOrder) {
  ProgramOrderGraphState state;
  buildProgramOrderTopology(module, inputs, structuralOrder, state.graph,
                            state.operationEvents, state.accessEvents,
                            state.resetBoundaryEvents,
                            state.reconfigurationBoundaryEvents);
  return state;
}

// Conditional runs match only under equivalent structured conditions.
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

using AccessRunMatchCallback = llvm::function_ref<LogicalResult(
    const AccessRun &, std::uint64_t, const AccessRun &, std::uint64_t,
    std::uint64_t)>;
using AccessRunPairCompatibility =
    llvm::function_ref<bool(const AccessRun &, const AccessRun &)>;

struct AccessRunMatchResult {
  std::size_t sourceIndex = 0;
  std::size_t targetIndex = 0;
  std::uint64_t sourceOffset = 0;
  std::uint64_t targetOffset = 0;

  bool fullyMatched(ArrayRef<const AccessRun *> sources,
                    ArrayRef<const AccessRun *> targets) const {
    return sourceIndex == sources.size() && targetIndex == targets.size();
  }
};

static bool accessRunsCanMatch(const AccessRun &source, const AccessRun &target,
                               LaunchNodeCoord node,
                               const LaunchNodeDomainState &domainState,
                               AccessRunPairCompatibility runsAreCompatible) {
  return source.access->numTiles == target.access->numTiles &&
         proveEquivalentConditionalRuns(source, target, node, domainState) &&
         runsAreCompatible(source, target);
}

static AccessRunMatchResult
matchAccessRunPrefix(ArrayRef<const AccessRun *> sources,
                     ArrayRef<const AccessRun *> targets, LaunchNodeCoord node,
                     const LaunchNodeDomainState &domainState,
                     AccessRunPairCompatibility runsAreCompatible,
                     AccessRunMatchCallback recordMatch) {
  AccessRunMatchResult result;
  while (result.sourceIndex < sources.size() &&
         result.targetIndex < targets.size()) {
    const AccessRun &source = *sources[result.sourceIndex];
    const AccessRun &target = *targets[result.targetIndex];
    if (!accessRunsCanMatch(source, target, node, domainState,
                            runsAreCompatible)) {
      break;
    }
    std::uint64_t matchedCount =
        std::min(source.executionCount - result.sourceOffset,
                 target.executionCount - result.targetOffset);
    if (failed(recordMatch(source, result.sourceOffset, target,
                           result.targetOffset, matchedCount))) {
      break;
    }
    result.sourceOffset += matchedCount;
    result.targetOffset += matchedCount;
    if (result.sourceOffset == source.executionCount) {
      ++result.sourceIndex;
      result.sourceOffset = 0;
    }
    if (result.targetOffset == target.executionCount) {
      ++result.targetIndex;
      result.targetOffset = 0;
    }
  }
  return result;
}

static bool matchAccessRuns(ArrayRef<const AccessRun *> sources,
                            ArrayRef<const AccessRun *> targets,
                            LaunchNodeCoord node,
                            const LaunchNodeDomainState &domainState,
                            AccessRunPairCompatibility runsAreCompatible,
                            AccessRunMatchCallback recordMatch) {
  return matchAccessRunPrefix(sources, targets, node, domainState,
                              runsAreCompatible, recordMatch)
      .fullyMatched(sources, targets);
}

static FailureOr<SmallVector<const AccessRun *>> orderProtocolRuns(
    ArrayRef<const AccessRun *> runs, const HappensBeforeGraph &graph,
    const StructuralOperationOrder &structuralOrder,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents);

static bool tryAddCumulativeQueueEdges(
    const DFBLogicalLifecycle &logicalDFB, HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, const AccessRuns &accessRuns,
    LaunchNodeCoord node, const LaunchNodeDomainState &domainState,
    const StructuralOperationOrder &structuralOrder,
    bool includeUnknownDomains);

static bool
hasRepeatedOpaqueEffectOperation(ArrayRef<const AccessRun *> protocolRuns) {
  DenseSet<Operation *> operations;
  return llvm::any_of(protocolRuns, [&](const AccessRun *run) {
    Operation *operation = run->access->operation;
    return isa<OpaqueCallOp>(operation) && !operations.insert(operation).second;
  });
}

static bool
supportsCumulativeQueueProof(ArrayRef<const AccessRun *> protocolRuns) {
  return !protocolRuns.empty() &&
         llvm::all_of(protocolRuns, [](const AccessRun *run) {
           return run->executionCount == 1 &&
                  isa<OpaqueCallOp>(run->access->operation);
         });
}

// Adds exact and cumulative producer-to-consumer synchronization edges.
static void addProtocolSynchronizationEdges(
    ArrayRef<DFBLogicalLifecycle> logicalDFBs, HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, const AccessRuns &accessRuns,
    LaunchNodeCoord node, const LaunchNodeDomainState &domainState,
    const StructuralOperationOrder &structuralOrder,
    bool includeUnknownDomains) {
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    SmallVector<const AccessRun *> reserves;
    SmallVector<const AccessRun *> pushes;
    SmallVector<const AccessRun *> waits;
    SmallVector<const AccessRun *> pops;
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      auto runIt = accessRuns.find(&access);
      if (runIt == accessRuns.end()) {
        continue;
      }
      if (access.isProtocolEffect(DFBProtocolEffectKind::Reserve)) {
        reserves.push_back(&runIt->second);
      } else if (access.isProtocolEffect(DFBProtocolEffectKind::Push)) {
        pushes.push_back(&runIt->second);
      } else if (access.isProtocolEffect(DFBProtocolEffectKind::Wait)) {
        waits.push_back(&runIt->second);
      } else if (access.isProtocolEffect(DFBProtocolEffectKind::Pop)) {
        pops.push_back(&runIt->second);
      }
    }

    SmallVector<const AccessRun *> producerProtocolRuns;
    llvm::append_range(producerProtocolRuns, reserves);
    llvm::append_range(producerProtocolRuns, pushes);
    SmallVector<const AccessRun *> consumerProtocolRuns;
    llvm::append_range(consumerProtocolRuns, waits);
    llvm::append_range(consumerProtocolRuns, pops);

    // Repeated effects in one opaque call share one completion event. Exact
    // matching would impose an all-before-all relation; cumulative analysis
    // can replace those edges only when it supports both complete schedules.
    if (supportsCumulativeQueueProof(producerProtocolRuns) &&
        supportsCumulativeQueueProof(consumerProtocolRuns) &&
        (hasRepeatedOpaqueEffectOperation(pushes) ||
         hasRepeatedOpaqueEffectOperation(waits))) {
      continue;
    }

    SmallVector<std::pair<unsigned, unsigned>> synchronizationEdges;
    bool matched = matchAccessRuns(
        pushes, waits, node, domainState,
        [](const AccessRun &, const AccessRun &) { return true; },
        [&](const AccessRun &push, std::uint64_t pushOffset,
            const AccessRun &wait, std::uint64_t waitOffset,
            std::uint64_t matchedCount) {
          std::optional<AccessEventSpan> pushEvents =
              getAccessEventSpan(*push.access, operationEvents, accessEvents);
          std::optional<AccessEventSpan> waitEvents =
              getAccessEventSpan(*wait.access, operationEvents, accessEvents);
          if (!pushEvents || !waitEvents) {
            return failure();
          }
          if (pushOffset == 0 && waitOffset == 0) {
            synchronizationEdges.emplace_back(pushEvents->first.completion,
                                              waitEvents->first.completion);
          }
          if (pushOffset + matchedCount == push.executionCount &&
              waitOffset + matchedCount == wait.executionCount) {
            synchronizationEdges.emplace_back(pushEvents->last.completion,
                                              waitEvents->last.completion);
          }
          return success();
        });
    if (matched) {
      for (auto [pushCompletion, waitCompletion] : synchronizationEdges) {
        graph.addEdge(pushCompletion, waitCompletion);
      }
    }
  }
  // Cumulative queue proofs must observe every exact push/wait edge. This also
  // makes the result independent of logical DFB collection order.
  graph.computeReachability();
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    tryAddCumulativeQueueEdges(logicalDFB, graph, operationEvents, accessEvents,
                               executionCounts, accessRuns, node, domainState,
                               structuralOrder, includeUnknownDomains);
  }
}

// Proves ordering between corresponding executions with equal counts and
// iteration domains, not all-before-all ordering across the complete runs.
static bool proveRunBeforeWithinEachIteration(
    const AccessRun &before, const AccessRun &after,
    const HappensBeforeGraph &graph,
    const StructuralOperationOrder &structuralOrder,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  if (before.executionCount != after.executionCount ||
      !(before.iterationDomain == after.iterationDomain)) {
    return false;
  }
  if (runPrecedesWithinEachIteration(before, after, structuralOrder)) {
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
    const StructuralOperationOrder &structuralOrder,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  if (before.executionCount == 1 && after.executionCount == 1 &&
      runPrecedesWithinEachIteration(before, after, structuralOrder)) {
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

// Run order may be pointwise within each iteration or all-before-all across
// the complete runs.
static bool
proveRunPrecedes(const AccessRun &before, const AccessRun &after,
                 const HappensBeforeGraph &graph,
                 const StructuralOperationOrder &structuralOrder,
                 const DenseMap<Operation *, EventPair> &operationEvents,
                 const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
                     &accessEvents) {
  return proveRunBeforeWithinEachIteration(before, after, graph,
                                           structuralOrder, operationEvents,
                                           accessEvents) ||
         proveAllRunExecutionsBefore(before, after, graph, structuralOrder,
                                     operationEvents, accessEvents);
}

static bool acquireReleaseRunsAlign(
    const AccessRun &acquire, const AccessRun &release,
    const HappensBeforeGraph &graph,
    const StructuralOperationOrder &structuralOrder,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  bool nativeAcquirePrecedesRelease =
      ((isa<CBReserveOp>(acquire.access->operation) &&
        isa<CBPushOp>(release.access->operation)) ||
       (isa<CBWaitOp>(acquire.access->operation) &&
        isa<CBPopOp>(release.access->operation))) &&
      acquire.access->operation->getBlock() ==
          release.access->operation->getBlock() &&
      acquire.access->operation->isBeforeInBlock(release.access->operation);
  return acquire.executionCount == release.executionCount &&
         acquire.iterationDomain == release.iterationDomain &&
         (nativeAcquirePrecedesRelease ||
          proveRunBeforeWithinEachIteration(acquire, release, graph,
                                            structuralOrder, operationEvents,
                                            accessEvents));
}

static bool proveAlignedAcquireReleaseRuns(
    ArrayRef<const AccessRun *> acquires, ArrayRef<const AccessRun *> releases,
    const HappensBeforeGraph &graph,
    const StructuralOperationOrder &structuralOrder,
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
        return acquire.access->numTiles == release.access->numTiles &&
               acquireReleaseRunsAlign(acquire, release, graph, structuralOrder,
                                       operationEvents, accessEvents);
      });
  if (!pairsAreAligned) {
    return false;
  }
  for (std::size_t runIndex = 1; runIndex < acquires.size(); ++runIndex) {
    if (!proveRunPrecedes(*releases[runIndex - 1], *acquires[runIndex], graph,
                          structuralOrder, operationEvents, accessEvents)) {
      return false;
    }
  }
  return true;
}

static bool
runIsInsideInterval(const AccessRun &use, const AccessRun &acquire,
                    const AccessRun &release, const HappensBeforeGraph &graph,
                    const StructuralOperationOrder &structuralOrder,
                    const DenseMap<Operation *, EventPair> &operationEvents,
                    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
                        &accessEvents) {
  return proveRunBeforeWithinEachIteration(acquire, use, graph, structuralOrder,
                                           operationEvents, accessEvents) &&
         proveRunBeforeWithinEachIteration(use, release, graph, structuralOrder,
                                           operationEvents, accessEvents);
}

static void appendTransactionRun(SmallVectorImpl<DFBTransactionRun> &runs,
                                 std::uint64_t executionCount,
                                 int64_t tilesPerExecution) {
  if (executionCount == 0) {
    return;
  }
  if (!runs.empty() && runs.back().tilesPerExecution == tilesPerExecution) {
    runs.back().executionCount += executionCount;
    return;
  }
  runs.push_back({executionCount, tilesPerExecution});
}

struct CumulativeQueueSide {
  SmallVector<const AccessRun *> orderedRuns;
  SmallVector<DFBTransactionRun> cursorRuns;
  SmallVector<std::pair<const AccessRun *, const AccessRun *>> intervals;
  std::optional<DFBPointerOwner> owner;
  std::uint64_t totalMovement = 0;
};

struct CumulativeQueueSideResult {
  std::optional<CumulativeQueueSide> side;
  DFBLifecycleCompletionProof failure;
};

static FailureOr<SmallVector<const AccessRun *>>
orderProtocolRuns(ArrayRef<const AccessRun *> runs,
                  const HappensBeforeGraph &graph,
                  const StructuralOperationOrder &structuralOrder,
                  const DenseMap<Operation *, EventPair> &operationEvents,
                  const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
                      &accessEvents) {
  Operation *firstOperation =
      runs.empty() ? nullptr : runs.front()->access->operation;
  bool effectsBelongToOneOperation =
      llvm::all_of(runs, [&](const AccessRun *run) {
        return run->access->operation == firstOperation;
      });
  // Effects from one operation are recorded in their execution order.
  if (effectsBelongToOneOperation) {
    SmallVector<const AccessRun *> ordered(runs.begin(), runs.end());
    llvm::sort(ordered, [](const AccessRun *lhs, const AccessRun *rhs) {
      return lhs->access->sequenceIndex < rhs->access->sequenceIndex;
    });
    return ordered;
  }

  SmallVector<unsigned> predecessorCounts(runs.size());
  for (auto [lhsIndex, lhs] : llvm::enumerate(runs)) {
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < runs.size(); ++rhsIndex) {
      const AccessRun *rhs = runs[rhsIndex];
      bool lhsBeforeRhs = proveRunPrecedes(*lhs, *rhs, graph, structuralOrder,
                                           operationEvents, accessEvents);
      bool rhsBeforeLhs = proveRunPrecedes(*rhs, *lhs, graph, structuralOrder,
                                           operationEvents, accessEvents);
      if (lhsBeforeRhs == rhsBeforeLhs) {
        return failure();
      }
      ++predecessorCounts[lhsBeforeRhs ? rhsIndex : lhsIndex];
    }
  }

  SmallVector<const AccessRun *> ordered(runs.size());
  llvm::BitVector occupiedRanks(runs.size());
  for (auto [runIndex, run] : llvm::enumerate(runs)) {
    unsigned rank = predecessorCounts[runIndex];
    if (rank >= ordered.size() || occupiedRanks.test(rank)) {
      return failure();
    }
    ordered[rank] = run;
    occupiedRanks.set(rank);
  }
  return ordered;
}

static CumulativeQueueSideResult proveCumulativeQueueSide(
    ArrayRef<const AccessRun *> unorderedRuns,
    DFBProtocolEffectKind acquireKind, DFBProtocolEffectKind releaseKind,
    int64_t physicalTileCount, LaunchNodeCoord node,
    const HappensBeforeGraph &graph,
    const StructuralOperationOrder &structuralOrder,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  FailureOr<SmallVector<const AccessRun *>> orderedRuns = orderProtocolRuns(
      unorderedRuns, graph, structuralOrder, operationEvents, accessEvents);
  if (failed(orderedRuns)) {
    return {std::nullopt,
            {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
             unorderedRuns.front()->access->operation}};
  }

  CumulativeQueueSide result;
  result.orderedRuns = *orderedRuns;
  SmallVector<const AccessRun *> activeAcquires;
  const AccessRun *lastRelease = nullptr;
  std::optional<std::uint64_t> readinessLimit;
  for (const AccessRun *run : *orderedRuns) {
    if (run->executionCount != 1 || run->access->numTiles <= 0 ||
        run->access->numTiles > physicalTileCount ||
        !run->access->getProtocolEffect()) {
      return {std::nullopt,
              {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
               run->access->operation}};
    }
    DFBProtocolEffectKind effect = *run->access->getProtocolEffect();
    if (effect != acquireKind && effect != releaseKind) {
      return {std::nullopt,
              {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
               run->access->operation}};
    }
    std::optional<DFBPointerOwner> owner =
        getPointerOwner(run->access->operation, node, effect);
    if (!owner || (result.owner && *result.owner != *owner)) {
      return {std::nullopt,
              {DFBLifecycleCompletionFailureReason::UnknownPointerOwner,
               run->access->operation}};
    }
    result.owner = owner;

    if (effect == acquireKind) {
      if (!activeAcquires.empty() && lastRelease) {
        for (const AccessRun *activeAcquire : activeAcquires) {
          result.intervals.emplace_back(activeAcquire, lastRelease);
        }
        activeAcquires.clear();
        lastRelease = nullptr;
      }
      activeAcquires.push_back(run);
      std::optional<std::uint64_t> acquiredLimit = llvm::checkedAddUnsigned(
          result.totalMovement,
          static_cast<std::uint64_t>(run->access->numTiles));
      if (!acquiredLimit) {
        return {std::nullopt,
                {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
                 run->access->operation}};
      }
      readinessLimit = std::max(readinessLimit.value_or(0), *acquiredLimit);
      continue;
    }

    if (activeAcquires.empty() || !readinessLimit) {
      return {std::nullopt,
              {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
               run->access->operation}};
    }
    std::optional<std::uint64_t> nextMovement = llvm::checkedAddUnsigned(
        result.totalMovement,
        static_cast<std::uint64_t>(run->access->numTiles));
    if (!nextMovement || *nextMovement > *readinessLimit) {
      return {std::nullopt,
              {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
               run->access->operation}};
    }
    result.totalMovement = *nextMovement;
    lastRelease = run;
    appendTransactionRun(result.cursorRuns, 1, run->access->numTiles);
  }
  if (activeAcquires.empty() || !lastRelease) {
    return {std::nullopt,
            {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
             unorderedRuns.front()->access->operation}};
  }
  for (const AccessRun *activeAcquire : activeAcquires) {
    result.intervals.emplace_back(activeAcquire, lastRelease);
  }
  return {std::move(result), {}};
}

static FailureOr<SmallVector<DFBTransactionRun>>
normalizeCumulativeTransactions(ArrayRef<DFBTransactionRun> writeRuns,
                                ArrayRef<DFBTransactionRun> readRuns) {
  SmallVector<std::uint64_t> boundaries;
  auto collectBoundaries = [&](ArrayRef<DFBTransactionRun> runs,
                               std::uint64_t &total) {
    std::uint64_t position = 0;
    for (const DFBTransactionRun &run : runs) {
      for (std::uint64_t execution = 0; execution < run.executionCount;
           ++execution) {
        std::optional<std::uint64_t> next = llvm::checkedAddUnsigned(
            position, static_cast<std::uint64_t>(run.tilesPerExecution));
        if (!next) {
          return failure();
        }
        position = *next;
        boundaries.push_back(position);
      }
    }
    total = position;
    return success();
  };
  std::uint64_t writeTotal = 0;
  if (failed(collectBoundaries(writeRuns, writeTotal))) {
    return failure();
  }
  std::uint64_t readTotal = 0;
  if (failed(collectBoundaries(readRuns, readTotal))) {
    return failure();
  }
  if (writeTotal == 0 || writeTotal != readTotal) {
    return failure();
  }
  llvm::sort(boundaries);
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                   boundaries.end());

  SmallVector<DFBTransactionRun> normalized;
  std::uint64_t previous = 0;
  for (std::uint64_t boundary : boundaries) {
    std::uint64_t tiles = boundary - previous;
    if (tiles == 0 || tiles > static_cast<std::uint64_t>(
                                  std::numeric_limits<int64_t>::max())) {
      return failure();
    }
    appendTransactionRun(normalized, 1, static_cast<int64_t>(tiles));
    previous = boundary;
  }
  return normalized;
}

struct CumulativeReleasePoint {
  std::uint64_t position = 0;
  const AccessRun *run = nullptr;
};

static FailureOr<SmallVector<CumulativeReleasePoint>>
collectCumulativeReleasePoints(ArrayRef<const AccessRun *> orderedRuns,
                               DFBProtocolEffectKind releaseKind) {
  SmallVector<CumulativeReleasePoint> releasePoints;
  std::uint64_t position = 0;
  for (const AccessRun *run : orderedRuns) {
    if (!run->access->isProtocolEffect(releaseKind)) {
      continue;
    }
    std::optional<std::uint64_t> next = llvm::checkedAddUnsigned(
        position, static_cast<std::uint64_t>(run->access->numTiles));
    if (!next) {
      return failure();
    }
    position = *next;
    releasePoints.push_back({position, run});
  }
  return releasePoints;
}

static const AccessRun *
findCumulativeRelease(ArrayRef<CumulativeReleasePoint> releasePoints,
                      std::uint64_t requiredPosition) {
  auto releaseIt = llvm::find_if(releasePoints, [&](const auto &release) {
    return release.position >= requiredPosition;
  });
  return releaseIt == releasePoints.end() ? nullptr : releaseIt->run;
}

static bool appendCumulativeEdge(
    SmallVectorImpl<std::pair<unsigned, unsigned>> &edges,
    const AccessRun &release, const AccessRun &acquire, LaunchNodeCoord node,
    const LaunchNodeDomainState &domainState,
    const StructuralOperationOrder &structuralOrder,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  if (!proveEquivalentConditionalRuns(release, acquire, node, domainState)) {
    return false;
  }
  std::optional<AccessEventSpan> releaseEvents =
      getAccessEventSpan(*release.access, operationEvents, accessEvents);
  std::optional<AccessEventSpan> acquireEvents =
      getAccessEventSpan(*acquire.access, operationEvents, accessEvents);
  if (!releaseEvents || !acquireEvents) {
    return false;
  }
  if (releaseEvents->last.completion == acquireEvents->last.completion) {
    return runPrecedesWithinEachIteration(release, acquire, structuralOrder);
  }
  edges.emplace_back(releaseEvents->last.completion,
                     acquireEvents->last.completion);
  return true;
}

static FailureOr<SmallVector<std::pair<unsigned, unsigned>>>
collectCumulativeSynchronizationEdges(
    const CumulativeQueueSide &producer, const CumulativeQueueSide &consumer,
    int64_t physicalTileCount, LaunchNodeCoord node,
    const LaunchNodeDomainState &domainState,
    const StructuralOperationOrder &structuralOrder,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  FailureOr<SmallVector<CumulativeReleasePoint>> published =
      collectCumulativeReleasePoints(producer.orderedRuns,
                                     DFBProtocolEffectKind::Push);
  FailureOr<SmallVector<CumulativeReleasePoint>> consumed =
      collectCumulativeReleasePoints(consumer.orderedRuns,
                                     DFBProtocolEffectKind::Pop);
  if (failed(published) || failed(consumed)) {
    return failure();
  }

  SmallVector<std::pair<unsigned, unsigned>> synchronizationEdges;
  std::uint64_t readPosition = 0;
  for (const AccessRun *run : consumer.orderedRuns) {
    if (run->access->isProtocolEffect(DFBProtocolEffectKind::Pop)) {
      std::optional<std::uint64_t> next = llvm::checkedAddUnsigned(
          readPosition, static_cast<std::uint64_t>(run->access->numTiles));
      if (!next) {
        return failure();
      }
      readPosition = *next;
      continue;
    }
    std::optional<std::uint64_t> required = llvm::checkedAddUnsigned(
        readPosition, static_cast<std::uint64_t>(run->access->numTiles));
    const AccessRun *publishingPush =
        required ? findCumulativeRelease(*published, *required) : nullptr;
    if (!publishingPush ||
        !appendCumulativeEdge(synchronizationEdges, *publishingPush, *run, node,
                              domainState, structuralOrder, operationEvents,
                              accessEvents)) {
      return failure();
    }
  }

  std::uint64_t writePosition = 0;
  std::uint64_t capacity = static_cast<std::uint64_t>(physicalTileCount);
  for (const AccessRun *run : producer.orderedRuns) {
    if (run->access->isProtocolEffect(DFBProtocolEffectKind::Push)) {
      std::optional<std::uint64_t> next = llvm::checkedAddUnsigned(
          writePosition, static_cast<std::uint64_t>(run->access->numTiles));
      if (!next) {
        return failure();
      }
      writePosition = *next;
      continue;
    }
    std::optional<std::uint64_t> reservationEnd = llvm::checkedAddUnsigned(
        writePosition, static_cast<std::uint64_t>(run->access->numTiles));
    if (!reservationEnd) {
      return failure();
    }
    if (*reservationEnd <= capacity) {
      continue;
    }
    const AccessRun *enablingPop =
        findCumulativeRelease(*consumed, *reservationEnd - capacity);
    if (!enablingPop ||
        !appendCumulativeEdge(synchronizationEdges, *enablingPop, *run, node,
                              domainState, structuralOrder, operationEvents,
                              accessEvents)) {
      return failure();
    }
  }
  return synchronizationEdges;
}

static bool
isSingleOpaqueCallQueueScheduleFeasible(const CumulativeQueueSide &producer,
                                        const CumulativeQueueSide &consumer,
                                        std::uint64_t capacity) {
  auto runsBelongToSingleOperation = [](ArrayRef<const AccessRun *> runs) {
    Operation *operation = runs.front()->access->operation;
    return llvm::all_of(runs, [&](const AccessRun *run) {
      return run->access->operation == operation;
    });
  };
  if (!runsBelongToSingleOperation(producer.orderedRuns) ||
      !runsBelongToSingleOperation(consumer.orderedRuns)) {
    return false;
  }

  Operation *producerOperation =
      producer.orderedRuns.front()->access->operation;
  Operation *consumerOperation =
      consumer.orderedRuns.front()->access->operation;
  bool hasRepeatedEffects =
      hasRepeatedOpaqueEffectOperation(producer.orderedRuns) ||
      hasRepeatedOpaqueEffectOperation(consumer.orderedRuns);
  if (producerOperation == consumerOperation || !hasRepeatedEffects) {
    return false;
  }

  std::size_t producerIndex = 0;
  std::size_t consumerIndex = 0;
  std::uint64_t occupiedTiles = 0;

  auto tryAdvance = [&](ArrayRef<const AccessRun *> runs,
                        std::size_t &runIndex) {
    if (runIndex == runs.size()) {
      return false;
    }
    const DFBAccessOccurrence &access = *runs[runIndex]->access;
    std::uint64_t tiles = static_cast<std::uint64_t>(access.numTiles);
    switch (*access.getProtocolEffect()) {
    case DFBProtocolEffectKind::Reserve:
      if (tiles > capacity - occupiedTiles) {
        return false;
      }
      break;
    case DFBProtocolEffectKind::Push:
      if (tiles > capacity - occupiedTiles) {
        return false;
      }
      occupiedTiles += tiles;
      break;
    case DFBProtocolEffectKind::Wait:
      if (tiles > occupiedTiles) {
        return false;
      }
      break;
    case DFBProtocolEffectKind::Pop:
      if (tiles > occupiedTiles) {
        return false;
      }
      occupiedTiles -= tiles;
      break;
    }
    ++runIndex;
    return true;
  };

  // Producer progress only adds occupancy and consumer progress only removes
  // it, so advancing either enabled side cannot disable the other side.
  while (producerIndex != producer.orderedRuns.size() ||
         consumerIndex != consumer.orderedRuns.size()) {
    bool producerAdvanced = tryAdvance(producer.orderedRuns, producerIndex);
    bool consumerAdvanced = tryAdvance(consumer.orderedRuns, consumerIndex);
    if (!producerAdvanced && !consumerAdvanced) {
      return false;
    }
  }
  return occupiedTiles == 0;
}

static bool tryAddCumulativeQueueEdges(
    const DFBLogicalLifecycle &logicalDFB, HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, const AccessRuns &accessRuns,
    LaunchNodeCoord node, const LaunchNodeDomainState &domainState,
    const StructuralOperationOrder &structuralOrder,
    bool includeUnknownDomains) {
  SmallVector<const AccessRun *> producerRuns;
  SmallVector<const AccessRun *> consumerRuns;
  for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
    if (!access.getProtocolEffect() ||
        !mayAccessLaunchNode(access, node, executionCounts,
                             includeUnknownDomains)) {
      continue;
    }
    bool producerEffect =
        access.isProtocolEffect(DFBProtocolEffectKind::Reserve) ||
        access.isProtocolEffect(DFBProtocolEffectKind::Push);
    bool consumerEffect =
        access.isProtocolEffect(DFBProtocolEffectKind::Wait) ||
        access.isProtocolEffect(DFBProtocolEffectKind::Pop);
    if (!producerEffect && !consumerEffect) {
      continue;
    }
    auto runIt = accessRuns.find(&access);
    if (runIt == accessRuns.end() || runIt->second.executionCount != 1 ||
        !isa<OpaqueCallOp>(access.operation)) {
      return false;
    }
    (producerEffect ? producerRuns : consumerRuns).push_back(&runIt->second);
  }
  if (producerRuns.empty() || consumerRuns.empty()) {
    return false;
  }

  int64_t physicalTileCount =
      cast<CircularBufferType>(logicalDFB.type).getTotalElements();
  if (physicalTileCount <= 0) {
    return false;
  }
  CumulativeQueueSideResult producer = proveCumulativeQueueSide(
      producerRuns, DFBProtocolEffectKind::Reserve, DFBProtocolEffectKind::Push,
      physicalTileCount, node, graph, structuralOrder, operationEvents,
      accessEvents);
  CumulativeQueueSideResult consumer = proveCumulativeQueueSide(
      consumerRuns, DFBProtocolEffectKind::Wait, DFBProtocolEffectKind::Pop,
      physicalTileCount, node, graph, structuralOrder, operationEvents,
      accessEvents);
  if (!producer.side || !consumer.side ||
      failed(normalizeCumulativeTransactions(producer.side->cursorRuns,
                                             consumer.side->cursorRuns))) {
    return false;
  }

  FailureOr<SmallVector<std::pair<unsigned, unsigned>>> synchronizationEdges =
      collectCumulativeSynchronizationEdges(
          *producer.side, *consumer.side, physicalTileCount, node, domainState,
          structuralOrder, operationEvents, accessEvents);
  if (failed(synchronizationEdges)) {
    return false;
  }

  bool scheduleFeasible = isSingleOpaqueCallQueueScheduleFeasible(
      *producer.side, *consumer.side,
      static_cast<std::uint64_t>(physicalTileCount));
  if (!graph.tryAddEdgesAndUpdateReachability(*synchronizationEdges,
                                              scheduleFeasible)) {
    return false;
  }
  return true;
}

} // namespace

FailureOr<std::uint64_t>
advanceDFBTransactionCursor(ArrayRef<DFBTransactionRun> transactionRuns,
                            std::uint64_t physicalTileCount,
                            std::uint64_t pointerOffset) {
  if (physicalTileCount == 0 || pointerOffset >= physicalTileCount) {
    return failure();
  }
  for (const DFBTransactionRun &run : transactionRuns) {
    if (run.executionCount == 0 || run.tilesPerExecution <= 0 ||
        static_cast<std::uint64_t>(run.tilesPerExecution) > physicalTileCount) {
      return failure();
    }
    std::uint64_t tilesPerExecution = run.tilesPerExecution;
    std::uint64_t remainingTiles = physicalTileCount - pointerOffset;
    std::uint64_t executionsBeforeBoundary = remainingTiles / tilesPerExecution;
    if (remainingTiles % tilesPerExecution != 0) {
      if (run.executionCount > executionsBeforeBoundary) {
        return failure();
      }
      pointerOffset += run.executionCount * tilesPerExecution;
      continue;
    }
    if (run.executionCount <= executionsBeforeBoundary) {
      pointerOffset += run.executionCount * tilesPerExecution;
      if (pointerOffset == physicalTileCount) {
        pointerOffset = 0;
      }
      continue;
    }

    std::uint64_t remainingExecutions =
        run.executionCount - executionsBeforeBoundary;
    std::uint64_t executionsPerCycle = physicalTileCount / tilesPerExecution;
    if (physicalTileCount % tilesPerExecution != 0) {
      if (remainingExecutions > executionsPerCycle) {
        return failure();
      }
      pointerOffset = remainingExecutions * tilesPerExecution;
      continue;
    }
    pointerOffset =
        (remainingExecutions % executionsPerCycle) * tilesPerExecution;
  }
  return pointerOffset;
}

namespace {

// Derives exact-domain or possible-domain per-node lifetime facts. Possible
// facts control reuse only after proving conditional boundedness.
static DFBLifecycleCompletionProof computeProtocolLifetime(
    DFBLogicalLifecycle &logicalDFB, LaunchNodeCoord node,
    SmallVectorImpl<DFBPerNodeLifetime> &lifetimes,
    SmallVectorImpl<DFBPerNodeLifetimeDiagnostics> *lifetimeDiagnostics,
    const HappensBeforeGraph &graph,
    const StructuralOperationOrder &structuralOrder,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, const AccessRuns &accessRuns,
    const LaunchNodeDomainState &domainState,
    bool includeUnknownDomains = false,
    ArrayRef<const DFBAccessOccurrence *> selectedAccesses = {},
    bool hasCanonicalResetTerminator = false,
    std::optional<std::uint64_t> expectedSelectedExecutionCount =
        std::nullopt) {
  assert((!expectedSelectedExecutionCount ||
          (*expectedSelectedExecutionCount > 0 && !selectedAccesses.empty())) &&
         "normalized selected accesses require a positive execution count");
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
    if (!selectedAccesses.empty() &&
        !llvm::is_contained(selectedAccesses, &access)) {
      continue;
    }
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
    if (access.getProtocolEffect()) {
      switch (*access.getProtocolEffect()) {
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
    if (!access.getProtocolEffect()) {
      continue;
    }
    switch (*access.getProtocolEffect()) {
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

  auto recordAccessFrontiers = [&]() -> LogicalResult {
    lifetime.earliestEntryEvents = findMinimalEntryEvents(
        activeAccesses, graph, operationEvents, accessEvents);
    SmallVector<const DFBAccessOccurrence *> terminalAccesses =
        findMaximalCompletionAccesses(activeAccesses, graph, operationEvents,
                                      accessEvents);
    if (lifetime.earliestEntryEvents.empty() || terminalAccesses.empty()) {
      return failure();
    }
    recordEntryFrontierEvidence(lifetime, diagnostics, activeAccesses,
                                logicalDFB, operationEvents, accessEvents);
    for (const DFBAccessOccurrence *terminalAccess : terminalAccesses) {
      std::optional<AccessEventSpan> events =
          getAccessEventSpan(*terminalAccess, operationEvents, accessEvents);
      if (!events) {
        return failure();
      }
      if (!llvm::is_contained(lifetime.terminalCompletionEvents,
                              events->last.completion)) {
        lifetime.terminalCompletionEvents.push_back(events->last.completion);
      }
      if (diagnostics) {
        diagnostics->terminalAccessOccurrenceIndices.push_back(
            static_cast<unsigned>(terminalAccess - logicalDFB.accesses.data()));
      }
    }
    return success();
  };

  auto opaqueExternalAccess =
      llvm::find_if(activeAccesses, [](const DFBAccessOccurrence *access) {
        return access->opaqueExternalAccess;
      });
  if (opaqueExternalAccess != activeAccesses.end()) {
    if (!hasCanonicalResetTerminator) {
      return {DFBLifecycleCompletionFailureReason::MissingProtocolEffect,
              (*opaqueExternalAccess)->operation};
    }
    auto unscopedOpaqueAccess =
        llvm::find_if(activeAccesses, [](const DFBAccessOccurrence *access) {
          return !access->getProtocolEffect() &&
                 isa<OpaqueCallOp>(access->operation) &&
                 !access->opaqueExternalAccess;
        });
    if (unscopedOpaqueAccess != activeAccesses.end()) {
      return {DFBLifecycleCompletionFailureReason::MissingProtocolEffect,
              (*unscopedOpaqueAccess)->operation};
    }

    if (failed(recordAccessFrontiers())) {
      return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
              activeAccesses.front()->operation};
    }
    return {};
  }

  bool hasProtocolAccess =
      llvm::any_of(activeAccesses, [](const DFBAccessOccurrence *access) {
        return access->getProtocolEffect();
      });
  if (!hasProtocolAccess) {
    bool inspectionOnly = !activeAccesses.empty() &&
                          llvm::all_of(activeAccesses, [](const auto *access) {
                            return access->isNonTransactionalAccess(
                                DFBNonTransactionalAccessKind::Inspect);
                          });
    if (!inspectionOnly) {
      return {DFBLifecycleCompletionFailureReason::MissingProtocolEffect,
              activeAccesses.empty() ? logicalDFB.declarations.front()
                                     : activeAccesses.front()->operation};
    }
    if (failed(recordAccessFrontiers())) {
      return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
              activeAccesses.front()->operation};
    }
    lifetime.conditionalExecutionProven = includeUnknownDomains;
    lifetime.inspectionOnly = true;
    return {};
  }

  bool resetTerminatedProducer = hasCanonicalResetTerminator && hasReserve &&
                                 hasPush && !hasWait && !hasPop;
  if ((!hasReserve || !hasPush || !hasWait || !hasPop) &&
      !resetTerminatedProducer) {
    return {DFBLifecycleCompletionFailureReason::MissingProtocolEffect,
            activeAccesses.empty() ? logicalDFB.declarations.front()
                                   : activeAccesses.front()->operation};
  }

  if (unsupportedAccess) {
    return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
            unsupportedAccess->operation};
  }
  assert(!reserves.empty() && !pushes.empty() &&
         (resetTerminatedProducer || (!waits.empty() && !pops.empty())) &&
         "supported protocol effects must have access runs");

  SmallVector<const AccessRun *> conditionalRuns;
  for (const DFBAccessOccurrence *activeAccess : activeAccesses) {
    const AccessRun &run = accessRuns.at(activeAccess);
    if (run.conditionalExecution) {
      conditionalRuns.push_back(&run);
    }
  }
  if (!conditionalRuns.empty()) {
    const AccessRun &reference = *conditionalRuns.front();
    bool sameCondition = conditionalRuns.size() == activeAccesses.size() &&
                         llvm::all_of(llvm::drop_begin(conditionalRuns),
                                      [&](const AccessRun *run) {
                                        return proveEquivalentConditionalRuns(
                                            reference, *run, node, domainState);
                                      });
    if (!sameCondition) {
      return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
              reference.access->operation};
    }
    lifetime.conditionalExecutionProven = true;
  }
  auto getIntervalExecutionCount = [&](const AccessRun &run) {
    if (expectedSelectedExecutionCount) {
      assert(run.executionCount == *expectedSelectedExecutionCount &&
             "selected repeated-reset accesses must execute once per interval");
      return std::uint64_t{1};
    }
    return run.executionCount;
  };
  auto getTransactionCount = [&](ArrayRef<const AccessRun *> runs) {
    std::optional<std::uint64_t> total = 0;
    for (const AccessRun *run : runs) {
      total = llvm::checkedAddUnsigned(*total, getIntervalExecutionCount(*run));
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
  bool countsMatch = reserveCount && pushCount && *reserveCount == *pushCount;
  if (!resetTerminatedProducer && countsMatch) {
    countsMatch = waitCount && popCount && *reserveCount == *waitCount &&
                  *reserveCount == *popCount;
  }
  int64_t physicalTileCount =
      cast<CircularBufferType>(logicalDFB.type).getTotalElements();
  if (physicalTileCount <= 0) {
    return {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
            reserves.front()->access->operation};
  }
  bool alignedTransactions =
      proveAlignedAcquireReleaseRuns(reserves, pushes, graph, structuralOrder,
                                     operationEvents, accessEvents) &&
      (resetTerminatedProducer ||
       proveAlignedAcquireReleaseRuns(waits, pops, graph, structuralOrder,
                                      operationEvents, accessEvents));
  SmallVector<const AccessRun *> producerProtocolRuns;
  llvm::append_range(producerProtocolRuns, reserves);
  llvm::append_range(producerProtocolRuns, pushes);
  SmallVector<const AccessRun *> consumerProtocolRuns;
  llvm::append_range(consumerProtocolRuns, waits);
  llvm::append_range(consumerProtocolRuns, pops);
  bool useCumulativeQueueProof =
      !expectedSelectedExecutionCount && !resetTerminatedProducer &&
      supportsCumulativeQueueProof(producerProtocolRuns) &&
      supportsCumulativeQueueProof(consumerProtocolRuns);
  if (!countsMatch && !useCumulativeQueueProof) {
    return {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
            activeAccesses.front()->operation};
  }
  if (!alignedTransactions && !useCumulativeQueueProof) {
    return {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
            activeAccesses.front()->operation};
  }

  SmallVector<std::pair<const AccessRun *, const AccessRun *>>
      producerIntervals;
  SmallVector<std::pair<const AccessRun *, const AccessRun *>>
      consumerIntervals;
  const DFBAccessOccurrence *cumulativeTerminalAccess = nullptr;
  std::optional<DFBPointerOwner> writeOwner;
  std::optional<DFBPointerOwner> readOwner;
  if (useCumulativeQueueProof) {
    CumulativeQueueSideResult producer = proveCumulativeQueueSide(
        producerProtocolRuns, DFBProtocolEffectKind::Reserve,
        DFBProtocolEffectKind::Push, physicalTileCount, node, graph,
        structuralOrder, operationEvents, accessEvents);
    if (!producer.side) {
      return producer.failure;
    }
    CumulativeQueueSideResult consumer = proveCumulativeQueueSide(
        consumerProtocolRuns, DFBProtocolEffectKind::Wait,
        DFBProtocolEffectKind::Pop, physicalTileCount, node, graph,
        structuralOrder, operationEvents, accessEvents);
    if (!consumer.side) {
      return consumer.failure;
    }
    FailureOr<SmallVector<DFBTransactionRun>> normalized =
        normalizeCumulativeTransactions(producer.side->cursorRuns,
                                        consumer.side->cursorRuns);
    if (failed(normalized) ||
        failed(advanceDFBTransactionCursor(
            producer.side->cursorRuns,
            static_cast<std::uint64_t>(physicalTileCount))) ||
        failed(advanceDFBTransactionCursor(
            consumer.side->cursorRuns,
            static_cast<std::uint64_t>(physicalTileCount)))) {
      return {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
              activeAccesses.front()->operation};
    }
    FailureOr<SmallVector<std::pair<unsigned, unsigned>>> synchronizationEdges =
        collectCumulativeSynchronizationEdges(
            *producer.side, *consumer.side, physicalTileCount, node,
            domainState, structuralOrder, operationEvents, accessEvents);
    if (failed(synchronizationEdges) ||
        !llvm::all_of(*synchronizationEdges, [&](auto edge) {
          return graph.strictlyPrecedes(edge.first, edge.second);
        })) {
      return {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
              activeAccesses.front()->operation};
    }
    lifetime.transactionRuns = std::move(*normalized);
    lifetime.writeCursorRuns = producer.side->cursorRuns;
    lifetime.readCursorRuns = consumer.side->cursorRuns;
    producerIntervals = producer.side->intervals;
    consumerIntervals = consumer.side->intervals;
    writeOwner = producer.side->owner;
    readOwner = consumer.side->owner;
    cumulativeTerminalAccess = consumerIntervals.back().second->access;
  }

  if (!useCumulativeQueueProof) {
    SmallVector<Operation *> nativeReserves;
    SmallVector<Operation *> nativeWaits;
    // Acquire ownership ends at the next same-kind acquire in the complete
    // kernel. Restricting these boundaries to one lifecycle epoch makes an
    // earlier acquisition incorrectly own direct DFB uses in subsequent epochs.
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      if (!mayContainLaunchNode(access.launchDomain, node,
                                includeUnknownDomains)) {
        continue;
      }
      auto executionCountIt = executionCounts.find(&access);
      assert(executionCountIt != executionCounts.end() &&
             "every DFB access must have an execution-count fact");
      if (executionCountIt->second && *executionCountIt->second == 0) {
        continue;
      }
      if (access.isProtocolEffect(DFBProtocolEffectKind::Reserve) &&
          isa<CBReserveOp>(access.operation)) {
        nativeReserves.push_back(access.operation);
      } else if (access.isProtocolEffect(DFBProtocolEffectKind::Wait) &&
                 isa<CBWaitOp>(access.operation)) {
        nativeWaits.push_back(access.operation);
      }
    }
    for (auto [reserve, push] : llvm::zip_equal(reserves, pushes)) {
      if (reserve->access->numTiles <= 0) {
        return {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
                reserve->access->operation};
      }
      if (isa<CBReserveOp>(reserve->access->operation) &&
          !releaseFollowsOwnedUses(reserve->access->operation,
                                   push->access->operation, nativeReserves,
                                   structuralOrder)) {
        return {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
                push->access->operation};
      }
      std::optional<DFBPointerOwner> reserveOwner = getPointerOwner(
          reserve->access->operation, node, DFBProtocolEffectKind::Reserve);
      std::optional<DFBPointerOwner> pushOwner = getPointerOwner(
          push->access->operation, node, DFBProtocolEffectKind::Push);
      if (!reserveOwner || !pushOwner || *reserveOwner != *pushOwner ||
          (writeOwner && *writeOwner != *reserveOwner)) {
        return {DFBLifecycleCompletionFailureReason::UnknownPointerOwner,
                reserve->access->operation};
      }
      writeOwner = reserveOwner;
    }
    for (auto [wait, pop] : llvm::zip_equal(waits, pops)) {
      if (wait->access->numTiles <= 0) {
        return {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
                wait->access->operation};
      }
      if (isa<CBWaitOp>(wait->access->operation) &&
          !releaseFollowsOwnedUses(wait->access->operation,
                                   pop->access->operation, nativeWaits,
                                   structuralOrder)) {
        return {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
                pop->access->operation};
      }
      std::optional<DFBPointerOwner> waitOwner = getPointerOwner(
          wait->access->operation, node, DFBProtocolEffectKind::Wait);
      std::optional<DFBPointerOwner> popOwner = getPointerOwner(
          pop->access->operation, node, DFBProtocolEffectKind::Pop);
      if (!waitOwner || !popOwner || *waitOwner != *popOwner ||
          (readOwner && *readOwner != *waitOwner)) {
        return {DFBLifecycleCompletionFailureReason::UnknownPointerOwner,
                wait->access->operation};
      }
      readOwner = waitOwner;
    }

    if (resetTerminatedProducer) {
      std::uint64_t occupiedTiles = 0;
      for (const AccessRun *reserve : reserves) {
        std::uint64_t intervalCount = getIntervalExecutionCount(*reserve);
        std::optional<std::uint64_t> runTiles = llvm::checkedMulUnsigned(
            intervalCount,
            static_cast<std::uint64_t>(reserve->access->numTiles));
        if (!runTiles) {
          return {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
                  reserve->access->operation};
        }
        std::optional<std::uint64_t> updatedTiles =
            llvm::checkedAddUnsigned(occupiedTiles, *runTiles);
        if (!updatedTiles ||
            *updatedTiles > static_cast<std::uint64_t>(physicalTileCount)) {
          return {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
                  reserve->access->operation};
        }
        occupiedTiles = *updatedTiles;
        appendTransactionRun(lifetime.transactionRuns, intervalCount,
                             reserve->access->numTiles);
      }
    } else {
      std::size_t reserveIndex = 0;
      std::size_t waitIndex = 0;
      std::uint64_t reserveOffset = 0;
      std::uint64_t waitOffset = 0;
      while (reserveIndex < reserves.size() && waitIndex < waits.size()) {
        const AccessRun &reserve = *reserves[reserveIndex];
        const AccessRun &wait = *waits[waitIndex];
        // An ordinary terminal lifecycle must return ring pointers to their
        // initial offsets. A synchronized reset establishes that state
        // directly.
        if (reserve.access->numTiles != wait.access->numTiles ||
            reserve.access->numTiles <= 0 ||
            reserve.access->numTiles > physicalTileCount ||
            (!hasCanonicalResetTerminator &&
             physicalTileCount % reserve.access->numTiles != 0)) {
          return {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
                  reserve.access->operation};
        }
        std::uint64_t reserveIntervalCount = getIntervalExecutionCount(reserve);
        std::uint64_t waitIntervalCount = getIntervalExecutionCount(wait);
        std::uint64_t matchedCount =
            std::min(reserveIntervalCount - reserveOffset,
                     waitIntervalCount - waitOffset);
        appendTransactionRun(lifetime.transactionRuns, matchedCount,
                             reserve.access->numTiles);
        reserveOffset += matchedCount;
        waitOffset += matchedCount;
        if (reserveOffset == reserveIntervalCount) {
          ++reserveIndex;
          reserveOffset = 0;
        }
        if (waitOffset == waitIntervalCount) {
          ++waitIndex;
          waitOffset = 0;
        }
      }
      if (reserveIndex != reserves.size() || waitIndex != waits.size()) {
        return {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
                reserves.front()->access->operation};
      }
    }
    if (hasCanonicalResetTerminator &&
        failed(advanceDFBTransactionCursor(
            lifetime.transactionRuns,
            static_cast<std::uint64_t>(physicalTileCount)))) {
      return {DFBLifecycleCompletionFailureReason::MismatchedTransaction,
              reserves.front()->access->operation};
    }
  }
  lifetime.writePointerOwner = writeOwner;
  lifetime.readPointerOwner = readOwner;
  if (!useCumulativeQueueProof) {
    for (auto [reserve, push] : llvm::zip_equal(reserves, pushes)) {
      producerIntervals.emplace_back(reserve, push);
    }
    for (auto [wait, pop] : llvm::zip_equal(waits, pops)) {
      consumerIntervals.emplace_back(wait, pop);
    }
  }

  for (const DFBAccessOccurrence *activeAccess : activeAccesses) {
    if (activeAccess->getProtocolEffect()) {
      continue;
    }
    const AccessRun &use = accessRuns.at(activeAccess);
    bool covered = false;
    for (auto [reserve, push] : producerIntervals) {
      covered |=
          runIsInsideInterval(use, *reserve, *push, graph, structuralOrder,
                              operationEvents, accessEvents);
    }
    for (auto [wait, pop] : consumerIntervals) {
      covered |= runIsInsideInterval(use, *wait, *pop, graph, structuralOrder,
                                     operationEvents, accessEvents);
    }
    if (!covered) {
      return {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
              activeAccess->operation};
    }
  }

  const DFBAccessOccurrence *terminalAccess =
      resetTerminatedProducer
          ? pushes.back()->access
          : (cumulativeTerminalAccess ? cumulativeTerminalAccess
                                      : pops.back()->access);
  std::optional<AccessEventSpan> terminalEvents =
      getAccessEventSpan(*terminalAccess, operationEvents, accessEvents);
  if (!terminalEvents) {
    return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
            terminalAccess->operation};
  }
  for (const DFBAccessOccurrence *activeAccess : activeAccesses) {
    std::optional<AccessEventSpan> useEvents =
        getAccessEventSpan(*activeAccess, operationEvents, accessEvents);
    if (!useEvents ||
        (useEvents->last.completion != terminalEvents->last.completion &&
         !graph.strictlyPrecedes(useEvents->last.completion,
                                 terminalEvents->last.completion))) {
      return {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
              activeAccess->operation};
    }
  }
  lifetime.earliestEntryEvents = findMinimalEntryEvents(
      activeAccesses, graph, operationEvents, accessEvents);
  lifetime.terminalCompletionEvents = {terminalEvents->last.completion};
  recordEntryFrontierEvidence(lifetime, diagnostics, activeAccesses, logicalDFB,
                              operationEvents, accessEvents);
  if (diagnostics) {
    diagnostics->terminalAccessOccurrenceIndices = {
        static_cast<unsigned>(terminalAccess - logicalDFB.accesses.data())};
  }
  if (lifetime.earliestEntryEvents.empty()) {
    return {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
            terminalAccess->operation};
  }
  lifetime.terminalTransactionRuns.assign(lifetime.transactionRuns.begin(),
                                          lifetime.transactionRuns.end());
  if (!useCumulativeQueueProof) {
    lifetime.writeCursorRuns.assign(lifetime.transactionRuns.begin(),
                                    lifetime.transactionRuns.end());
    if (!resetTerminatedProducer) {
      lifetime.readCursorRuns.assign(lifetime.transactionRuns.begin(),
                                     lifetime.transactionRuns.end());
    }
  }
  lifetime.terminalWriteCursorRuns.assign(lifetime.writeCursorRuns.begin(),
                                          lifetime.writeCursorRuns.end());
  lifetime.terminalReadCursorRuns.assign(lifetime.readCursorRuns.begin(),
                                         lifetime.readCursorRuns.end());
  lifetime.terminalWritePointerOwner = lifetime.writePointerOwner;
  lifetime.terminalReadPointerOwner = lifetime.readPointerOwner;
  return {};
}

static std::optional<std::pair<Operation *, const StaticIterationDomain *>>
getParticipantIteration(const ValidatedSynchronizedReset &reset,
                        Operation *accessOperation) {
  func::FuncOp accessFunction =
      accessOperation->getParentOfType<func::FuncOp>();
  if (!accessFunction) {
    return std::nullopt;
  }
  for (auto [participant, iterationDomain] : llvm::zip_equal(
           reset.participantOperations, reset.participantIterationDomains)) {
    if (participant &&
        participant->getParentOfType<func::FuncOp>() == accessFunction) {
      return std::make_pair(participant, &iterationDomain);
    }
  }
  return std::nullopt;
}

// Models one representative interval when every access and its terminating
// collective reset execute once per iteration of the same sequential loop.
// The reset restores canonical cursor state, so physical allocation validates
// the per-iteration transactions rather than their dispatch-wide sum.
static std::optional<DFBLifecycleCompletionProof>
tryComputeRepeatedResetLifetime(
    DFBLogicalLifecycle &logicalDFB, unsigned logicalIndex,
    LaunchNodeCoord node, SmallVectorImpl<DFBPerNodeLifetime> &lifetimes,
    SmallVectorImpl<DFBPerNodeLifetimeDiagnostics> *lifetimeDiagnostics,
    ArrayRef<ValidatedSynchronizedReset> synchronizedResets,
    const ResetBoundaryEvents &resetBoundaryEvents,
    ArrayRef<ValidatedDFBReconfiguration> reconfigurations,
    const ReconfigurationBoundaryEvents &reconfigurationBoundaryEvents,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, const AccessRuns &accessRuns,
    const LaunchNodeDomainState &domainState,
    const StructuralOperationOrder &structuralOrder,
    bool includeUnknownDomains) {
  SmallVector<const DFBAccessOccurrence *> activeAccesses;
  for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
    if (!mayContainLaunchNode(access.launchDomain, node,
                              includeUnknownDomains)) {
      continue;
    }
    auto executionCountIt = executionCounts.find(&access);
    assert(executionCountIt != executionCounts.end() &&
           "every DFB access must have an execution-count fact");
    if (executionCountIt->second && *executionCountIt->second == 0) {
      continue;
    }
    activeAccesses.push_back(&access);
  }
  if (activeAccesses.empty()) {
    return std::nullopt;
  }

  SmallVector<const ValidatedSynchronizedReset *> candidates;
  for (const ValidatedSynchronizedReset &reset : synchronizedResets) {
    if (reset.executionCount <= 1 || reset.conditionalExecution ||
        !llvm::is_contained(reset.targetLogicalIndices, logicalIndex)) {
      continue;
    }
    auto resetEventsIt = resetBoundaryEvents.find(reset.reset);
    if (resetEventsIt == resetBoundaryEvents.end()) {
      continue;
    }
    bool terminatesEveryAccess =
        llvm::all_of(activeAccesses, [&](const DFBAccessOccurrence *access) {
          auto runIt = accessRuns.find(access);
          std::optional<std::pair<Operation *, const StaticIterationDomain *>>
              participant = getParticipantIteration(reset, access->operation);
          std::optional<AccessEventSpan> events =
              getAccessEventSpan(*access, operationEvents, accessEvents);
          if (runIt == accessRuns.end() || !participant || !events ||
              runIt->second.conditionalExecution ||
              runIt->second.executionCount != reset.executionCount ||
              !(runIt->second.iterationDomain == *participant->second) ||
              !structuralOrder.precedes(access->operation,
                                        participant->first)) {
            return false;
          }
          const AccessEventSpan &resetEvents = resetEventsIt->second;
          return graph.strictlyPrecedes(events->first.completion,
                                        resetEvents.first.entry) &&
                 graph.strictlyPrecedes(events->last.completion,
                                        resetEvents.last.entry);
        });
    if (terminatesEveryAccess) {
      candidates.push_back(&reset);
    }
  }
  if (candidates.empty()) {
    return std::nullopt;
  }

  const ValidatedSynchronizedReset *terminator = nullptr;
  for (const ValidatedSynchronizedReset *candidate : candidates) {
    const AccessEventSpan &candidateEvents =
        resetBoundaryEvents.at(candidate->reset);
    bool precedesEveryOtherCandidate =
        llvm::all_of(candidates, [&](const ValidatedSynchronizedReset *other) {
          if (candidate == other) {
            return true;
          }
          const AccessEventSpan &otherEvents =
              resetBoundaryEvents.at(other->reset);
          return graph.strictlyPrecedes(candidateEvents.first.completion,
                                        otherEvents.first.entry) &&
                 graph.strictlyPrecedes(candidateEvents.last.completion,
                                        otherEvents.last.entry);
        });
    if (precedesEveryOtherCandidate) {
      terminator = candidate;
      break;
    }
  }
  if (!terminator) {
    return std::nullopt;
  }

  const AccessEventSpan &terminatorEvents =
      resetBoundaryEvents.at(terminator->reset);
  SmallVector<AccessEventSpan> activeEventSpans;
  activeEventSpans.reserve(activeAccesses.size());
  for (const DFBAccessOccurrence *access : activeAccesses) {
    std::optional<AccessEventSpan> events =
        getAccessEventSpan(*access, operationEvents, accessEvents);
    if (!events) {
      return std::nullopt;
    }
    activeEventSpans.push_back(*events);
  }
  auto boundaryPrecedesRepeatedInterval = [&](const EventPair &boundary) {
    return graph.strictlyPrecedes(boundary.completion,
                                  terminatorEvents.first.entry) &&
           llvm::all_of(activeEventSpans, [&](const AccessEventSpan &events) {
             return graph.strictlyPrecedes(boundary.completion,
                                           events.first.entry);
           });
  };
  auto boundaryFollowsRepeatedInterval = [&](const EventPair &boundary) {
    return graph.strictlyPrecedes(terminatorEvents.last.completion,
                                  boundary.entry) &&
           llvm::all_of(activeEventSpans, [&](const AccessEventSpan &events) {
             return graph.strictlyPrecedes(events.last.completion,
                                           boundary.entry);
           });
  };

  for (const ValidatedSynchronizedReset &reset : synchronizedResets) {
    if (!reset.isModeledLifetimeBoundary() ||
        !llvm::is_contained(reset.targetLogicalIndices, logicalIndex)) {
      continue;
    }
    auto eventsIt = resetBoundaryEvents.find(reset.reset);
    if (eventsIt == resetBoundaryEvents.end() ||
        (!boundaryPrecedesRepeatedInterval(eventsIt->second.first) &&
         !boundaryFollowsRepeatedInterval(eventsIt->second.last))) {
      return std::nullopt;
    }
  }

  const ValidatedDFBReconfiguration *entryReconfiguration = nullptr;
  std::optional<EventPair> entryReconfigurationEvents;
  for (const ValidatedDFBReconfiguration &reconfiguration : reconfigurations) {
    auto eventsIt =
        reconfigurationBoundaryEvents.find(reconfiguration.boundary);
    if (eventsIt == reconfigurationBoundaryEvents.end()) {
      return std::nullopt;
    }
    bool precedes = boundaryPrecedesRepeatedInterval(eventsIt->second);
    bool follows = boundaryFollowsRepeatedInterval(eventsIt->second);
    if (precedes == follows) {
      return std::nullopt;
    }
    if (precedes &&
        (!entryReconfigurationEvents ||
         graph.strictlyPrecedes(entryReconfigurationEvents->completion,
                                eventsIt->second.entry))) {
      entryReconfiguration = &reconfiguration;
      entryReconfigurationEvents = eventsIt->second;
    }
  }
  if (entryReconfiguration && entryReconfiguration->conditionalExecution) {
    for (const DFBAccessOccurrence *access : activeAccesses) {
      auto runIt = accessRuns.find(access);
      if (runIt == accessRuns.end() || !runIt->second.conditionalExecution ||
          !proveEquivalentConditionalExecutionAtLaunchNodes(
              access->operation, node,
              entryReconfiguration->participantOperations.front(), node,
              domainState)) {
        return std::nullopt;
      }
    }
  }

  SmallVector<DFBPerNodeLifetime, 0> intervalLifetimes;
  SmallVector<DFBPerNodeLifetimeDiagnostics, 0> intervalDiagnostics;
  DFBLifecycleCompletionProof proof = computeProtocolLifetime(
      logicalDFB, node, intervalLifetimes,
      lifetimeDiagnostics ? &intervalDiagnostics : nullptr, graph,
      structuralOrder, operationEvents, accessEvents, executionCounts,
      accessRuns, domainState, includeUnknownDomains, activeAccesses,
      /*hasCanonicalResetTerminator=*/true,
      /*expectedSelectedExecutionCount=*/terminator->executionCount);
  assert(intervalLifetimes.size() == 1 &&
         "one repeated interval must produce one protocol lifetime");
  assert((!lifetimeDiagnostics || intervalDiagnostics.size() == 1) &&
         "one repeated interval must produce one diagnostic lifetime");
  lifetimes.push_back(std::move(intervalLifetimes.front()));
  if (lifetimeDiagnostics) {
    lifetimeDiagnostics->push_back(std::move(intervalDiagnostics.front()));
  }
  DFBPerNodeLifetime &lifetime = lifetimes.back();
  lifetime.completionProof = proof;
  if (!proof.proven()) {
    return proof;
  }

  lifetime.terminalCompletionEvents = {terminatorEvents.last.completion};
  lifetime.terminalStateCanonical = true;

  DFBLifecycleEpoch epoch;
  epoch.executionCount = terminator->executionCount;
  for (const DFBAccessOccurrence *access : activeAccesses) {
    epoch.accessOccurrenceIndices.push_back(
        static_cast<unsigned>(access - logicalDFB.accesses.data()));
  }
  epoch.earliestEntryEvents = lifetime.earliestEntryEvents;
  epoch.terminalCompletionEvents = lifetime.terminalCompletionEvents;
  epoch.transactionRuns = lifetime.transactionRuns;
  epoch.writeCursorRuns = lifetime.writeCursorRuns;
  epoch.readCursorRuns = lifetime.readCursorRuns;
  epoch.writePointerOwner = lifetime.writePointerOwner;
  epoch.readPointerOwner = lifetime.readPointerOwner;
  epoch.terminalWritePointerOwner = lifetime.terminalWritePointerOwner;
  epoch.terminalReadPointerOwner = lifetime.terminalReadPointerOwner;
  if (entryReconfiguration) {
    epoch.entryReconfigurationOrdinal =
        entryReconfiguration->boundary.getOrdinal();
  }
  epoch.terminalResetOrdinal = terminator->reset.getOrdinal();
  epoch.inspectionOnly = lifetime.inspectionOnly;
  epoch.terminalStateCanonical = true;
  epoch.completionProof = proof;
  lifetime.epochs.push_back(std::move(epoch));
  return proof;
}

struct OrderedLifecycleBoundary {
  const ValidatedSynchronizedReset *reset = nullptr;
  const ValidatedDFBReconfiguration *reconfiguration = nullptr;
  EventPair events;

  bool isConditional() const {
    return reset ? reset->conditionalExecution
                 : reconfiguration->conditionalExecution;
  }

  Operation *getEvidenceOperation() const {
    return reset ? reset->participantOperations.front()
                 : reconfiguration->participantOperations.front();
  }
};

static Operation *findConditionalExecutionMismatch(
    const OrderedLifecycleBoundary &boundary,
    ArrayRef<const DFBAccessOccurrence *> accesses, LaunchNodeCoord node,
    const AccessRuns &accessRuns, const LaunchNodeDomainState &domainState) {
  if (!boundary.isConditional()) {
    return nullptr;
  }
  for (const DFBAccessOccurrence *access : accesses) {
    auto runIt = accessRuns.find(access);
    if (runIt == accessRuns.end() || !runIt->second.conditionalExecution ||
        !proveEquivalentConditionalExecutionAtLaunchNodes(
            access->operation, node, boundary.getEvidenceOperation(), node,
            domainState)) {
      return access->operation;
    }
  }
  return nullptr;
}

// Proves complete protocol intervals between lifecycle boundaries. A reset may
// discard unread blocks. An incomplete protocol crosses reconfiguration
// unchanged and remains active in every configuration epoch that it spans.
static DFBLifecycleCompletionProof computePerNodeLifetime(
    DFBLogicalLifecycle &logicalDFB, unsigned logicalIndex,
    LaunchNodeCoord node, SmallVectorImpl<DFBPerNodeLifetime> &lifetimes,
    SmallVectorImpl<DFBPerNodeLifetimeDiagnostics> *lifetimeDiagnostics,
    ArrayRef<ValidatedSynchronizedReset> synchronizedResets,
    const ResetBoundaryEvents &resetBoundaryEvents,
    ArrayRef<ValidatedDFBReconfiguration> reconfigurations,
    const ReconfigurationBoundaryEvents &reconfigurationBoundaryEvents,
    const HappensBeforeGraph &graph,
    const StructuralOperationOrder &structuralOrder,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, const AccessRuns &accessRuns,
    const LaunchNodeDomainState &domainState,
    bool includeUnknownDomains = false) {
  if (std::optional<DFBLifecycleCompletionProof> repeatedResetProof =
          tryComputeRepeatedResetLifetime(
              logicalDFB, logicalIndex, node, lifetimes, lifetimeDiagnostics,
              synchronizedResets, resetBoundaryEvents, reconfigurations,
              reconfigurationBoundaryEvents, graph, operationEvents,
              accessEvents, executionCounts, accessRuns, domainState,
              structuralOrder, includeUnknownDomains)) {
    return *repeatedResetProof;
  }

  SmallVector<OrderedLifecycleBoundary> boundaries;
  for (const ValidatedSynchronizedReset &reset : synchronizedResets) {
    if (!reset.isModeledLifetimeBoundary()) {
      continue;
    }
    if (!llvm::is_contained(reset.targetLogicalIndices, logicalIndex)) {
      continue;
    }
    auto eventsIt = resetBoundaryEvents.find(reset.reset);
    if (eventsIt == resetBoundaryEvents.end()) {
      DFBPerNodeLifetime &lifetime = lifetimes.emplace_back();
      lifetime.node = node;
      if (lifetimeDiagnostics) {
        lifetimeDiagnostics->emplace_back();
      }
      return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
              reset.participantOperations.front()};
    }
    assert(eventsIt->second.first.entry == eventsIt->second.last.entry &&
           "single reset instance must use one boundary event");
    boundaries.push_back({&reset, nullptr, eventsIt->second.first});
  }
  for (const ValidatedDFBReconfiguration &reconfiguration : reconfigurations) {
    auto eventsIt =
        reconfigurationBoundaryEvents.find(reconfiguration.boundary);
    if (eventsIt == reconfigurationBoundaryEvents.end()) {
      DFBPerNodeLifetime &lifetime = lifetimes.emplace_back();
      lifetime.node = node;
      if (lifetimeDiagnostics) {
        lifetimeDiagnostics->emplace_back();
      }
      return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
              reconfiguration.participantOperations.front()};
    }
    boundaries.push_back({nullptr, &reconfiguration, eventsIt->second});
  }
  if (boundaries.empty()) {
    return computeProtocolLifetime(
        logicalDFB, node, lifetimes, lifetimeDiagnostics, graph,
        structuralOrder, operationEvents, accessEvents, executionCounts,
        accessRuns, domainState, includeUnknownDomains);
  }

  for (auto [lhsIndex, lhs] : llvm::enumerate(boundaries)) {
    for (auto [rhsIndex, rhs] : llvm::enumerate(boundaries)) {
      if (lhsIndex >= rhsIndex) {
        continue;
      }
      bool lhsBeforeRhs =
          graph.strictlyPrecedes(lhs.events.completion, rhs.events.entry);
      bool rhsBeforeLhs =
          graph.strictlyPrecedes(rhs.events.completion, lhs.events.entry);
      if (!lhsBeforeRhs && !rhsBeforeLhs) {
        DFBPerNodeLifetime &lifetime = lifetimes.emplace_back();
        lifetime.node = node;
        if (lifetimeDiagnostics) {
          lifetimeDiagnostics->emplace_back();
        }
        return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
                rhs.getEvidenceOperation()};
      }
    }
  }
  llvm::sort(boundaries, [&](const OrderedLifecycleBoundary &lhs,
                             const OrderedLifecycleBoundary &rhs) {
    return graph.strictlyPrecedes(lhs.events.completion, rhs.events.entry);
  });

  DFBPerNodeLifetime &lifetime = lifetimes.emplace_back();
  lifetime.node = node;
  DFBPerNodeLifetimeDiagnostics *diagnostics = nullptr;
  if (lifetimeDiagnostics) {
    diagnostics = &lifetimeDiagnostics->emplace_back();
  }
  SmallVector<SmallVector<const DFBAccessOccurrence *>> epochAccesses(
      boundaries.size() + 1);
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
    std::optional<AccessEventSpan> events =
        getAccessEventSpan(access, operationEvents, accessEvents);
    if (!events) {
      return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
              access.operation};
    }
    unsigned epochIndex = 0;
    for (const OrderedLifecycleBoundary &boundary : boundaries) {
      bool beforeBoundary = graph.strictlyPrecedes(events->last.completion,
                                                   boundary.events.entry);
      bool afterBoundary = graph.strictlyPrecedes(boundary.events.completion,
                                                  events->first.entry);
      if (beforeBoundary == afterBoundary) {
        return {DFBLifecycleCompletionFailureReason::IncompleteUseOrder,
                access.operation};
      }
      epochIndex += afterBoundary;
    }
    epochAccesses[epochIndex].push_back(&access);
  }

  for (const OrderedLifecycleBoundary &boundary : boundaries) {
    if (!boundary.reset) {
      continue;
    }
    Operation *conditionalMismatch = nullptr;
    for (ArrayRef<const DFBAccessOccurrence *> accesses : epochAccesses) {
      conditionalMismatch = findConditionalExecutionMismatch(
          boundary, accesses, node, accessRuns, domainState);
      if (conditionalMismatch) {
        break;
      }
    }
    if (conditionalMismatch) {
      return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
              conditionalMismatch};
    }
  }

  bool hasActiveEpoch = false;
  SmallVector<const DFBAccessOccurrence *> lifecycleAccesses;
  std::optional<unsigned> firstBoundaryInterval;
  for (auto [boundaryInterval, intervalAccesses] :
       llvm::enumerate(epochAccesses)) {
    if (!intervalAccesses.empty() && !firstBoundaryInterval) {
      firstBoundaryInterval = boundaryInterval;
    }
    llvm::append_range(lifecycleAccesses, intervalAccesses);
    if (lifecycleAccesses.empty()) {
      continue;
    }
    const OrderedLifecycleBoundary *terminalBoundary =
        boundaryInterval < boundaries.size() ? &boundaries[boundaryInterval]
                                             : nullptr;
    bool resetTerminated = terminalBoundary && terminalBoundary->reset;
    SmallVector<DFBPerNodeLifetime, 0> epochLifetimes;
    SmallVector<DFBPerNodeLifetimeDiagnostics, 0> epochDiagnostics;
    DFBLifecycleCompletionProof proof = computeProtocolLifetime(
        logicalDFB, node, epochLifetimes,
        diagnostics ? &epochDiagnostics : nullptr, graph, structuralOrder,
        operationEvents, accessEvents, executionCounts, accessRuns, domainState,
        includeUnknownDomains, lifecycleAccesses,
        /*hasCanonicalResetTerminator=*/resetTerminated);
    assert(epochLifetimes.size() == 1 &&
           "one selected epoch must produce one protocol lifetime");
    assert((!diagnostics || epochDiagnostics.size() == 1) &&
           "one selected epoch must produce one diagnostic lifetime");
    DFBPerNodeLifetime &epochLifetime = epochLifetimes.front();
    const DFBPerNodeLifetimeDiagnostics *epochDiagnostic =
        diagnostics ? &epochDiagnostics.front() : nullptr;
    epochLifetime.completionProof = proof;
    if (terminalBoundary && terminalBoundary->reconfiguration &&
        !proof.proven()) {
      continue;
    }
    if (!proof.proven()) {
      return proof;
    }
    DFBLifecycleEpoch epoch;
    for (const DFBAccessOccurrence *access : lifecycleAccesses) {
      epoch.accessOccurrenceIndices.push_back(
          static_cast<unsigned>(access - logicalDFB.accesses.data()));
    }
    epoch.transactionRuns = epochLifetime.transactionRuns;
    epoch.writeCursorRuns = epochLifetime.writeCursorRuns;
    epoch.readCursorRuns = epochLifetime.readCursorRuns;
    epoch.writePointerOwner = epochLifetime.writePointerOwner;
    epoch.readPointerOwner = epochLifetime.readPointerOwner;
    epoch.completionProof = proof;
    epoch.inspectionOnly = epochLifetime.inspectionOnly;
    assert(firstBoundaryInterval &&
           "active lifecycle must have a first boundary interval");
    const OrderedLifecycleBoundary *entryBoundary = nullptr;
    for (unsigned boundaryIndex = 0; boundaryIndex < *firstBoundaryInterval;
         ++boundaryIndex) {
      if (const ValidatedDFBReconfiguration *entryReconfiguration =
              boundaries[boundaryIndex].reconfiguration) {
        epoch.entryReconfigurationOrdinal =
            entryReconfiguration->boundary.getOrdinal();
        entryBoundary = &boundaries[boundaryIndex];
      }
    }
    if (entryBoundary) {
      if (Operation *conditionalMismatch = findConditionalExecutionMismatch(
              *entryBoundary, lifecycleAccesses, node, accessRuns,
              domainState)) {
        return {DFBLifecycleCompletionFailureReason::UnsupportedControlFlow,
                conditionalMismatch};
      }
    }
    epoch.activeConfigurationEpochs.push_back(
        epoch.entryReconfigurationOrdinal);
    for (unsigned boundaryIndex = *firstBoundaryInterval;
         boundaryIndex < boundaryInterval; ++boundaryIndex) {
      assert(!boundaries[boundaryIndex].reset &&
             "a DFB lifecycle cannot cross a synchronized reset");
      epoch.activeConfigurationEpochs.push_back(
          boundaries[boundaryIndex].reconfiguration->boundary.getOrdinal());
    }
    if (terminalBoundary) {
      if (terminalBoundary->reset) {
        if (terminalBoundary->reset->conditionalExecution) {
          epochLifetime.conditionalExecutionProven = true;
        }
        epoch.terminalResetOrdinal =
            terminalBoundary->reset->reset.getOrdinal();
      } else {
        epoch.terminalReconfigurationOrdinal =
            terminalBoundary->reconfiguration->boundary.getOrdinal();
      }
      epoch.terminalStateCanonical = true;
      epochLifetime.terminalCompletionEvents = {
          terminalBoundary->events.completion};
      epochLifetime.terminalStateCanonical = true;
    }
    epoch.earliestEntryEvents = epochLifetime.earliestEntryEvents;
    epoch.terminalCompletionEvents = epochLifetime.terminalCompletionEvents;
    epoch.terminalWritePointerOwner = epochLifetime.terminalWritePointerOwner;
    epoch.terminalReadPointerOwner = epochLifetime.terminalReadPointerOwner;
    lifetime.epochs.push_back(std::move(epoch));

    if (!hasActiveEpoch) {
      lifetime.earliestEntryEvents = epochLifetime.earliestEntryEvents;
      lifetime.entryEvidence = epochLifetime.entryEvidence;
      if (diagnostics) {
        diagnostics->earliestAccessOccurrenceIndices =
            epochDiagnostic->earliestAccessOccurrenceIndices;
      }
      lifetime.transactionRuns = epochLifetime.transactionRuns;
      lifetime.writeCursorRuns = epochLifetime.writeCursorRuns;
      lifetime.readCursorRuns = epochLifetime.readCursorRuns;
      lifetime.writePointerOwner = epochLifetime.writePointerOwner;
      lifetime.readPointerOwner = epochLifetime.readPointerOwner;
      lifetime.inspectionOnly = epochLifetime.inspectionOnly;
      hasActiveEpoch = true;
    }
    lifetime.conditionalExecutionProven |=
        epochLifetime.conditionalExecutionProven;
    lifetime.terminalCompletionEvents = epochLifetime.terminalCompletionEvents;
    if (diagnostics) {
      diagnostics->terminalAccessOccurrenceIndices =
          epochDiagnostic->terminalAccessOccurrenceIndices;
    }
    lifetime.terminalTransactionRuns = epochLifetime.terminalTransactionRuns;
    lifetime.terminalWriteCursorRuns = epochLifetime.terminalWriteCursorRuns;
    lifetime.terminalReadCursorRuns = epochLifetime.terminalReadCursorRuns;
    lifetime.terminalWritePointerOwner =
        epochLifetime.terminalWritePointerOwner;
    lifetime.terminalReadPointerOwner = epochLifetime.terminalReadPointerOwner;
    lifetime.inspectionOnly =
        lifetime.inspectionOnly && epochLifetime.inspectionOnly;
    lifetime.terminalStateCanonical = epochLifetime.terminalStateCanonical;
    lifecycleAccesses.clear();
    firstBoundaryInterval.reset();
  }

  if (!hasActiveEpoch) {
    if (includeUnknownDomains) {
      lifetime.mayBeActive = false;
      return {};
    }
    return {DFBLifecycleCompletionFailureReason::MissingProtocolEffect,
            logicalDFB.declarations.front()};
  }
  return {};
}

// Proves non-overlap only when every possible end of `before` strictly precedes
// every possible start of `after`.
static bool proveOrderedBefore(const DFBPerNodeLifetime &before,
                               const DFBPerNodeLifetime &after,
                               const HappensBeforeGraph &graph) {
  if (!before.completionProof.proven() || !after.completionProof.proven()) {
    return false;
  }
  return llvm::all_of(before.terminalCompletionEvents, [&](unsigned terminal) {
    return llvm::all_of(after.earliestEntryEvents, [&](unsigned earliest) {
      return graph.strictlyPrecedes(terminal, earliest);
    });
  });
}

static SmallVector<llvm::BitVector>
buildLogicalOrdering(ArrayRef<const DFBPerNodeLifetime *> lifetimes,
                     const HappensBeforeGraph &graph) {
  SmallVector<llvm::BitVector> ordering(lifetimes.size(),
                                        llvm::BitVector(lifetimes.size()));
  for (auto [beforeIndex, before] : llvm::enumerate(lifetimes)) {
    if (!before) {
      continue;
    }
    for (auto [afterIndex, after] : llvm::enumerate(lifetimes)) {
      if (after && proveOrderedBefore(*before, *after, graph)) {
        ordering[beforeIndex].set(afterIndex);
      }
    }
  }
  return ordering;
}

static bool intervalIsOutsideReset(ArrayRef<unsigned> earliestEntryEvents,
                                   ArrayRef<unsigned> terminalCompletionEvents,
                                   const AccessEventSpan &resetEvents,
                                   const HappensBeforeGraph &graph) {
  if (earliestEntryEvents.empty() || terminalCompletionEvents.empty()) {
    return false;
  }
  bool beforeReset =
      llvm::all_of(terminalCompletionEvents, [&](unsigned terminalCompletion) {
        return graph.strictlyPrecedes(terminalCompletion,
                                      resetEvents.first.entry);
      });
  bool afterReset =
      llvm::all_of(earliestEntryEvents, [&](unsigned earliestEntry) {
        return graph.strictlyPrecedes(resetEvents.last.completion,
                                      earliestEntry);
      });
  return beforeReset || afterReset;
}

static bool lifetimeIsOutsideReset(const DFBPerNodeLifetime &lifetime,
                                   SynchronizedDFBResetAttr reset,
                                   const AccessEventSpan &resetEvents,
                                   const HappensBeforeGraph &graph) {
  if (lifetime.epochs.empty()) {
    return intervalIsOutsideReset(lifetime.earliestEntryEvents,
                                  lifetime.terminalCompletionEvents,
                                  resetEvents, graph);
  }
  return llvm::all_of(lifetime.epochs, [&](const DFBLifecycleEpoch &epoch) {
    if (epoch.terminalResetOrdinal == reset.getOrdinal()) {
      return true;
    }
    return intervalIsOutsideReset(epoch.earliestEntryEvents,
                                  epoch.terminalCompletionEvents, resetEvents,
                                  graph);
  });
}

enum class ResetSide { Before, After };

using AccessResetSides = DenseMap<const DFBAccessOccurrence *, ResetSide>;

static bool protocolRunsCrossReset(
    const DFBLogicalLifecycle &logicalDFB, DFBProtocolEffectKind sourceEffect,
    DFBProtocolEffectKind targetEffect, const AccessResetSides &accessSides,
    const AccessRuns &accessRuns, LaunchNodeCoord node,
    const LaunchNodeDomainState &domainState,
    AccessRunPairCompatibility runsAreCompatible) {
  SmallVector<const AccessRun *> sourcesBeforeReset;
  SmallVector<const AccessRun *> targetsBeforeReset;
  SmallVector<const AccessRun *> sourcesAfterReset;
  SmallVector<const AccessRun *> targetsAfterReset;
  bool hasUnmodeledSourceBeforeReset = false;
  bool hasUnmodeledTargetAfterReset = false;
  for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
    auto side = accessSides.find(&access);
    const DFBProtocolEffectKind *protocolEffect = access.getProtocolEffect();
    if (side == accessSides.end() || !protocolEffect ||
        (*protocolEffect != sourceEffect && *protocolEffect != targetEffect)) {
      continue;
    }
    bool isSource = *protocolEffect == sourceEffect;
    auto run = accessRuns.find(&access);
    if (run == accessRuns.end()) {
      hasUnmodeledSourceBeforeReset |=
          isSource && side->second == ResetSide::Before;
      hasUnmodeledTargetAfterReset |=
          !isSource && side->second == ResetSide::After;
      continue;
    }
    SmallVector<const AccessRun *> &sideRuns =
        side->second == ResetSide::Before
            ? (isSource ? sourcesBeforeReset : targetsBeforeReset)
            : (isSource ? sourcesAfterReset : targetsAfterReset);
    sideRuns.push_back(&run->second);
  }

  auto ignoreMatch = [](const AccessRun &, std::uint64_t, const AccessRun &,
                        std::uint64_t, std::uint64_t) { return success(); };
  AccessRunMatchResult beforeResetMatch =
      matchAccessRunPrefix(sourcesBeforeReset, targetsBeforeReset, node,
                           domainState, runsAreCompatible, ignoreMatch);
  AccessRunMatchResult afterResetMatch =
      matchAccessRunPrefix(sourcesAfterReset, targetsAfterReset, node,
                           domainState, runsAreCompatible, ignoreMatch);

  bool hasModeledSourceBeforeReset =
      beforeResetMatch.sourceIndex < sourcesBeforeReset.size();
  bool hasModeledTargetAfterReset =
      afterResetMatch.targetIndex < targetsAfterReset.size();
  return (hasUnmodeledSourceBeforeReset || hasModeledSourceBeforeReset) &&
         (hasUnmodeledTargetAfterReset || hasModeledTargetAfterReset);
}

static bool unprovenLifecycleIsOutsideReset(
    const DFBLogicalLifecycle &logicalDFB, LaunchNodeCoord node,
    const AccessExecutionCounts &executionCounts, bool includeUnknownDomains,
    const AccessEventSpan &resetEvents, const HappensBeforeGraph &graph,
    const StructuralOperationOrder &structuralOrder,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessRuns &accessRuns, const LaunchNodeDomainState &domainState) {
  AccessResetSides accessSides;
  for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
    if (!mayAccessLaunchNode(access, node, executionCounts,
                             includeUnknownDomains)) {
      continue;
    }
    std::optional<AccessEventSpan> events =
        getAccessEventSpan(access, operationEvents, accessEvents);
    if (!events) {
      return false;
    }
    bool beforeReset = graph.strictlyPrecedes(events->last.completion,
                                              resetEvents.first.entry);
    bool afterReset = graph.strictlyPrecedes(resetEvents.last.completion,
                                             events->first.entry);
    if (beforeReset == afterReset) {
      return false;
    }
    accessSides.try_emplace(&access,
                            beforeReset ? ResetSide::Before : ResetSide::After);
  }
  return !protocolRunsCrossReset(
             logicalDFB, DFBProtocolEffectKind::Reserve,
             DFBProtocolEffectKind::Push, accessSides, accessRuns, node,
             domainState,
             [&](const AccessRun &reserve, const AccessRun &push) {
               return acquireReleaseRunsAlign(reserve, push, graph,
                                              structuralOrder, operationEvents,
                                              accessEvents);
             }) &&
         !protocolRunsCrossReset(
             logicalDFB, DFBProtocolEffectKind::Push,
             DFBProtocolEffectKind::Wait, accessSides, accessRuns, node,
             domainState,
             [](const AccessRun &, const AccessRun &) { return true; }) &&
         !protocolRunsCrossReset(
             logicalDFB, DFBProtocolEffectKind::Wait,
             DFBProtocolEffectKind::Pop, accessSides, accessRuns, node,
             domainState, [&](const AccessRun &wait, const AccessRun &pop) {
               return acquireReleaseRunsAlign(wait, pop, graph, structuralOrder,
                                              operationEvents, accessEvents);
             });
}

static Operation *getResetOverlapEvidence(
    const DFBLogicalLifecycle &logicalDFB, const DFBPerNodeLifetime &lifetime,
    LaunchNodeCoord node, const AccessExecutionCounts &executionCounts,
    bool includeUnknownDomains,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &accessEvents) {
  if (lifetime.completionProof.evidence) {
    return lifetime.completionProof.evidence;
  }
  for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
    if (!mayAccessLaunchNode(access, node, executionCounts,
                             includeUnknownDomains)) {
      continue;
    }
    std::optional<AccessEventSpan> events =
        getAccessEventSpan(access, operationEvents, accessEvents);
    if (events &&
        llvm::is_contained(lifetime.earliestEntryEvents, events->first.entry)) {
      return access.operation;
    }
  }
  return logicalDFB.declarations.front();
}

static void collectResetAllocationConflicts(
    ArrayRef<DFBLogicalLifecycle> logicalDFBs,
    ArrayRef<const DFBPerNodeLifetime *> lifetimes,
    ArrayRef<ValidatedSynchronizedReset> synchronizedResets,
    const ResetBoundaryEvents &resetBoundaryEvents,
    const HappensBeforeGraph &graph, LaunchNodeCoord node,
    const StructuralOperationOrder &structuralOrder,
    const AccessExecutionCounts &executionCounts,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessRuns &accessRuns, const LaunchNodeDomainState &domainState,
    bool usePossibleLifetimes,
    SmallVectorImpl<DFBResetAllocationConflict> &conflicts) {
  assert(lifetimes.size() == logicalDFBs.size());
  for (const ValidatedSynchronizedReset &reset : synchronizedResets) {
    auto resetEvents = resetBoundaryEvents.find(reset.reset);
    if (resetEvents == resetBoundaryEvents.end()) {
      continue;
    }
    for (unsigned targetLogicalIndex : reset.targetLogicalIndices) {
      for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
        unsigned overlappingLogicalIndex =
            static_cast<unsigned>(indexedLogicalDFB.index());
        const DFBLogicalLifecycle &logicalDFB = indexedLogicalDFB.value();
        if (targetLogicalIndex == overlappingLogicalIndex) {
          continue;
        }
        bool sharesAllocationGroup =
            logicalDFBs[targetLogicalIndex].allocationGroup &&
            logicalDFBs[targetLogicalIndex].allocationGroup ==
                logicalDFB.allocationGroup;
        if (llvm::is_contained(reset.targetLogicalIndices,
                               overlappingLogicalIndex) &&
            !sharesAllocationGroup) {
          continue;
        }
        const DFBPerNodeLifetime *lifetime = lifetimes[overlappingLogicalIndex];
        if (!lifetime || !lifetime->mayBeActive) {
          continue;
        }
        bool outsideReset =
            lifetime->completionProof.proven()
                ? lifetimeIsOutsideReset(*lifetime, reset.reset,
                                         resetEvents->second, graph)
                : unprovenLifecycleIsOutsideReset(
                      logicalDFB, node, executionCounts,
                      /*includeUnknownDomains=*/usePossibleLifetimes,
                      resetEvents->second, graph, structuralOrder,
                      operationEvents, accessEvents, accessRuns, domainState);
        if (outsideReset) {
          continue;
        }
        bool alreadyRecorded = llvm::any_of(
            conflicts, [&](const DFBResetAllocationConflict &conflict) {
              return conflict.targetLogicalIndex == targetLogicalIndex &&
                     conflict.overlappingLogicalIndex ==
                         overlappingLogicalIndex &&
                     conflict.node == node && conflict.reset == reset.reset;
            });
        if (alreadyRecorded) {
          continue;
        }
        conflicts.push_back({targetLogicalIndex, overlappingLogicalIndex, node,
                             reset.reset, reset.participantOperations.front(),
                             getResetOverlapEvidence(
                                 logicalDFB, *lifetime, node, executionCounts,
                                 /*includeUnknownDomains=*/usePossibleLifetimes,
                                 operationEvents, accessEvents)});
      }
    }
  }
}

static unsigned getLifecycleEpochCount(const DFBPerNodeLifetime *lifetime) {
  if (!lifetime || !lifetime->mayBeActive) {
    return 0;
  }
  return std::max<unsigned>(1, lifetime->epochs.size());
}

static ArrayRef<unsigned>
getEpochEarliestEntryEvents(const DFBPerNodeLifetime &lifetime,
                            unsigned epochIndex) {
  if (lifetime.epochs.empty()) {
    assert(epochIndex == 0 && "complete lifetime has one epoch");
    return lifetime.earliestEntryEvents;
  }
  assert(epochIndex < lifetime.epochs.size());
  return lifetime.epochs[epochIndex].earliestEntryEvents;
}

static ArrayRef<unsigned>
getEpochTerminalCompletionEvents(const DFBPerNodeLifetime &lifetime,
                                 unsigned epochIndex) {
  if (lifetime.epochs.empty()) {
    assert(epochIndex == 0 && "complete lifetime has one epoch");
    return lifetime.terminalCompletionEvents;
  }
  assert(epochIndex < lifetime.epochs.size());
  return lifetime.epochs[epochIndex].terminalCompletionEvents;
}

static const DFBLifecycleCompletionProof &
getEpochCompletionProof(const DFBPerNodeLifetime &lifetime,
                        unsigned epochIndex) {
  if (lifetime.epochs.empty()) {
    assert(epochIndex == 0 && "complete lifetime has one epoch");
    return lifetime.completionProof;
  }
  assert(epochIndex < lifetime.epochs.size());
  return lifetime.epochs[epochIndex].completionProof;
}

static bool proveEpochOrderedBefore(const DFBPerNodeLifetime &before,
                                    unsigned beforeEpochIndex,
                                    const DFBPerNodeLifetime &after,
                                    unsigned afterEpochIndex,
                                    const HappensBeforeGraph &graph) {
  if (!getEpochCompletionProof(before, beforeEpochIndex).proven() ||
      !getEpochCompletionProof(after, afterEpochIndex).proven()) {
    return false;
  }
  return llvm::all_of(
      getEpochTerminalCompletionEvents(before, beforeEpochIndex),
      [&](unsigned terminal) {
        return llvm::all_of(getEpochEarliestEntryEvents(after, afterEpochIndex),
                            [&](unsigned earliest) {
                              return graph.strictlyPrecedes(terminal, earliest);
                            });
      });
}

static void appendInconsistentEpochAccessEvents(
    const DFBLogicalLifecycle &logicalDFB, const DFBPerNodeLifetime &lifetime,
    unsigned epochIndex, LaunchNodeCoord node, const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, bool includeUnknownDomains,
    SmallVectorImpl<unsigned> &inconsistentEvents) {
  auto appendAccess = [&](const DFBAccessOccurrence &access) {
    std::optional<AccessEventSpan> span =
        getAccessEventSpan(access, operationEvents, accessEvents);
    if (!span) {
      return;
    }
    unsigned candidates[] = {span->first.entry, span->first.completion,
                             span->last.entry, span->last.completion};
    for (unsigned event : candidates) {
      if (graph.eventParticipatesInInconsistentOrder(event) &&
          !llvm::is_contained(inconsistentEvents, event)) {
        inconsistentEvents.push_back(event);
      }
    }
  };

  if (!lifetime.epochs.empty()) {
    assert(epochIndex < lifetime.epochs.size());
    for (unsigned accessIndex :
         lifetime.epochs[epochIndex].accessOccurrenceIndices) {
      assert(accessIndex < logicalDFB.accesses.size());
      appendAccess(logicalDFB.accesses[accessIndex]);
    }
    return;
  }

  assert(epochIndex == 0 && "complete lifetime has one epoch");
  for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
    if (mayAccessLaunchNode(access, node, executionCounts,
                            includeUnknownDomains)) {
      appendAccess(access);
    }
  }
}

static void buildEpochOrdering(
    ArrayRef<DFBLogicalLifecycle> logicalDFBs,
    ArrayRef<const DFBPerNodeLifetime *> lifetimes, LaunchNodeCoord node,
    const HappensBeforeGraph &graph,
    const DenseMap<Operation *, EventPair> &operationEvents,
    const DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents,
    const AccessExecutionCounts &executionCounts, bool includeUnknownDomains,
    SmallVectorImpl<unsigned> &logicalOffsets,
    SmallVectorImpl<llvm::BitVector> &orderedBefore,
    SmallVectorImpl<llvm::BitVector> &inconsistent) {
  struct LifecycleEpochIdentity {
    unsigned logicalIndex;
    unsigned epochIndex;
    const DFBPerNodeLifetime *lifetime;
  };

  assert(lifetimes.size() == logicalDFBs.size());
  SmallVector<LifecycleEpochIdentity> epochIdentities;
  logicalOffsets.reserve(logicalDFBs.size() + 1);
  logicalOffsets.push_back(0);
  for (auto [logicalIndex, lifetime] : llvm::enumerate(lifetimes)) {
    unsigned epochCount = getLifecycleEpochCount(lifetime);
    for (unsigned epochIndex = 0; epochIndex < epochCount; ++epochIndex) {
      epochIdentities.push_back(
          {static_cast<unsigned>(logicalIndex), epochIndex, lifetime});
    }
    logicalOffsets.push_back(epochIdentities.size());
  }

  orderedBefore.assign(epochIdentities.size(),
                       llvm::BitVector(epochIdentities.size()));
  SmallVector<SmallVector<unsigned>> inconsistentEventsByEpoch(
      epochIdentities.size());
  for (auto [beforeFlatIndex, beforeIdentity] :
       llvm::enumerate(epochIdentities)) {
    unsigned beforeLogicalIndex = beforeIdentity.logicalIndex;
    unsigned beforeEpochIndex = beforeIdentity.epochIndex;
    const DFBLogicalLifecycle &beforeLogicalDFB =
        logicalDFBs[beforeLogicalIndex];
    const DFBPerNodeLifetime *beforeLifetime = beforeIdentity.lifetime;
    assert(beforeLifetime && "flattened epoch must have a lifetime");
    appendInconsistentEpochAccessEvents(
        beforeLogicalDFB, *beforeLifetime, beforeEpochIndex, node, graph,
        operationEvents, accessEvents, executionCounts, includeUnknownDomains,
        inconsistentEventsByEpoch[beforeFlatIndex]);
    for (auto [afterFlatIndex, afterIdentity] :
         llvm::enumerate(epochIdentities)) {
      unsigned afterEpochIndex = afterIdentity.epochIndex;
      const DFBPerNodeLifetime *afterLifetime = afterIdentity.lifetime;
      assert(afterLifetime && "flattened epoch must have a lifetime");
      if (proveEpochOrderedBefore(*beforeLifetime, beforeEpochIndex,
                                  *afterLifetime, afterEpochIndex, graph)) {
        orderedBefore[beforeFlatIndex].set(afterFlatIndex);
      }
    }
  }

  inconsistent.assign(epochIdentities.size(),
                      llvm::BitVector(epochIdentities.size()));
  for (unsigned lhsIndex = 0; lhsIndex < epochIdentities.size(); ++lhsIndex) {
    for (unsigned rhsIndex = lhsIndex; rhsIndex < epochIdentities.size();
         ++rhsIndex) {
      bool inconsistentOrder = llvm::any_of(
          inconsistentEventsByEpoch[lhsIndex], [&](unsigned lhsEvent) {
            return llvm::any_of(
                inconsistentEventsByEpoch[rhsIndex], [&](unsigned rhsEvent) {
                  return graph.hasInconsistentOrder(lhsEvent, rhsEvent);
                });
          });
      if (inconsistentOrder) {
        inconsistent[lhsIndex].set(rhsIndex);
        inconsistent[rhsIndex].set(lhsIndex);
      }
    }
  }
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
  SmallVector<SynchronizedResetOccurrence> resetOccurrences;
  SmallVector<DFBReconfigurationOccurrence> reconfigurationOccurrences;
  if (failed(collectLogicalDFBs(module, identityAnalysis, logicalDFBs,
                                unknownAccessOperations, resetOccurrences,
                                reconfigurationOccurrences, analysisFailure,
                                dependsOnLaunchNode))) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  if (logicalDFBs.empty()) {
    if (!resetOccurrences.empty()) {
      errorOperation = resetOccurrences.front().operation;
      errorMessage =
          "synchronized DFB reset requires at least one DFB allocation";
      return;
    }
    if (reconfigurationOccurrences.empty()) {
      return;
    }
  }

  LivenessDomainState domainState;
  domainState.initialize(module);
  exactLaunchGridAvailable = domainState.hasLaunchGrid;
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
    if (llvm::any_of(logicalDFB.accesses,
                     [](const DFBAccessOccurrence &access) {
                       return !access.launchDomain.known &&
                              (access.getProtocolEffect() ||
                               access.opaqueExternalAccess);
                     })) {
      logicalDFB.launchDomain = LaunchNodeDomain::unknown();
    }
  }
  for (SynchronizedResetOccurrence &reset : resetOccurrences) {
    const AccessDomain &resetDomain = getRefinedAccessDomain(reset.operation);
    reset.launchDomain = resetDomain.domain;
  }
  for (DFBReconfigurationOccurrence &reconfiguration :
       reconfigurationOccurrences) {
    auto domainIt = domainState.accessDomains.find(reconfiguration.operation);
    AccessDomain boundaryDomain =
        domainIt == domainState.accessDomains.end()
            ? AccessDomain{LaunchNodeDomain::unknown(),
                           reconfiguration.operation}
            : domainIt->second;
    boundaryDomain = refineUnknownAccessDomainFromExecutionCounts(
        reconfiguration.operation, boundaryDomain, domainState);
    reconfiguration.launchDomain = boundaryDomain.domain;
  }
  if (failed(validateSynchronizedResetDeclarations(resetOccurrences,
                                                   analysisFailure))) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  if (failed(validateDFBReconfigurationDeclarations(reconfigurationOccurrences,
                                                    analysisFailure))) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
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
      logicalDFB.accesses.push_back({unknownAccess, std::monostate{}, 0, 0,
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

  StructuralOperationOrder structuralOrder(module);
  launchNodes.append(domainState.baseDomain.nodes.begin(),
                     domainState.baseDomain.nodes.end());
  SmallVector<int64_t> declaredReconfigurationOrdinals;
  DenseMap<int64_t, unsigned> reconfigurationDeclarationOrder;
  DenseMap<int64_t, DenseSet<int64_t>> reconfigurationSuccessors;
  DenseMap<int64_t, Operation *> reconfigurationEvidence;
  for (const DFBReconfigurationOccurrence &occurrence :
       reconfigurationOccurrences) {
    int64_t ordinal = occurrence.boundary.getOrdinal();
    if (!reconfigurationDeclarationOrder.contains(ordinal)) {
      reconfigurationDeclarationOrder[ordinal] =
          declaredReconfigurationOrdinals.size();
      declaredReconfigurationOrdinals.push_back(ordinal);
      reconfigurationEvidence[ordinal] = occurrence.operation;
    }
  }
  orderedBeforeByNode.reserve(launchNodes.size());
  conditionallyOrderedBeforeByNode.reserve(launchNodes.size());
  inconsistentOrderByNode.reserve(launchNodes.size());
  conditionallyInconsistentOrderByNode.reserve(launchNodes.size());
  epochOrderedBeforeByNode.reserve(launchNodes.size());
  conditionallyEpochOrderedBeforeByNode.reserve(launchNodes.size());
  bool collectAllocationDiagnostics = false;
  LLVM_DEBUG(collectAllocationDiagnostics = true);
  if (collectAllocationDiagnostics) {
    for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
      logicalDFB.allocationDiagnostics =
          std::make_unique<DFBLogicalLifecycleDiagnostics>();
    }
  }
  for (LaunchNodeCoord node : launchNodes) {
    SmallVector<ValidatedSynchronizedReset> validatedResets;
    if (failed(validateSynchronizedResetsAtNode(resetOccurrences, node,
                                                domainState, validatedResets,
                                                analysisFailure))) {
      errorOperation = analysisFailure.operation;
      errorMessage = std::move(analysisFailure.message);
      return;
    }
    SmallVector<ValidatedDFBReconfiguration> validatedReconfigurations;
    if (failed(validateDFBReconfigurationsAtNode(
            reconfigurationOccurrences, node, domainState, structuralOrder,
            validatedReconfigurations, analysisFailure))) {
      errorOperation = analysisFailure.operation;
      errorMessage = std::move(analysisFailure.message);
      return;
    }
    AccessExecutionCounts executionCounts =
        collectAccessExecutionCounts(logicalDFBs, node, domainState);
    AccessRuns accessRuns =
        collectAccessRuns(logicalDFBs, node, domainState, executionCounts,
                          /*includeUnknownDomains=*/false);
    ProgramOrderTopologyInputs graphTopologyInputs =
        collectProgramOrderTopologyInputs(
            logicalDFBs, validatedResets, validatedReconfigurations, node,
            executionCounts, accessRuns, structuralOrder,
            /*includeUnknownDomains=*/false);
    ProgramOrderGraphState graphState = buildProgramOrderTopologyState(
        module, graphTopologyInputs, structuralOrder);
    std::optional<AccessRuns> possibleAccessRuns;
    std::optional<ProgramOrderGraphState> possibleGraphState;
    if (hasUnknownDFBLaunchDomain) {
      possibleAccessRuns.emplace(
          collectAccessRuns(logicalDFBs, node, domainState, executionCounts,
                            /*includeUnknownDomains=*/true));
      ProgramOrderTopologyInputs possibleGraphTopologyInputs =
          collectProgramOrderTopologyInputs(
              logicalDFBs, validatedResets, validatedReconfigurations, node,
              executionCounts, *possibleAccessRuns, structuralOrder,
              /*includeUnknownDomains=*/true);
      bool graphTopologiesMatch =
          graphTopologyInputs == possibleGraphTopologyInputs;
      possibleGraphState.emplace(
          graphTopologiesMatch
              ? graphState
              : buildProgramOrderTopologyState(
                    module, possibleGraphTopologyInputs, structuralOrder));
#ifndef NDEBUG
      // Small graphs retain an independent reconstruction oracle without
      // repeating model-scale graph construction in assertion-enabled builds.
      if (graphTopologiesMatch && graphState.graph.getEventCount() <= 128) {
        ProgramOrderGraphState reconstructedPossibleGraphState =
            buildProgramOrderTopologyState(module, possibleGraphTopologyInputs,
                                           structuralOrder);
        assert(reconstructedPossibleGraphState == *possibleGraphState &&
               "equal program-order inputs must produce equal graph topology");
      }
#endif
    }
    HappensBeforeGraph &graph = graphState.graph;
    DenseMap<Operation *, EventPair> &operationEvents =
        graphState.operationEvents;
    DenseMap<const DFBAccessOccurrence *, AccessEventSpan> &accessEvents =
        graphState.accessEvents;
    ResetBoundaryEvents &resetBoundaryEvents = graphState.resetBoundaryEvents;
    ReconfigurationBoundaryEvents &reconfigurationBoundaryEvents =
        graphState.reconfigurationBoundaryEvents;
    addSynchronizedResetAccessEdges(logicalDFBs, validatedResets, node, graph,
                                    operationEvents, accessEvents,
                                    resetBoundaryEvents, executionCounts,
                                    accessRuns, structuralOrder,
                                    /*includeUnknownDomains=*/false);
    addDFBReconfigurationAccessEdges(
        logicalDFBs, validatedReconfigurations, node, graph, operationEvents,
        accessEvents, reconfigurationBoundaryEvents, executionCounts,
        structuralOrder, /*includeUnknownDomains=*/false);
    addProtocolSynchronizationEdges(logicalDFBs, graph, operationEvents,
                                    accessEvents, executionCounts, accessRuns,
                                    node, domainState, structuralOrder,
                                    /*includeUnknownDomains=*/false);

    SmallVector<const ValidatedDFBReconfiguration *> orderedReconfigurations;
    orderedReconfigurations.reserve(validatedReconfigurations.size());
    for (const ValidatedDFBReconfiguration &reconfiguration :
         validatedReconfigurations) {
      orderedReconfigurations.push_back(&reconfiguration);
    }
    for (auto [lhsIndex, lhs] : llvm::enumerate(orderedReconfigurations)) {
      for (auto [rhsIndex, rhs] : llvm::enumerate(orderedReconfigurations)) {
        if (lhsIndex >= rhsIndex) {
          continue;
        }
        const EventPair &lhsEvents =
            reconfigurationBoundaryEvents.lookup(lhs->boundary);
        const EventPair &rhsEvents =
            reconfigurationBoundaryEvents.lookup(rhs->boundary);
        bool lhsBeforeRhs =
            graph.strictlyPrecedes(lhsEvents.completion, rhsEvents.entry);
        bool rhsBeforeLhs =
            graph.strictlyPrecedes(rhsEvents.completion, lhsEvents.entry);
        if (lhsBeforeRhs == rhsBeforeLhs) {
          errorOperation = rhs->participantOperations.front();
          errorMessage =
              "DFB reconfiguration boundary execution order is not proved";
          return;
        }
      }
    }
    llvm::sort(orderedReconfigurations, [&](const auto *lhs, const auto *rhs) {
      const EventPair &lhsEvents =
          reconfigurationBoundaryEvents.lookup(lhs->boundary);
      const EventPair &rhsEvents =
          reconfigurationBoundaryEvents.lookup(rhs->boundary);
      return graph.strictlyPrecedes(lhsEvents.completion, rhsEvents.entry);
    });
    for (auto adjacent : llvm::zip(orderedReconfigurations,
                                   llvm::drop_begin(orderedReconfigurations))) {
      int64_t beforeOrdinal = std::get<0>(adjacent)->boundary.getOrdinal();
      int64_t afterOrdinal = std::get<1>(adjacent)->boundary.getOrdinal();
      reconfigurationSuccessors[beforeOrdinal].insert(afterOrdinal);
    }

    SmallVector<const DFBPerNodeLifetime *> nodeLifetimes(logicalDFBs.size());
    for (auto [logicalIndex, logicalDFB] : llvm::enumerate(logicalDFBs)) {
      if (!knownLaunchNodeDomainContains(logicalDFB.launchDomain, node)) {
        continue;
      }
      DFBLogicalLifecycleDiagnostics *allocationDiagnostics =
          logicalDFB.allocationDiagnostics.get();
      DFBLifecycleCompletionProof proof = computePerNodeLifetime(
          logicalDFB, logicalIndex, node, logicalDFB.nodeLifetimes,
          allocationDiagnostics
              ? &allocationDiagnostics->nodeLifetimeDiagnostics
              : nullptr,
          validatedResets, resetBoundaryEvents, validatedReconfigurations,
          reconfigurationBoundaryEvents, graph, structuralOrder,
          operationEvents, accessEvents, executionCounts, accessRuns,
          domainState);
      logicalDFB.nodeLifetimes.back().completionProof = proof;
      nodeLifetimes[logicalIndex] = &logicalDFB.nodeLifetimes.back();
    }

    collectResetAllocationConflicts(
        logicalDFBs, nodeLifetimes, validatedResets, resetBoundaryEvents, graph,
        node, structuralOrder, executionCounts, operationEvents, accessEvents,
        accessRuns, domainState, /*usePossibleLifetimes=*/false,
        resetAllocationConflicts);

    orderedBeforeByNode.push_back(buildLogicalOrdering(nodeLifetimes, graph));
    inconsistentOrderByNode.push_back(collectInconsistentAccessOrder(
        logicalDFBs, node, graph, operationEvents, accessEvents,
        executionCounts, /*includeUnknownDomains=*/false));
    EpochOrdering epochOrdering;
    buildEpochOrdering(logicalDFBs, nodeLifetimes, node, graph, operationEvents,
                       accessEvents, executionCounts,
                       /*includeUnknownDomains=*/false,
                       epochOrdering.logicalOffsets,
                       epochOrdering.orderedBefore, epochOrdering.inconsistent);
    epochOrderedBeforeByNode.push_back(std::move(epochOrdering));

    // Possible-domain reachability cannot affect exact-domain reuse.
    if (!hasUnknownDFBLaunchDomain) {
      conditionallyOrderedBeforeByNode.emplace_back(
          logicalDFBs.size(), llvm::BitVector(logicalDFBs.size()));
      conditionallyInconsistentOrderByNode.emplace_back(
          logicalDFBs.size(), llvm::BitVector(logicalDFBs.size()));
      continue;
    }

    assert(possibleAccessRuns && possibleGraphState &&
           "unknown domains must construct a possible graph");
    HappensBeforeGraph &possibleGraph = possibleGraphState->graph;
    DenseMap<Operation *, EventPair> &possibleOperationEvents =
        possibleGraphState->operationEvents;
    DenseMap<const DFBAccessOccurrence *, AccessEventSpan>
        &possibleAccessEvents = possibleGraphState->accessEvents;
    ResetBoundaryEvents &possibleResetBoundaryEvents =
        possibleGraphState->resetBoundaryEvents;
    ReconfigurationBoundaryEvents &possibleReconfigurationBoundaryEvents =
        possibleGraphState->reconfigurationBoundaryEvents;
    addSynchronizedResetAccessEdges(
        logicalDFBs, validatedResets, node, possibleGraph,
        possibleOperationEvents, possibleAccessEvents,
        possibleResetBoundaryEvents, executionCounts, *possibleAccessRuns,
        structuralOrder,
        /*includeUnknownDomains=*/true);
    addDFBReconfigurationAccessEdges(
        logicalDFBs, validatedReconfigurations, node, possibleGraph,
        possibleOperationEvents, possibleAccessEvents,
        possibleReconfigurationBoundaryEvents, executionCounts, structuralOrder,
        /*includeUnknownDomains=*/true);
    addProtocolSynchronizationEdges(
        logicalDFBs, possibleGraph, possibleOperationEvents,
        possibleAccessEvents, executionCounts, *possibleAccessRuns, node,
        domainState, structuralOrder, /*includeUnknownDomains=*/true);
    SmallVector<const DFBPerNodeLifetime *> possibleNodeLifetimes(
        logicalDFBs.size());
    for (auto [logicalIndex, logicalDFB] : llvm::enumerate(logicalDFBs)) {
      if (logicalDFB.launchDomain.known) {
        continue;
      }
      DFBLogicalLifecycleDiagnostics *allocationDiagnostics =
          logicalDFB.allocationDiagnostics.get();
      DFBLifecycleCompletionProof proof = computePerNodeLifetime(
          logicalDFB, logicalIndex, node, logicalDFB.possibleNodeLifetimes,
          allocationDiagnostics
              ? &allocationDiagnostics->possibleNodeLifetimeDiagnostics
              : nullptr,
          validatedResets, possibleResetBoundaryEvents,
          validatedReconfigurations, possibleReconfigurationBoundaryEvents,
          possibleGraph, structuralOrder, possibleOperationEvents,
          possibleAccessEvents, executionCounts, *possibleAccessRuns,
          domainState,
          /*includeUnknownDomains=*/true);
      logicalDFB.possibleNodeLifetimes.back().completionProof = proof;
      possibleNodeLifetimes[logicalIndex] =
          &logicalDFB.possibleNodeLifetimes.back();
    }

    collectResetAllocationConflicts(
        logicalDFBs, possibleNodeLifetimes, validatedResets,
        possibleResetBoundaryEvents, possibleGraph, node, structuralOrder,
        executionCounts, possibleOperationEvents, possibleAccessEvents,
        *possibleAccessRuns, domainState, /*usePossibleLifetimes=*/true,
        resetAllocationConflicts);

    conditionallyOrderedBeforeByNode.push_back(
        buildLogicalOrdering(possibleNodeLifetimes, possibleGraph));
    conditionallyInconsistentOrderByNode.push_back(
        collectInconsistentAccessOrder(logicalDFBs, node, possibleGraph,
                                       possibleOperationEvents,
                                       possibleAccessEvents, executionCounts,
                                       /*includeUnknownDomains=*/true));
    EpochOrdering conditionalEpochOrdering;
    SmallVector<const DFBPerNodeLifetime *> conditionalEpochLifetimes;
    conditionalEpochLifetimes.reserve(logicalDFBs.size());
    for (unsigned logicalIndex = 0; logicalIndex < logicalDFBs.size();
         ++logicalIndex) {
      conditionalEpochLifetimes.push_back(
          logicalDFBs[logicalIndex].launchDomain.known
              ? nodeLifetimes[logicalIndex]
              : possibleNodeLifetimes[logicalIndex]);
    }
    buildEpochOrdering(
        logicalDFBs, conditionalEpochLifetimes, node, possibleGraph,
        possibleOperationEvents, possibleAccessEvents, executionCounts,
        /*includeUnknownDomains=*/true, conditionalEpochOrdering.logicalOffsets,
        conditionalEpochOrdering.orderedBefore,
        conditionalEpochOrdering.inconsistent);
    conditionallyEpochOrderedBeforeByNode.push_back(
        std::move(conditionalEpochOrdering));
  }

  DenseMap<int64_t, unsigned> reconfigurationPredecessorCount;
  for (int64_t ordinal : declaredReconfigurationOrdinals) {
    reconfigurationPredecessorCount[ordinal] = 0;
  }
  for (const auto &[beforeOrdinal, successors] : reconfigurationSuccessors) {
    (void)beforeOrdinal;
    for (int64_t successor : successors) {
      ++reconfigurationPredecessorCount[successor];
    }
  }
  SmallVector<int64_t> readyReconfigurations;
  for (int64_t ordinal : declaredReconfigurationOrdinals) {
    if (reconfigurationPredecessorCount.lookup(ordinal) == 0) {
      readyReconfigurations.push_back(ordinal);
    }
  }
  auto sortReadyReconfigurations = [&] {
    llvm::sort(readyReconfigurations, [&](int64_t lhs, int64_t rhs) {
      return reconfigurationDeclarationOrder.lookup(lhs) >
             reconfigurationDeclarationOrder.lookup(rhs);
    });
  };
  sortReadyReconfigurations();
  while (!readyReconfigurations.empty()) {
    int64_t ordinal = readyReconfigurations.pop_back_val();
    reconfigurationBoundaryOrdinals.push_back(ordinal);
    for (int64_t successor : reconfigurationSuccessors[ordinal]) {
      unsigned &predecessorCount = reconfigurationPredecessorCount[successor];
      assert(predecessorCount > 0 && "boundary predecessor count underflow");
      --predecessorCount;
      if (predecessorCount == 0) {
        readyReconfigurations.push_back(successor);
      }
    }
    sortReadyReconfigurations();
  }
  if (reconfigurationBoundaryOrdinals.size() !=
      declaredReconfigurationOrdinals.size()) {
    int64_t cyclicOrdinal =
        *llvm::find_if(declaredReconfigurationOrdinals, [&](int64_t ordinal) {
          return reconfigurationPredecessorCount.lookup(ordinal) != 0;
        });
    errorOperation = reconfigurationEvidence.lookup(cyclicOrdinal);
    errorMessage =
        "DFB reconfiguration boundaries execute in different orders across "
        "launch nodes";
    return;
  }

  assert(orderedBeforeByNode.size() == launchNodes.size() &&
         conditionallyOrderedBeforeByNode.size() == launchNodes.size() &&
         inconsistentOrderByNode.size() == launchNodes.size() &&
         conditionallyInconsistentOrderByNode.size() == launchNodes.size() &&
         "per-node order relations must cover the launch grid");

  for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    bool exactLifecyclesComplete =
        !logicalDFB.nodeLifetimes.empty() &&
        llvm::all_of(logicalDFB.nodeLifetimes,
                     [](const DFBPerNodeLifetime &lifetime) {
                       return lifetime.completionProof.proven();
                     });
    bool possibleLifecyclesComplete =
        !logicalDFB.possibleNodeLifetimes.empty() &&
        llvm::all_of(logicalDFB.possibleNodeLifetimes,
                     [](const DFBPerNodeLifetime &lifetime) {
                       return !lifetime.mayBeActive ||
                              lifetime.completionProof.proven();
                     });
    bool opaqueExternalAccessesComplete = logicalDFB.launchDomain.known
                                              ? exactLifecyclesComplete
                                              : possibleLifecyclesComplete;
    logicalDFB.accessCompletionProven = llvm::all_of(
        logicalDFB.accesses, [&](const DFBAccessOccurrence &access) {
          if (access.getProtocolEffect() ||
              access.getNonTransactionalAccess()) {
            return true;
          }
          if (access.opaqueExternalAccess) {
            return opaqueExternalAccessesComplete;
          }
          // Separate acquire and release operations establish queue ownership
          // for the slot transferred by ttl.copy.
          return isa<CopyOp>(access.operation);
        });
    logicalDFB.bounded = logicalDFB.accessCompletionProven &&
                         logicalDFB.launchDomain.known &&
                         exactLifecyclesComplete;
    bool hasProvenConditionalLifecycle =
        llvm::any_of(logicalDFB.possibleNodeLifetimes,
                     [](const DFBPerNodeLifetime &lifetime) {
                       return lifetime.mayBeActive &&
                              lifetime.conditionalExecutionProven &&
                              lifetime.completionProof.proven();
                     });
    logicalDFB.conditionallyBounded =
        logicalDFB.accessCompletionProven && !logicalDFB.launchDomain.known &&
        hasProvenConditionalLifecycle &&
        llvm::all_of(logicalDFB.possibleNodeLifetimes,
                     [](const DFBPerNodeLifetime &lifetime) {
                       return !lifetime.mayBeActive ||
                              (lifetime.conditionalExecutionProven &&
                               lifetime.completionProof.proven());
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

bool DFBConcurrentKernelLivenessAnalysis::hasInconsistentOrder(
    unsigned lhsIndex, unsigned rhsIndex, LaunchNodeCoord node) const {
  auto nodeIt = llvm::find(launchNodes, node);
  assert(nodeIt != launchNodes.end() && "node must be in the launch grid");
  unsigned nodeIndex = nodeIt - launchNodes.begin();
  assert(lhsIndex < inconsistentOrderByNode[nodeIndex].size() &&
         rhsIndex < inconsistentOrderByNode[nodeIndex].size());
  return inconsistentOrderByNode[nodeIndex][lhsIndex].test(rhsIndex);
}

bool DFBConcurrentKernelLivenessAnalysis::hasConditionallyInconsistentOrder(
    unsigned lhsIndex, unsigned rhsIndex, LaunchNodeCoord node) const {
  auto nodeIt = llvm::find(launchNodes, node);
  assert(nodeIt != launchNodes.end() && "node must be in the launch grid");
  unsigned nodeIndex = nodeIt - launchNodes.begin();
  assert(lhsIndex < conditionallyInconsistentOrderByNode[nodeIndex].size() &&
         rhsIndex < conditionallyInconsistentOrderByNode[nodeIndex].size());
  return conditionallyInconsistentOrderByNode[nodeIndex][lhsIndex].test(
      rhsIndex);
}

bool DFBConcurrentKernelLivenessAnalysis::isEpochOrderedBefore(
    unsigned beforeIndex, unsigned beforeEpochIndex, unsigned afterIndex,
    unsigned afterEpochIndex, LaunchNodeCoord node) const {
  return queryEpochRelation(epochOrderedBeforeByNode, beforeIndex,
                            beforeEpochIndex, afterIndex, afterEpochIndex, node,
                            /*inconsistent=*/false);
}

bool DFBConcurrentKernelLivenessAnalysis::isConditionallyEpochOrderedBefore(
    unsigned beforeIndex, unsigned beforeEpochIndex, unsigned afterIndex,
    unsigned afterEpochIndex, LaunchNodeCoord node) const {
  return queryEpochRelation(conditionallyEpochOrderedBeforeByNode, beforeIndex,
                            beforeEpochIndex, afterIndex, afterEpochIndex, node,
                            /*inconsistent=*/false);
}

bool DFBConcurrentKernelLivenessAnalysis::hasInconsistentEpochOrder(
    unsigned lhsIndex, unsigned lhsEpochIndex, unsigned rhsIndex,
    unsigned rhsEpochIndex, LaunchNodeCoord node) const {
  return queryEpochRelation(epochOrderedBeforeByNode, lhsIndex, lhsEpochIndex,
                            rhsIndex, rhsEpochIndex, node,
                            /*inconsistent=*/true);
}

bool DFBConcurrentKernelLivenessAnalysis::
    hasConditionallyInconsistentEpochOrder(unsigned lhsIndex,
                                           unsigned lhsEpochIndex,
                                           unsigned rhsIndex,
                                           unsigned rhsEpochIndex,
                                           LaunchNodeCoord node) const {
  return queryEpochRelation(conditionallyEpochOrderedBeforeByNode, lhsIndex,
                            lhsEpochIndex, rhsIndex, rhsEpochIndex, node,
                            /*inconsistent=*/true);
}

bool DFBConcurrentKernelLivenessAnalysis::queryEpochRelation(
    ArrayRef<EpochOrdering> orderings, unsigned lhsIndex,
    unsigned lhsEpochIndex, unsigned rhsIndex, unsigned rhsEpochIndex,
    LaunchNodeCoord node, bool inconsistent) const {
  auto nodeIt = llvm::find(launchNodes, node);
  assert(nodeIt != launchNodes.end() && "node must be in the launch grid");
  unsigned nodeIndex = nodeIt - launchNodes.begin();
  assert(nodeIndex < orderings.size());
  const EpochOrdering &ordering = orderings[nodeIndex];
  assert(lhsIndex + 1 < ordering.logicalOffsets.size() &&
         rhsIndex + 1 < ordering.logicalOffsets.size());
  unsigned lhsFlatIndex = ordering.logicalOffsets[lhsIndex] + lhsEpochIndex;
  unsigned rhsFlatIndex = ordering.logicalOffsets[rhsIndex] + rhsEpochIndex;
  assert(lhsFlatIndex < ordering.logicalOffsets[lhsIndex + 1] &&
         rhsFlatIndex < ordering.logicalOffsets[rhsIndex + 1]);
  ArrayRef<llvm::BitVector> relation =
      inconsistent ? ordering.inconsistent : ordering.orderedBefore;
  return relation[lhsFlatIndex].test(rhsFlatIndex);
}

} // namespace mlir::tt::ttl
