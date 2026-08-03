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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
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
    unsigned entry = successors.size();
    successors.emplace_back();
    unsigned completion = successors.size();
    successors.emplace_back();
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
  SmallVector<SmallVector<unsigned>> successors;
  SmallVector<llvm::BitVector> reachable;
};

struct AccessDomain {
  LaunchNodeDomain domain = LaunchNodeDomain::unknown();
  Operation *unanalyzableOperation = nullptr;
};

struct LivenessDomainState : LaunchNodeDomainState {
  DenseMap<Operation *, AccessDomain> accessDomains;
};

static DFBProtocolEffect classifyEffect(Operation *operation) {
  if (isa<CBReserveOp>(operation)) {
    return DFBProtocolEffect::Reserve;
  }
  if (isa<CBPushOp>(operation)) {
    return DFBProtocolEffect::Push;
  }
  if (isa<CBWaitOp>(operation)) {
    return DFBProtocolEffect::Wait;
  }
  if (isa<CBPopOp>(operation)) {
    return DFBProtocolEffect::Pop;
  }
  return DFBProtocolEffect::OpaqueAccess;
}

static bool effectAdvancesWritePointer(DFBProtocolEffect effect) {
  return effect == DFBProtocolEffect::Reserve ||
         effect == DFBProtocolEffect::Push;
}

static bool effectAdvancesReadPointer(DFBProtocolEffect effect) {
  return effect == DFBProtocolEffect::Wait || effect == DFBProtocolEffect::Pop;
}

static std::optional<DFBPointerOwner>
getPointerOwner(Operation *operation, LaunchNodeCoord node,
                DFBProtocolEffect effect) {
  if (!effectAdvancesWritePointer(effect) &&
      !effectAdvancesReadPointer(effect)) {
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

  DFBPointerDirection direction = effectAdvancesWritePointer(effect)
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
  int64_t nocIndex = getNocIndex(operation);
  if (nocIndex != 0 && nocIndex != 1) {
    return std::nullopt;
  }
  DFBPointerProcessor processor =
      nocIndex == 0 ? DFBPointerProcessor::Noc0 : DFBPointerProcessor::Noc1;
  return DFBPointerOwner{node, processor, direction};
}

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

static bool releaseFollowsOwnedUses(Operation *acquire, Operation *release) {
  if (acquire->getBlock() != release->getBlock()) {
    return false;
  }
  SmallVector<Operation *> acquires = {acquire};
  DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);
  Operation *lastOwnedUse = findLastDFBAcquireOwnedUse(interval);
  return lastOwnedUse == acquire || lastOwnedUse->isBeforeInBlock(release);
}

static SmallVector<Operation *>
findMinimalOperations(ArrayRef<Operation *> operations,
                      const HappensBeforeGraph &graph,
                      const DenseMap<Operation *, EventPair> &operationEvents) {
  SmallVector<Operation *> minimal;
  for (Operation *candidate : operations) {
    std::optional<EventPair> candidateEvents =
        getProjectedEvents(candidate, operationEvents);
    if (!candidateEvents) {
      continue;
    }
    bool hasPredecessor = llvm::any_of(operations, [&](Operation *other) {
      std::optional<EventPair> otherEvents =
          getProjectedEvents(other, operationEvents);
      return otherEvents &&
             graph.strictlyPrecedes(otherEvents->entry, candidateEvents->entry);
    });
    if (!hasPredecessor && !llvm::is_contained(minimal, candidate)) {
      minimal.push_back(candidate);
    }
  }
  return minimal;
}

static LogicalResult verifyCustomFunctionIndexDependencies(
    OpaqueCallOp call, const DFBLogicalIdentityAnalysis &identityAnalysis,
    DFBAnalysisFailure &analysisFailure) {
  SmallVector<int64_t> dependencyIds;
  for (Value operand : call.getArgOperands()) {
    if (!isa<CircularBufferType>(operand.getType())) {
      continue;
    }
    FailureOr<int64_t> logicalId = identityAnalysis.getLogicalId(operand);
    if (succeeded(logicalId)) {
      dependencyIds.push_back(*logicalId);
    }
  }

  auto verifyIndexDependency = [&](Value operand) {
    auto getId = operand.getDefiningOp<GetDfbIdOp>();
    if (!getId) {
      return success();
    }
    FailureOr<int64_t> logicalId =
        identityAnalysis.getLogicalId(getId.getDfb());
    if (failed(logicalId)) {
      return success();
    }
    if (!llvm::is_contained(dependencyIds, *logicalId)) {
      analysisFailure.set(
          call, "custom function consumes the physical index for logical DFB " +
                    std::to_string(*logicalId) +
                    " without listing that DFB as a dependency operand");
      return failure();
    }
    return success();
  };

  for (Value operand : call.getTemplateArgVals()) {
    if (failed(verifyIndexDependency(operand))) {
      return failure();
    }
  }
  for (Value operand : call.getArgOperands()) {
    if (failed(verifyIndexDependency(operand))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult collectLogicalDFBs(
    ModuleOp module, const DFBLogicalIdentityAnalysis &identityAnalysis,
    SmallVectorImpl<DFBLogicalLifecycle> &logicalDFBs,
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
    dependsOnLaunchNode |=
        isa<CoreXOp, CoreYOp, CreatePipeOp, PipeNetPredicateOpInterface>(
            operation);
    if (auto call = dyn_cast<OpaqueCallOp>(operation);
        call && failed(verifyCustomFunctionIndexDependencies(
                    call, identityAnalysis, analysisFailure))) {
      return WalkResult::interrupt();
    }
    if (!mayAccessDFBStorage(operation)) {
      return WalkResult::advance();
    }
    SmallVector<unsigned> operationLogicalIndices;
    for (Value operand : operation->getOperands()) {
      if (!isa<CircularBufferType>(operand.getType())) {
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
      assert(logicalIt != logicalIndexById.end());
      if (!llvm::is_contained(operationLogicalIndices, logicalIt->second)) {
        operationLogicalIndices.push_back(logicalIt->second);
      }
    }
    for (unsigned logicalIndex : operationLogicalIndices) {
      logicalDFBs[logicalIndex].accesses.push_back(
          {operation, classifyEffect(operation), LaunchNodeDomain::unknown(),
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
    auto hasEffect = [&](DFBProtocolEffect effect) {
      return llvm::any_of(logicalDFB.accesses,
                          [&](const DFBAccessOccurrence &access) {
                            return access.effect == effect;
                          });
    };
    StringRef missingOperation;
    if (!hasEffect(DFBProtocolEffect::Reserve)) {
      missingOperation = "ttl.cb_reserve";
    } else if (!hasEffect(DFBProtocolEffect::Push)) {
      missingOperation = "ttl.cb_push";
    } else if (!hasEffect(DFBProtocolEffect::Wait)) {
      missingOperation = "ttl.cb_wait";
    } else if (!hasEffect(DFBProtocolEffect::Pop)) {
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

static void
buildProgramOrderGraph(ModuleOp module,
                       ArrayRef<DFBLogicalLifecycle> logicalDFBs,
                       LaunchNodeCoord node, HappensBeforeGraph &graph,
                       DenseMap<Operation *, EventPair> &operationEvents) {
  llvm::DenseSet<Operation *> modeledOperations;
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
      if (!knownLaunchNodeDomainContains(access.launchDomain, node)) {
        continue;
      }
      if (Operation *projected = getTopLevelKernelOperation(access.operation)) {
        modeledOperations.insert(projected);
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
      previousEvents = events;
    }
  }
}

static DFBQuiescenceProof
computePerNodeLifetime(DFBLogicalLifecycle &logicalDFB, LaunchNodeCoord node,
                       const HappensBeforeGraph &graph,
                       const DenseMap<Operation *, EventPair> &operationEvents,
                       const LaunchNodeDomainState &domainState) {
  DFBPerNodeLifetime &lifetime = logicalDFB.nodeLifetimes.emplace_back();
  lifetime.node = node;
  SmallVector<Operation *> reserves;
  SmallVector<Operation *> pushes;
  SmallVector<Operation *> waits;
  SmallVector<Operation *> pops;
  SmallVector<Operation *> activeOperations;
  for (auto [accessIndex, access] : llvm::enumerate(logicalDFB.accesses)) {
    if (!knownLaunchNodeDomainContains(access.launchDomain, node)) {
      continue;
    }
    lifetime.occurrenceIndices.push_back(accessIndex);
    activeOperations.push_back(access.operation);
    switch (access.effect) {
    case DFBProtocolEffect::Reserve:
      reserves.push_back(access.operation);
      break;
    case DFBProtocolEffect::Push:
      pushes.push_back(access.operation);
      break;
    case DFBProtocolEffect::Wait:
      waits.push_back(access.operation);
      break;
    case DFBProtocolEffect::Pop:
      pops.push_back(access.operation);
      break;
    case DFBProtocolEffect::OpaqueAccess:
      break;
    }
  }

  if (reserves.empty() || pushes.empty() || waits.empty() || pops.empty()) {
    return {DFBQuiescenceFailureReason::MissingProtocolEffect,
            activeOperations.empty() ? logicalDFB.declarations.front()
                                     : activeOperations.front()};
  }
  if (reserves.size() != 1 || pushes.size() != 1 || waits.size() != 1 ||
      pops.size() != 1) {
    return {DFBQuiescenceFailureReason::RepeatedProtocolEffect,
            activeOperations.front()};
  }

  Operation *reserve = reserves.front();
  Operation *push = pushes.front();
  Operation *wait = waits.front();
  Operation *pop = pops.front();
  for (Operation *protocolOperation : {reserve, push, wait, pop}) {
    std::optional<std::uint64_t> executionCount =
        getExactExecutionCountAtLaunchNode(protocolOperation, node,
                                           domainState);
    if (!executionCount || *executionCount != 1) {
      return {DFBQuiescenceFailureReason::UnsupportedControlFlow,
              protocolOperation};
    }
  }
  if (reserve->getBlock() != push->getBlock() ||
      wait->getBlock() != pop->getBlock() || !reserve->isBeforeInBlock(push) ||
      !wait->isBeforeInBlock(pop) || !releaseFollowsOwnedUses(reserve, push) ||
      !releaseFollowsOwnedUses(wait, pop)) {
    return {DFBQuiescenceFailureReason::IncompleteUseOrder, pop};
  }

  int64_t transactionTileCount = getDFBLifecycleTileCount(reserve);
  int64_t physicalTileCount =
      cast<CircularBufferType>(logicalDFB.type).getTotalElements();
  if (transactionTileCount <= 0 || physicalTileCount <= 0 ||
      physicalTileCount % transactionTileCount != 0 ||
      transactionTileCount != getDFBLifecycleTileCount(push) ||
      transactionTileCount != getDFBLifecycleTileCount(wait) ||
      transactionTileCount != getDFBLifecycleTileCount(pop)) {
    return {DFBQuiescenceFailureReason::MismatchedTransaction, reserve};
  }
  lifetime.transactionTileCount = transactionTileCount;

  std::optional<DFBPointerOwner> reserveOwner =
      getPointerOwner(reserve, node, DFBProtocolEffect::Reserve);
  std::optional<DFBPointerOwner> pushOwner =
      getPointerOwner(push, node, DFBProtocolEffect::Push);
  std::optional<DFBPointerOwner> waitOwner =
      getPointerOwner(wait, node, DFBProtocolEffect::Wait);
  std::optional<DFBPointerOwner> popOwner =
      getPointerOwner(pop, node, DFBProtocolEffect::Pop);
  if (!reserveOwner || !pushOwner || !waitOwner || !popOwner ||
      *reserveOwner != *pushOwner || *waitOwner != *popOwner) {
    return {DFBQuiescenceFailureReason::UnknownPointerOwner, reserve};
  }
  lifetime.writePointerOwner = reserveOwner;
  lifetime.readPointerOwner = waitOwner;

  std::optional<EventPair> popEvents = getProjectedEvents(pop, operationEvents);
  if (!popEvents) {
    return {DFBQuiescenceFailureReason::UnsupportedControlFlow, pop};
  }
  for (Operation *activeOperation : activeOperations) {
    std::optional<EventPair> useEvents =
        getProjectedEvents(activeOperation, operationEvents);
    if (!useEvents || (useEvents->completion != popEvents->completion &&
                       !graph.strictlyPrecedes(useEvents->completion,
                                               popEvents->completion))) {
      return {DFBQuiescenceFailureReason::IncompleteUseOrder, activeOperation};
    }
  }
  lifetime.earliestOperations =
      findMinimalOperations(activeOperations, graph, operationEvents);
  lifetime.terminalOperations = {pop};
  if (lifetime.earliestOperations.empty()) {
    return {DFBQuiescenceFailureReason::IncompleteUseOrder, pop};
  }
  return {};
}

static bool
proveOrderedBefore(const DFBPerNodeLifetime &before,
                   const DFBPerNodeLifetime &after,
                   const HappensBeforeGraph &graph,
                   const DenseMap<Operation *, EventPair> &operationEvents) {
  if (!before.quiescence.proven() || !after.quiescence.proven()) {
    return false;
  }
  return llvm::all_of(before.terminalOperations, [&](Operation *terminal) {
    std::optional<EventPair> terminalEvents =
        getProjectedEvents(terminal, operationEvents);
    return terminalEvents &&
           llvm::all_of(after.earliestOperations, [&](Operation *earliest) {
             std::optional<EventPair> earliestEvents =
                 getProjectedEvents(earliest, operationEvents);
             return earliestEvents &&
                    graph.strictlyPrecedes(terminalEvents->completion,
                                           earliestEvents->entry);
           });
  });
}

} // namespace

const DFBPerNodeLifetime *
DFBLogicalLifecycle::findNodeLifetime(LaunchNodeCoord node) const {
  auto lifetimeIt = llvm::find_if(nodeLifetimes, [&](const auto &lifetime) {
    return lifetime.node == node;
  });
  return lifetimeIt == nodeLifetimes.end() ? nullptr : &*lifetimeIt;
}

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

void DFBConcurrentKernelLivenessAnalysis::analyze(
    Operation *operation, const DFBLogicalIdentityAnalysis &identityAnalysis) {
  ModuleOp module = cast<ModuleOp>(operation);
  DFBAnalysisFailure analysisFailure;
  bool dependsOnLaunchNode = false;
  if (failed(collectLogicalDFBs(module, identityAnalysis, logicalDFBs,
                                analysisFailure, dependsOnLaunchNode))) {
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

  for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    if (logicalDFB.accesses.empty()) {
      logicalDFB.launchDomain = LaunchNodeDomain::unknown();
      continue;
    }
    for (DFBAccessOccurrence &access : logicalDFB.accesses) {
      auto domainIt = domainState.accessDomains.find(access.operation);
      if (domainIt == domainState.accessDomains.end()) {
        access.launchDomain = LaunchNodeDomain::unknown();
        access.unanalyzableDomainOperation = access.operation;
      } else {
        access.launchDomain = domainIt->second.domain;
        access.unanalyzableDomainOperation =
            domainIt->second.unanalyzableOperation;
      }
      logicalDFB.launchDomain =
          logicalDFB.launchDomain.unionWith(access.launchDomain);
    }
  }

  launchNodes.append(domainState.baseDomain.nodes.begin(),
                     domainState.baseDomain.nodes.end());
  orderedBeforeByNode.reserve(launchNodes.size());
  for (LaunchNodeCoord node : launchNodes) {
    HappensBeforeGraph graph;
    DenseMap<Operation *, EventPair> operationEvents;
    buildProgramOrderGraph(module, logicalDFBs, node, graph, operationEvents);

    for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
      SmallVector<Operation *> pushes;
      SmallVector<Operation *> waits;
      for (const DFBAccessOccurrence &access : logicalDFB.accesses) {
        if (!knownLaunchNodeDomainContains(access.launchDomain, node)) {
          continue;
        }
        if (access.effect == DFBProtocolEffect::Push) {
          pushes.push_back(access.operation);
        } else if (access.effect == DFBProtocolEffect::Wait) {
          waits.push_back(access.operation);
        }
      }
      if (pushes.size() == 1 && waits.size() == 1) {
        std::optional<EventPair> pushEvents =
            getProjectedEvents(pushes.front(), operationEvents);
        std::optional<EventPair> waitEvents =
            getProjectedEvents(waits.front(), operationEvents);
        if (pushEvents && waitEvents) {
          graph.addEdge(pushEvents->completion, waitEvents->completion);
        }
      }
    }
    graph.computeReachability();

    for (DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
      if (!knownLaunchNodeDomainContains(logicalDFB.launchDomain, node)) {
        continue;
      }
      DFBQuiescenceProof proof = computePerNodeLifetime(
          logicalDFB, node, graph, operationEvents, domainState);
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
        if (after &&
            proveOrderedBefore(*before, *after, graph, operationEvents)) {
          nodeOrdering[beforeIndex].set(afterIndex);
        }
      }
    }
    orderedBeforeByNode.push_back(std::move(nodeOrdering));
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
