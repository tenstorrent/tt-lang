// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Copy Wait
//===----------------------------------------------------------------------===//
//
// Completes every ttl.copy without changing the result of a readiness
// selection.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/SliceWalk.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-copy-wait"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTCOPYWAIT
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static bool completionOperandSelectsCopyImpl(Value completionOperand,
                                             CopyOp copy,
                                             DenseSet<Value> &activeValues) {
  if (completionOperand == copy.getXf()) {
    return true;
  }
  if (!activeValues.insert(completionOperand).second) {
    return false;
  }
  auto removeActiveValue =
      llvm::scope_exit([&] { activeValues.erase(completionOperand); });

  if (auto cast =
          completionOperand.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1 || cast.getOutputs().size() != 1) {
      return false;
    }
    return completionOperandSelectsCopyImpl(cast.getInputs().front(), copy,
                                            activeValues);
  }

  if (auto blockArgument = dyn_cast<BlockArgument>(completionOperand)) {
    Block *argumentBlock = blockArgument.getOwner();
    if (argumentBlock->isEntryBlock() ||
        argumentBlock->getParent() != copy->getParentRegion()) {
      return false;
    }
    std::optional<SmallVector<Value>> predecessors =
        getControlFlowPredecessors(completionOperand);
    if (!predecessors) {
      return false;
    }
    SmallVector<Block *> predecessorBlocks(argumentBlock->getPredecessors());
    if (predecessorBlocks.size() != predecessors->size()) {
      return false;
    }

    bool hasContinuationFromCopy = false;
    for (auto [predecessorBlock, predecessorValue] :
         llvm::zip(predecessorBlocks, *predecessors)) {
      if (copy->getBlock() != predecessorBlock) {
        llvm::SmallPtrSet<Block *, 16> excludedBlocks;
        excludedBlocks.insert(argumentBlock);
        if (!copy->getBlock()->isReachable(predecessorBlock,
                                           std::move(excludedBlocks))) {
          continue;
        }
      }
      hasContinuationFromCopy = true;
      if (!completionOperandSelectsCopyImpl(predecessorValue, copy,
                                            activeValues)) {
        return false;
      }
    }
    return hasContinuationFromCopy;
  }

  auto result = dyn_cast<OpResult>(completionOperand);
  if (!result) {
    return false;
  }
  auto ifOp = dyn_cast<scf::IfOp>(result.getOwner());
  if (!ifOp) {
    return false;
  }
  if (ifOp.getElseRegion().empty()) {
    return false;
  }

  auto thenYield =
      dyn_cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
  auto elseYield =
      dyn_cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
  unsigned resultIndex = result.getResultNumber();
  if (!thenYield || !elseYield ||
      resultIndex >= thenYield.getResults().size() ||
      resultIndex >= elseYield.getResults().size()) {
    return false;
  }

  Value thenValue = thenYield.getResults()[resultIndex];
  Value elseValue = elseYield.getResults()[resultIndex];
  Region *copyRegion = copy->getParentRegion();
  if (ifOp.getThenRegion().isAncestor(copyRegion)) {
    return completionOperandSelectsCopyImpl(thenValue, copy, activeValues);
  }
  if (ifOp.getElseRegion().isAncestor(copyRegion)) {
    return completionOperandSelectsCopyImpl(elseValue, copy, activeValues);
  }

  return completionOperandSelectsCopyImpl(thenValue, copy, activeValues) &&
         completionOperandSelectsCopyImpl(elseValue, copy, activeValues);
}

// Return whether the operand denotes this copy on every continuation from the
// copy to the operand.
static bool completionOperandSelectsCopy(Value completionOperand, CopyOp copy) {
  DenseSet<Value> activeValues;
  return completionOperandSelectsCopyImpl(completionOperand, copy,
                                          activeValues);
}

// Every copy execution must reach its completion before the same copy
// operation can execute again.
static bool preservesCopyExecutionCorrespondence(CopyOp copy,
                                                 Operation *completion) {
  if (copy->getParentRegion() != completion->getParentRegion() ||
      copy->getBlock() == completion->getBlock()) {
    return true;
  }

  Block *copyBlock = copy->getBlock();
  Block *completionBlock = completion->getBlock();
  for (Block *successor : copyBlock->getSuccessors()) {
    if (successor == completionBlock) {
      continue;
    }
    if (successor == copyBlock) {
      return false;
    }
    llvm::SmallPtrSet<Block *, 16> excludedBlocks;
    excludedBlocks.insert(completionBlock);
    if (successor->isReachable(copyBlock, std::move(excludedBlocks))) {
      return false;
    }
  }
  return true;
}

class CopyCompletionIndex {
public:
  CopyCompletionIndex(func::FuncOp func, ValueOriginAnalysis &valueOrigins) {
    func.walk([&](CopyOp copy) { copies.push_back(copy.getOperation()); });

    auto recordCompletedOrigins = [&](Operation *completion,
                                      Value completionOperand) {
      for (Value origin : valueOrigins.getOrigins(completionOperand)) {
        auto copy = origin.getDefiningOp<CopyOp>();
        if (copy && completionOperandSelectsCopy(completionOperand, copy) &&
            preservesCopyExecutionCorrespondence(copy, completion)) {
          completedCopies[completion].push_back(copy.getOperation());
          completionsByCopy[copy.getOperation()].push_back(completion);
        }
      }
    };

    func.walk([&](Operation *operation) {
      if (auto wait = dyn_cast<WaitOp>(operation)) {
        recordCompletedOrigins(wait, wait.getXf());
        return;
      }
      auto waitAny = dyn_cast<WaitAnyOp>(operation);
      if (waitAny && waitAny.getRequests().size() == 1) {
        recordCompletedOrigins(waitAny, waitAny.getRequests().front());
      }
    });
  }

  ArrayRef<Operation *> getCopies() const { return copies; }

  ArrayRef<Operation *> getCompletedCopies(Operation *operation) const {
    auto found = completedCopies.find(operation);
    if (found == completedCopies.end()) {
      return {};
    }
    return found->second;
  }

  ArrayRef<Operation *> getCompletions(Operation *copy) const {
    auto found = completionsByCopy.find(copy);
    if (found == completionsByCopy.end()) {
      return {};
    }
    return found->second;
  }

private:
  SmallVector<Operation *> copies;
  DenseMap<Operation *, SmallVector<Operation *>> completedCopies;
  DenseMap<Operation *, SmallVector<Operation *>> completionsByCopy;
};

class CopyCompletionLattice : public dataflow::AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyCompletionLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const dataflow::AbstractDenseLattice &rhs) override {
    const auto &other = static_cast<const CopyCompletionLattice &>(rhs);
    return join(other.initialized, other.possiblyOutstandingCopies);
  }

  ChangeResult join(bool otherInitialized,
                    const DenseSet<Operation *> &otherOutstandingCopies) {
    ChangeResult changed = ChangeResult::NoChange;
    if (otherInitialized && !initialized) {
      initialized = true;
      changed |= ChangeResult::Change;
    }
    for (Operation *copy : otherOutstandingCopies) {
      if (possiblyOutstandingCopies.insert(copy).second) {
        changed |= ChangeResult::Change;
      }
    }
    return changed;
  }

  ChangeResult initialize(ArrayRef<Operation *> outstandingCopies = {}) {
    DenseSet<Operation *> outstanding(outstandingCopies.begin(),
                                      outstandingCopies.end());
    return join(/*otherInitialized=*/true, outstanding);
  }

  bool isInitialized() const { return initialized; }

  const DenseSet<Operation *> &getPossiblyOutstandingCopies() const {
    return possiblyOutstandingCopies;
  }

  void print(raw_ostream &output) const override {
    output << (initialized ? "initialized" : "unreachable");
    for (Operation *copy : possiblyOutstandingCopies) {
      output << "\n  outstanding: " << copy->getName();
    }
  }

private:
  bool initialized = false;
  DenseSet<Operation *> possiblyOutstandingCopies;
};

static std::optional<LaunchNodeDomain>
getPipeRoleRegionDomain(Operation *operation,
                        const LaunchNodeDomain &baseDomain) {
  if (auto ifSrc = dyn_cast<IfSrcOp>(operation)) {
    return getPipeSourceLaunchNodeDomain(
        cast<PipeType>(ifSrc.getPipe().getType()));
  }
  if (auto ifDst = dyn_cast<IfDstOp>(operation)) {
    return getPipeDestinationLaunchNodeDomain(
        cast<PipeType>(ifDst.getPipe().getType()), baseDomain);
  }
  if (auto foreachSrc = dyn_cast<PipeNetForeachSrcOp>(operation)) {
    return getPipeRecordsRoleLaunchNodeDomain(foreachSrc.getRecords(),
                                              PipeRole::Source);
  }
  if (auto foreachDst = dyn_cast<PipeNetForeachDstOp>(operation)) {
    return getPipeRecordsRoleLaunchNodeDomain(foreachDst.getRecords(),
                                              PipeRole::Destination);
  }
  return std::nullopt;
}

static std::optional<LaunchNodeDomain>
getEnclosingPipeRoleDomain(Operation *operation,
                           const LaunchNodeDomain &baseDomain) {
  std::optional<LaunchNodeDomain> domain;
  for (Operation *ancestor = operation->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    std::optional<LaunchNodeDomain> ancestorDomain =
        getPipeRoleRegionDomain(ancestor, baseDomain);
    if (!ancestorDomain) {
      continue;
    }
    domain = domain ? domain->intersectWith(*ancestorDomain) : ancestorDomain;
  }
  return domain;
}

// Propagate possible outstanding-copy state through region and block control
// flow. A copy is complete only when no reachable function exit retains it.
class CopyCompletionDataFlow
    : public dataflow::DenseForwardDataFlowAnalysis<CopyCompletionLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyCompletionDataFlow)

  CopyCompletionDataFlow(DataFlowSolver &solver, func::FuncOp func,
                         const CopyCompletionIndex &completionIndex,
                         const LaunchNodeDomainState &launchNodeDomains)
      : DenseForwardDataFlowAnalysis(solver), func(func),
        completionIndex(completionIndex), launchNodeDomains(launchNodeDomains) {
  }

  void setToEntryState(CopyCompletionLattice *lattice) override {
    ProgramPoint *point = lattice->getAnchor().dyn_cast<ProgramPoint *>();
    bool isFunctionEntry = point &&
                           point->getBlock() == &func.getBody().front() &&
                           point->isBlockStart();
    ArrayRef<Operation *> outstandingCopies =
        isFunctionEntry ? ArrayRef<Operation *>() : completionIndex.getCopies();
    propagateIfChanged(lattice, lattice->initialize(outstandingCopies));
  }

  LogicalResult visitOperation(Operation *operation,
                               const CopyCompletionLattice &before,
                               CopyCompletionLattice *after) override {
    DenseSet<Operation *> transferred(before.getPossiblyOutstandingCopies());
    if (isa<CopyOp>(operation)) {
      transferred.insert(operation);
    }
    for (Operation *completedCopy :
         completionIndex.getCompletedCopies(operation)) {
      transferred.erase(completedCopy);
    }
    propagateIfChanged(after, after->join(before.isInitialized(), transferred));
    return success();
  }

  void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo, const CopyCompletionLattice &before,
      CopyCompletionLattice *after) override {
    if (regionFrom || regionTo) {
      propagateIfChanged(after, after->join(before));
      return;
    }

    std::optional<LaunchNodeDomain> regionDomain = getPipeRoleRegionDomain(
        branch.getOperation(), launchNodeDomains.baseDomain);
    if (!regionDomain) {
      propagateIfChanged(after, after->join(before));
      return;
    }

    // The parent-to-parent edge represents nodes that do not enter the role
    // region. A copy restricted to that region cannot be outstanding there.
    DenseSet<Operation *> transferred(before.getPossiblyOutstandingCopies());
    for (Operation *copyOperation : before.getPossiblyOutstandingCopies()) {
      std::optional<LaunchNodeDomain> copyDomain = getEnclosingPipeRoleDomain(
          copyOperation, launchNodeDomains.baseDomain);
      if (copyDomain && copyDomain->isSubsetOf(*regionDomain)) {
        transferred.erase(copyOperation);
      }
    }
    propagateIfChanged(after, after->join(before.isInitialized(), transferred));
  }

private:
  func::FuncOp func;
  const CopyCompletionIndex &completionIndex;
  const LaunchNodeDomainState &launchNodeDomains;
};

struct CopyCompletionResult {
  struct ReturnState {
    Operation *returnOperation;
    DenseSet<Operation *> outstandingCopies;
  };

  DenseSet<Operation *> completedCopies;
  SmallVector<ReturnState> returnStates;
};

static FailureOr<CopyCompletionResult>
findCopiesCompletedOnAllContinuations(func::FuncOp func,
                                      const CopyCompletionIndex &index,
                                      const LaunchNodeDomainState &domains) {
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<CopyCompletionDataFlow>(func, index, domains);
  if (failed(solver.initializeAndRun(func))) {
    return failure();
  }

  CopyCompletionResult result;
  result.completedCopies.insert(index.getCopies().begin(),
                                index.getCopies().end());
  bool foundReachableReturn = false;
  bool missingReturnState = false;
  func.walk([&](func::ReturnOp returnOp) {
    ProgramPoint *blockStart =
        solver.getProgramPointBefore(returnOp->getBlock());
    const auto *executable =
        solver.lookupState<dataflow::Executable>(blockStart);
    if (executable && !executable->isLive()) {
      return;
    }
    foundReachableReturn = true;
    ProgramPoint *beforeReturn = solver.getProgramPointBefore(returnOp);
    const auto *lattice =
        solver.lookupState<CopyCompletionLattice>(beforeReturn);
    if (!lattice || !lattice->isInitialized()) {
      missingReturnState = true;
      return;
    }
    CopyCompletionResult::ReturnState returnState{returnOp, {}};
    for (Operation *copy : lattice->getPossiblyOutstandingCopies()) {
      returnState.outstandingCopies.insert(copy);
      result.completedCopies.erase(copy);
    }
    result.returnStates.push_back(std::move(returnState));
  });
  if (missingReturnState) {
    return failure();
  }
  if (!foundReachableReturn) {
    result.completedCopies.clear();
  }
  return result;
}

struct ReceiveSelectionObservation {
  Operation *waitAny;
  Value request;
  bool requestSelectsCopy;
};

static DenseMap<Operation *, SmallVector<ReceiveSelectionObservation>>
indexReceiveSelectionObservations(func::FuncOp func,
                                  ValueOriginAnalysis &valueOrigins) {
  DenseMap<Operation *, SmallVector<ReceiveSelectionObservation>> observations;
  func.walk([&](WaitAnyOp waitAny) {
    if (waitAny.getRequests().size() == 1) {
      return;
    }
    for (Value request : waitAny.getRequests()) {
      for (Value origin : valueOrigins.getOrigins(request)) {
        auto copy = origin.getDefiningOp<CopyOp>();
        if (copy) {
          observations[copy.getOperation()].push_back(
              {waitAny.getOperation(), request,
               completionOperandSelectsCopy(request, copy) &&
                   preservesCopyExecutionCorrespondence(copy, waitAny)});
        }
      }
    }
  });
  return observations;
}

struct CopyWaitPlan {
  Value handle;
  Operation *insertAfter = nullptr;
  Operation *insertBefore = nullptr;
  Location location;
};

// Same-block program order completes the copy before any block exit or
// backedge can execute.
static bool hasGuaranteedSameBlockWait(CopyOp copy) {
  return llvm::any_of(copy.getXf().getUsers(), [&](Operation *user) {
    auto wait = dyn_cast<WaitOp>(user);
    return wait && wait->getBlock() == copy->getBlock() &&
           copy->isBeforeInBlock(wait);
  });
}

static FailureOr<SmallVector<CopyWaitPlan>>
tryPlanLocalCopyWaits(func::FuncOp func) {
  SmallVector<CopyWaitPlan> plans;
  WalkResult result = func.walk([&](CopyOp copy) {
    if (hasGuaranteedSameBlockWait(copy)) {
      return WalkResult::advance();
    }
    if (!isa<ReceiveRequestType>(copy.getXf().getType()) &&
        copy.getXf().use_empty()) {
      plans.push_back(
          {copy.getXf(), copy.getOperation(), nullptr, copy.getLoc()});
      return WalkResult::advance();
    }
    return WalkResult::interrupt();
  });
  if (result.wasInterrupted()) {
    return failure();
  }
  return plans;
}

static FailureOr<Operation *>
getImplicitWaitAnchor(CopyOp copy, const CopyCompletionIndex &index,
                      const LaunchNodeDomain &baseDomain) {
  if (!isa<ReceiveRequestType>(copy.getXf().getType())) {
    return copy.getOperation();
  }

  // Receive completion uses a monotonic threshold. A fallback after nested
  // role regions covers entered and bypassed nodes without preceding sends.
  llvm::SetVector<Operation *> destinationDependentOperations;
  getForwardSlice(copy.getDst(), &destinationDependentOperations);
  Operation *latestAnchor = copy.getOperation();
  for (Operation *completion : index.getCompletions(copy.getOperation())) {
    Operation *ancestor = completion;
    while (ancestor && ancestor->getParentRegion() != copy->getParentRegion()) {
      ancestor = ancestor->getParentOp();
    }
    if (!ancestor || ancestor->getBlock() != copy->getBlock() ||
        !getPipeRoleRegionDomain(ancestor, baseDomain) ||
        !copy->isBeforeInBlock(ancestor)) {
      continue;
    }
    for (Operation *crossed = copy->getNextNode(); crossed != ancestor;
         crossed = crossed->getNextNode()) {
      bool dependsOnDestination = llvm::any_of(
          destinationDependentOperations,
          [crossed](Operation *destinationDependentOperation) {
            return crossed == destinationDependentOperation ||
                   crossed->isAncestor(destinationDependentOperation);
          });
      if (isMemoryEffectFree(crossed) && !dependsOnDestination) {
        continue;
      }
      InFlightDiagnostic diagnostic = copy.emitOpError(
          "cannot place an implicit receive wait across an operation that "
          "may access its destination; add an explicit wait after the "
          "matching send and before the destination is used");
      diagnostic.attachNote(crossed->getLoc())
          << "operation prevents safe implicit wait placement";
      return failure();
    }
    if (latestAnchor == copy.getOperation() ||
        latestAnchor->isBeforeInBlock(ancestor)) {
      latestAnchor = ancestor;
    }
  }
  return latestAnchor;
}

static bool hasEquivalentPlan(ArrayRef<CopyWaitPlan> plans,
                              const CopyWaitPlan &candidate) {
  return llvm::any_of(plans, [&](const CopyWaitPlan &plan) {
    return plan.handle == candidate.handle &&
           plan.insertAfter == candidate.insertAfter &&
           plan.insertBefore == candidate.insertBefore;
  });
}

static void applyCopyWaitPlans(func::FuncOp func,
                               ArrayRef<CopyWaitPlan> plans) {
  OpBuilder builder(func.getContext());
  DenseMap<Operation *, Operation *> lastInsertionAfter;
  DenseMap<Operation *, Operation *> lastInsertionBefore;
  for (const CopyWaitPlan &plan : plans) {
    if (plan.insertAfter) {
      Operation *anchor = plan.insertAfter;
      auto previous = lastInsertionAfter.find(anchor);
      if (previous != lastInsertionAfter.end()) {
        builder.setInsertionPointAfter(previous->second);
      } else {
        builder.setInsertionPointAfter(anchor);
      }
      WaitOp wait = WaitOp::create(builder, plan.location, plan.handle);
      lastInsertionAfter[anchor] = wait.getOperation();
      continue;
    }

    assert(plan.insertBefore && "copy wait plan requires an anchor");
    auto previous = lastInsertionBefore.find(plan.insertBefore);
    if (previous != lastInsertionBefore.end()) {
      builder.setInsertionPointAfter(previous->second);
    } else {
      builder.setInsertionPoint(plan.insertBefore);
    }
    WaitOp wait = WaitOp::create(builder, plan.location, plan.handle);
    lastInsertionBefore[plan.insertBefore] = wait.getOperation();
  }
}

struct TTLInsertCopyWaitPass
    : public impl::TTLInsertCopyWaitBase<TTLInsertCopyWaitPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    FailureOr<SmallVector<CopyWaitPlan>> localPlans =
        tryPlanLocalCopyWaits(func);
    if (succeeded(localPlans)) {
      applyCopyWaitPlans(func, *localPlans);
      return;
    }

    LaunchNodeDomainState launchNodeDomains;
    launchNodeDomains.initialize(func->getParentOfType<ModuleOp>());
    ValueOriginAnalysis valueOrigins(func);
    CopyCompletionIndex completionIndex(func, valueOrigins);
    FailureOr<CopyCompletionResult> maybeCompletionResult =
        findCopiesCompletedOnAllContinuations(func, completionIndex,
                                              launchNodeDomains);
    if (failed(maybeCompletionResult)) {
      func.emitOpError("failed to analyze copy completion coverage");
      signalPassFailure();
      return;
    }

    DenseMap<Operation *, SmallVector<ReceiveSelectionObservation>>
        observations = indexReceiveSelectionObservations(func, valueOrigins);
    DominanceInfo dominanceInfo(func);
    SmallVector<CopyWaitPlan> plans;
    for (Operation *copyOperation : completionIndex.getCopies()) {
      if (maybeCompletionResult->completedCopies.contains(copyOperation)) {
        continue;
      }
      auto copy = cast<CopyOp>(copyOperation);
      auto found = observations.find(copyOperation);
      if (found == observations.end()) {
        FailureOr<Operation *> maybeAnchor = getImplicitWaitAnchor(
            copy, completionIndex, launchNodeDomains.baseDomain);
        if (failed(maybeAnchor)) {
          signalPassFailure();
          return;
        }
        plans.push_back({copy.getXf(), *maybeAnchor, nullptr, copy.getLoc()});
        continue;
      }

      bool hasUnplannedReturn = false;
      for (const CopyCompletionResult::ReturnState &returnState :
           maybeCompletionResult->returnStates) {
        if (!returnState.outstandingCopies.contains(copyOperation)) {
          continue;
        }
        Operation *returnOperation = returnState.returnOperation;
        Value completionHandle;
        for (const ReceiveSelectionObservation &observation : found->second) {
          if (observation.requestSelectsCopy &&
              dominanceInfo.dominates(observation.request, returnOperation)) {
            completionHandle = observation.request;
            break;
          }
        }
        if (!completionHandle &&
            preservesCopyExecutionCorrespondence(copy, returnOperation) &&
            dominanceInfo.dominates(copy.getXf(), returnOperation)) {
          completionHandle = copy.getXf();
        }
        if (!completionHandle) {
          hasUnplannedReturn = true;
          continue;
        }
        CopyWaitPlan returnPlan{completionHandle, nullptr, returnOperation,
                                copy.getLoc()};
        if (!hasEquivalentPlan(plans, returnPlan)) {
          plans.push_back(returnPlan);
        }
      }
      if (hasUnplannedReturn) {
        copy.emitOpError(
            "cannot place an implicit wait after every wait_any observation; "
            "add explicit waits after the final selection on each "
            "continuation");
        signalPassFailure();
        return;
      }
    }

    applyCopyWaitPlans(func, plans);
  }
};

} // namespace

} // namespace mlir::tt::ttl
