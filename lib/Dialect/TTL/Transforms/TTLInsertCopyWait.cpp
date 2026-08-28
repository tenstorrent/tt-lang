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

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-copy-wait"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTCOPYWAIT
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

// Return whether the operand denotes this copy whenever the copy executes.
// Branch-local origins are correlated with the selected scf.if alternative;
// an outside origin must be preserved by every alternative.
static bool completionOperandSelectsCopy(Value completionOperand, CopyOp copy) {
  if (completionOperand == copy.getXf()) {
    return true;
  }

  if (auto cast =
          completionOperand.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1 || cast.getOutputs().size() != 1) {
      return false;
    }
    return completionOperandSelectsCopy(cast.getInputs().front(), copy);
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
    return completionOperandSelectsCopy(thenValue, copy);
  }
  if (ifOp.getElseRegion().isAncestor(copyRegion)) {
    return completionOperandSelectsCopy(elseValue, copy);
  }

  return completionOperandSelectsCopy(thenValue, copy) &&
         completionOperandSelectsCopy(elseValue, copy);
}

class CopyCompletionIndex {
public:
  CopyCompletionIndex(func::FuncOp func, ValueOriginAnalysis &valueOrigins) {
    func.walk([&](CopyOp copy) { copies.push_back(copy.getOperation()); });

    auto recordCompletedOrigins = [&](Operation *completion,
                                      Value completionOperand) {
      for (Value origin : valueOrigins.getOrigins(completionOperand)) {
        auto copy = origin.getDefiningOp<CopyOp>();
        if (copy && completionOperandSelectsCopy(completionOperand, copy)) {
          completedCopies[completion].push_back(copy.getOperation());
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

private:
  SmallVector<Operation *> copies;
  DenseMap<Operation *, SmallVector<Operation *>> completedCopies;
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

// Propagate possible outstanding-copy state through region and block control
// flow. A copy is complete only when no reachable function exit retains it.
class CopyCompletionDataFlow
    : public dataflow::DenseForwardDataFlowAnalysis<CopyCompletionLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyCompletionDataFlow)

  CopyCompletionDataFlow(DataFlowSolver &solver, func::FuncOp func,
                         const CopyCompletionIndex &completionIndex)
      : DenseForwardDataFlowAnalysis(solver), func(func),
        completionIndex(completionIndex) {}

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

private:
  func::FuncOp func;
  const CopyCompletionIndex &completionIndex;
};

static FailureOr<DenseSet<Operation *>>
findCopiesCompletedOnAllContinuations(func::FuncOp func,
                                      const CopyCompletionIndex &index) {
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<CopyCompletionDataFlow>(func, index);
  if (failed(solver.initializeAndRun(func))) {
    return failure();
  }

  DenseSet<Operation *> coveredCopies(index.getCopies().begin(),
                                      index.getCopies().end());
  bool foundReachableReturn = false;
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
      coveredCopies.clear();
      return;
    }
    for (Operation *copy : lattice->getPossiblyOutstandingCopies()) {
      coveredCopies.erase(copy);
    }
  });
  if (!foundReachableReturn) {
    coveredCopies.clear();
  }
  return coveredCopies;
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
               completionOperandSelectsCopy(request, copy)});
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

static Operation *findAncestorInBlock(Operation *operation, Block *block) {
  for (Operation *ancestor = operation; ancestor;
       ancestor = ancestor->getParentOp()) {
    if (ancestor->getBlock() == block) {
      return ancestor;
    }
  }
  return nullptr;
}

static std::optional<CopyWaitPlan> planAfterSelectionObservations(
    CopyOp copy, ArrayRef<ReceiveSelectionObservation> observations,
    const DominanceInfo &dominanceInfo,
    const PostDominanceInfo &postDominanceInfo) {
  Block *copyBlock = copy->getBlock();
  bool copyBlockContainsEveryObservation = llvm::all_of(
      observations, [&](const ReceiveSelectionObservation &observation) {
        Operation *ancestor =
            findAncestorInBlock(observation.waitAny, copyBlock);
        return ancestor && copy->isBeforeInBlock(ancestor);
      });
  if (copyBlockContainsEveryObservation) {
    return CopyWaitPlan{copy.getXf(), nullptr, copyBlock->getTerminator(),
                        copy.getLoc()};
  }

  for (const ReceiveSelectionObservation &candidate : observations) {
    if (!candidate.requestSelectsCopy) {
      continue;
    }
    Operation *candidateWaitAny = candidate.waitAny;
    if (!postDominanceInfo.postDominates(candidateWaitAny,
                                         copy.getOperation())) {
      continue;
    }
    bool followsEveryObservation = llvm::all_of(
        observations, [&](const ReceiveSelectionObservation &observation) {
          return observation.waitAny == candidate.waitAny ||
                 postDominanceInfo.postDominates(candidateWaitAny,
                                                 observation.waitAny);
        });
    if (followsEveryObservation) {
      return CopyWaitPlan{candidate.request, nullptr,
                          candidateWaitAny->getBlock()->getTerminator(),
                          copy.getLoc()};
    }
  }

  Block *commonPostDominator = copyBlock;
  for (const ReceiveSelectionObservation &observation : observations) {
    commonPostDominator = postDominanceInfo.findNearestCommonDominator(
        commonPostDominator, observation.waitAny->getBlock());
    if (!commonPostDominator) {
      return std::nullopt;
    }
  }
  if (commonPostDominator == copyBlock || commonPostDominator->empty()) {
    return std::nullopt;
  }

  Operation *terminator = commonPostDominator->getTerminator();
  if (!dominanceInfo.dominates(copy.getXf(), terminator)) {
    return std::nullopt;
  }
  return CopyWaitPlan{copy.getXf(), nullptr, terminator, copy.getLoc()};
}

static bool hasEquivalentPlan(ArrayRef<CopyWaitPlan> plans,
                              const CopyWaitPlan &candidate) {
  return llvm::any_of(plans, [&](const CopyWaitPlan &plan) {
    return plan.handle == candidate.handle &&
           plan.insertAfter == candidate.insertAfter &&
           plan.insertBefore == candidate.insertBefore;
  });
}

struct TTLInsertCopyWaitPass
    : public impl::TTLInsertCopyWaitBase<TTLInsertCopyWaitPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    ValueOriginAnalysis valueOrigins(func);
    CopyCompletionIndex completionIndex(func, valueOrigins);
    FailureOr<DenseSet<Operation *>> maybeCoveredCopies =
        findCopiesCompletedOnAllContinuations(func, completionIndex);
    if (failed(maybeCoveredCopies)) {
      func.emitOpError("failed to analyze copy completion coverage");
      signalPassFailure();
      return;
    }

    DenseMap<Operation *, SmallVector<ReceiveSelectionObservation>>
        observations = indexReceiveSelectionObservations(func, valueOrigins);
    DominanceInfo dominanceInfo(func);
    PostDominanceInfo postDominanceInfo(func);
    SmallVector<CopyWaitPlan> plans;
    for (Operation *copyOperation : completionIndex.getCopies()) {
      if (maybeCoveredCopies->contains(copyOperation)) {
        continue;
      }
      auto copy = cast<CopyOp>(copyOperation);
      auto found = observations.find(copyOperation);
      if (found == observations.end()) {
        plans.push_back({copy.getXf(), copyOperation, nullptr, copy.getLoc()});
        continue;
      }

      std::optional<CopyWaitPlan> plan = planAfterSelectionObservations(
          copy, found->second, dominanceInfo, postDominanceInfo);
      if (!plan) {
        bool plannedAtReturn = false;
        func.walk([&](func::ReturnOp returnOp) {
          if (!dominanceInfo.dominates(copy.getXf(), returnOp)) {
            return;
          }
          plannedAtReturn = true;
          CopyWaitPlan returnPlan{copy.getXf(), nullptr,
                                  returnOp.getOperation(), copy.getLoc()};
          if (!hasEquivalentPlan(plans, returnPlan)) {
            plans.push_back(returnPlan);
          }
        });
        if (!plannedAtReturn) {
          copy.emitOpError(
              "cannot place an implicit wait after every wait_any "
              "observation; add explicit waits after the final selection on "
              "each continuation");
          signalPassFailure();
          return;
        }
        continue;
      }
      if (!hasEquivalentPlan(plans, *plan)) {
        plans.push_back(*plan);
      }
    }

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
};

} // namespace

} // namespace mlir::tt::ttl
