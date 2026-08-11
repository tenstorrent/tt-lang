// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert CB Sync
//===----------------------------------------------------------------------===//
//
// Auto-inserts a cb_push / cb_pop after each cb_reserve / cb_wait whose
// matching release is absent in the input IR, placing each release after
// the last use of the acquired slot so the slot is not recycled before
// the consumer is done with it. "Last use" classification handles two
// different valid IR situations -- direct-CB uses and tensor-SSA uses --
// under different rules; see `docs/development/DFBManagement.md` for the
// rules and correctness argument.
//
//===----------------------------------------------------------------------===//

#include "DFBAcquireReleaseAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-cb-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTCBSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct MissingReleasePlan {
  Operation *acquire = nullptr;
  Operation *insertionAfter = nullptr;
  Value dfb;
  SmallVector<Operation *> nestedConcreteReleases;
};

struct WaitAnyCandidate {
  WaitAnyOp waitAny;
  unsigned candidateIndex = 0;
};

struct ConditionalReceiveReleasePlan {
  DenseSet<Operation *> reserves;
  SmallVector<Operation *> reserveOrder;
  DenseMap<Operation *, SmallVector<WaitAnyCandidate>> candidatesByReserve;
  DenseMap<Operation *, SmallVector<WaitOp>> exactWaitsByReserve;
};

template <typename ConcreteReleaseOp>
static PlanningResult<SmallVector<MissingReleasePlan>>
planMissingReleases(ArrayRef<Operation *> acquires,
                    ArrayRef<Operation *> releases,
                    const DenseSet<Operation *> &explicitReleaseAcquires,
                    StringRef effectName) {
  SmallVector<MissingReleasePlan> plans;

  for (Operation *acquire : acquires) {
    if (explicitReleaseAcquires.contains(acquire)) {
      continue;
    }
    DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);

    // Tensor SSA uses can keep this acquired slot live past the next same-DFB
    // acquire. An existing release after that final use still belongs to this
    // acquire, so pass the final use into the release search.
    Operation *last = findLastDFBAcquireOwnedUse(interval);
    DFBReleaseSearch releaseSearch =
        findOwnedDFBReleases(interval, last, releases);

    if (releaseSearch.hasSameLevelRelease()) {
      continue;
    }

    for (Operation *nestedRelease : releaseSearch.nestedReleases) {
      if (!isa<ConcreteReleaseOp>(nestedRelease)) {
        return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
            nestedRelease,
            ("external DFB " + effectName +
             " effect must be in the same block as its acquisition")
                .str());
      }
    }

    plans.push_back(
        {acquire, last, interval.dfb, std::move(releaseSearch.nestedReleases)});
  }
  return PlanningResult<SmallVector<MissingReleasePlan>>::planned(
      std::move(plans));
}

template <typename CreateReleaseFn>
static void applyMissingReleases(ArrayRef<MissingReleasePlan> plans,
                                 DenseSet<Operation *> &erased,
                                 OpBuilder &builder,
                                 CreateReleaseFn createRelease) {
  for (const MissingReleasePlan &plan : plans) {
    for (Operation *nestedRelease : plan.nestedConcreteReleases) {
      if (erased.insert(nestedRelease).second) {
        nestedRelease->erase();
      }
    }
    builder.setInsertionPointAfter(plan.insertionAfter);
    createRelease(builder, plan.acquire->getLoc(), plan.dfb);
  }
}

static FailureOr<ConditionalReceiveReleasePlan>
buildConditionalReceiveReleasePlan(func::FuncOp func,
                                   ValueOriginAnalysis &valueOrigins) {
  ConditionalReceiveReleasePlan plan;
  WalkResult result = func.walk([&](WaitAnyOp waitAny) {
    for (auto [candidateIndex, request] :
         llvm::enumerate(waitAny.getRequests())) {
      FailureOr<SmallVector<CopyOp>> receiveCopies =
          findPipeReceiveCopies(valueOrigins, request);
      if (failed(receiveCopies)) {
        waitAny.emitOpError()
            << "requires every request origin to be a pipe receive ttl.copy";
        return WalkResult::interrupt();
      }
      for (CopyOp receiveCopy : *receiveCopies) {
        CBReserveOp reserve = findCBReserveForPipeReceive(receiveCopy.getDst());
        assert(reserve && "pipe receive verifier requires a DFB reservation");
        Operation *reserveOperation = reserve.getOperation();
        if (plan.reserves.insert(reserveOperation).second) {
          plan.reserveOrder.push_back(reserveOperation);
        }
        plan.candidatesByReserve[reserveOperation].push_back(
            WaitAnyCandidate{waitAny, static_cast<unsigned>(candidateIndex)});
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }

  WalkResult exactWaitResult =
      func.walk([&](WaitOp wait) {
        if (!isa<ReceiveRequestType>(wait.getXf().getType())) {
          return WalkResult::advance();
        }
        FailureOr<SmallVector<CopyOp>> receiveCopies =
            findPipeReceiveCopies(valueOrigins, wait.getXf());
        if (failed(receiveCopies)) {
          wait.emitOpError()
              << "requires every request origin to be a pipe receive ttl.copy";
          return WalkResult::interrupt();
        }
        for (CopyOp receiveCopy : *receiveCopies) {
          if (CBReserveOp reserve =
                  findCBReserveForPipeReceive(receiveCopy.getDst())) {
            plan.exactWaitsByReserve[reserve.getOperation()].push_back(wait);
          }
        }
        return WalkResult::advance();
      });
  if (exactWaitResult.wasInterrupted()) {
    return failure();
  }
  return plan;
}

static LogicalResult
validateConditionalReceiveReleases(ArrayRef<Operation *> pushes,
                                   const ConditionalReceiveReleasePlan &plan,
                                   const DFBAcquireReleaseIndex &lifecycles,
                                   const DominanceInfo &dominanceInfo) {
  DenseSet<Operation *> publishedReserves;
  auto isOrderedBefore = [&](Operation *before, Operation *after) {
    return dominanceInfo.properlyDominates(before, after);
  };
  for (Operation *push : pushes) {
    for (Operation *reserve : plan.reserveOrder) {
      auto candidates = plan.candidatesByReserve.find(reserve);
      assert(candidates != plan.candidatesByReserve.end() &&
             "planned wait-any reserve must have a candidate");
      Value dfb = getDFBAcquireDFB(reserve);
      if (getDFBReleaseDFB(push) != dfb) {
        continue;
      }
      for (WaitAnyCandidate candidate : candidates->second) {
        bool sharesStream =
            llvm::any_of(plan.reserveOrder, [&](Operation *otherReserve) {
              if (otherReserve == reserve ||
                  getDFBAcquireDFB(otherReserve) != dfb) {
                return false;
              }
              auto otherCandidates =
                  plan.candidatesByReserve.find(otherReserve);
              assert(otherCandidates != plan.candidatesByReserve.end() &&
                     "planned wait-any reserve must have a candidate");
              return llvm::any_of(
                  otherCandidates->second, [&](WaitAnyCandidate other) {
                    return other.waitAny == candidate.waitAny &&
                           other.candidateIndex != candidate.candidateIndex;
                  });
            });
        if (sharesStream && isInReadyReceiveSelectionRegion(
                                push, candidate.waitAny,
                                static_cast<int64_t>(candidate.candidateIndex),
                                isOrderedBefore)) {
          push->emitError(
              "wait-any candidates published according to selection must use "
              "separate destination dataflow buffer streams");
          return failure();
        }
      }
    }

    const DFBReleaseOwnership &ownership = lifecycles.getReleaseOwnership(push);
    ArrayRef<Operation *> owners = ownership.candidateOwners;
    if (ownership.ownership == DFBReleaseOwnershipKind::Unresolved) {
      ArrayRef<Operation *> intervalOwners =
          lifecycles.getReleaseIntervalOwners(push);
      if (!intervalOwners.empty()) {
        owners = intervalOwners;
      }
    }
    for (Operation *reserve : owners) {
      if (!plan.reserves.contains(reserve)) {
        continue;
      }
      publishedReserves.insert(reserve);
      auto exactWaits = plan.exactWaitsByReserve.find(reserve);
      bool hasExactWait = exactWaits != plan.exactWaitsByReserve.end() &&
                          llvm::any_of(exactWaits->second, [&](WaitOp wait) {
                            return isOrderedBefore(wait, push);
                          });
      auto candidates = plan.candidatesByReserve.find(reserve);
      bool hasSelectedCandidate =
          candidates != plan.candidatesByReserve.end() &&
          llvm::any_of(candidates->second, [&](WaitAnyCandidate candidate) {
            return isInReadyReceiveSelectionRegion(
                push, candidate.waitAny,
                static_cast<int64_t>(candidate.candidateIndex),
                isOrderedBefore);
          });
      if (!hasExactWait && !hasSelectedCandidate) {
        push->emitError(
            "publishes a wait-any receive reservation without proving that "
            "candidate complete");
        return failure();
      }
    }
  }
  for (Operation *reserve : plan.reserveOrder) {
    if (publishedReserves.contains(reserve)) {
      continue;
    }
    reserve->emitError("wait-any receive reservation is never published");
    return failure();
  }
  return success();
}

struct TTLInsertCBSyncPass
    : public impl::TTLInsertCBSyncBase<TTLInsertCBSyncPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    ValueOriginAnalysis valueOrigins(func);
    FailureOr<ConditionalReceiveReleasePlan> conditionalReleasePlan =
        buildConditionalReceiveReleasePlan(func, valueOrigins);
    if (failed(conditionalReleasePlan)) {
      signalPassFailure();
      return;
    }

    DFBAcquireReleaseOperations operations = collectDFBAcquireReleaseOps(func);
    if (!conditionalReleasePlan->reserves.empty()) {
      PlanningResult<std::unique_ptr<DFBAcquireReleaseIndex>> lifecycleResult =
          DFBAcquireReleaseIndex::create(func);
      if (lifecycleResult.isInvalidIR()) {
        const PlanningDiagnostic &diagnostic = lifecycleResult.getInvalidIR();
        diagnostic.operation->emitError(diagnostic.message);
        signalPassFailure();
        return;
      }
      assert(lifecycleResult.isPlanned() &&
             "DFB lifecycle indexing has no recoverable rejection");
      std::unique_ptr<DFBAcquireReleaseIndex> lifecycles =
          std::move(lifecycleResult).takePlan();
      DominanceInfo dominanceInfo(func);
      if (failed(validateConditionalReceiveReleases(
              operations.pushes, *conditionalReleasePlan, *lifecycles,
              dominanceInfo))) {
        signalPassFailure();
        return;
      }
    }

    auto producerPlan = planMissingReleases<CBPushOp>(
        operations.reserves, operations.producerProtocolReleases,
        conditionalReleasePlan->reserves, "push");
    if (producerPlan.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = producerPlan.getInvalidIR();
      diagnostic.operation->emitError(diagnostic.message);
      signalPassFailure();
      return;
    }
    const DenseSet<Operation *> noExplicitReleaseAcquires;
    auto consumerPlan = planMissingReleases<CBPopOp>(
        operations.waits, operations.consumerProtocolReleases,
        noExplicitReleaseAcquires, "pop");
    if (consumerPlan.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = consumerPlan.getInvalidIR();
      diagnostic.operation->emitError(diagnostic.message);
      signalPassFailure();
      return;
    }

    OpBuilder builder(func.getContext());

    // One nested release may satisfy multiple planned acquisition intervals.
    DenseSet<Operation *> erased;

    applyMissingReleases(producerPlan.getPlan(), erased, builder,
                         [](OpBuilder &builder, Location location, Value dfb) {
                           CBPushOp::create(builder, location, dfb,
                                            /*num_tiles=*/IntegerAttr{});
                         });

    applyMissingReleases(consumerPlan.getPlan(), erased, builder,
                         [](OpBuilder &builder, Location location, Value dfb) {
                           CBPopOp::create(builder, location, dfb,
                                           /*num_tiles=*/IntegerAttr{});
                         });
  }
};

} // namespace

} // namespace mlir::tt::ttl
