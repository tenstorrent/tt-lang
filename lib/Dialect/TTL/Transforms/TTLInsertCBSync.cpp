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
// different valid IR situations -- direct-DFB uses and tensor-SSA uses --
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

#include <optional>

#define DEBUG_TYPE "ttl-insert-cb-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTCBSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

enum class ReleaseInsertionKind { AfterOperation, GuardedAfterOperation };

// Defers one rewrite until all producer and consumer intervals are valid.
struct MissingReleasePlan {
  Operation *acquire = nullptr;

  // Last owned use after which the closing operation is inserted.
  Operation *insertionAfter = nullptr;

  ReleaseInsertionKind insertionKind = ReleaseInsertionKind::AfterOperation;

  Value guardCondition;

  Value dfb;

  IntegerAttr releaseNumTiles;

  // Nested concrete releases are retained until every candidate is valid.
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

struct GuardedLocalReleaseInfo {
  Operation *lastLocalUse = nullptr;
  SmallVector<Operation *> releases;
};

struct GuardedAcquireUseInfo {
  bool hasNonLocalUse = false;
};

static scf::IfOp getGuardedAcquireIf(Operation *acquire) {
  auto ifOp = dyn_cast_or_null<scf::IfOp>(acquire->getParentOp());
  if (!ifOp || acquire->getBlock()->getParent() != &ifOp.getThenRegion()) {
    return {};
  }
  return ifOp;
}

static bool hasProtocolEffect(Operation *operation, Value dfb,
                              DFBProtocolEffectKind kind) {
  auto access = dyn_cast<DFBAccessOpInterface>(operation);
  if (!access) {
    return false;
  }
  for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
    if (effect.dfb == dfb && effect.kind == kind) {
      return true;
    }
  }
  return false;
}

static IntegerAttr getAcquireNumTilesAttr(Operation *acquire) {
  if (auto reserve = dyn_cast<CBReserveOp>(acquire)) {
    return reserve.getNumTilesAttr();
  }
  return cast<CBWaitOp>(acquire).getNumTilesAttr();
}

static bool hasAcquireKind(Operation *operation, DFBAcquireReleaseKind kind) {
  switch (kind) {
  case DFBAcquireReleaseKind::Producer:
    return isa<CBReserveOp>(operation);
  case DFBAcquireReleaseKind::Consumer:
    return isa<CBWaitOp>(operation);
  }
  llvm_unreachable("unknown DFB acquire/release kind");
}

static Operation *findLocalKindBoundary(DFBAcquireInterval interval) {
  for (Operation &operation :
       llvm::make_range(std::next(interval.acquire->getIterator()),
                        interval.acquire->getBlock()->end())) {
    if (hasAcquireKind(&operation, interval.kind) &&
        getDFBAcquireDFB(&operation) == interval.dfb) {
      return &operation;
    }
  }
  return nullptr;
}

static bool isBeforeLocalKindBoundary(Operation *operation,
                                      DFBAcquireInterval interval,
                                      Operation *localKindBoundary) {
  if (!localKindBoundary) {
    return true;
  }
  Operation *projected =
      operation->getBlock() == interval.acquire->getBlock()
          ? operation
          : interval.acquire->getBlock()->findAncestorOpInBlock(*operation);
  return projected && projected->isBeforeInBlock(localKindBoundary);
}

static bool updateLocalSlotValuesAndTestUse(DFBAcquireInterval interval,
                                            Operation *operation,
                                            DenseSet<Value> &slotValues,
                                            Operation *localKindBoundary) {
  bool usesSlot = false;
  for (Value operand : operation->getOperands()) {
    if (slotValues.contains(operand)) {
      usesSlot = true;
      break;
    }
  }
  if (usesSlot) {
    for (Value result : operation->getResults()) {
      slotValues.insert(result);
    }
    return !isa<AttachCBOp, UnrealizedConversionCastOp, scf::YieldOp>(
        operation);
  }

  if (operation->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }

  if (!isBeforeLocalKindBoundary(operation, interval, localKindBoundary)) {
    return false;
  }
  return operationMayDirectlyUseAcquiredDFBSlot(interval, operation);
}

static bool nestedRegionMayUseLocalSlot(DFBAcquireInterval interval,
                                        Operation *operation,
                                        DenseSet<Value> slotValues,
                                        Operation *localKindBoundary) {
  bool foundUse = false;
  operation->walk([&](Operation *nested) {
    if (nested == operation) {
      return;
    }
    foundUse |= updateLocalSlotValuesAndTestUse(interval, nested, slotValues,
                                                localKindBoundary);
  });
  return foundUse;
}

static bool operationMayUseLocalSlot(DFBAcquireInterval interval,
                                     Operation *operation,
                                     DenseSet<Value> &slotValues,
                                     Operation *localKindBoundary) {
  if (updateLocalSlotValuesAndTestUse(interval, operation, slotValues,
                                      localKindBoundary)) {
    for (Value result : operation->getResults()) {
      slotValues.insert(result);
    }
    return true;
  }
  if (nestedRegionMayUseLocalSlot(interval, operation, slotValues,
                                  localKindBoundary)) {
    for (Value result : operation->getResults()) {
      slotValues.insert(result);
    }
    return true;
  }
  return false;
}

static bool isNestedUnder(Operation *operation, Operation *ancestor) {
  for (Operation *current = operation; current;
       current = current->getParentOp()) {
    if (current == ancestor) {
      return true;
    }
  }
  return false;
}

static bool isOrderedAfterAcquireInGuard(Operation *operation,
                                         DFBAcquireInterval interval,
                                         scf::IfOp guard,
                                         Operation *externalKindBoundary) {
  if (isNestedUnder(operation, guard.getOperation())) {
    Operation *projected =
        operation->getBlock() == interval.acquire->getBlock()
            ? operation
            : interval.acquire->getBlock()->findAncestorOpInBlock(*operation);
    return projected && interval.acquire->isBeforeInBlock(projected);
  }

  Operation *projected =
      operation->getBlock() == guard->getBlock()
          ? operation
          : guard->getBlock()->findAncestorOpInBlock(*operation);
  if (!projected || !guard->isBeforeInBlock(projected)) {
    return false;
  }
  return !externalKindBoundary ||
         projected->isBeforeInBlock(externalKindBoundary);
}

static Operation *projectToGuardBlock(Operation *operation, scf::IfOp guard) {
  return operation->getBlock() == guard->getBlock()
             ? operation
             : guard->getBlock()->findAncestorOpInBlock(*operation);
}

static Operation *
findGuardedExternalKindBoundary(DFBAcquireInterval interval, scf::IfOp guard,
                                ArrayRef<Operation *> acquires) {
  Operation *boundary = nullptr;
  for (Operation *other : acquires) {
    if (other == interval.acquire || getDFBAcquireDFB(other) != interval.dfb) {
      continue;
    }
    Operation *projected = projectToGuardBlock(other, guard);
    if (!projected || !guard->isBeforeInBlock(projected)) {
      continue;
    }
    if (!boundary || projected->isBeforeInBlock(boundary)) {
      boundary = projected;
    }
  }
  return boundary;
}

static std::optional<PlanningDiagnostic> validateGuardedExternalReleases(
    DFBAcquireInterval interval, scf::IfOp guard, Operation *lastOwnedUse,
    Operation *externalKindBoundary, ArrayRef<Operation *> releases,
    DFBProtocolEffectKind releaseEffectKind, StringRef effectName) {
  Operation *projectedLast =
      lastOwnedUse ? projectToGuardBlock(lastOwnedUse, guard) : nullptr;
  for (Operation *release : releases) {
    if (!hasProtocolEffect(release, interval.dfb, releaseEffectKind) ||
        isNestedUnder(release, guard.getOperation())) {
      continue;
    }

    Operation *projectedRelease = projectToGuardBlock(release, guard);
    if (!projectedRelease || !guard->isBeforeInBlock(projectedRelease)) {
      continue;
    }
    if (externalKindBoundary &&
        !projectedRelease->isBeforeInBlock(externalKindBoundary)) {
      continue;
    }

    if (!isOperationInThenRegionGuardedBy(release, guard.getCondition())) {
      return PlanningDiagnostic(release,
                                ("conditional dataflow buffer " + effectName +
                                 " must execute under the acquiring condition")
                                    .str());
    }
    if (projectedLast && !projectedLast->isBeforeInBlock(projectedRelease)) {
      return PlanningDiagnostic(
          release, ("conditional dataflow buffer " + effectName +
                    " must follow all uses under the acquiring condition")
                       .str());
    }
  }
  return std::nullopt;
}

static bool hasGuardedExternalRelease(DFBAcquireInterval interval,
                                      scf::IfOp guard, Operation *lastOwnedUse,
                                      Operation *externalKindBoundary,
                                      ArrayRef<Operation *> releases,
                                      DFBProtocolEffectKind releaseEffectKind) {
  Operation *projectedLast =
      lastOwnedUse ? projectToGuardBlock(lastOwnedUse, guard) : nullptr;
  for (Operation *release : releases) {
    if (!hasProtocolEffect(release, interval.dfb, releaseEffectKind) ||
        isNestedUnder(release, guard.getOperation()) ||
        !isOperationInThenRegionGuardedBy(release, guard.getCondition())) {
      continue;
    }
    Operation *projectedRelease = projectToGuardBlock(release, guard);
    if (!projectedRelease || !guard->isBeforeInBlock(projectedRelease)) {
      continue;
    }
    if (externalKindBoundary &&
        !projectedRelease->isBeforeInBlock(externalKindBoundary)) {
      continue;
    }
    if (projectedLast && !projectedLast->isBeforeInBlock(projectedRelease)) {
      continue;
    }
    return true;
  }
  return false;
}

static PlanningResult<GuardedAcquireUseInfo>
analyzeGuardedAcquireUses(DFBAcquireInterval interval, scf::IfOp guard,
                          Operation *externalKindBoundary) {
  GuardedAcquireUseInfo info;

  auto classifyUse = [&](Operation *user) -> std::optional<PlanningDiagnostic> {
    if (isNestedUnder(user, guard.getOperation())) {
      return std::nullopt;
    }
    if (!isOperationInThenRegionGuardedBy(user, guard.getCondition())) {
      return PlanningDiagnostic(
          user,
          "conditional dataflow buffer slot use must be under the acquiring "
          "condition");
    }
    info.hasNonLocalUse = true;
    return std::nullopt;
  };

  DenseSet<Value> visitedValues;
  SmallVector<Value, 8> worklist;
  worklist.push_back(interval.acquire->getResult(0));

  auto drainWorklist = [&]() -> std::optional<PlanningDiagnostic> {
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      if (!visitedValues.insert(value).second) {
        continue;
      }
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (isa<CBPushOp, CBPopOp>(user)) {
          continue;
        }
        if (auto yield = dyn_cast<scf::YieldOp>(user)) {
          if (auto ifOp = dyn_cast<scf::IfOp>(yield->getParentOp())) {
            unsigned resultIndex = use.getOperandNumber();
            if (resultIndex < ifOp.getNumResults()) {
              worklist.push_back(ifOp.getResult(resultIndex));
            }
          }
          continue;
        }
        if (std::optional<PlanningDiagnostic> diagnostic = classifyUse(user)) {
          return diagnostic;
        }
        for (Value result : user->getResults()) {
          worklist.push_back(result);
        }
      }
    }
    return std::nullopt;
  };

  if (std::optional<PlanningDiagnostic> diagnostic = drainWorklist()) {
    return PlanningResult<GuardedAcquireUseInfo>::invalidIR(
        diagnostic->operation, diagnostic->message);
  }

  DenseSet<Operation *> visitedDirectUsers;
  for (OpOperand &use : interval.dfb.getUses()) {
    Operation *user = use.getOwner();
    if (!visitedDirectUsers.insert(user).second) {
      continue;
    }
    if (!operationMayDirectlyUseAcquiredDFBSlot(interval, user)) {
      continue;
    }
    if (!isOrderedAfterAcquireInGuard(user, interval, guard,
                                      externalKindBoundary)) {
      continue;
    }
    if (std::optional<PlanningDiagnostic> diagnostic = classifyUse(user)) {
      return PlanningResult<GuardedAcquireUseInfo>::invalidIR(
          diagnostic->operation, diagnostic->message);
    }
    for (Value result : user->getResults()) {
      worklist.push_back(result);
    }
  }
  if (std::optional<PlanningDiagnostic> diagnostic = drainWorklist()) {
    return PlanningResult<GuardedAcquireUseInfo>::invalidIR(
        diagnostic->operation, diagnostic->message);
  }

  return PlanningResult<GuardedAcquireUseInfo>::planned(info);
}

static PlanningResult<GuardedLocalReleaseInfo> analyzeGuardedLocalReleases(
    DFBAcquireInterval interval, ArrayRef<Operation *> releases,
    DFBProtocolEffectKind releaseEffectKind, StringRef effectName) {
  GuardedLocalReleaseInfo info;
  Operation *localKindBoundary = findLocalKindBoundary(interval);
  DenseSet<Operation *> candidateReleases;
  for (Operation *release : releases) {
    if (!hasProtocolEffect(release, interval.dfb, releaseEffectKind) ||
        release->getBlock() != interval.acquire->getBlock() ||
        !interval.acquire->isBeforeInBlock(release)) {
      continue;
    }
    info.releases.push_back(release);
    candidateReleases.insert(release);
  }

  DenseSet<Value> slotValues;
  slotValues.insert(interval.acquire->getResult(0));
  info.lastLocalUse = interval.acquire;
  for (Operation &operation :
       llvm::make_range(std::next(interval.acquire->getIterator()),
                        interval.acquire->getBlock()->end())) {
    if (operation.hasTrait<OpTrait::IsTerminator>()) {
      break;
    }
    if (candidateReleases.contains(&operation)) {
      continue;
    }
    if (operationMayUseLocalSlot(interval, &operation, slotValues,
                                 localKindBoundary)) {
      info.lastLocalUse = &operation;
    }
  }

  bool localUseExtendsPastBoundary =
      localKindBoundary && info.lastLocalUse != interval.acquire &&
      !info.lastLocalUse->isBeforeInBlock(localKindBoundary);
  llvm::erase_if(info.releases, [&](Operation *release) {
    return localKindBoundary && !release->isBeforeInBlock(localKindBoundary) &&
           !localUseExtendsPastBoundary;
  });

  for (Operation *release : info.releases) {
    if (!info.lastLocalUse->isBeforeInBlock(release)) {
      return PlanningResult<GuardedLocalReleaseInfo>::invalidIR(
          release, ("guarded local dataflow buffer " + effectName +
                    " must follow all uses in its acquiring region")
                       .str());
    }
  }
  return PlanningResult<GuardedLocalReleaseInfo>::planned(std::move(info));
}

// External release effects cannot be relocated as concrete operations. Validate
// every interval before mutation so failure leaves the input IR unchanged.
template <typename ConcreteReleaseOp>
static PlanningResult<SmallVector<MissingReleasePlan>> planMissingReleases(
    ArrayRef<Operation *> acquires, ArrayRef<Operation *> releases,
    DFBProtocolEffectKind releaseEffectKind, StringRef effectName,
    const DenseSet<Operation *> &acquisitionsRequiringExplicitRelease) {
  SmallVector<MissingReleasePlan> plans;
  for (Operation *acquire : acquires) {
    DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);

    // Tensor SSA uses can keep this acquired slot live past the next same-DFB
    // acquire. An existing release after that final use still belongs to this
    // acquire, so pass the final use into the release search.
    Operation *last = findLastDFBAcquireOwnedUse(interval);
    DFBReleaseSearch releaseSearch =
        findOwnedDFBReleases(interval, last, releases);

    if (scf::IfOp guard = getGuardedAcquireIf(acquire)) {
      Operation *externalKindBoundary =
          findGuardedExternalKindBoundary(interval, guard, acquires);
      auto localReleaseInfo = analyzeGuardedLocalReleases(
          interval, releases, releaseEffectKind, effectName);
      if (localReleaseInfo.isInvalidIR()) {
        const PlanningDiagnostic &diagnostic = localReleaseInfo.getInvalidIR();
        return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
            diagnostic.operation, diagnostic.message);
      }

      auto guardedUseInfo =
          analyzeGuardedAcquireUses(interval, guard, externalKindBoundary);
      if (guardedUseInfo.isInvalidIR()) {
        const PlanningDiagnostic &diagnostic = guardedUseInfo.getInvalidIR();
        return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
            diagnostic.operation, diagnostic.message);
      }

      if (std::optional<PlanningDiagnostic> diagnostic =
              validateGuardedExternalReleases(interval, guard, last,
                                              externalKindBoundary, releases,
                                              releaseEffectKind, effectName)) {
        return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
            diagnostic->operation, diagnostic->message);
      }

      if (releaseSearch.hasSameLevelRelease()) {
        continue;
      }

      if (hasGuardedExternalRelease(interval, guard, last, externalKindBoundary,
                                    releases, releaseEffectKind)) {
        continue;
      }

      if (!guardedUseInfo.getPlan().hasNonLocalUse) {
        if (!localReleaseInfo.getPlan().releases.empty()) {
          continue;
        }
        plans.push_back({acquire,
                         localReleaseInfo.getPlan().lastLocalUse,
                         ReleaseInsertionKind::AfterOperation,
                         Value{},
                         interval.dfb,
                         getAcquireNumTilesAttr(acquire),
                         {}});
        continue;
      }

      for (Operation *release : localReleaseInfo.getPlan().releases) {
        if (!isa<ConcreteReleaseOp>(release)) {
          return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
              release,
              ("external dataflow buffer " + effectName +
               " effect cannot be relocated out of a guarded acquisition "
               "region")
                  .str());
        }
      }

      plans.push_back(
          {acquire, last, ReleaseInsertionKind::GuardedAfterOperation,
           guard.getCondition(), interval.dfb, getAcquireNumTilesAttr(acquire),
           localReleaseInfo.getPlan().releases});
      continue;
    }

    if (interval.kind == DFBAcquireReleaseKind::Producer) {
      for (Operation *release : releaseSearch.releasesBeforeOwnedUses) {
        if (hasProducerDFBAcquireStorageUseAfterRelease(interval, release)) {
          return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
              release, ("dataflow buffer " + effectName +
                        " must follow all uses owned by its acquisition")
                           .str());
        }
      }
    }

    if (!releaseSearch.releasesBeforeOwnedUses.empty()) {
      continue;
    }

    if (acquisitionsRequiringExplicitRelease.contains(acquire)) {
      continue;
    }

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

    plans.push_back({acquire, last, ReleaseInsertionKind::AfterOperation,
                     Value{}, interval.dfb, getAcquireNumTilesAttr(acquire),
                     std::move(releaseSearch.nestedReleases)});
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
    if (plan.insertionKind == ReleaseInsertionKind::GuardedAfterOperation) {
      auto ifOp = scf::IfOp::create(builder, plan.acquire->getLoc(),
                                    plan.guardCondition);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }
    createRelease(builder, plan.acquire->getLoc(), plan.dfb,
                  plan.releaseNumTiles);
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

    const DenseSet<Operation *> noExplicitReleaseAcquisitions;
    auto producerPlan = planMissingReleases<CBPushOp>(
        operations.reserves, operations.producerProtocolReleases,
        DFBProtocolEffectKind::Push, "push", conditionalReleasePlan->reserves);
    if (producerPlan.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = producerPlan.getInvalidIR();
      diagnostic.operation->emitError(diagnostic.message);
      signalPassFailure();
      return;
    }
    auto consumerPlan = planMissingReleases<CBPopOp>(
        operations.waits, operations.consumerProtocolReleases,
        DFBProtocolEffectKind::Pop, "pop", noExplicitReleaseAcquisitions);
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
                         [](OpBuilder &builder, Location location, Value dfb,
                            IntegerAttr numTiles) {
                           CBPushOp::create(builder, location, dfb, numTiles);
                         });

    applyMissingReleases(consumerPlan.getPlan(), erased, builder,
                         [](OpBuilder &builder, Location location, Value dfb,
                            IntegerAttr numTiles) {
                           CBPopOp::create(builder, location, dfb, numTiles);
                         });
  }
};

} // namespace

} // namespace mlir::tt::ttl
