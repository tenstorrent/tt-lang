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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

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

static scf::IfOp getGuardedAcquireIf(Operation *acquire) {
  if (!isGuardedDFBAcquire(acquire)) {
    return {};
  }
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

static bool updateLocalSlotValuesAndTestUse(DFBAcquireInterval interval,
                                            Operation *operation,
                                            DenseSet<Value> &slotValues) {
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
    return true;
  }

  if (operation->hasTrait<OpTrait::IsTerminator>() ||
      isDFBReleaseOp(operation) || !mayAccessDFBStorage(operation)) {
    return false;
  }

  if (auto access = dyn_cast<DFBAccessOpInterface>(operation)) {
    if (access.hasUnknownDFBAccess()) {
      return true;
    }
    for (Value dependency : access.getDFBDependencyOperands()) {
      if (dependency == interval.dfb) {
        return true;
      }
    }
    return false;
  }

  for (Value operand : operation->getOperands()) {
    if (operand == interval.dfb) {
      return true;
    }
  }
  return false;
}

static bool nestedRegionMayUseLocalSlot(DFBAcquireInterval interval,
                                        Operation *operation,
                                        DenseSet<Value> slotValues) {
  bool foundUse = false;
  operation->walk([&](Operation *nested) {
    if (nested == operation) {
      return;
    }
    foundUse |= updateLocalSlotValuesAndTestUse(interval, nested, slotValues);
  });
  return foundUse;
}

static bool operationMayUseLocalSlot(DFBAcquireInterval interval,
                                     Operation *operation,
                                     DenseSet<Value> &slotValues) {
  if (updateLocalSlotValuesAndTestUse(interval, operation, slotValues)) {
    for (Value result : operation->getResults()) {
      slotValues.insert(result);
    }
    return true;
  }
  if (nestedRegionMayUseLocalSlot(interval, operation, slotValues)) {
    for (Value result : operation->getResults()) {
      slotValues.insert(result);
    }
    return true;
  }
  return false;
}

template <typename ConcreteReleaseOp>
static PlanningResult<SmallVector<Operation *>> collectLocalConcreteReleases(
    DFBAcquireInterval interval, ArrayRef<Operation *> releases,
    DFBProtocolEffectKind releaseEffectKind, StringRef effectName) {
  SmallVector<Operation *> localReleases;
  DenseSet<Operation *> candidateReleases;
  for (Operation *release : releases) {
    if (!hasProtocolEffect(release, interval.dfb, releaseEffectKind) ||
        release->getBlock() != interval.acquire->getBlock() ||
        !interval.acquire->isBeforeInBlock(release)) {
      continue;
    }
    if (!isa<ConcreteReleaseOp>(release)) {
      return PlanningResult<SmallVector<Operation *>>::invalidIR(
          release,
          ("external dataflow buffer " + effectName +
           " effect cannot be relocated out of a guarded acquisition region")
              .str());
    }
    localReleases.push_back(release);
    candidateReleases.insert(release);
  }

  DenseSet<Value> slotValues;
  slotValues.insert(interval.acquire->getResult(0));
  Operation *lastLocalUse = interval.acquire;
  for (Operation &operation :
       llvm::make_range(std::next(interval.acquire->getIterator()),
                        interval.acquire->getBlock()->end())) {
    if (operation.hasTrait<OpTrait::IsTerminator>()) {
      break;
    }
    if (candidateReleases.contains(&operation)) {
      continue;
    }
    if (operationMayUseLocalSlot(interval, &operation, slotValues)) {
      lastLocalUse = &operation;
    }
  }

  for (Operation *release : localReleases) {
    if (!lastLocalUse->isBeforeInBlock(release)) {
      return PlanningResult<SmallVector<Operation *>>::invalidIR(
          release, ("guarded local dataflow buffer " + effectName +
                    " must follow all uses in its acquiring region")
                       .str());
    }
  }
  return PlanningResult<SmallVector<Operation *>>::planned(
      std::move(localReleases));
}

// External release effects cannot be relocated as concrete operations. Validate
// every interval before mutation so failure leaves the input IR unchanged.
template <typename ConcreteReleaseOp>
static PlanningResult<SmallVector<MissingReleasePlan>> planMissingReleases(
    ArrayRef<Operation *> acquires, ArrayRef<Operation *> releases,
    DFBProtocolEffectKind releaseEffectKind, StringRef effectName) {
  SmallVector<MissingReleasePlan> plans;
  for (Operation *acquire : acquires) {
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

    if (isGuardedDFBAcquire(acquire)) {
      scf::IfOp guard = getGuardedAcquireIf(acquire);
      if (guard && last == guard.getOperation()) {
        auto localReleases = collectLocalConcreteReleases<ConcreteReleaseOp>(
            interval, releases, releaseEffectKind, effectName);
        if (localReleases.isInvalidIR()) {
          const PlanningDiagnostic &diagnostic = localReleases.getInvalidIR();
          return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
              diagnostic.operation, diagnostic.message);
        }
        plans.push_back({acquire, guard.getOperation(),
                         ReleaseInsertionKind::GuardedAfterOperation,
                         guard.getCondition(), interval.dfb,
                         getAcquireNumTilesAttr(acquire),
                         std::move(localReleases).takePlan()});
        continue;
      }

      return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
          acquire,
          ("conditional dataflow buffer " + effectName +
           " requires an explicit release after its guarded uses under the "
           "same condition")
              .str());
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

struct TTLInsertCBSyncPass
    : public impl::TTLInsertCBSyncBase<TTLInsertCBSyncPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    DFBAcquireReleaseOperations operations = collectDFBAcquireReleaseOps(func);
    auto producerPlan = planMissingReleases<CBPushOp>(
        operations.reserves, operations.producerProtocolReleases,
        DFBProtocolEffectKind::Push, "push");
    if (producerPlan.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = producerPlan.getInvalidIR();
      diagnostic.operation->emitError(diagnostic.message);
      signalPassFailure();
      return;
    }
    auto consumerPlan = planMissingReleases<CBPopOp>(
        operations.waits, operations.consumerProtocolReleases,
        DFBProtocolEffectKind::Pop, "pop");
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
