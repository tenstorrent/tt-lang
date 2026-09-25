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
#include "mlir/Interfaces/LoopLikeInterface.h"
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

static bool isSameKindAcquisition(Operation *operation,
                                  DFBAcquireInterval interval) {
  return hasAcquireKind(operation, interval.kind) &&
         getDFBAcquireDFB(operation) == interval.dfb;
}

static Operation *findLocalKindBoundary(DFBAcquireInterval interval) {
  for (Operation &operation :
       llvm::make_range(std::next(interval.acquire->getIterator()),
                        interval.acquire->getBlock()->end())) {
    if (isSameKindAcquisition(&operation, interval)) {
      return &operation;
    }
  }
  return nullptr;
}

// The next same-kind acquisition of an interval's DFB lies inside `boundary`,
// a region operation of the ordering block; `firstAcquire` is the first such
// acquisition inside it in program order.
struct NestedAcquisitionBoundary {
  Operation *boundary = nullptr;
  Operation *firstAcquire = nullptr;
};

template <typename Root>
static Operation *findFirstSameKindAcquisition(Root &root,
                                               DFBAcquireInterval interval) {
  Operation *found = nullptr;
  root.walk([&](Operation *operation) {
    if (!isSameKindAcquisition(operation, interval)) {
      return WalkResult::advance();
    }
    found = operation;
    return WalkResult::interrupt();
  });
  return found;
}

static std::optional<NestedAcquisitionBoundary>
getNestedAcquisitionBoundary(DFBAcquireInterval interval) {
  Operation *boundary = interval.kindBoundary;
  if (!boundary || isDFBAcquireOp(boundary)) {
    return std::nullopt;
  }
  Operation *firstAcquire = findFirstSameKindAcquisition(*boundary, interval);
  assert(firstAcquire && "nested boundary must contain an acquisition");
  return NestedAcquisitionBoundary{boundary, firstAcquire};
}

// Whether every execution of `op` enters one of its regions.
static bool coversEveryPath(Operation *op) {
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    return !ifOp.getElseRegion().empty();
  }
  if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
    return ifOp.hasElse();
  }
  return isa<scf::IndexSwitchOp, scf::ExecuteRegionOp>(op);
}

static DFBProtocolEffectKind getAcquireEffectKind(DFBAcquireInterval interval) {
  return interval.kind == DFBAcquireReleaseKind::Producer
             ? DFBProtocolEffectKind::Reserve
             : DFBProtocolEffectKind::Wait;
}

// Whether `region` contains an acquisition or release of one kind on `dfb`.
static bool regionHasProtocolActionOfKind(Region &region, Value dfb,
                                          DFBProtocolEffectKind acquireKind,
                                          DFBProtocolEffectKind releaseKind) {
  return region
      .walk([&](Operation *operation) {
        return hasProtocolEffect(operation, dfb, acquireKind) ||
                       hasProtocolEffect(operation, dfb, releaseKind)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

// Protocol actions of one kind on the DFB along one path through a region of
// a nested-acquisition boundary: the region's top-level operations and, when
// the path leads to a same-kind acquisition in further nested regions, the
// operations on the way to it.
struct BoundaryPathOutline {
  // First same-kind acquisition on the path, or null.
  Operation *firstAcquire = nullptr;

  // Releases that no acquisition on the path precedes, in program order.
  SmallVector<Operation *> unownedReleases;

  // A protocol action whose execution on the path is not decided by entering
  // the region: inside a nested region off the path before the first
  // acquisition, inside a loop, or an unowned release after the first
  // acquisition at the path's level.
  Operation *indefiniteAction = nullptr;
};

class BoundaryPathOutliner {
public:
  BoundaryPathOutliner(Value dfb, DFBProtocolEffectKind acquireKind,
                       DFBProtocolEffectKind releaseKind)
      : dfb(dfb), acquireKind(acquireKind), releaseKind(releaseKind) {}

  // `target` is the first same-kind acquisition inside `region`, or null.
  BoundaryPathOutline outline(Region &region, Operation *target) {
    BoundaryPathOutline result;
    if (region.empty()) {
      return result;
    }
    if (!region.hasOneBlock()) {
      result.indefiniteAction = region.getParentOp();
      return result;
    }
    int64_t open = 0;
    visitBlock(region.front(), target, /*repeated=*/false, open, result);
    return result;
  }

private:
  void visitBlock(Block &block, Operation *target, bool repeated, int64_t &open,
                  BoundaryPathOutline &result) {
    Operation *targetAncestor =
        target ? block.findAncestorOpInBlock(*target) : nullptr;
    for (Operation &operation : block) {
      if (result.indefiniteAction) {
        return;
      }
      if (&operation == targetAncestor && &operation != target) {
        Region *next = target->getParentRegion();
        while (next->getParentOp() != &operation) {
          next = next->getParentOp()->getParentRegion();
        }
        for (Region &other : operation.getRegions()) {
          if (&other != next && regionHasProtocolActionOfKind(
                                    other, dfb, acquireKind, releaseKind)) {
            result.indefiniteAction = &operation;
            return;
          }
        }
        if (!next->hasOneBlock()) {
          result.indefiniteAction = &operation;
          return;
        }
        visitBlock(next->front(), target,
                   repeated || !executesRegionsAtMostOnce(&operation), open,
                   result);
        continue;
      }
      if (operation.getNumRegions() != 0) {
        // Actions after the path's first acquisition matter only when they
        // release the held block; a nested region there is left to its own
        // acquisitions.
        if (result.firstAcquire) {
          continue;
        }
        for (Region &region : operation.getRegions()) {
          if (regionHasProtocolActionOfKind(region, dfb, acquireKind,
                                            releaseKind)) {
            result.indefiniteAction = &operation;
            return;
          }
        }
        continue;
      }
      auto access = dyn_cast<DFBAccessOpInterface>(&operation);
      if (!access) {
        continue;
      }
      for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
        if (effect.dfb != dfb) {
          continue;
        }
        if (effect.kind == acquireKind) {
          if (!result.firstAcquire) {
            result.firstAcquire = &operation;
          }
          ++open;
          continue;
        }
        if (effect.kind != releaseKind) {
          continue;
        }
        if (open > 0) {
          --open;
          continue;
        }
        if (repeated || result.firstAcquire) {
          result.indefiniteAction = &operation;
          return;
        }
        result.unownedReleases.push_back(&operation);
      }
    }
  }

  Value dfb;
  DFBProtocolEffectKind acquireKind;
  DFBProtocolEffectKind releaseKind;
};

// The first release from `begin` to the end of its block that no acquisition
// in that range precedes, or null.
static Operation *findUnownedReleaseFrom(Value dfb, Block::iterator begin,
                                         Block *block,
                                         DFBProtocolEffectKind acquireKind,
                                         DFBProtocolEffectKind releaseKind) {
  int64_t open = 0;
  Operation *unowned = nullptr;
  for (Operation &operation : llvm::make_range(begin, block->end())) {
    operation.walk<WalkOrder::PreOrder>([&](Operation *nested) {
      auto access = dyn_cast<DFBAccessOpInterface>(nested);
      if (!access) {
        return WalkResult::advance();
      }
      for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
        if (effect.dfb != dfb) {
          continue;
        }
        if (effect.kind == acquireKind) {
          ++open;
        } else if (effect.kind == releaseKind) {
          if (open == 0) {
            unowned = nested;
            return WalkResult::interrupt();
          }
          --open;
        }
      }
      return WalkResult::advance();
    });
    if (unowned) {
      return unowned;
    }
  }
  return nullptr;
}

enum class NestedAcquisitionResolution {
  // No release of the held block exists inside the boundary; the normal
  // placement inserts one after the last owned use, before the boundary.
  InsertBeforeBoundary,
  // Every path through the boundary releases the held block before acquiring
  // the DFB again; the releases stay where they are.
  KeepReleases,
  // Some paths release the held block inside the boundary and others do not;
  // those releases move before the boundary.
  HoistReleases,
};

struct NestedAcquisitionPlan {
  NestedAcquisitionResolution resolution =
      NestedAcquisitionResolution::InsertBeforeBoundary;
  SmallVector<Operation *> releases;
};

// Decide how a block held at a nested-acquisition boundary is released. The
// block is released on a path through the boundary when that path performs
// exactly one release of its kind on the DFB that no acquisition on the path
// precedes, at the top level of the region or on the way to the path's first
// acquisition through at-most-once regions. Anything else that releases the
// DFB inside the boundary, and any release after the boundary that no later
// acquisition owns, marks the program invalid.
static PlanningResult<NestedAcquisitionPlan> planNestedAcquisitionBoundary(
    DFBAcquireInterval interval, const NestedAcquisitionBoundary &nested,
    Operation *lastOwnedUse, DFBProtocolEffectKind releaseEffectKind,
    StringRef effectName) {
  using Result = PlanningResult<NestedAcquisitionPlan>;
  Block *orderingBlock = nested.boundary->getBlock();
  Operation *start = orderingBlock->findAncestorOpInBlock(*interval.acquire);
  assert(start && "acquisition must project into the ordering block");
  bool guarded = start != interval.acquire;
  DFBProtocolEffectKind acquireKind = getAcquireEffectKind(interval);

  std::string beforeRegionAdvice =
      ("release it before that region (" + effectName +
       " it) or acquire the block inside the region")
          .str();
  std::string everyPathAdvice =
      ("release it before that region (" + effectName +
       " it) or on every path through the region")
          .str();
  auto invalid = [&](const std::string &advice) {
    return Result::invalidIR(
        interval.acquire,
        "dataflow buffer block is still acquired when a nested region "
        "acquires the same buffer again; " +
            advice,
        nested.firstAcquire, "the buffer is acquired again here");
  };

  BoundaryPathOutliner outliner(interval.dfb, acquireKind, releaseEffectKind);
  NestedAcquisitionPlan plan;
  bool everyPathReleased = coversEveryPath(nested.boundary);
  for (Region &region : nested.boundary->getRegions()) {
    BoundaryPathOutline outline = outliner.outline(
        region, findFirstSameKindAcquisition(region, interval));
    if (outline.indefiniteAction || outline.unownedReleases.size() > 1) {
      return invalid(beforeRegionAdvice);
    }
    if (outline.unownedReleases.empty()) {
      everyPathReleased = false;
      continue;
    }
    plan.releases.push_back(outline.unownedReleases.front());
  }
  if (findUnownedReleaseFrom(interval.dfb,
                             std::next(nested.boundary->getIterator()),
                             orderingBlock, acquireKind, releaseEffectKind)) {
    return invalid(beforeRegionAdvice);
  }
  if (everyPathReleased) {
    plan.resolution = NestedAcquisitionResolution::KeepReleases;
    return Result::planned(std::move(plan));
  }

  Operation *projectedLast =
      lastOwnedUse ? orderingBlock->findAncestorOpInBlock(*lastOwnedUse)
                   : nullptr;
  bool usedInOrAfterBoundary =
      lastOwnedUse && lastOwnedUse != start &&
      (!projectedLast || !projectedLast->isBeforeInBlock(nested.boundary));
  if (usedInOrAfterBoundary) {
    return invalid(plan.releases.empty() ? beforeRegionAdvice
                                         : everyPathAdvice);
  }
  if (plan.releases.empty()) {
    return Result::planned(std::move(plan));
  }
  // A guarded acquisition places its release under the acquiring condition;
  // moving a release out of the boundary into that placement is not modeled.
  if (guarded) {
    return invalid(everyPathAdvice);
  }
  plan.resolution = NestedAcquisitionResolution::HoistReleases;
  return Result::planned(std::move(plan));
}

// Whether a release the search attributes to the acquisition projects before
// `boundary` in the ordering block. Guarded local releases lie inside the
// guard, which precedes the boundary.
static bool hasOwnedReleaseBeforeBoundary(const DFBReleaseSearch &search,
                                          Operation *boundary) {
  if (!search.guardedLocalReleases.empty()) {
    return true;
  }
  Block *orderingBlock = boundary->getBlock();
  auto precedesBoundary = [&](Operation *release) {
    Operation *projected = orderingBlock->findAncestorOpInBlock(*release);
    return projected && projected->isBeforeInBlock(boundary);
  };
  return llvm::any_of(search.sameLevelReleases, precedesBoundary) ||
         llvm::any_of(search.releasesBeforeOwnedUses, precedesBoundary);
}

// Whether a release of `releaseKind` on the interval's DFB, direct or nested,
// lies strictly between the acquisition and `boundary` in their block.
static bool hasReleaseBefore(DFBAcquireInterval interval, Operation *boundary,
                             DFBProtocolEffectKind releaseKind) {
  for (Operation &operation :
       llvm::make_range(std::next(interval.acquire->getIterator()),
                        boundary->getIterator())) {
    bool found =
        operation
            .walk([&](Operation *nested) {
              return hasProtocolEffect(nested, interval.dfb, releaseKind)
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
            .wasInterrupted();
    if (found) {
      return true;
    }
  }
  return false;
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

static bool mayPropagateAcquiredDFBSlot(Operation *operation) {
  // A scalar read copies its value out of the DFB without retaining the slot.
  return !isa<ReadIndexOp>(operation);
}

static bool updateLocalSlotValuesAndTestUse(
    DFBAcquireInterval interval, Operation *operation,
    DenseSet<Value> &slotValues, Operation *localKindBoundary,
    ArrayRef<Operation *> acquires, const DominanceInfo &dominanceInfo) {
  bool usesSlot = false;
  for (Value operand : operation->getOperands()) {
    if (slotValues.contains(operand)) {
      usesSlot = true;
      break;
    }
  }
  if (usesSlot) {
    if (mayPropagateAcquiredDFBSlot(operation)) {
      for (Value result : operation->getResults()) {
        slotValues.insert(result);
      }
    }
    return !isa<AttachCBOp, UnrealizedConversionCastOp, scf::YieldOp>(
               operation) &&
           !getSingletonDimensionShapeViewSource(operation);
  }

  if (operation->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }

  if (!isBeforeLocalKindBoundary(operation, interval, localKindBoundary)) {
    return false;
  }
  // A later acquisition owns dominated direct DFB accesses because it advances
  // the queue pointer. SSA uses derived from this acquisition remain exact.
  if (llvm::any_of(acquires, [&](Operation *otherAcquire) {
        return otherAcquire != interval.acquire &&
               getDFBAcquireDFB(otherAcquire) == interval.dfb &&
               dominanceInfo.properlyDominates(interval.acquire,
                                               otherAcquire) &&
               dominanceInfo.properlyDominates(otherAcquire, operation);
      })) {
    return false;
  }
  return operationMayDirectlyUseAcquiredDFBSlot(interval, operation);
}

static bool nestedRegionMayUseLocalSlot(DFBAcquireInterval interval,
                                        Operation *operation,
                                        DenseSet<Value> slotValues,
                                        Operation *localKindBoundary,
                                        ArrayRef<Operation *> acquires,
                                        const DominanceInfo &dominanceInfo) {
  bool foundUse = false;
  operation->walk([&](Operation *nested) {
    if (nested == operation) {
      return;
    }
    foundUse |= updateLocalSlotValuesAndTestUse(interval, nested, slotValues,
                                                localKindBoundary, acquires,
                                                dominanceInfo);
  });
  return foundUse;
}

static bool operationMayUseLocalSlot(DFBAcquireInterval interval,
                                     Operation *operation,
                                     DenseSet<Value> &slotValues,
                                     Operation *localKindBoundary,
                                     ArrayRef<Operation *> acquires,
                                     const DominanceInfo &dominanceInfo) {
  if (updateLocalSlotValuesAndTestUse(interval, operation, slotValues,
                                      localKindBoundary, acquires,
                                      dominanceInfo)) {
    if (mayPropagateAcquiredDFBSlot(operation)) {
      for (Value result : operation->getResults()) {
        slotValues.insert(result);
      }
    }
    return true;
  }
  if (nestedRegionMayUseLocalSlot(interval, operation, slotValues,
                                  localKindBoundary, acquires, dominanceInfo)) {
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
        if (mayPropagateAcquiredDFBSlot(user)) {
          for (Value result : user->getResults()) {
            worklist.push_back(result);
          }
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
    DFBAcquireInterval interval, ArrayRef<Operation *> acquires,
    ArrayRef<Operation *> releases, DFBProtocolEffectKind releaseEffectKind,
    StringRef effectName, const DominanceInfo &dominanceInfo) {
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
                                 localKindBoundary, acquires, dominanceInfo)) {
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
    const DenseSet<Operation *> &acquisitionsRequiringExplicitRelease,
    const DominanceInfo &dominanceInfo, bool syncUserDFBs) {
  SmallVector<MissingReleasePlan> plans;
  for (Operation *acquire : acquires) {
    if (!syncUserDFBs && isUserManagedDFB(getDFBAcquireDFB(acquire))) {
      continue;
    }
    DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);

    // Tensor SSA uses can keep this acquired slot live past the next same-DFB
    // acquire. An existing release after that final use still belongs to this
    // acquire, so pass the final use into the release search.
    Operation *last = findLastDFBAcquireOwnedUse(interval);
    DFBReleaseSearch releaseSearch =
        findOwnedDFBReleases(interval, last, releases);

    // A data-movement kernel addresses a DFB through one read or write
    // pointer, so a same-kind acquisition while an earlier block is still
    // acquired returns that block again and the copies meant for either
    // address one slot. Compute kernels may hold several blocks because
    // consecutive acquisitions are coalesced into one multi-block acquisition
    // with offset views.
    if (Operation *localBoundary = findLocalKindBoundary(interval);
        localBoundary &&
        getKernelThreadType(acquire->getParentOfType<func::FuncOp>()) !=
            ttkernel::ThreadType::Compute &&
        !hasReleaseBefore(interval, localBoundary, releaseEffectKind)) {
      // Direct uses after the next acquisition in the acquiring block belong
      // to that acquisition, also for a guarded acquisition whose ordering
      // block is the guard's.
      DFBAcquireInterval localInterval = interval;
      localInterval.kindBoundary = localBoundary;
      Operation *localLast = acquire->getBlock()->findAncestorOpInBlock(
          *findLastDFBAcquireOwnedUse(localInterval));
      if (!localLast || localLast == acquire) {
        return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
            localBoundary,
            ("a data-movement kernel cannot hold two acquired blocks of one "
             "dataflow buffer; the earlier block has no use before this "
             "acquisition, so use and " +
             effectName + " it before this acquisition or drop it")
                .str());
      }
      // Tensor views may extend the earlier block past the boundary; a
      // block whose uses all precede it receives its release there, so a
      // later release that no later acquisition owns is the misplaced
      // release of the earlier block.
      if (localLast->isBeforeInBlock(localBoundary) &&
          findUnownedReleaseFrom(
              interval.dfb, localBoundary->getIterator(), acquire->getBlock(),
              getAcquireEffectKind(interval), releaseEffectKind)) {
        return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
            localBoundary,
            ("a data-movement kernel cannot hold two acquired blocks of one "
             "dataflow buffer; the earlier block is released after this "
             "acquisition, so " +
             effectName + " it before this acquisition")
                .str());
      }
    }

    // A block still acquired when a nested region acquires the same DFB with
    // the same kind would alias that acquisition, because cb_wait_front and
    // cb_reserve_back address the front or write slot regardless of an
    // earlier open acquisition. A release before the region settles it;
    // otherwise the releases inside the region decide (see
    // planNestedAcquisitionBoundary).
    if (std::optional<NestedAcquisitionBoundary> nested =
            getNestedAcquisitionBoundary(interval)) {
      if (!hasOwnedReleaseBeforeBoundary(releaseSearch, nested->boundary)) {
        PlanningResult<NestedAcquisitionPlan> nestedPlan =
            planNestedAcquisitionBoundary(interval, *nested, last,
                                          releaseEffectKind, effectName);
        if (nestedPlan.isInvalidIR()) {
          const PlanningDiagnostic &diagnostic = nestedPlan.getInvalidIR();
          return PlanningResult<SmallVector<MissingReleasePlan>>::invalidIR(
              diagnostic.operation, diagnostic.message,
              diagnostic.noteOperation, diagnostic.note);
        }
        switch (nestedPlan.getPlan().resolution) {
        case NestedAcquisitionResolution::KeepReleases:
          continue;
        case NestedAcquisitionResolution::HoistReleases:
          llvm::append_range(releaseSearch.nestedReleases,
                             nestedPlan.getPlan().releases);
          break;
        case NestedAcquisitionResolution::InsertBeforeBoundary:
          break;
        }
      }
    }

    if (scf::IfOp guard = getGuardedAcquireIf(acquire)) {
      Operation *externalKindBoundary =
          findGuardedExternalKindBoundary(interval, guard, acquires);
      auto localReleaseInfo = analyzeGuardedLocalReleases(
          interval, acquires, releases, releaseEffectKind, effectName,
          dominanceInfo);
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
  using impl::TTLInsertCBSyncBase<TTLInsertCBSyncPass>::TTLInsertCBSyncBase;

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
        emitPlanningDiagnostic(diagnostic);
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
    DominanceInfo dominanceInfo(func);
    auto producerPlan = planMissingReleases<CBPushOp>(
        operations.reserves, operations.producerProtocolReleases,
        DFBProtocolEffectKind::Push, "push", conditionalReleasePlan->reserves,
        dominanceInfo, syncUserDFBs);
    if (producerPlan.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = producerPlan.getInvalidIR();
      emitPlanningDiagnostic(diagnostic);
      signalPassFailure();
      return;
    }
    auto consumerPlan = planMissingReleases<CBPopOp>(
        operations.waits, operations.consumerProtocolReleases,
        DFBProtocolEffectKind::Pop, "pop", noExplicitReleaseAcquisitions,
        dominanceInfo, syncUserDFBs);
    if (consumerPlan.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = consumerPlan.getInvalidIR();
      emitPlanningDiagnostic(diagnostic);
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
