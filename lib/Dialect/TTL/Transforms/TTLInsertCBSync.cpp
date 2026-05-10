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

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-cb-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTCBSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

enum class DFBSyncClass { Producer, Consumer };

struct ReleaseSearch {
  bool hasSameLevelRelease = false;
  SmallVector<Operation *> nestedReleases;
};

struct AcquireInterval {
  Operation *acquire;
  Value cb;
  DFBSyncClass syncClass;
  Operation *syncClassBoundary;
};

static bool isBefore(Operation *a, Operation *b) {
  return a->isBeforeInBlock(b);
}

static bool isAcquireOp(Operation *op) {
  return isa<CBReserveOp, CBWaitOp>(op);
}

static bool isReleaseOp(Operation *op) { return isa<CBPushOp, CBPopOp>(op); }

static Value getAcquireCB(Operation *op) {
  if (auto reserve = dyn_cast<CBReserveOp>(op)) {
    return reserve.getCb();
  }
  return cast<CBWaitOp>(op).getCb();
}

static Value getReleaseCB(Operation *op) {
  if (auto push = dyn_cast<CBPushOp>(op)) {
    return push.getCb();
  }
  return cast<CBPopOp>(op).getCb();
}

static DFBSyncClass getDFBSyncClass(Operation *op) {
  if (isa<CBReserveOp>(op)) {
    return DFBSyncClass::Producer;
  }
  assert(isa<CBWaitOp>(op) && "unsupported DFB acquire op");
  return DFBSyncClass::Consumer;
}

static bool isLifecycleOrAttachOp(Operation *op) {
  return isAcquireOp(op) || isReleaseOp(op) || isa<AttachCBOp>(op);
}

static bool directDFBUseMatchesAcquire(AcquireInterval interval,
                                       Operation *user) {
  auto copy = dyn_cast<CopyOp>(user);
  if (!copy) {
    return true;
  }

  switch (interval.syncClass) {
  case DFBSyncClass::Producer:
    return copy.getDst() == interval.cb;
  case DFBSyncClass::Consumer:
    return copy.getSrc() == interval.cb;
  }
  llvm_unreachable("unknown DFB sync class");
}

static bool projectToAcquireBlock(AcquireInterval interval, Operation *op,
                                  Operation *&projected,
                                  bool ignoreBoundary = false) {
  Block *block = interval.acquire->getBlock();
  projected = op->getBlock() == block ? op : block->findAncestorOpInBlock(*op);
  if (!projected) {
    return false;
  }
  if (!isBefore(interval.acquire, projected)) {
    return false;
  }
  if (!ignoreBoundary && interval.syncClassBoundary &&
      !isBefore(projected, interval.syncClassBoundary)) {
    return false;
  }
  return true;
}

static void updateLatestUse(Operation *candidate, Operation *&latest) {
  if (isBefore(latest, candidate)) {
    latest = candidate;
  }
}

/// Find releases owned by this acquire interval. When `lastOwnedUse` is
/// non-null and falls past the next-acquire boundary, also accept releases
/// in that extended range so the pass is idempotent on re-run.
static ReleaseSearch findOwnedReleases(AcquireInterval interval,
                                       Operation *lastOwnedUse,
                                       ArrayRef<Operation *> allReleases,
                                       const DenseSet<Operation *> &erased) {
  ReleaseSearch result;
  Block *block = interval.acquire->getBlock();

  bool useExtendsPastBoundary =
      lastOwnedUse && lastOwnedUse != interval.acquire &&
      interval.syncClassBoundary &&
      !isBefore(lastOwnedUse, interval.syncClassBoundary);

  for (Operation *release : allReleases) {
    if (erased.contains(release)) {
      continue;
    }
    if (getReleaseCB(release) != interval.cb) {
      continue;
    }

    if (release->getBlock() == block) {
      Operation *projected = nullptr;
      if (projectToAcquireBlock(interval, release, projected)) {
        result.hasSameLevelRelease = true;
        continue;
      }
      // Re-check past the boundary: a release at or after the acquire's
      // last owned use is one this pass would have inserted on a prior run.
      if (useExtendsPastBoundary &&
          projectToAcquireBlock(interval, release, projected,
                                /*ignoreBoundary=*/true) &&
          !isBefore(projected, lastOwnedUse)) {
        result.hasSameLevelRelease = true;
      }
      continue;
    }

    Operation *projected = nullptr;
    if (!projectToAcquireBlock(interval, release, projected)) {
      continue;
    }
    result.nestedReleases.push_back(release);
  }

  return result;
}

static void updateBoundary(Value cb, Operation *acquire,
                           ArrayRef<Operation *> acquires,
                           Operation *&boundary) {
  Block *block = acquire->getBlock();
  for (Operation *other : acquires) {
    if (other == acquire) {
      continue;
    }
    if (getAcquireCB(other) != cb) {
      continue;
    }
    Operation *ancestor = block->findAncestorOpInBlock(*other);
    if (!ancestor) {
      continue;
    }
    if (!isBefore(acquire, ancestor)) {
      continue;
    }
    if (!boundary || isBefore(ancestor, boundary)) {
      boundary = ancestor;
    }
  }
}

/// Return the closest later acquire on `cb` in the same DFB sync class,
/// projected into `acquire`'s block.
static Operation *findNextSyncClassAcquire(Value cb, Operation *acquire,
                                           ArrayRef<Operation *> acquires) {
  Operation *boundary = nullptr;
  updateBoundary(cb, acquire, acquires, boundary);
  return boundary;
}

/// Return the last op in `acquire`'s block that consumes the acquired
/// slot. See `docs/development/DFBManagement.md` for the asymmetric
/// classification of direct-DFB vs. tensor-SSA uses that this walk
/// implements.
static Operation *findLastOwnedUse(AcquireInterval interval) {
  Operation *last = interval.acquire;
  DenseSet<Operation *> visited;
  SmallVector<Value, 8> worklist;

  auto extend = [&](Operation *user, bool ignoreBoundary) {
    Operation *projected = nullptr;
    if (!projectToAcquireBlock(interval, user, projected, ignoreBoundary)) {
      return false;
    }
    if (!visited.insert(user).second) {
      return false;
    }
    updateLatestUse(projected, last);
    for (Value result : user->getResults()) {
      worklist.push_back(result);
    }
    return true;
  };

  auto drainWorklist = [&](bool ignoreBoundary) {
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (isa<CBPushOp, CBPopOp>(user)) {
          continue;
        }
        extend(user, ignoreBoundary);
      }
    }
  };

  // Direct-DFB uses. The walk recurses through each user's SSA results
  // because the *true* end of the use can be a downstream op (e.g.
  // ttl.copy returns a transfer_handle whose ttl.wait marks the actual
  // end of the transfer). The next-acquire boundary applies: two
  // direct-DFB uses straddling that boundary belong to different
  // intervals.
  for (OpOperand &use : interval.cb.getUses()) {
    Operation *user = use.getOwner();
    if (user == interval.acquire) {
      continue;
    }
    if (isLifecycleOrAttachOp(user)) {
      continue;
    }
    if (!directDFBUseMatchesAcquire(interval, user)) {
      continue;
    }
    extend(user, /*ignoreBoundary=*/false);
  }
  drainWorklist(/*ignoreBoundary=*/false);

  // Tensor-SSA uses. The next-acquire boundary does NOT apply: a tile
  // produced by `cb_wait t1` may legitimately be consumed after
  // `cb_wait t2`, since the consumer reads through the SSA value, not
  // the slot's identity. Applying the boundary here was the root cause
  // of the issue #536 follow-up miscompile.
  assert(interval.acquire->getNumResults() == 1 &&
         "DFB acquire ops produce exactly one tensor result");
  worklist.push_back(interval.acquire->getResult(0));
  drainWorklist(/*ignoreBoundary=*/true);

  return last;
}

static AcquireInterval makeAcquireInterval(Operation *acquire,
                                           ArrayRef<Operation *> acquires) {
  Value cb = getAcquireCB(acquire);
  return {acquire, cb, getDFBSyncClass(acquire),
          findNextSyncClassAcquire(cb, acquire, acquires)};
}

template <typename CreateReleaseFn>
static void insertMissingReleases(ArrayRef<Operation *> acquires,
                                  ArrayRef<Operation *> releases,
                                  DenseSet<Operation *> &erased,
                                  OpBuilder &builder,
                                  CreateReleaseFn createRelease) {
  for (Operation *acquire : acquires) {
    AcquireInterval interval = makeAcquireInterval(acquire, acquires);
    // Cheap check first: any release inside the strict next-acquire range?
    ReleaseSearch releaseSearch =
        findOwnedReleases(interval, /*lastOwnedUse=*/nullptr, releases, erased);
    if (releaseSearch.hasSameLevelRelease) {
      continue;
    }

    // Compute the last owned use; it both bounds the idempotency recheck
    // and pinpoints the insertion point.
    Operation *last = findLastOwnedUse(interval);
    if (last != interval.acquire) {
      releaseSearch = findOwnedReleases(interval, last, releases, erased);
      if (releaseSearch.hasSameLevelRelease) {
        continue;
      }
    }

    for (Operation *nestedRelease : releaseSearch.nestedReleases) {
      erased.insert(nestedRelease);
      nestedRelease->erase();
    }

    builder.setInsertionPointAfter(last);
    createRelease(builder, acquire->getLoc(), interval.cb);
  }
}

struct TTLInsertCBSyncPass
    : public impl::TTLInsertCBSyncBase<TTLInsertCBSyncPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<Operation *> reserves;
    SmallVector<Operation *> waits;
    SmallVector<Operation *> pushes;
    SmallVector<Operation *> pops;

    func.walk([&](Operation *op) {
      if (isa<CBReserveOp>(op)) {
        reserves.push_back(op);
      } else if (isa<CBWaitOp>(op)) {
        waits.push_back(op);
      } else if (isa<CBPushOp>(op)) {
        pushes.push_back(op);
      } else if (isa<CBPopOp>(op)) {
        pops.push_back(op);
      }
    });

    OpBuilder builder(func.getContext());

    // Track erased ops so later iterations skip them before any accessor
    // call. The set holds raw pointers to freed ops; `findOwnedReleases` must
    // check `erased.contains(...)` before touching any op wrapper method.
    DenseSet<Operation *> erased;

    insertMissingReleases(reserves, pushes, erased, builder,
                          [](OpBuilder &b, Location loc, Value cb) {
                            CBPushOp::create(b, loc, cb,
                                             /*num_tiles=*/IntegerAttr{});
                          });

    insertMissingReleases(waits, pops, erased, builder,
                          [](OpBuilder &b, Location loc, Value cb) {
                            CBPopOp::create(b, loc, cb,
                                            /*num_tiles=*/IntegerAttr{});
                          });
  }
};

} // namespace

} // namespace mlir::tt::ttl
