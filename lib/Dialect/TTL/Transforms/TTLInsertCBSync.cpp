// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert CB Sync
//===----------------------------------------------------------------------===//
//
// Inserts missing cb_push / cb_pop for unmatched cb_reserve / cb_wait ops.
// Computes a transitive use closure to find the last operation that touches
// the CB's data, and inserts the release after that point.
//
// Releases nested inside structured control flow (scf.if branches) are
// hoisted: the nested release is erased and a single release is placed
// after the enclosing structured op. This keeps push/pop at the same
// scope level as their acquire.
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

/// Return true if `a` is before `b` in their common block.
static bool isBefore(Operation *a, Operation *b) {
  return a->isBeforeInBlock(b);
}

/// Find releases on `cb` between `acquire` and `bound` in the acquire's block.
/// Same-level releases are "matching" (the acquire is already handled).
/// Nested releases are collected into `toHoist` for erasure.
template <typename ReleaseOpTy>
static bool findReleases(Value cb, Operation *acquire, Operation *bound,
                         const SmallVectorImpl<ReleaseOpTy> &allReleases,
                         SmallVectorImpl<ReleaseOpTy> &toHoist,
                         const DenseSet<Operation *> &erased) {
  Block *block = acquire->getBlock();
  bool hasSameLevelRelease = false;

  for (auto release : allReleases) {
    if (erased.contains(release)) {
      continue;
    }
    if (release.getCb() != cb) {
      continue;
    }

    // Same-level: release is directly in the acquire's block.
    if (release->getBlock() == block) {
      if (!isBefore(acquire, release)) {
        continue;
      }
      if (bound && !isBefore(release, bound)) {
        continue;
      }
      hasSameLevelRelease = true;
      continue;
    }

    // Nested: release is inside a structured op in the acquire's block.
    Operation *ancestor = block->findAncestorOpInBlock(*release);
    if (!ancestor) {
      continue;
    }
    if (!isBefore(acquire, ancestor)) {
      continue;
    }
    if (bound && !isBefore(ancestor, bound)) {
      continue;
    }
    toHoist.push_back(release);
  }

  return hasSameLevelRelease;
}

/// Compute the last operation (in the acquire's block) in the transitive
/// use closure of an acquire.
///
/// Uses in nested regions are projected up to their ancestor in the
/// acquire's block (e.g., an add inside an scf.if projects to the scf.if).
static Operation *findLastTransitiveUse(Value cb, Operation *acquire,
                                        Operation *bound) {
  Block *block = acquire->getBlock();
  Operation *last = acquire;
  DenseSet<Operation *> visited;
  SmallVector<Value, 8> worklist;

  if (acquire->getNumResults() > 0) {
    worklist.push_back(acquire->getResult(0));
  }

  // Returns the ancestor in the acquire's block, or nullptr if out of range.
  auto getInRangeAncestor = [&](Operation *op) -> Operation * {
    Operation *ancestor = block->findAncestorOpInBlock(*op);
    if (!ancestor) {
      return nullptr;
    }
    if (!isBefore(acquire, ancestor) && ancestor != acquire) {
      return nullptr;
    }
    if (bound && !isBefore(ancestor, bound)) {
      return nullptr;
    }
    return ancestor;
  };

  for (auto &use : cb.getUses()) {
    Operation *user = use.getOwner();
    if (user == acquire) {
      continue;
    }
    if (isa<CBPushOp, CBPopOp, CBReserveOp, CBWaitOp>(user)) {
      continue;
    }
    Operation *ancestor = getInRangeAncestor(user);
    if (!ancestor) {
      continue;
    }
    if (isBefore(last, ancestor)) {
      last = ancestor;
    }
    for (auto result : user->getResults()) {
      worklist.push_back(result);
    }
  }

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (auto &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (!visited.insert(user).second) {
        continue;
      }
      if (isa<CBPushOp, CBPopOp>(user)) {
        continue;
      }
      Operation *ancestor = getInRangeAncestor(user);
      if (!ancestor) {
        continue;
      }
      if (isBefore(last, ancestor)) {
        last = ancestor;
      }
      for (auto result : user->getResults()) {
        worklist.push_back(result);
      }
    }
  }

  return last;
}

struct TTLInsertCBSyncPass
    : public impl::TTLInsertCBSyncBase<TTLInsertCBSyncPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<CBReserveOp> reserves;
    SmallVector<CBWaitOp> waits;
    SmallVector<CBPushOp> pushes;
    SmallVector<CBPopOp> pops;

    func.walk([&](Operation *op) {
      if (auto r = dyn_cast<CBReserveOp>(op)) {
        reserves.push_back(r);
      } else if (auto w = dyn_cast<CBWaitOp>(op)) {
        waits.push_back(w);
      } else if (auto p = dyn_cast<CBPushOp>(op)) {
        pushes.push_back(p);
      } else if (auto p = dyn_cast<CBPopOp>(op)) {
        pops.push_back(p);
      }
    });

    OpBuilder builder(func.getContext());

    // Track erased ops so later iterations don't access dangling pointers.
    DenseSet<Operation *> erased;

    // Find the next op on the same CB, in the same block, after `after`.
    auto findNextOnCB = [](Value cb, Operation *after, auto &candidates) {
      Operation *next = nullptr;
      for (auto candidate : candidates) {
        if (candidate.getCb() != cb) {
          continue;
        }
        if (candidate->getBlock() != after->getBlock()) {
          continue;
        }
        if (!isBefore(after, candidate)) {
          continue;
        }
        if (!next || isBefore(candidate, next)) {
          next = candidate;
        }
      }
      return next;
    };

    auto insertMissingReleases = [&](auto acquires, auto &releases,
                                     auto createRelease,
                                     auto &extraBoundCandidates) {
      for (auto acquire : acquires) {
        Value cb = acquire.getCb();

        // Bound = earliest of next same-type acquire and next
        // extra-bound candidate (e.g., next cb_wait for reserves).
        Operation *bound = findNextOnCB(cb, acquire, acquires);
        Operation *extra = findNextOnCB(cb, acquire, extraBoundCandidates);
        if (extra) {
          bound = bound ? (isBefore(bound, extra) ? bound : extra) : extra;
        }

        using ReleaseOpTy =
            typename std::remove_reference_t<decltype(releases)>::value_type;
        SmallVector<ReleaseOpTy> nested;
        if (findReleases(cb, acquire, bound, releases, nested, erased)) {
          continue;
        }

        for (auto nestedOp : nested) {
          erased.insert(nestedOp);
          nestedOp.erase();
        }

        Operation *last = findLastTransitiveUse(cb, acquire, bound);
        builder.setInsertionPointAfter(last);
        createRelease(builder, acquire.getLoc(), cb);
      }
    };

    // For reserves, bound push placement by the next cb_wait on the same
    // CB. Intra-thread DFBs use the same CB for both reserve and wait,
    // so push must precede wait to avoid deadlock.
    insertMissingReleases(
        reserves, pushes,
        [](OpBuilder &b, Location loc, Value cb) {
          CBPushOp::create(b, loc, cb, /*num_tiles=*/IntegerAttr{});
        },
        waits);

    // For waits, only bound by the next same-CB wait (no extra bound).
    SmallVector<CBWaitOp> noExtraBound;
    insertMissingReleases(
        waits, pops,
        [](OpBuilder &b, Location loc, Value cb) {
          CBPopOp::create(b, loc, cb);
        },
        noExtraBound);
  }
};

} // namespace

} // namespace mlir::tt::ttl
