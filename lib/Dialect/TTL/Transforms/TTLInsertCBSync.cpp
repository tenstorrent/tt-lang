// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert CB Sync
//===----------------------------------------------------------------------===//
//
// Inserts missing cb_push / cb_pop for unmatched cb_reserve / cb_wait ops.
//
// Placement is driven by the live interval of the acquire's tensor result:
// the SSA tensor value produced by cb_reserve / cb_wait flows through
// attach_cb, ttl.store, and downstream compute ops. The last operation in
// the acquire's block that transitively consumes the tensor marks the end
// of the interval; the release is inserted immediately after it. Uses in
// descendant regions (scf.for / scf.if bodies) project to their ancestor
// in the acquire's block, so a tensor read inside a loop body correctly
// extends the interval through the enclosing structured op.
//
// Releases nested inside structured control flow are hoisted: the nested
// release is erased and a single release is placed at the acquire's block
// scope. Pre-existing same-level releases mark the acquire as already
// handled and the pass leaves it alone (idempotency).
//
// Legality invariants:
//   P1. cb_push must follow the store into the reserved slot and precede
//       any cb_wait that consumes the slot, including waits nested in
//       descendant regions.
//   P2. cb_pop must follow the last transitive use of the waited value,
//       including uses nested in descendant regions.
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

/// Find releases on `cb` at or after `acquire` in the acquire's block.
/// Same-level releases are "matching" (the acquire is already handled).
/// Nested releases are collected into `toHoist` for erasure.
template <typename ReleaseOpTy>
static bool findReleases(Value cb, Operation *acquire,
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
    toHoist.push_back(release);
  }

  return hasSameLevelRelease;
}

/// Live-interval endpoint for `acquire->getResult(0)`: the last op in
/// `acquire->getBlock()` that transitively consumes the tensor value. Uses
/// in descendant regions project up to their ancestor in the acquire's
/// block. If the tensor has no uses, returns `acquire`.
static Operation *findLastTensorUse(Operation *acquire) {
  Operation *last = acquire;
  if (acquire->getNumResults() == 0) {
    return last;
  }

  Block *block = acquire->getBlock();
  DenseSet<Operation *> visited;
  SmallVector<Value, 8> worklist;
  worklist.push_back(acquire->getResult(0));

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (!visited.insert(user).second) {
        continue;
      }
      // Sync ops don't extend the live interval.
      if (isa<CBPushOp, CBPopOp>(user)) {
        continue;
      }
      Operation *ancestor = block->findAncestorOpInBlock(*user);
      if (!ancestor) {
        continue;
      }
      if (isBefore(last, ancestor)) {
        last = ancestor;
      }
      for (Value result : user->getResults()) {
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

    auto insertMissingReleases = [&](auto acquires, auto &releases,
                                     auto createRelease) {
      for (auto acquire : acquires) {
        Value cb = acquire.getCb();

        using ReleaseOpTy =
            typename std::remove_reference_t<decltype(releases)>::value_type;
        SmallVector<ReleaseOpTy> nested;
        if (findReleases(cb, acquire, releases, nested, erased)) {
          continue;
        }

        for (auto nestedOp : nested) {
          erased.insert(nestedOp);
          nestedOp.erase();
        }

        Operation *last = findLastTensorUse(acquire);
        builder.setInsertionPointAfter(last);
        createRelease(builder, acquire.getLoc(), cb);
      }
    };

    insertMissingReleases(
        reserves, pushes, [](OpBuilder &b, Location loc, Value cb) {
          CBPushOp::create(b, loc, cb, /*num_tiles=*/IntegerAttr{});
        });

    insertMissingReleases(waits, pops,
                          [](OpBuilder &b, Location loc, Value cb) {
                            CBPopOp::create(b, loc, cb);
                          });
  }
};

} // namespace

} // namespace mlir::tt::ttl
