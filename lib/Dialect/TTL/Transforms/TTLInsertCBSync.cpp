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
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-cb-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTCBSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Return true if `a` is before `b` in their common block. Both ops must
/// be in the same block.
static bool isBefore(Operation *a, Operation *b) { return a->isBeforeInBlock(b); }

/// Check if `releaseOps` contains a release on `cb` that is between
/// `acquire` (exclusive) and `bound` (exclusive). When `bound` is null,
/// the end of the block is the boundary.
template <typename ReleaseOpTy>
static bool hasMatchingRelease(Value cb, Operation *acquire, Operation *bound,
                               SmallVectorImpl<ReleaseOpTy> &releaseOps) {
  for (auto release : releaseOps) {
    if (release.getCb() != cb)
      continue;
    if (!isBefore(acquire, release))
      continue;
    if (bound && !isBefore(release, bound))
      continue;
    return true;
  }
  return false;
}

/// Compute the last operation in the transitive use closure of an acquire.
///
/// Starting from the acquire result and all uses of `cb` between `acquire`
/// and `bound`, chase produced values one level deep (handles the
/// copy -> transfer_handle -> wait chain).
static Operation *findLastTransitiveUse(Value cb, Operation *acquire,
                                        Operation *bound) {
  Operation *last = acquire;
  DenseSet<Operation *> visited;
  SmallVector<Value, 8> worklist;

  // Seed with the acquire result (the tensor view).
  if (acquire->getNumResults() > 0)
    worklist.push_back(acquire->getResult(0));

  auto updateLast = [&](Operation *op) {
    if (op->getBlock() == acquire->getBlock() && isBefore(last, op))
      last = op;
  };

  auto inRange = [&](Operation *op) {
    if (op->getBlock() != acquire->getBlock())
      return false;
    if (!isBefore(acquire, op) && op != acquire)
      return false;
    if (bound && !isBefore(op, bound))
      return false;
    return true;
  };

  // Collect direct uses of the CB value in range. These are ops like
  // copy(%slice, %cb) or store(%tensor, %reserve_view) that reference %cb.
  for (auto &use : cb.getUses()) {
    Operation *user = use.getOwner();
    if (user == acquire)
      continue;
    if (isa<CBPushOp, CBPopOp, CBReserveOp, CBWaitOp>(user))
      continue;
    if (!inRange(user))
      continue;
    updateLast(user);
    for (auto result : user->getResults())
      worklist.push_back(result);
  }

  // Chase one level of def-use from seeded values. This handles:
  //   attach_cb result -> used by store, arithmetic, etc.
  //   copy result (transfer_handle) -> used by wait
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (auto &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (!visited.insert(user).second)
        continue;
      if (isa<CBPushOp, CBPopOp>(user))
        continue;
      if (!inRange(user))
        continue;
      updateLast(user);
      // Chase one more level for results.
      for (auto result : user->getResults())
        worklist.push_back(result);
    }
  }

  return last;
}

struct TTLInsertCBSyncPass
    : public impl::TTLInsertCBSyncBase<TTLInsertCBSyncPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect all CB sync ops per CB value.
    SmallVector<CBReserveOp> reserves;
    SmallVector<CBWaitOp> waits;
    SmallVector<CBPushOp> pushes;
    SmallVector<CBPopOp> pops;

    func.walk([&](Operation *op) {
      if (auto r = dyn_cast<CBReserveOp>(op))
        reserves.push_back(r);
      else if (auto w = dyn_cast<CBWaitOp>(op))
        waits.push_back(w);
      else if (auto p = dyn_cast<CBPushOp>(op))
        pushes.push_back(p);
      else if (auto p = dyn_cast<CBPopOp>(op))
        pops.push_back(p);
    });

    OpBuilder builder(func.getContext());

    // For each cb_reserve, check if a matching cb_push exists.
    for (auto reserve : reserves) {
      Value cb = reserve.getCb();

      // Find the next cb_reserve on the same CB (scope boundary).
      Operation *nextReserve = nullptr;
      for (auto other : reserves) {
        if (other == reserve)
          continue;
        if (other.getCb() != cb)
          continue;
        if (other->getBlock() != reserve->getBlock())
          continue;
        if (!isBefore(reserve, other))
          continue;
        if (!nextReserve || isBefore(other, nextReserve))
          nextReserve = other;
      }

      if (hasMatchingRelease(cb, reserve, nextReserve, pushes))
        continue;

      Operation *last = findLastTransitiveUse(cb, reserve, nextReserve);
      builder.setInsertionPointAfter(last);
      CBPushOp::create(builder, reserve.getLoc(), cb,
                       /*num_tiles=*/IntegerAttr{});
    }

    // For each cb_wait, check if a matching cb_pop exists.
    for (auto wait : waits) {
      Value cb = wait.getCb();

      // Find the next cb_wait on the same CB (scope boundary).
      Operation *nextWait = nullptr;
      for (auto other : waits) {
        if (other == wait)
          continue;
        if (other.getCb() != cb)
          continue;
        if (other->getBlock() != wait->getBlock())
          continue;
        if (!isBefore(wait, other))
          continue;
        if (!nextWait || isBefore(other, nextWait))
          nextWait = other;
      }

      if (hasMatchingRelease(cb, wait, nextWait, pops))
        continue;

      Operation *last = findLastTransitiveUse(cb, wait, nextWait);
      builder.setInsertionPointAfter(last);
      CBPopOp::create(builder, wait.getLoc(), cb);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
