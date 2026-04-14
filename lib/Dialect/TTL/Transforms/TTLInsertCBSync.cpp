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
                         SmallVectorImpl<ReleaseOpTy> &allReleases,
                         SmallVectorImpl<ReleaseOpTy> &toHoist,
                         const DenseSet<Operation *> &erased) {
  Block *block = acquire->getBlock();
  bool hasSameLevelRelease = false;

  for (auto release : allReleases) {
    if (erased.contains(release))
      continue;
    if (release.getCb() != cb)
      continue;

    // Same-level: release is directly in the acquire's block.
    if (release->getBlock() == block) {
      if (!isBefore(acquire, release))
        continue;
      if (bound && !isBefore(release, bound))
        continue;
      hasSameLevelRelease = true;
      continue;
    }

    // Nested: release is inside a structured op in the acquire's block.
    Operation *ancestor = block->findAncestorOpInBlock(*release);
    if (!ancestor)
      continue;
    if (!isBefore(acquire, ancestor))
      continue;
    if (bound && !isBefore(ancestor, bound))
      continue;
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

  if (acquire->getNumResults() > 0)
    worklist.push_back(acquire->getResult(0));

  auto updateLast = [&](Operation *op) {
    Operation *ancestor = block->findAncestorOpInBlock(*op);
    if (!ancestor)
      return;
    if (isBefore(last, ancestor))
      last = ancestor;
  };

  auto inRange = [&](Operation *op) {
    Operation *ancestor = block->findAncestorOpInBlock(*op);
    if (!ancestor)
      return false;
    if (!isBefore(acquire, ancestor) && ancestor != acquire)
      return false;
    if (bound && !isBefore(ancestor, bound))
      return false;
    return true;
  };

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

    // Track erased ops so later iterations don't access dangling pointers.
    DenseSet<Operation *> erased;

    for (auto reserve : reserves) {
      Value cb = reserve.getCb();

      Operation *nextReserve = nullptr;
      for (auto other : reserves) {
        if (other == reserve || other.getCb() != cb)
          continue;
        if (other->getBlock() != reserve->getBlock())
          continue;
        if (!isBefore(reserve, other))
          continue;
        if (!nextReserve || isBefore(other, nextReserve))
          nextReserve = other;
      }

      SmallVector<CBPushOp> nestedPushes;
      if (findReleases(cb, reserve, nextReserve, pushes, nestedPushes,
                       erased))
        continue;

      for (auto nested : nestedPushes) {
        erased.insert(nested);
        nested.erase();
      }

      Operation *last = findLastTransitiveUse(cb, reserve, nextReserve);
      builder.setInsertionPointAfter(last);
      CBPushOp::create(builder, reserve.getLoc(), cb,
                       /*num_tiles=*/IntegerAttr{});
    }

    for (auto wait : waits) {
      Value cb = wait.getCb();

      Operation *nextWait = nullptr;
      for (auto other : waits) {
        if (other == wait || other.getCb() != cb)
          continue;
        if (other->getBlock() != wait->getBlock())
          continue;
        if (!isBefore(wait, other))
          continue;
        if (!nextWait || isBefore(other, nextWait))
          nextWait = other;
      }

      SmallVector<CBPopOp> nestedPops;
      if (findReleases(cb, wait, nextWait, pops, nestedPops, erased))
        continue;

      for (auto nested : nestedPops) {
        erased.insert(nested);
        nested.erase();
      }

      Operation *last = findLastTransitiveUse(cb, wait, nextWait);
      builder.setInsertionPointAfter(last);
      CBPopOp::create(builder, wait.getLoc(), cb);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
