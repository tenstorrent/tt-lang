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
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-cb-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTCBSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

template <typename CreateReleaseFn>
static void insertMissingReleases(ArrayRef<Operation *> acquires,
                                  ArrayRef<Operation *> releases,
                                  DenseSet<Operation *> &erased,
                                  OpBuilder &builder,
                                  CreateReleaseFn createRelease) {
  for (Operation *acquire : acquires) {
    DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);
    // Cheap check first: any release inside the strict next-acquire range?
    DFBReleaseSearch releaseSearch = findOwnedDFBReleases(
        interval, /*lastOwnedUse=*/nullptr, releases, &erased);
    if (releaseSearch.hasSameLevelRelease()) {
      continue;
    }

    // Compute the last owned use; it both bounds the idempotency recheck
    // and pinpoints the insertion point.
    Operation *last = findLastDFBAcquireOwnedUse(interval);
    if (last != interval.acquire) {
      releaseSearch = findOwnedDFBReleases(interval, last, releases, &erased);
      if (releaseSearch.hasSameLevelRelease()) {
        continue;
      }
    }

    for (Operation *nestedRelease : releaseSearch.nestedReleases) {
      erased.insert(nestedRelease);
      nestedRelease->erase();
    }

    builder.setInsertionPointAfter(last);
    createRelease(builder, acquire->getLoc(), interval.dfb);
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

    collectDFBAcquireReleaseOps(func, reserves, waits, pushes, pops);

    OpBuilder builder(func.getContext());

    // Track erased ops so later iterations skip them before any accessor
    // call. The set holds raw pointers to freed ops; release ownership search
    // must check the set before touching any op wrapper method.
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
