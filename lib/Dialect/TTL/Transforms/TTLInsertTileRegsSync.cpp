// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Tile Regs Sync Pass
//===----------------------------------------------------------------------===//
//
// The TTLInsertTileRegsSyncPass inserts DST register synchronization operations
// for ttl.compute ops to enforce the MATH/PACK thread synchronization protocol
// required by the hardware DST register bank.
//
// This pass only inserts sync ops (acquire/commit/wait/release). Common init
// ops (init_sfpu, binary_op_init_common) and per-op init ops are inserted
// later by the ttkernel-insert-inits pass after conversion to TTKernel.
//
// Sync op placement depends on whether the compute is subblocked or not:
//
// 1. Subblocked computes (has ttl.full_linearization_strides):
//    Sync ops go OUTSIDE the compute body. One sync cycle covers all tiles
//    in the subblock. Lower-to-loops then unrolls the body into N tile copies
//    that share the single sync cycle. Store hoisting later moves pack/store
//    ops after the wait barrier.
//    DST lifecycle per subblock:
//      acquire -> [N x compute] -> commit -> wait -> [N x pack] -> release
//
// 2. Non-subblocked computes (no ttl.full_linearization_strides):
//    Sync ops go INSIDE the compute body (per-tile sync). This is the
//    original behavior for computes that haven't been through the subblocking
//    pass. Lower-to-loops creates tile loops, and each iteration has its own
//    sync cycle.
//    DST lifecycle per tile:
//      acquire -> [compute] -> commit -> wait -> [pack] -> release
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "ttl-insert-tile-regs-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTTILEREGSSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLInsertTileRegsSyncPass
    : public impl::TTLInsertTileRegsSyncBase<TTLInsertTileRegsSyncPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    WalkResult result = funcOp.walk([&](ComputeOp computeOp) -> WalkResult {
      Operation *computeOperation = computeOp.getOperation();
      Location loc = computeOp.getLoc();

      // Find existing acquire preceding this compute. Stop at another compute
      // op since each compute has its own lifecycle ops.
      auto stopAtCompute = [](Operation *op) { return isa<ComputeOp>(op); };
      TileRegsAcquireOp existingAcquire =
          findPrecedingOp<TileRegsAcquireOp>(computeOperation, stopAtCompute);

      OpBuilder builder(computeOp);

      // Check if this compute was processed by the subblocking pass.
      // If so, place sync outside (one cycle per subblock). Otherwise,
      // place sync inside the body (one cycle per tile iteration).
      bool isSubblocked = computeOp->hasAttr(kFullLinStridesAttrName);

      if (isSubblocked) {
        // Outside placement: one sync cycle covers all tiles in the subblock.
        // After lower-to-loops unrolls the body, all tile ops appear inside
        // the sync region (between acquire and release).
        if (!existingAcquire) {
          builder.setInsertionPoint(computeOperation);
          builder.create<TileRegsAcquireOp>(loc);
        }

        // Insert commit, wait, release after the compute op.
        builder.setInsertionPointAfter(computeOperation);
        builder.create<TileRegsCommitOp>(loc);
        builder.create<TileRegsWaitOp>(loc);
        builder.create<TileRegsReleaseOp>(loc);
      } else {
        // Inside placement: per-tile sync (original behavior).
        Block &body = computeOp.getRegion().front();

        if (!existingAcquire) {
          builder.setInsertionPointToStart(&body);
          builder.create<TileRegsAcquireOp>(loc);
        }
        auto *terminator = body.getTerminator();

        SmallVector<TileStoreOp> storeOps;
        TileRegsCommitOp commitOp = nullptr;
        TileRegsWaitOp waitOp = nullptr;
        for (Operation &op : body.without_terminator()) {
          TypeSwitch<Operation *>(&op)
              .Case<TileStoreOp>([&](auto store) { storeOps.push_back(store); })
              .Case<TileRegsCommitOp>([&](auto commit) { commitOp = commit; })
              .Case<TileRegsWaitOp>([&](auto wait) { waitOp = wait; });
        }

        // Insert commit + wait before the first tile_store.
        if (!storeOps.empty()) {
          builder.setInsertionPoint(storeOps.front());
        } else {
          builder.setInsertionPoint(terminator);
        }

        if (!commitOp) {
          commitOp = builder.create<TileRegsCommitOp>(loc);
        }
        if (!waitOp) {
          waitOp = builder.create<TileRegsWaitOp>(loc);
        }
        if (!commitOp->isBeforeInBlock(waitOp)) {
          commitOp->moveBefore(waitOp);
        }

        // Release: at end of compute body (before yield).
        builder.setInsertionPoint(terminator);
        builder.create<TileRegsReleaseOp>(loc);
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
