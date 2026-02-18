// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Tile Regs Sync Pass
//===----------------------------------------------------------------------===//
//
// This pass inserts DST register synchronization operations inside ttl.compute
// bodies to enforce the MATH/PACK thread synchronization protocol required by
// the hardware DST register bank.
//
// The pass inserts DST lifecycle ops relative to existing ttl.tile_store ops:
//    - Inserts init_sfpu before the compute op (if not present)
//    - Inserts tile_regs_acquire at the beginning of the body
//    - Inserts tile_regs_commit + tile_regs_wait before existing tile_stores
//    - Inserts tile_regs_release at the end (before yield)
//
// This pass does NOT create stores or CB lifecycle ops. Stores come from
// Python and are transformed to tile_store by convert-ttl-to-compute.
// If a tile_store is missing, that is a bug in an earlier pass.
//
// DST lifecycle per tile:
//   acquire -> [compute] -> commit -> wait -> [pack via tile_store] -> release
//
//===----------------------------------------------------------------------===//

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

      Value icb = getAttachedCB(computeOp.getInputs().front());
      Value ocb = getAttachedCB(computeOp.getOutputs().front());
      Location loc = computeOp.getLoc();

      // Find existing sync ops preceding this compute. Stop at another compute
      // op since each compute has its own lifecycle ops.
      auto stopAtCompute = [](Operation *op) { return isa<ComputeOp>(op); };
      TileRegsAcquireOp existingAcquire =
          findPrecedingOp<TileRegsAcquireOp>(computeOperation, stopAtCompute);
      InitSFPUOp existingInitSfpu =
          findPrecedingOp<InitSFPUOp>(computeOperation, stopAtCompute);

      OpBuilder builder(computeOp);

      if (!existingInitSfpu) {
        Operation *insertBefore =
            existingAcquire ? existingAcquire : computeOperation;
        builder.setInsertionPoint(insertBefore);
        builder.create<InitSFPUOp>(loc, icb, ocb);
      }

      Block &body = computeOp.getRegion().front();

      // Insert acquire at start of compute body.
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

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
