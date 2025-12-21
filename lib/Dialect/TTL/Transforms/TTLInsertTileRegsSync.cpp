// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Tile Regs Sync Pass
//===----------------------------------------------------------------------===//
//
// This pass inserts DST register synchronization operations around ttl.compute
// regions to enforce the MATH/PACK thread synchronization protocol required by
// the hardware DST register bank.
//
// The pass performs the following transformations:
//
// 1. Inside ttl.compute body:
//    - Inserts tile_regs_acquire at the beginning (if not present)
//    - Inserts tile_regs_commit immediately before ttl.store/ttl.yield
//    - Inserts tile_regs_wait before ttl.store
//    - Inserts ttl.store for outputs that lack one (requires existing CB view)
//
// 2. Outside ttl.compute (in parent block):
//    - Inserts tile_regs_release immediately after the ttl.compute operation
//
// This establishes the correct DST lifecycle:
//   MATH thread:  acquire -> [compute] -> commit
//   PACK thread:  wait -> release
//
// The pass is idempotent - it checks if sync ops are already present before
// inserting them, allowing it to be run multiple times safely.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

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
      Block *parent = computeOperation->getBlock();
      assert(parent && "ComputeOp must have parent block");

      if (computeOperation->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
        computeOp.emitOpError()
            << "ttl-insert-tile-regs-sync expects ttl.compute to be "
            << "non-IsolatedFromAbove; it rematerializes cb_reserve inside "
            << "the body for auto-inserted stores";
        return WalkResult::interrupt();
      }

      // Acquire: before the compute op in parent block.
      Operation *prev = computeOperation->getPrevNode();
      if (!isa_and_nonnull<TileRegsAcquireOp>(prev)) {
        OpBuilder beforeBuilder(parent, Block::iterator(computeOperation));
        beforeBuilder.create<TileRegsAcquireOp>(computeOp.getLoc());
      }

      Block &body = computeOp.getRegion().front();
      auto *terminator = body.getTerminator();
      auto yieldOp = cast<YieldOp>(terminator);

      // Collect existing sync and store ops.
      SmallVector<StoreOp> storeOps;
      TileRegsCommitOp commitOp = nullptr;
      TileRegsWaitOp waitOp = nullptr;
      for (Operation &op : body.without_terminator()) {
        if (auto store = dyn_cast<StoreOp>(&op)) {
          storeOps.push_back(store);
          continue;
        }
        if (auto commit = dyn_cast<TileRegsCommitOp>(&op)) {
          commitOp = commit;
          continue;
        }
        if (auto wait = dyn_cast<TileRegsWaitOp>(&op)) {
          waitOp = wait;
          continue;
        }
      }

      // Ensure commit and wait exist near the end of the block.
      OpBuilder endBuilder(terminator);
      if (!commitOp) {
        commitOp = endBuilder.create<TileRegsCommitOp>(computeOp.getLoc());
      }
      if (!waitOp) {
        waitOp = endBuilder.create<TileRegsWaitOp>(computeOp.getLoc());
      }
      // Enforce ordering: commit -> wait.
      if (!commitOp->isBeforeInBlock(waitOp)) {
        commitOp->moveBefore(waitOp);
      }

      // Map yielded outputs to existing stores.
      SmallVector<Value> yielded(yieldOp.getValues().begin(),
                                 yieldOp.getValues().end());
      llvm::DenseMap<size_t, StoreOp> storeForOutput;
      for (StoreOp store : storeOps) {
        auto it = llvm::find(yielded, store.getTile());
        if (it != yielded.end()) {
          storeForOutput.try_emplace(static_cast<size_t>(it - yielded.begin()),
                                     store);
        }
      }

      // NOTE: This pass requires ttl.compute to be non-IsolatedFromAbove (the
      // pass emits an error otherwise). We prefer existing views and only
      // materialize a new reserve if none exists.

      // Helper: find a cb_reserve view for the given CB. Prefer in-body
      // reserves; otherwise look in the parent block before the compute.
      auto findViewForCB = [&](Value cb) -> Value {
        // Check inside the compute body.
        for (Operation &op : body.without_terminator()) {
          if (auto reserve = dyn_cast<CBReserveOp>(&op)) {
            if (reserve.getCb() == cb) {
              return reserve.getResult();
            }
          }
        }
        // Check the parent block before the compute op.
        for (Operation &op : *parent) {
          if (&op == computeOperation) {
            break;
          }
          if (auto reserve = dyn_cast<CBReserveOp>(&op)) {
            if (reserve.getCb() == cb) {
              return reserve.getResult();
            }
          }
        }
        return Value();
      };

      // Reorder existing stores after wait in original order.
      Operation *tail = waitOp.getOperation();
      for (StoreOp store : storeOps) {
        store.getOperation()->moveAfter(tail);
        tail = store.getOperation();
      }

      // Insert missing stores for outputs using existing views.
      for (auto [idx, tile] : llvm::enumerate(yielded)) {
        if (storeForOutput.contains(idx)) {
          continue;
        }

        Value cb = getAttachedCB(computeOp.getOutputs()[idx]);
        Value view = findViewForCB(cb);
        if (!view) {
          OpBuilder storeBuilder(terminator);
          storeBuilder.setInsertionPointAfter(tail);
          auto newReserve = storeBuilder.create<CBReserveOp>(
              computeOp.getLoc(), computeOp.getOutputs()[idx].getType(), cb);
          view = newReserve.getResult();
          tail = newReserve.getOperation();
        }

        OpBuilder storeBuilder(terminator);
        storeBuilder.setInsertionPointAfter(tail);
        auto newStore =
            storeBuilder.create<StoreOp>(computeOp.getLoc(), tile, view);
        tail = newStore.getOperation();
        storeForOutput.try_emplace(idx, newStore);
      }

      // Release: after compute in parent block.
      Operation *next = computeOperation->getNextNode();
      OpBuilder afterBuilder(parent, ++Block::iterator(computeOperation));
      if (!isa_and_nonnull<TileRegsReleaseOp>(next)) {
        afterBuilder.create<TileRegsReleaseOp>(computeOp.getLoc());
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
