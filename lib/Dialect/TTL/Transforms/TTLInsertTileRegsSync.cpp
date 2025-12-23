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
// The pass is designed to run once during lowering; it does not check for
// existing sync ops.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
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
      // Also insert init_sfpu before tile_regs_acquire if CBs are available.
      Operation *prev = computeOperation->getPrevNode();
      if (!isa_and_nonnull<TileRegsAcquireOp>(prev)) {
        OpBuilder beforeBuilder(parent, Block::iterator(computeOperation));

        // Get first input CB and output CB for init_sfpu.
        Value icb, ocb;
        if (!computeOp.getInputs().empty()) {
          icb = getAttachedCB(computeOp.getInputs()[0]);
        }
        if (!computeOp.getOutputs().empty()) {
          ocb = getAttachedCB(computeOp.getOutputs()[0]);
        }
        if (icb && ocb) {
          beforeBuilder.create<InitSFPUOp>(computeOp.getLoc(), icb, ocb);
        }

        beforeBuilder.create<TileRegsAcquireOp>(computeOp.getLoc());
      }

      Block &body = computeOp.getRegion().front();
      auto *terminator = body.getTerminator();
      auto yieldOp = cast<YieldOp>(terminator);

      // Collect existing sync, store, cb_reserve, and cb_push ops.
      SmallVector<StoreOp> storeOps;
      SmallVector<CBReserveOp> reserveOps;
      SmallVector<CBPushOp> pushOps;
      TileRegsCommitOp commitOp = nullptr;
      TileRegsWaitOp waitOp = nullptr;
      for (Operation &op : body.without_terminator()) {
        TypeSwitch<Operation *>(&op)
            .Case<StoreOp>([&](auto store) { storeOps.push_back(store); })
            .Case<CBReserveOp>(
                [&](auto reserve) { reserveOps.push_back(reserve); })
            .Case<CBPushOp>([&](auto push) { pushOps.push_back(push); })
            .Case<TileRegsCommitOp>([&](auto commit) { commitOp = commit; })
            .Case<TileRegsWaitOp>([&](auto wait) { waitOp = wait; });
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

      // Helper: find a cb_reserve view for the given CB that dominates the
      // store insertion point. Prefer the last in-body reserve before the
      // insertion; otherwise use the nearest reserve in the parent block before
      // the compute.
      auto findViewForCB = [&](Value cb, Operation *insertAfter) -> Value {
        Value candidate;

        // Scan the compute body up to (but not including) insertAfter.
        for (Operation &op : body) {
          if (&op == insertAfter) {
            break;
          }
          if (auto reserve = dyn_cast<CBReserveOp>(&op)) {
            if (reserve.getCb() == cb) {
              candidate = reserve.getResult(); // keep the last dominating one
            }
          }
        }
        if (candidate) {
          return candidate;
        }

        // Scan the parent block backwards from the compute op.
        for (Operation *curr = computeOperation->getPrevNode(); curr;
             curr = curr->getPrevNode()) {
          if (auto reserve = dyn_cast<CBReserveOp>(curr)) {
            if (reserve.getCb() == cb) {
              return reserve.getResult();
            }
          }
        }

        return Value();
      };

      // Reorder existing cb_reserve, stores, and cb_push after wait in original
      // order. Order: commit → wait → [cb_reserve → store → cb_push] → yield
      Operation *tail = waitOp.getOperation();
      for (CBReserveOp reserve : reserveOps) {
        reserve.getOperation()->moveAfter(tail);
        tail = reserve.getOperation();
      }
      for (StoreOp store : storeOps) {
        store.getOperation()->moveAfter(tail);
        tail = store.getOperation();
      }
      for (CBPushOp push : pushOps) {
        push.getOperation()->moveAfter(tail);
        tail = push.getOperation();
      }

      // Insert missing stores for outputs using existing views.
      for (auto [idx, tile] : llvm::enumerate(yielded)) {
        if (storeForOutput.contains(idx)) {
          continue;
        }

        Value cb = getAttachedCB(computeOp.getOutputs()[idx]);
        Value view = findViewForCB(cb, tail->getNextNode());
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
