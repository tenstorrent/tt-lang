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

      if (!existingAcquire) {
        builder.setInsertionPoint(computeOperation);
        builder.create<TileRegsAcquireOp>(loc);
      }

      Block &body = computeOp.getRegion().front();
      auto *terminator = body.getTerminator();
      auto yieldOp = cast<YieldOp>(terminator);
      OperandRange yieldedValues = yieldOp.getValues();

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

      builder.setInsertionPoint(terminator);
      if (!commitOp) {
        commitOp = builder.create<TileRegsCommitOp>(loc);
      }
      if (!waitOp) {
        waitOp = builder.create<TileRegsWaitOp>(loc);
      }
      if (!commitOp->isBeforeInBlock(waitOp)) {
        commitOp->moveBefore(waitOp);
      }

      // NOTE: try_emplace ensures only the first store per output is tracked.
      // The ComputeOp verifier already rejects duplicate stores per output,
      // so at most one store exists per output index.
      llvm::DenseMap<size_t, StoreOp> storeForOutput;
      for (StoreOp store : storeOps) {
        auto it = llvm::find(yieldedValues, store.getTile());
        if (it != yieldedValues.end()) {
          storeForOutput.try_emplace(
              static_cast<size_t>(it - yieldedValues.begin()), store);
        }
      }

      // NOTE: This pass requires ttl.compute to be non-IsolatedFromAbove (the
      // pass emits an error otherwise). We prefer existing views and only
      // materialize a new reserve if none exists.

      // Helper: find a cb_reserve view for auto-inserted stores. Only
      // cb_reserve views are valid (verifier rejects cb_wait). Prefers the last
      // in-body reserve before the insertion point; otherwise uses the nearest
      // reserve in the parent block before the compute.
      auto findReserveViewForStore = [&](Value cb,
                                         Operation *insertAfter) -> Value {
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
      // NOTE: Relative ordering of stores across different outputs is preserved
      // (moveAfter maintains order). Stores to different CBs are independent.
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
      for (auto [idx, tile] : llvm::enumerate(yieldedValues)) {
        if (storeForOutput.contains(idx)) {
          continue;
        }

        Value cb = getAttachedCB(computeOp.getOutputs()[idx]);
        Value view = findReserveViewForStore(cb, tail->getNextNode());
        if (!view) {
          builder.setInsertionPointAfter(tail);
          auto newReserve = builder.create<CBReserveOp>(
              loc, computeOp.getOutputs()[idx].getType(), cb);
          view = newReserve.getResult();
          tail = newReserve.getOperation();
        }

        builder.setInsertionPointAfter(tail);
        auto newStore = builder.create<StoreOp>(loc, tile, view);
        tail = newStore.getOperation();
        storeForOutput.try_emplace(idx, newStore);
      }

      // Release: after compute in parent block.
      Operation *next = computeOperation->getNextNode();
      if (!isa_and_nonnull<TileRegsReleaseOp>(next)) {
        builder.setInsertionPointAfter(computeOperation);
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
