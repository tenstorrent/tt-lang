// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Tile Regs Sync Pass
//===----------------------------------------------------------------------===//
//
// This pass inserts DST register synchronization operations inside scf.for
// loop bodies (marked with ttl.tile_loop) to enforce the MATH/PACK thread
// synchronization protocol required by the hardware DST register bank.
//
// The pass runs AFTER ttl-lower-to-loops and operates on loops marked with
// the ttl.tile_loop attribute. It performs the following transformations:
//    - Inserts init_sfpu before the outermost loop (if not present)
//    - Inserts tile_regs_acquire at the beginning of the innermost loop body
//    - Inserts tile_regs_commit immediately before ttl.store
//    - Inserts tile_regs_wait before ttl.store
//    - Inserts ttl.store for outputs that lack one (requires existing CB view)
//    - Inserts tile_regs_release at the end of the loop body (before scf.yield)
//
// This establishes the correct DST lifecycle per tile:
//   acquire -> [compute] -> commit -> wait -> [pack] -> release
//
// The pass is designed to run once during lowering; it does not check for
// existing sync ops.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "ttl-insert-tile-regs-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTTILEREGSSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Find a cb_reserve view for auto-inserted stores. Searches for cb_reserve
/// ops in the loop body and in the parent block before the outermost loop.
static Value findReserveViewForStore(scf::ForOp forOp, scf::ForOp outermostLoop,
                                     Value cb, Operation *insertAfter) {
  Value candidate;
  Block &body = forOp.getRegion().front();

  // Scan the loop body up to (but not including) insertAfter.
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

  // Scan the parent block backwards from the outermost loop.
  Operation *loopOp = outermostLoop.getOperation();
  for (Operation *curr = loopOp->getPrevNode(); curr;
       curr = curr->getPrevNode()) {
    if (auto reserve = dyn_cast<CBReserveOp>(curr)) {
      if (reserve.getCb() == cb) {
        return reserve.getResult();
      }
    }
  }

  return Value();
}

struct TTLInsertTileRegsSyncPass
    : public impl::TTLInsertTileRegsSyncBase<TTLInsertTileRegsSyncPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    WalkResult result = funcOp.walk([&](scf::ForOp forOp) -> WalkResult {
      // Only process loops marked with the tile loop attribute.
      if (!forOp->hasAttr(kTileLoopAttrName)) {
        return WalkResult::advance();
      }

      Location loc = forOp.getLoc();
      Block &body = forOp.getRegion().front();
      OpBuilder builder(forOp);

      // Find outermost compute loop for init_sfpu placement.
      scf::ForOp outermostLoop = findOutermostComputeLoop(forOp);

      // Find existing sync ops preceding the outermost loop.
      auto stopAtLoop = [](Operation *op) { return isa<scf::ForOp>(op); };
      InitSFPUOp existingInitSfpu =
          findPrecedingOp<InitSFPUOp>(outermostLoop.getOperation(), stopAtLoop);

      // Check if the loop body contains broadcast ops (have bcast_dim
      // attribute). Broadcast operations use their own init functions, not
      // init_sfpu.
      bool hasBroadcastOps = false;
      forOp.walk([&](Operation *innerOp) {
        if (innerOp->hasAttr("bcast_dim")) {
          hasBroadcastOps = true;
        }
      });

      // Insert init_sfpu before outermost loop if not present.
      // Use first input/output CB for init_sfpu (hardware only needs one pair).
      // Skip for broadcast ops - they use their own bcast init functions.
      if (!existingInitSfpu && !hasBroadcastOps) {
        SmallVector<Value> inputCBs =
            getCBValuesFromLoopAttr(funcOp, forOp, kTileLoopInputCBsAttrName);
        SmallVector<Value> outputCBs =
            getCBValuesFromLoopAttr(funcOp, forOp, kTileLoopOutputCBsAttrName);

        if (!inputCBs.empty() && !outputCBs.empty()) {
          builder.setInsertionPoint(outermostLoop);
          builder.create<InitSFPUOp>(loc, inputCBs.front(), outputCBs.front());
        }
      }

      // Collect operations in the loop body.
      SmallVector<StoreOp> storeOps;
      SmallVector<CBReserveOp> reserveOps;
      SmallVector<CBPushOp> pushOps;
      SmallVector<tensor::InsertOp> insertOps;
      TileRegsAcquireOp acquireOp = nullptr;
      TileRegsCommitOp commitOp = nullptr;
      TileRegsWaitOp waitOp = nullptr;
      TileRegsReleaseOp releaseOp = nullptr;

      for (Operation &op : body.without_terminator()) {
        TypeSwitch<Operation *>(&op)
            .Case<StoreOp>([&](auto store) { storeOps.push_back(store); })
            .Case<CBReserveOp>(
                [&](auto reserve) { reserveOps.push_back(reserve); })
            .Case<CBPushOp>([&](auto push) { pushOps.push_back(push); })
            .Case<tensor::InsertOp>(
                [&](auto insert) { insertOps.push_back(insert); })
            .Case<TileRegsAcquireOp>([&](auto acquire) { acquireOp = acquire; })
            .Case<TileRegsCommitOp>([&](auto commit) { commitOp = commit; })
            .Case<TileRegsWaitOp>([&](auto wait) { waitOp = wait; })
            .Case<TileRegsReleaseOp>(
                [&](auto release) { releaseOp = release; });
      }

      auto *terminator = body.getTerminator();

      // Insert acquire at start of loop body if not present.
      if (!acquireOp) {
        builder.setInsertionPointToStart(&body);
        builder.create<TileRegsAcquireOp>(loc);
      }

      // Insert commit and wait before terminator.
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

      // Track which tiles (from tensor.insert) have stores.
      // After loop lowering, tensor.insert ops correspond to outputs.
      llvm::DenseMap<Value, StoreOp> storeForTile;
      for (StoreOp store : storeOps) {
        storeForTile.try_emplace(store.getTile(), store);
      }

      // Reorder existing cb_reserve, stores, and cb_push after wait in original
      // order. Order: commit -> wait -> [cb_reserve -> store -> cb_push] ->
      // yield
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

      // Auto-insert stores for tiles being inserted into tensors that lack
      // explicit stores. Each tensor.insert corresponds to an output.
      // Match each tensor.insert to its corresponding output CB by finding
      // which iter_arg it writes to.
      SmallVector<Value> outputCBs =
          getCBValuesFromLoopAttr(funcOp, forOp, kTileLoopOutputCBsAttrName);

      // Build a map from iter_arg (output tensor) to output CB index.
      // The iter_args are in the same order as the ComputeOp outputs.
      llvm::DenseMap<Value, size_t> iterArgToOutputIdx;
      for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        iterArgToOutputIdx[iterArg] = idx;
      }

      for (tensor::InsertOp insertOp : insertOps) {
        Value tile = insertOp.getScalar();
        if (storeForTile.contains(tile)) {
          continue;
        }

        // Find which output CB corresponds to this tensor.insert.
        // The dest of tensor.insert is an iter_arg, which maps to an output.
        Value destTensor = insertOp.getDest();
        auto it = iterArgToOutputIdx.find(destTensor);
        if (it == iterArgToOutputIdx.end()) {
          continue; // Not writing to an iter_arg, skip
        }
        size_t outputIdx = it->second;
        if (outputIdx >= outputCBs.size()) {
          continue; // No CB for this output, skip
        }
        Value cb = outputCBs[outputIdx];

        Value view = findReserveViewForStore(forOp, outermostLoop, cb,
                                             tail->getNextNode());
        if (!view) {
          // Create a new cb_reserve.
          builder.setInsertionPointAfter(tail);
          auto newReserve =
              builder.create<CBReserveOp>(loc, destTensor.getType(), cb);
          view = newReserve.getResult();
          tail = newReserve.getOperation();
        }

        builder.setInsertionPointAfter(tail);
        auto newStore = builder.create<StoreOp>(loc, tile, view);
        tail = newStore.getOperation();
        storeForTile.try_emplace(tile, newStore);
      }

      // Release: at end of loop body (before scf.yield) if not present.
      if (!releaseOp) {
        builder.setInsertionPoint(terminator);
        builder.create<TileRegsReleaseOp>(loc);
      }

      // Clean up marker attributes after processing.
      forOp->removeAttr(kTileLoopAttrName);
      forOp->removeAttr(kTileLoopInputCBsAttrName);
      forOp->removeAttr(kTileLoopOutputCBsAttrName);
      outermostLoop->removeAttr(kTileLoopOuterAttrName);

      return WalkResult::skip();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
