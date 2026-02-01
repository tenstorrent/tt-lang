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
//    - Inserts tile_regs_commit before the terminator
//    - Inserts tile_regs_wait after commit
//    - Moves ttl.store ops after wait (stores pack each tile to the output CB)
//    - Inserts tile_regs_release at the end of the loop body (before scf.yield)
//
// This establishes the correct DST lifecycle per tile:
//   acquire -> [compute] -> commit -> wait -> [store/pack] -> release
//
// For tensor.insert ops without explicit stores, the pass auto-inserts:
//   - cb_reserve before the outermost loop
//   - store inside the loop (after wait)
//   - cb_push after the outermost loop
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "ttl-insert-tile-regs-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTTILEREGSSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Find an unconsumed cb_reserve for the given CB that dominates the target.
/// A reserve is "unconsumed" if there is no cb_push for the same CB that:
/// - properly dominates the target (comes before on all paths), AND
/// - is dominated by the reserve (comes after the reserve)
/// Returns the reserve's result Value, or std::nullopt if none found.
static std::optional<Value>
findUnconsumedDominatingReserve(func::FuncOp funcOp, Value cb,
                                Operation *target, DominanceInfo &domInfo) {
  // Collect all push ops for this CB that properly dominate the target.
  SmallVector<CBPushOp> dominatingPushes;
  funcOp.walk([&](CBPushOp push) {
    if (push.getCb() == cb &&
        domInfo.properlyDominates(push.getOperation(), target)) {
      dominatingPushes.push_back(push);
    }
  });

  // Find a reserve for this CB that dominates target and has no intervening
  // push.
  std::optional<Value> result;
  funcOp.walk([&](CBReserveOp reserve) {
    if (reserve.getCb() != cb ||
        !domInfo.properlyDominates(reserve.getOperation(), target)) {
      return;
    }

    // Check if there's a push that "consumes" this reserve:
    // - The push must properly dominate the target (already filtered above)
    // - The push must be dominated by the reserve (comes after it)
    bool hasConsumingPush = llvm::any_of(dominatingPushes, [&](CBPushOp push) {
      return domInfo.dominates(reserve.getOperation(), push.getOperation());
    });

    if (!hasConsumingPush) {
      result = reserve.getResult();
    }
  });

  return result;
}

/// Find an unmatched tile_regs_acquire that dominates the given operation.
/// An acquire is "unmatched" if there is no tile_regs_release that:
/// - properly dominates the target (comes before on all paths), AND
/// - is dominated by the acquire (comes after the acquire)
/// Uses DominanceInfo for correctness across complex control flow.
static TileRegsAcquireOp
findUnmatchedDominatingAcquire(func::FuncOp funcOp, Operation *target,
                               DominanceInfo &domInfo) {
  TileRegsAcquireOp unmatchedAcquire = nullptr;

  // Collect all release ops that properly dominate the target.
  SmallVector<TileRegsReleaseOp> dominatingReleases;
  funcOp.walk([&](TileRegsReleaseOp release) {
    if (domInfo.properlyDominates(release.getOperation(), target)) {
      dominatingReleases.push_back(release);
    }
  });

  // Check each acquire that properly dominates the target.
  funcOp.walk([&](TileRegsAcquireOp acquire) {
    if (!domInfo.properlyDominates(acquire.getOperation(), target)) {
      return;
    }

    // Check if there's a release that "intercepts" this acquire:
    // - The release must properly dominate the target (already filtered above)
    // - The release must be dominated by the acquire (comes after it)
    bool hasMatchingRelease =
        llvm::any_of(dominatingReleases, [&](TileRegsReleaseOp release) {
          return domInfo.dominates(acquire.getOperation(),
                                   release.getOperation());
        });

    if (!hasMatchingRelease) {
      unmatchedAcquire = acquire;
    }
  });

  return unmatchedAcquire;
}

/// Get CB values from a loop's array attribute by looking up bind_cb ops.
/// Returns empty vector if attribute is not present or CBs are not found (e.g.,
/// when the loop is a user loop, not a tile loop generated from a compute op)
static SmallVector<Value> getCBValuesFromLoopAttr(func::FuncOp funcOp,
                                                  scf::ForOp forOp,
                                                  llvm::StringRef attrName) {
  SmallVector<Value> cbs;
  auto cbArrayAttr = forOp->getAttrOfType<ArrayAttr>(attrName);
  if (!cbArrayAttr) {
    return cbs;
  }
  for (Attribute attr : cbArrayAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    assert(intAttr);

    if (auto bindOp = findBindCBByIndex(funcOp, intAttr.getInt())) {
      cbs.push_back(bindOp.getResult());
    }
  }
  return cbs;
}

struct TTLInsertTileRegsSyncPass
    : public impl::TTLInsertTileRegsSyncBase<TTLInsertTileRegsSyncPass> {
  using Base::Base;

  // Steps performed for the tile loop resulting from ttl.compute
  // lowering:
  //   1. Validate: reject unmatched acquire outside loop (would cause
  //   imbalance)
  //   2. init_sfpu: insert before outermost loop for hardware configuration
  //   3. DST sync: insert acquire/commit/wait/release in loop body
  //   4. Validate CB op placement: error if cb_reserve/cb_push inside loop
  //   5. Auto-store: for tensor.inserts without stores, insert reserve (before
  //      loop), store (inside loop), push (after loop)
  //   6. Cleanup: remove tile_loop marker attributes
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    DominanceInfo domInfo(funcOp);

    WalkResult result = funcOp.walk([&](scf::ForOp forOp) -> WalkResult {
      if (!forOp->hasAttr(kTileLoopAttrName)) {
        return WalkResult::advance();
      }

      Location loc = forOp.getLoc();
      Block &body = forOp.getRegion().front();
      OpBuilder builder(forOp);

      scf::ForOp outermostLoop = findOutermostComputeLoop(forOp);

      auto stopAtLoop = [](Operation *op) { return isa<scf::ForOp>(op); };
      InitSFPUOp existingInitSfpu =
          findPrecedingOp<InitSFPUOp>(outermostLoop.getOperation(), stopAtLoop);

      // Step 1: Validate - reject pre-existing acquire without matching
      // release.
      TileRegsAcquireOp unmatchedAcquire = findUnmatchedDominatingAcquire(
          funcOp, outermostLoop.getOperation(), domInfo);
      if (unmatchedAcquire) {
        unmatchedAcquire.emitError()
            << "tile_regs_acquire outside tile loop without matching release; "
            << "sync ops must be inside the loop body";
        return WalkResult::interrupt();
      }

      // Step 2: init_sfpu - configure unpack/pack hardware for tile operations.
      //
      // init_sfpu configures hardware based on the CB's tile data format
      // (element type, num_faces, face_r_dim, tile_size). The CB's shape
      // (number of tiles) doesn't matter - only the tile format does.
      // init_sfpu forwards icb and ocb to unary_op_init_common,
      // which performs the low-level LLK configuration:
      //    tt_metal/include/compute_kernel_api/eltwise_unary/eltwise_unary.h
      //
      // We use the first input CB following tt-metal convention (LHS for
      // binary ops). When CBs have different formats, copy_tile_init is used
      // to reconfigure the hardware dynamically. For broadcast scenarios where
      // inputs have different tile geometries, the hardware is reconfigured
      // dynamically (with unary_bcast_init), so the init CB just needs to
      // provide a valid initial state.
      if (!existingInitSfpu) {
        SmallVector<Value> inputCBs =
            getCBValuesFromLoopAttr(funcOp, forOp, kTileLoopInputCBsAttrName);
        SmallVector<Value> outputCBs =
            getCBValuesFromLoopAttr(funcOp, forOp, kTileLoopOutputCBsAttrName);

        if (!inputCBs.empty() && !outputCBs.empty()) {
          builder.setInsertionPoint(outermostLoop);
          builder.create<InitSFPUOp>(loc, inputCBs.front(), outputCBs.front());
        }
      }

      // Step 3: DST sync - collect existing ops, then insert missing sync ops.
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

      llvm::DenseMap<Value, StoreOp> storeForTile;
      for (StoreOp store : storeOps) {
        storeForTile.try_emplace(store.getTile(), store);
      }

      // Step 4: Validate CB op placement - cb_reserve and cb_push must not be
      // inside the loop body. The correct pattern is: reserve BEFORE loop,
      // store INSIDE loop, push AFTER loop.
      for (CBReserveOp reserve : reserveOps) {
        reserve.emitError()
            << "cb_reserve inside tile loop body; must be placed before "
            << "the loop (reserves space for all tiles at once)";
        return WalkResult::interrupt();
      }
      for (CBPushOp push : pushOps) {
        push.emitError()
            << "cb_push inside tile loop body; must be placed after "
            << "the loop (signals all tiles ready at once)";
        return WalkResult::interrupt();
      }

      // Move existing stores after wait (stores stay inside loop for per-tile
      // packing). Packing multiple tiles simultaneously is a future
      // optimization.
      Operation *tail = waitOp.getOperation();
      for (StoreOp store : storeOps) {
        store.getOperation()->moveAfter(tail);
        tail = store.getOperation();
      }

      // Step 5: Auto-store - insert stores for tensor.inserts that lack them.
      // Pattern: reserve BEFORE loop, store INSIDE loop, push AFTER loop.
      SmallVector<Value> outputCBs =
          getCBValuesFromLoopAttr(funcOp, forOp, kTileLoopOutputCBsAttrName);

      auto findExistingReserve = [&](Value cb) -> std::optional<Value> {
        return findUnconsumedDominatingReserve(
            funcOp, cb, outermostLoop.getOperation(), domInfo);
      };

      auto hasExistingPush = [&](Value cb) -> bool {
        for (Operation *curr = outermostLoop->getNextNode(); curr;
             curr = curr->getNextNode()) {
          if (auto push = dyn_cast<CBPushOp>(curr)) {
            if (push.getCb() == cb) {
              return true;
            }
          }
        }
        return false;
      };

      llvm::DenseMap<Value, Value> newReserveViews; // cb -> reserve view

      // Match each tensor.insert to its output CB via iter_arg index.
      for (tensor::InsertOp insertOp : insertOps) {
        Value tile = insertOp.getScalar();
        if (storeForTile.contains(tile)) {
          continue;
        }

        if (outputCBs.empty()) {
          continue;
        }

        // Determine which output CB this insert corresponds to.
        // For scf.for, block args are: [iv, iter_arg_0, iter_arg_1, ...]
        // The iter_arg index maps directly to the output CB index.
        Value cb;
        Value dest = insertOp.getDest();
        if (auto blockArg = dyn_cast<BlockArgument>(dest)) {
          // iter_arg index = block_arg_number - 1 (subtract 1 for the iv)
          size_t iterArgIdx = blockArg.getArgNumber() - 1;
          if (iterArgIdx < outputCBs.size()) {
            cb = outputCBs[iterArgIdx];
          }
        }

        // Error if we couldn't determine the output CB mapping.
        // This indicates an unexpected IR structure (e.g., insert destination
        // is not a block argument from the loop's iter_args).
        if (!cb) {
          insertOp.emitError()
              << "could not determine output CB for tensor.insert; "
              << "destination must be an iter_arg block argument";
          return WalkResult::interrupt();
        }

        std::optional<Value> existingReserve = findExistingReserve(cb);
        Value view;
        if (existingReserve) {
          view = *existingReserve;
        } else {
          auto it = newReserveViews.find(cb);
          if (it != newReserveViews.end()) {
            view = it->second;
          } else {
            // Reserve before outermost loop (reserves space for all tiles).
            auto tensorType = insertOp.getDest().getType();
            builder.setInsertionPoint(outermostLoop);
            auto newReserve = builder.create<CBReserveOp>(loc, tensorType, cb);
            view = newReserve.getResult();
            newReserveViews.try_emplace(cb, view);
          }
        }

        // Store inside loop (packs each tile).
        builder.setInsertionPointAfter(tail);
        auto newStore = builder.create<StoreOp>(loc, tile, view);
        tail = newStore.getOperation();
        storeForTile.try_emplace(tile, newStore);
      }

      // Push after outermost loop (signals all tiles ready).
      for (auto &[cb, reserveView] : newReserveViews) {
        if (!hasExistingPush(cb)) {
          builder.setInsertionPointAfter(outermostLoop);
          builder.create<CBPushOp>(loc, cb);
        }
      }

      // Release at end of loop body.
      if (!releaseOp) {
        builder.setInsertionPoint(terminator);
        builder.create<TileRegsReleaseOp>(loc);
      }

      // Step 6: Cleanup - remove marker attributes.
      forOp->removeAttr(kTileLoopAttrName);
      forOp->removeAttr(kTileLoopInputCBsAttrName);
      forOp->removeAttr(kTileLoopOutputCBsAttrName);
      if (outermostLoop != forOp) {
        Operation *current = forOp.getOperation();
        while (auto parentFor = current->getParentOfType<scf::ForOp>()) {
          parentFor->removeAttr(kTileLoopAttrName);
          if (parentFor == outermostLoop) {
            break;
          }
          current = parentFor.getOperation();
        }
      }

      // Skip outer loops (already processed via marker removal).
      return WalkResult::skip();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
