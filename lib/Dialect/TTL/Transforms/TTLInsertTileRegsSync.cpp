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
//    - Inserts tile_regs_release at the end of the loop body (before scf.yield)
//
// This establishes the correct DST lifecycle per tile:
//   acquire -> [compute] -> commit -> wait -> [pack] -> release
//
// The pass requires that all tensor.insert ops in the loop body have a
// corresponding ttl.store op. Missing stores result in an error.
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

/// Find an input CB that is extracted in the loop body, preferring one whose
/// shape matches the output CB shape, or the largest input CB. This handles:
/// - Broadcast ops: prefers full-sized input over broadcast input
/// - Reduction ops: prefers largest input (the one being reduced)
/// - Broadcast + reduction: prefers largest input
///
/// Only considers CBs from AttachCBOp/CBWaitOp, not iter_args (accumulators).
/// Returns nullptr if no CB is found.
static Value findExtractedInputCB(scf::ForOp forOp, Value outputCB) {
  // Get output CB shape for matching.
  ArrayRef<int64_t> outputShape;
  if (outputCB) {
    if (auto cbType = dyn_cast<CircularBufferType>(outputCB.getType())) {
      outputShape = cbType.getShape();
    }
  }

  Value matchingShapeCB;
  Value largestCB;
  int64_t largestSize = 0;

  forOp.walk([&](tensor::ExtractOp extractOp) {
    Value tensor = extractOp.getTensor();
    if (Value cb = getAttachedCB(tensor)) {
      auto cbType = dyn_cast<CircularBufferType>(cb.getType());
      if (!cbType) {
        return WalkResult::advance();
      }

      // Track the largest CB (by total number of elements).
      ArrayRef<int64_t> shape = cbType.getShape();
      int64_t size = 1;
      for (int64_t dim : shape) {
        size *= dim;
      }
      if (size > largestSize) {
        largestSize = size;
        largestCB = cb;
      }

      // Check if this CB's shape matches the output shape.
      if (!matchingShapeCB && !outputShape.empty()) {
        if (shape == outputShape) {
          matchingShapeCB = cb;
        }
      }
    }
    return WalkResult::advance();
  });

  // Prefer: 1) CB matching output shape, 2) largest CB.
  return matchingShapeCB ? matchingShapeCB : largestCB;
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

struct TTLInsertTileRegsSyncPass
    : public impl::TTLInsertTileRegsSyncBase<TTLInsertTileRegsSyncPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    DominanceInfo domInfo(funcOp);

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

      // Check for an unmatched tile_regs_acquire that dominates the loop.
      // An acquire is "unmatched" if there's no release between it and the
      // loop. This would create an unbalanced acquire/release pair since this
      // pass inserts acquire inside the loop body.
      TileRegsAcquireOp unmatchedAcquire = findUnmatchedDominatingAcquire(
          funcOp, outermostLoop.getOperation(), domInfo);
      if (unmatchedAcquire) {
        unmatchedAcquire.emitError()
            << "tile_regs_acquire outside tile loop without matching release; "
            << "sync ops must be inside the loop body";
        return WalkResult::interrupt();
      }

      // Insert init_sfpu before outermost loop if not present.
      // init_sfpu only uses CB metadata (format, num_faces, face_dim) to
      // configure hardware. It does not access CB data. We use the CB that is
      // actually extracted in the loop body, preferring one whose shape matches
      // the output shape. This ensures correct configuration for broadcast ops
      // where inputs have different shapes.
      if (!existingInitSfpu) {
        SmallVector<Value> outputCBs =
            getCBValuesFromLoopAttr(funcOp, forOp, kTileLoopOutputCBsAttrName);

        // Find the input CB that is extracted in the loop body.
        // Prefer one whose shape matches the output CB shape.
        Value outputCB = outputCBs.empty() ? Value() : outputCBs.front();
        Value extractedInputCB = findExtractedInputCB(forOp, outputCB);

        // Error if loop attributes declare input CBs but nothing extracts from
        // them - init_sfpu needs an extracted CB for hardware configuration.
        if (!extractedInputCB) {
          SmallVector<Value> inputCBs =
              getCBValuesFromLoopAttr(funcOp, forOp, kTileLoopInputCBsAttrName);
          if (!inputCBs.empty()) {
            forOp.emitOpError()
                << "init_sfpu: input CBs declared but none extracted in loop "
                   "body; ensure compute body uses its inputs";
            return WalkResult::interrupt();
          }
        }

        if (extractedInputCB && !outputCBs.empty()) {
          builder.setInsertionPoint(outermostLoop);
          builder.create<InitSFPUOp>(loc, extractedInputCB, outputCBs.front());
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

      // Get output CBs from loop attributes for auto-inserting stores.
      SmallVector<Value> outputCBs =
          getCBValuesFromLoopAttr(funcOp, forOp, kTileLoopOutputCBsAttrName);

      // Helper: find an existing cb_reserve view for the given CB.
      // Searches both the loop body and the parent block before the loop.
      auto findExistingReserve = [&](Value cb) -> Value {
        // Scan the loop body for a reserve.
        for (Operation &op : body) {
          if (auto reserve = dyn_cast<CBReserveOp>(&op)) {
            if (reserve.getCb() == cb) {
              return reserve.getResult();
            }
          }
        }

        // Scan the parent block backwards from the outermost loop.
        for (Operation *curr = outermostLoop->getPrevNode(); curr;
             curr = curr->getPrevNode()) {
          if (auto reserve = dyn_cast<CBReserveOp>(curr)) {
            if (reserve.getCb() == cb) {
              return reserve.getResult();
            }
          }
        }

        return Value();
      };

      // Helper: check if a cb_push exists for the given CB after the loop.
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

      // Track CBs that need new reserve/push (created by this pass).
      llvm::DenseMap<Value, Value> newReserveViews; // cb -> reserve view

      // Insert missing stores for tiles that go through tensor.insert.
      // Each tensor.insert corresponds to an output that should be stored.
      // For multi-output computes, we match each insert to its output CB by
      // tracing the insert destination back to the iter_arg block argument.
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

        // Find or create reserve view for this CB.
        Value view = findExistingReserve(cb);
        if (!view) {
          // Check if we already created a reserve for this CB.
          auto it = newReserveViews.find(cb);
          if (it != newReserveViews.end()) {
            view = it->second;
          } else {
            // Create reserve BEFORE outermost loop (not inside loop body).
            auto tensorType = insertOp.getDest().getType();
            builder.setInsertionPoint(outermostLoop);
            auto newReserve = builder.create<CBReserveOp>(loc, tensorType, cb);
            view = newReserve.getResult();
            newReserveViews.try_emplace(cb, view);
          }
        }

        // Store goes INSIDE the loop body, after wait.
        builder.setInsertionPointAfter(tail);
        auto newStore = builder.create<StoreOp>(loc, tile, view);
        tail = newStore.getOperation();
        storeForTile.try_emplace(tile, newStore);
      }

      // Create cb_push AFTER outermost loop for CBs that need it.
      // Only create push if we created a new reserve (no existing
      // reserve/push).
      for (auto &[cb, reserveView] : newReserveViews) {
        if (!hasExistingPush(cb)) {
          builder.setInsertionPointAfter(outermostLoop);
          builder.create<CBPushOp>(loc, cb);
        }
      }

      // Release: at end of loop body (before scf.yield) if not present.
      if (!releaseOp) {
        builder.setInsertionPoint(terminator);
        builder.create<TileRegsReleaseOp>(loc);
      }

      // Clean up marker attributes after processing.
      // Remove from innermost loop.
      forOp->removeAttr(kTileLoopAttrName);
      forOp->removeAttr(kTileLoopInputCBsAttrName);
      forOp->removeAttr(kTileLoopOutputCBsAttrName);
      // Remove from all outer loops (outermostLoop walks up to find them).
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

      // Skip: post-order walk processes innermost first; outer loops already
      // had their markers removed above.
      return WalkResult::skip();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
