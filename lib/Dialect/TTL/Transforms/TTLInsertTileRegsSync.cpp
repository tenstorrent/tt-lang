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

/// Find a bind_cb op with the given cb_index in the function.
static BindCBOp findBindCBByIndex(func::FuncOp funcOp, int64_t cbIndex) {
  BindCBOp result = nullptr;
  funcOp.walk([&](BindCBOp bindOp) {
    if (bindOp.getCbIndex() == cbIndex) {
      result = bindOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

/// Find the outermost scf.for loop containing this operation.
static scf::ForOp findOutermostLoop(Operation *op) {
  scf::ForOp outermost = nullptr;
  Operation *current = op;
  while (auto parentFor = current->getParentOfType<scf::ForOp>()) {
    outermost = parentFor;
    current = parentFor.getOperation();
  }
  return outermost;
}

/// Lookup a CB value from a loop attribute that stores a cb_index.
static FailureOr<Value> getCBFromAttr(func::FuncOp funcOp, scf::ForOp forOp,
                                      llvm::StringRef attrName) {
  auto cbAttr = forOp->getAttrOfType<IntegerAttr>(attrName);
  if (!cbAttr) {
    return failure();
  }
  if (auto bindOp = findBindCBByIndex(funcOp, cbAttr.getInt())) {
    return bindOp.getResult();
  }
  return failure();
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

      // Find outermost loop for init_sfpu placement.
      scf::ForOp outermostLoop = forOp->getParentOfType<scf::ForOp>()
                                     ? findOutermostLoop(forOp)
                                     : forOp;

      // Find existing sync ops preceding the outermost loop.
      auto stopAtLoop = [](Operation *op) { return isa<scf::ForOp>(op); };
      InitSFPUOp existingInitSfpu =
          findPrecedingOp<InitSFPUOp>(outermostLoop.getOperation(), stopAtLoop);

      // Check for an unmatched tile_regs_acquire that dominates the loop.
      // An acquire is "unmatched" if there's no release between it and the
      // loop. This would create an unbalanced acquire/release pair since the
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
      // Use stored CB indices from loop attributes to find the CBs.
      if (!existingInitSfpu) {
        auto icbOrFailure =
            getCBFromAttr(funcOp, forOp, kTileLoopInputCBAttrName);
        auto ocbOrFailure =
            getCBFromAttr(funcOp, forOp, kTileLoopOutputCBAttrName);

        if (failed(icbOrFailure)) {
          forOp.emitError() << "tile loop missing '" << kTileLoopInputCBAttrName
                            << "' attribute required for init_sfpu insertion";
          return WalkResult::interrupt();
        }
        if (failed(ocbOrFailure)) {
          forOp.emitError()
              << "tile loop missing '" << kTileLoopOutputCBAttrName
              << "' attribute required for init_sfpu insertion";
          return WalkResult::interrupt();
        }

        Value icb = *icbOrFailure;
        Value ocb = *ocbOrFailure;
        builder.setInsertionPoint(outermostLoop);
        builder.create<InitSFPUOp>(loc, icb, ocb);
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

      // Verify all tensor.insert ops have corresponding ttl.store ops.
      // Each tensor.insert corresponds to an output that must be stored to a
      // CB.
      for (tensor::InsertOp insertOp : insertOps) {
        Value tile = insertOp.getScalar();
        if (!storeForTile.contains(tile)) {
          insertOp.emitError()
              << "tensor.insert tile has no corresponding ttl.store op; "
              << "all compute outputs must be explicitly stored to a CB";
          return WalkResult::interrupt();
        }
      }

      // Release: at end of loop body (before scf.yield) if not present.
      if (!releaseOp) {
        builder.setInsertionPoint(terminator);
        builder.create<TileRegsReleaseOp>(loc);
      }

      // Clean up marker attributes after processing.
      forOp->removeAttr(kTileLoopAttrName);
      forOp->removeAttr(kTileLoopInputCBAttrName);
      forOp->removeAttr(kTileLoopOutputCBAttrName);

      return WalkResult::skip();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
