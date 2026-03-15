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
//    Sync ops go OUTSIDE the compute body. One sync region covers all tiles
//    in the subblock. Lower-to-loops then unrolls the body into N tile copies
//    that share the single sync region. Store hoisting later moves pack/store
//    ops after the wait barrier.
//    DST sync region per subblock:
//      acquire -> [N x compute] -> commit -> wait -> [N x pack] -> release
//
// 2. Non-subblocked computes (no ttl.full_linearization_strides):
//    Sync ops go INSIDE the compute body (per-tile sync). This is the
//    original behavior for computes that haven't been through the subblocking
//    pass. Lower-to-loops creates tile loops, and each iteration has its own
//    sync region.
//    DST sync region per tile:
//      acquire -> [compute] -> commit -> wait -> [pack] -> release
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "ttl-insert-tile-regs-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTTILEREGSSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLInsertTileRegsSyncPass
    : public impl::TTLInsertTileRegsSyncBase<TTLInsertTileRegsSyncPass> {
  using Base::Base;

  /// Insert group-level sync for accumulation groups. One sync region
  /// wraps all computes in the group (acquire before first, release after
  /// last).
  void insertGroupSync(func::FuncOp funcOp) {
    // Collect computes by group ID.
    DenseMap<int64_t, SmallVector<ComputeOp>> groups;
    funcOp.walk([&](ComputeOp computeOp) {
      if (auto groupAttr =
              computeOp->getAttrOfType<IntegerAttr>(kAccGroupIdAttrName)) {
        groups[groupAttr.getInt()].push_back(computeOp);
      }
    });

    for (auto &[groupId, computes] : groups) {
      if (computes.empty()) {
        continue;
      }

      // Find the common parent block for all computes in the group.
      // Walk up from each compute to find its top-level ancestor in the
      // common block.
      Block *commonBlock = nullptr;
      SmallVector<Operation *> ancestors;

      for (ComputeOp compute : computes) {
        // Find the top-level ancestor in the function body first.
        Operation *ancestor = compute;
        while (ancestor->getParentOp() &&
               !isa<func::FuncOp>(ancestor->getParentOp())) {
          ancestor = ancestor->getParentOp();
        }

        if (!commonBlock) {
          commonBlock = ancestor->getBlock();
        }
        ancestors.push_back(ancestor);
      }

      if (!commonBlock || ancestors.empty()) {
        continue;
      }

      // Find first and last ancestors in block order.
      Operation *firstOp = ancestors[0];
      Operation *lastOp = ancestors[0];
      for (Operation *anc : ancestors) {
        if (anc->isBeforeInBlock(firstOp)) {
          firstOp = anc;
        }
        if (lastOp->isBeforeInBlock(anc)) {
          lastOp = anc;
        }
      }

      Location loc = firstOp->getLoc();
      OpBuilder builder(firstOp);

      // Place acquire before the first op. Mark it as an accumulation group
      // sync so TTKernelInsertInits uses binary_op_init_common.
      builder.setInsertionPoint(firstOp);
      auto acquireOp = TileRegsAcquireOp::create(builder, loc);
      acquireOp->setAttr(kAccGroupSyncAttrName, builder.getUnitAttr());

      // Place commit, wait, release after the last op.
      builder.setInsertionPointAfter(lastOp);
      TileRegsCommitOp::create(builder, loc);
      TileRegsWaitOp::create(builder, loc);
      TileRegsReleaseOp::create(builder, loc);
    }
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Build set of grouped computes so we can skip them in per-compute walk.
    DenseSet<Operation *> groupedComputes;
    funcOp.walk([&](ComputeOp computeOp) {
      if (computeOp->hasAttr(kAccGroupIdAttrName)) {
        groupedComputes.insert(computeOp);
      }
    });

    funcOp.walk([&](ComputeOp computeOp) {
      // Skip grouped computes; they get group-level sync below.
      if (groupedComputes.contains(computeOp)) {
        return;
      }
      Location loc = computeOp.getLoc();

      // Find existing acquire preceding this compute. Stop at another compute
      // op since each compute has its own lifecycle ops.
      auto stopAtCompute = [](Operation *op) { return isa<ComputeOp>(op); };
      TileRegsAcquireOp existingAcquire =
          findPrecedingOp<TileRegsAcquireOp>(computeOp, stopAtCompute);

      OpBuilder builder(computeOp);

      // Check if this compute was processed by the subblocking pass.
      // If so, place sync outside (one cycle per subblock). Otherwise,
      // place sync inside the body (one cycle per tile iteration).
      bool isSubblocked = computeOp->hasAttr(kFullLinStridesAttrName);

      if (isSubblocked) {
        // Outside placement: one sync region covers all tiles in the subblock.
        // After lower-to-loops unrolls the body, all tile ops appear inside
        // the sync region (between acquire and release).
        if (!existingAcquire) {
          builder.setInsertionPoint(computeOp);
          TileRegsAcquireOp::create(builder, loc);
        }

        // Scan for existing sync ops after this compute, stopping at the
        // next compute op since each compute has its own sync lifecycle.
        TileRegsCommitOp existingCommit = nullptr;
        TileRegsWaitOp existingWait = nullptr;
        TileRegsReleaseOp existingRelease = nullptr;
        for (auto it = std::next(Block::iterator(computeOp)),
                  end = computeOp->getBlock()->end();
             it != end; ++it) {
          if (isa<ComputeOp>(&*it)) {
            break;
          }
          if (auto commit = dyn_cast<TileRegsCommitOp>(&*it)) {
            existingCommit = commit;
          } else if (auto wait = dyn_cast<TileRegsWaitOp>(&*it)) {
            existingWait = wait;
          } else if (auto release = dyn_cast<TileRegsReleaseOp>(&*it)) {
            existingRelease = release;
          }
        }

        // Insert commit, wait, release after the compute op if not present.
        builder.setInsertionPointAfter(computeOp);
        if (!existingCommit) {
          TileRegsCommitOp::create(builder, loc);
        }
        if (!existingWait) {
          TileRegsWaitOp::create(builder, loc);
        }
        if (!existingRelease) {
          TileRegsReleaseOp::create(builder, loc);
        }
      } else {
        // Inside placement: per-tile sync.
        Block &body = computeOp.getRegion().front();
        auto *terminator = body.getTerminator();

        // Scan the body for existing sync and store ops.
        SmallVector<TileStoreOp> storeOps;
        TileRegsAcquireOp acquireOp = nullptr;
        TileRegsCommitOp commitOp = nullptr;
        TileRegsWaitOp waitOp = nullptr;
        TileRegsReleaseOp releaseOp = nullptr;
        for (Operation &op : body.without_terminator()) {
          if (auto store = dyn_cast<TileStoreOp>(&op)) {
            storeOps.push_back(store);
          } else if (auto acquire = dyn_cast<TileRegsAcquireOp>(&op)) {
            acquireOp = acquire;
          } else if (auto commit = dyn_cast<TileRegsCommitOp>(&op)) {
            commitOp = commit;
          } else if (auto wait = dyn_cast<TileRegsWaitOp>(&op)) {
            waitOp = wait;
          } else if (auto release = dyn_cast<TileRegsReleaseOp>(&op)) {
            releaseOp = release;
          }
        }

        // Acquire: at start of compute body.
        if (!existingAcquire && !acquireOp) {
          builder.setInsertionPointToStart(&body);
          TileRegsAcquireOp::create(builder, loc);
        }

        // Insert commit + wait before the first non-accumulating tile_store.
        // Accumulating stores (identified by acc_dst_idx from DST assignment)
        // are compute ops (add_binary_tile), not pack ops, so they stay in
        // the compute phase.
        auto firstNonAccStore = llvm::find_if(storeOps, [](TileStoreOp store) {
          return !isAccumulatingStore(store);
        });
        if (firstNonAccStore != storeOps.end()) {
          builder.setInsertionPoint(*firstNonAccStore);
        } else if (!storeOps.empty()) {
          // All stores are accumulating: commit+wait goes after the last
          // acc store (before terminator or release).
          builder.setInsertionPointAfter(storeOps.back());
        } else {
          builder.setInsertionPoint(terminator);
        }

        if (!commitOp) {
          commitOp = TileRegsCommitOp::create(builder, loc);
        }
        if (!waitOp) {
          waitOp = TileRegsWaitOp::create(builder, loc);
        }
        if (!commitOp->isBeforeInBlock(waitOp)) {
          commitOp->moveBefore(waitOp);
        }

        // Release: at end of compute body (before yield).
        if (!releaseOp) {
          builder.setInsertionPoint(terminator);
          TileRegsReleaseOp::create(builder, loc);
        }
      }
    });

    // Insert group-level sync for accumulation groups.
    insertGroupSync(funcOp);
  }
};

} // namespace

} // namespace mlir::tt::ttl
