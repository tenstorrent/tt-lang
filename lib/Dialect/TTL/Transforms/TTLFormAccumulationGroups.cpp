// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Form Accumulation Groups Pass
//===----------------------------------------------------------------------===//
//
// Detects cross-compute accumulation patterns and marks computes that
// participate in accumulation chains with a shared `ttl.acc_group` ID.
//
// After ConvertTTLToCompute, each ttl.store becomes a separate ttl.compute
// with a single tile_store. For accumulation patterns (init + loop, or
// consecutive acc stores), DST Assignment and sync insertion must treat
// multiple computes as a single group sharing an accumulator register.
//
// Phase A: Group Detection
//   Walk the function body, collect computes that store to each view,
//   and form groups based on acc attribute and loop context.
//
// Phase C: Attribute Assignment
//   Set `ttl.acc_group = <id>` on all computes in each group.
//   Convert any non-acc store in the group to acc=true.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-form-accumulation-groups"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFORMACCUMULATIONGROUPS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Information about a compute and its tile_store(s).
struct ComputeStoreInfo {
  ComputeOp compute;
  SmallVector<TileStoreOp> stores;
  /// The top-level ancestor op in the function body block.
  /// For a compute inside an scf.for, this is the scf.for.
  /// For a compute directly in the function body, this is the compute itself.
  Operation *topLevelAncestor = nullptr;
};

/// Check if an scf.for is a user loop (not a tile loop or subblock loop).
static bool isUserLoop(scf::ForOp forOp) {
  return !forOp->hasAttr(kTileLoopAttrName) &&
         !forOp->hasAttr(kSubblockStrideAttrName);
}

/// Find the top-level ancestor of an operation in the given block.
/// Walks up through parent operations until reaching an op whose parent
/// block is the target block.
static Operation *findTopLevelAncestor(Operation *op, Block *targetBlock) {
  Operation *current = op;
  while (current && current->getBlock() != targetBlock) {
    current = current->getParentOp();
  }
  return current;
}

/// Check if an operation is nested inside a user scf.for loop.
static bool isInsideUserLoop(Operation *op, Block *funcBody) {
  Operation *parent = op->getParentOp();
  while (parent) {
    // Check if parent is a user loop before checking if we've reached the
    // function body. The scf.for itself lives in the function body block,
    // but the compute is nested inside it.
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (isUserLoop(forOp)) {
        return true;
      }
    }
    if (parent->getBlock() == funcBody) {
      return false; // Reached function body level without finding user loop.
    }
    parent = parent->getParentOp();
  }
  return false;
}

struct TTLFormAccumulationGroupsPass
    : public impl::TTLFormAccumulationGroupsBase<
          TTLFormAccumulationGroupsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    Block &funcBody = funcOp.getBody().front();

    //=== Phase A: Group Detection ===

    // Step 1: Collect all computes and their store targets.
    // Map: view (cb_reserve result) -> list of compute+store info.
    llvm::MapVector<Value, SmallVector<ComputeStoreInfo>> viewToComputes;

    funcOp.walk([&](ComputeOp computeOp) {
      Block &body = computeOp.getRegion().front();
      SmallVector<TileStoreOp> stores;
      for (Operation &op : body) {
        if (auto store = dyn_cast<TileStoreOp>(&op)) {
          stores.push_back(store);
        }
      }

      if (stores.empty()) {
        return;
      }

      Operation *ancestor = findTopLevelAncestor(computeOp, &funcBody);

      // Group by target view.
      DenseMap<Value, SmallVector<TileStoreOp>> storesByView;
      for (TileStoreOp store : stores) {
        storesByView[store.getView()].push_back(store);
      }

      for (auto &[view, viewStores] : storesByView) {
        ComputeStoreInfo info;
        info.compute = computeOp;
        info.stores = viewStores;
        info.topLevelAncestor = ancestor;
        viewToComputes[view].push_back(info);
      }
    });

    // Step 2: Form groups.
    int64_t nextGroupId = 0;

    for (auto &[view, computeInfos] : viewToComputes) {
      bool hasAccStore = false;
      for (auto &info : computeInfos) {
        for (TileStoreOp store : info.stores) {
          if (store.getAcc()) {
            hasAccStore = true;
            break;
          }
        }
        if (hasAccStore) {
          break;
        }
      }

      if (!hasAccStore) {
        continue; // No acc stores for this view, nothing to do.
      }

      bool needsGroup = false;

      if (computeInfos.size() > 1) {
        // Multi-compute case: at least 2 computes store to the same view.
        needsGroup = true;
        LLVM_DEBUG(llvm::dbgs()
                   << "Phase A: Multi-compute group for view " << view << " ("
                   << computeInfos.size() << " computes)\n");
      } else if (computeInfos.size() == 1) {
        // Single compute: only needs a group if inside a user loop.
        ComputeOp compute = computeInfos[0].compute;
        if (isInsideUserLoop(compute, &funcBody)) {
          needsGroup = true;
          LLVM_DEBUG(llvm::dbgs()
                     << "Phase A: Single-compute-in-loop group for view "
                     << view << "\n");
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Phase A: Skipping single acc store (no loop) for view "
                     << view << "\n");
        }
      }

      if (!needsGroup) {
        continue;
      }

      // Phase 1 restriction: only form groups for 1x1 tile domains.
      // Multi-tile domains require an outer tile loop (Phase 2) to process
      // one tile at a time; without it, all tiles mix in a single DST
      // accumulator.
      auto viewTy = mlir::cast<RankedTensorType>(view.getType());
      bool isSingleTile = true;
      for (int64_t dim = 0; dim < viewTy.getRank() - 1; ++dim) {
        if (viewTy.getDimSize(dim) != 1) {
          isSingleTile = false;
          break;
        }
      }
      if (!isSingleTile) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Phase A: Skipping multi-tile group for view " << view
                   << " (requires Phase 2 outer tile loop)\n");
        continue;
      }

      //=== Phase C: Attribute Assignment ===
      int64_t groupId = nextGroupId++;
      OpBuilder builder(funcOp.getContext());

      for (auto &info : computeInfos) {
        info.compute->setAttr(kAccGroupIdAttrName,
                              builder.getI64IntegerAttr(groupId));
        LLVM_DEBUG(llvm::dbgs() << "Phase C: Set acc_group=" << groupId
                                << " on compute " << info.compute << "\n");

        // Convert any non-acc store to acc=true. The initializer becomes
        // zero-init + add via the accumulation lowering path.
        for (TileStoreOp store : info.stores) {
          if (!store.getAcc()) {
            store.setAcc(true);
            LLVM_DEBUG(llvm::dbgs()
                       << "Phase C: Converted non-acc store to acc=true in "
                          "group "
                       << groupId << "\n");
          }
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Formed " << nextGroupId << " accumulation group(s)\n");
  }
};

} // namespace

} // namespace mlir::tt::ttl
