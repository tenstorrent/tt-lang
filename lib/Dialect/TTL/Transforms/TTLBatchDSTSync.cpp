// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Batch DST Sync Pass
//===----------------------------------------------------------------------===//
//
// After lower-to-loops, each tile loop iteration has its own DST sync cycle:
//   acquire -> copy -> compute -> commit -> wait -> store -> release
//
// When all tiles fit in DST, this pass fully unrolls the tile loop and
// rewrites to a single batched sync cycle. Each unrolled tile gets a unique
// DST slot, eliminating per-tile sync overhead.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "ttl-batch-dst-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLBATCHDSTSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Default DST capacities by element type (double-buffered mode).
constexpr uint32_t kDSTCapacityF32 = 4;
constexpr uint32_t kDSTCapacityDefault = 8;

//===----------------------------------------------------------------------===//
// Analysis helpers
//===----------------------------------------------------------------------===//

/// Collect the loop nest containing `inner`, innermost first.
static SmallVector<scf::ForOp> collectLoopNest(scf::ForOp inner) {
  SmallVector<scf::ForOp> nest;
  nest.push_back(inner);
  for (Operation *parent = inner->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      nest.push_back(forOp);
  }
  return nest;
}

/// Return constant trip count for a loop with lb=0, step=1, constant ub.
static std::optional<int64_t> getConstantTripCount(scf::ForOp loop) {
  auto lb = getConstantIntValue(loop.getLowerBound());
  auto ub = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  if (!lb || *lb != 0 || !step || *step != 1 || !ub)
    return std::nullopt;
  return *ub;
}

/// Compute total trip count for a loop nest. Returns nullopt on failure.
static std::optional<int64_t>
getTotalTripCount(ArrayRef<scf::ForOp> nest) {
  int64_t total = 1;
  for (scf::ForOp loop : nest) {
    auto tc = getConstantTripCount(loop);
    if (!tc)
      return std::nullopt;
    total *= *tc;
  }
  return total;
}

/// Max dst_idx attribute value + 1 in a block (DST registers per iteration).
static int64_t computeDSTPerIter(scf::ForOp innerLoop) {
  int64_t maxIdx = -1;
  innerLoop.walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName))
      maxIdx = std::max(maxIdx, attr.getInt());
  });
  return maxIdx + 1;
}

/// Infer DST capacity from tile element types in the loop body.
static uint32_t inferDSTCapacity(scf::ForOp innerLoop) {
  bool hasF32 = false;
  innerLoop.walk([&](CopyTileOp op) {
    if (auto tileType = dyn_cast<ttcore::TileType>(op.getSrc().getType()))
      if (tileType.getElementType().isF32())
        hasF32 = true;
  });
  return hasF32 ? kDSTCapacityF32 : kDSTCapacityDefault;
}

/// Return true if any loop in the nest contains ops that are unsafe to batch:
/// CB-input ops (bcast, reduce) or CB lifecycle ops (reserve, push, pop, wait).
static bool hasUnsafeOps(ArrayRef<scf::ForOp> nest) {
  // Check the outermost loop, which covers all ops in the nest.
  scf::ForOp outermost = nest.back();
  bool unsafe = false;
  outermost.walk([&](Operation *op) {
    if (op->hasTrait<TTLCBInputTileOpTrait>())
      unsafe = true;
    if (isa<CBReserveOp, CBPushOp, CBPopOp, CBWaitOp>(op))
      unsafe = true;
  });
  return unsafe;
}

/// Return true if `loop` is the innermost for-loop (no nested scf.for).
static bool isInnermostLoop(scf::ForOp loop) {
  bool hasNested = false;
  loop.walk([&](scf::ForOp nested) {
    if (nested != loop)
      hasNested = true;
  });
  return !hasNested;
}

//===----------------------------------------------------------------------===//
// DST index patching
//===----------------------------------------------------------------------===//

/// Patch dst_idx attribute: newIdx = original + iterK * dstPerIter.
static void patchDstIdxAttr(Operation *op, int64_t iterK, int64_t dstPerIter,
                            OpBuilder &builder) {
  auto attr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName);
  if (!attr)
    return;
  int64_t newIdx = attr.getInt() + iterK * dstPerIter;
  op->setAttr(kDstIdxAttrName,
              builder.getI32IntegerAttr(static_cast<int32_t>(newIdx)));
}

/// Replace copy_tile dst_index operand with a patched constant.
static void patchCopyTileDstIndex(CopyTileOp copyTile, int64_t iterK,
                                  int64_t dstPerIter, OpBuilder &builder) {
  auto constOp =
      copyTile.getDstIndex().getDefiningOp<arith::ConstantIndexOp>();
  if (!constOp)
    return;
  int64_t newIdx = constOp.value() + iterK * dstPerIter;
  builder.setInsertionPoint(copyTile);
  Value newDstIdx =
      builder.create<arith::ConstantIndexOp>(copyTile.getLoc(), newIdx);
  copyTile.getDstIndexMutable().assign(newDstIdx);
}

//===----------------------------------------------------------------------===//
// Sync rewrite
//===----------------------------------------------------------------------===//

/// Walk the unrolled region, patch DST indices per iteration, and rewrite
/// per-tile sync ops into a single batched sync cycle.
static void rewriteUnrolledRegion(Block::iterator startIt,
                                  Block::iterator endIt, int64_t dstPerIter) {
  // First pass: patch indices and collect ops to move/erase.
  OpBuilder builder(startIt->getContext());
  int64_t iterK = -1;
  SmallVector<Operation *> syncOps;
  SmallVector<TileStoreOp> allStores;
  Operation *firstDSTOp = nullptr;

  for (auto it = startIt; it != endIt; ++it) {
    Operation *op = &*it;

    if (isa<TileRegsAcquireOp>(op)) {
      iterK++;
      syncOps.push_back(op);
      continue;
    }
    if (isa<TileRegsCommitOp, TileRegsWaitOp, TileRegsReleaseOp>(op)) {
      syncOps.push_back(op);
      continue;
    }
    if (iterK < 0)
      continue;

    // Patch tile compute ops and copy_dst.
    if (isTileComputeOp(op) || isa<CopyDstOp>(op))
      patchDstIdxAttr(op, iterK, dstPerIter, builder);

    // Patch copy_tile: both dst_idx attribute and dst_index operand.
    if (auto copyTile = dyn_cast<CopyTileOp>(op)) {
      patchDstIdxAttr(op, iterK, dstPerIter, builder);
      patchCopyTileDstIndex(copyTile, iterK, dstPerIter, builder);
      if (!firstDSTOp)
        firstDSTOp = op;
    }

    // Track first DST-using op for acquire placement.
    if (!firstDSTOp && (isTileComputeOp(op) || isa<CopyDstOp>(op)))
      firstDSTOp = op;

    // Set tile_offset on tile_store for downstream CB index computation.
    if (auto store = dyn_cast<TileStoreOp>(op)) {
      store->setAttr(kTileOffsetAttrName, builder.getI64IntegerAttr(iterK));
      allStores.push_back(store);
    }
  }

  if (!firstDSTOp || allStores.empty())
    return;

  // Erase all per-tile sync ops.
  for (Operation *op : llvm::reverse(syncOps))
    op->erase();

  // Move all stores to end of region (before endIt), preserving order.
  Operation *storeAnchor =
      (endIt != startIt->getBlock()->end()) ? &*endIt : nullptr;
  for (TileStoreOp store : allStores) {
    if (storeAnchor)
      store->moveBefore(storeAnchor);
    else
      store->moveAfter(&store->getBlock()->back());
  }

  // Insert batched sync.
  Location loc = firstDSTOp->getLoc();
  builder.setInsertionPoint(firstDSTOp);
  builder.create<TileRegsAcquireOp>(loc);

  builder.setInsertionPoint(allStores.front());
  builder.create<TileRegsCommitOp>(loc);
  builder.create<TileRegsWaitOp>(loc);

  builder.setInsertionPointAfter(allStores.back());
  builder.create<TileRegsReleaseOp>(loc);
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct TTLBatchDSTSyncPass
    : public impl::TTLBatchDSTSyncBase<TTLBatchDSTSyncPass> {
  using Base = impl::TTLBatchDSTSyncBase<TTLBatchDSTSyncPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Collect candidate innermost loops containing DST sync ops.
    SmallVector<scf::ForOp> candidates;
    funcOp.walk([&](TileRegsAcquireOp acquireOp) {
      auto forOp = acquireOp->getParentOfType<scf::ForOp>();
      if (forOp && isInnermostLoop(forOp) &&
          !llvm::is_contained(candidates, forOp))
        candidates.push_back(forOp);
    });

    for (scf::ForOp innerLoop : candidates) {
      SmallVector<scf::ForOp> nest = collectLoopNest(innerLoop);
      if (hasUnsafeOps(nest))
        continue;
      auto totalTrip = getTotalTripCount(nest);
      if (!totalTrip || *totalTrip <= 1)
        continue;

      int64_t dstPerIter = computeDSTPerIter(innerLoop);
      if (dstPerIter <= 0)
        continue;

      uint32_t capacity =
          dstCapacity > 0 ? dstCapacity : inferDSTCapacity(innerLoop);
      if (*totalTrip * dstPerIter > static_cast<int64_t>(capacity))
        continue;

      // Record anchors before unrolling.
      scf::ForOp outerLoop = nest.back();
      Operation *beforeLoop = outerLoop->getPrevNode();
      Operation *afterLoop = outerLoop->getNextNode();
      Block *parentBlock = outerLoop->getBlock();

      // Fully unroll: innermost first.
      bool unrollFailed = false;
      for (scf::ForOp loop : nest) {
        if (failed(loopUnrollFull(loop))) {
          unrollFailed = true;
          break;
        }
      }
      if (unrollFailed)
        continue;

      // Locate the unrolled ops between the anchors.
      auto startIt = beforeLoop
                         ? std::next(Block::iterator(beforeLoop))
                         : parentBlock->begin();
      auto endIt = afterLoop ? Block::iterator(afterLoop)
                             : parentBlock->end();

      rewriteUnrolledRegion(startIt, endIt, dstPerIter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
