// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernel Insert L1 Accumulation
//===----------------------------------------------------------------------===//
//
// Inserts pack_reconfig_l1_acc guards inside reduction loops. When a
// tile_regs_acquire is inside a reduction loop, the packer must switch
// to L1 accumulation mode from the second iteration onwards so that
// pack_tile adds to the existing L1 value instead of overwriting.
//
// See docs/development/AccumulatingComputeLowering.md for design details.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "ttkernel-insert-l1-accumulation"

namespace mlir::tt::ttl {

namespace ttk = mlir::tt::ttkernel;

#define GEN_PASS_DEF_TTKERNELINSERTL1ACCUMULATION
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Find the enclosing loop that controls L1 accumulation.
/// Prefers kL1AccLoopAttrName (user-annotated). Falls back to innermost
/// kReductionLoopAttrName (compiler-generated, for reduce ops).
static scf::ForOp findL1AccLoop(Operation *op) {
  scf::ForOp reductionFallback;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (forOp->hasAttr(kL1AccLoopAttrName)) {
        return forOp;
      }
      if (forOp->hasAttr(kReductionLoopAttrName) && !reductionFallback) {
        reductionFallback = forOp;
      }
    }
  }
  return reductionFallback;
}

/// Find the outermost enclosing L1 acc or reduction loop for the disable guard.
static scf::ForOp findOutermostL1AccLoop(Operation *op) {
  scf::ForOp outermost;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (forOp->hasAttr(kL1AccLoopAttrName) ||
          forOp->hasAttr(kReductionLoopAttrName)) {
        outermost = forOp;
      }
    }
  }
  return outermost;
}

struct TTKernelInsertL1AccumulationPass
    : public impl::TTKernelInsertL1AccumulationBase<
          TTKernelInsertL1AccumulationPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Collect L1 acc loops (kL1AccLoopAttrName or kReductionLoopAttrName)
    // that contain pack_tile activity.
    SmallVector<scf::ForOp> l1AccLoops;
    llvm::SmallDenseSet<Operation *> seenLoops;
    moduleOp->walk([&](ttk::TileRegsAcquireOp acquireOp) {
      auto loop = findL1AccLoop(acquireOp);
      if (!loop || !seenLoops.insert(loop).second) {
        return;
      }
      bool hasMaxReduce = false;
      loop->walk([&](ttk::ReduceTileOp reduceOp) {
        if (reduceOp.getReduceType() == ttk::ReduceType::Max) {
          hasMaxReduce = true;
        }
      });
      if (!hasMaxReduce) {
        l1AccLoops.push_back(loop);
      }
    });

    // Insert pack_reconfig_l1_acc matching the tt-metal minimal_matmul
    // pattern: enable at the END of the first K iteration (after all
    // DstSections complete), disable after the loop. The enable guard
    // uses `if (k == lb)` so it fires once when the first iteration
    // finishes, and L1 acc stays enabled for all subsequent iterations.
    llvm::SmallDenseSet<Operation *> disabledLoops;
    for (scf::ForOp loop : l1AccLoops) {
      OpBuilder builder(loop->getContext());
      Location loc = loop.getLoc();

      // Disable L1 acc before the loop to ensure clean state.
      builder.setInsertionPoint(loop);
      Value disablePre = arith::ConstantOp::create(
          builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
      ttk::PackReconfigL1AccOp::create(builder, loc, disablePre);

      // Enable at end of first iteration, matching tt-metal:
      //   if (k_block == 0) { PACK((llk_pack_reconfig_l1_acc(1))); }
      Operation *yield = loop.getBody()->getTerminator();
      builder.setInsertionPoint(yield);
      Value loopIV = loop.getInductionVar();
      Value loopLB = loop.getLowerBound();
      Value isFirstIter = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, loopIV, loopLB);
      auto ifOp = scf::IfOp::create(builder, loc, isFirstIter);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value enableFlag = arith::ConstantOp::create(
          builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
      ttk::PackReconfigL1AccOp::create(builder, loc, enableFlag);

      // Disable after each L1 acc loop to prevent L1 acc state from
      // leaking into outer loops or subsequent code.
      if (disabledLoops.insert(loop.getOperation()).second) {
        // For the outermost loop, place disable after cb_push_back.
        // For inner loops, place directly after the loop.
        auto outermostLoop = findOutermostL1AccLoop(loop);
        bool isOutermost = !outermostLoop || outermostLoop == loop;
        if (isOutermost) {
          // Scan forward for cb_push_back.
          Operation *insertPoint = loop->getNextNode();
          while (insertPoint && !isa<ttk::CBPushBackOp>(insertPoint)) {
            insertPoint = insertPoint->getNextNode();
          }
          if (insertPoint) {
            builder.setInsertionPointAfter(insertPoint);
          } else {
            builder.setInsertionPointAfter(loop);
          }
        } else {
          builder.setInsertionPointAfter(loop);
        }
        Value disableFlag = arith::ConstantOp::create(
            builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
        ttk::PackReconfigL1AccOp::create(builder, loc, disableFlag);
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
