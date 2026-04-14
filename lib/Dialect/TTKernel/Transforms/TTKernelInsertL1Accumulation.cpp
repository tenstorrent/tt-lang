// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernel Insert L1 Accumulation
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

    // L1 accumulation guard placement. For any loop that
    // accumulates in L1 (matmul K loop or reduce loop), the pattern is:
    //
    //   pack_reconfig_l1_acc(0)                // disable before loop
    //   for (iv = lb; ...) {
    //     [subblock 0: acquire...pack...release]
    //     [subblock N: acquire...pack...release]
    //     if (iv == lb) pack_reconfig_l1_acc(1) // enable once after first
    //                                           // iteration's last pack
    //   }
    //   [cb_push_back if present]
    //   pack_reconfig_l1_acc(0)                // disable after loop
    //
    // The L1 acc state persists across multiple dst sections, so the enable
    // call only needs to happen once (after the first iteration completes
    // all its packs). Disable guards are inserted once per outermost
    // reduction loop (parallel loops are not considered).

    // Find the insertion point for the enable guard: the top-level op in
    // the loop body that contains the last tile_regs_release.
    auto findTopLevelAncestor = [](Operation *op,
                                   Block *loopBody) -> Operation * {
      while (op && op->getBlock() != loopBody) {
        op = op->getParentOp();
      }
      return op;
    };

    llvm::SmallDenseMap<Operation *, Operation *> enablePointPerLoop;
    for (auto loop : l1AccLoops) {
      Operation *lastTopLevel = nullptr;
      loop->walk([&](ttk::TileRegsReleaseOp releaseOp) {
        Operation *topLevel = findTopLevelAncestor(releaseOp, loop.getBody());
        if (topLevel) {
          lastTopLevel = topLevel;
        }
      });
      if (lastTopLevel) {
        enablePointPerLoop[loop.getOperation()] = lastTopLevel;
      }
    }

    llvm::SmallDenseSet<Operation *> disabledLoops;
    for (auto loop : l1AccLoops) {
      auto iter = enablePointPerLoop.find(loop.getOperation());
      if (iter == enablePointPerLoop.end()) {
        continue;
      }
      Operation *enablePoint = iter->second;
      OpBuilder builder(loop->getContext());
      Location loc = enablePoint->getLoc();

      // Enable L1 acc once, at the end of the first iteration of the
      // reduction loop. All packs in iteration 0 write without
      // accumulation; subsequent iterations add to the existing L1 value.
      builder.setInsertionPointAfter(enablePoint);
      Value loopIV = loop.getInductionVar();
      Value loopLB = loop.getLowerBound();
      Value firstIter = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, loopIV, loopLB);
      auto ifOp = scf::IfOp::create(builder, loc, firstIter);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value enableFlag = arith::ConstantOp::create(
          builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
      ttk::PackReconfigL1AccOp::create(builder, loc, enableFlag);

      // Bracket the outermost accumulation loop with disable guards.
      // Both kL1AccLoopAttrName and kReductionLoopAttrName mean "all
      // iterations write to the same CB slot," so the outermost such
      // loop is the correct accumulation boundary.
      auto outermostLoop = findOutermostL1AccLoop(loop);
      if (!outermostLoop) {
        outermostLoop = loop;
      }
      if (disabledLoops.insert(outermostLoop.getOperation()).second) {
        Location disableLoc = outermostLoop->getLoc();
        // Disable before the loop.
        builder.setInsertionPoint(outermostLoop);
        Value disablePre =
            arith::ConstantOp::create(builder, disableLoc, builder.getI32Type(),
                                      builder.getI32IntegerAttr(0));
        ttk::PackReconfigL1AccOp::create(builder, disableLoc, disablePre);

        // Disable after any consecutive cb_push_back ops that follow the
        // loop. Multi-output computes produce one push per output CB.
        Operation *lastPush = nullptr;
        for (Operation *op = outermostLoop->getNextNode();
             op && isa<ttk::CBPushBackOp>(op); op = op->getNextNode()) {
          lastPush = op;
        }
        if (lastPush) {
          builder.setInsertionPointAfter(lastPush);
        } else {
          builder.setInsertionPointAfter(outermostLoop);
        }
        Value disablePost =
            arith::ConstantOp::create(builder, disableLoc, builder.getI32Type(),
                                      builder.getI32IntegerAttr(0));
        ttk::PackReconfigL1AccOp::create(builder, disableLoc, disablePost);
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
