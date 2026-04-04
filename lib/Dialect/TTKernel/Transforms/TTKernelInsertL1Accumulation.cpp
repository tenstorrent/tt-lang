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

/// Find the innermost enclosing reduction loop for an operation.
static scf::ForOp findInnermostReductionLoop(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (forOp->hasAttr(kReductionLoopAttrName)) {
        return forOp;
      }
    }
  }
  return nullptr;
}

/// Find the outermost enclosing reduction loop for an operation.
static scf::ForOp findOutermostReductionLoop(Operation *op) {
  scf::ForOp outermost;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (forOp->hasAttr(kReductionLoopAttrName)) {
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

    // Collect all acquire ops inside reduction loops. Collecting first
    // avoids invalidation issues from modifying IR during iteration.
    SmallVector<std::pair<ttk::TileRegsAcquireOp, scf::ForOp>> targets;
    moduleOp->walk([&](ttk::TileRegsAcquireOp acquireOp) {
      auto reductionLoop = findInnermostReductionLoop(acquireOp);
      if (!reductionLoop) {
        return;
      }
      // L1 accumulation uses additive packing -- only valid for sum
      // reductions. Max reductions require DST accumulation (Phase 2)
      // where the hardware max operation accumulates across iterations.
      bool hasMaxReduce = false;
      reductionLoop->walk([&](ttk::ReduceTileOp reduceOp) {
        if (reduceOp.getReduceType() == ttk::ReduceType::Max) {
          hasMaxReduce = true;
        }
      });
      if (!hasMaxReduce) {
        targets.emplace_back(acquireOp, reductionLoop);
      }
    });

    llvm::SmallDenseSet<Operation *> disabledLoops;
    for (auto [acquireOp, reductionLoop] : targets) {
      OpBuilder builder(acquireOp->getContext());
      builder.setInsertionPointAfter(acquireOp);
      Location loc = acquireOp.getLoc();

      // Guard: if (loop_iv != lower_bound) pack_reconfig_l1_acc(1)
      Value loopIV = reductionLoop.getInductionVar();
      Value loopLB = reductionLoop.getLowerBound();
      Value notFirstIter = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::ne, loopIV, loopLB);
      auto ifOp = scf::IfOp::create(builder, loc, notFirstIter);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value enableFlag = arith::ConstantOp::create(
          builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
      ttk::PackReconfigL1AccOp::create(builder, loc, enableFlag);

      // Disable L1 accumulation after the outermost reduction loop.
      auto outermostLoop = findOutermostReductionLoop(acquireOp);
      if (disabledLoops.insert(outermostLoop).second) {
        builder.setInsertionPointAfter(outermostLoop);
        Value disableFlag = arith::ConstantOp::create(
            builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
        ttk::PackReconfigL1AccOp::create(builder, loc, disableFlag);
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
