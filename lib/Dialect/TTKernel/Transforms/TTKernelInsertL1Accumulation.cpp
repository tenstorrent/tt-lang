// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernel Insert L1 Accumulation
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
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

/// Find the innermost enclosing L1 acc or reduction loop.
/// User-written += loops (kL1AccLoopAttrName) take precedence over
/// compiler-generated reduction loops because the user-specified loop
/// structure determines the accumulation granularity.
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

/// Walk from loop up through parent ops, returning the outermost
/// annotated ancestor. Returns loop itself if no annotated ancestor exists.
static scf::ForOp findOutermostAnnotatedAncestor(scf::ForOp loop) {
  scf::ForOp outermost = loop;
  for (Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
      if (parentFor->hasAttr(kL1AccLoopAttrName) ||
          parentFor->hasAttr(kReductionLoopAttrName)) {
        outermost = parentFor;
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

    // Walk from TileRegsAcquireOp upward to find annotated loops —
    // only loops with actual pack activity need L1 acc guards.
    SmallVector<scf::ForOp> l1AccLoops;
    llvm::SmallDenseSet<Operation *> visitedLoops;
    moduleOp->walk([&](ttk::TileRegsAcquireOp acquireOp) {
      auto loop = findL1AccLoop(acquireOp);
      if (!loop || !visitedLoops.insert(loop).second) {
        return;
      }
      // Skip if this pass already ran (idempotency).
      bool alreadyProcessed = false;
      loop->walk([&](ttk::PackReconfigL1AccOp) {
        alreadyProcessed = true;
        return WalkResult::interrupt();
      });
      if (alreadyProcessed) {
        return;
      }
      // Max reduce is not additive — L1 acc would corrupt the running max.
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

    // The enable guard goes after the last pack in the first iteration.
    // Packs live inside tile_regs_acquire/release sections, which may be
    // nested in subblock loops. The top-level ancestor of the last release
    // in the loop body is the correct insertion point.
    llvm::SmallDenseMap<Operation *, Operation *> l1AccEnablePoint;
    for (auto loop : l1AccLoops) {
      Operation *lastReleaseAncestor = nullptr;
      loop->walk([&](ttk::TileRegsReleaseOp releaseOp) {
        if (auto *ancestor =
                loop.getBody()->findAncestorOpInBlock(*releaseOp)) {
          lastReleaseAncestor = ancestor;
        }
      });
      if (lastReleaseAncestor) {
        l1AccEnablePoint[loop.getOperation()] = lastReleaseAncestor;
      }
    }

    // Step 1: Group loops into accumulation scopes. Consecutive sibling
    // loops that pack to the same CB share a single disable pair. Nested
    // annotated loops are folded into the outermost ancestor.
    struct AccGroup {
      scf::ForOp rootLoop;
      SmallVector<scf::ForOp> loops;
      Operation *scopeEnd = nullptr;
    };
    SmallVector<AccGroup> groups;
    llvm::SmallDenseSet<Operation *> assignedToGroup;

    for (auto loop : l1AccLoops) {
      if (!l1AccEnablePoint.count(loop.getOperation())) {
        continue;
      }
      if (assignedToGroup.contains(loop.getOperation())) {
        continue;
      }

      scf::ForOp rootLoop = findOutermostAnnotatedAncestor(loop);

      AccGroup group;
      group.rootLoop = rootLoop;
      group.loops.push_back(loop);
      assignedToGroup.insert(loop.getOperation());

      // Collect sibling annotated loops that share a pack CB target.
      for (Operation *op = rootLoop->getNextNode(); op;
           op = op->getNextNode()) {
        if (isa<ttk::CBPushBackOp>(op)) {
          break;
        }
        auto sibling = dyn_cast<scf::ForOp>(op);
        if (!sibling) {
          continue;
        }
        if (!sibling->hasAttr(kL1AccLoopAttrName) &&
            !sibling->hasAttr(kReductionLoopAttrName)) {
          break;
        }
        if (!sharePackCB(rootLoop, sibling)) {
          break;
        }
        group.loops.push_back(sibling);
        assignedToGroup.insert(sibling.getOperation());
      }

      // Scope ends at the last trailing cb_push_back.
      Operation *lastInGroup = group.loops.size() > 1
                                   ? group.loops.back().getOperation()
                                   : rootLoop.getOperation();
      group.scopeEnd = lastInGroup;
      for (Operation *op = lastInGroup->getNextNode(); op;
           op = op->getNextNode()) {
        if (isa<ttk::CBPushBackOp>(op)) {
          group.scopeEnd = op;
        } else {
          break;
        }
      }

      groups.push_back(std::move(group));
    }

    // Step 2: Emit guards per group.
    for (auto &group : groups) {
      OpBuilder builder(group.rootLoop->getContext());
      Location disableLoc = group.rootLoop->getLoc();

      // Disable before the group.
      builder.setInsertionPoint(group.rootLoop);
      Value disableFlag =
          arith::ConstantOp::create(builder, disableLoc, builder.getI32Type(),
                                    builder.getI32IntegerAttr(0));
      ttk::PackReconfigL1AccOp::create(builder, disableLoc, disableFlag);

      for (size_t idx = 0; idx < group.loops.size(); ++idx) {
        scf::ForOp loop = group.loops[idx];
        auto iter = l1AccEnablePoint.find(loop.getOperation());
        if (iter == l1AccEnablePoint.end()) {
          continue;
        }

        // For the 2nd+ loop in a group, re-enable L1 acc before
        // the loop because init ops between loops reset packer state.
        if (idx > 0) {
          builder.setInsertionPoint(loop);
          Value enableFlag = arith::ConstantOp::create(
              builder, loop->getLoc(), builder.getI32Type(),
              builder.getI32IntegerAttr(1));
          ttk::PackReconfigL1AccOp::create(builder, loop->getLoc(), enableFlag);
        }

        // Conditional enable after the first iteration's last pack.
        Operation *afterOp = iter->second;
        Location loc = afterOp->getLoc();
        builder.setInsertionPointAfter(afterOp);
        Value firstIter =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                  loop.getInductionVar(), loop.getLowerBound());
        auto ifOp = scf::IfOp::create(builder, loc, firstIter);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        Value enableFlag = arith::ConstantOp::create(
            builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
        ttk::PackReconfigL1AccOp::create(builder, loc, enableFlag);
      }

      // Disable after the scope end.
      builder.setInsertionPointAfter(group.scopeEnd);
      ttk::PackReconfigL1AccOp::create(builder, disableLoc, disableFlag);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
