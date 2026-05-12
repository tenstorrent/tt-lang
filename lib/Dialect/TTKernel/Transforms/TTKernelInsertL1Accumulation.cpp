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
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "ttkernel-insert-l1-accumulation"

namespace mlir::tt::ttl {

namespace ttk = mlir::tt::ttkernel;

#define GEN_PASS_DEF_TTKERNELINSERTL1ACCUMULATION
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Build an i32 constant at the builder's insertion point.
static Value buildI32Const(OpBuilder &builder, Location loc, int32_t value) {
  return arith::ConstantOp::create(builder, loc, builder.getI32Type(),
                                   builder.getI32IntegerAttr(value));
}

/// Returns true if every CB in `packCBs` has a prior value in L1 from a
/// non-accumulating pack that precedes `rootLoop` in its parent block.
/// When true, the reconfig before the group must enable L1 acc instead of
/// disabling it.
///
/// See `docs/development/AccumulatingComputeLowering.md` ("Guard placement
/// around L1 accumulation loops") for the full set of detection rules,
/// boundary conditions, and the multi-output coverage requirement.
static bool
precededByNonAccumulatingPack(scf::ForOp rootLoop,
                              const llvm::SmallDenseSet<Value, 2> &packCBs) {
  assert(!packCBs.empty() && "L1-acc loop must have at least one pack CB");
  // Same-block only: an outer-scope pack wouldn't re-execute on each
  // iteration if `rootLoop` is nested, so its value would go stale.
  Block *block = rootLoop->getBlock();
  if (!block) {
    return false;
  }
  auto it = Block::iterator(rootLoop);
  if (it == block->begin()) {
    return false;
  }

  // Track which of `packCBs` are confirmed by a preceding pack; return true
  // only when all are covered, because L1 acc is a single switch for the
  // whole sync region — partial coverage would corrupt iteration 0 of the
  // uncovered CBs (acc onto stale L1). Invariant: covered ⊆ packCBs
  // (enforced by `record`).
  llvm::SmallDenseSet<Value, 2> covered;
  auto record = [&](Value cb) {
    if (packCBs.contains(cb)) {
      covered.insert(cb);
    }
  };
  auto allCovered = [&] { return covered.size() == packCBs.size(); };

  for (auto revIt = Block::reverse_iterator(it); revIt != block->rend();
       ++revIt) {
    Operation *op = &*revIt;
    if (auto pack = dyn_cast<ttk::PackTileOp>(op)) {
      record(pack.getOutCb());
      if (allCovered()) {
        return true;
      }
      continue;
    }
    if (auto reserve = dyn_cast<ttk::CBReserveBackOp>(op)) {
      if (packCBs.contains(reserve.getCb())) {
        return false;
      }
      continue;
    }
    if (auto push = dyn_cast<ttk::CBPushBackOp>(op)) {
      if (packCBs.contains(push.getCb())) {
        return false;
      }
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // An annotated scf.for has its own L1 acc lifecycle; its packs do
      // not provide a prior value to us. A non-annotated scf.for (e.g., a
      // compiler-generated tile-loop wrapper) packs with L1 acc disabled,
      // but only reaches L1 when it actually executes. Require a
      // provably-positive trip count (both bounds constant with lb < ub);
      // otherwise fall back to the no-prior-pack lowering.
      bool isAnnotated = forOp->hasAttr(kL1AccLoopAttrName) ||
                         forOp->hasAttr(kReductionLoopAttrName);
      bool executes = false;
      if (!isAnnotated) {
        auto tripCounts = getConstLoopTripCounts(forOp);
        executes = !tripCounts.empty() && !tripCounts.front().isZero();
      }
      bool touchedOurs = false;
      for (Value cb : getPackTileCBs(forOp)) {
        if (!packCBs.contains(cb)) {
          continue;
        }
        if (!executes) {
          return false;
        }
        covered.insert(cb);
        touchedOurs = true;
      }
      if (touchedOurs && allCovered()) {
        return true;
      }
      continue;
    }
    // Conservative fallback for any other region-bearing op (scf.if,
    // scf.while, custom region ops): if its body packs to one of our CBs
    // we cannot reason about its execution semantics, so treat as a
    // boundary.
    if (op->getNumRegions() > 0) {
      bool packsOurs = false;
      op->walk([&](ttk::PackTileOp pack) {
        if (packCBs.contains(pack.getOutCb())) {
          packsOurs = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (packsOurs) {
        return false;
      }
    }
  }
  return false;
}

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
      bool alreadyProcessed = false;
      if (auto *prev = loop->getPrevNode()) {
        alreadyProcessed = isa<ttk::PackReconfigL1AccOp>(prev);
      }
      if (!alreadyProcessed) {
        loop->walk([&](ttk::PackReconfigL1AccOp) {
          alreadyProcessed = true;
          return WalkResult::interrupt();
        });
      }
      if (alreadyProcessed) {
        return;
      }
      // L1 acc adds; max reduce would be corrupted by an additive store.
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

    // Insertion point for the per-iteration enable: the top-level ancestor
    // of the last tile_regs_release in the loop body, since packs may be
    // nested in subblock loops.
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

    // Group consecutive sibling loops that pack to the same CB.
    auto groups = collectLoopGroups(l1AccLoops, l1AccEnablePoint);

    // Emit guards per group.
    for (auto &group : groups) {
      OpBuilder builder(group.rootLoop->getContext());
      Location disableLoc = group.rootLoop->getLoc();

      // Reconfig L1 acc immediately before the first loop in the group:
      // ENABLE when L1 already holds a prior value from a non-accumulating
      // pack ahead of the group (so iteration 0 accumulates onto it),
      // DISABLE otherwise (so iteration 0 overwrites stale L1). See
      // precededByNonAccumulatingPack for the structural detection rules.
      auto rootPackCBs = getPackTileCBs(group.rootLoop);
      bool l1HasPriorValue =
          precededByNonAccumulatingPack(group.rootLoop, rootPackCBs);

      builder.setInsertionPoint(group.rootLoop);
      Value beforeGroupFlag =
          buildI32Const(builder, disableLoc, l1HasPriorValue ? 1 : 0);
      ttk::PackReconfigL1AccOp::create(builder, disableLoc, beforeGroupFlag);

      for (size_t idx = 0; idx < group.loops.size(); ++idx) {
        scf::ForOp loop = group.loops[idx];
        auto iter = l1AccEnablePoint.find(loop.getOperation());
        if (iter == l1AccEnablePoint.end()) {
          continue;
        }

        if (idx == 0 && l1HasPriorValue) {
          continue;
        }

        // Init ops between sibling loops reset packer state, so each
        // non-first loop needs an unconditional enable.
        if (idx > 0) {
          builder.setInsertionPoint(loop);
          Value enableFlag = buildI32Const(builder, loop->getLoc(), 1);
          ttk::PackReconfigL1AccOp::create(builder, loop->getLoc(), enableFlag);
        }

        // Enable runs once after the first iteration's last pack so
        // subsequent iterations accumulate.
        Operation *afterOp = iter->second;
        Location loc = afterOp->getLoc();
        builder.setInsertionPointAfter(afterOp);
        Value firstIter =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                  loop.getInductionVar(), loop.getLowerBound());
        auto ifOp = scf::IfOp::create(builder, loc, firstIter);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        Value enableFlag = buildI32Const(builder, loc, 1);
        ttk::PackReconfigL1AccOp::create(builder, loc, enableFlag);
      }

      // Disable L1 acc after the group's scope end (typically cb_push_back).
      builder.setInsertionPointAfter(group.scopeEnd);
      Value afterGroupFlag = l1HasPriorValue
                                 ? buildI32Const(builder, disableLoc, 0)
                                 : beforeGroupFlag;
      ttk::PackReconfigL1AccOp::create(builder, disableLoc, afterGroupFlag);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
