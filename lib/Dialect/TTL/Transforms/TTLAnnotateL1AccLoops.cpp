// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Annotate L1 Acc Loops
//===----------------------------------------------------------------------===//
//
// Detects user-written scf.for loops containing accumulating stores
// (ttl.store with the {accumulate} attribute, emitted by +=) and annotates
// them with kL1AccLoopAttrName for L1 packer accumulation.
//
// Uses dominance: for each accumulating store, verifies the destination
// cb_reserve properly dominates the enclosing loop (the reserve is outside
// the loop, so the same L1 slot persists across iterations).
//
// TTKernelInsertL1Accumulation uses the annotated loops to find enable
// points, and groups consecutive sibling loops by shared pack CB targets
// to determine the accumulation scope for disable guards.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"

#define DEBUG_TYPE "ttl-annotate-l1-acc-loops"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLANNOTATEL1ACCLOOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Returns true if the loop carries any ttl.* annotation, indicating it
/// was generated or already processed by a compiler pass.
static bool hasCompilerAnnotation(scf::ForOp loop) {
  for (auto attr : loop->getAttrs()) {
    if (attr.getName().getValue().starts_with("ttl.")) {
      return true;
    }
  }
  return false;
}

struct TTLAnnotateL1AccLoopsPass
    : public impl::TTLAnnotateL1AccLoopsBase<TTLAnnotateL1AccLoopsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DominanceInfo domInfo(func);
    bool failed = false;

    func.walk([&](StoreOp store) {
      if (!store.getAccumulate()) {
        return;
      }

      auto enclosingLoop = store->getParentOfType<scf::ForOp>();
      if (!enclosingLoop) {
        return;
      }
      if (hasCompilerAnnotation(enclosingLoop)) {
        return;
      }

      // Conditional += is not supported: the L1 acc enable guard is conditional
      // based on the loop induction variable, not on whether a pack actually
      // executed. If the condition is false on iteration 0, subsequent
      // iterations accumulate into uninitialized L1.
      if (store->getParentOp() != enclosingLoop.getOperation()) {
        store->emitError(
            "+= inside a conditional is not supported (#504); move "
            "the condition outside the accumulation loop or use a "
            "separate loop for the conditional path");
        failed = true;
        return;
      }

      // The reserve must properly dominate the enclosing loop: the
      // reserve is outside the loop so the same L1 slot persists across
      // iterations. If the reserve is inside the loop, each iteration
      // gets a fresh slot and accumulation is meaningless.
      Value reserve = store.getView();
      Operation *reserveOp = reserve.getDefiningOp();
      if (reserveOp && !domInfo.properlyDominates(reserveOp, enclosingLoop)) {
        return;
      }

      enclosingLoop->setAttr(kL1AccLoopAttrName,
                             UnitAttr::get(enclosingLoop->getContext()));
    });

    if (failed) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
