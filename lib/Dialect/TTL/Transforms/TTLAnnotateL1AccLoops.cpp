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
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"

#include <optional>

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

/// Return true when `operation` contains a non-accumulating store to the exact
/// view SSA value. Exact matching is required because this pass does not have
/// alias metadata for proving that two different slice values identify the
/// same output tile set.
static bool containsPlainStoreToView(Operation *operation, Value view) {
  bool found = false;
  operation->walk([&](StoreOp store) {
    if (!store.getAccumulate() && store.getView() == view) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Determine whether iteration 0 should overwrite L1 or accumulate onto a value
/// produced by a preceding non-accumulating store to the same output view.
static FailureOr<AccumulationInitialMode>
getInitialModeForAccumulatingStore(StoreOp store, scf::ForOp loop) {
  Value view = store.getView();
  Value targetCB;
  if (auto reserve = findCBReserveForView(view)) {
    targetCB = reserve.getCb();
  } else {
    targetCB = getAttachedCB(view);
  }

  Block *block = loop->getBlock();
  if (!block || block->begin() == Block::iterator(loop)) {
    return AccumulationInitialMode::Overwrite;
  }

  auto isSameCB = [&](Value cb) { return targetCB && cb == targetCB; };

  for (auto iter = Block::reverse_iterator(Block::iterator(loop));
       iter != block->rend(); ++iter) {
    Operation *operation = &*iter;
    if (auto priorStore = dyn_cast<StoreOp>(operation)) {
      if (priorStore.getView() == view) {
        if (priorStore.getAccumulate()) {
          return failure();
        }
        return AccumulationInitialMode::AccumulateExisting;
      }
      continue;
    }
    if (auto reserve = dyn_cast<CBReserveOp>(operation)) {
      if (isSameCB(reserve.getCb())) {
        return AccumulationInitialMode::Overwrite;
      }
      continue;
    }
    if (auto push = dyn_cast<CBPushOp>(operation)) {
      if (isSameCB(push.getCb())) {
        return AccumulationInitialMode::Overwrite;
      }
      continue;
    }
    if (operation->getNumRegions() > 0 &&
        containsPlainStoreToView(operation, view)) {
      return failure();
    }
  }

  return AccumulationInitialMode::Overwrite;
}

struct TTLAnnotateL1AccLoopsPass
    : public impl::TTLAnnotateL1AccLoopsBase<TTLAnnotateL1AccLoopsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DominanceInfo domInfo(func);
    bool hadFailure = false;
    llvm::DenseMap<Operation *, SmallVector<StoreOp, 2>> loopStores;

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
        hadFailure = true;
        return;
      }

      // The reserve must properly dominate the enclosing loop so each
      // iteration writes the same L1 slot. If the reserve is inside the loop,
      // each iteration reserves a different output slot.
      Operation *reserveOp = nullptr;
      if (auto reserve = findCBReserveForView(store.getView())) {
        reserveOp = reserve.getOperation();
      } else {
        reserveOp = store.getView().getDefiningOp();
      }
      if (reserveOp && !domInfo.properlyDominates(reserveOp, enclosingLoop)) {
        return;
      }

      loopStores[enclosingLoop.getOperation()].push_back(store);
    });

    if (hadFailure) {
      signalPassFailure();
      return;
    }

    for (auto &[loopOperation, stores] : loopStores) {
      auto loop = cast<scf::ForOp>(loopOperation);
      std::optional<AccumulationInitialMode> loopMode;
      for (StoreOp store : stores) {
        FailureOr<AccumulationInitialMode> mode =
            getInitialModeForAccumulatingStore(store, loop);
        if (failed(mode)) {
          store.emitError("cannot determine L1 accumulation initial mode");
          signalPassFailure();
          return;
        }
        if (!loopMode) {
          loopMode = *mode;
          continue;
        }
        if (*loopMode != *mode) {
          loop.emitOpError()
              << "has accumulating stores requiring different L1 initial "
                 "modes";
          signalPassFailure();
          return;
        }
      }

      loop->setAttr(kL1AccLoopAttrName, UnitAttr::get(loop->getContext()));
      loop->setAttr(kL1AccInitialAttrName, AccumulationInitialModeAttr::get(
                                               loop.getContext(), *loopMode));
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
