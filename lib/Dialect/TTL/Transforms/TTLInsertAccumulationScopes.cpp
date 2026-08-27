// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Accumulation Scopes
//===----------------------------------------------------------------------===//
//
// Inserts semantic accumulation regions before a later strategy-selection pass
// chooses DST, L1 packer, or explicit dataflow buffer state lowering.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#define DEBUG_TYPE "ttl-insert-accumulation-scopes"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTACCUMULATIONSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static Value stripDFBAssociation(Value view) {
  view = traceUnrealizedCasts(view);
  while (auto attach = view.getDefiningOp<AttachCBOp>()) {
    view = traceUnrealizedCasts(attach.getTensor());
  }
  return view;
}

static bool isSameStoredView(Value lhs, Value rhs) {
  if (!lhs || !rhs) {
    return lhs == rhs;
  }
  return lhs == rhs || stripDFBAssociation(lhs) == stripDFBAssociation(rhs);
}

/// Return true when `operation` contains a non-accumulating store to the same
/// storage view. DFB association ops do not change storage identity. Slice and
/// extract ops remain distinct because this pass has no alias proof for them.
static bool containsPlainStoreToView(Operation *operation, Value view) {
  bool found = false;
  operation->walk([&](StoreOp store) {
    if (!store.getAccumulate() && isSameStoredView(store.getView(), view)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static Value getGuardedThenYieldedView(Value view, Operation *use) {
  view = traceUnrealizedCasts(view);
  if (auto attach = view.getDefiningOp<AttachCBOp>()) {
    view = traceUnrealizedCasts(attach.getTensor());
  }

  auto result = dyn_cast<OpResult>(view);
  if (!result) {
    return {};
  }
  auto ifOp = dyn_cast<scf::IfOp>(result.getOwner());
  if (!ifOp || ifOp.getElseRegion().empty() ||
      !isOperationInThenRegionGuardedBy(use, ifOp.getCondition())) {
    return {};
  }

  unsigned resultIndex = result.getResultNumber();
  auto thenYield =
      dyn_cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
  auto elseYield =
      dyn_cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
  if (!thenYield || !elseYield ||
      resultIndex >= thenYield.getResults().size() ||
      resultIndex >= elseYield.getResults().size() ||
      !isInactiveGuardedDFBYield(elseYield.getResults()[resultIndex])) {
    return {};
  }
  return thenYield.getResults()[resultIndex];
}

/// Determine whether iteration 0 should overwrite L1 or accumulate onto a value
/// produced by a preceding non-accumulating store to the same output view.
static FailureOr<AccumulationInitialMode>
getInitialModeForAccumulatingStore(StoreOp store, scf::ForOp loop) {
  Value view = store.getView();
  Value guardedThenView = getGuardedThenYieldedView(view, store.getOperation());
  Value targetDFB;
  if (auto reserve = findCBReserveForView(view, store.getOperation())) {
    targetDFB = reserve.getCb();
  } else {
    targetDFB = getAttachedCB(view);
  }

  auto isSameDFB = [&](Value cb) { return targetDFB && cb == targetDFB; };

  Operation *cursor = loop.getOperation();
  while (Block *block = cursor->getBlock()) {
    for (auto iter = Block::reverse_iterator(Block::iterator(cursor));
         iter != block->rend(); ++iter) {
      Operation *operation = &*iter;
      if (auto priorStore = dyn_cast<StoreOp>(operation)) {
        if (isSameStoredView(priorStore.getView(), view) ||
            isSameStoredView(priorStore.getView(), guardedThenView)) {
          if (priorStore.getAccumulate()) {
            return failure();
          }
          return AccumulationInitialMode::AccumulateExisting;
        }
        continue;
      }
      if (auto reserve = dyn_cast<CBReserveOp>(operation)) {
        if (isSameDFB(reserve.getCb())) {
          return AccumulationInitialMode::Overwrite;
        }
        continue;
      }
      if (auto push = dyn_cast<CBPushOp>(operation)) {
        if (isSameDFB(push.getCb())) {
          return AccumulationInitialMode::Overwrite;
        }
        continue;
      }
      if (operation->getNumRegions() > 0 &&
          containsPlainStoreToView(operation, view)) {
        return failure();
      }
      if (guardedThenView && operation->getNumRegions() > 0 &&
          containsPlainStoreToView(operation, guardedThenView)) {
        return AccumulationInitialMode::AccumulateExisting;
      }
    }
    Operation *parentOp = block->getParentOp();
    if (!parentOp || isa<func::FuncOp>(parentOp)) {
      break;
    }
    cursor = parentOp;
  }

  return AccumulationInitialMode::Overwrite;
}

static bool isGuardedDFBAccumulatingStore(StoreOp store) {
  return !findCBReserveForView(store.getView()) &&
         findCBReserveForView(store.getView(), store.getOperation());
}

/// Collect accumulating stores whose nearest enclosing loop is `loop`.
/// Conditional stores are accepted only when the output view is the
/// same-guard DFB value produced by a conditional acquire.
static FailureOr<SmallVector<StoreOp, 2>>
collectDFBAccumulationStores(scf::ForOp loop, bool &hadFailure) {
  SmallVector<StoreOp, 2> stores;
  SmallVector<StoreOp, 2> plainStores;
  loop->walk([&](StoreOp store) {
    if (store->getParentOfType<scf::ForOp>() != loop) {
      return WalkResult::advance();
    }
    if (!store.getAccumulate()) {
      plainStores.push_back(store);
      return WalkResult::advance();
    }
    if (store->getParentOp() != loop.getOperation() &&
        !isGuardedDFBAccumulatingStore(store)) {
      store.emitOpError()
          << "+= inside a conditional is not supported (#504); move the "
             "condition outside the accumulation loop or use a separate loop "
             "for the conditional branch";
      hadFailure = true;
      return WalkResult::interrupt();
    }
    stores.push_back(store);
    return WalkResult::advance();
  });
  if (hadFailure) {
    return failure();
  }
  if (!stores.empty() && !plainStores.empty()) {
    plainStores.front().emitOpError()
        << "non-accumulating store inside a += loop is not supported (#648); "
           "packer L1 accumulation state applies to every pack in the loop, "
           "including stores to other outputs; move the plain store outside "
           "the accumulation loop or split the loop";
    hadFailure = true;
    return failure();
  }
  return stores;
}

/// Insert a semantic accumulation scope around one user-written DFB
/// accumulation loop. The scope carries the initial-mode decision so later
/// lowering does not rediscover it from neighboring stores or dataflow buffer
/// operations.
static LogicalResult insertDFBAccumulationScope(scf::ForOp loop,
                                                DominanceInfo &domInfo,
                                                bool &hadFailure,
                                                RewriterBase &rewriter) {
  if (loop->getParentOfType<AccumulationScopeOp>() ||
      hasTTLDialectAttribute(loop)) {
    return failure();
  }

  FailureOr<SmallVector<StoreOp, 2>> stores =
      collectDFBAccumulationStores(loop, hadFailure);
  if (failed(stores)) {
    return failure();
  }
  if (stores->empty()) {
    return failure();
  }

  MLIRContext *context = loop.getContext();
  SmallVector<Value, 2> outputs;
  SmallVector<Attribute, 2> initialModes;
  SmallVector<Value, 2> seenOutputs;
  std::optional<AccumulationInitialMode> loopMode;

  for (StoreOp store : *stores) {
    Operation *reserveOp = nullptr;
    if (auto reserve = findCBReserveForView(store.getView())) {
      reserveOp = reserve.getOperation();
    } else {
      reserveOp = store.getView().getDefiningOp();
    }

    // The reserve must dominate the loop so every iteration updates the same
    // output slot.
    if (reserveOp && !domInfo.properlyDominates(reserveOp, loop)) {
      hadFailure = true;
      return store.emitOpError(
          "accumulating store requires an output reserve that dominates the "
          "accumulation loop; move the reserve before the loop");
    }

    if (llvm::any_of(seenOutputs, [&](Value output) {
          return isSameStoredView(output, store.getView());
        })) {
      hadFailure = true;
      return store.emitOpError(
          "multiple accumulating stores to the same output view in one loop "
          "are not supported; combine the updates before storing or split them "
          "into separate loops");
    }
    seenOutputs.push_back(store.getView());

    FailureOr<AccumulationInitialMode> mode =
        getInitialModeForAccumulatingStore(store, loop);
    if (failed(mode)) {
      hadFailure = true;
      return store.emitOpError(
          "cannot determine L1 accumulation initial mode; keep initialization "
          "as a straight-line store before the loop or split the loop");
    }
    if (!loopMode) {
      loopMode = *mode;
    } else if (*loopMode != *mode) {
      hadFailure = true;
      return loop.emitOpError()
             << "has accumulating stores requiring different L1 initial modes; "
                "split outputs with different initialization requirements into "
                "separate loops";
    }

    outputs.push_back(store.getView());
    initialModes.push_back(AccumulationInitialModeAttr::get(context, *mode));
  }

  rewriter.setInsertionPoint(loop);
  auto scope = AccumulationScopeOp::create(rewriter, loop.getLoc(), outputs,
                                           ValueRange{},
                                           rewriter.getArrayAttr(initialModes));

  SmallVector<Type, 2> outputTypes;
  SmallVector<Location, 2> outputLocs;
  for (Value output : outputs) {
    outputTypes.push_back(output.getType());
    outputLocs.push_back(output.getLoc());
  }
  Block *body =
      rewriter.createBlock(&scope.getBody(), {}, outputTypes, outputLocs);

  rewriter.moveOpBefore(loop, body, body->end());
  rewriter.setInsertionPointToEnd(body);
  YieldOp::create(rewriter, loop.getLoc(), body->getArguments());
  return success();
}

struct TTLInsertAccumulationScopesPass
    : public impl::TTLInsertAccumulationScopesBase<
          TTLInsertAccumulationScopesPass> {
  using impl::TTLInsertAccumulationScopesBase<
      TTLInsertAccumulationScopesPass>::TTLInsertAccumulationScopesBase;

  void runOnOperation() override {
    if (kind != "dfb") {
      getOperation().emitOpError()
          << "invalid accumulation scope insertion kind `" << kind
          << "`; expected `dfb`";
      signalPassFailure();
      return;
    }

    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>(
        [&](scf::ForOp loop) { loops.push_back(loop); });

    IRRewriter rewriter(&getContext());
    DominanceInfo domInfo(getOperation());
    bool hadFailure = false;
    for (scf::ForOp loop : loops) {
      (void)insertDFBAccumulationScope(loop, domInfo, hadFailure, rewriter);
      if (hadFailure) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
