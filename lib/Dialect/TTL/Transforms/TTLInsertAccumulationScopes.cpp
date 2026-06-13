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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#define DEBUG_TYPE "ttl-insert-accumulation-scopes"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTACCUMULATIONSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Returns true if the loop carries any ttl.* annotation, indicating it was
/// generated or already processed by a compiler pass.
static bool hasCompilerAnnotation(scf::ForOp loop) {
  for (NamedAttribute attr : loop->getAttrs()) {
    if (attr.getName().getValue().starts_with("ttl.")) {
      return true;
    }
  }
  return false;
}

/// Return true when the loop-to-store range contains only operations that the
/// accumulation scope insertion can preserve or remove without changing
/// visible program effects.
static bool
isContiguousSingleTensorAccumulator(scf::ForOp loop,
                                    TensorAccumulationMatch &match) {
  if (loop.getNumResults() != 1 || match.resultIndex != 0) {
    return false;
  }

  if (!loop->isBeforeInBlock(match.finalStore)) {
    return false;
  }

  llvm::SmallPtrSet<Operation *, 4> removableOps;
  removableOps.insert(match.reserve.getOperation());
  for (AttachCBOp attach : match.deadReserveAttachOps) {
    removableOps.insert(attach.getOperation());
  }

  // Insertion normalizes the reserve before the loop and removes dead attach
  // views. Other intervening operations would need explicit strategy-lowering
  // support to preserve their relative execution order.
  // TODO(#640): Preserve post-loop pure users by lowering them through a staged
  // finalize region instead of requiring an immediate final store.
  for (Operation *operation = loop->getNextNode();
       operation != match.finalStore.getOperation();
       operation = operation->getNextNode()) {
    if (!removableOps.contains(operation)) {
      return false;
    }
  }
  return true;
}

/// Insert a semantic accumulation scope around a matched tensor recurrence.
static LogicalResult insertTensorAccumulationScope(scf::ForOp loop,
                                                   RewriterBase &rewriter) {
  if (loop->getParentOfType<AccumulationScopeOp>()) {
    return failure();
  }

  FailureOr<TensorAccumulationMatch> match =
      matchAdditiveTensorAccumulation(loop, /*resultIndex=*/0);
  if (failed(match) || !isContiguousSingleTensorAccumulator(loop, *match)) {
    return failure();
  }
  // Strategy lowering consumes the whole matched loop. A loop-local store is a
  // side effect not represented by the tensor recurrence scope contract.
  bool hasLoopLocalStore = false;
  loop->walk([&](StoreOp) {
    hasLoopLocalStore = true;
    return WalkResult::interrupt();
  });
  if (hasLoopLocalStore) {
    return failure();
  }

  MLIRContext *context = loop.getContext();
  // TODO(#646): Select the combiner from the matched recurrence instead of
  // inserting only additive tensor accumulation scopes.
  ArrayAttr combiners = rewriter.getArrayAttr(
      {AccumulationCombinerAttr::get(context, AccumulationCombiner::Add)});
  ArrayAttr initialModes =
      rewriter.getArrayAttr({AccumulationInitialModeAttr::get(
          context, AccumulationInitialMode::Explicit)});

  // The reserve defines the output view for the semantic scope. Moving it
  // before the loop represents the required single output slot that persists
  // across all accumulation iterations.
  if (!match->reserve->isBeforeInBlock(loop)) {
    rewriter.moveOpBefore(match->reserve, loop);
  }
  for (AttachCBOp attach : match->deadReserveAttachOps) {
    rewriter.eraseOp(attach);
  }

  rewriter.setInsertionPoint(loop);
  auto scope = AccumulationScopeOp::create(
      rewriter, loop.getLoc(), ValueRange{match->reserve.getResult()},
      ValueRange{match->initialValue}, combiners, initialModes);

  Block *body = rewriter.createBlock(&scope.getBody());
  rewriter.setInsertionPointToEnd(body);
  YieldOp::create(rewriter, loop.getLoc());

  Operation *terminator = body->getTerminator();
  rewriter.moveOpBefore(loop, terminator);
  rewriter.moveOpBefore(match->finalStore, terminator);
  return success();
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
  Value targetDFB;
  if (auto reserve = findCBReserveForView(view)) {
    targetDFB = reserve.getCb();
  } else {
    targetDFB = getAttachedCB(view);
  }

  Block *block = loop->getBlock();
  if (!block || block->begin() == Block::iterator(loop)) {
    return AccumulationInitialMode::Overwrite;
  }

  auto isSameDFB = [&](Value cb) { return targetDFB && cb == targetDFB; };

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
  }

  return AccumulationInitialMode::Overwrite;
}

/// Collect direct accumulating stores whose nearest enclosing loop is `loop`.
/// Conditional accumulation is rejected before inserting scopes because the L1
/// packer enable point is tied to loop iteration 0, not to dynamic control flow
/// inside the loop.
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
    if (store->getParentOp() != loop.getOperation()) {
      store->emitError("+= inside a conditional is not supported (#504); move "
                       "the condition outside the accumulation loop or use a "
                       "separate loop for the conditional branch");
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
    plainStores.front()->emitError(
        "non-accumulating store inside a += loop is not supported (#648); "
        "move it outside the accumulation loop or split the loop");
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
      hasCompilerAnnotation(loop)) {
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
  SmallVector<Attribute, 2> combiners;
  SmallVector<Attribute, 2> initialModes;
  llvm::DenseSet<Value> seenOutputs;
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
      return store.emitError(
          "accumulating store requires an output reserve that dominates the "
          "accumulation loop; move the reserve before the loop");
    }

    if (!seenOutputs.insert(store.getView()).second) {
      hadFailure = true;
      return store.emitError(
          "multiple accumulating stores to the same output view in one loop "
          "are not supported; combine the updates before storing or split them "
          "into separate loops");
    }

    FailureOr<AccumulationInitialMode> mode =
        getInitialModeForAccumulatingStore(store, loop);
    if (failed(mode)) {
      hadFailure = true;
      return store.emitError(
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
    // TODO(#646): Carry the source combiner when DFB accumulations support
    // non-additive update operations.
    combiners.push_back(
        AccumulationCombinerAttr::get(context, AccumulationCombiner::Add));
    initialModes.push_back(AccumulationInitialModeAttr::get(context, *mode));
  }

  rewriter.setInsertionPoint(loop);
  auto scope = AccumulationScopeOp::create(
      rewriter, loop.getLoc(), outputs, ValueRange{},
      rewriter.getArrayAttr(combiners), rewriter.getArrayAttr(initialModes));

  Block *body = rewriter.createBlock(&scope.getBody());
  rewriter.setInsertionPointToEnd(body);
  YieldOp::create(rewriter, loop.getLoc());

  Operation *terminator = body->getTerminator();
  rewriter.moveOpBefore(loop, terminator);
  return success();
}

struct TTLInsertAccumulationScopesPass
    : public impl::TTLInsertAccumulationScopesBase<
          TTLInsertAccumulationScopesPass> {
  using impl::TTLInsertAccumulationScopesBase<
      TTLInsertAccumulationScopesPass>::TTLInsertAccumulationScopesBase;

  void runOnOperation() override {
    if (kind != "tensor" && kind != "dfb") {
      getOperation().emitOpError()
          << "invalid accumulation scope insertion kind `" << kind
          << "`; expected `tensor` or `dfb`";
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
      if (kind == "tensor") {
        (void)insertTensorAccumulationScope(loop, rewriter);
        continue;
      }
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
