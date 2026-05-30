// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Form Accumulation Scopes
//===----------------------------------------------------------------------===//
//
// Forms semantic accumulation regions before a later strategy-selection pass
// chooses DST, L1 packer, or explicit dataflow buffer state lowering.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-form-accumulation-scopes"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFORMACCUMULATIONSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Return true when the loop-to-store range contains only operations that the
/// accumulation scope formation can preserve or remove without changing
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

  // Formation normalizes the reserve before the loop and removes dead attach
  // views. Other intervening operations would need explicit strategy-lowering
  // support to preserve their relative execution order.
  for (Operation *operation = loop->getNextNode();
       operation != match.finalStore.getOperation();
       operation = operation->getNextNode()) {
    if (!removableOps.contains(operation)) {
      return false;
    }
  }
  return true;
}

/// Form a semantic accumulation scope around a matched tensor recurrence.
static LogicalResult formTensorAccumulationScope(scf::ForOp loop,
                                                 RewriterBase &rewriter) {
  if (loop->getParentOfType<AccumulationScopeOp>()) {
    return failure();
  }

  FailureOr<TensorAccumulationMatch> match =
      matchAdditiveTensorAccumulation(loop, /*resultIndex=*/0);
  if (failed(match) || !isContiguousSingleTensorAccumulator(loop, *match)) {
    return failure();
  }

  MLIRContext *context = loop.getContext();
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

struct TTLFormAccumulationScopesPass
    : public impl::TTLFormAccumulationScopesBase<
          TTLFormAccumulationScopesPass> {
  using impl::TTLFormAccumulationScopesBase<
      TTLFormAccumulationScopesPass>::TTLFormAccumulationScopesBase;

  void runOnOperation() override {
    if (kind != "tensor") {
      getOperation().emitOpError()
          << "invalid accumulation scope formation kind `" << kind
          << "`; expected `tensor`";
      signalPassFailure();
      return;
    }

    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>(
        [&](scf::ForOp loop) { loops.push_back(loop); });

    IRRewriter rewriter(&getContext());
    for (scf::ForOp loop : loops) {
      (void)formTensorAccumulationScope(loop, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
