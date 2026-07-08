// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Form Accumulation Scopes
//===----------------------------------------------------------------------===//
//
// Inserts accumulation regions around tensor recurrences that can be lowered
// to streaming DST-resident recurrence sections.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "DFBAcquireReleaseAnalysis.h"
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

/// Immutable formation facts collected before any loop is moved. The DFB
/// lifecycle index is valid only for this pre-mutation function version.
struct TensorAccumulationScopeFormation {
  scf::ForOp loop;
  TensorAccumulationMatch match;
};

/// Return true when a loop already carries TTL compiler metadata. Such loops
/// are owned by another lowering decision, and moving them into a new region
/// would make the metadata's scope ambiguous.
static bool hasCompilerAnnotation(scf::ForOp loop) {
  for (NamedAttribute attr : loop->getAttrs()) {
    if (attr.getName().getValue().starts_with("ttl.")) {
      return true;
    }
  }
  return false;
}

/// Return true when the loop and final store form the complete accumulator
/// publication sequence. The scope lowerer expects a normalized body, so only
/// dead attachments to the reserved output may appear between them.
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

  for (Operation *operation = loop->getNextNode();
       operation != match.finalStore.getOperation();
       operation = operation->getNextNode()) {
    if (!removableOps.contains(operation)) {
      return false;
    }
  }
  return true;
}

/// Return the facts needed to form a tensor accumulation scope. The caller runs
/// this during an immutable scan so DFB lifecycle analysis observes one
/// function version.
static FailureOr<TensorAccumulationScopeFormation>
getTensorAccumulationScopeFormation(scf::ForOp loop,
                                    const DFBAcquireReleaseIndex &dfbIndex) {
  if (loop->getParentOfType<AccumulationScopeOp>() ||
      hasCompilerAnnotation(loop)) {
    return failure();
  }

  FailureOr<TensorAccumulationMatch> match =
      matchAdditiveTensorAccumulation(loop, /*resultIndex=*/0);
  if (failed(match) || !isContiguousSingleTensorAccumulator(loop, *match) ||
      failed(analyzeTensorAccumulationForDst(*match, loop, &dfbIndex))) {
    return failure();
  }

  bool hasLoopLocalStore = false;
  loop->walk([&](StoreOp) {
    hasLoopLocalStore = true;
    return WalkResult::interrupt();
  });
  // Stores inside the source loop are side effects independent of the final
  // accumulator publication. Wrapping only the final store would not preserve
  // their ordering contract after the loop is deleted by DST lowering.
  if (hasLoopLocalStore) {
    return failure();
  }

  return TensorAccumulationScopeFormation{loop, *match};
}

/// Move a pre-verified recurrence into an accumulation scope.
static void
formTensorAccumulationScope(const TensorAccumulationScopeFormation &formation,
                            RewriterBase &rewriter) {
  scf::ForOp loop = formation.loop;
  TensorAccumulationMatch match = formation.match;

  MLIRContext *context = loop.getContext();
  ArrayAttr initialModes =
      rewriter.getArrayAttr({AccumulationInitialModeAttr::get(
          context, AccumulationInitialMode::Init)});

  if (!match.reserve->isBeforeInBlock(loop)) {
    rewriter.moveOpBefore(match.reserve, loop);
  }
  for (AttachCBOp attach : match.deadReserveAttachOps) {
    rewriter.eraseOp(attach);
  }

  rewriter.setInsertionPoint(loop);
  auto scope = AccumulationScopeOp::create(
      rewriter, loop.getLoc(), ValueRange{match.reserve.getResult()},
      ValueRange{match.initialValue}, initialModes);

  Block *body =
      rewriter.createBlock(&scope.getBody(), {}, match.initialValue.getType(),
                           SmallVector<Location>{match.initialValue.getLoc()});

  rewriter.moveOpBefore(loop, body, body->end());
  loop.getInitsMutable()[0].set(body->getArgument(0));
  rewriter.moveOpBefore(match.finalStore, body, body->end());
  rewriter.setInsertionPointToEnd(body);
  YieldOp::create(rewriter, loop.getLoc(), loop->getResults());
}

/// Forms tensor accumulation scopes opportunistically. Non-matching loops are
/// left unchanged so the existing loop-state materialization remains the
/// fallback for recurrences that need per-iteration work.
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

    DFBAcquireReleaseIndex dfbIndex(getOperation());
    SmallVector<TensorAccumulationScopeFormation> formations;
    formations.reserve(loops.size());
    for (scf::ForOp loop : loops) {
      // Post-order selection keeps the innermost valid recurrence. Forming a
      // nested loop first changes the body an outer recurrence was matched
      // against, so the outer candidate is not rewritten.
      bool containsSelectedLoop = llvm::any_of(
          formations, [&](const TensorAccumulationScopeFormation &formation) {
            return loop->isAncestor(formation.loop);
          });
      if (containsSelectedLoop) {
        continue;
      }

      FailureOr<TensorAccumulationScopeFormation> formation =
          getTensorAccumulationScopeFormation(loop, dfbIndex);
      if (succeeded(formation)) {
        formations.push_back(*formation);
      }
    }

    IRRewriter rewriter(&getContext());
    for (const TensorAccumulationScopeFormation &formation : formations) {
      formTensorAccumulationScope(formation, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
