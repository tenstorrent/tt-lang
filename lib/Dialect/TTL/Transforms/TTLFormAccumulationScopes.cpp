// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Form Accumulation Scopes
//===----------------------------------------------------------------------===//
//
// Forms semantic accumulation scopes for additive tensor recurrences before
// storage strategy lowering.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <iterator>
#include <memory>
#include <utility>

#define DEBUG_TYPE "ttl-form-accumulation-scopes"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFORMACCUMULATIONSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Return true when the loop and final store form the complete accumulator
/// publication sequence. The scope lowerer expects a normalized body, so only
/// dead attachments to the reserved output may appear between them.
static bool
isContiguousSingleTensorAccumulator(const TensorAccumulationMatch &match) {
  scf::ForOp loop = match.loop;
  if (loop.getNumResults() != 1 || match.resultIndex != 0) {
    return false;
  }
  if (!loop->isBeforeInBlock(match.finalStore)) {
    return false;
  }

  CBReserveOp reserve = match.reserve;
  llvm::SmallPtrSet<Operation *, 4> removableOps;
  removableOps.insert(reserve.getOperation());
  for (AttachCBOp attach : match.deadReserveAttachOps) {
    removableOps.insert(attach.getOperation());
  }

  for (auto operationIt = std::next(loop->getIterator()),
            finalStoreIt = match.finalStore->getIterator();
       operationIt != finalStoreIt; ++operationIt) {
    if (!removableOps.contains(&*operationIt)) {
      return false;
    }
  }
  return true;
}

/// Return true if formation should wrap the recurrence for `strategy`.
static bool
shouldFormTensorAccumulationForStrategy(const TensorAccumulationMatch &match,
                                        const DFBAcquireReleaseIndex &dfbIndex,
                                        AccumulationStrategy strategy) {
  if (strategy == AccumulationStrategy::Dst ||
      strategy == AccumulationStrategy::L1Pack) {
    return true;
  }

  if (succeeded(analyzeTensorAccumulationForDst(match, dfbIndex))) {
    return true;
  }

  return succeeded(analyzeTensorAccumulationForL1Pack(match, &dfbIndex));
}

static FailureOr<TensorAccumulationMatch>
getTensorAccumulationScopeMatch(scf::ForOp loop,
                                const DFBAcquireReleaseIndex &dfbIndex,
                                AccumulationStrategy strategy) {
  // TTL attributes identify loops owned by another lowering decision. Moving
  // such a loop would make the attribute's scope ambiguous.
  if (loop->getParentOfType<AccumulationScopeOp>() ||
      hasTTLDialectAttribute(loop)) {
    return failure();
  }

  FailureOr<TensorAccumulationMatch> match =
      matchAdditiveTensorAccumulation(loop, /*resultIndex=*/0);
  if (failed(match) || !isContiguousSingleTensorAccumulator(*match) ||
      !shouldFormTensorAccumulationForStrategy(*match, dfbIndex, strategy)) {
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

  return *match;
}

/// Apply one pre-verified tensor accumulation scope rewrite.
static void rewriteTensorAccumulationScope(const TensorAccumulationMatch &match,
                                           RewriterBase &rewriter) {
  scf::ForOp loop = match.loop;
  CBReserveOp reserve = match.reserve;

  MLIRContext *context = loop.getContext();
  ArrayAttr initialModes =
      rewriter.getArrayAttr({AccumulationInitialModeAttr::get(
          context, AccumulationInitialMode::Init)});

  if (!reserve->isBeforeInBlock(loop)) {
    rewriter.moveOpBefore(reserve, loop);
  }
  for (AttachCBOp attach : match.deadReserveAttachOps) {
    rewriter.eraseOp(attach);
  }

  rewriter.setInsertionPoint(loop);
  auto scope = AccumulationScopeOp::create(
      rewriter, loop.getLoc(), ValueRange{reserve.getResult()},
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

/// Rewrites normalized single-result `scf.for` recurrences of the form
/// `iter_arg + contribution`, with DFB-backed initial and contribution values
/// and a contiguous final store, into `ttl.accumulation_scope`. Other tensor
/// recurrences remain unchanged so `ttl-materialize-loop-state` preserves their
/// original loop-carried tensor semantics through dataflow buffer state.
struct TTLFormAccumulationScopesPass
    : public impl::TTLFormAccumulationScopesBase<
          TTLFormAccumulationScopesPass> {
  using impl::TTLFormAccumulationScopesBase<
      TTLFormAccumulationScopesPass>::TTLFormAccumulationScopesBase;

  void runOnOperation() override {
    FailureOr<AccumulationStrategy> selectedStrategy =
        parseAccumulationStrategy(strategy);
    if (failed(selectedStrategy)) {
      getOperation().emitOpError()
          << "invalid accumulation strategy `" << strategy
          << "`; expected auto, dst, or l1-pack";
      signalPassFailure();
      return;
    }

    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>(
        [&](scf::ForOp loop) { loops.push_back(loop); });

    PlanningResult<std::unique_ptr<DFBAcquireReleaseIndex>> indexResult =
        DFBAcquireReleaseIndex::create(getOperation());
    if (indexResult.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = indexResult.getInvalidIR();
      diagnostic.operation->emitOpError(diagnostic.message);
      signalPassFailure();
      return;
    }
    assert(indexResult.isPlanned() &&
           "DFB lifecycle indexing has no recoverable rejection");
    std::unique_ptr<DFBAcquireReleaseIndex> dfbIndex =
        std::move(indexResult).takePlan();
    SmallVector<TensorAccumulationMatch> matches;
    matches.reserve(loops.size());
    for (scf::ForOp loop : loops) {
      // Rewriting both nested candidates would invalidate the outer match,
      // which was collected before any IR mutation.
      bool containsSelectedLoop =
          llvm::any_of(matches, [&](const TensorAccumulationMatch &match) {
            return loop->isAncestor(match.loop);
          });
      if (containsSelectedLoop) {
        continue;
      }

      FailureOr<TensorAccumulationMatch> match =
          getTensorAccumulationScopeMatch(loop, *dfbIndex, *selectedStrategy);
      if (succeeded(match)) {
        matches.push_back(*match);
      }
    }

    IRRewriter rewriter(&getContext());
    for (const TensorAccumulationMatch &match : matches) {
      rewriteTensorAccumulationScope(match, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
