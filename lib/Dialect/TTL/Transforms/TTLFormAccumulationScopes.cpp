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

#include <iterator>

#define DEBUG_TYPE "ttl-form-accumulation-scopes"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFORMACCUMULATIONSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Immutable information collected before any loop is moved. The DFB lifecycle
/// index is valid only for this pre-mutation function version.
struct TensorAccumulationScopeInfo {
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

  for (auto operationIt = std::next(loop->getIterator()),
            finalStoreIt = match.finalStore->getIterator();
       operationIt != finalStoreIt; ++operationIt) {
    if (!removableOps.contains(&*operationIt)) {
      return false;
    }
  }
  return true;
}

/// Return the information needed to rewrite one tensor accumulation scope. The
/// caller runs this during an immutable scan so DFB lifecycle analysis observes
/// one function version.
static FailureOr<TensorAccumulationScopeInfo>
getTensorAccumulationScopeInfo(scf::ForOp loop,
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

  return TensorAccumulationScopeInfo{loop, *match};
}

/// Apply one pre-verified tensor accumulation scope rewrite.
static void
rewriteTensorAccumulationScope(const TensorAccumulationScopeInfo &scopeInfo,
                               RewriterBase &rewriter) {
  scf::ForOp loop = scopeInfo.loop;
  TensorAccumulationMatch match = scopeInfo.match;

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
    FailureOr<AccumulationScopeKind> scopeKind =
        parseAccumulationScopeKind(kind);
    if (failed(scopeKind)) {
      getOperation().emitOpError() << "invalid accumulation scope kind `"
                                   << kind << "`; expected `tensor` or `dfb`";
      signalPassFailure();
      return;
    }
    if (*scopeKind == AccumulationScopeKind::DFB) {
      getOperation().emitOpError()
          << "DFB accumulation scopes are not supported yet";
      signalPassFailure();
      return;
    }

    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>(
        [&](scf::ForOp loop) { loops.push_back(loop); });

    DFBAcquireReleaseIndex dfbIndex(getOperation());
    SmallVector<TensorAccumulationScopeInfo> scopeInfos;
    scopeInfos.reserve(loops.size());
    for (scf::ForOp loop : loops) {
      // Rewriting both nested candidates would invalidate the outer match,
      // which was collected before any IR mutation.
      bool containsSelectedLoop = llvm::any_of(
          scopeInfos, [&](const TensorAccumulationScopeInfo &scopeInfo) {
            return loop->isAncestor(scopeInfo.loop);
          });
      if (containsSelectedLoop) {
        continue;
      }

      FailureOr<TensorAccumulationScopeInfo> scopeInfo =
          getTensorAccumulationScopeInfo(loop, dfbIndex);
      if (succeeded(scopeInfo)) {
        scopeInfos.push_back(*scopeInfo);
      }
    }

    IRRewriter rewriter(&getContext());
    for (const TensorAccumulationScopeInfo &scopeInfo : scopeInfos) {
      rewriteTensorAccumulationScope(scopeInfo, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
