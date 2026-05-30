// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Lower Accumulation Scopes
//===----------------------------------------------------------------------===//
//
// Selects a concrete storage strategy for semantic accumulation scopes.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringSwitch.h"

#define DEBUG_TYPE "ttl-lower-accumulation-scopes"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLLOWERACCUMULATIONSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

enum class AccumulationStrategy {
  Auto,
  Dst,
  L1Pack,
};

/// Parse the user-facing strategy spelling used by the pass option.
static FailureOr<AccumulationStrategy>
parseAccumulationStrategy(StringRef value) {
  return llvm::StringSwitch<FailureOr<AccumulationStrategy>>(value)
      .Case("auto", AccumulationStrategy::Auto)
      .Case("dst", AccumulationStrategy::Dst)
      .Case("l1-pack", AccumulationStrategy::L1Pack)
      .Default(failure());
}

/// Verify the scope policy accepted by the initial tensor lowering strategy.
static LogicalResult
verifySingleAddExplicitTensorScope(AccumulationScopeOp scope) {
  if (scope.getOutputs().size() != 1) {
    return scope.emitOpError("tensor lowering requires exactly one output");
  }
  if (scope.getExplicitInits().size() != 1) {
    return scope.emitOpError("tensor lowering requires one explicit init");
  }

  SmallVector<AccumulationCombiner> combiners =
      scope.getAccumulationCombiners();
  if (combiners.front() != AccumulationCombiner::Add) {
    return scope.emitOpError("tensor lowering requires add combiner");
  }

  SmallVector<AccumulationInitialMode> initialModes =
      scope.getAccumulationInitialModes();
  if (initialModes.front() != AccumulationInitialMode::Explicit) {
    return scope.emitOpError("tensor lowering requires explicit initial mode");
  }

  if (!scope.getOutputs().front().getDefiningOp<CBReserveOp>()) {
    return scope.emitOpError(
        "tensor lowering requires output from ttl.cb_reserve");
  }
  return success();
}

/// Find the single top-level loop and final store represented by a tensor
/// accumulation scope. The lowering owns only the normalized form emitted by
/// scope formation, so additional top-level operations are rejected before any
/// mutation.
static FailureOr<scf::ForOp>
getSingleTensorAccumulationLoop(AccumulationScopeOp scope,
                                StoreOp &finalStore) {
  scf::ForOp loop;
  Block &body = scope.getBody().front();
  for (Operation &operation : body.without_terminator()) {
    if (auto candidateLoop = dyn_cast<scf::ForOp>(&operation)) {
      if (loop) {
        return failure();
      }
      loop = candidateLoop;
      continue;
    }

    if (auto candidateStore = dyn_cast<StoreOp>(&operation)) {
      if (finalStore) {
        return failure();
      }
      finalStore = candidateStore;
      continue;
    }

    return failure();
  }

  if (!loop || !finalStore || !loop->isBeforeInBlock(finalStore) ||
      finalStore.getView() != scope.getOutputs().front()) {
    return failure();
  }
  return loop;
}

/// Match a tensor accumulation scope to the additive recurrence consumed by
/// the concrete strategy lowerings.
static FailureOr<TensorAccumulationMatch>
matchTensorAccumulationScope(AccumulationScopeOp scope) {
  if (failed(verifySingleAddExplicitTensorScope(scope))) {
    return failure();
  }

  StoreOp finalStore;
  FailureOr<scf::ForOp> loop =
      getSingleTensorAccumulationLoop(scope, finalStore);
  if (failed(loop)) {
    (void)scope.emitOpError(
        "tensor lowering requires a top-level scf.for followed by ttl.store");
    return failure();
  }

  FailureOr<TensorAccumulationMatch> match = matchAdditiveTensorAccumulation(
      *loop, /*resultIndex=*/0,
      TensorAccumulationReservePlacement::ExternalAllowed,
      ArrayRef<Operation *>{scope.getOperation()});
  if (failed(match)) {
    (void)scope.emitOpError(
        "tensor lowering requires acc = acc + contribution recurrence");
    return failure();
  }

  if (match->finalStore != finalStore ||
      match->initialValue != scope.getExplicitInits().front()) {
    (void)scope.emitOpError(
        "tensor lowering requires scope policy to match the loop recurrence");
    return failure();
  }

  return match;
}

/// Inline a lowered accumulation body and erase the semantic wrapper.
static void eraseAccumulationScopeWrapper(AccumulationScopeOp scope,
                                          RewriterBase &rewriter) {
  Block &body = scope.getBody().front();
  rewriter.eraseOp(body.getTerminator());
  rewriter.inlineBlockBefore(&body, scope, ValueRange{});
  rewriter.eraseOp(scope);
}

/// Lower one tensor accumulation scope according to the selected strategy.
static LogicalResult lowerTensorAccumulationScope(AccumulationScopeOp scope,
                                                  AccumulationStrategy strategy,
                                                  RewriterBase &rewriter) {
  FailureOr<TensorAccumulationMatch> match =
      matchTensorAccumulationScope(scope);
  if (failed(match)) {
    return failure();
  }

  scf::ForOp loop = match->add->getParentOfType<scf::ForOp>();
  assert(loop && "matched add must be inside an scf.for");

  if (strategy == AccumulationStrategy::Dst ||
      strategy == AccumulationStrategy::Auto) {
    if (succeeded(lowerTensorAccumulationToDst(*match, loop, rewriter))) {
      eraseAccumulationScopeWrapper(scope, rewriter);
      return success();
    }
    if (strategy == AccumulationStrategy::Dst) {
      return scope.emitOpError("cannot lower tensor accumulation scope to DST");
    }
  }

  if (failed(lowerTensorAccumulationToL1Pack(*match, loop, rewriter))) {
    return scope.emitOpError(
        "cannot lower tensor accumulation scope to L1 packer accumulation");
  }
  eraseAccumulationScopeWrapper(scope, rewriter);
  return success();
}

struct TTLLowerAccumulationScopesPass
    : public impl::TTLLowerAccumulationScopesBase<
          TTLLowerAccumulationScopesPass> {
  using impl::TTLLowerAccumulationScopesBase<
      TTLLowerAccumulationScopesPass>::TTLLowerAccumulationScopesBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (kind != "tensor") {
      func.emitOpError() << "invalid accumulation scope lowering kind `" << kind
                         << "`; expected `tensor`";
      signalPassFailure();
      return;
    }

    FailureOr<AccumulationStrategy> selectedStrategy =
        parseAccumulationStrategy(strategy);
    if (failed(selectedStrategy)) {
      func.emitOpError() << "invalid accumulation strategy `" << strategy
                         << "`; expected auto, dst, or l1-pack";
      signalPassFailure();
      return;
    }

    SmallVector<AccumulationScopeOp> scopes;
    func.walk([&](AccumulationScopeOp scope) { scopes.push_back(scope); });

    IRRewriter rewriter(&getContext());
    for (AccumulationScopeOp scope : scopes) {
      if (failed(lowerTensorAccumulationScope(scope, *selectedStrategy,
                                              rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
