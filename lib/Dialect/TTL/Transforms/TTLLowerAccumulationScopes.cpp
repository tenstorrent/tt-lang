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
#include "ttlang/Dialect/TTL/Transforms/AccumulationAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "ttl-lower-accumulation-scopes"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLLOWERACCUMULATIONSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Verify the scope policy for single-output tensor accumulation.
static LogicalResult
verifySingleAddExplicitTensorScope(AccumulationScopeOp scope) {
  if (scope.getOutputs().size() != 1) {
    return scope.emitOpError(
        "tensor accumulation lowering supports exactly one output; split "
        "multiple accumulators into separate scopes");
  }
  if (scope.getExplicitInits().size() != 1) {
    return scope.emitOpError(
        "tensor accumulation lowering requires one explicit init; use DFB "
        "accumulation lowering for overwrite or accumulate_existing modes");
  }

  SmallVector<AccumulationCombiner> combiners =
      scope.getAccumulationCombiners();
  // TODO(#646): Replace add-only validation with a combiner/strategy legality
  // table shared by all accumulation scope lowerings.
  if (combiners.front() != AccumulationCombiner::Add) {
    return scope.emitOpError(
        "tensor accumulation lowering supports only the add combiner (#646)");
  }

  SmallVector<AccumulationInitialMode> initialModes =
      scope.getAccumulationInitialModes();
  if (initialModes.front() != AccumulationInitialMode::Explicit) {
    return scope.emitOpError(
        "tensor accumulation lowering requires explicit initial mode; use DFB "
        "accumulation lowering for overwrite or accumulate_existing modes");
  }

  if (!scope.getOutputs().front().getDefiningOp<CBReserveOp>()) {
    return scope.emitOpError(
        "tensor accumulation lowering requires output from ttl.cb_reserve");
  }
  return success();
}

/// Return the normalized tensor accumulation body: one loop followed by the
/// final store to the scope output.
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

/// Return the additive recurrence represented by a normalized tensor scope.
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
        "tensor accumulation lowering requires a normalized scope body with "
        "one "
        "top-level scf.for followed by the final ttl.store; run "
        "ttl-insert-accumulation-scopes or split other operations outside the "
        "scope");
    return failure();
  }

  FailureOr<TensorAccumulationMatch> match = matchAdditiveTensorAccumulation(
      *loop, /*resultIndex=*/0,
      TensorAccumulationReservePlacement::ExternalAllowed,
      ArrayRef<Operation *>{scope.getOperation()});
  if (failed(match)) {
    (void)scope.emitOpError(
        "tensor accumulation lowering requires a loop-carried additive "
        "recurrence of the form acc = acc + contribution; rewrite the loop");
    return failure();
  }

  if (match->finalStore != finalStore ||
      match->initialValue != scope.getExplicitInits().front()) {
    (void)scope.emitOpError(
        "tensor accumulation scope policy must match the loop recurrence; "
        "rebuild the scope with ttl-insert-accumulation-scopes");
    return failure();
  }

  return match;
}

/// Remove the semantic wrapper after storage strategy lowering has made the
/// accumulation state explicit.
static void
eraseAccumulationScopeWrapper(AccumulationScopeOp scope, RewriterBase &rewriter,
                              ValueRange blockArgReplacements = ValueRange()) {
  Block &body = scope.getBody().front();
  rewriter.eraseOp(body.getTerminator());
  rewriter.inlineBlockBefore(&body, scope, blockArgReplacements);
  rewriter.eraseOp(scope);
}

/// Return true when a scope body carries explicit state through region
/// arguments and ttl.yield values.
static bool hasStatefulBody(AccumulationScopeOp scope) {
  Block &body = scope.getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  return body.getNumArguments() != 0 || !yield.getValues().empty();
}

/// Verify the policy required for tensor stateful scope fallback.
static LogicalResult verifyStatefulTensorScope(AccumulationScopeOp scope) {
  assert(hasStatefulBody(scope) && "expected stateful accumulation scope");
  for (auto [outputIndex, output] : llvm::enumerate(scope.getOutputs())) {
    if (!output.getDefiningOp<CBReserveOp>()) {
      return scope.emitOpError(
                 "stateful tensor accumulation lowering requires output ")
             << outputIndex << " from ttl.cb_reserve";
    }
  }
  return success();
}

/// Lower a stateful tensor scope to ordinary stores and tensor loop-carried
/// state. ttl-materialize-loop-state later assigns compiler-allocated DFB
/// storage to the remaining tensor iter_args.
static LogicalResult
lowerStatefulTensorAccumulationScope(AccumulationScopeOp scope,
                                     AccumulationStrategy strategy,
                                     RewriterBase &rewriter) {
  if (strategy == AccumulationStrategy::Dst) {
    return scope.emitOpError(
        "cannot lower stateful tensor accumulation scope to DST: grouped DST "
        "lowering is not implemented");
  }
  if (strategy == AccumulationStrategy::L1Pack) {
    return scope.emitOpError(
        "cannot lower stateful tensor accumulation scope to L1 packer "
        "accumulation: grouped L1 packer lowering is not implemented");
  }

  if (failed(verifyStatefulTensorScope(scope))) {
    return failure();
  }

  auto yield = cast<YieldOp>(scope.getBody().front().getTerminator());
  rewriter.setInsertionPoint(yield);
  for (auto [output, yieldedValue] :
       llvm::zip_equal(scope.getOutputs(), yield.getValues())) {
    StoreOp::create(rewriter, yield.getLoc(), yieldedValue, output,
                    /*accumulate=*/nullptr);
  }
  eraseAccumulationScopeWrapper(scope, rewriter, scope.getExplicitInits());
  return success();
}

/// Verify the scope policy for user-written dataflow-buffer accumulation.
static LogicalResult verifyAddDFBScope(AccumulationScopeOp scope) {
  if (scope.getOutputs().empty()) {
    return scope.emitOpError(
        "DFB accumulation lowering requires at least one output");
  }
  if (!scope.getExplicitInits().empty()) {
    return scope.emitOpError(
        "DFB accumulation lowering does not accept explicit inits; use "
        "overwrite or accumulate_existing initial modes");
  }

  // TODO(#646): Replace add-only validation with a combiner/strategy legality
  // table shared by all accumulation scope lowerings.
  for (AccumulationCombiner combiner : scope.getAccumulationCombiners()) {
    if (combiner != AccumulationCombiner::Add) {
      return scope.emitOpError(
          "DFB accumulation lowering supports only the add combiner (#646)");
    }
  }

  for (AccumulationInitialMode mode : scope.getAccumulationInitialModes()) {
    if (mode != AccumulationInitialMode::Overwrite &&
        mode != AccumulationInitialMode::AccumulateExisting) {
      return scope.emitOpError(
          "DFB accumulation lowering requires overwrite or accumulate_existing "
          "initial mode");
    }
  }

  return success();
}

/// Return the single loop governed by a dataflow-buffer accumulation scope.
static FailureOr<scf::ForOp>
getSingleDFBAccumulationLoop(AccumulationScopeOp scope) {
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
    return failure();
  }

  if (!loop) {
    return failure();
  }
  return loop;
}

/// Lower a dataflow-buffer accumulation scope to L1 packer metadata.
static LogicalResult lowerDFBAccumulationScope(AccumulationScopeOp scope,
                                               int64_t scopeId,
                                               RewriterBase &rewriter) {
  if (failed(verifyAddDFBScope(scope))) {
    return failure();
  }

  FailureOr<scf::ForOp> loop = getSingleDFBAccumulationLoop(scope);
  if (failed(loop)) {
    return scope.emitOpError(
        "DFB accumulation lowering requires one top-level scf.for; split "
        "multiple loops into separate scopes");
  }

  SmallVector<AccumulationInitialMode> initialModes =
      scope.getAccumulationInitialModes();
  AccumulationInitialMode initialMode = initialModes.front();
  if (!llvm::all_of(initialModes, [&](AccumulationInitialMode mode) {
        return mode == initialMode;
      })) {
    return scope.emitOpError(
        "DFB L1 accumulation lowering requires one initial mode for all "
        "outputs; split outputs with different initialization requirements "
        "into "
        "separate loops");
  }

  (*loop)->setAttr(kL1AccLoopAttrName, UnitAttr::get(scope.getContext()));
  (*loop)->setAttr(kL1AccInitialAttrName, AccumulationInitialModeAttr::get(
                                              scope.getContext(), initialMode));
  (*loop)->setAttr(kL1AccScopeIdAttrName, rewriter.getI64IntegerAttr(scopeId));
  eraseAccumulationScopeWrapper(scope, rewriter);
  return success();
}

/// Lower one tensor accumulation scope according to the selected strategy.
static LogicalResult lowerTensorAccumulationScope(AccumulationScopeOp scope,
                                                  AccumulationStrategy strategy,
                                                  int64_t scopeId,
                                                  RewriterBase &rewriter) {
  if (hasStatefulBody(scope)) {
    return lowerStatefulTensorAccumulationScope(scope, strategy, rewriter);
  }

  FailureOr<TensorAccumulationMatch> match =
      matchTensorAccumulationScope(scope);
  if (failed(match)) {
    return failure();
  }

  scf::ForOp loop = match->add->getParentOfType<scf::ForOp>();
  assert(loop && "matched add must be inside an scf.for");

  AccumulationCostModel costModel =
      AccumulationCostModel::forOperation(scope.getOperation());
  FailureOr<AccumulationStrategyPlan> plan =
      planTensorAccumulationStrategy(scope, *match, loop, strategy, costModel);
  AccumulationStrategy selectedStrategy = strategy;
  if (failed(plan)) {
    if (strategy == AccumulationStrategy::Dst) {
      return scope.emitOpError(
          "cannot lower tensor accumulation scope to DST: expected a "
          "DST-compatible same-type additive recurrence with one loop-carried "
          "accumulator and one final store; select the automatic accumulation "
          "strategy or l1-pack");
    }
    if (strategy == AccumulationStrategy::Auto) {
      return scope.emitOpError(
          "automatic accumulation strategy selection found no legal tensor "
          "accumulation strategy; rewrite the loop or select a required "
          "strategy "
          "for a more specific diagnostic");
    }
  } else {
    selectedStrategy = plan->strategy;
  }

  if (selectedStrategy == AccumulationStrategy::Dst) {
    if (succeeded(lowerTensorAccumulationToDst(*match, loop, rewriter))) {
      eraseAccumulationScopeWrapper(scope, rewriter);
      return success();
    }
    return scope.emitOpError("cannot lower tensor accumulation scope to DST "
                             "after strategy planning");
  }

  auto emitL1PackError = [&](StringRef reason) {
    return scope.emitOpError()
           << "cannot lower tensor accumulation scope to L1 packer "
              "accumulation: "
           << reason;
  };
  if (match->contribution.getType() != match->tensorType) {
    return emitL1PackError(
        "the addend must have the same tensor type as the accumulator; select "
        "the automatic accumulation strategy or rewrite the loop as a "
        "same-type "
        "additive recurrence");
  }
  if (loop.getNumResults() != 1 || match->resultIndex != 0) {
    return emitL1PackError(
        "the current strategy supports exactly one loop-carried tensor "
        "accumulator; select the automatic accumulation strategy or split the "
        "accumulators into separate loops");
  }

  bool hasLoopLocalStore = false;
  loop->walk([&](StoreOp) {
    hasLoopLocalStore = true;
    return WalkResult::interrupt();
  });
  if (hasLoopLocalStore) {
    return emitL1PackError(
        "the accumulation loop contains a store not owned by the recurrence; "
        "select the automatic accumulation strategy, move that store outside "
        "the loop, or split the loop");
  }

  if (failed(
          lowerTensorAccumulationToL1Pack(*match, loop, scopeId, rewriter))) {
    // TODO(#650): Use explicit DFB state as the correctness fallback for
    // semantically valid scopes when no hardware accumulation strategy is
    // legal.
    return emitL1PackError(
        "expected one same-type additive recurrence with one final store; "
        "select the automatic accumulation strategy or rewrite the loop");
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
    if (kind != "tensor" && kind != "dfb") {
      func.emitOpError() << "invalid accumulation scope lowering kind `" << kind
                         << "`; expected `tensor` or `dfb`";
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
    int64_t nextScopeId = getNextL1AccScopeId(func);
    for (AccumulationScopeOp scope : scopes) {
      int64_t scopeId = nextScopeId++;
      LogicalResult result =
          kind == "tensor"
              ? lowerTensorAccumulationScope(scope, *selectedStrategy, scopeId,
                                             rewriter)
              : lowerDFBAccumulationScope(scope, scopeId, rewriter);
      if (failed(result)) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
