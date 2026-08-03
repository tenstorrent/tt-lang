// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Lower Accumulation Scopes
//===----------------------------------------------------------------------===//
//
// Lowers semantic tensor accumulation scopes to recurrence sections whose
// accumulator stays resident in DST across the source loop.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include <memory>
#include <utility>

#define DEBUG_TYPE "ttl-lower-accumulation-scopes"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLLOWERACCUMULATIONSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Keeps the semantic scope init operand separate from the recurrence match.
/// The recurrence matcher reports the loop-carried value inside the scope; DST
/// lowering copies the external init tensor into the accumulator.
struct TensorAccumulationScopeMatch {
  TensorAccumulationMatch recurrence;
  Value initialValue;
};

/// Check the structural policy encoded by tensor accumulation scopes. The
/// scope lowerer relies on this policy before it looks through the contained
/// loop recurrence.
static LogicalResult verifySingleAddInitTensorScope(AccumulationScopeOp scope) {
  if (scope.getOutputs().size() != 1) {
    return scope.emitOpError(
        "tensor accumulation lowering supports exactly one output; split "
        "multiple accumulators into separate scopes");
  }
  if (scope.getInits().size() != 1) {
    return scope.emitOpError(
        "tensor accumulation lowering requires one init operand");
  }

  SmallVector<AccumulationInitialMode> initialModes =
      scope.getAccumulationInitialModes();
  if (initialModes.front() != AccumulationInitialMode::Init) {
    return scope.emitOpError(
        "tensor accumulation lowering requires init initial mode");
  }

  if (!scope.getOutputs().front().getDefiningOp<CBReserveOp>()) {
    return scope.emitOpError(
        "tensor accumulation lowering requires output from ttl.cb_reserve");
  }
  return success();
}

/// Return the normalized loop/store pair from a tensor accumulation scope.
/// Extra top-level operations cannot be reordered across the loop or final
/// store without changing side-effect ordering, so they are rejected.
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

/// Reuse the shared recurrence matcher so scope formation and scope lowering
/// accept the same tensor recurrence.
static FailureOr<TensorAccumulationScopeMatch>
matchTensorAccumulationScope(AccumulationScopeOp scope,
                             bool emitDiagnostics = true) {
  LogicalResult verified =
      emitDiagnostics ? verifySingleAddInitTensorScope(scope) : success();
  if (!emitDiagnostics) {
    if (scope.getOutputs().size() != 1 || scope.getInits().size() != 1 ||
        scope.getAccumulationInitialModes().front() !=
            AccumulationInitialMode::Init ||
        !scope.getOutputs().front().getDefiningOp<CBReserveOp>()) {
      verified = failure();
    }
  }
  if (failed(verified)) {
    return failure();
  }

  StoreOp finalStore;
  FailureOr<scf::ForOp> loop =
      getSingleTensorAccumulationLoop(scope, finalStore);
  if (failed(loop)) {
    if (emitDiagnostics) {
      (void)scope.emitOpError(
          "tensor accumulation lowering requires a normalized scope body with "
          "one top-level scf.for followed by the final ttl.store; run "
          "ttl-form-accumulation-scopes or split other operations outside the "
          "scope");
    }
    return failure();
  }

  FailureOr<TensorAccumulationMatch> match = matchAdditiveTensorAccumulation(
      *loop, /*resultIndex=*/0,
      TensorAccumulationReservePlacement::ExternalAllowed,
      ArrayRef<Operation *>{scope.getOperation()},
      ArrayRef<Operation *>{scope.getBody().front().getTerminator()});
  if (failed(match)) {
    if (emitDiagnostics) {
      (void)scope.emitOpError(
          "tensor accumulation lowering requires a loop-carried additive "
          "recurrence of the form acc = acc + contribution");
    }
    return failure();
  }

  if (match->finalStore != finalStore ||
      match->initialValue != scope.getBody().front().getArgument(0)) {
    if (emitDiagnostics) {
      (void)scope.emitOpError(
          "tensor accumulation scope policy must match the loop recurrence");
    }
    return failure();
  }
  return TensorAccumulationScopeMatch{*match, scope.getInits().front()};
}

/// Remove the region wrapper after its contents no longer depend on region
/// isolation.
static void eraseAccumulationScopeWrapper(AccumulationScopeOp scope,
                                          RewriterBase &rewriter,
                                          ValueRange replacements = {}) {
  Block &body = scope.getBody().front();
  rewriter.eraseOp(body.getTerminator());
  rewriter.inlineBlockBefore(&body, scope, replacements);
  rewriter.eraseOp(scope);
}

/// Disconnect the terminator from loop results before the loop is erased.
/// This keeps the wrapper body valid until it is inlined into the parent block.
static void replaceYieldOperandsWithStateArguments(AccumulationScopeOp scope) {
  Block &body = scope.getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  for (auto [index, stateArgument] : llvm::enumerate(body.getArguments())) {
    yield->setOperand(index, stateArgument);
  }
}

/// Return true when a scope body carries explicit state through region
/// arguments and `ttl.yield` values.
static bool hasStatefulBody(AccumulationScopeOp scope) {
  Block &body = scope.getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  return body.getNumArguments() != 0 || !yield.getValues().empty();
}

/// Return true if the scope already contains an output store owned by the
/// additive tensor form.
static bool hasTopLevelOutputStore(AccumulationScopeOp scope) {
  Block &body = scope.getBody().front();
  for (Operation &operation : body.without_terminator()) {
    auto store = dyn_cast<StoreOp>(&operation);
    if (!store) {
      continue;
    }
    for (Value output : scope.getOutputs()) {
      if (store.getView() == output) {
        return true;
      }
    }
  }
  return false;
}

/// Return values used when removing the scope body block. Init-mode outputs use
/// their init operands; other modes preserve the output view SSA value.
static SmallVector<Value>
getScopeBlockArgumentReplacements(AccumulationScopeOp scope) {
  SmallVector<Value> replacements;
  SmallVector<AccumulationInitialMode> initialModes =
      scope.getAccumulationInitialModes();
  unsigned initIndex = 0;
  for (auto [outputIndex, output] : llvm::enumerate(scope.getOutputs())) {
    if (initialModes[outputIndex] == AccumulationInitialMode::Init) {
      replacements.push_back(scope.getInits()[initIndex++]);
      continue;
    }
    replacements.push_back(output);
  }
  return replacements;
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
/// state. `ttl-materialize-loop-state` later assigns compiler-managed DFB
/// storage to the remaining tensor iter_args.
static LogicalResult
lowerStatefulTensorAccumulationScope(AccumulationScopeOp scope,
                                     AccumulationStrategy strategy,
                                     RewriterBase &rewriter) {
  if (strategy == AccumulationStrategy::Dst) {
    return scope.emitOpError(
        "cannot lower stateful tensor accumulation scope to DST: stateful DST "
        "lowering is not implemented (at this point)");
  }
  if (strategy == AccumulationStrategy::L1Pack) {
    return scope.emitOpError(
        "cannot lower stateful tensor accumulation scope to L1 packer "
        "accumulation: stateful L1 packer lowering is not implemented (at "
        "this point)");
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
  eraseAccumulationScopeWrapper(scope, rewriter,
                                getScopeBlockArgumentReplacements(scope));
  return success();
}

/// Verify the scope policy for user-written DFB accumulation.
static LogicalResult verifyAddDFBScope(AccumulationScopeOp scope) {
  if (scope.getOutputs().empty()) {
    return scope.emitOpError(
        "DFB accumulation lowering requires at least one output");
  }
  if (!scope.getInits().empty()) {
    return scope.emitOpError(
        "DFB accumulation lowering does not accept inits; use "
        "overwrite or accumulate_existing initial modes");
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

/// Return the single loop governed by a DFB accumulation scope.
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

/// Lower a DFB accumulation scope to L1 packer metadata.
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
        "into separate loops");
  }

  (*loop)->setAttr(kL1AccLoopAttrName, UnitAttr::get(scope.getContext()));
  (*loop)->setAttr(kL1AccInitialAttrName, AccumulationInitialModeAttr::get(
                                              scope.getContext(), initialMode));
  (*loop)->setAttr(kL1AccScopeIdAttrName, rewriter.getI64IntegerAttr(scopeId));
  eraseAccumulationScopeWrapper(scope, rewriter,
                                getScopeBlockArgumentReplacements(scope));
  return success();
}

/// Lower one tensor accumulation scope according to the selected strategy.
static LogicalResult lowerTensorAccumulationScope(
    AccumulationScopeOp scope, AccumulationStrategy strategy, int64_t scopeId,
    const DFBAcquireReleaseIndex &dfbIndex, RewriterBase &rewriter) {
  FailureOr<TensorAccumulationScopeMatch> match =
      matchTensorAccumulationScope(scope, /*emitDiagnostics=*/false);
  if (failed(match)) {
    if (hasTopLevelOutputStore(scope)) {
      (void)matchTensorAccumulationScope(scope, /*emitDiagnostics=*/true);
      return failure();
    }
    return lowerStatefulTensorAccumulationScope(scope, strategy, rewriter);
  }

  TensorAccumulationMatch recurrence = match->recurrence;
  recurrence.initialValue = match->initialValue;

  AccumulationCostModel costModel =
      AccumulationCostModel::forOperation(scope.getOperation());
  FailureOr<AccumulationStrategyPlan> plan = planTensorAccumulationStrategy(
      scope, recurrence, strategy, dfbIndex, costModel);
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
          "strategy for a more specific diagnostic");
    }
  } else {
    selectedStrategy = plan->strategy;
  }

  if (selectedStrategy == AccumulationStrategy::Dst) {
    FailureOr<TensorDstAccumulationInfo> dstInfo =
        analyzeTensorAccumulationForDst(recurrence, match->initialValue,
                                        dfbIndex);
    if (failed(dstInfo)) {
      return scope.emitOpError(
          "cannot lower tensor accumulation scope to DST after strategy "
          "planning");
    }
    replaceYieldOperandsWithStateArguments(scope);
    lowerTensorAccumulationToDst(recurrence, *dstInfo, rewriter);
    eraseAccumulationScopeWrapper(scope, rewriter,
                                  getScopeBlockArgumentReplacements(scope));
    return success();
  }

  auto emitL1PackError = [&](StringRef reason) {
    return scope.emitOpError()
           << "cannot lower tensor accumulation scope to L1 packer "
              "accumulation: "
           << reason;
  };
  if (recurrence.contribution.getType() != recurrence.tensorType) {
    return emitL1PackError(
        "the addend must have the same tensor type as the accumulator; select "
        "the automatic accumulation strategy or rewrite the loop as a "
        "same-type additive recurrence");
  }
  if (recurrence.loop.getNumResults() != 1 || recurrence.resultIndex != 0) {
    return emitL1PackError(
        "the current strategy supports exactly one loop-carried tensor "
        "accumulator; select the automatic accumulation strategy or split the "
        "accumulators into separate loops");
  }

  replaceYieldOperandsWithStateArguments(scope);
  if (failed(lowerTensorAccumulationToL1Pack(recurrence, scopeId, rewriter))) {
    return emitL1PackError(
        "expected one same-type additive recurrence with one final store; "
        "select the automatic accumulation strategy or rewrite the loop");
  }
  eraseAccumulationScopeWrapper(scope, rewriter,
                                getScopeBlockArgumentReplacements(scope));
  return success();
}

/// Lowers only verified tensor accumulation scopes. The pass scans first and
/// mutates second because MLIR diagnostics from `runOnOperation` must coincide
/// with pass failure, not leave mixed old/new IR.
struct TTLLowerAccumulationScopesPass
    : public impl::TTLLowerAccumulationScopesBase<
          TTLLowerAccumulationScopesPass> {
  using impl::TTLLowerAccumulationScopesBase<
      TTLLowerAccumulationScopesPass>::TTLLowerAccumulationScopesBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    FailureOr<AccumulationScopeKind> scopeKind =
        parseAccumulationScopeKind(kind);
    if (failed(scopeKind)) {
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

    PlanningResult<std::unique_ptr<DFBAcquireReleaseIndex>> indexResult =
        DFBAcquireReleaseIndex::create(func);
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
    IRRewriter rewriter(&getContext());
    int64_t nextScopeId = getNextL1AccScopeId(func);
    for (AccumulationScopeOp scope : scopes) {
      int64_t scopeId = nextScopeId++;
      LogicalResult result =
          *scopeKind == AccumulationScopeKind::Tensor
              ? lowerTensorAccumulationScope(scope, *selectedStrategy, scopeId,
                                             *dfbIndex, rewriter)
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
