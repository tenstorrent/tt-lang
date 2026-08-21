// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Lower Accumulation Scopes
//===----------------------------------------------------------------------===//
//
// Lowers semantic accumulation scopes to concrete storage strategy metadata or
// operations.
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <memory>
#include <optional>
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

enum class TensorAccumulationScopeLoweringKind { Stateful, Dst, L1Pack };

/// Records all tensor scope facts that lowering depends on.
struct TensorAccumulationScopeLoweringPlan {
  AccumulationScopeOp scope;
  TensorAccumulationScopeLoweringKind kind =
      TensorAccumulationScopeLoweringKind::Stateful;
  std::optional<TensorAccumulationScopeMatch> match;
  std::optional<TensorDstAccumulationInfo> dstInfo;
  int64_t scopeId = 0;
  bool synthesizeResidentContributionPop = false;
};

/// Check the structural policy encoded by tensor accumulation scopes. The
/// scope lowerer relies on this policy before it looks through the contained
/// loop recurrence.
static LogicalResult verifySingleAddInitTensorScope(AccumulationScopeOp scope,
                                                    bool emitDiagnostics) {
  if (scope.getOutputs().size() != 1) {
    if (!emitDiagnostics) {
      return failure();
    }
    return scope.emitOpError(
        "tensor accumulation lowering supports exactly one output; split "
        "multiple accumulators into separate scopes");
  }
  if (scope.getInits().size() != 1) {
    if (!emitDiagnostics) {
      return failure();
    }
    return scope.emitOpError(
        "tensor accumulation lowering requires one init operand");
  }

  SmallVector<AccumulationInitialMode> initialModes =
      scope.getAccumulationInitialModes();
  if (initialModes.front() != AccumulationInitialMode::Init) {
    if (!emitDiagnostics) {
      return failure();
    }
    return scope.emitOpError(
        "tensor accumulation lowering requires init initial mode");
  }

  if (!scope.getOutputs().front().getDefiningOp<CBReserveOp>()) {
    if (!emitDiagnostics) {
      return failure();
    }
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
  if (failed(verifySingleAddInitTensorScope(scope, emitDiagnostics))) {
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
  for (auto [outputIndex, output] : llvm::enumerate(scope.getOutputs())) {
    if (!output.getDefiningOp<CBReserveOp>()) {
      return scope.emitOpError(
                 "stateful tensor accumulation lowering requires output ")
             << outputIndex << " from ttl.cb_reserve";
    }
  }
  return success();
}

static LogicalResult
verifyStatefulTensorScopeLowering(AccumulationScopeOp scope,
                                  AccumulationStrategy strategy) {
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

  return verifyStatefulTensorScope(scope);
}

/// Lower a stateful tensor scope to ordinary stores and tensor loop-carried
/// state. `ttl-materialize-loop-state` later assigns compiler-managed DFB
/// storage to the remaining tensor iter_args.
static LogicalResult
lowerStatefulTensorAccumulationScope(AccumulationScopeOp scope,
                                     RewriterBase &rewriter) {
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

static LogicalResult
emitTensorStrategyPlanningFailure(AccumulationScopeOp scope,
                                  AccumulationStrategy strategy) {
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
  if (strategy == AccumulationStrategy::L1Pack) {
    return scope.emitOpError(
        "cannot lower tensor accumulation scope to L1 packer accumulation: "
        "expected one same-type additive recurrence with one final store; "
        "select the automatic accumulation strategy or rewrite the loop");
  }
  return failure();
}

static LogicalResult
emitTensorUseAfterReleaseError(AccumulationScopeOp scope,
                               AccumulationStrategy strategy,
                               TensorAccumulationReleasedValue releasedValue) {
  if (strategy == AccumulationStrategy::Dst) {
    return scope.emitOpError()
           << "cannot lower tensor accumulation scope to DST: the "
           << (releasedValue == TensorAccumulationReleasedValue::Initial
                   ? "initial value"
                   : "contribution")
           << " would be used after release; move the release after the "
              "recurrence or keep the accumulator stateful";
  }
  if (strategy == AccumulationStrategy::L1Pack) {
    return scope.emitOpError()
           << "cannot lower tensor accumulation scope to L1 packer "
              "accumulation: the "
           << (releasedValue == TensorAccumulationReleasedValue::Initial
                   ? "initial value"
                   : "contribution")
           << " would be used after release; move the release after the "
              "recurrence or keep the accumulator stateful";
  }
  return failure();
}

static LogicalResult
verifyTensorL1PackLowering(AccumulationScopeOp scope,
                           const TensorAccumulationMatch &recurrence,
                           const DFBAcquireReleaseIndex &dfbIndex) {
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
  if (recurrence.loop->getNumResults() != 1 || recurrence.resultIndex != 0) {
    return emitL1PackError(
        "the current strategy supports exactly one loop-carried tensor "
        "accumulator; select the automatic accumulation strategy or split the "
        "accumulators into separate loops");
  }
  if (std::optional<TensorAccumulationReleasedValue> releasedValue =
          getTensorAccumulationUseAfterOwnedRelease(recurrence, dfbIndex)) {
    if (*releasedValue == TensorAccumulationReleasedValue::Initial) {
      return emitL1PackError(
          "the initial value would be used after release; move the release "
          "after the recurrence or keep the accumulator stateful");
    }
    return emitL1PackError(
        "the contribution would be used after release; move the release after "
        "the recurrence use or keep the accumulator stateful");
  }
  if (failed(analyzeTensorAccumulationForL1Pack(recurrence, &dfbIndex))) {
    return emitL1PackError(
        "expected one same-type additive recurrence with one final store; "
        "select the automatic accumulation strategy or rewrite the loop");
  }
  return success();
}

/// Verify every tensor scope before rewriting any scope.
static FailureOr<TensorAccumulationScopeLoweringPlan>
getTensorScopeLoweringPlan(AccumulationScopeOp scope,
                           AccumulationStrategy strategy, int64_t scopeId,
                           const DFBAcquireReleaseIndex &dfbIndex) {
  TensorAccumulationScopeLoweringPlan loweringPlan;
  loweringPlan.scope = scope;
  loweringPlan.scopeId = scopeId;

  FailureOr<TensorAccumulationScopeMatch> match =
      matchTensorAccumulationScope(scope, /*emitDiagnostics=*/false);
  if (failed(match)) {
    if (hasTopLevelOutputStore(scope)) {
      (void)matchTensorAccumulationScope(scope, /*emitDiagnostics=*/true);
      return failure();
    }
    if (failed(verifyStatefulTensorScopeLowering(scope, strategy))) {
      return failure();
    }
    loweringPlan.kind = TensorAccumulationScopeLoweringKind::Stateful;
    return loweringPlan;
  }

  TensorAccumulationMatch recurrence = match->recurrence;
  recurrence.initialValue = match->initialValue;

  if (strategy != AccumulationStrategy::Auto) {
    if (std::optional<TensorAccumulationReleasedValue> releasedValue =
            getTensorAccumulationUseAfterOwnedRelease(recurrence, dfbIndex)) {
      (void)emitTensorUseAfterReleaseError(scope, strategy, *releasedValue);
      return failure();
    }
  }

  FailureOr<AccumulationCostModel> costModel =
      AccumulationCostModel::forOperation(scope.getOperation());
  if (failed(costModel)) {
    return failure();
  }
  FailureOr<AccumulationStrategyPlan> strategyPlan =
      planTensorAccumulationStrategy(scope, recurrence, strategy, dfbIndex,
                                     *costModel);
  if (failed(strategyPlan)) {
    (void)emitTensorStrategyPlanningFailure(scope, strategy);
    return failure();
  }

  loweringPlan.match = *match;
  if (strategyPlan->strategy == AccumulationStrategy::Dst) {
    FailureOr<TensorDstAccumulationInfo> dstInfo =
        analyzeTensorAccumulationForDst(recurrence, dfbIndex);
    if (failed(dstInfo)) {
      (void)scope.emitOpError(
          "cannot lower tensor accumulation scope to DST after strategy "
          "planning");
      return failure();
    }
    loweringPlan.kind = TensorAccumulationScopeLoweringKind::Dst;
    loweringPlan.dstInfo = *dstInfo;
    return loweringPlan;
  }

  if (strategyPlan->strategy == AccumulationStrategy::L1Pack) {
    if (failed(verifyTensorL1PackLowering(scope, recurrence, dfbIndex))) {
      return failure();
    }
    loweringPlan.kind = TensorAccumulationScopeLoweringKind::L1Pack;
    return loweringPlan;
  }

  (void)scope.emitOpError(
      "automatic accumulation strategy selection returned an unresolved "
      "strategy");
  return failure();
}

/// Assign one missing resident release to the final lowering plan that uses
/// each acquisition. Multiple scopes may reuse one resident contribution, but
/// the acquisition owns exactly one release after its final use.
static void assignResidentContributionReleases(
    MutableArrayRef<TensorAccumulationScopeLoweringPlan> plans) {
  llvm::SmallPtrSet<Operation *, 4> assignedWaits;
  for (TensorAccumulationScopeLoweringPlan &plan : llvm::reverse(plans)) {
    if (plan.kind != TensorAccumulationScopeLoweringKind::Dst ||
        !plan.dstInfo) {
      continue;
    }
    TensorDstAccumulationInfo &dstInfo = *plan.dstInfo;
    if (dstInfo.contributionResidency !=
            TensorAccumulationContributionResidency::Resident ||
        dstInfo.residentContributionPop) {
      continue;
    }
    plan.synthesizeResidentContributionPop =
        assignedWaits.insert(dstInfo.contributionWait.getOperation()).second;
  }
}

/// Lower one tensor accumulation scope according to the selected strategy.
static LogicalResult
lowerTensorAccumulationScope(const TensorAccumulationScopeLoweringPlan &plan,
                             const DFBAcquireReleaseIndex &dfbIndex,
                             RewriterBase &rewriter) {
  AccumulationScopeOp scope = plan.scope;
  if (plan.kind == TensorAccumulationScopeLoweringKind::Stateful) {
    return lowerStatefulTensorAccumulationScope(scope, rewriter);
  }

  assert(plan.match && "planned tensor strategy requires a recurrence match");
  TensorAccumulationMatch recurrence = plan.match->recurrence;
  recurrence.initialValue = plan.match->initialValue;
  replaceYieldOperandsWithStateArguments(scope);

  if (plan.kind == TensorAccumulationScopeLoweringKind::Dst) {
    assert(plan.dstInfo && "DST lowering plan requires DST analysis facts");
    lowerTensorAccumulationToDst(recurrence, *plan.dstInfo,
                                 plan.synthesizeResidentContributionPop,
                                 rewriter);
    eraseAccumulationScopeWrapper(scope, rewriter,
                                  getScopeBlockArgumentReplacements(scope));
    return success();
  }

  [[maybe_unused]] LogicalResult lowered = lowerTensorAccumulationToL1Pack(
      recurrence, plan.scopeId, dfbIndex, rewriter);
  assert(succeeded(lowered) && "L1 pack legality was checked before mutation");
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
    if (*scopeKind == AccumulationScopeKind::Tensor) {
      SmallVector<TensorAccumulationScopeLoweringPlan, 4> plans;
      plans.reserve(scopes.size());
      bool hasInvalidScope = false;
      for (AccumulationScopeOp scope : scopes) {
        FailureOr<TensorAccumulationScopeLoweringPlan> plan =
            getTensorScopeLoweringPlan(scope, *selectedStrategy, nextScopeId++,
                                       *dfbIndex);
        if (failed(plan)) {
          hasInvalidScope = true;
          continue;
        }
        plans.push_back(*plan);
      }
      if (hasInvalidScope) {
        signalPassFailure();
        return;
      }
      assignResidentContributionReleases(plans);
      for (const TensorAccumulationScopeLoweringPlan &plan : plans) {
        if (failed(lowerTensorAccumulationScope(plan, *dfbIndex, rewriter))) {
          signalPassFailure();
          return;
        }
      }
      return;
    }

    for (AccumulationScopeOp scope : scopes) {
      if (failed(lowerDFBAccumulationScope(scope, nextScopeId++, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
