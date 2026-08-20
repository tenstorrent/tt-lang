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
#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

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

/// Caches all facts needed by lowering before any scope is rewritten. The pass
/// emits diagnostics for every invalid scope before mutating IR.
struct TensorAccumulationScopeLoweringPlan {
  AccumulationScopeOp scope;
  TensorAccumulationScopeMatch match;
  TensorDstAccumulationInfo dstInfo;
  bool synthesizeResidentContributionPop = false;
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
matchTensorAccumulationScope(AccumulationScopeOp scope) {
  if (failed(verifySingleAddInitTensorScope(scope))) {
    return failure();
  }

  StoreOp finalStore;
  FailureOr<scf::ForOp> loop =
      getSingleTensorAccumulationLoop(scope, finalStore);
  if (failed(loop)) {
    (void)scope.emitOpError(
        "tensor accumulation lowering requires a normalized scope body with "
        "one top-level scf.for followed by the final ttl.store; run "
        "ttl-form-accumulation-scopes or split other operations outside the "
        "scope");
    return failure();
  }

  FailureOr<TensorAccumulationMatch> match = matchAdditiveTensorAccumulation(
      *loop, /*resultIndex=*/0,
      TensorAccumulationReservePlacement::ExternalAllowed,
      ArrayRef<Operation *>{scope.getOperation()},
      ArrayRef<Operation *>{scope.getBody().front().getTerminator()});
  if (failed(match)) {
    (void)scope.emitOpError(
        "tensor accumulation lowering requires a loop-carried additive "
        "recurrence of the form acc = acc + contribution");
    return failure();
  }

  if (match->finalStore != finalStore ||
      match->initialValue != scope.getBody().front().getArgument(0)) {
    (void)scope.emitOpError(
        "tensor accumulation scope policy must match the loop recurrence");
    return failure();
  }
  return TensorAccumulationScopeMatch{*match, scope.getInits().front()};
}

/// Remove the region wrapper after its contents no longer depend on region
/// isolation. The single block argument is replaced with the verified init
/// operand.
static void eraseAccumulationScopeWrapper(AccumulationScopeOp scope,
                                          RewriterBase &rewriter,
                                          Value initialValue) {
  Block &body = scope.getBody().front();
  rewriter.eraseOp(body.getTerminator());
  rewriter.inlineBlockBefore(&body, scope, ValueRange{initialValue});
  rewriter.eraseOp(scope);
}

/// Disconnect the terminator from loop results before the loop is erased.
/// This keeps the wrapper body valid until it is inlined into the parent block.
static void replaceYieldOperandsWithStateArguments(AccumulationScopeOp scope) {
  Block &body = scope.getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  yield->setOperand(0, body.getArgument(0));
}

/// Verify every condition that the mutating rewrite depends on. Diagnostics
/// are emitted here, before any scope is rewritten, to avoid partially lowered
/// IR when one scope is invalid.
static FailureOr<TensorAccumulationScopeLoweringPlan>
getTensorScopeLoweringPlan(AccumulationScopeOp scope,
                           const DFBAcquireReleaseIndex &dfbIndex) {
  FailureOr<TensorAccumulationScopeMatch> match =
      matchTensorAccumulationScope(scope);
  if (failed(match)) {
    return failure();
  }

  FailureOr<TensorDstAccumulationInfo> dstInfo =
      analyzeTensorAccumulationForDst(match->recurrence, match->initialValue,
                                      dfbIndex);
  if (failed(dstInfo)) {
    (void)scope.emitOpError(
        "cannot lower tensor accumulation scope to DST: expected a "
        "DST-compatible same-type "
        "additive recurrence with an attached init tensor, a streamed or "
        "resident contribution ttl.cb_wait using the default block size, "
        "balanced contribution releases, and a static output tile count that "
        "fits in DST");
    return failure();
  }
  return TensorAccumulationScopeLoweringPlan{
      scope, *match, *dstInfo,
      /*synthesizeResidentContributionPop=*/
      false};
}

/// Assign one missing resident release to the final lowering plan that uses
/// each acquisition. Multiple scopes may reuse one resident contribution, but
/// the acquisition owns exactly one release after its final use.
static void assignResidentContributionReleases(
    MutableArrayRef<TensorAccumulationScopeLoweringPlan> plans) {
  llvm::SmallPtrSet<Operation *, 4> assignedWaits;
  for (TensorAccumulationScopeLoweringPlan &plan : llvm::reverse(plans)) {
    TensorDstAccumulationInfo &dstInfo = plan.dstInfo;
    if (dstInfo.contributionResidency !=
            TensorAccumulationContributionResidency::Resident ||
        dstInfo.residentContributionPop) {
      continue;
    }
    plan.synthesizeResidentContributionPop =
        assignedWaits.insert(dstInfo.contributionWait.getOperation()).second;
  }
}

/// Rewrite one verified tensor accumulation scope to a DST-resident accumulator
/// section and remove the now-empty scope wrapper.
static void
lowerTensorAccumulationScope(const TensorAccumulationScopeLoweringPlan &plan,
                             RewriterBase &rewriter) {
  AccumulationScopeOp scope = plan.scope;
  Value initialValue = scope.getInits().front();
  replaceYieldOperandsWithStateArguments(scope);
  lowerTensorAccumulationToDst(plan.match.recurrence, plan.dstInfo,
                               plan.synthesizeResidentContributionPop,
                               rewriter);

  eraseAccumulationScopeWrapper(scope, rewriter, initialValue);
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
    SmallVector<TensorAccumulationScopeLoweringPlan> plans;
    plans.reserve(scopes.size());
    bool hasInvalidScope = false;
    for (AccumulationScopeOp scope : scopes) {
      FailureOr<TensorAccumulationScopeLoweringPlan> plan =
          getTensorScopeLoweringPlan(scope, *dfbIndex);
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

    IRRewriter rewriter(&getContext());
    for (const TensorAccumulationScopeLoweringPlan &plan : plans) {
      lowerTensorAccumulationScope(plan, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
