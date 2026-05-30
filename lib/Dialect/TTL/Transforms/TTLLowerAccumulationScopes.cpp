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
#include "llvm/ADT/STLExtras.h"
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

/// Verify the scope policy accepted by the initial DFB L1 lowering strategy.
/// DFB accumulation has no explicit initial tensor: overwrite and
/// accumulate-existing modes are implemented by TTKernel L1 packer guards.
static LogicalResult verifyAddDFBScope(AccumulationScopeOp scope) {
  if (scope.getOutputs().empty()) {
    return scope.emitOpError("DFB lowering requires at least one output");
  }
  if (!scope.getExplicitInits().empty()) {
    return scope.emitOpError("DFB lowering does not accept explicit inits");
  }

  for (AccumulationCombiner combiner : scope.getAccumulationCombiners()) {
    if (combiner != AccumulationCombiner::Add) {
      return scope.emitOpError("DFB lowering requires add combiner");
    }
  }

  for (AccumulationInitialMode mode : scope.getAccumulationInitialModes()) {
    if (mode != AccumulationInitialMode::Overwrite &&
        mode != AccumulationInitialMode::AccumulateExisting) {
      return scope.emitOpError(
          "DFB lowering requires overwrite or accumulate_existing initial "
          "mode");
    }
  }

  return success();
}

/// Find the single top-level loop represented by a DFB accumulation scope. The
/// formation pass emits one loop per scope so L1 metadata can be attached to
/// the loop before the semantic wrapper is erased.
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

/// Lower a DFB accumulation scope to explicit L1 packer metadata on its loop.
/// TTKernel lowering consumes the metadata after TTL stores have been converted
/// to packs, so no TTKernel operation ordering is inspected here.
static LogicalResult lowerDFBAccumulationScope(AccumulationScopeOp scope,
                                               AccumulationStrategy strategy,
                                               RewriterBase &rewriter) {
  if (strategy == AccumulationStrategy::Dst) {
    return scope.emitOpError("cannot lower DFB accumulation scope to DST");
  }
  if (failed(verifyAddDFBScope(scope))) {
    return failure();
  }

  FailureOr<scf::ForOp> loop = getSingleDFBAccumulationLoop(scope);
  if (failed(loop)) {
    return scope.emitOpError("DFB lowering requires one top-level scf.for");
  }

  SmallVector<AccumulationInitialMode> initialModes =
      scope.getAccumulationInitialModes();
  AccumulationInitialMode initialMode = initialModes.front();
  if (!llvm::all_of(initialModes, [&](AccumulationInitialMode mode) {
        return mode == initialMode;
      })) {
    return scope.emitOpError(
        "DFB L1 lowering requires one initial mode for all outputs");
  }

  (*loop)->setAttr(kL1AccLoopAttrName, UnitAttr::get(scope.getContext()));
  (*loop)->setAttr(kL1AccInitialAttrName, AccumulationInitialModeAttr::get(
                                              scope.getContext(), initialMode));
  eraseAccumulationScopeWrapper(scope, rewriter);
  return success();
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
    for (AccumulationScopeOp scope : scopes) {
      LogicalResult result =
          kind == "tensor"
              ? lowerTensorAccumulationScope(scope, *selectedStrategy, rewriter)
              : lowerDFBAccumulationScope(scope, *selectedStrategy, rewriter);
      if (failed(result)) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
