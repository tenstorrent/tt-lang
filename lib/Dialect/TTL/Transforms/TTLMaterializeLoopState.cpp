// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Materialize Loop State
//===----------------------------------------------------------------------===//
//
// Eliminates tensor-valued scf.for iter_args before compute lowering. Eligible
// additive recurrence state lowers to an in-DST reduction compute; the
// remaining additive cases use accumulate stores. All other tensor state lowers
// through compiler-allocated DFB state slots.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"
#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#define DEBUG_TYPE "ttl-materialize-loop-state"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLMATERIALIZELOOPSTATE
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TensorLoopState {
  unsigned resultIndex;
  RankedTensorType tensorType;
  Value initialValue;
  BlockArgument iterArg;
  Value yieldedValue;
  std::optional<TensorAccumulationMatch> accumulator;
  BindCBOp stateDFB;
};

// Collects tensor loop-carried values and records whether each one matches the
// additive recurrence form used by the accumulation-specific lowering.
static SmallVector<TensorLoopState> collectTensorLoopStates(scf::ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());

  SmallVector<TensorLoopState> states;
  for (unsigned resultIndex = 0; resultIndex < loop.getNumResults();
       ++resultIndex) {
    if (!isTensorLoopState(loop, resultIndex)) {
      continue;
    }

    std::optional<TensorAccumulationMatch> accumulator;
    FailureOr<TensorAccumulationMatch> match =
        matchAdditiveTensorAccumulation(loop, resultIndex);
    if (succeeded(match)) {
      accumulator = *match;
    }

    states.push_back(TensorLoopState{
        resultIndex,
        cast<RankedTensorType>(loop.getInitArgs()[resultIndex].getType()),
        loop.getInitArgs()[resultIndex], loop.getRegionIterArgs()[resultIndex],
        yield.getOperand(resultIndex), std::move(accumulator), BindCBOp()});
  }
  return states;
}

// Tests whether an old loop result is one of the tensor states removed from the
// reconstructed scf.for signature.
static bool isTensorStateIndex(ArrayRef<TensorLoopState> states,
                               unsigned resultIndex) {
  return llvm::any_of(states, [&](const TensorLoopState &state) {
    return state.resultIndex == resultIndex;
  });
}

// Seeds materialized state before the rewritten loop. Accumulators reuse the
// user output dataflow buffer; other tensor states use compiler-allocated DFBs.
static void createInitialStores(ArrayRef<TensorLoopState> states,
                                scf::ForOp loop, RewriterBase &rewriter) {
  for (TensorLoopState state : states) {
    if (!state.accumulator) {
      continue;
    }
    CBReserveOp reserve = state.accumulator->reserve;
    if (!reserve->isBeforeInBlock(loop)) {
      rewriter.moveOpBefore(reserve, loop);
    }

    rewriter.setInsertionPoint(loop);
    StoreOp::create(rewriter, state.accumulator->finalStore.getLoc(),
                    state.initialValue, reserve.getResult(),
                    /*accumulate=*/nullptr);
  }

  for (TensorLoopState state : states) {
    if (state.accumulator) {
      continue;
    }

    rewriter.setInsertionPoint(loop);
    createDFBStore(state.initialValue, state.stateDFB.getResult(), rewriter);
  }
}

// Rebuilds the loop with only non-tensor iter_args so tensor state is carried
// by explicit dataflow buffer operations instead of scf.for results.
static scf::ForOp createLoopWithoutTensorState(scf::ForOp loop,
                                               ArrayRef<TensorLoopState> states,
                                               RewriterBase &rewriter) {
  SmallVector<Value> newInitArgs;
  for (unsigned resultIndex = 0; resultIndex < loop.getNumResults();
       ++resultIndex) {
    if (isTensorStateIndex(states, resultIndex)) {
      continue;
    }
    newInitArgs.push_back(loop.getInitArgs()[resultIndex]);
  }

  rewriter.setInsertionPoint(loop);
  auto newLoop =
      scf::ForOp::create(rewriter, loop.getLoc(), loop.getLowerBound(),
                         loop.getUpperBound(), loop.getStep(), newInitArgs);
  for (NamedAttribute attr : loop->getAttrs()) {
    newLoop->setAttr(attr.getName(), attr.getValue());
  }

  Block *newBody = newLoop.getBody();
  if (!newBody->empty() && isa<scf::YieldOp>(newBody->back())) {
    rewriter.eraseOp(&newBody->back());
  }

  return newLoop;
}

// Maps old loop-carried SSA values into the rebuilt loop and materializes
// non-accumulator tensor iter_args from their DFB state slots at loop entry.
static void mapLoopCarriedValues(scf::ForOp loop, scf::ForOp newLoop,
                                 ArrayRef<TensorLoopState> states,
                                 IRMapping &mapper, RewriterBase &rewriter) {
  mapper.map(loop.getInductionVar(), newLoop.getInductionVar());

  unsigned newRegionArgIndex = 0;
  for (unsigned resultIndex = 0; resultIndex < loop.getNumResults();
       ++resultIndex) {
    if (isTensorStateIndex(states, resultIndex)) {
      continue;
    }
    mapper.map(loop.getRegionIterArgs()[resultIndex],
               newLoop.getRegionIterArgs()[newRegionArgIndex]);
    ++newRegionArgIndex;
  }

  rewriter.setInsertionPointToStart(newLoop.getBody());
  for (TensorLoopState state : states) {
    if (state.accumulator) {
      continue;
    }

    auto attach = createDFBWaitAndAttach(
        state.stateDFB.getResult(), state.tensorType, loop.getLoc(), rewriter);
    mapper.map(state.iterArg, attach.getResult());
  }
}

// For each non-accumulator state, emits a reserve and store of the next
// iteration value immediately after the op that produced it: at body entry
// when the value enters from outside the loop body, inline when it is
// produced by a cloned body op, and at body end as a fallback. Accumulator
// states emit an in-place accumulate `store` at the `add` site instead.
static void cloneBodyAndMaterializeNextState(scf::ForOp loop,
                                             scf::ForOp newLoop,
                                             ArrayRef<TensorLoopState> states,
                                             IRMapping &mapper,
                                             RewriterBase &rewriter) {
  DenseSet<Operation *> accumulatorAdds;
  DenseSet<unsigned> storedStateIndices;
  for (TensorLoopState state : states) {
    if (state.accumulator) {
      accumulatorAdds.insert(state.accumulator->add.getOperation());
    }
  }

  auto storeNextState = [&](TensorLoopState state) {
    Value nextState = mapper.lookupOrDefault(state.yieldedValue);
    createDFBStore(nextState, state.stateDFB.getResult(), rewriter);
    storedStateIndices.insert(state.resultIndex);
  };

  for (TensorLoopState state : states) {
    if (state.accumulator) {
      continue;
    }

    Operation *definingOp = state.yieldedValue.getDefiningOp();
    if (!definingOp || definingOp->getBlock() != loop.getBody()) {
      storeNextState(state);
    }
  }

  for (Operation &bodyOp : *loop.getBody()) {
    if (isa<scf::YieldOp>(bodyOp)) {
      continue;
    }

    if (accumulatorAdds.contains(&bodyOp)) {
      for (TensorLoopState state : states) {
        if (!state.accumulator ||
            state.accumulator->add.getOperation() != &bodyOp) {
          continue;
        }
        Value contribution =
            mapper.lookupOrDefault(state.accumulator->contribution);
        StoreOp::create(rewriter, bodyOp.getLoc(), contribution,
                        state.accumulator->reserve.getResult(),
                        rewriter.getUnitAttr());
      }
      continue;
    }

    rewriter.clone(bodyOp, mapper);

    for (TensorLoopState state : states) {
      if (state.accumulator || storedStateIndices.contains(state.resultIndex)) {
        continue;
      }
      Operation *definingOp = state.yieldedValue.getDefiningOp();
      if (definingOp == &bodyOp) {
        storeNextState(state);
      }
    }
  }

  for (TensorLoopState state : states) {
    if (!state.accumulator && !storedStateIndices.contains(state.resultIndex)) {
      storeNextState(state);
    }
  }

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<Value> newYieldOperands;
  for (unsigned resultIndex = 0; resultIndex < yield.getNumOperands();
       ++resultIndex) {
    if (isTensorStateIndex(states, resultIndex)) {
      continue;
    }
    newYieldOperands.push_back(
        mapper.lookupOrDefault(yield.getOperand(resultIndex)));
  }
  scf::YieldOp::create(rewriter, yield.getLoc(), newYieldOperands);
}

// Reconnects users of the old loop results after tensor state has been
// materialized, and removes final stores absorbed by accumulation state.
static void replaceLoopResults(scf::ForOp loop, scf::ForOp newLoop,
                               ArrayRef<TensorLoopState> states,
                               RewriterBase &rewriter) {
  DenseMap<unsigned, Value> tensorReplacements;
  rewriter.setInsertionPointAfter(newLoop);
  for (TensorLoopState state : states) {
    if (state.accumulator) {
      rewriter.eraseOp(state.accumulator->finalStore);
      for (AttachCBOp attach : state.accumulator->deadReserveAttachOps) {
        rewriter.eraseOp(attach);
      }
      continue;
    }

    auto attach = createDFBWaitAndAttach(
        state.stateDFB.getResult(), state.tensorType, loop.getLoc(), rewriter);
    tensorReplacements[state.resultIndex] = attach.getResult();
  }

  unsigned newResultIndex = 0;
  for (unsigned resultIndex = 0; resultIndex < loop.getNumResults();
       ++resultIndex) {
    if (auto replacement = tensorReplacements.lookup(resultIndex)) {
      rewriter.replaceAllUsesWith(loop.getResult(resultIndex), replacement);
      continue;
    }
    if (isTensorStateIndex(states, resultIndex)) {
      continue;
    }
    rewriter.replaceAllUsesWith(loop.getResult(resultIndex),
                                newLoop.getResult(newResultIndex));
    ++newResultIndex;
  }
}

// Applies tensor state materialization to one loop. A loop without tensor
// iter_args is not a match and is left untouched by the pass driver.
static LogicalResult materializeLoopState(scf::ForOp loop,
                                          RewriterBase &rewriter) {
  SmallVector<TensorLoopState> states = collectTensorLoopStates(loop);
  if (states.empty()) {
    return failure();
  }

  // The DST strategy replaces the whole loop with one reduction compute.
  // Mixed loop-carried state needs the existing per-iteration materialization
  // to preserve ordering with non-accumulator updates.
  if (states.size() == 1 && states.front().accumulator &&
      succeeded(lowerTensorAccumulationToDst(*states.front().accumulator, loop,
                                             rewriter))) {
    return success();
  }

  auto funcOp = loop->getParentOfType<func::FuncOp>();
  assert(funcOp && "pass runs on func.func");
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  assert(moduleOp && "func.func must be nested in a module");

  for (TensorLoopState &state : states) {
    if (state.accumulator) {
      continue;
    }
    OpBuilder::InsertionGuard guard(rewriter);
    state.stateDFB = createCompilerAllocatedDFB(state.tensorType, loop.getLoc(),
                                                funcOp, moduleOp, rewriter);
  }

  createInitialStores(states, loop, rewriter);
  scf::ForOp newLoop = createLoopWithoutTensorState(loop, states, rewriter);

  IRMapping mapper;
  mapLoopCarriedValues(loop, newLoop, states, mapper, rewriter);
  cloneBodyAndMaterializeNextState(loop, newLoop, states, mapper, rewriter);
  replaceLoopResults(loop, newLoop, states, rewriter);

  rewriter.eraseOp(loop);
  return success();
}

struct TTLMaterializeLoopStatePass
    : public impl::TTLMaterializeLoopStateBase<TTLMaterializeLoopStatePass> {
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>(
        [&](scf::ForOp loop) { loops.push_back(loop); });

    // This transform moves and erases sibling ops around the matched loop, so
    // it is driven explicitly instead of relying on a greedy pattern worklist.
    // Postorder collection ensures nested loops are handled before a parent can
    // clone or erase its body, and RewriterBase reports every mutation.
    IRRewriter rewriter(&getContext());
    for (scf::ForOp loop : loops) {
      (void)materializeLoopState(loop, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
