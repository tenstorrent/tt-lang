// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Materialize Loop State
//===----------------------------------------------------------------------===//
//
// Eliminates tensor-valued scf.for iter_args before compute lowering. Additive
// recurrence state lowers to accumulate stores; all other tensor state lowers
// through compiler-allocated DFB state slots.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

struct AccumulatorState {
  StoreOp finalStore;
  CBReserveOp reserve;
  AddOp add;
  Value contribution;
  SmallVector<AttachCBOp> deadReserveAttachOps;
};

struct TensorLoopState {
  unsigned resultIndex;
  RankedTensorType tensorType;
  Value initialValue;
  BlockArgument iterArg;
  Value yieldedValue;
  std::optional<AccumulatorState> accumulator;
  BindCBOp stateDFB;
};

static bool isTensorLoopState(scf::ForOp loop, unsigned resultIndex) {
  return isa<RankedTensorType>(loop.getInitArgs()[resultIndex].getType());
}

static std::optional<AccumulatorState> matchAccumulator(scf::ForOp loop,
                                                        unsigned resultIndex) {
  auto loopResult = loop.getResult(resultIndex);
  if (!loopResult.hasOneUse()) {
    return std::nullopt;
  }

  auto finalStore = dyn_cast<StoreOp>(*loopResult.getUsers().begin());
  if (!finalStore || finalStore.getAccumulate()) {
    return std::nullopt;
  }

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto add = yield.getOperand(resultIndex).getDefiningOp<AddOp>();
  if (!add || add->getBlock() != loop.getBody() ||
      !add.getResult().hasOneUse()) {
    return std::nullopt;
  }

  BlockArgument iterArg = loop.getRegionIterArgs()[resultIndex];
  Value contribution;
  if (add.getLhs() == iterArg) {
    contribution = add.getRhs();
  } else if (add.getRhs() == iterArg) {
    contribution = add.getLhs();
  } else {
    return std::nullopt;
  }
  if (contribution == iterArg) {
    return std::nullopt;
  }

  for (OpOperand &use : iterArg.getUses()) {
    if (use.getOwner() != add.getOperation()) {
      return std::nullopt;
    }
  }

  auto reserve = finalStore.getView().getDefiningOp<CBReserveOp>();
  if (!reserve) {
    return std::nullopt;
  }

  SmallVector<AttachCBOp> deadReserveAttachOps;
  for (OpOperand &reserveUse : reserve.getResult().getUses()) {
    Operation *owner = reserveUse.getOwner();
    if (owner == finalStore.getOperation()) {
      continue;
    }

    auto attach = dyn_cast<AttachCBOp>(owner);
    if (!attach || !attach.getResult().use_empty()) {
      return std::nullopt;
    }
    deadReserveAttachOps.push_back(attach);
  }

  if (finalStore->getBlock() != loop->getBlock() ||
      reserve->getBlock() != loop->getBlock()) {
    return std::nullopt;
  }

  return AccumulatorState{finalStore, reserve, add, contribution,
                          deadReserveAttachOps};
}

static SmallVector<TensorLoopState> collectTensorLoopStates(scf::ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());

  SmallVector<TensorLoopState> states;
  for (unsigned resultIndex = 0; resultIndex < loop.getNumResults();
       ++resultIndex) {
    if (!isTensorLoopState(loop, resultIndex)) {
      continue;
    }

    states.push_back(TensorLoopState{
        resultIndex,
        cast<RankedTensorType>(loop.getInitArgs()[resultIndex].getType()),
        loop.getInitArgs()[resultIndex], loop.getRegionIterArgs()[resultIndex],
        yield.getOperand(resultIndex), matchAccumulator(loop, resultIndex),
        BindCBOp()});
  }
  return states;
}

static bool isTensorStateIndex(ArrayRef<TensorLoopState> states,
                               unsigned resultIndex) {
  return llvm::any_of(states, [&](const TensorLoopState &state) {
    return state.resultIndex == resultIndex;
  });
}

static void createInitialStores(ArrayRef<TensorLoopState> states,
                                scf::ForOp loop, PatternRewriter &rewriter) {
  for (TensorLoopState state : states) {
    if (!state.accumulator) {
      continue;
    }
    CBReserveOp reserve = state.accumulator->reserve;
    if (!reserve->isBeforeInBlock(loop)) {
      reserve->moveBefore(loop);
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

static scf::ForOp createLoopWithoutTensorState(scf::ForOp loop,
                                               ArrayRef<TensorLoopState> states,
                                               PatternRewriter &rewriter) {
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

static void mapLoopCarriedValues(scf::ForOp loop, scf::ForOp newLoop,
                                 ArrayRef<TensorLoopState> states,
                                 IRMapping &mapper, PatternRewriter &rewriter) {
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

static void cloneBodyAndMaterializeNextState(scf::ForOp loop,
                                             scf::ForOp newLoop,
                                             ArrayRef<TensorLoopState> states,
                                             IRMapping &mapper,
                                             PatternRewriter &rewriter) {
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

static void replaceLoopResults(scf::ForOp loop, scf::ForOp newLoop,
                               ArrayRef<TensorLoopState> states,
                               PatternRewriter &rewriter) {
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
      loop.getResult(resultIndex).replaceAllUsesWith(replacement);
      continue;
    }
    if (isTensorStateIndex(states, resultIndex)) {
      continue;
    }
    loop.getResult(resultIndex)
        .replaceAllUsesWith(newLoop.getResult(newResultIndex));
    ++newResultIndex;
  }
}

struct MaterializeLoopState : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != loop.getNumResults()) {
      return rewriter.notifyMatchFailure(loop,
                                         "loop terminator is not scf.yield");
    }

    SmallVector<TensorLoopState> states = collectTensorLoopStates(loop);
    if (states.empty()) {
      return rewriter.notifyMatchFailure(loop, "loop has no tensor iter_args");
    }

    auto moduleOp = loop->getParentOfType<ModuleOp>();
    auto funcOp = loop->getParentOfType<func::FuncOp>();
    if (!moduleOp || !funcOp) {
      return rewriter.notifyMatchFailure(
          loop, "loop is not nested in a module function");
    }

    for (TensorLoopState &state : states) {
      if (state.accumulator) {
        continue;
      }
      OpBuilder::InsertionGuard guard(rewriter);
      state.stateDFB = createCompilerAllocatedDFB(
          state.tensorType, loop.getLoc(), funcOp, moduleOp, rewriter);
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
};

struct TTLMaterializeLoopStatePass
    : public impl::TTLMaterializeLoopStateBase<TTLMaterializeLoopStatePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MaterializeLoopState>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
