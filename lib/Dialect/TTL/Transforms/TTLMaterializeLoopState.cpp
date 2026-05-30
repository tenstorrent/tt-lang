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
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
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

// Returns the number of tiles represented by a statically ranked tensor. This
// pass only reaches this helper after rejecting dynamic tensor types.
static int64_t getTileCount(RankedTensorType tensorType) {
  assert(tensorType.hasStaticShape() && "expected static tensor shape");
  int64_t tileCount = 1;
  for (int64_t dim : tensorType.getShape()) {
    tileCount *= dim;
  }
  return tileCount;
}

// Identifies loop-carried tensors that must be removed before compute lowering.
static bool isTensorLoopState(scf::ForOp loop, unsigned resultIndex) {
  return isa<RankedTensorType>(loop.getInitArgs()[resultIndex].getType());
}

// Matches a recurrence `acc = add(acc, x)` whose loop result is consumed by
// a single non-accumulate store to a user-declared dataflow buffer.
// Preconditions: the add has a single use (the yield); the iter_arg is read
// only by the add; the contribution side is not the iter_arg itself; the
// reserve fed to the store has no live attach users; reserve and store sit in
// the loop's parent block.
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

// Finds the contribution wait only when it is immediately owned by the loop.
// Ancestor containment would admit nested-region effects that cannot be
// coalesced into one pre-loop wait without changing execution order.
static CBWaitOp getLoopLocalContributionWait(AccumulatorState &accumulator,
                                             scf::ForOp loop,
                                             AttachCBOp &attachedContribution) {
  Value contribution = accumulator.contribution;
  if (auto attach = contribution.getDefiningOp<AttachCBOp>()) {
    if (attach->getParentOp() != loop) {
      return {};
    }
    attachedContribution = attach;
    contribution = attach.getTensor();
  }

  auto wait = contribution.getDefiningOp<CBWaitOp>();
  if (!wait || wait->getParentOp() != loop) {
    return {};
  }
  return wait;
}

// Ensures replacing the whole loop with one reduction compute does not discard
// side effects other than the matched wait/attach/add recurrence.
static bool onlyContainsDstReductionOps(scf::ForOp loop,
                                        AccumulatorState &accumulator,
                                        CBWaitOp contributionWait,
                                        AttachCBOp attachedContribution) {
  DenseSet<Operation *> allowedOps;
  allowedOps.insert(accumulator.add.getOperation());
  allowedOps.insert(contributionWait.getOperation());
  if (attachedContribution) {
    allowedOps.insert(attachedContribution.getOperation());
  }

  for (Operation &bodyOp : loop.getBody()->without_terminator()) {
    if (!allowedOps.contains(&bodyOp)) {
      return false;
    }
  }
  return true;
}

// Prepends the reduction dimension so the coalesced wait exposes every
// per-iteration contribution as one compute input tensor.
static RankedTensorType
buildCoalescedContributionType(RankedTensorType unitType, int64_t tripCount) {
  SmallVector<int64_t> shape;
  shape.push_back(tripCount);
  llvm::append_range(shape, unitType.getShape());
  return RankedTensorType::get(shape, unitType.getElementType());
}

// Builds maps for a reduction domain whose final dimension indexes the
// coalesced contribution while the output and initial accumulator ignore it.
static SmallVector<Attribute>
buildDstReductionIndexingMaps(MLIRContext *ctx, int64_t outputRank) {
  int64_t domainRank = outputRank + 1;
  SmallVector<AffineExpr> parallelExprs;
  for (int64_t dim = 0; dim < outputRank; ++dim) {
    parallelExprs.push_back(getAffineDimExpr(dim, ctx));
  }

  SmallVector<AffineExpr> contributionExprs;
  contributionExprs.push_back(getAffineDimExpr(outputRank, ctx));
  llvm::append_range(contributionExprs, parallelExprs);

  AffineMap outputMap = AffineMap::get(domainRank, 0, parallelExprs, ctx);
  AffineMap contributionMap =
      AffineMap::get(domainRank, 0, contributionExprs, ctx);
  return {AffineMapAttr::get(outputMap), AffineMapAttr::get(contributionMap),
          AffineMapAttr::get(outputMap)};
}

// Marks output dimensions parallel and appends the single reduction dimension
// that drives repeated tile accumulation into DST.
static SmallVector<Attribute>
buildDstReductionIteratorTypes(RewriterBase &rewriter, int64_t outputRank) {
  SmallVector<Attribute> iteratorTypes;
  for (int64_t dim = 0; dim < outputRank; ++dim) {
    iteratorTypes.push_back(rewriter.getStringAttr("parallel"));
  }
  iteratorTypes.push_back(rewriter.getStringAttr("reduction"));
  return iteratorTypes;
}

// Materializes the restricted in-DST strategy:
//
//   %acc = scf.for ... iter_args(%acc = %init) {
//     %contribution = ttl.cb_wait %input
//     %next = ttl.add %acc, %contribution
//     scf.yield %next
//   }
//   ttl.store %acc, %reserved_output
//
// becomes:
//
//   %all_contributions = ttl.cb_wait %input, num_tiles = trip_count * tile_count
//   ttl.compute ins(%init, %all_contributions) outs(%reserved_output) {
//     %next = ttl.tile_accumulate_add %init_tile, %contribution_tile
//     ttl.tile_store %next, %reserved_output
//   }
//   ttl.cb_pop %input, num_tiles = trip_count * tile_count
//
// The generated compute owns the reduction loop so DST is acquired before the
// first contribution tile and released only after the final accumulated store.
static LogicalResult
tryMaterializeDstAccumulatingCompute(scf::ForOp loop, TensorLoopState &state,
                                     RewriterBase &rewriter) {
  // The in-DST strategy currently handles one tensor recurrence whose update is
  // exactly `state = state + contribution`, with matching per-iteration tensor
  // types. Other recurrence forms use the general materialization below.
  if (!state.accumulator) {
    return failure();
  }

  AccumulatorState &accumulator = *state.accumulator;
  if (state.accumulator->contribution.getType() != state.tensorType) {
    return failure();
  }

  // The initial accumulator must already be dataflow-buffer backed because the
  // generated compute reads it as the first input before reusing DST.
  if (!getAttachedCB(state.initialValue)) {
    return failure();
  }

  // Coalescing replaces one wait per iteration with one pre-compute wait, so the
  // total tile count must be known at compile time.
  std::optional<llvm::APInt> tripCount = loop.getStaticTripCount();
  if (!tripCount || tripCount->isZero() || tripCount->getActiveBits() > 63) {
    return failure();
  }
  int64_t tripCountValue = static_cast<int64_t>(tripCount->getZExtValue());

  // The contribution wait must be the canonical one-tensor wait immediately in
  // the loop body. Explicit num_tiles would need separate accounting, and nested
  // waits cannot be hoisted without moving region-local effects.
  AttachCBOp attachedContribution;
  CBWaitOp contributionWait =
      getLoopLocalContributionWait(accumulator, loop, attachedContribution);
  if (!contributionWait || contributionWait.getNumTiles().has_value()) {
    return failure();
  }
  if (!onlyContainsDstReductionOps(loop, accumulator, contributionWait,
                                   attachedContribution)) {
    return failure();
  }

  // The reduction compute preserves the original output tensor domain and adds
  // only a leading reduction dimension for the coalesced contributions.
  auto contributionType =
      dyn_cast<RankedTensorType>(contributionWait.getResult().getType());
  if (!contributionType || contributionType != state.tensorType ||
      !contributionType.hasStaticShape()) {
    return failure();
  }

  // The single coalesced wait must fit in the producer dataflow buffer. This is
  // a compile-time strategy selection, not a runtime capacity check.
  int64_t unitTileCount = getTileCount(contributionType);
  if (unitTileCount <= 0 ||
      tripCountValue > std::numeric_limits<int64_t>::max() / unitTileCount) {
    return failure();
  }
  int64_t totalContributionTiles = tripCountValue * unitTileCount;
  auto contributionCBType =
      cast<CircularBufferType>(contributionWait.getCb().getType());
  if (totalContributionTiles > contributionCBType.getTotalElements()) {
    return failure();
  }

  Location loc = loop.getLoc();
  CBReserveOp outputReserve = accumulator.reserve;
  // The output reserve must dominate both the generated compute output view and
  // the tile stores that reuse its explicit DST indices.
  if (!outputReserve->isBeforeInBlock(loop)) {
    rewriter.moveOpBefore(outputReserve, loop);
  }

  rewriter.setInsertionPoint(loop);
  IntegerAttr totalTilesAttr =
      rewriter.getI64IntegerAttr(totalContributionTiles);
  RankedTensorType coalescedType =
      buildCoalescedContributionType(contributionType, tripCountValue);
  CBWaitOp coalescedWait = CBWaitOp::create(
      rewriter, loc, coalescedType, contributionWait.getCb(), totalTilesAttr);
  AttachCBOp coalescedContribution =
      AttachCBOp::create(rewriter, loc, coalescedType,
                         coalescedWait.getResult(), contributionWait.getCb());

  // The compute output operand is a tensor view of the reserved output dataflow
  // buffer. The placeholder tensor has no data dependence; it only supplies the
  // tensor type needed by AttachCBOp.
  Value outputInit =
      tensor::EmptyOp::create(rewriter, loc, state.tensorType.getShape(),
                              state.tensorType.getElementType());
  Value output = AttachCBOp::create(rewriter, loc, state.tensorType, outputInit,
                                    outputReserve.getCb());

  SmallVector<Attribute> indexingMaps = buildDstReductionIndexingMaps(
      rewriter.getContext(), state.tensorType.getRank());
  SmallVector<Attribute> iteratorTypes =
      buildDstReductionIteratorTypes(rewriter, state.tensorType.getRank());

  // Input 0 is the initial accumulator tile, input 1 is the contribution tile
  // for the current reduction iteration, and output 0 is the reserved DST slot.
  auto compute = ComputeOp::create(
      rewriter, loc, TypeRange{state.tensorType},
      ValueRange{state.initialValue, coalescedContribution.getResult()},
      ValueRange{output}, rewriter.getArrayAttr(indexingMaps),
      rewriter.getArrayAttr(iteratorTypes));

  Block *body = rewriter.createBlock(&compute.getBody());
  Type tileType = state.tensorType.getElementType();
  body->addArgument(tileType, loc);
  body->addArgument(tileType, loc);
  body->addArgument(tileType, loc);

  rewriter.setInsertionPointToStart(body);
  SmallVector<Value> outputIndices;
  for (int64_t dim = 0; dim < state.tensorType.getRank(); ++dim) {
    outputIndices.push_back(
        IterIndexOp::create(rewriter, loc, rewriter.getI64IntegerAttr(dim)));
  }

  // TileAccumulateAddOp leaves the result in DST; the store records the final
  // output location without releasing DST between reduction iterations.
  auto accumulated = createTileOpWithPlaceholderDstIndex<TileAccumulateAddOp>(
      rewriter, loc, body->getArgument(0), body->getArgument(1));
  createTileOpWithPlaceholderDstIndex<TileStoreOp>(
      rewriter, loc, accumulated.getResult(), outputReserve.getResult(),
      outputIndices);
  YieldOp::create(rewriter, loc);

  rewriter.setInsertionPointAfter(compute);
  // The original per-iteration pops disappear with the loop, so the coalesced
  // wait needs one matching pop after compute consumes all contribution tiles.
  CBPopOp::create(rewriter, loc, contributionWait.getCb(), totalTilesAttr);

  // The new compute writes the original final store target, and the loop body
  // with its local wait/attach/add has been replaced completely.
  rewriter.eraseOp(accumulator.finalStore);
  for (AttachCBOp attach : accumulator.deadReserveAttachOps) {
    rewriter.eraseOp(attach);
  }
  rewriter.eraseOp(loop);
  return success();
}

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

    states.push_back(TensorLoopState{
        resultIndex,
        cast<RankedTensorType>(loop.getInitArgs()[resultIndex].getType()),
        loop.getInitArgs()[resultIndex], loop.getRegionIterArgs()[resultIndex],
        yield.getOperand(resultIndex), matchAccumulator(loop, resultIndex),
        BindCBOp()});
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
  if (states.size() == 1 && succeeded(tryMaterializeDstAccumulatingCompute(
                                loop, states.front(), rewriter))) {
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
