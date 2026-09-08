// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

namespace mlir::tt::ttkernel {

namespace {

/// Return whether `op` is a side-effect-free TTKernel value computation that
/// can be moved out of loops and single-region conditionals.
static bool isHoistableTTKernelValueComputation(Operation *op) {
  Dialect *dialect = op->getDialect();
  bool supportedDialect = isa<arith::ArithDialect, TTKernelDialect>(dialect);
  return supportedDialect && isPure(op);
}

/// Return whether `value` is defined outside `region`.
static bool isDefinedOutsideRegion(Value value, Region *region) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return true;
  }
  return !region->isAncestor(definingOp->getParentRegion());
}

/// Return whether two operations cannot execute on the same loop iteration.
static bool haveMutuallyExclusiveExecution(Operation *lhs, Operation *rhs,
                                           Operation *loop) {
  if (insideMutuallyExclusiveRegions(lhs, rhs)) {
    return true;
  }
  return haveDisjointExecutionCoreRanges(lhs, rhs, loop);
}

/// Deduplicate consecutive barriers of the same type and NoC. Barriers only
/// wait for transactions issued on the selected NoC.
template <typename BarrierOp>
struct DeduplicateConsecutiveBarriers : OpRewritePattern<BarrierOp> {
  using OpRewritePattern<BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter &rewriter) const override {
    if (auto *prev = op->getPrevNode()) {
      if (auto previousBarrier = dyn_cast<BarrierOp>(prev)) {
        if (previousBarrier.getNoc() == op.getNoc()) {
          rewriter.eraseOp(op);
          return success();
        }
      }
    }
    return failure();
  }
};

/// Move pure value computations whose operands are defined outside `op`.
static size_t hoistIfRegionInvariantValueOps(scf::IfOp op,
                                             PatternRewriter &rewriter) {
  SmallVector<Region *> regions{&op.getThenRegion()};
  if (!op.getElseRegion().empty()) {
    regions.push_back(&op.getElseRegion());
  }
  return moveLoopInvariantCode(
      regions,
      [](Value value, Region *region) {
        return isDefinedOutsideRegion(value, region);
      },
      [](Operation *candidate, Region *) {
        return isHoistableTTKernelValueComputation(candidate);
      },
      [&](Operation *candidate, Region *region) {
        rewriter.moveOpBefore(candidate, region->getParentOp());
      });
}

/// Hoist pure value computations from `scf.if` regions when all operands are
/// already available before the conditional.
struct HoistIfRegionInvariantValueOps : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    return success(hoistIfRegionInvariantValueOps(op, rewriter) != 0);
  }
};

/// Move pure value computations whose operands are defined outside `loop`.
static size_t hoistLoopInvariantValueOps(scf::ForOp loop,
                                         PatternRewriter &rewriter) {
  LoopLikeOpInterface loopInterface =
      cast<LoopLikeOpInterface>(loop.getOperation());
  return moveLoopInvariantCode(
      loopInterface.getLoopRegions(),
      [&](Value value, Region *) {
        return loopInterface.isDefinedOutsideOfLoop(value);
      },
      [](Operation *candidate, Region *) {
        return isHoistableTTKernelValueComputation(candidate);
      },
      [&](Operation *candidate, Region *) {
        rewriter.moveOpBefore(candidate, loop);
      });
}

/// Hoist pure value computations from loops when their operands are not defined
/// by the loop body.
struct HoistLoopInvariantValueOps : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    return success(hoistLoopInvariantValueOps(op, rewriter) != 0);
  }
};

// Check whether `operation` preserves the unpack/math configuration for copies
// from `sourceDFB`. Unclassified hardware operations and calls are
// conservative.
static bool preservesCopyTileConfiguration(Operation *operation,
                                           Value sourceDFB) {
  if (auto copy = dyn_cast<CopyTileOp>(operation)) {
    return copy.getCb0() == sourceDFB;
  }
  return isa<CBWaitFrontOp, CBPopFrontOp, CBReserveBackOp, CBPushBackOp,
             TileRegsAcquireOp, TileRegsCommitOp, TileRegsWaitOp,
             TileRegsReleaseOp, PackTileOp, PackTileBlockOp, scf::YieldOp>(
             operation) ||
         isHoistableTTKernelValueComputation(operation);
}

// Share one copy initialization across a nonempty loop whose straight-line
// body preserves that configuration and uses an invariant source DFB.
struct HoistInvariantCopyTileInit : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    std::optional<APInt> tripCount = loop.getStaticTripCount();
    if (!tripCount || tripCount->isZero()) {
      return rewriter.notifyMatchFailure(loop,
                                         "loop execution is not guaranteed");
    }
    SmallVector<CopyTileInitOp> initializations;
    Value sourceDFB;
    bool foundCopy = false;
    for (Operation &operation : loop.getBody()->getOperations()) {
      if (operation.getNumRegions() != 0) {
        return rewriter.notifyMatchFailure(
            loop, "copy loop contains nested control flow");
      }
      if (auto init = dyn_cast<CopyTileInitOp>(operation)) {
        if (!loop.isDefinedOutsideOfLoop(init.getCb0()) ||
            (sourceDFB && sourceDFB != init.getCb0())) {
          return rewriter.notifyMatchFailure(
              loop, "copy source is not loop invariant");
        }
        sourceDFB = init.getCb0();
        initializations.push_back(init);
        continue;
      }
      if (!preservesCopyTileConfiguration(&operation, sourceDFB)) {
        return rewriter.notifyMatchFailure(
            loop, "operation may change copy configuration");
      }
      foundCopy |= isa<CopyTileOp>(operation);
    }
    if (initializations.empty() || !foundCopy) {
      return failure();
    }

    // All iterations execute the same initialization before their first copy;
    // the complete body has been checked before moving or erasing any init.
    rewriter.moveOpBefore(initializations.front(), loop);
    for (CopyTileInitOp redundant : llvm::drop_begin(initializations)) {
      rewriter.eraseOp(redundant);
    }
    return success();
  }
};

/// A loop and the predicates that must guard its stateful write setup.
struct StatefulWriteLoop {
  scf::ForOp loop;
  SmallVector<scf::IfOp> predicates;
};

// Check `operation`, its nested operations, and their callees for uses or
// overwrites of resident write-command state.
static bool
mayUseOrOverwriteWriteCommand(Operation *operation,
                              NocCommandEffectsAnalysis &commandEffects) {
  WalkResult result = operation->walk([&](Operation *nested) {
    NocCommandEffects effects = commandEffects.getEffects(nested);
    return effects.mayReprogram || effects.mayUseState ? WalkResult::interrupt()
                                                       : WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Collect pure same-block definitions that must precede `anchor`.
// Dependencies precede their users in `definitions`.
static LogicalResult
collectValueDefinitionsToMove(Value value, Operation *anchor,
                              SmallPtrSetImpl<Operation *> &visited,
                              SmallVectorImpl<Operation *> &definitions) {
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getBlock() != anchor->getBlock() ||
      definition->isBeforeInBlock(anchor)) {
    return success();
  }
  if (!visited.insert(definition).second) {
    return success();
  }
  if (!isHoistableTTKernelValueComputation(definition)) {
    return failure();
  }
  for (Value operand : definition->getOperands()) {
    if (failed(collectValueDefinitionsToMove(operand, anchor, visited,
                                             definitions))) {
      return failure();
    }
  }
  definitions.push_back(definition);
  return success();
}

// Schedule one-packet write-state configuration before a blocking wait so the
// NoC command setup overlaps sender synchronization.
struct SchedulePostedNocWriteStateBeforeWait
    : OpRewritePattern<NocAsyncWriteOp> {
  using OpRewritePattern<NocAsyncWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NocAsyncWriteOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<bool> posted = op.getPosted();
    if (!posted || !*posted) {
      return rewriter.notifyMatchFailure(op, "write is not posted");
    }
    if (op.getDstCoreXY().size() != 2 || !op.getDstBankId().empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "write is not a unicast core write");
    }
    std::optional<int64_t> transferSize = getConstantIntValue(op.getSize());
    if (!transferSize || *transferSize <= 0 ||
        *transferSize > getTargetNocMaxBurstBytes(op)) {
      return rewriter.notifyMatchFailure(
          op, "write size is not a valid one-packet transfer");
    }
    if (op->getParentOfType<scf::ForOp>()) {
      return rewriter.notifyMatchFailure(
          op, "loop writes use loop-invariant state selection");
    }

    Operation *firstBlockingWait = nullptr;
    NocCommandEffectsAnalysis commandEffects(NocCommandClass::Write);
    for (Operation *previous = op->getPrevNode(); previous;
         previous = previous->getPrevNode()) {
      if (mayUseOrOverwriteWriteCommand(previous, commandEffects)) {
        break;
      }
      if (isa<CBWaitFrontOp, SemaphoreWaitOp, SemaphoreWaitMinOp>(previous)) {
        firstBlockingWait = previous;
      }
    }
    if (!firstBlockingWait) {
      return rewriter.notifyMatchFailure(
          op, "no preceding blocking wait can hide command setup");
    }

    SmallPtrSet<Operation *, 8> movedDefinitions;
    SmallVector<Operation *, 8> definitionsToMove;
    for (Value coordinate : op.getDstCoreXY()) {
      if (failed(collectValueDefinitionsToMove(coordinate, firstBlockingWait,
                                               movedDefinitions,
                                               definitionsToMove))) {
        return rewriter.notifyMatchFailure(
            op, "destination coordinate cannot dominate the blocking wait");
      }
    }
    if (failed(collectValueDefinitionsToMove(op.getSize(), firstBlockingWait,
                                             movedDefinitions,
                                             definitionsToMove)) ||
        (op.getNoc() && failed(collectValueDefinitionsToMove(
                            op.getNoc(), firstBlockingWait, movedDefinitions,
                            definitionsToMove)))) {
      return rewriter.notifyMatchFailure(
          op, "write configuration cannot dominate the blocking wait");
    }
    for (Operation *definition : definitionsToMove) {
      rewriter.moveOpBefore(definition, firstBlockingWait);
    }

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(firstBlockingWait);
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value destinationNocAddress =
        GetNocAddrOp::create(rewriter, loc, op.getDstCoreXY()[0],
                             op.getDstCoreXY()[1], zero, op.getNoc());
    NocAsyncWriteOnePacketSetStateOp::create(
        rewriter, loc, destinationNocAddress, op.getSize(), op.getNoc(),
        op.getPostedAttr());

    rewriter.setInsertionPoint(op);
    NocAsyncWriteOnePacketWithStateOp::create(
        rewriter, loc, op.getSrcLocalL1Addr(), op.getDstAddress(), op.getNoc(),
        op.getPostedAttr());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Return whether `loop` preserves the write command and setup predicate.
static LogicalResult analyzeStatefulWriteLoop(NocAsyncWriteOp op,
                                              scf::ForOp loop,
                                              StatefulWriteLoop &result) {
  std::optional<APInt> tripCount = loop.getStaticTripCount();
  if (!tripCount || tripCount->isZero()) {
    return failure();
  }

  auto isLoopInvariant = [&](Value value) {
    return !value || loop.isDefinedOutsideOfLoop(value);
  };
  if (!llvm::all_of(op.getDstCoreXY(), isLoopInvariant) ||
      !isLoopInvariant(op.getSize()) || !isLoopInvariant(op.getNoc())) {
    return failure();
  }

  SmallVector<scf::IfOp> enclosingPredicates;
  for (Operation *ancestor = op->getParentOp();
       ancestor && ancestor != loop.getOperation();
       ancestor = ancestor->getParentOp()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(ancestor)) {
      if (!ifOp.getElseRegion().empty() ||
          !ifOp.getThenRegion().isAncestor(op->getParentRegion()) ||
          !isLoopInvariant(ifOp.getCondition())) {
        return failure();
      }
      enclosingPredicates.push_back(ifOp);
      continue;
    }
    if (isa<scf::ForOp>(ancestor)) {
      continue;
    }
    if (ancestor->getNumRegions() != 0) {
      return failure();
    }
  }

  NocCommandEffectsAnalysis commandEffects(NocCommandClass::Write);
  WalkResult commandCheck = loop.walk([&](Operation *nestedOp) {
    if (nestedOp == op.getOperation()) {
      return WalkResult::advance();
    }
    NocCommandEffects effects = commandEffects.getEffects(nestedOp);
    if ((effects.mayReprogram || effects.mayUseState) &&
        !haveMutuallyExclusiveExecution(op, nestedOp, loop)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (commandCheck.wasInterrupted()) {
    return failure();
  }

  result = StatefulWriteLoop{loop, std::move(enclosingPredicates)};
  return success();
}

/// Return the outermost loop across which write command state remains valid.
static FailureOr<StatefulWriteLoop> findStatefulWriteLoop(NocAsyncWriteOp op) {
  std::optional<StatefulWriteLoop> selected;
  for (scf::ForOp loop = op->getParentOfType<scf::ForOp>(); loop;
       loop = loop->getParentOfType<scf::ForOp>()) {
    StatefulWriteLoop candidate;
    if (succeeded(analyzeStatefulWriteLoop(op, loop, candidate))) {
      selected = std::move(candidate);
    }
  }
  if (!selected) {
    return failure();
  }
  return std::move(*selected);
}

/// Select stateful one-packet writes when a loop preserves the resident write
/// command.
struct UseStatefulNocWriteInLoop : OpRewritePattern<NocAsyncWriteOp> {
  using OpRewritePattern<NocAsyncWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NocAsyncWriteOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getDstCoreXY().size() != 2 || !op.getDstBankId().empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "write is not a unicast core write");
    }

    std::optional<int64_t> transferSize = getConstantIntValue(op.getSize());
    if (!transferSize || *transferSize <= 0 ||
        *transferSize > getTargetNocMaxBurstBytes(op)) {
      return rewriter.notifyMatchFailure(
          op, "write size is not a valid one-packet transfer");
    }

    size_t numMoved = 0;
    for (Operation *ancestor = op->getParentOp(); ancestor;
         ancestor = ancestor->getParentOp()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(ancestor)) {
        numMoved += hoistIfRegionInvariantValueOps(ifOp, rewriter);
      }
    }
    for (scf::ForOp loop = op->getParentOfType<scf::ForOp>(); loop;
         loop = loop->getParentOfType<scf::ForOp>()) {
      numMoved += hoistLoopInvariantValueOps(loop, rewriter);
    }

    FailureOr<StatefulWriteLoop> maybeStatefulLoop = findStatefulWriteLoop(op);
    if (failed(maybeStatefulLoop)) {
      if (numMoved != 0) {
        return success();
      }
      return rewriter.notifyMatchFailure(
          op, "no enclosing loop preserves the NoC write command");
    }
    StatefulWriteLoop &statefulLoop = *maybeStatefulLoop;

    OpBuilder::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPoint(statefulLoop.loop);
    Location loc = op.getLoc();
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value destinationNocAddress =
        GetNocAddrOp::create(rewriter, loc, op.getDstCoreXY()[0],
                             op.getDstCoreXY()[1], zero, op.getNoc());

    for (scf::IfOp predicate : llvm::reverse(statefulLoop.predicates)) {
      auto setupIf = scf::IfOp::create(rewriter, loc, predicate.getCondition(),
                                       /*withElseRegion=*/false);
      if (Attribute executionCoreRanges =
              predicate->getAttr(kExecutionCoreRangesAttrName)) {
        setupIf->setAttr(kExecutionCoreRangesAttrName, executionCoreRanges);
      }
      rewriter.setInsertionPointToStart(&setupIf.getThenRegion().front());
    }
    NocAsyncWriteOnePacketSetStateOp::create(
        rewriter, loc, destinationNocAddress, op.getSize(), op.getNoc(),
        op.getPostedAttr());

    rewriter.setInsertionPoint(op);
    NocAsyncWriteOnePacketWithStateOp::create(
        rewriter, loc, op.getSrcLocalL1Addr(), op.getDstAddress(), op.getNoc(),
        op.getPostedAttr());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTTKernelCleanupPatterns(RewritePatternSet &patterns) {
  patterns.add<DeduplicateConsecutiveBarriers<NocAsyncReadBarrierOp>>(
      patterns.getContext());
  patterns.add<DeduplicateConsecutiveBarriers<NocAsyncWriteBarrierOp>>(
      patterns.getContext());
  patterns
      .add<HoistIfRegionInvariantValueOps, HoistLoopInvariantValueOps,
           HoistInvariantCopyTileInit, SchedulePostedNocWriteStateBeforeWait,
           UseStatefulNocWriteInLoop>(patterns.getContext());
}

} // namespace mlir::tt::ttkernel
