// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/DenseMap.h"

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

/// Return execution-domain constraints between `op` and `limit`.
static SmallVector<ArrayAttr>
getEnclosingExecutionCoreRanges(Operation *op, Operation *limit) {
  SmallVector<ArrayAttr> domains;
  for (Operation *ancestor = op->getParentOp(); ancestor && ancestor != limit;
       ancestor = ancestor->getParentOp()) {
    if (auto ranges =
            ancestor->getAttrOfType<ArrayAttr>(kExecutionCoreRangesAttrName)) {
      domains.push_back(ranges);
    }
  }
  return domains;
}

/// Return whether two core-range arrays have an empty intersection.
static bool haveDisjointCoreRanges(ArrayAttr lhs, ArrayAttr rhs) {
  if (lhs.empty() || rhs.empty()) {
    return false;
  }
  for (Attribute lhsAttr : lhs) {
    auto lhsRange = dyn_cast<ttcore::CoreRangeAttr>(lhsAttr);
    if (!lhsRange) {
      return false;
    }
    for (Attribute rhsAttr : rhs) {
      auto rhsRange = dyn_cast<ttcore::CoreRangeAttr>(rhsAttr);
      if (!rhsRange || lhsRange.intersects(rhsRange)) {
        return false;
      }
    }
  }
  return true;
}

/// Return whether two operations cannot execute on the same loop iteration.
static bool haveMutuallyExclusiveExecution(Operation *lhs, Operation *rhs,
                                           Operation *loop) {
  if (insideMutuallyExclusiveRegions(lhs, rhs)) {
    return true;
  }

  SmallVector<ArrayAttr> lhsDomains =
      getEnclosingExecutionCoreRanges(lhs, loop);
  SmallVector<ArrayAttr> rhsDomains =
      getEnclosingExecutionCoreRanges(rhs, loop);
  return llvm::any_of(lhsDomains, [&](ArrayAttr lhsDomain) {
    return llvm::any_of(rhsDomains, [&](ArrayAttr rhsDomain) {
      return haveDisjointCoreRanges(lhsDomain, rhsDomain);
    });
  });
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

/// A loop and the predicates that must guard its stateful write setup.
struct StatefulWriteLoop {
  scf::ForOp loop;
  SmallVector<scf::IfOp> predicates;
};

/// Cached transitive write-command interference for a callable operation.
enum class CallableWriteCommandInterference {
  Analyzing,
  Preserves,
  Interferes,
};

/// Return whether `operation` or a called function may interfere with setup.
static bool mayTransitivelyInterfereWithWriteCommand(
    Operation *operation,
    DenseMap<Operation *, CallableWriteCommandInterference> &callableEffects) {
  if (mayReprogramNocCommand(operation, NocCommandClass::Write) ||
      usesNocCommandState(operation, NocCommandClass::Write)) {
    return true;
  }

  auto call = dyn_cast<CallOpInterface>(operation);
  if (!call) {
    return false;
  }
  Operation *callableOperation = call.resolveCallable();
  auto callable = dyn_cast_or_null<CallableOpInterface>(callableOperation);
  Region *callableRegion = callable ? callable.getCallableRegion() : nullptr;
  if (!callableRegion) {
    return true;
  }

  auto cachedEffect = callableEffects.find(callableOperation);
  if (cachedEffect != callableEffects.end()) {
    // The remaining operations in an active recursive component have not been
    // analyzed, so the component cannot yet be proven to preserve command
    // state.
    return cachedEffect->second != CallableWriteCommandInterference::Preserves;
  }

  callableEffects[callableOperation] =
      CallableWriteCommandInterference::Analyzing;
  WalkResult walkResult = callableRegion->walk([&](Operation *nested) {
    if (mayTransitivelyInterfereWithWriteCommand(nested, callableEffects)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  bool interferes = walkResult.wasInterrupted();
  callableEffects[callableOperation] =
      interferes ? CallableWriteCommandInterference::Interferes
                 : CallableWriteCommandInterference::Preserves;
  return interferes;
}

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

  DenseMap<Operation *, CallableWriteCommandInterference> callableEffects;
  WalkResult commandCheck = loop.walk([&](Operation *nestedOp) {
    if (nestedOp == op.getOperation()) {
      return WalkResult::advance();
    }
    if (mayTransitivelyInterfereWithWriteCommand(nestedOp, callableEffects) &&
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
        rewriter, loc, destinationNocAddress, op.getSize(), op.getNoc());

    rewriter.setInsertionPoint(op);
    NocAsyncWriteOnePacketWithStateOp::create(
        rewriter, loc, op.getSrcLocalL1Addr(), op.getDstAddress(), op.getNoc());
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
  patterns.add<HoistIfRegionInvariantValueOps, HoistLoopInvariantValueOps,
               UseStatefulNocWriteInLoop>(patterns.getContext());
}

} // namespace mlir::tt::ttkernel
