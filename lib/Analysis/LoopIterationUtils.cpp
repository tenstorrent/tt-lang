// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/LoopIterationUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ScopeExit.h"

#include <utility>

namespace mlir::tt {
namespace {

/// Evaluate an attribute or SSA expression for one enumerated iteration.
std::optional<llvm::APInt> evaluateIntegerAPInt(
    OpFoldResult expression, const LoopInductionBindings &bindings,
    const IntegerExpressionEvaluator::ValueEvaluator &valueEvaluator) {
  if (auto integer = dyn_cast<Attribute>(expression)) {
    auto integerAttr = dyn_cast<IntegerAttr>(integer);
    return integerAttr ? std::optional(integerAttr.getValue()) : std::nullopt;
  }
  return createLoopIntegerEvaluator(bindings, valueEvaluator)
      .evaluate(cast<Value>(expression));
}

/// Compute an `scf.for` trip count after evaluating its bound expressions.
///
/// `constantTripCount` retains MLIR's signedness and overflow semantics instead
/// of duplicating the trip-count formula here.
std::optional<llvm::APInt> getSCFForTripCount(
    scf::ForOp forOp, const LoopInductionBindings &bindings,
    const IntegerExpressionEvaluator::ValueEvaluator &valueEvaluator) {
  std::optional<llvm::APInt> maybeLowerBound =
      evaluateIntegerAPInt(forOp.getLowerBound(), bindings, valueEvaluator);
  std::optional<llvm::APInt> maybeUpperBound =
      evaluateIntegerAPInt(forOp.getUpperBound(), bindings, valueEvaluator);
  std::optional<llvm::APInt> maybeStep =
      evaluateIntegerAPInt(forOp.getStep(), bindings, valueEvaluator);
  if (!maybeLowerBound || !maybeUpperBound || !maybeStep ||
      maybeStep->isZero()) {
    return std::nullopt;
  }

  IntegerAttr lowerBoundAttr =
      IntegerAttr::get(forOp.getLowerBound().getType(), *maybeLowerBound);
  IntegerAttr upperBoundAttr =
      IntegerAttr::get(forOp.getUpperBound().getType(), *maybeUpperBound);
  IntegerAttr stepAttr =
      IntegerAttr::get(forOp.getStep().getType(), *maybeStep);
  return constantTripCount(lowerBoundAttr, upperBoundAttr, stepAttr,
                           /*isSigned=*/!forOp.getUnsignedCmp(),
                           scf::computeUbMinusLb);
}

LogicalResult enumerateLoopNestImpl(
    ArrayRef<LoopLikeOpInterface> loops, std::size_t loopIndex,
    LoopInductionBindings &bindings, EnumerationBudget &budget,
    function_ref<LogicalResult(const LoopInductionBindings &)> visitAssignment,
    const IntegerExpressionEvaluator::ValueEvaluator &valueEvaluator);

/// Enumerate every induction-variable dimension exposed by one loop.
class LoopDimensionEnumerator {
public:
  LoopDimensionEnumerator(
      ArrayRef<LoopLikeOpInterface> loops, std::size_t loopIndex,
      ArrayRef<Value> inductionVariables, ArrayRef<OpFoldResult> lowerBounds,
      ArrayRef<OpFoldResult> upperBounds, ArrayRef<OpFoldResult> steps,
      bool isUnsigned, LoopInductionBindings &bindings,
      EnumerationBudget &budget,
      function_ref<LogicalResult(const LoopInductionBindings &)>
          visitAssignment,
      const IntegerExpressionEvaluator::ValueEvaluator &valueEvaluator)
      : loops(loops), loopIndex(loopIndex),
        inductionVariables(inductionVariables), lowerBounds(lowerBounds),
        upperBounds(upperBounds), steps(steps), isUnsigned(isUnsigned),
        bindings(bindings), budget(budget), visitAssignment(visitAssignment),
        valueEvaluator(valueEvaluator) {}

  LogicalResult enumerate(std::size_t dimension) {
    if (dimension == inductionVariables.size()) {
      return enumerateLoopNestImpl(loops, loopIndex + 1, bindings, budget,
                                   visitAssignment, valueEvaluator);
    }

    std::optional<llvm::APInt> maybeLower =
        evaluateIntegerAPInt(lowerBounds[dimension], bindings, valueEvaluator);
    std::optional<llvm::APInt> maybeUpper =
        evaluateIntegerAPInt(upperBounds[dimension], bindings, valueEvaluator);
    std::optional<llvm::APInt> maybeStep =
        evaluateIntegerAPInt(steps[dimension], bindings, valueEvaluator);
    if (!maybeLower || !maybeUpper || !maybeStep ||
        maybeLower->getBitWidth() != maybeUpper->getBitWidth() ||
        maybeLower->getBitWidth() != maybeStep->getBitWidth() ||
        (isUnsigned ? maybeStep->isZero() : !maybeStep->isStrictlyPositive())) {
      return failure();
    }

    Value inductionVariable = inductionVariables[dimension];
    auto previousBinding = bindings.find(inductionVariable);
    std::optional<llvm::APInt> maybePreviousBinding =
        previousBinding == bindings.end()
            ? std::nullopt
            : std::optional<llvm::APInt>(previousBinding->second);
    llvm::scope_exit restoreBinding([&] {
      if (maybePreviousBinding) {
        bindings[inductionVariable] = *maybePreviousBinding;
      } else {
        bindings.erase(inductionVariable);
      }
    });

    llvm::APInt value = *maybeLower;
    auto isBeforeUpperBound = [&] {
      return isUnsigned ? value.ult(*maybeUpper) : value.slt(*maybeUpper);
    };
    while (isBeforeUpperBound()) {
      if (!budget.tryConsume()) {
        return failure();
      }
      bindings[inductionVariable] = value;
      if (failed(enumerate(dimension + 1))) {
        return failure();
      }

      bool overflow = false;
      value = isUnsigned ? value.uadd_ov(*maybeStep, overflow)
                         : value.sadd_ov(*maybeStep, overflow);
      if (overflow) {
        return failure();
      }
    }
    return success();
  }

private:
  ArrayRef<LoopLikeOpInterface> loops;
  std::size_t loopIndex;
  ArrayRef<Value> inductionVariables;
  ArrayRef<OpFoldResult> lowerBounds;
  ArrayRef<OpFoldResult> upperBounds;
  ArrayRef<OpFoldResult> steps;
  bool isUnsigned;
  LoopInductionBindings &bindings;
  EnumerationBudget &budget;
  function_ref<LogicalResult(const LoopInductionBindings &)> visitAssignment;
  const IntegerExpressionEvaluator::ValueEvaluator &valueEvaluator;
};

/// Recursively enumerate loop nesting and multi-dimensional loop interfaces.
///
/// Every active induction variable is present in `bindings` when
/// `visitAssignment` runs. Existing bindings are restored before return.
LogicalResult enumerateLoopNestImpl(
    ArrayRef<LoopLikeOpInterface> loops, std::size_t loopIndex,
    LoopInductionBindings &bindings, EnumerationBudget &budget,
    function_ref<LogicalResult(const LoopInductionBindings &)> visitAssignment,
    const IntegerExpressionEvaluator::ValueEvaluator &valueEvaluator) {
  if (loopIndex == loops.size()) {
    return visitAssignment(bindings);
  }

  LoopLikeOpInterface loop = loops[loopIndex];
  std::optional<SmallVector<Value>> inductionVariables =
      loop.getLoopInductionVars();
  std::optional<SmallVector<OpFoldResult>> lowerBounds =
      loop.getLoopLowerBounds();
  std::optional<SmallVector<OpFoldResult>> upperBounds =
      loop.getLoopUpperBounds();
  std::optional<SmallVector<OpFoldResult>> steps = loop.getLoopSteps();
  if (!inductionVariables || !lowerBounds || !upperBounds || !steps ||
      inductionVariables->size() != lowerBounds->size() ||
      inductionVariables->size() != upperBounds->size() ||
      inductionVariables->size() != steps->size()) {
    return failure();
  }

  bool isUnsigned = false;
  if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
    isUnsigned = forOp.getUnsignedCmp();
  }
  // Other LoopLike implementations expose no unsigned-comparison property.
  return LoopDimensionEnumerator(loops, loopIndex, *inductionVariables,
                                 *lowerBounds, *upperBounds, *steps, isUnsigned,
                                 bindings, budget, visitAssignment,
                                 valueEvaluator)
      .enumerate(0);
}

} // namespace

IntegerExpressionEvaluator createLoopIntegerEvaluator(
    const LoopInductionBindings &bindings,
    IntegerExpressionEvaluator::ValueEvaluator valueEvaluator) {
  return IntegerExpressionEvaluator(
      [&bindings, valueEvaluator = std::move(valueEvaluator)](Value value) {
        auto binding = bindings.find(value);
        if (binding != bindings.end()) {
          return std::optional<llvm::APInt>(binding->second);
        }
        return valueEvaluator ? valueEvaluator(value) : std::nullopt;
      });
}

std::optional<std::int64_t> evaluateIndexExpression(
    OpFoldResult expression, const LoopInductionBindings &bindings,
    IntegerExpressionEvaluator::ValueEvaluator valueEvaluator) {
  std::optional<llvm::APInt> maybeValue =
      evaluateIntegerAPInt(expression, bindings, std::move(valueEvaluator));
  if (!maybeValue || !maybeValue->isSignedIntN(64)) {
    return std::nullopt;
  }
  return maybeValue->getSExtValue();
}

std::optional<std::uint64_t>
getLoopTripCount(LoopLikeOpInterface loop,
                 const LoopInductionBindings &bindings,
                 IntegerExpressionEvaluator::ValueEvaluator valueEvaluator) {
  std::optional<llvm::APInt> maybeTripCount = loop.getStaticTripCount();
  if (!maybeTripCount) {
    if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
      maybeTripCount = getSCFForTripCount(forOp, bindings, valueEvaluator);
    }
  }
  if (!maybeTripCount || maybeTripCount->getActiveBits() > 64) {
    return std::nullopt;
  }
  return maybeTripCount->getZExtValue();
}

LogicalResult enumerateLoopNest(
    ArrayRef<LoopLikeOpInterface> loops, LoopInductionBindings &bindings,
    EnumerationBudget &budget,
    function_ref<LogicalResult(const LoopInductionBindings &)> visitAssignment,
    IntegerExpressionEvaluator::ValueEvaluator valueEvaluator) {
  return enumerateLoopNestImpl(loops, 0, bindings, budget, visitAssignment,
                               valueEvaluator);
}

} // namespace mlir::tt
