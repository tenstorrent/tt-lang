// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Loop Iteration Utilities
//===----------------------------------------------------------------------===//
//
// This file declares shared utilities for evaluating loop trip counts and
// enumerating finite LoopLikeOpInterface induction-variable assignments within
// a bounded compile-time work budget.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_ANALYSIS_LOOPITERATIONUTILS_H
#define TTLANG_ANALYSIS_LOOPITERATIONUTILS_H

#include "ttlang/Analysis/IntegerExpressionEvaluator.h"

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <optional>

namespace mlir::tt {

/// Maps each induction variable to its value in one enumerated iteration.
using LoopInductionBindings = llvm::DenseMap<Value, llvm::APInt>;

/// Enforces one shared work limit across recursive compile-time enumeration.
class EnumerationBudget {
public:
  /// Set the number of items that the enumeration may still examine.
  explicit EnumerationBudget(std::uint64_t remaining) : remaining(remaining) {}

  /// Return whether `count` additional items fit without consuming them.
  bool canConsume(std::uint64_t count) const { return count <= remaining; }

  /// Consume one item, or return false when no budget remains.
  bool tryConsume() {
    if (remaining == 0) {
      return false;
    }
    --remaining;
    return true;
  }

private:
  /// Number of items that may still be consumed.
  std::uint64_t remaining;
};

/// Create an integer evaluator for one enumerated loop iteration.
///
/// Induction-variable bindings take precedence because they define the current
/// iteration. `valueEvaluator` supplies context-dependent values not present in
/// those bindings. The returned evaluator stores a reference to `bindings` and
/// must not outlive it.
IntegerExpressionEvaluator createLoopIntegerEvaluator(
    const LoopInductionBindings &bindings,
    IntegerExpressionEvaluator::ValueEvaluator valueEvaluator = {});

/// Evaluate an integer expression as a signed 64-bit value.
std::optional<std::int64_t> evaluateIndexExpression(
    OpFoldResult expression,
    const LoopInductionBindings &bindings = LoopInductionBindings(),
    IntegerExpressionEvaluator::ValueEvaluator valueEvaluator = {});

/// Return the exact trip count under the current induction-variable bindings.
///
/// The LoopLike interface supplies static counts. For `scf.for`, evaluable
/// bounds are passed to MLIR's `constantTripCount`. Other unknown counts remain
/// unknown.
std::optional<std::uint64_t> getLoopTripCount(
    LoopLikeOpInterface loop,
    const LoopInductionBindings &bindings = LoopInductionBindings(),
    IntegerExpressionEvaluator::ValueEvaluator valueEvaluator = {});

/// Enumerate the induction-variable assignments of `loops`, ordered from
/// outermost to innermost.
///
/// Each loop must expose aligned induction variables, bounds, and steps through
/// LoopLikeOpInterface. The function restores all prior bindings before it
/// returns. Unsupported expressions, invalid iteration ranges, arithmetic
/// overflow, callback failure, and exhausted budget return failure.
LogicalResult enumerateLoopNest(
    ArrayRef<LoopLikeOpInterface> loops, LoopInductionBindings &bindings,
    EnumerationBudget &budget,
    function_ref<LogicalResult(const LoopInductionBindings &)> visitAssignment,
    IntegerExpressionEvaluator::ValueEvaluator valueEvaluator = {});

} // namespace mlir::tt

#endif // TTLANG_ANALYSIS_LOOPITERATIONUTILS_H
