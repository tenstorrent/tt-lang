// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_ANALYSIS_INTEGEREXPRESSIONEVALUATOR_H
#define TTLANG_ANALYSIS_INTEGEREXPRESSIONEVALUATOR_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"

#include <functional>
#include <optional>

namespace mlir::tt {

/// Evaluates exact scalar integer and index expressions without changing IR.
class IntegerExpressionEvaluator {
public:
  /// Supplies context-specific facts that are not encoded as IR constants.
  /// The returned bit width must match the SSA value type.
  using ValueEvaluator = std::function<std::optional<llvm::APInt>(Value)>;

  explicit IntegerExpressionEvaluator(ValueEvaluator valueEvaluator = {});

  /// Returns an exact value or nullopt when the expression is not proven.
  /// The IR and callback facts must remain stable for the evaluator's lifetime
  /// because results are cached by SSA value.
  std::optional<llvm::APInt> evaluate(Value value);

private:
  ValueEvaluator valueEvaluator;
  llvm::DenseMap<Value, std::optional<llvm::APInt>> cache;
};

} // namespace mlir::tt

#endif // TTLANG_ANALYSIS_INTEGEREXPRESSIONEVALUATOR_H
