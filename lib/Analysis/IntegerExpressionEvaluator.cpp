// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/IntegerExpressionEvaluator.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <utility>

namespace mlir::tt {

namespace {

enum class EvaluationTaskKind { Discover, Fold, ResolveReplacement };

struct EvaluationTask {
  Value value;
  EvaluationTaskKind kind = EvaluationTaskKind::Discover;
  Value replacement;
};

std::optional<std::uint32_t> getIntegerBitWidth(Type type) {
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    return integerType.getWidth();
  }
  if (type.isIndex()) {
    return IndexType::kInternalStorageBitWidth;
  }
  return std::nullopt;
}

} // namespace

IntegerExpressionEvaluator::IntegerExpressionEvaluator(
    ValueEvaluator valueEvaluator)
    : valueEvaluator(std::move(valueEvaluator)) {}

std::optional<llvm::APInt>
IntegerExpressionEvaluator::evaluate(Value requestedValue) {
  if (!requestedValue) {
    return std::nullopt;
  }
  if (auto cached = cache.find(requestedValue); cached != cache.end()) {
    return cached->second;
  }

  SmallVector<EvaluationTask> worklist{
      {requestedValue, EvaluationTaskKind::Discover, {}}};
  llvm::DenseSet<Value> activeValues;
  while (!worklist.empty()) {
    EvaluationTask task = worklist.pop_back_val();
    if (task.kind == EvaluationTaskKind::Discover) {
      if (cache.contains(task.value)) {
        continue;
      }
      std::optional<std::uint32_t> maybeBitWidth =
          getIntegerBitWidth(task.value.getType());
      if (!maybeBitWidth) {
        cache.try_emplace(task.value, std::nullopt);
        continue;
      }

      Attribute constant;
      if (matchPattern(task.value, m_Constant(&constant))) {
        if (auto integer = dyn_cast_or_null<IntegerAttr>(constant)) {
          cache.try_emplace(task.value,
                            integer.getValue().getBitWidth() == *maybeBitWidth
                                ? std::optional(integer.getValue())
                                : std::nullopt);
          continue;
        }
      }
      if (valueEvaluator) {
        if (std::optional<llvm::APInt> maybeValue =
                valueEvaluator(task.value)) {
          cache.try_emplace(task.value,
                            maybeValue->getBitWidth() == *maybeBitWidth
                                ? maybeValue
                                : std::nullopt);
          continue;
        }
      }

      auto result = dyn_cast<OpResult>(task.value);
      Operation *operation = task.value.getDefiningOp();
      if (!result || !operation || operation->getNumRegions() != 0) {
        cache.try_emplace(task.value, std::nullopt);
        continue;
      }
      if (!activeValues.insert(task.value).second) {
        cache.try_emplace(task.value, std::nullopt);
        continue;
      }

      worklist.push_back({task.value, EvaluationTaskKind::Fold, Value()});
      for (Value operand : llvm::reverse(operation->getOperands())) {
        if (operand.getType().isIntOrIndex()) {
          worklist.push_back({operand, EvaluationTaskKind::Discover, Value()});
        }
      }
      continue;
    }

    if (task.kind == EvaluationTaskKind::ResolveReplacement) {
      activeValues.erase(task.value);
      auto replacement = cache.find(task.replacement);
      cache.try_emplace(task.value, replacement != cache.end()
                                        ? replacement->second
                                        : std::nullopt);
      continue;
    }

    Operation *operation = task.value.getDefiningOp();
    assert(operation && "fold task requires a defining operation");
    SmallVector<Attribute> operandConstants;
    operandConstants.reserve(operation->getNumOperands());
    for (Value operand : operation->getOperands()) {
      if (operand.getType().isIntOrIndex()) {
        auto cached = cache.find(operand);
        assert(cached != cache.end() && "integer operand must be evaluated");
        if (cached->second) {
          operandConstants.push_back(
              IntegerAttr::get(operand.getType(), *cached->second));
          continue;
        }
      }
      Attribute constant;
      operandConstants.push_back(matchPattern(operand, m_Constant(&constant))
                                     ? constant
                                     : Attribute());
    }

    SmallVector<Value> originalOperands(operation->getOperands());
    DictionaryAttr originalAttrs = operation->getAttrDictionary();
    SmallVector<OpFoldResult> foldResults;
    LogicalResult foldStatus = operation->fold(operandConstants, foldResults);
    llvm::scope_exit restoreOperation([&] {
      operation->setOperands(originalOperands);
      operation->setAttrs(originalAttrs);
    });
    if (failed(foldStatus)) {
      activeValues.erase(task.value);
      cache.try_emplace(task.value, std::nullopt);
      continue;
    }
    if (foldResults.empty()) {
      activeValues.erase(task.value);
      cache.try_emplace(task.value, std::nullopt);
      continue;
    }
    if (foldResults.size() != operation->getNumResults()) {
      activeValues.erase(task.value);
      cache.try_emplace(task.value, std::nullopt);
      continue;
    }

    unsigned resultNumber = cast<OpResult>(task.value).getResultNumber();
    OpFoldResult foldResult = foldResults[resultNumber];
    if (Attribute attribute = dyn_cast<Attribute>(foldResult)) {
      auto integer = dyn_cast<IntegerAttr>(attribute);
      std::optional<std::uint32_t> maybeBitWidth =
          getIntegerBitWidth(task.value.getType());
      activeValues.erase(task.value);
      cache.try_emplace(task.value, integer && maybeBitWidth &&
                                            integer.getValue().getBitWidth() ==
                                                *maybeBitWidth
                                        ? std::optional(integer.getValue())
                                        : std::nullopt);
      continue;
    }

    Value replacement = cast<Value>(foldResult);
    if (!replacement || replacement == task.value ||
        !replacement.getType().isIntOrIndex()) {
      activeValues.erase(task.value);
      cache.try_emplace(task.value, std::nullopt);
      continue;
    }
    auto cachedReplacement = cache.find(replacement);
    if (cachedReplacement != cache.end()) {
      activeValues.erase(task.value);
      cache.try_emplace(task.value, cachedReplacement->second);
      continue;
    }
    worklist.push_back(
        {task.value, EvaluationTaskKind::ResolveReplacement, replacement});
    worklist.push_back({replacement, EvaluationTaskKind::Discover, Value()});
  }

  auto result = cache.find(requestedValue);
  assert(result != cache.end() && "evaluation must cache its result");
  return result->second;
}

} // namespace mlir::tt
