// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/IntegerExpressionEvaluator.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir::tt {

namespace {

/// Action performed when the evaluator pops a worklist task.
enum class EvaluationTaskKind { Discover, Fold, ResolveReplacement };

/// Deferred work needed to evaluate one SSA value without recursion.
struct EvaluationTask {
  Value value;
  EvaluationTaskKind kind = EvaluationTaskKind::Discover;
  Value replacement;
};

/// Return the scalar bit width used to evaluate an integer or index value.
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
      if (!result || !operation || operation->getNumRegions() != 0 ||
          operation->getNumSuccessors() != 0) {
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

    // Fold a detached clone because a fold hook may modify its operation in
    // place. Repeat after a modification until the fold returns a constant or
    // external replacement, or the operation state repeats.
    OwningOpRef<Operation *> foldedOperation(operation->cloneWithoutRegions());
    llvm::SmallDenseSet<llvm::hash_code, 4> seenFoldStates;
    SmallVector<OpFoldResult> foldResults;
    bool deferredReplacement = false;
    while (true) {
      llvm::hash_code stateHash =
          OperationEquivalence::computeHash(*foldedOperation);
      if (!seenFoldStates.insert(stateHash).second) {
        break;
      }

      SmallVector<Attribute> operandConstants;
      operandConstants.reserve((*foldedOperation)->getNumOperands());
      for (Value operand : (*foldedOperation)->getOperands()) {
        if (operand.getType().isIntOrIndex()) {
          auto cached = cache.find(operand);
          if (cached != cache.end() && cached->second) {
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

      foldResults.clear();
      if (failed((*foldedOperation)->fold(operandConstants, foldResults))) {
        break;
      }
      if (foldResults.empty()) {
        continue;
      }
      if (foldResults.size() != (*foldedOperation)->getNumResults()) {
        break;
      }

      std::size_t resultNumber = cast<OpResult>(task.value).getResultNumber();
      OpFoldResult foldResult = foldResults[resultNumber];
      if (Attribute attribute = dyn_cast<Attribute>(foldResult)) {
        auto integer = dyn_cast<IntegerAttr>(attribute);
        std::optional<std::uint32_t> maybeBitWidth =
            getIntegerBitWidth(task.value.getType());
        activeValues.erase(task.value);
        cache.try_emplace(task.value,
                          integer && maybeBitWidth &&
                                  integer.getValue().getBitWidth() ==
                                      *maybeBitWidth
                              ? std::optional(integer.getValue())
                              : std::nullopt);
        break;
      }

      Value replacement = cast<Value>(foldResult);
      if (replacement && replacement.getDefiningOp() == foldedOperation.get()) {
        continue;
      }
      if (!replacement || replacement == task.value ||
          !replacement.getType().isIntOrIndex()) {
        break;
      }
      auto cachedReplacement = cache.find(replacement);
      if (cachedReplacement != cache.end()) {
        activeValues.erase(task.value);
        cache.try_emplace(task.value, cachedReplacement->second);
        break;
      }
      worklist.push_back(
          {task.value, EvaluationTaskKind::ResolveReplacement, replacement});
      worklist.push_back({replacement, EvaluationTaskKind::Discover, Value()});
      deferredReplacement = true;
      break;
    }

    if (!cache.contains(task.value) && !deferredReplacement) {
      activeValues.erase(task.value);
      cache.try_emplace(task.value, std::nullopt);
    }
  }

  auto result = cache.find(requestedValue);
  assert(result != cache.end() && "evaluation must cache its result");
  return result->second;
}

} // namespace mlir::tt
