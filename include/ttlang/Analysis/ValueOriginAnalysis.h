// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Value Origin Analysis
//===----------------------------------------------------------------------===//
//
// This file declares a conservative analysis for finding the SSA definitions
// that may supply a value or one extracted tensor element.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_ANALYSIS_VALUEORIGINANALYSIS_H
#define TTLANG_ANALYSIS_VALUEORIGINANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::tt {

/// Deterministic possible origins returned by ValueOriginAnalysis.
class OriginSet {
public:
  using const_iterator = llvm::SmallVector<Value, 4>::const_iterator;

  OriginSet() = default;
  explicit OriginSet(llvm::SmallVector<Value, 4> values)
      : values(std::move(values)) {}

  bool empty() const { return values.empty(); }
  const_iterator begin() const { return values.begin(); }
  const_iterator end() const { return values.end(); }

  /// Return true when the set is nonempty and every origin matches.
  bool allMatch(llvm::function_ref<bool(Value)> predicate) const {
    return !empty() && llvm::all_of(values, predicate);
  }

  /// Return the single property produced by every origin.
  template <typename ResultT>
  FailureOr<ResultT>
  uniqueMapped(llvm::function_ref<FailureOr<ResultT>(Value)> mapOrigin) const {
    std::optional<ResultT> result;
    for (Value origin : values) {
      FailureOr<ResultT> mapped = mapOrigin(origin);
      if (failed(mapped) || (result && *result != *mapped)) {
        return failure();
      }
      result = *mapped;
    }
    if (!result) {
      return failure();
    }
    return *result;
  }

  /// Return the defining op shared by every origin.
  template <typename OpTy>
  FailureOr<OpTy>
  uniqueDefiningOp(llvm::function_ref<bool(OpTy)> accept) const {
    return uniqueMapped<OpTy>([&](Value origin) -> FailureOr<OpTy> {
      OpTy definition = origin.getDefiningOp<OpTy>();
      if (!definition || !accept(definition)) {
        return failure();
      }
      return definition;
    });
  }

  template <typename OpTy>
  FailureOr<OpTy> uniqueDefiningOp() const {
    return uniqueDefiningOp<OpTy>([](OpTy) { return true; });
  }

private:
  llvm::SmallVector<Value, 4> values;
};

/// Finds the possible source definitions of an SSA value.
///
/// The analysis follows values through control-flow merges, loop-carried
/// arguments and results, and one-input/one-result unrealized casts. For
/// `tensor.extract`, it also follows the extracted tensor element through
/// `tensor.cast` and `tensor.insert`. When traversal reaches an unmodeled
/// producer, its result remains an opaque origin so consumers cannot treat
/// unsupported value flow as a proven source.
class ValueOriginAnalysis {
public:
  /// Limits compile-time enumeration of finite loop and tensor index domains.
  struct Options {
    /// Maximum loop iterations examined by one origin query.
    std::uint64_t maxEnumeratedLoopIterations = 4096;
    /// Maximum index tuples produced by one origin query.
    std::uint64_t maxEnumeratedIndexTuples = 4096;
  };

  explicit ValueOriginAnalysis(Operation *root);
  ValueOriginAnalysis(Operation *root, Options options);
  ~ValueOriginAnalysis();

  ValueOriginAnalysis(ValueOriginAnalysis &&);
  ValueOriginAnalysis &operator=(ValueOriginAnalysis &&);
  ValueOriginAnalysis(const ValueOriginAnalysis &) = delete;
  ValueOriginAnalysis &operator=(const ValueOriginAnalysis &) = delete;

  /// Returns a conservative origin set in deterministic order.
  /// An unresolved tensor producer remains a tensor-valued origin.
  const OriginSet &getOrigins(Value value) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::tt

#endif // TTLANG_ANALYSIS_VALUEORIGINANALYSIS_H
