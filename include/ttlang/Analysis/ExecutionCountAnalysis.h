// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_ANALYSIS_EXECUTIONCOUNTANALYSIS_H
#define TTLANG_ANALYSIS_EXECUTIONCOUNTANALYSIS_H

#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

namespace mlir::tt {

/// Computes exact operation execution counts within one region invocation.
///
/// The analysis composes exact loop trip counts and exact region invocation
/// counts. When a branch depends on an induction variable, it may enumerate a
/// bounded number of loop iterations. A result is unknown when the relevant
/// control flow is dynamic, unsupported, exceeds the enumeration limit, or
/// overflows a 64-bit count.
class ExecutionCountAnalysis {
public:
  /// Evaluates context-specific integer values, such as launch coordinates.
  /// Return nullopt for other values so the analysis can evaluate constants,
  /// induction variables, and supported integer arithmetic.
  using SymbolValueEvaluator = std::function<std::optional<llvm::APInt>(Value)>;

  /// Returns an exact invocation count for a context-specific region.
  /// Returning nullopt delegates to the region branch and loop interfaces.
  using RegionInvocationCountEvaluator =
      std::function<std::optional<std::uint64_t>(Region &)>;

  struct Options {
    explicit Options(std::uint64_t maxEnumeratedIterations = 1'000'000)
        : maxEnumeratedIterations(maxEnumeratedIterations) {}

    /// Maximum number of loop iterations examined while proving a count.
    std::uint64_t maxEnumeratedIterations;
  };

  /// `rootRegion` is assumed to execute once. Counts are relative to that
  /// invocation and are unknown for operations outside the region.
  explicit ExecutionCountAnalysis(
      Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator = {},
      RegionInvocationCountEvaluator regionInvocationCountEvaluator = {},
      Options options = Options());
  ~ExecutionCountAnalysis();

  ExecutionCountAnalysis(ExecutionCountAnalysis &&) noexcept;
  ExecutionCountAnalysis &operator=(ExecutionCountAnalysis &&) noexcept;

  ExecutionCountAnalysis(const ExecutionCountAnalysis &) = delete;
  ExecutionCountAnalysis &operator=(const ExecutionCountAnalysis &) = delete;

  /// Returns the exact number of executions, or nullopt when it is not proven.
  /// The analyzed IR and callback results must remain unchanged between
  /// queries because results are cached by operation.
  std::optional<std::uint64_t> getExecutionCount(Operation *operation);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::tt

#endif // TTLANG_ANALYSIS_EXECUTIONCOUNTANALYSIS_H
