// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_ANALYSIS_EXECUTIONCOUNTANALYSIS_H
#define TTLANG_ANALYSIS_EXECUTIONCOUNTANALYSIS_H

#include "ttlang/Analysis/IntegerExpressionEvaluator.h"

#include "mlir/IR/Region.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::tt {

class ExecutionCountAnalysis;

/// Owns context-independent dataflow facts for execution-count queries.
///
/// The analyzed IR must remain unchanged while this state and its dependent
/// analyses exist. Constructing multiple ExecutionCountAnalysis instances from
/// one state reuses the dataflow solution for the root region.
class ExecutionCountAnalysisSharedState {
public:
  explicit ExecutionCountAnalysisSharedState(Region &rootRegion);
  ~ExecutionCountAnalysisSharedState();

  ExecutionCountAnalysisSharedState(
      ExecutionCountAnalysisSharedState &&) noexcept;
  ExecutionCountAnalysisSharedState &
  operator=(ExecutionCountAnalysisSharedState &&) noexcept;

  ExecutionCountAnalysisSharedState(const ExecutionCountAnalysisSharedState &) =
      delete;
  ExecutionCountAnalysisSharedState &
  operator=(const ExecutionCountAnalysisSharedState &) = delete;

private:
  class Impl;
  std::unique_ptr<Impl> impl;

  friend class ExecutionCountAnalysis;
};

/// Computes exact operation execution counts within one region invocation.
///
/// The analysis composes exact loop trip counts and exact region invocation
/// counts. When a branch depends on an induction variable, it may enumerate a
/// bounded number of loop iterations. A result is unknown when an exact count
/// depends on unresolved runtime selection or data, unsupported control
/// flow, more iterations than the enumeration limit, or 64-bit overflow.
class ExecutionCountAnalysis {
public:
  /// Evaluates context-specific integer values, such as launch coordinates.
  /// The returned bit width must match the SSA value type. Return nullopt for
  /// other values so the analysis can evaluate supported integer expressions.
  using SymbolValueEvaluator = IntegerExpressionEvaluator::ValueEvaluator;

  /// Returns an exact invocation count for a context-specific non-loop region.
  /// Returning nullopt delegates to RegionBranchOpInterface.
  using RegionInvocationCountEvaluator =
      std::function<std::optional<std::uint64_t>(Region &)>;

  /// Configures bounded loop-iteration enumeration.
  struct Options {
    /// Maximum number of loop iterations examined across all proof attempts.
    std::uint64_t maxEnumeratedIterations = 1'000'000;
  };

  /// `rootRegion` is assumed to execute once. Counts are relative to that
  /// invocation and are unknown for operations outside the region.
  explicit ExecutionCountAnalysis(
      Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator = {},
      RegionInvocationCountEvaluator regionInvocationCountEvaluator = {});
  /// Allows the consumer to set a different enumeration limit.
  ExecutionCountAnalysis(
      Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator,
      RegionInvocationCountEvaluator regionInvocationCountEvaluator,
      Options options);

  /// The shared state's root region is assumed to execute once. Counts are
  /// relative to that invocation and are unknown for operations outside the
  /// region. `sharedState` must outlive this analysis.
  explicit ExecutionCountAnalysis(
      ExecutionCountAnalysisSharedState &sharedState,
      SymbolValueEvaluator symbolValueEvaluator = {},
      RegionInvocationCountEvaluator regionInvocationCountEvaluator = {});
  /// Allows the consumer to set a different enumeration limit.
  ExecutionCountAnalysis(
      ExecutionCountAnalysisSharedState &sharedState,
      SymbolValueEvaluator symbolValueEvaluator,
      RegionInvocationCountEvaluator regionInvocationCountEvaluator,
      Options options);
  ~ExecutionCountAnalysis();

  ExecutionCountAnalysis(ExecutionCountAnalysis &&) noexcept;
  ExecutionCountAnalysis &operator=(ExecutionCountAnalysis &&) noexcept;

  ExecutionCountAnalysis(const ExecutionCountAnalysis &) = delete;
  ExecutionCountAnalysis &operator=(const ExecutionCountAnalysis &) = delete;

  /// Returns the exact number of executions, or nullopt when it is not proven.
  /// This includes null operations, parentless operations, and operations
  /// outside `rootRegion`. The analyzed IR and callback results must remain
  /// unchanged between queries because results are cached by enclosing block.
  std::optional<std::uint64_t> getExecutionCount(Operation *operation);

private:
  class Impl;
  std::unique_ptr<ExecutionCountAnalysisSharedState> ownedSharedState;
  std::unique_ptr<Impl> impl;
};

inline constexpr std::size_t kMaxRetainedExecutionCountQueries = 64;

/// Retains a bounded number of context-specific execution-count analyses.
/// Inserting a missing key at capacity invalidates all returned references.
template <typename Key>
class ExecutionCountAnalysisQueryCache {
public:
  template <typename Factory>
  ExecutionCountAnalysis &getOrCreate(const Key &key, Factory &&factory) {
    auto analysisIt = analyses.find(key);
    if (analysisIt != analyses.end()) {
      return *analysisIt->second;
    }
    if (analyses.size() >= kMaxRetainedExecutionCountQueries) {
      analyses.clear();
    }
    std::unique_ptr<ExecutionCountAnalysis> analysis = factory();
    assert(analysis && "execution-count analysis factory returned null");
    analysisIt = analyses.emplace(key, std::move(analysis)).first;
    return *analysisIt->second;
  }

  std::size_t size() const { return analyses.size(); }

private:
  std::map<Key, std::unique_ptr<ExecutionCountAnalysis>> analyses;
};

} // namespace mlir::tt

#endif // TTLANG_ANALYSIS_EXECUTIONCOUNTANALYSIS_H
