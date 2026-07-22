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
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt {

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
    /// Maximum loop iterations examined by one access-domain comparison.
    std::uint64_t maxEnumeratedLoopIterations = 1'000'000;
    /// Maximum index tuples produced by one access-domain comparison.
    std::uint64_t maxEnumeratedIndexTuples = 1'000'000;
  };

  ValueOriginAnalysis() = default;
  explicit ValueOriginAnalysis(Options options) : options(options) {}

  /// Returns a conservative origin set in deterministic order.
  /// An unresolved tensor producer remains a tensor-valued origin.
  llvm::SmallVector<Value> getOrigins(Value value) const;

private:
  Options options;
};

} // namespace mlir::tt

#endif // TTLANG_ANALYSIS_VALUEORIGINANALYSIS_H
