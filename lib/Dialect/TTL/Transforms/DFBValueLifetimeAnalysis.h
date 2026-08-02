// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBVALUELIFETIMEANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBVALUELIFETIMEANALYSIS_H

#include "DFBAcquireReleaseAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace mlir::tt::ttl {

/// Availability of one tensor value at a kernel program point.
enum class DFBValueAvailability {
  /// The value does not carry a DFB association.
  NotDFBBacked,

  /// Every reachable execution reaching the point retains the value's DFB
  /// storage.
  DefinitelyAvailable,

  /// The analysis cannot prove that every reachable execution retains the
  /// associated storage before the point.
  MayBeReleased,
};

/// Computes DFB-backed tensor availability without modifying IR.
///
/// Values derived from `ttl.cb_wait` or `ttl.cb_reserve` use that exact
/// acquisition identity. A general `ttl.attach_cb` association represents
/// storage available at kernel entry because the association itself has no DFB
/// protocol effect. Any reachable release on its DFB invalidates it
/// conservatively; a later association cannot reacquire that storage. MLIR's
/// dense dataflow framework provides region, CFG, and loop propagation.
///
/// The propagation and executable-state handling use MLIR's upstream dense
/// forward dataflow analyses:
/// https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/mlir/include/mlir/Analysis/DataFlow/DenseAnalysis.h
/// https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h
///
/// Availability is not range-sensitive: any partial release invalidates the
/// whole associated tensor, which may require an unnecessary materialization
/// but cannot permit a read from released storage. The analysis result is valid
/// only while `kernel` remains unchanged.
class DFBValueLifetimeAnalysis {
public:
  /// Builds and runs the analysis for `kernel`.
  static PlanningResult<std::unique_ptr<DFBValueLifetimeAnalysis>>
  create(func::FuncOp kernel);

  ~DFBValueLifetimeAnalysis();

  DFBValueLifetimeAnalysis(const DFBValueLifetimeAnalysis &) = delete;
  DFBValueLifetimeAnalysis &
  operator=(const DFBValueLifetimeAnalysis &) = delete;

  /// Returns the availability of `value` immediately before `consumer`.
  /// `consumer` must belong to the kernel passed to `create`.
  DFBValueAvailability getAvailability(Value value, Operation *consumer) const;

  /// Returns true when any DFB-backed value may be released before
  /// `consumer`.
  bool anyValueMayBeReleased(ValueRange values, Operation *consumer) const;

  /// Returns the lifecycle index consumed by the availability analysis.
  const DFBAcquireReleaseIndex &getAcquireReleaseIndex() const;

private:
  class Impl;

  explicit DFBValueLifetimeAnalysis(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBVALUELIFETIMEANALYSIS_H
