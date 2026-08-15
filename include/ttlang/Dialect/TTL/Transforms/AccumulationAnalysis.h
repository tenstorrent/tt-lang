// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONANALYSIS_H

#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"

#include <optional>
#include <string>

namespace mlir::tt::ttl {

/// Storage mechanism requested for a semantic accumulation scope.
enum class AccumulationStrategy {
  Auto,
  Dst,
  L1Pack,
};

/// Target architecture used to select accumulation cost weights.
enum class AccumulationTargetArch {
  Unknown,
  WormholeB0,
  Blackhole,
};

/// Architecture-specific weights for accumulation storage traffic.
struct AccumulationCostWeights {
  int64_t dfbHopFixedCost = 0;
  int64_t dfbHopPerTileCost = 0;
};

/// Parse the user-facing spelling accepted by accumulation strategy options.
FailureOr<AccumulationStrategy> parseAccumulationStrategy(StringRef value);

/// Return the user-facing spelling for `strategy`.
StringRef stringifyAccumulationStrategy(AccumulationStrategy strategy);

/// Cross-output state dependence kind within one accumulation scope.
enum class AccumulationDependenceKind {
  /// The target output update reads the source output's incoming state.
  PreviousState,
  /// The target output update reads the source output's yielded state.
  UpdatedState,
};

/// Policy and state values for one accumulation scope output.
struct AccumulationSlot {
  unsigned index;
  Value output;
  Value init;
  AccumulationInitialMode initialMode;
  BlockArgument stateArgument;
  Value yieldedValue;
};

/// State dependence from one accumulation scope output to another.
struct AccumulationDependence {
  unsigned sourceIndex;
  unsigned targetIndex;
  AccumulationDependenceKind kind;
};

/// Dependence-connected output slots that must be planned together.
struct AccumulationGroup {
  SmallVector<unsigned> slotIndices;
  SmallVector<AccumulationDependence> dependences;
};

/// Partitions accumulation scope outputs by cross-output state dependence.
class AccumulationGroupAnalysis {
public:
  /// Analyze output state dependences in `scope`.
  explicit AccumulationGroupAnalysis(AccumulationScopeOp scope);

  /// Return the analyzed accumulation scope.
  AccumulationScopeOp getScope() const { return scope; }

  /// Return output slots in accumulation scope output order.
  ArrayRef<AccumulationSlot> getSlots() const { return slots; }

  /// Return dependences between distinct accumulation scope outputs.
  ArrayRef<AccumulationDependence> getDependences() const {
    return dependences;
  }

  /// Return output groups that can be planned independently.
  ArrayRef<AccumulationGroup> getGroups() const { return groups; }

  /// Return true if any output update reads another output's state.
  bool hasCrossAccumulatorDependence() const { return !dependences.empty(); }

private:
  AccumulationScopeOp scope;
  SmallVector<AccumulationSlot> slots;
  SmallVector<AccumulationDependence> dependences;
  SmallVector<AccumulationGroup> groups;
};

/// Storage-traffic features used to compare legal accumulation strategies.
struct AccumulationCost {
  /// Loop iterations used to scale per-iteration costs when statically known.
  std::optional<int64_t> iterations;
  /// Dataflow-buffer handoffs that do not scale with loop iteration count.
  int64_t oneTimeDfbHops = 0;
  /// Dataflow-buffer handoffs incurred by each loop iteration.
  int64_t perIterationDfbHops = 0;
  /// Pack/unpack tile traffic that does not scale with loop iteration count.
  int64_t oneTimePackUnpackTiles = 0;
  /// Pack/unpack tile traffic incurred by each loop iteration.
  int64_t perIterationPackUnpackTiles = 0;
  /// Tiles that must remain live in DST while the strategy executes.
  int64_t dstLiveTiles = 0;
  /// Packer L1-accumulation reconfiguration operations.
  int64_t packReconfigs = 0;
  /// Relative cost score when all scaled cost features are statically known.
  std::optional<int64_t> estimatedCost;
};

/// Legalization and cost result for one candidate strategy.
struct AccumulationStrategyCandidate {
  AccumulationStrategy strategy;
  bool legal = false;
  AccumulationCost cost;
  std::string reason;
};

/// Selected accumulation strategy and the candidates considered.
struct AccumulationStrategyPlan {
  AccumulationStrategy strategy;
  AccumulationCost cost;
  SmallVector<AccumulationStrategyCandidate> candidates;
};

/// Compares legal accumulation strategies using storage-traffic cost features.
class AccumulationCostModel {
public:
  explicit AccumulationCostModel(
      AccumulationTargetArch targetArch = AccumulationTargetArch::Unknown);

  /// Create a cost model using the target architecture recorded on `op`.
  /// Fails when the recorded architecture cannot be resolved.
  static FailureOr<AccumulationCostModel> forOperation(Operation *op);

  /// Return the target architecture used by this model.
  AccumulationTargetArch getTargetArch() const { return targetArch; }

  /// Compute the cost features for a DST-resident tensor accumulation.
  AccumulationCost computeDstCost(const TensorDstAccumulationInfo &info) const;

  /// Compute the cost features for packer L1 accumulation.
  AccumulationCost
  computeL1PackCost(const TensorL1PackAccumulationInfo &info) const;

  /// Return true if `lhs` is preferred over `rhs`.
  bool isLessCostly(const AccumulationCost &lhs,
                    const AccumulationCost &rhs) const;

private:
  AccumulationTargetArch targetArch;
  AccumulationCostWeights weights;
};

/// Select a legal tensor accumulation strategy using immutable DFB lifecycle
/// facts from the IR version that will be lowered.
FailureOr<AccumulationStrategyPlan> planTensorAccumulationStrategy(
    AccumulationScopeOp scope, TensorAccumulationMatch &match,
    AccumulationStrategy requestedStrategy,
    const DFBAcquireReleaseIndex &dfbIndex,
    const AccumulationCostModel &costModel = AccumulationCostModel());

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONANALYSIS_H
