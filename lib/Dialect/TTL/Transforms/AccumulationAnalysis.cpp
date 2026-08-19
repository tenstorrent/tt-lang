// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/AccumulationAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

#define DEBUG_TYPE "ttl-lower-accumulation-scopes"

namespace mlir::tt::ttl {

FailureOr<AccumulationStrategy> parseAccumulationStrategy(StringRef value) {
  return llvm::StringSwitch<FailureOr<AccumulationStrategy>>(value)
      .Case("auto", AccumulationStrategy::Auto)
      .Case("dst", AccumulationStrategy::Dst)
      .Case("l1-pack", AccumulationStrategy::L1Pack)
      .Default(failure());
}

StringRef stringifyAccumulationStrategy(AccumulationStrategy strategy) {
  switch (strategy) {
  case AccumulationStrategy::Auto:
    return "auto";
  case AccumulationStrategy::Dst:
    return "dst";
  case AccumulationStrategy::L1Pack:
    return "l1-pack";
  }
  llvm_unreachable("unknown accumulation strategy");
}

namespace {

static bool isInsideScope(Operation *operation, AccumulationScopeOp scope) {
  return operation &&
         operation->getParentOfType<AccumulationScopeOp>() == scope;
}

static bool valueDependsOn(Value value, Value root, AccumulationScopeOp scope,
                           llvm::SmallPtrSetImpl<Operation *> &visitedOps) {
  if (!value || !root) {
    return false;
  }
  if (value == root) {
    return true;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!isInsideScope(definingOp, scope) ||
      !visitedOps.insert(definingOp).second) {
    return false;
  }

  for (Value operand : definingOp->getOperands()) {
    if (valueDependsOn(operand, root, scope, visitedOps)) {
      return true;
    }
  }
  return false;
}

static bool backwardSliceUses(Value value, Value root,
                              AccumulationScopeOp scope) {
  if (!value || !root) {
    return false;
  }
  if (value == root) {
    return true;
  }

  BackwardSliceOptions options;
  options.inclusive = true;
  options.omitBlockArguments = false;
  options.omitUsesFromAbove = false;
  options.filter = [scope](Operation *operation) {
    return isInsideScope(operation, scope);
  };

  llvm::SetVector<Operation *> backwardSlice;
  if (failed(getBackwardSlice(value, &backwardSlice, options))) {
    llvm::SmallPtrSet<Operation *, 8> visitedOps;
    return valueDependsOn(value, root, scope, visitedOps);
  }

  Operation *rootDefiningOp = root.getDefiningOp();
  for (Operation *operation : backwardSlice) {
    if (operation == rootDefiningOp) {
      return true;
    }
    for (Value operand : operation->getOperands()) {
      if (operand == root) {
        return true;
      }
    }
  }

  return false;
}

static void addDependence(SmallVectorImpl<AccumulationDependence> &dependences,
                          unsigned sourceIndex, unsigned targetIndex,
                          AccumulationDependenceKind kind) {
  for (const AccumulationDependence &dependence : dependences) {
    if (dependence.sourceIndex == sourceIndex &&
        dependence.targetIndex == targetIndex && dependence.kind == kind) {
      return;
    }
  }
  dependences.push_back({sourceIndex, targetIndex, kind});
}

static bool isLegalSingleSlotGroup(const AccumulationGroupAnalysis &analysis) {
  return analysis.getGroups().size() == 1 &&
         analysis.getGroups().front().slotIndices.size() == 1 &&
         !analysis.hasCrossAccumulatorDependence();
}

static std::optional<int64_t> multiplyCost(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0) {
    return std::nullopt;
  }
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
    return std::nullopt;
  }
  return lhs * rhs;
}

static std::optional<int64_t> addCost(std::optional<int64_t> lhs,
                                      std::optional<int64_t> rhs) {
  if (!lhs || !rhs) {
    return std::nullopt;
  }
  if (*rhs > 0 && *lhs > std::numeric_limits<int64_t>::max() - *rhs) {
    return std::nullopt;
  }
  return *lhs + *rhs;
}

static AccumulationCostWeights
getCostWeights(AccumulationTargetArch targetArch) {
  switch (targetArch) {
  case AccumulationTargetArch::Blackhole:
    return {/*dfbHopFixedCost=*/210, /*dfbHopPerTileCost=*/67};
  case AccumulationTargetArch::WormholeB0:
    return {/*dfbHopFixedCost=*/286, /*dfbHopPerTileCost=*/136};
  case AccumulationTargetArch::Unknown:
    return {};
  }
  llvm_unreachable("unknown accumulation target architecture");
}

static StringRef stringifyTargetArch(AccumulationTargetArch targetArch) {
  switch (targetArch) {
  case AccumulationTargetArch::Blackhole:
    return "blackhole";
  case AccumulationTargetArch::WormholeB0:
    return "wormhole_b0";
  case AccumulationTargetArch::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown accumulation target architecture");
}

static void printOptionalCost(llvm::raw_ostream &os,
                              std::optional<int64_t> cost) {
  if (cost) {
    os << *cost;
    return;
  }
  os << "unknown";
}

static void printCandidate(llvm::raw_ostream &os,
                           const AccumulationStrategyCandidate &candidate) {
  os << "  candidate strategy="
     << stringifyAccumulationStrategy(candidate.strategy)
     << " legal=" << (candidate.legal ? "true" : "false");
  if (!candidate.legal) {
    os << " reason=\"" << candidate.reason << "\"\n";
    return;
  }

  const AccumulationCost &cost = candidate.cost;
  os << " estimated_cost=";
  printOptionalCost(os, cost.estimatedCost);
  os << " one_time_dfb_hops=" << cost.oneTimeDfbHops
     << " per_iteration_dfb_hops=" << cost.perIterationDfbHops
     << " one_time_pack_unpack_tiles=" << cost.oneTimePackUnpackTiles
     << " per_iteration_pack_unpack_tiles=" << cost.perIterationPackUnpackTiles
     << " dst_live_tiles=" << cost.dstLiveTiles
     << " pack_reconfigs=" << cost.packReconfigs << "\n";
}

static std::optional<int64_t>
estimateCost(const AccumulationCost &cost,
             const AccumulationCostWeights &weights) {
  if (!cost.iterations) {
    return std::nullopt;
  }

  std::optional<int64_t> iterationDfbHops =
      multiplyCost(cost.perIterationDfbHops, *cost.iterations);
  std::optional<int64_t> totalDfbHops =
      addCost(cost.oneTimeDfbHops, iterationDfbHops);
  std::optional<int64_t> dfbHopCost =
      totalDfbHops ? multiplyCost(*totalDfbHops, weights.dfbHopFixedCost)
                   : std::nullopt;

  std::optional<int64_t> iterationTiles =
      multiplyCost(cost.perIterationPackUnpackTiles, *cost.iterations);
  std::optional<int64_t> totalTiles =
      addCost(cost.oneTimePackUnpackTiles, iterationTiles);
  std::optional<int64_t> tileCost =
      totalTiles ? multiplyCost(*totalTiles, weights.dfbHopPerTileCost)
                 : std::nullopt;

  return addCost(dfbHopCost, tileCost);
}

static AccumulationStrategyCandidate
buildDstCandidate(TensorAccumulationMatch &match,
                  const DFBAcquireReleaseIndex &dfbIndex,
                  const AccumulationCostModel &costModel) {
  AccumulationStrategyCandidate candidate;
  candidate.strategy = AccumulationStrategy::Dst;

  FailureOr<TensorDstAccumulationInfo> info =
      analyzeTensorAccumulationForDst(match, dfbIndex);
  if (failed(info)) {
    candidate.reason =
        "expected a DST-compatible same-type additive recurrence with one "
        "loop-carried accumulator and one final store";
    return candidate;
  }

  candidate.legal = true;
  candidate.cost = costModel.computeDstCost(*info);
  return candidate;
}

static AccumulationStrategyCandidate
buildL1PackCandidate(TensorAccumulationMatch &match,
                     const DFBAcquireReleaseIndex &dfbIndex,
                     const AccumulationCostModel &costModel) {
  AccumulationStrategyCandidate candidate;
  candidate.strategy = AccumulationStrategy::L1Pack;

  FailureOr<TensorL1PackAccumulationInfo> info =
      analyzeTensorAccumulationForL1Pack(match, &dfbIndex);
  if (failed(info)) {
    candidate.reason =
        "expected one same-type additive recurrence with one final store";
    return candidate;
  }

  candidate.legal = true;
  candidate.cost = costModel.computeL1PackCost(*info);
  return candidate;
}

} // namespace

AccumulationGroupAnalysis::AccumulationGroupAnalysis(AccumulationScopeOp scope)
    : scope(scope) {
  SmallVector<AccumulationInitialMode> initialModes =
      scope.getAccumulationInitialModes();
  Block &body = scope.getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());

  unsigned initIndex = 0;
  for (auto [outputIndex, output] : llvm::enumerate(scope.getOutputs())) {
    Value init;
    if (initialModes[outputIndex] == AccumulationInitialMode::Init) {
      init = scope.getInits()[initIndex++];
    }

    BlockArgument stateArgument;
    stateArgument = body.getArgument(outputIndex);

    Value yieldedValue;
    yieldedValue = yield.getValues()[outputIndex];

    slots.push_back({static_cast<unsigned>(outputIndex), output, init,
                     initialModes[outputIndex], stateArgument, yieldedValue});
  }

  for (const AccumulationSlot &targetSlot : slots) {
    for (const AccumulationSlot &sourceSlot : slots) {
      if (targetSlot.index == sourceSlot.index) {
        continue;
      }
      bool yieldedValueIsUpdated =
          sourceSlot.yieldedValue != sourceSlot.stateArgument;
      if (yieldedValueIsUpdated &&
          backwardSliceUses(targetSlot.yieldedValue, sourceSlot.yieldedValue,
                            scope)) {
        addDependence(dependences, sourceSlot.index, targetSlot.index,
                      AccumulationDependenceKind::UpdatedState);
        continue;
      }
      if (backwardSliceUses(targetSlot.yieldedValue, sourceSlot.stateArgument,
                            scope)) {
        addDependence(dependences, sourceSlot.index, targetSlot.index,
                      AccumulationDependenceKind::PreviousState);
      }
    }
  }

  llvm::EquivalenceClasses<unsigned> equivalenceClasses;
  for (const AccumulationSlot &slot : slots) {
    equivalenceClasses.insert(slot.index);
  }
  for (const AccumulationDependence &dependence : dependences) {
    equivalenceClasses.unionSets(dependence.sourceIndex,
                                 dependence.targetIndex);
  }

  llvm::DenseMap<unsigned, unsigned> groupIndexByLeader;
  for (const AccumulationSlot &slot : slots) {
    unsigned leader = equivalenceClasses.getLeaderValue(slot.index);
    auto [it, inserted] = groupIndexByLeader.try_emplace(leader, groups.size());
    if (inserted) {
      groups.push_back({});
    }
    groups[it->second].slotIndices.push_back(slot.index);
  }

  for (const AccumulationDependence &dependence : dependences) {
    unsigned leader = equivalenceClasses.getLeaderValue(dependence.sourceIndex);
    unsigned groupIndex = groupIndexByLeader.lookup(leader);
    groups[groupIndex].dependences.push_back(dependence);
  }
}

AccumulationCostModel::AccumulationCostModel(AccumulationTargetArch targetArch)
    : targetArch(targetArch), weights(getCostWeights(targetArch)) {}

FailureOr<AccumulationCostModel>
AccumulationCostModel::forOperation(Operation *op) {
  std::string failureReason;
  FailureOr<std::optional<ttcore::Arch>> arch =
      resolveTargetArch(op, failureReason);
  if (failed(arch)) {
    op->emitOpError(failureReason);
    return failure();
  }
  if (!*arch) {
    return AccumulationCostModel(AccumulationTargetArch::Unknown);
  }
  switch (**arch) {
  case ttcore::Arch::Blackhole:
    return AccumulationCostModel(AccumulationTargetArch::Blackhole);
  case ttcore::Arch::WormholeB0:
    return AccumulationCostModel(AccumulationTargetArch::WormholeB0);
  case ttcore::Arch::Quasar:
    return AccumulationCostModel(AccumulationTargetArch::Unknown);
  }
  return AccumulationCostModel(AccumulationTargetArch::Unknown);
}

AccumulationCost AccumulationCostModel::computeDstCost(
    const TensorDstAccumulationInfo &info) const {
  AccumulationCost cost;
  cost.iterations = info.tripCount;
  cost.oneTimeDfbHops = 1;
  cost.oneTimePackUnpackTiles = info.unitTileCount;
  if (info.contributionResidency ==
      TensorAccumulationContributionResidency::Streamed) {
    cost.perIterationDfbHops = 2;
    cost.perIterationPackUnpackTiles = info.unitTileCount;
  } else {
    cost.oneTimeDfbHops += 1;
    cost.oneTimePackUnpackTiles += info.unitTileCount;
  }
  cost.dstLiveTiles = info.unitTileCount;
  if (targetArch != AccumulationTargetArch::Unknown) {
    cost.estimatedCost = estimateCost(cost, weights);
  }
  return cost;
}

AccumulationCost AccumulationCostModel::computeL1PackCost(
    const TensorL1PackAccumulationInfo &info) const {
  AccumulationCost cost;
  cost.iterations = info.tripCount;
  cost.oneTimeDfbHops = 1;
  cost.perIterationDfbHops = 2;
  if (info.unitTileCount) {
    cost.oneTimePackUnpackTiles = *info.unitTileCount;
    cost.perIterationPackUnpackTiles = 2 * *info.unitTileCount;
  }
  cost.packReconfigs = 2;
  if (targetArch != AccumulationTargetArch::Unknown) {
    cost.estimatedCost = estimateCost(cost, weights);
  }
  return cost;
}

bool AccumulationCostModel::isLessCostly(const AccumulationCost &lhs,
                                         const AccumulationCost &rhs) const {
  if (lhs.estimatedCost && rhs.estimatedCost &&
      *lhs.estimatedCost != *rhs.estimatedCost) {
    return *lhs.estimatedCost < *rhs.estimatedCost;
  }
  if (lhs.estimatedCost && !rhs.estimatedCost) {
    return true;
  }
  if (!lhs.estimatedCost && rhs.estimatedCost) {
    return false;
  }

  if (lhs.perIterationDfbHops != rhs.perIterationDfbHops) {
    return lhs.perIterationDfbHops < rhs.perIterationDfbHops;
  }
  if (lhs.oneTimeDfbHops != rhs.oneTimeDfbHops) {
    return lhs.oneTimeDfbHops < rhs.oneTimeDfbHops;
  }

  if (lhs.perIterationPackUnpackTiles != rhs.perIterationPackUnpackTiles) {
    return lhs.perIterationPackUnpackTiles < rhs.perIterationPackUnpackTiles;
  }
  if (lhs.oneTimePackUnpackTiles != rhs.oneTimePackUnpackTiles) {
    return lhs.oneTimePackUnpackTiles < rhs.oneTimePackUnpackTiles;
  }

  if (lhs.packReconfigs != rhs.packReconfigs) {
    return lhs.packReconfigs < rhs.packReconfigs;
  }

  return lhs.dstLiveTiles < rhs.dstLiveTiles;
}

FailureOr<AccumulationStrategyPlan>
planTensorAccumulationStrategy(AccumulationScopeOp scope,
                               TensorAccumulationMatch &match,
                               AccumulationStrategy requestedStrategy,
                               const DFBAcquireReleaseIndex &dfbIndex,
                               const AccumulationCostModel &costModel) {
  AccumulationStrategyPlan plan;
  AccumulationGroupAnalysis groupAnalysis(scope);
  assert(isLegalSingleSlotGroup(groupAnalysis) &&
         "tensor recurrence strategy planning requires one independent slot");

  if (requestedStrategy == AccumulationStrategy::Dst) {
    plan.candidates.push_back(buildDstCandidate(match, dfbIndex, costModel));
  } else if (requestedStrategy == AccumulationStrategy::L1Pack) {
    plan.candidates.push_back(buildL1PackCandidate(match, dfbIndex, costModel));
  } else {
    plan.candidates.push_back(buildDstCandidate(match, dfbIndex, costModel));
    plan.candidates.push_back(buildL1PackCandidate(match, dfbIndex, costModel));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "accumulation cost model target_arch="
                 << stringifyTargetArch(costModel.getTargetArch()) << "\n";
    for (const AccumulationStrategyCandidate &candidate : plan.candidates) {
      printCandidate(llvm::dbgs(), candidate);
    }
  });

  AccumulationStrategyCandidate *selected = nullptr;
  for (AccumulationStrategyCandidate &candidate : plan.candidates) {
    if (!candidate.legal) {
      continue;
    }
    if (!selected || costModel.isLessCostly(candidate.cost, selected->cost)) {
      selected = &candidate;
    }
  }
  if (!selected) {
    return failure();
  }

  plan.strategy = selected->strategy;
  plan.cost = selected->cost;
  LLVM_DEBUG(llvm::dbgs() << "  selected strategy="
                          << stringifyAccumulationStrategy(plan.strategy)
                          << "\n");
  return plan;
}

} // namespace mlir::tt::ttl
