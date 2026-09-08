// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocator.h"
#include "SRAMAllocator_Internal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace mlir::tt::ttl {
namespace {

using llvm::ArrayRef;
using llvm::SmallVector;

struct ExactSearchBudget {
  uint64_t workLimit;
  uint64_t consumedWork = 0;
  bool limitReached = false;

  bool consumeWork() {
    if (consumedWork == workLimit) {
      limitReached = true;
      return false;
    }
    ++consumedWork;
    return true;
  }
};

static std::optional<uint64_t> getComponentLowerBound(
    ArrayRef<unsigned> component, ArrayRef<uint64_t> regionSizeUnits,
    const SRAMAllocationProblem &problem, uint64_t capacityUnits) {
  uint64_t lowerBound = 0;
  for (unsigned seedIndex : component) {
    SmallVector<unsigned> clique{seedIndex};
    SmallVector<unsigned> candidates;
    for (unsigned regionIndex : component) {
      if (problem.conflicts.interferes(seedIndex, regionIndex)) {
        candidates.push_back(regionIndex);
      }
    }
    llvm::stable_sort(candidates, [&](unsigned leftIndex, unsigned rightIndex) {
      if (regionSizeUnits[leftIndex] != regionSizeUnits[rightIndex]) {
        return regionSizeUnits[leftIndex] > regionSizeUnits[rightIndex];
      }
      unsigned leftDegree = problem.conflicts.degree(leftIndex);
      unsigned rightDegree = problem.conflicts.degree(rightIndex);
      if (leftDegree != rightDegree) {
        return leftDegree > rightDegree;
      }
      return leftIndex < rightIndex;
    });
    uint64_t cliqueWeight = regionSizeUnits[seedIndex];
    for (unsigned candidateIndex : candidates) {
      bool conflictsWithClique =
          llvm::all_of(clique, [&](unsigned memberIndex) {
            return problem.conflicts.interferes(candidateIndex, memberIndex);
          });
      if (!conflictsWithClique) {
        continue;
      }
      clique.push_back(candidateIndex);
      std::optional<uint64_t> nextCliqueWeight = llvm::checkedAddUnsigned(
          cliqueWeight, regionSizeUnits[candidateIndex]);
      if (!nextCliqueWeight || *nextCliqueWeight > capacityUnits) {
        return std::nullopt;
      }
      cliqueWeight = *nextCliqueWeight;
    }
    lowerBound = std::max(lowerBound, cliqueWeight);
  }
  return lowerBound;
}

static FailureOr<SmallVector<uint64_t>> getSubsetSumOffsets(
    ArrayRef<unsigned> component, ArrayRef<uint64_t> regionSizeUnits,
    uint64_t maximumOffsetUnits, ExactSearchBudget &searchBudget) {
  SmallVector<uint64_t> offsets{0};
  llvm::DenseSet<uint64_t> seenOffsets;
  seenOffsets.insert(0);
  if (!searchBudget.consumeWork()) {
    return failure();
  }
  for (unsigned regionIndex : component) {
    SmallVector<uint64_t> currentOffsets(offsets);
    for (uint64_t offsetUnits : currentOffsets) {
      uint64_t sizeUnits = regionSizeUnits[regionIndex];
      if (sizeUnits > maximumOffsetUnits ||
          offsetUnits > maximumOffsetUnits - sizeUnits) {
        continue;
      }
      uint64_t nextOffsetUnits = offsetUnits + sizeUnits;
      if (nextOffsetUnits > maximumOffsetUnits ||
          !seenOffsets.insert(nextOffsetUnits).second) {
        continue;
      }
      if (!searchBudget.consumeWork()) {
        return failure();
      }
      offsets.push_back(nextOffsetUnits);
    }
  }
  llvm::sort(offsets);
  return offsets;
}

struct ExactComponentSolution {
  SmallVector<uint64_t> offsetUnits;
  uint64_t highWaterUnits;
};

class ExactComponentSearch {
public:
  ExactComponentSearch(const SRAMAllocationProblem &problem,
                       ArrayRef<unsigned> component,
                       ArrayRef<uint64_t> regionSizeUnits,
                       ArrayRef<uint64_t> candidateOffsets,
                       uint64_t lowerBoundUnits,
                       std::optional<ExactComponentSolution> initialSolution,
                       uint64_t capacityUnits, ExactSearchBudget &searchBudget)
      : problem(problem), component(component.begin(), component.end()),
        regionSizeUnits(regionSizeUnits),
        candidateOffsets(candidateOffsets.begin(), candidateOffsets.end()),
        lowerBoundUnits(lowerBoundUnits), best(std::move(initialSolution)),
        capacityUnits(capacityUnits), searchBudget(searchBudget),
        currentOffsets(problem.regionBytes.size(), unassignedOffset) {
    order.assign(component.begin(), component.end());
    llvm::stable_sort(order, [&](unsigned leftIndex, unsigned rightIndex) {
      unsigned leftDegree = problem.conflicts.degree(leftIndex);
      unsigned rightDegree = problem.conflicts.degree(rightIndex);
      if (leftDegree != rightDegree) {
        return leftDegree > rightDegree;
      }
      if (regionSizeUnits[leftIndex] != regionSizeUnits[rightIndex]) {
        return regionSizeUnits[leftIndex] > regionSizeUnits[rightIndex];
      }
      return leftIndex < rightIndex;
    });
  }

  FailureOr<ExactComponentSolution> solve() {
    if (best && best->highWaterUnits == lowerBoundUnits) {
      return *best;
    }
    search(0, 0);
    if (searchBudget.limitReached) {
      return failure();
    }
    if (best) {
      return *best;
    }
    return failure();
  }

private:
  bool overlapsPlacedConflict(unsigned regionIndex, uint64_t offsetUnits,
                              uint64_t endUnits) const {
    for (unsigned otherIndex : component) {
      if (currentOffsets[otherIndex] == unassignedOffset ||
          !problem.conflicts.interferes(regionIndex, otherIndex)) {
        continue;
      }
      uint64_t otherEndUnits =
          currentOffsets[otherIndex] + regionSizeUnits[otherIndex];
      if (endUnits > currentOffsets[otherIndex] &&
          otherEndUnits > offsetUnits) {
        return true;
      }
    }
    return false;
  }

  void search(unsigned position, uint64_t highWaterUnits) {
    if (best && best->highWaterUnits == lowerBoundUnits) {
      return;
    }
    if (!searchBudget.consumeWork()) {
      return;
    }
    if (position == order.size()) {
      SmallVector<uint64_t> solutionOffsets;
      solutionOffsets.reserve(component.size());
      for (unsigned regionIndex : component) {
        solutionOffsets.push_back(currentOffsets[regionIndex]);
      }
      best = ExactComponentSolution{std::move(solutionOffsets), highWaterUnits};
      return;
    }

    unsigned regionIndex = order[position];
    uint64_t sizeUnits = regionSizeUnits[regionIndex];
    uint64_t maximumHighWater = best ? best->highWaterUnits - 1 : capacityUnits;
    for (uint64_t offsetUnits : candidateOffsets) {
      if (sizeUnits > maximumHighWater ||
          offsetUnits > maximumHighWater - sizeUnits) {
        break;
      }
      uint64_t endUnits = offsetUnits + sizeUnits;
      if (overlapsPlacedConflict(regionIndex, offsetUnits, endUnits)) {
        continue;
      }
      currentOffsets[regionIndex] = offsetUnits;
      search(position + 1, std::max(highWaterUnits, endUnits));
      currentOffsets[regionIndex] = unassignedOffset;
      if (searchBudget.limitReached ||
          (best && best->highWaterUnits == lowerBoundUnits)) {
        return;
      }
      maximumHighWater = best ? best->highWaterUnits - 1 : capacityUnits;
    }
  }

  static constexpr uint64_t unassignedOffset =
      std::numeric_limits<uint64_t>::max();
  const SRAMAllocationProblem &problem;
  SmallVector<unsigned> component;
  ArrayRef<uint64_t> regionSizeUnits;
  SmallVector<uint64_t> candidateOffsets;
  uint64_t lowerBoundUnits;
  std::optional<ExactComponentSolution> best;
  uint64_t capacityUnits;
  ExactSearchBudget &searchBudget;
  SmallVector<unsigned> order;
  SmallVector<uint64_t> currentOffsets;
};

static FailureOr<SRAMAllocationSolution>
allocateExactly(const SRAMAllocationProblem &problem, uint64_t searchWorkLimit,
                std::string &failureReason) {
  if (searchWorkLimit == 0) {
    failureReason = "exact allocation search limit must be positive";
    return failure();
  }
  if (problem.regionBytes.empty()) {
    return SRAMAllocationSolution{{}, 0};
  }

  std::string heuristicFailure;
  FailureOr<SRAMAllocationSolution> firstFit = detail::allocateGreedy(
      problem, detail::GreedyGapSelection::FirstFit, heuristicFailure);
  heuristicFailure.clear();
  FailureOr<SRAMAllocationSolution> bestFit = detail::allocateGreedy(
      problem, detail::GreedyGapSelection::BestFit, heuristicFailure);
  const SRAMAllocationSolution *heuristic = nullptr;
  if (succeeded(firstFit)) {
    heuristic = &*firstFit;
  }
  if (succeeded(bestFit) &&
      (!heuristic || bestFit->arenaBytes < heuristic->arenaBytes)) {
    heuristic = &*bestFit;
  }

  uint64_t alignment = problem.alignmentBytes;
  uint64_t capacityUnits =
      (problem.budgetBytes - problem.payloadBaseOffset) / alignment;
  SmallVector<uint64_t> regionSizeUnits;
  regionSizeUnits.reserve(problem.regionBytes.size());
  for (uint64_t regionBytes : problem.regionBytes) {
    regionSizeUnits.push_back(regionBytes / alignment);
  }

  SmallVector<uint64_t> exactOffsets(problem.regionBytes.size());
  uint64_t exactHighWaterUnits = 0;
  ExactSearchBudget searchBudget{searchWorkLimit};
  for (ArrayRef<unsigned> component :
       getInterferenceGraphConnectedComponents(problem.conflicts)) {
    std::optional<uint64_t> lowerBoundUnits = getComponentLowerBound(
        component, regionSizeUnits, problem, capacityUnits);
    if (!lowerBoundUnits || *lowerBoundUnits > capacityUnits) {
      failureReason = "exact placement proves that no allocation fits SRAM "
                      "budget " +
                      std::to_string(problem.budgetBytes) + " bytes";
      return failure();
    }

    std::optional<ExactComponentSolution> initialSolution;
    if (heuristic) {
      ExactComponentSolution componentSolution;
      componentSolution.highWaterUnits = 0;
      for (unsigned regionIndex : component) {
        uint64_t offsetUnits =
            (heuristic->offsets[regionIndex] - problem.payloadBaseOffset) /
            alignment;
        componentSolution.offsetUnits.push_back(offsetUnits);
        componentSolution.highWaterUnits =
            std::max(componentSolution.highWaterUnits,
                     offsetUnits + regionSizeUnits[regionIndex]);
      }
      if (componentSolution.highWaterUnits <= capacityUnits) {
        initialSolution = std::move(componentSolution);
      }
    }

    if (initialSolution &&
        initialSolution->highWaterUnits == *lowerBoundUnits) {
      for (auto [componentPosition, regionIndex] : llvm::enumerate(component)) {
        exactOffsets[regionIndex] =
            initialSolution->offsetUnits[componentPosition];
      }
      exactHighWaterUnits =
          std::max(exactHighWaterUnits, initialSolution->highWaterUnits);
      continue;
    }

    uint64_t maximumOffsetUnits =
        initialSolution ? initialSolution->highWaterUnits - 1 : capacityUnits;
    FailureOr<SmallVector<uint64_t>> candidates = getSubsetSumOffsets(
        component, regionSizeUnits, maximumOffsetUnits, searchBudget);
    if (failed(candidates)) {
      break;
    }
    ExactComponentSearch search(
        problem, component, regionSizeUnits, *candidates, *lowerBoundUnits,
        std::move(initialSolution), capacityUnits, searchBudget);
    FailureOr<ExactComponentSolution> componentSolution = search.solve();
    if (failed(componentSolution)) {
      if (searchBudget.limitReached) {
        break;
      }
      failureReason = "exact placement proves that no allocation fits SRAM "
                      "budget " +
                      std::to_string(problem.budgetBytes) + " bytes";
      return failure();
    }
    for (auto [componentPosition, regionIndex] : llvm::enumerate(component)) {
      exactOffsets[regionIndex] =
          componentSolution->offsetUnits[componentPosition];
    }
    exactHighWaterUnits =
        std::max(exactHighWaterUnits, componentSolution->highWaterUnits);
  }

  if (searchBudget.limitReached) {
    failureReason = "exact allocation examined " +
                    std::to_string(searchBudget.consumedWork) +
                    " work items and reached the " +
                    std::to_string(searchWorkLimit) + "-item limit";
    if (heuristic && heuristic->arenaBytes <= problem.budgetBytes) {
      failureReason += " after finding a feasible " +
                       std::to_string(heuristic->arenaBytes) +
                       "-byte arena without proving it is minimal";
    } else {
      failureReason += " before proving a minimum that fits the SRAM budget";
    }
    return failure();
  }

  uint64_t arenaBytes =
      problem.payloadBaseOffset + exactHighWaterUnits * alignment;
  for (uint64_t &offsetUnits : exactOffsets) {
    offsetUnits = problem.payloadBaseOffset + offsetUnits * alignment;
  }
  return SRAMAllocationSolution{std::move(exactOffsets), arenaBytes};
}

class ExactAllocator final : public SRAMAllocator {
public:
  explicit ExactAllocator(uint64_t searchWorkLimit)
      : searchWorkLimit(searchWorkLimit) {}

  llvm::StringRef getName() const override { return kExactSRAMAllocator; }

  FailureOr<SRAMAllocationSolution>
  allocateImpl(const SRAMAllocationProblem &problem,
               std::string &failureReason) const override {
    return allocateExactly(problem, searchWorkLimit, failureReason);
  }

private:
  uint64_t searchWorkLimit;
};

} // namespace

std::unique_ptr<SRAMAllocator>
detail::createExactSRAMAllocator(uint64_t searchWorkLimit) {
  return std::make_unique<ExactAllocator>(searchWorkLimit);
}

} // namespace mlir::tt::ttl
