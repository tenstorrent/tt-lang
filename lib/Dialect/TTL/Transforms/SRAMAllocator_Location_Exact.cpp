// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocator_Internal.h"
#include "SRAMAllocator_Location_Internal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace mlir::tt::ttl::detail {
namespace {

using llvm::ArrayRef;
using llvm::SmallVector;

struct ExactLocationSearchBudget {
  uint64_t limit;
  uint64_t consumed = 0;
  bool limitReached = false;

  bool consume() {
    if (consumed == limit) {
      limitReached = true;
      return false;
    }
    ++consumed;
    return true;
  }
};

static FailureOr<SmallVector<uint64_t>>
buildExactCandidateOffsets(const SRAMLocationAllocationProblem &problem,
                           ExactLocationSearchBudget &searchBudget) {
  SmallVector<uint64_t> candidates;
  llvm::DenseSet<uint64_t> seenCandidates;
  uint64_t maximumBudget = 0;
  auto insertCandidate = [&](uint64_t offset) {
    if (!seenCandidates.insert(offset).second) {
      return true;
    }
    if (!searchBudget.consume()) {
      return false;
    }
    candidates.push_back(offset);
    return true;
  };
  for (const auto &location : problem.locations) {
    maximumBudget = std::max(maximumBudget, location.budgetBytes);
    if (!insertCandidate(location.payloadBaseOffset)) {
      return failure();
    }
  }
  for (const auto &region : problem.regions) {
    if (region.fixedOffset &&
        (!insertCandidate(*region.fixedOffset) ||
         !insertCandidate(*region.fixedOffset + region.bytes))) {
      return failure();
    }
  }
  for (const auto &region : problem.regions) {
    SmallVector<uint64_t> previousCandidates(candidates);
    for (uint64_t offset : previousCandidates) {
      if (offset > maximumBudget || region.bytes > maximumBudget - offset) {
        continue;
      }
      if (!insertCandidate(offset + region.bytes)) {
        return failure();
      }
    }
  }
  llvm::sort(candidates);
  return candidates;
}

class ExactLocationSearch {
public:
  ExactLocationSearch(const SRAMLocationAllocationProblem &problem,
                      ArrayRef<uint64_t> candidateOffsets,
                      ExactLocationSearchBudget &searchBudget,
                      std::optional<SRAMLocationAllocationSolution> initial)
      : problem(problem), variables(buildLocationPlacementVariables(problem)),
        candidateOffsets(candidateOffsets.begin(), candidateOffsets.end()),
        state(makeEmptyLocationState(problem)), searchBudget(searchBudget),
        best(std::move(initial)) {
    order = getLocationVariableOrder(variables, problem, true);
  }

  FailureOr<SRAMLocationAllocationSolution> solve(std::string &failureReason) {
    search(0);
    if (searchBudget.limitReached) {
      failureReason = "exact location placement examined " +
                      std::to_string(searchBudget.consumed) +
                      " work items and reached the " +
                      std::to_string(searchBudget.limit) + "-item limit";
      return failure();
    }
    if (!best) {
      failureReason = "exact placement proves that no location allocation fits";
      return failure();
    }
    return *best;
  }

private:
  uint64_t currentCost() const {
    return getLocationReservationBytes(problem, state.highWaterBytes);
  }

  void search(unsigned position) {
    if (!searchBudget.consume()) {
      return;
    }
    if (position == order.size()) {
      SRAMLocationAllocationSolution candidate{state.offsets,
                                               state.highWaterBytes};
      if (!best ||
          getLocationReservationBytes(problem, candidate.highWaterBytes) <
              getLocationReservationBytes(problem, best->highWaterBytes) ||
          (getLocationReservationBytes(problem, candidate.highWaterBytes) ==
               getLocationReservationBytes(problem, best->highWaterBytes) &&
           candidate.offsets < best->offsets)) {
        best = std::move(candidate);
      }
      return;
    }
    if (best && currentCost() >= getLocationReservationBytes(
                                     problem, best->highWaterBytes)) {
      return;
    }
    const LocationPlacementVariable &variable = variables[order[position]];
    ArrayRef<uint64_t> candidates = candidateOffsets;
    SmallVector<uint64_t> fixedCandidate;
    if (variable.fixedOffset) {
      fixedCandidate.push_back(*variable.fixedOffset);
      candidates = fixedCandidate;
    }
    for (uint64_t candidateOffset : candidates) {
      if (!fitsLocationVariableAtOffset(variable, candidateOffset, problem,
                                        state)) {
        continue;
      }
      LocationPlacementState savedState = state;
      placeLocationVariable(variable, candidateOffset, problem, state);
      search(position + 1);
      state = std::move(savedState);
      if (searchBudget.limitReached) {
        return;
      }
    }
  }

  const SRAMLocationAllocationProblem &problem;
  SmallVector<LocationPlacementVariable> variables;
  SmallVector<unsigned> order;
  SmallVector<uint64_t> candidateOffsets;
  LocationPlacementState state;
  ExactLocationSearchBudget &searchBudget;
  std::optional<SRAMLocationAllocationSolution> best;
};

} // namespace

FailureOr<SRAMLocationAllocationSolution>
allocateLocationsExactly(const SRAMLocationAllocationProblem &problem,
                         uint64_t searchWorkLimit, std::string &failureReason) {
  if (searchWorkLimit == 0) {
    failureReason = "exact allocation search limit must be positive";
    return failure();
  }
  if (problem.regions.empty()) {
    SRAMLocationAllocationSolution solution;
    for (const auto &location : problem.locations) {
      solution.highWaterBytes.push_back(location.payloadBaseOffset);
    }
    return solution;
  }
  std::string heuristicFailure;
  auto initial = allocateLocationsGreedy(problem, GreedyGapSelection::FirstFit,
                                         true, heuristicFailure);
  std::optional<SRAMLocationAllocationSolution> initialSolution;
  if (succeeded(initial)) {
    initialSolution = std::move(*initial);
  }
  ExactLocationSearchBudget searchBudget{searchWorkLimit};
  auto candidates = buildExactCandidateOffsets(problem, searchBudget);
  if (failed(candidates)) {
    failureReason = "exact location placement examined " +
                    std::to_string(searchBudget.consumed) +
                    " work items and reached the " +
                    std::to_string(searchBudget.limit) + "-item limit";
    return failure();
  }
  return ExactLocationSearch(problem, *candidates, searchBudget,
                             std::move(initialSolution))
      .solve(failureReason);
}

} // namespace mlir::tt::ttl::detail
