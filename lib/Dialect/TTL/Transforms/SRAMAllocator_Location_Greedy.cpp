// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocator_Internal.h"
#include "SRAMAllocator_Location_Internal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <tuple>

namespace mlir::tt::ttl::detail {

using llvm::SmallVector;

static uint64_t getMinimumOffset(const LocationPlacementVariable &variable,
                                 const SRAMLocationAllocationProblem &problem) {
  uint64_t minimumOffset = 0;
  for (unsigned regionIndex : variable.regionIndices) {
    const auto &region = problem.regions[regionIndex];
    minimumOffset =
        std::max(minimumOffset,
                 problem.locations[region.locationIndex].payloadBaseOffset);
  }
  return minimumOffset;
}

static FailureOr<SmallVector<uint64_t>>
getCandidateOffsets(const LocationPlacementVariable &variable,
                    const SRAMLocationAllocationProblem &problem,
                    const LocationPlacementState &state,
                    std::string &failureReason) {
  if (variable.fixedOffset) {
    return SmallVector<uint64_t>{*variable.fixedOffset};
  }
  SmallVector<uint64_t> candidates{getMinimumOffset(variable, problem)};
  for (unsigned regionIndex : variable.regionIndices) {
    for (unsigned placedRegionIndex : state.placedRegions) {
      if (!problem.conflicts.interferes(regionIndex, placedRegionIndex)) {
        continue;
      }
      std::optional<uint64_t> placedEnd =
          llvm::checkedAddUnsigned(state.offsets[placedRegionIndex],
                                   problem.regions[placedRegionIndex].bytes);
      if (!placedEnd || *placedEnd > std::numeric_limits<uint64_t>::max() -
                                         (problem.alignmentBytes - 1)) {
        failureReason = "location placement offset overflowed during alignment";
        return failure();
      }
      candidates.push_back(llvm::alignTo(*placedEnd, problem.alignmentBytes));
    }
  }
  llvm::sort(candidates);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  return candidates;
}

static uint64_t getPlacementCost(const LocationPlacementVariable &variable,
                                 uint64_t offset,
                                 const SRAMLocationAllocationProblem &problem,
                                 const LocationPlacementState &state) {
  SmallVector<uint64_t> proposedHighWater = state.highWaterBytes;
  for (unsigned regionIndex : variable.regionIndices) {
    const auto &region = problem.regions[regionIndex];
    proposedHighWater[region.locationIndex] = std::max(
        proposedHighWater[region.locationIndex], offset + region.bytes);
  }
  return getLocationReservationBytes(problem, proposedHighWater) -
         getLocationReservationBytes(problem, state.highWaterBytes);
}

static FailureOr<uint64_t>
selectOffset(const LocationPlacementVariable &variable,
             const SRAMLocationAllocationProblem &problem,
             const LocationPlacementState &state, GreedyGapSelection selection,
             std::string &failureReason) {
  auto candidates =
      getCandidateOffsets(variable, problem, state, failureReason);
  if (failed(candidates)) {
    return failure();
  }
  std::optional<uint64_t> selectedOffset;
  std::optional<uint64_t> selectedCost;
  for (uint64_t candidate : *candidates) {
    if (!fitsLocationVariableAtOffset(variable, candidate, problem, state)) {
      continue;
    }
    uint64_t cost = getPlacementCost(variable, candidate, problem, state);
    if (!selectedOffset || selection == GreedyGapSelection::FirstFit ||
        std::tie(cost, candidate) < std::tie(*selectedCost, *selectedOffset)) {
      selectedOffset = candidate;
      selectedCost = cost;
    }
    if (selection == GreedyGapSelection::FirstFit) {
      break;
    }
  }
  if (!selectedOffset) {
    failureReason = "no placement offset fits every participating location";
    return failure();
  }
  return *selectedOffset;
}

static FailureOr<SRAMLocationAllocationSolution>
allocateGreedily(const SRAMLocationAllocationProblem &problem,
                 GreedyGapSelection selection, bool degreeAware,
                 std::string &failureReason) {
  SmallVector<LocationPlacementVariable> variables =
      buildLocationPlacementVariables(problem);
  LocationPlacementState state = makeEmptyLocationState(problem);
  for (unsigned variableIndex :
       getLocationVariableOrder(variables, problem, degreeAware)) {
    auto offset = selectOffset(variables[variableIndex], problem, state,
                               selection, failureReason);
    if (failed(offset)) {
      return failure();
    }
    placeLocationVariable(variables[variableIndex], *offset, problem, state);
  }
  return SRAMLocationAllocationSolution{std::move(state.offsets),
                                        std::move(state.highWaterBytes)};
}

FailureOr<SRAMLocationAllocationSolution>
allocateLocationsGreedy(const SRAMLocationAllocationProblem &problem,
                        GreedyGapSelection selection, bool degreeAware,
                        std::string &failureReason) {
  return allocateGreedily(problem, selection, degreeAware, failureReason);
}

} // namespace mlir::tt::ttl::detail
