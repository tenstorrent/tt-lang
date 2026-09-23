// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocator_Location_Internal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

namespace mlir::tt::ttl::detail {

using llvm::ArrayRef;
using llvm::SmallVector;

constexpr uint64_t kUnassignedOffset = std::numeric_limits<uint64_t>::max();

SmallVector<LocationPlacementVariable>
buildLocationPlacementVariables(const SRAMLocationAllocationProblem &problem) {
  SmallVector<LocationPlacementVariable> variables;
  llvm::DenseSet<unsigned> groupedRegions;
  for (const auto &group : problem.equalOffsetGroups) {
    LocationPlacementVariable variable;
    variable.regionIndices = group.regionIndices;
    for (unsigned regionIndex : group.regionIndices) {
      groupedRegions.insert(regionIndex);
      if (problem.regions[regionIndex].fixedOffset) {
        variable.fixedOffset = problem.regions[regionIndex].fixedOffset;
      }
    }
    variables.push_back(std::move(variable));
  }
  for (unsigned regionIndex = 0; regionIndex < problem.regions.size();
       ++regionIndex) {
    if (!groupedRegions.contains(regionIndex)) {
      variables.push_back(
          {{regionIndex}, problem.regions[regionIndex].fixedOffset});
    }
  }
  return variables;
}

static uint64_t getVariableBytes(const LocationPlacementVariable &variable,
                                 const SRAMLocationAllocationProblem &problem) {
  uint64_t bytes = 0;
  for (unsigned regionIndex : variable.regionIndices) {
    uint64_t regionBytes = problem.regions[regionIndex].bytes;
    if (regionBytes > std::numeric_limits<uint64_t>::max() - bytes) {
      return std::numeric_limits<uint64_t>::max();
    }
    bytes += regionBytes;
  }
  return bytes;
}

static unsigned
getVariableDegree(const LocationPlacementVariable &variable,
                  const SRAMLocationAllocationProblem &problem) {
  llvm::DenseSet<unsigned> neighbors;
  for (unsigned regionIndex : variable.regionIndices) {
    for (int neighborIndex :
         problem.conflicts.getNeighbors(regionIndex).set_bits()) {
      neighbors.insert(static_cast<unsigned>(neighborIndex));
    }
  }
  return neighbors.size();
}

static bool regionsOverlap(uint64_t leftOffset, uint64_t leftBytes,
                           uint64_t rightOffset, uint64_t rightBytes) {
  return leftOffset < rightOffset + rightBytes &&
         rightOffset < leftOffset + leftBytes;
}

static uint64_t
getProposedHighWater(unsigned locationIndex,
                     const LocationPlacementVariable &variable, uint64_t offset,
                     const SRAMLocationAllocationProblem &problem,
                     const LocationPlacementState &state) {
  uint64_t highWater = state.highWaterBytes[locationIndex];
  for (unsigned regionIndex : variable.regionIndices) {
    const auto &region = problem.regions[regionIndex];
    if (region.locationIndex == locationIndex) {
      highWater = std::max(highWater, offset + region.bytes);
    }
  }
  return highWater;
}

bool fitsLocationVariableAtOffset(const LocationPlacementVariable &variable,
                                  uint64_t offset,
                                  const SRAMLocationAllocationProblem &problem,
                                  const LocationPlacementState &state) {
  for (unsigned regionIndex : variable.regionIndices) {
    const auto &region = problem.regions[regionIndex];
    const auto &location = problem.locations[region.locationIndex];
    if (offset < location.payloadBaseOffset || offset > location.budgetBytes ||
        region.bytes > location.budgetBytes - offset) {
      return false;
    }
    for (unsigned placedRegionIndex : state.placedRegions) {
      if (!problem.conflicts.interferes(regionIndex, placedRegionIndex)) {
        continue;
      }
      const auto &placedRegion = problem.regions[placedRegionIndex];
      if (regionsOverlap(offset, region.bytes, state.offsets[placedRegionIndex],
                         placedRegion.bytes)) {
        return false;
      }
    }
  }
  for (const auto &group : problem.equalCapacityGroups) {
    uint64_t capacity = 0;
    for (unsigned locationIndex : group.locationIndices) {
      capacity =
          std::max(capacity, getProposedHighWater(locationIndex, variable,
                                                  offset, problem, state));
    }
    for (unsigned locationIndex : group.locationIndices) {
      if (capacity > problem.locations[locationIndex].budgetBytes) {
        return false;
      }
    }
  }
  return true;
}

void placeLocationVariable(const LocationPlacementVariable &variable,
                           uint64_t offset,
                           const SRAMLocationAllocationProblem &problem,
                           LocationPlacementState &state) {
  for (unsigned regionIndex : variable.regionIndices) {
    const auto &region = problem.regions[regionIndex];
    state.offsets[regionIndex] = offset;
    state.placedRegions.push_back(regionIndex);
    state.highWaterBytes[region.locationIndex] = std::max(
        state.highWaterBytes[region.locationIndex], offset + region.bytes);
  }
}

SmallVector<unsigned>
getLocationVariableOrder(ArrayRef<LocationPlacementVariable> variables,
                         const SRAMLocationAllocationProblem &problem,
                         bool degreeAware) {
  SmallVector<unsigned> order(variables.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(order, [&](unsigned leftIndex, unsigned rightIndex) {
    bool leftFixed = variables[leftIndex].fixedOffset.has_value();
    bool rightFixed = variables[rightIndex].fixedOffset.has_value();
    if (leftFixed != rightFixed) {
      return leftFixed;
    }
    uint64_t leftBytes = getVariableBytes(variables[leftIndex], problem);
    uint64_t rightBytes = getVariableBytes(variables[rightIndex], problem);
    if (leftBytes != rightBytes) {
      return leftBytes > rightBytes;
    }
    if (degreeAware) {
      unsigned leftDegree = getVariableDegree(variables[leftIndex], problem);
      unsigned rightDegree = getVariableDegree(variables[rightIndex], problem);
      if (leftDegree != rightDegree) {
        return leftDegree > rightDegree;
      }
    }
    return leftIndex < rightIndex;
  });
  return order;
}

LocationPlacementState
makeEmptyLocationState(const SRAMLocationAllocationProblem &problem) {
  LocationPlacementState state;
  state.offsets.assign(problem.regions.size(), kUnassignedOffset);
  for (const auto &location : problem.locations) {
    state.highWaterBytes.push_back(location.payloadBaseOffset);
  }
  return state;
}

uint64_t
getLocationReservationBytes(const SRAMLocationAllocationProblem &problem,
                            ArrayRef<uint64_t> highWaterBytes) {
  llvm::DenseSet<unsigned> groupedLocations;
  uint64_t reservationBytes = 0;
  for (const auto &group : problem.equalCapacityGroups) {
    uint64_t capacity = 0;
    for (unsigned locationIndex : group.locationIndices) {
      groupedLocations.insert(locationIndex);
      capacity = std::max(capacity, highWaterBytes[locationIndex]);
    }
    reservationBytes += capacity * group.locationIndices.size();
  }
  for (auto [locationIndex, highWater] : llvm::enumerate(highWaterBytes)) {
    if (!groupedLocations.contains(locationIndex)) {
      reservationBytes += highWater;
    }
  }
  return reservationBytes;
}

} // namespace mlir::tt::ttl::detail
