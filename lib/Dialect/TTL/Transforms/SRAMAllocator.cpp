// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/SRAMAllocator.h"
#include "SRAMAllocator_Internal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>

namespace mlir::tt::ttl {
namespace {

using llvm::SmallVector;

LogicalResult validateProblem(const SRAMAllocationProblem &problem,
                              std::string &failureReason) {
  size_t regionCount = problem.regionBytes.size();
  if (!llvm::isPowerOf2_64(problem.alignmentBytes)) {
    failureReason = "allocation alignment must be a nonzero power of two";
    return failure();
  }
  if (problem.payloadBaseOffset % problem.alignmentBytes != 0) {
    failureReason = "payload base offset does not satisfy allocation alignment";
    return failure();
  }
  if (problem.payloadBaseOffset > problem.budgetBytes) {
    failureReason = "payload base offset exceeds the SRAM budget";
    return failure();
  }
  if (problem.conflicts.size() != regionCount) {
    failureReason = "conflict graph size does not match the region count";
    return failure();
  }
  for (unsigned regionIndex = 0; regionIndex < regionCount; ++regionIndex) {
    if (problem.regionBytes[regionIndex] == 0 ||
        problem.regionBytes[regionIndex] % problem.alignmentBytes != 0) {
      failureReason = "region size does not satisfy allocation alignment";
      return failure();
    }
  }
  return success();
}

LogicalResult validateSolution(const SRAMAllocationProblem &problem,
                               const SRAMAllocationSolution &solution,
                               std::optional<unsigned> &failureRegionIndex,
                               std::string &failureReason) {
  if (solution.offsets.size() != problem.regionBytes.size()) {
    failureReason = "allocator returned the wrong number of offsets";
    return failure();
  }
  uint64_t expectedArenaBytes =
      problem.regionBytes.empty() ? uint64_t{0} : problem.payloadBaseOffset;
  SmallVector<uint64_t> ends(problem.regionBytes.size());
  for (unsigned regionIndex = 0; regionIndex < problem.regionBytes.size();
       ++regionIndex) {
    uint64_t offset = solution.offsets[regionIndex];
    if (offset < problem.payloadBaseOffset ||
        offset % problem.alignmentBytes != 0) {
      failureRegionIndex = regionIndex;
      failureReason = "allocator returned a misaligned payload offset";
      return failure();
    }
    std::optional<uint64_t> end =
        llvm::checkedAddUnsigned(offset, problem.regionBytes[regionIndex]);
    if (!end) {
      failureRegionIndex = regionIndex;
      failureReason = "placed interval overflowed the address range";
      return failure();
    }
    if (*end > problem.budgetBytes) {
      failureRegionIndex = regionIndex;
      failureReason = "placement exceeds SRAM budget " +
                      std::to_string(problem.budgetBytes) + " bytes";
      return failure();
    }
    ends[regionIndex] = *end;
    expectedArenaBytes = std::max(expectedArenaBytes, *end);
  }
  for (unsigned leftIndex = 0; leftIndex < problem.regionBytes.size();
       ++leftIndex) {
    for (unsigned rightIndex = leftIndex + 1;
         rightIndex < problem.regionBytes.size(); ++rightIndex) {
      if (!problem.conflicts.interferes(leftIndex, rightIndex)) {
        continue;
      }
      if (ends[leftIndex] > solution.offsets[rightIndex] &&
          ends[rightIndex] > solution.offsets[leftIndex]) {
        failureRegionIndex = rightIndex;
        failureReason = "allocator overlapped conflicting payload regions";
        return failure();
      }
    }
  }
  if (solution.arenaBytes != expectedArenaBytes) {
    failureReason = "allocator returned an incorrect arena high-water mark";
    return failure();
  }
  return success();
}

LogicalResult
validateLocationProblem(const SRAMLocationAllocationProblem &problem,
                        std::optional<unsigned> &failureRegionIndex,
                        std::string &failureReason) {
  if (!llvm::isPowerOf2_64(problem.alignmentBytes)) {
    failureReason = "allocation alignment must be a nonzero power of two";
    return failure();
  }
  uint64_t combinedBudgetBytes = 0;
  for (const auto &location : problem.locations) {
    if (location.payloadBaseOffset % problem.alignmentBytes != 0) {
      failureReason =
          "location payload base does not satisfy allocation alignment";
      return failure();
    }
    if (location.payloadBaseOffset > location.budgetBytes) {
      failureReason = "location payload base exceeds its SRAM budget";
      return failure();
    }
    std::optional<uint64_t> nextCombinedBudget =
        llvm::checkedAddUnsigned(combinedBudgetBytes, location.budgetBytes);
    if (!nextCombinedBudget) {
      failureReason = "combined SRAM location budgets overflow address space";
      return failure();
    }
    combinedBudgetBytes = *nextCombinedBudget;
  }
  if (problem.conflicts.size() != problem.regions.size()) {
    failureReason = "conflict graph size does not match the region count";
    return failure();
  }
  llvm::DenseSet<std::pair<unsigned, unsigned>> ownerLocations;
  for (auto [regionIndex, region] : llvm::enumerate(problem.regions)) {
    if (region.locationIndex >= problem.locations.size()) {
      failureRegionIndex = regionIndex;
      failureReason = "region references an unknown SRAM location";
      return failure();
    }
    if (region.bytes == 0 || region.bytes % problem.alignmentBytes != 0) {
      failureRegionIndex = regionIndex;
      failureReason = "region size does not satisfy allocation alignment";
      return failure();
    }
    if (!ownerLocations.insert({region.ownerIndex, region.locationIndex})
             .second) {
      failureRegionIndex = regionIndex;
      failureReason = "storage owner has duplicate regions at one location";
      return failure();
    }
    if (!region.fixedOffset) {
      continue;
    }
    const auto &location = problem.locations[region.locationIndex];
    std::optional<uint64_t> end =
        llvm::checkedAddUnsigned(*region.fixedOffset, region.bytes);
    if (*region.fixedOffset < location.payloadBaseOffset ||
        *region.fixedOffset % problem.alignmentBytes != 0 || !end ||
        *end > location.budgetBytes) {
      failureRegionIndex = regionIndex;
      failureReason = "fixed region does not fit its SRAM location";
      return failure();
    }
  }
  for (unsigned leftIndex = 0; leftIndex < problem.regions.size();
       ++leftIndex) {
    for (unsigned rightIndex = leftIndex + 1;
         rightIndex < problem.regions.size(); ++rightIndex) {
      if (problem.conflicts.interferes(leftIndex, rightIndex) &&
          problem.regions[leftIndex].locationIndex !=
              problem.regions[rightIndex].locationIndex) {
        failureRegionIndex = rightIndex;
        failureReason = "conflicting regions must occupy the same location";
        return failure();
      }
    }
  }
  llvm::DenseSet<unsigned> groupedRegions;
  for (const auto &group : problem.equalOffsetGroups) {
    if (group.regionIndices.size() < 2) {
      failureReason = "equal-offset group must contain at least two regions";
      return failure();
    }
    llvm::DenseSet<unsigned> locations;
    std::optional<uint64_t> fixedOffset;
    for (unsigned regionIndex : group.regionIndices) {
      if (regionIndex >= problem.regions.size()) {
        failureReason = "equal-offset group references an unknown region";
        return failure();
      }
      if (!groupedRegions.insert(regionIndex).second) {
        failureRegionIndex = regionIndex;
        failureReason = "region belongs to multiple equal-offset groups";
        return failure();
      }
      const auto &region = problem.regions[regionIndex];
      if (!locations.insert(region.locationIndex).second) {
        failureRegionIndex = regionIndex;
        failureReason =
            "equal-offset group contains multiple regions at one location";
        return failure();
      }
      if (region.fixedOffset && fixedOffset &&
          region.fixedOffset != fixedOffset) {
        failureRegionIndex = regionIndex;
        failureReason = "equal-offset group has incompatible fixed offsets";
        return failure();
      }
      if (region.fixedOffset) {
        fixedOffset = region.fixedOffset;
      }
    }
  }
  llvm::DenseSet<unsigned> groupedCapacityLocations;
  for (const auto &group : problem.equalCapacityGroups) {
    if (group.locationIndices.size() < 2) {
      failureReason =
          "equal-capacity group must contain at least two locations";
      return failure();
    }
    uint64_t minimumBudget = std::numeric_limits<uint64_t>::max();
    uint64_t maximumPayloadBase = 0;
    for (unsigned locationIndex : group.locationIndices) {
      if (locationIndex >= problem.locations.size()) {
        failureReason =
            "equal-capacity group references an unknown SRAM location";
        return failure();
      }
      if (!groupedCapacityLocations.insert(locationIndex).second) {
        failureReason =
            "SRAM location belongs to multiple equal-capacity groups";
        return failure();
      }
      const auto &location = problem.locations[locationIndex];
      minimumBudget = std::min(minimumBudget, location.budgetBytes);
      maximumPayloadBase =
          std::max(maximumPayloadBase, location.payloadBaseOffset);
    }
    if (maximumPayloadBase > minimumBudget) {
      failureReason =
          "equal-capacity group payload base exceeds a member SRAM budget";
      return failure();
    }
  }
  return success();
}

LogicalResult
validateLocationSolution(const SRAMLocationAllocationProblem &problem,
                         const SRAMLocationAllocationSolution &solution,
                         std::optional<unsigned> &failureRegionIndex,
                         std::string &failureReason) {
  if (solution.offsets.size() != problem.regions.size() ||
      solution.highWaterBytes.size() != problem.locations.size()) {
    failureReason = "allocator returned an incorrect location result size";
    return failure();
  }
  SmallVector<uint64_t> ends(problem.regions.size());
  SmallVector<uint64_t> expectedHighWater;
  expectedHighWater.reserve(problem.locations.size());
  for (const auto &location : problem.locations) {
    expectedHighWater.push_back(location.payloadBaseOffset);
  }
  for (auto [regionIndex, region] : llvm::enumerate(problem.regions)) {
    uint64_t offset = solution.offsets[regionIndex];
    const auto &location = problem.locations[region.locationIndex];
    std::optional<uint64_t> end =
        llvm::checkedAddUnsigned(offset, region.bytes);
    if (offset < location.payloadBaseOffset ||
        offset % problem.alignmentBytes != 0 || !end ||
        *end > location.budgetBytes ||
        (region.fixedOffset && offset != *region.fixedOffset)) {
      failureRegionIndex = regionIndex;
      failureReason = "allocator returned an invalid per-location offset";
      return failure();
    }
    ends[regionIndex] = *end;
    expectedHighWater[region.locationIndex] =
        std::max(expectedHighWater[region.locationIndex], *end);
  }
  for (const auto &group : problem.equalOffsetGroups) {
    uint64_t offset = solution.offsets[group.regionIndices.front()];
    for (unsigned regionIndex : llvm::drop_begin(group.regionIndices)) {
      if (solution.offsets[regionIndex] != offset) {
        failureRegionIndex = regionIndex;
        failureReason = "allocator violated an equal-offset constraint";
        return failure();
      }
    }
  }
  for (unsigned leftIndex = 0; leftIndex < problem.regions.size();
       ++leftIndex) {
    for (unsigned rightIndex = leftIndex + 1;
         rightIndex < problem.regions.size(); ++rightIndex) {
      if (!problem.conflicts.interferes(leftIndex, rightIndex)) {
        continue;
      }
      if (ends[leftIndex] > solution.offsets[rightIndex] &&
          ends[rightIndex] > solution.offsets[leftIndex]) {
        failureRegionIndex = rightIndex;
        failureReason = "allocator overlapped conflicting location regions";
        return failure();
      }
    }
  }
  if (solution.highWaterBytes != expectedHighWater) {
    failureReason = "allocator returned incorrect location high-water marks";
    return failure();
  }
  for (const auto &group : problem.equalCapacityGroups) {
    uint64_t capacity = 0;
    for (unsigned locationIndex : group.locationIndices) {
      capacity = std::max(capacity, solution.highWaterBytes[locationIndex]);
    }
    for (unsigned locationIndex : group.locationIndices) {
      if (capacity > problem.locations[locationIndex].budgetBytes) {
        failureReason = "allocator returned an equal capacity that exceeds a "
                        "location budget";
        return failure();
      }
    }
  }
  return success();
}

} // namespace

FailureOr<SRAMAllocationSolution>
SRAMAllocator::allocate(const SRAMAllocationProblem &problem,
                        std::optional<unsigned> &failureRegionIndex,
                        std::string &failureReason) const {
  failureRegionIndex = std::nullopt;
  if (failed(validateProblem(problem, failureReason))) {
    return failure();
  }
  FailureOr<SRAMAllocationSolution> solution =
      allocateImpl(problem, failureReason);
  if (failed(solution) ||
      failed(validateSolution(problem, *solution, failureRegionIndex,
                              failureReason))) {
    return failure();
  }
  return solution;
}

FailureOr<SmallVector<SRAMAllocationDomainSolution>>
SRAMAllocator::allocateDomains(
    llvm::ArrayRef<SRAMAllocationDomainProblem> domains,
    SRAMAllocationDomainFailure &failureDetail) const {
  for (auto [domainIndex, domain] : llvm::enumerate(domains)) {
    failureDetail = {static_cast<unsigned>(domainIndex), std::nullopt, ""};
    if (domain.storageIndices.size() != domain.allocation.regionBytes.size()) {
      failureDetail.reason =
          "domain storage count does not match the region count";
      return failure();
    }
    llvm::DenseSet<unsigned> owners;
    for (unsigned storageIndex : domain.storageIndices) {
      if (!owners.insert(storageIndex).second) {
        failureDetail.storageIndex = storageIndex;
        failureDetail.reason = "domain contains a duplicate storage owner";
        return failure();
      }
    }
    if (failed(validateProblem(domain.allocation, failureDetail.reason))) {
      return failure();
    }
  }

  SmallVector<SRAMAllocationDomainSolution> result;
  for (auto [domainIndex, domain] : llvm::enumerate(domains)) {
    failureDetail = {static_cast<unsigned>(domainIndex), std::nullopt, ""};
    std::optional<unsigned> failedRegion;
    auto solution = allocateImpl(domain.allocation, failureDetail.reason);
    if (failed(solution) ||
        failed(validateSolution(domain.allocation, *solution, failedRegion,
                                failureDetail.reason))) {
      if (failedRegion) {
        failureDetail.storageIndex = domain.storageIndices[*failedRegion];
      }
      return failure();
    }
    SRAMAllocationDomainSolution placement;
    placement.arenaBytes =
        std::max(domain.allocation.payloadBaseOffset, solution->arenaBytes);
    for (auto [regionIndex, storageIndex] :
         llvm::enumerate(domain.storageIndices)) {
      placement.placements.push_back(
          {storageIndex, solution->offsets[regionIndex]});
    }
    result.push_back(std::move(placement));
  }
  return result;
}

FailureOr<SRAMLocationAllocationSolution>
SRAMAllocator::allocateLocations(const SRAMLocationAllocationProblem &problem,
                                 std::optional<unsigned> &failureRegionIndex,
                                 std::string &failureReason) const {
  failureRegionIndex = std::nullopt;
  if (failed(validateLocationProblem(problem, failureRegionIndex,
                                     failureReason))) {
    return failure();
  }
  auto solution = allocateLocationsImpl(problem, failureReason);
  if (failed(solution) ||
      failed(validateLocationSolution(problem, *solution, failureRegionIndex,
                                      failureReason))) {
    return failure();
  }
  return solution;
}

FailureOr<std::unique_ptr<SRAMAllocator>>
createSRAMAllocator(llvm::StringRef name, const SRAMAllocatorOptions &options,
                    std::string &failureReason) {
  if (name == kMultiOrderDecreasingSRAMAllocator) {
    return detail::createMultiOrderDecreasingSRAMAllocator();
  }
  if (name == kFirstFitDecreasingSRAMAllocator) {
    return detail::createFirstFitDecreasingSRAMAllocator();
  }
  if (name == kBestFitDecreasingSRAMAllocator) {
    return detail::createBestFitDecreasingSRAMAllocator();
  }
  if (name == kExactSRAMAllocator) {
    return detail::createExactSRAMAllocator(options.exactSearchLimit);
  }
  failureReason = "unknown SRAM allocation strategy '" + name.str() +
                  "'; expected multi-order-decreasing, first-fit-decreasing, "
                  "best-fit-decreasing, or "
                  "exact";
  return failure();
}

} // namespace mlir::tt::ttl
