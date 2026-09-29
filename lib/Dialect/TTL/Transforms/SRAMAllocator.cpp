// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocator.h"
#include "SRAMAllocator_Internal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>

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
                               SRAMPlacementFailure &failureDetail) {
  if (solution.offsets.size() != problem.regionBytes.size()) {
    failureDetail.reason = "allocator returned the wrong number of offsets";
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
      failureDetail.regionIndex = regionIndex;
      failureDetail.reason = "allocator returned a misaligned payload offset";
      return failure();
    }
    std::optional<uint64_t> end =
        llvm::checkedAddUnsigned(offset, problem.regionBytes[regionIndex]);
    if (!end) {
      failureDetail.regionIndex = regionIndex;
      failureDetail.reason = "placed interval overflowed the address range";
      return failure();
    }
    if (*end > problem.budgetBytes) {
      failureDetail.kind = SRAMPlacementFailureKind::BudgetExceeded;
      failureDetail.regionIndex = regionIndex;
      failureDetail.reason = "placement exceeds SRAM budget " +
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
        failureDetail.regionIndex = rightIndex;
        failureDetail.reason =
            "allocator overlapped conflicting payload regions";
        return failure();
      }
    }
  }
  if (solution.arenaBytes != expectedArenaBytes) {
    failureDetail.reason =
        "allocator returned an incorrect arena high-water mark";
    return failure();
  }
  return success();
}

} // namespace

FailureOr<SRAMAllocationSolution>
SRAMAllocator::allocate(const SRAMAllocationProblem &problem,
                        SRAMPlacementFailure &failureDetail) const {
  failureDetail = {};
  if (failed(validateProblem(problem, failureDetail.reason))) {
    return failure();
  }
  failureDetail.kind = SRAMPlacementFailureKind::StrategyFailure;
  FailureOr<SRAMAllocationSolution> solution =
      allocateImpl(problem, failureDetail.reason);
  if (failed(solution)) {
    assert(!failureDetail.reason.empty() &&
           "failed SRAM strategy must explain its failure");
    return failure();
  }
  failureDetail.kind = SRAMPlacementFailureKind::InvalidSolution;
  if (failed(validateSolution(problem, *solution, failureDetail))) {
    return failure();
  }
  return solution;
}

FailureOr<SmallVector<SRAMAllocationDomainSolution>>
SRAMAllocator::allocateDomains(
    llvm::ArrayRef<SRAMAllocationDomainProblem> domains,
    SRAMAllocationDomainFailure &failureDetail) const {
  for (auto [domainIndex, domain] : llvm::enumerate(domains)) {
    failureDetail = {static_cast<unsigned>(domainIndex), std::nullopt,
                     SRAMPlacementFailureKind::InvalidProblem, ""};
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
    failureDetail = {static_cast<unsigned>(domainIndex), std::nullopt,
                     SRAMPlacementFailureKind::InvalidProblem, ""};
    SRAMPlacementFailure placementFailure;
    auto solution = allocate(domain.allocation, placementFailure);
    if (failed(solution)) {
      failureDetail.kind = placementFailure.kind;
      failureDetail.reason = std::move(placementFailure.reason);
      if (placementFailure.regionIndex) {
        failureDetail.storageIndex =
            domain.storageIndices[*placementFailure.regionIndex];
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
  if (name == kMinimumArenaSRAMAllocator) {
    if (options.minimumArenaSearchLimit == 0) {
      failureReason = "minimum-arena search limit must be positive";
      return failure();
    }
    return detail::createMinimumArenaSRAMAllocator(
        options.minimumArenaSearchLimit);
  }
  failureReason = "unknown compiler-sram allocation strategy '" + name.str() +
                  "'; expected multi-order-decreasing, first-fit-decreasing, "
                  "best-fit-decreasing, or minimum-arena";
  return failure();
}

} // namespace mlir::tt::ttl
