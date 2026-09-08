// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocator.h"
#include "SRAMAllocator_Internal.h"

#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

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
  failureReason = "unknown compiler-l1 allocation strategy '" + name.str() +
                  "'; expected multi-order-decreasing, first-fit-decreasing, "
                  "best-fit-decreasing, or "
                  "exact";
  return failure();
}

} // namespace mlir::tt::ttl
