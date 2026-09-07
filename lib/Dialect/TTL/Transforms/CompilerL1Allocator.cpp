// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "CompilerL1Allocator.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::tt::ttl {
namespace {

using llvm::ArrayRef;
using llvm::SmallVector;

struct BlockingInterval {
  uint64_t begin;
  uint64_t end;
};

enum class GapSelection { FirstFit, BestFit };

static LogicalResult validateProblem(const CompilerL1AllocationProblem &problem,
                                     std::string &failureReason);

static LogicalResult
validateSolution(const CompilerL1AllocationProblem &problem,
                 const CompilerL1AllocationSolution &solution,
                 std::optional<unsigned> &failureRegionIndex,
                 std::string &failureReason);

static FailureOr<uint64_t> alignOffset(uint64_t offset, uint64_t alignment,
                                       std::string &failureReason) {
  if (offset > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
    failureReason = "placement offset overflowed during alignment";
    return failure();
  }
  return llvm::alignTo(offset, alignment);
}

static FailureOr<SmallVector<BlockingInterval>>
getBlockingIntervals(const CompilerL1AllocationProblem &problem,
                     unsigned regionIndex, ArrayRef<unsigned> placedRegions,
                     ArrayRef<uint64_t> offsets, std::string &failureReason) {
  SmallVector<BlockingInterval> intervals;
  for (unsigned placedRegionIndex : placedRegions) {
    if (!problem.conflicts[regionIndex].test(placedRegionIndex)) {
      continue;
    }
    std::optional<uint64_t> end = llvm::checkedAddUnsigned(
        offsets[placedRegionIndex], problem.regionBytes[placedRegionIndex]);
    if (!end) {
      failureReason = "placed interval overflowed the address range";
      return failure();
    }
    intervals.push_back({offsets[placedRegionIndex], *end});
  }
  llvm::sort(intervals, [](const BlockingInterval &left,
                           const BlockingInterval &right) {
    return std::tie(left.begin, left.end) < std::tie(right.begin, right.end);
  });
  return intervals;
}

static SmallVector<unsigned>
getDecreasingSizeOrder(const CompilerL1AllocationProblem &problem) {
  SmallVector<unsigned> order;
  for (unsigned regionIndex = 0; regionIndex < problem.regionBytes.size();
       ++regionIndex) {
    order.push_back(regionIndex);
  }
  llvm::stable_sort(order, [&](unsigned leftIndex, unsigned rightIndex) {
    return problem.regionBytes[leftIndex] > problem.regionBytes[rightIndex];
  });
  return order;
}

static bool fitsBefore(uint64_t offset, uint64_t regionBytes,
                       uint64_t intervalBegin) {
  return offset <= intervalBegin && regionBytes <= intervalBegin - offset;
}

static FailureOr<uint64_t> advanceBeyond(uint64_t offset,
                                         const BlockingInterval &interval,
                                         uint64_t alignment,
                                         std::string &failureReason) {
  if (offset >= interval.end) {
    return offset;
  }
  return alignOffset(interval.end, alignment, failureReason);
}

static FailureOr<uint64_t>
selectOffset(const CompilerL1AllocationProblem &problem, unsigned regionIndex,
             ArrayRef<unsigned> placedRegions, ArrayRef<uint64_t> offsets,
             GapSelection selection, std::string &failureReason) {
  FailureOr<SmallVector<BlockingInterval>> intervals = getBlockingIntervals(
      problem, regionIndex, placedRegions, offsets, failureReason);
  if (failed(intervals)) {
    return failure();
  }
  uint64_t offset = problem.payloadBaseOffset;
  std::optional<std::pair<uint64_t, uint64_t>> bestGap;
  for (const BlockingInterval &interval : *intervals) {
    if (fitsBefore(offset, problem.regionBytes[regionIndex], interval.begin)) {
      if (selection == GapSelection::FirstFit) {
        return offset;
      }
      uint64_t unusedBytes =
          interval.begin - offset - problem.regionBytes[regionIndex];
      std::pair<uint64_t, uint64_t> gap{unusedBytes, offset};
      if (!bestGap || gap < *bestGap) {
        bestGap = gap;
      }
    }
    FailureOr<uint64_t> nextOffset =
        advanceBeyond(offset, interval, problem.alignmentBytes, failureReason);
    if (failed(nextOffset)) {
      return failure();
    }
    offset = *nextOffset;
  }
  return bestGap ? bestGap->second : offset;
}

static FailureOr<CompilerL1AllocationSolution>
allocateDecreasing(const CompilerL1AllocationProblem &problem,
                   GapSelection selection, std::string &failureReason) {
  SmallVector<uint64_t> offsets(problem.regionBytes.size());
  SmallVector<unsigned> placedRegions;
  for (unsigned regionIndex : getDecreasingSizeOrder(problem)) {
    FailureOr<uint64_t> offset = selectOffset(
        problem, regionIndex, placedRegions, offsets, selection, failureReason);
    if (failed(offset)) {
      return failure();
    }
    offsets[regionIndex] = *offset;
    placedRegions.push_back(regionIndex);
  }
  uint64_t arenaBytes =
      problem.regionBytes.empty() ? uint64_t{0} : problem.payloadBaseOffset;
  for (auto [offset, regionBytes] :
       llvm::zip_equal(offsets, problem.regionBytes)) {
    std::optional<uint64_t> end = llvm::checkedAddUnsigned(offset, regionBytes);
    if (!end) {
      failureReason = "placed interval overflowed the address range";
      return failure();
    }
    arenaBytes = std::max(arenaBytes, *end);
  }
  return CompilerL1AllocationSolution{std::move(offsets), arenaBytes};
}

class FirstFitDecreasingAllocator final : public CompilerL1Allocator {
public:
  llvm::StringRef getName() const override {
    return kFirstFitDecreasingL1Allocator;
  }

  FailureOr<CompilerL1AllocationSolution>
  allocate(const CompilerL1AllocationProblem &problem,
           std::string &failureReason) const override {
    return allocateDecreasing(problem, GapSelection::FirstFit, failureReason);
  }
};

class BestFitDecreasingAllocator final : public CompilerL1Allocator {
public:
  llvm::StringRef getName() const override {
    return kBestFitDecreasingL1Allocator;
  }

  FailureOr<CompilerL1AllocationSolution>
  allocate(const CompilerL1AllocationProblem &problem,
           std::string &failureReason) const override {
    return allocateDecreasing(problem, GapSelection::BestFit, failureReason);
  }
};

} // namespace

FailureOr<std::unique_ptr<CompilerL1Allocator>>
createCompilerL1Allocator(llvm::StringRef name, std::string &failureReason) {
  if (name == kFirstFitDecreasingL1Allocator) {
    std::unique_ptr<CompilerL1Allocator> allocator =
        std::make_unique<FirstFitDecreasingAllocator>();
    return FailureOr<std::unique_ptr<CompilerL1Allocator>>(
        std::move(allocator));
  }
  if (name == kBestFitDecreasingL1Allocator) {
    std::unique_ptr<CompilerL1Allocator> allocator =
        std::make_unique<BestFitDecreasingAllocator>();
    return FailureOr<std::unique_ptr<CompilerL1Allocator>>(
        std::move(allocator));
  }
  failureReason = "unknown compiler-l1 allocation strategy '" + name.str() +
                  "'; expected first-fit-decreasing or best-fit-decreasing";
  return failure();
}

namespace {

LogicalResult validateProblem(const CompilerL1AllocationProblem &problem,
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
    failureReason = "payload base offset exceeds the L1 budget";
    return failure();
  }
  if (problem.conflicts.size() != regionCount) {
    failureReason = "conflict row count does not match the region count";
    return failure();
  }
  for (unsigned regionIndex = 0; regionIndex < regionCount; ++regionIndex) {
    if (problem.regionBytes[regionIndex] == 0 ||
        problem.regionBytes[regionIndex] % problem.alignmentBytes != 0) {
      failureReason = "region size does not satisfy allocation alignment";
      return failure();
    }
    const llvm::BitVector &row = problem.conflicts[regionIndex];
    if (row.size() != regionCount || row.test(regionIndex)) {
      failureReason = "conflict matrix has an invalid row";
      return failure();
    }
  }
  for (unsigned regionIndex = 0; regionIndex < regionCount; ++regionIndex) {
    const llvm::BitVector &row = problem.conflicts[regionIndex];
    for (unsigned otherIndex = regionIndex + 1; otherIndex < regionCount;
         ++otherIndex) {
      if (row.test(otherIndex) !=
          problem.conflicts[otherIndex].test(regionIndex)) {
        failureReason = "conflict matrix is not symmetric";
        return failure();
      }
    }
  }
  return success();
}

LogicalResult validateSolution(const CompilerL1AllocationProblem &problem,
                               const CompilerL1AllocationSolution &solution,
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
      failureReason = "placement exceeds L1 budget " +
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
      if (!problem.conflicts[leftIndex].test(rightIndex)) {
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

FailureOr<CompilerL1AllocationSolution>
solveCompilerL1Allocation(const CompilerL1Allocator &allocator,
                          const CompilerL1AllocationProblem &problem,
                          std::optional<unsigned> &failureRegionIndex,
                          std::string &failureReason) {
  failureRegionIndex = std::nullopt;
  if (failed(validateProblem(problem, failureReason))) {
    return failure();
  }
  FailureOr<CompilerL1AllocationSolution> solution =
      allocator.allocate(problem, failureReason);
  if (failed(solution) ||
      failed(validateSolution(problem, *solution, failureRegionIndex,
                              failureReason))) {
    return failure();
  }
  return solution;
}

} // namespace mlir::tt::ttl
