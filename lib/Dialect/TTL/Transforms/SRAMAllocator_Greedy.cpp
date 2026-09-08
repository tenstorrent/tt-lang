// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocator.h"
#include "SRAMAllocator_Internal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
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

using detail::GreedyGapSelection;
enum class PlacementOrder { Stable, DegreeAware };

static FailureOr<uint64_t> alignOffset(uint64_t offset, uint64_t alignment,
                                       std::string &failureReason) {
  if (offset > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
    failureReason = "placement offset overflowed during alignment";
    return failure();
  }
  return llvm::alignTo(offset, alignment);
}

static FailureOr<SmallVector<BlockingInterval>>
getBlockingIntervals(const SRAMAllocationProblem &problem, unsigned regionIndex,
                     ArrayRef<unsigned> placedRegions,
                     ArrayRef<uint64_t> offsets, std::string &failureReason) {
  SmallVector<BlockingInterval> intervals;
  for (unsigned placedRegionIndex : placedRegions) {
    if (!problem.conflicts.interferes(regionIndex, placedRegionIndex)) {
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
getDecreasingSizeOrder(const SRAMAllocationProblem &problem,
                       PlacementOrder ordering) {
  SmallVector<unsigned> order;
  for (unsigned regionIndex = 0; regionIndex < problem.regionBytes.size();
       ++regionIndex) {
    order.push_back(regionIndex);
  }
  llvm::stable_sort(order, [&](unsigned leftIndex, unsigned rightIndex) {
    if (problem.regionBytes[leftIndex] != problem.regionBytes[rightIndex]) {
      return problem.regionBytes[leftIndex] > problem.regionBytes[rightIndex];
    }
    return ordering == PlacementOrder::DegreeAware &&
           problem.conflicts.degree(leftIndex) >
               problem.conflicts.degree(rightIndex);
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
selectOffset(const SRAMAllocationProblem &problem, unsigned regionIndex,
             ArrayRef<unsigned> placedRegions, ArrayRef<uint64_t> offsets,
             GreedyGapSelection selection, std::string &failureReason) {
  FailureOr<SmallVector<BlockingInterval>> intervals = getBlockingIntervals(
      problem, regionIndex, placedRegions, offsets, failureReason);
  if (failed(intervals)) {
    return failure();
  }
  uint64_t offset = problem.payloadBaseOffset;
  std::optional<std::pair<uint64_t, uint64_t>> bestGap;
  for (const BlockingInterval &interval : *intervals) {
    if (fitsBefore(offset, problem.regionBytes[regionIndex], interval.begin)) {
      if (selection == GreedyGapSelection::FirstFit) {
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

static FailureOr<SRAMAllocationSolution>
allocateDecreasing(const SRAMAllocationProblem &problem,
                   GreedyGapSelection selection, std::string &failureReason,
                   PlacementOrder ordering = PlacementOrder::Stable) {
  SmallVector<uint64_t> offsets(problem.regionBytes.size());
  SmallVector<unsigned> placedRegions;
  for (unsigned regionIndex : getDecreasingSizeOrder(problem, ordering)) {
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
  return SRAMAllocationSolution{std::move(offsets), arenaBytes};
}

class FirstFitDecreasingAllocator final : public SRAMAllocator {
public:
  llvm::StringRef getName() const override {
    return kFirstFitDecreasingSRAMAllocator;
  }

  FailureOr<SRAMAllocationSolution>
  allocateImpl(const SRAMAllocationProblem &problem,
               std::string &failureReason) const override {
    return allocateDecreasing(problem, GreedyGapSelection::FirstFit,
                              failureReason);
  }
};

class BestFitDecreasingAllocator final : public SRAMAllocator {
public:
  llvm::StringRef getName() const override {
    return kBestFitDecreasingSRAMAllocator;
  }

  FailureOr<SRAMAllocationSolution>
  allocateImpl(const SRAMAllocationProblem &problem,
               std::string &failureReason) const override {
    return allocateDecreasing(problem, GreedyGapSelection::BestFit,
                              failureReason);
  }
};

class MultiOrderDecreasingAllocator final : public SRAMAllocator {
public:
  llvm::StringRef getName() const override {
    return kMultiOrderDecreasingSRAMAllocator;
  }

private:
  FailureOr<SRAMAllocationSolution>
  allocateImpl(const SRAMAllocationProblem &problem,
               std::string &failureReason) const override {
    auto stable = allocateDecreasing(problem, GreedyGapSelection::FirstFit,
                                     failureReason);
    std::string degreeFailure;
    auto degreeAware =
        allocateDecreasing(problem, GreedyGapSelection::FirstFit, degreeFailure,
                           PlacementOrder::DegreeAware);
    if (succeeded(degreeAware) &&
        (failed(stable) || degreeAware->arenaBytes < stable->arenaBytes)) {
      return degreeAware;
    }
    return stable;
  }
};

} // namespace

FailureOr<SRAMAllocationSolution>
detail::allocateGreedy(const SRAMAllocationProblem &problem,
                       GreedyGapSelection selection,
                       std::string &failureReason) {
  return allocateDecreasing(problem, selection, failureReason);
}

std::unique_ptr<SRAMAllocator>
detail::createMultiOrderDecreasingSRAMAllocator() {
  return std::make_unique<MultiOrderDecreasingAllocator>();
}

std::unique_ptr<SRAMAllocator> detail::createFirstFitDecreasingSRAMAllocator() {
  return std::make_unique<FirstFitDecreasingAllocator>();
}

std::unique_ptr<SRAMAllocator> detail::createBestFitDecreasingSRAMAllocator() {
  return std::make_unique<BestFitDecreasingAllocator>();
}

} // namespace mlir::tt::ttl
