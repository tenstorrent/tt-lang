// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocator.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace {

using mlir::tt::ttl::InterferenceGraph;
using mlir::tt::ttl::SRAMAllocationProblem;
using mlir::tt::ttl::SRAMAllocationSolution;
using mlir::tt::ttl::SRAMAllocator;
using mlir::tt::ttl::SRAMLocationAllocationProblem;
using mlir::tt::ttl::SRAMPlacementFailure;
using mlir::tt::ttl::SRAMPlacementFailureKind;

class FixedSolutionAllocator final : public SRAMAllocator {
public:
  explicit FixedSolutionAllocator(SRAMAllocationSolution solution)
      : solution(std::move(solution)) {}

  llvm::StringRef getName() const override { return "test-fixed"; }

private:
  mlir::FailureOr<SRAMAllocationSolution>
  allocateImpl(const SRAMAllocationProblem &, std::string &) const override {
    return solution;
  }

  SRAMAllocationSolution solution;
};

SRAMAllocationProblem makeProblem() {
  SRAMAllocationProblem problem;
  problem.regionBytes = {64, 64};
  problem.conflicts = InterferenceGraph(2);
  problem.conflicts.addInterference(0, 1);
  problem.alignmentBytes = 64;
  problem.payloadBaseOffset = 64;
  problem.budgetBytes = 256;
  return problem;
}

bool rejects(SRAMAllocationProblem problem, SRAMAllocationSolution solution,
             SRAMPlacementFailureKind expectedKind,
             llvm::StringRef expectedReason) {
  FixedSolutionAllocator allocator(std::move(solution));
  SRAMPlacementFailure detail;
  if (mlir::succeeded(allocator.allocate(problem, detail)) ||
      detail.kind != expectedKind || detail.reason != expectedReason) {
    llvm::errs() << "allocator contract rejected wrong outcome: "
                 << detail.reason << "\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  SRAMLocationAllocationProblem locationProblem;
  locationProblem.locations.push_back({0, 64});
  locationProblem.regions.push_back({0, 0, 16, std::nullopt});
  locationProblem.conflicts = InterferenceGraph(1);
  locationProblem.alignmentBytes = 16;
  FixedSolutionAllocator scalarOnly({{}, 0});
  SRAMPlacementFailure locationFailure;
  if (mlir::succeeded(
          scalarOnly.allocateLocations(locationProblem, locationFailure)) ||
      locationFailure.kind != SRAMPlacementFailureKind::StrategyFailure ||
      locationFailure.reason !=
          "strategy does not support per-location SRAM allocation") {
    llvm::errs() << "scalar-only allocator did not reject location placement\n";
    return 1;
  }
  SRAMAllocationProblem problem = makeProblem();
  if (!rejects({}, {{}, 0}, SRAMPlacementFailureKind::InvalidProblem,
               "allocation alignment must be a nonzero power of two") ||
      !rejects(problem, {{64}, 128}, SRAMPlacementFailureKind::InvalidSolution,
               "allocator returned the wrong number of offsets") ||
      !rejects(problem, {{65, 128}, 192},
               SRAMPlacementFailureKind::InvalidSolution,
               "allocator returned a misaligned payload offset") ||
      !rejects(problem, {{64, 64}, 128},
               SRAMPlacementFailureKind::InvalidSolution,
               "allocator overlapped conflicting payload regions") ||
      !rejects(problem, {{64, 256}, 320},
               SRAMPlacementFailureKind::BudgetExceeded,
               "placement exceeds SRAM budget 256 bytes") ||
      !rejects(problem, {{64, 128}, 128},
               SRAMPlacementFailureKind::InvalidSolution,
               "allocator returned an incorrect arena high-water mark") ||
      !rejects(problem, {{64, UINT64_MAX - 63}, 256},
               SRAMPlacementFailureKind::InvalidSolution,
               "placed interval overflowed the address range")) {
    return 1;
  }
  problem.conflicts = InterferenceGraph(1);
  if (!rejects(problem, {{64, 128}, 192},
               SRAMPlacementFailureKind::InvalidProblem,
               "conflict graph size does not match the region count")) {
    return 1;
  }
  llvm::outs() << "allocator_contract_cases=8\n";
  return 0;
}
