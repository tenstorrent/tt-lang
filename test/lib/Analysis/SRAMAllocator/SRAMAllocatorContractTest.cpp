// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "CompilerL1Allocator.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <utility>

namespace {

using mlir::tt::ttl::CompilerL1AllocationProblem;
using mlir::tt::ttl::CompilerL1AllocationSolution;
using mlir::tt::ttl::CompilerL1Allocator;
using mlir::tt::ttl::solveCompilerL1Allocation;
using mlir::tt::ttl::SRAMPlacementFailure;
using mlir::tt::ttl::SRAMPlacementFailureKind;

class FixedSolutionAllocator final : public CompilerL1Allocator {
public:
  explicit FixedSolutionAllocator(CompilerL1AllocationSolution solution)
      : solution(std::move(solution)) {}

  llvm::StringRef getName() const override { return "test-fixed"; }

private:
  mlir::FailureOr<CompilerL1AllocationSolution>
  allocate(const CompilerL1AllocationProblem &, std::string &) const override {
    return solution;
  }

  CompilerL1AllocationSolution solution;
};

CompilerL1AllocationProblem makeProblem() {
  CompilerL1AllocationProblem problem;
  problem.regionBytes = {64, 64};
  problem.conflicts = {llvm::BitVector(2), llvm::BitVector(2)};
  problem.conflicts[0].set(1);
  problem.conflicts[1].set(0);
  problem.alignmentBytes = 64;
  problem.payloadBaseOffset = 64;
  problem.budgetBytes = 256;
  return problem;
}

bool rejects(CompilerL1AllocationProblem problem,
             CompilerL1AllocationSolution solution,
             SRAMPlacementFailureKind expectedKind,
             llvm::StringRef expectedReason) {
  FixedSolutionAllocator allocator(std::move(solution));
  SRAMPlacementFailure detail;
  if (mlir::succeeded(solveCompilerL1Allocation(allocator, problem, detail)) ||
      detail.kind != expectedKind || detail.reason != expectedReason) {
    llvm::errs() << "allocator contract rejected wrong outcome: "
                 << detail.reason << "\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  CompilerL1AllocationProblem problem = makeProblem();
  if (!rejects(problem, {{64}, 128}, SRAMPlacementFailureKind::InvalidSolution,
               "allocator returned the wrong number of offsets") ||
      !rejects(problem, {{65, 128}, 192},
               SRAMPlacementFailureKind::InvalidSolution,
               "allocator returned a misaligned payload offset") ||
      !rejects(problem, {{64, 64}, 128},
               SRAMPlacementFailureKind::InvalidSolution,
               "allocator overlapped conflicting payload regions") ||
      !rejects(problem, {{64, 256}, 320},
               SRAMPlacementFailureKind::BudgetExceeded,
               "placement exceeds L1 budget 256 bytes") ||
      !rejects(problem, {{64, 128}, 128},
               SRAMPlacementFailureKind::InvalidSolution,
               "allocator returned an incorrect arena high-water mark") ||
      !rejects(problem, {{64, UINT64_MAX - 63}, 256},
               SRAMPlacementFailureKind::InvalidSolution,
               "placed interval overflowed the address range")) {
    return 1;
  }
  problem.conflicts[1].reset(0);
  if (!rejects(problem, {{64, 128}, 192},
               SRAMPlacementFailureKind::InvalidProblem,
               "conflict matrix is not symmetric")) {
    return 1;
  }
  llvm::outs() << "allocator_contract_cases=7\n";
  return 0;
}
