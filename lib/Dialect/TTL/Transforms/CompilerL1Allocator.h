//===- CompilerL1Allocator.h - Compiler L1 placement API ------*- C++ -*-===//
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_COMPILERL1ALLOCATOR_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_COMPILERL1ALLOCATOR_H

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

inline constexpr llvm::StringLiteral kFirstFitDecreasingL1Allocator =
    "first-fit-decreasing";
inline constexpr llvm::StringLiteral kBestFitDecreasingL1Allocator =
    "best-fit-decreasing";

/// Byte-placement input independent of MLIR and architecture identities.
struct CompilerL1AllocationProblem {
  llvm::SmallVector<uint64_t> regionBytes;
  llvm::SmallVector<llvm::BitVector> conflicts;
  uint64_t alignmentBytes;
  uint64_t payloadBaseOffset;
  uint64_t budgetBytes;
};

/// Byte offsets selected for every input region and their high-water mark.
struct CompilerL1AllocationSolution {
  llvm::SmallVector<uint64_t> offsets;
  uint64_t arenaBytes;
};

/// Selects payload offsets without inspecting or modifying compiler IR.
class CompilerL1Allocator {
public:
  virtual ~CompilerL1Allocator() = default;

  virtual llvm::StringRef getName() const = 0;

private:
  friend FailureOr<CompilerL1AllocationSolution>
  solveCompilerL1Allocation(const CompilerL1Allocator &allocator,
                            const CompilerL1AllocationProblem &problem,
                            std::optional<unsigned> &failureRegionIndex,
                            std::string &failureReason);

  virtual FailureOr<CompilerL1AllocationSolution>
  allocate(const CompilerL1AllocationProblem &problem,
           std::string &failureReason) const = 0;
};

/// Creates a built-in allocator selected by its stable compiler option name.
FailureOr<std::unique_ptr<CompilerL1Allocator>>
createCompilerL1Allocator(llvm::StringRef name, std::string &failureReason);

/// Runs a strategy between common input and output validation.
FailureOr<CompilerL1AllocationSolution>
solveCompilerL1Allocation(const CompilerL1Allocator &allocator,
                          const CompilerL1AllocationProblem &problem,
                          std::optional<unsigned> &failureRegionIndex,
                          std::string &failureReason);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_COMPILERL1ALLOCATOR_H
