//===- SRAMAllocator.h - SRAM placement API ------*- C++ -*-===//
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATOR_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATOR_H

#include "ttlang/Dialect/TTL/Transforms/InterferenceGraphColoring.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

inline constexpr llvm::StringLiteral kMultiOrderDecreasingSRAMAllocator =
    "multi-order-decreasing";
inline constexpr llvm::StringLiteral kFirstFitDecreasingSRAMAllocator =
    "first-fit-decreasing";
inline constexpr llvm::StringLiteral kBestFitDecreasingSRAMAllocator =
    "best-fit-decreasing";
inline constexpr llvm::StringLiteral kExactSRAMAllocator = "exact";

/// Strategy-specific limits supplied independently of the allocation problem.
struct SRAMAllocatorOptions {
  /// Maximum generated candidate offsets and visited partial placements.
  uint64_t exactSearchLimit;
};

/// Byte-placement input independent of MLIR and architecture identities.
struct SRAMAllocationProblem {
  llvm::SmallVector<uint64_t> regionBytes;
  InterferenceGraph conflicts{0};
  uint64_t alignmentBytes;
  uint64_t payloadBaseOffset;
  uint64_t budgetBytes;
};

/// Byte offsets selected for every input region and their high-water mark.
struct SRAMAllocationSolution {
  llvm::SmallVector<uint64_t> offsets;
  uint64_t arenaBytes;
};

/// Selects payload offsets without inspecting or modifying compiler IR.
class SRAMAllocator {
public:
  virtual ~SRAMAllocator() = default;

  virtual llvm::StringRef getName() const = 0;

  /// Returns a validated placement without modifying the problem.
  FailureOr<SRAMAllocationSolution>
  allocate(const SRAMAllocationProblem &problem,
           std::optional<unsigned> &failureRegionIndex,
           std::string &failureReason) const;

private:
  virtual FailureOr<SRAMAllocationSolution>
  allocateImpl(const SRAMAllocationProblem &problem,
               std::string &failureReason) const = 0;
};

/// Creates a built-in allocator selected by its stable compiler option name.
FailureOr<std::unique_ptr<SRAMAllocator>>
createSRAMAllocator(llvm::StringRef name, const SRAMAllocatorOptions &options,
                    std::string &failureReason);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATOR_H
