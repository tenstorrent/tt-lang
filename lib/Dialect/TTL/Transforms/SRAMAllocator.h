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
  /// Maximum generated candidate offsets and partial placements per domain.
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

/// One independently placed layout. Region indices are local to allocation;
/// storageIndices maps them to caller-owned storage identities.
struct SRAMAllocationDomainProblem {
  SRAMAllocationProblem allocation;
  llvm::SmallVector<unsigned> storageIndices;
};

/// One storage owner's payload offset within its domain's arena.
struct SRAMStoragePlacement {
  unsigned storageIndex;
  uint64_t offset;
};

/// Domain order matches the request; placement order matches storageIndices.
struct SRAMAllocationDomainSolution {
  llvm::SmallVector<SRAMStoragePlacement> placements;
  uint64_t arenaBytes;
};

/// Valid only on failure: the domain and, when available, its storage owner.
struct SRAMAllocationDomainFailure {
  unsigned domainIndex;
  std::optional<unsigned> storageIndex;
  std::string reason;
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

  /// Allocates independently addressable domains using this strategy. The
  /// caller proves that domain bindings do not overlap and supplies each
  /// domain's full conflict relation. All inputs are validated before
  /// placement; failure returns no partial solution. Control-only domains
  /// retain their prefix.
  FailureOr<llvm::SmallVector<SRAMAllocationDomainSolution>>
  allocateDomains(llvm::ArrayRef<SRAMAllocationDomainProblem> domains,
                  SRAMAllocationDomainFailure &failureDetail) const;

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
