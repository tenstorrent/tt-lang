// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATIONLIMITS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATIONLIMITS_H

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

class DFBLogicalIdentityAnalysis;

/// Returns the per-node L1 bytes occupied by one physical DFB descriptor.
/// On failure, `failureReason` describes the invalid allocation type.
FailureOr<uint64_t> getDFBAllocationSizeBytes(CircularBufferType type,
                                              std::string &failureReason);

/// Rounds one runtime allocation to the target's maximum L1 quantum.
FailureOr<uint64_t> getL1AllocationSizeBytes(ModuleOp module,
                                             uint64_t payloadBytes);

/// Returns the target-aligned L1 allocation for one physical DFB descriptor.
FailureOr<uint64_t> getDFBL1AllocationSizeBytes(ModuleOp module,
                                                CircularBufferType type,
                                                std::string &failureReason);

/// Returns the per-node L1 bytes reserved for all unique reconfiguration
/// boundaries, or failure when the total is not representable.
FailureOr<uint64_t> getDFBReconfigurationStateBytes(ModuleOp module);

/// Verifies that the selected target implements DFB reconfiguration.
LogicalResult validateDFBReconfigurationTarget(ModuleOp module);

/// Per-node L1 footprint aggregated by allocation identity.
class DFBAllocationFootprint {
public:
  /// Adds one assignment and returns true when it increases its identity size.
  /// On failure, `failureReason` describes the invalid allocation type.
  FailureOr<bool> add(ModuleOp module, int64_t physicalIndex,
                      CircularBufferType type, std::string &failureReason);

  bool empty() const { return maxBytesByIndex.empty(); }
  /// Returns the total allocation size, or failure when the sum overflows.
  FailureOr<uint64_t> getTotalBytes() const;
  /// Returns the total after applying per-index minimum allocation sizes.
  FailureOr<uint64_t> getTotalBytesWithMinimumAllocations(
      const llvm::DenseMap<int64_t, uint64_t> &minimumBytesByIndex) const;
  uint64_t getBytes(int64_t physicalIndex) const;
  llvm::SmallVector<int64_t> getSortedPhysicalIndices() const;

private:
  llvm::DenseMap<int64_t, uint64_t> maxBytesByIndex;
};

/// Returns the per-node DFB footprint of all declarations in `module`.
FailureOr<DFBAllocationFootprint> getDFBAllocationFootprint(ModuleOp module);

/// Returns a conservative footprint that assigns each logical DFB separate
/// storage. Tensor-backed declarations do not allocate additional L1.
FailureOr<DFBAllocationFootprint>
getLogicalDFBAllocationFootprint(ModuleOp module,
                                 const DFBLogicalIdentityAnalysis &identities);

/// Returns the per-core L1 allocation reserved for GlobalSemaphore objects.
FailureOr<uint64_t> getGlobalSemaphoreL1Bytes(ModuleOp module,
                                              int64_t semaphoreCount);

/// Validates the exact finalized DFB, PipeNet, and reconfiguration allocation.
LogicalResult validateCombinedDFBResourceL1Bytes(
    ModuleOp module, const DFBAllocationFootprint &allocationFootprint,
    uint64_t pipeScratchBytes, int64_t globalSemaphoreCount,
    std::optional<uint64_t> overrideBytes = std::nullopt);

/// Returns the target's usable per-node L1 bytes or the supported fallback.
uint64_t
getUsableDFBL1Bytes(ModuleOp module,
                    std::optional<uint64_t> overrideBytes = std::nullopt);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATIONLIMITS_H
