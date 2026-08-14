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

/// Returns the per-node L1 bytes occupied by one physical DFB descriptor.
/// On failure, `failureReason` describes the invalid allocation type.
FailureOr<uint64_t> getDFBAllocationSizeBytes(CircularBufferType type,
                                              std::string &failureReason);

/// Per-node L1 footprint aggregated by unique physical DFB index.
class DFBAllocationFootprint {
public:
  /// Adds one assignment and returns true when it increases the index size.
  /// On failure, `failureReason` describes the invalid allocation type.
  FailureOr<bool> add(int64_t physicalIndex, CircularBufferType type,
                      std::string &failureReason);

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

/// Returns the target's usable per-node L1 bytes or the supported fallback.
uint64_t
getUsableDFBL1Bytes(ModuleOp module,
                    std::optional<uint64_t> overrideBytes = std::nullopt);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATIONLIMITS_H
