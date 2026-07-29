// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATION_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdint>

namespace mlir::tt::ttl {

/// Default WH/BH usable L1 size when target information is unavailable.
///
/// This matches `ttl.constants.DEFAULT_L1_CB_BUDGET_BYTES`. Modules with a
/// system descriptor use `ChipDescAttr::getUsableL1Size()` instead.
inline constexpr uint64_t kDefaultL1DFBBudgetBytes =
    static_cast<uint64_t>(1432 * 1024);

/// Largest allocation associated with one hardware DFB index.
struct DFBIndexAllocation {
  uint64_t bytes = 0;
  BindCBOp representative;
};

/// Per-core DFB allocation accounting for a module.
struct DFBAllocationSummary {
  llvm::DenseMap<int64_t, DFBIndexAllocation> allocations;
  uint64_t totalBytes = 0;

  /// Return the total after requiring `minimumBytes` at `dfbIndex`.
  FailureOr<uint64_t>
  getTotalBytesWithMinimumAllocation(int64_t dfbIndex,
                                     uint64_t minimumBytes) const;

  /// Return the total after applying per-index minimum allocations.
  FailureOr<uint64_t> getTotalBytesWithMinimumAllocations(
      const llvm::DenseMap<int64_t, uint64_t> &minimumBytesByIndex) const;
};

/// Return the usable L1 size selected for DFB allocations.
uint64_t getL1DFBBudgetBytes(ModuleOp moduleOp, uint64_t overrideBytes = 0);

/// Return the backing allocation size for `dfbType`.
FailureOr<uint64_t> getDFBAllocationSizeBytes(CircularBufferType dfbType);

/// Summarize the largest allocation at each hardware DFB index.
FailureOr<DFBAllocationSummary> getDFBAllocationSummary(ModuleOp moduleOp);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATION_H
