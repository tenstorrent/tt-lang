// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATION_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir::tt::ttl {

class DFBConcurrentKernelLivenessAnalysis;
class DFBLogicalIdentityAnalysis;
struct DFBAssumedAllocationGroup;
struct DFBStaticConfigurationConflict;

/// Plans and materializes compiler-managed SRAM offsets. Failure leaves IR
/// unchanged.
LogicalResult allocateSRAM(
    ModuleOp module, const DFBLogicalIdentityAnalysis &identities,
    uint64_t budgetOverride, bool reuseStorage,
    llvm::StringRef allocationStrategy, uint64_t exactSearchLimit,
    bool reportAllocation, const DFBConcurrentKernelLivenessAnalysis &liveness,
    llvm::ArrayRef<DFBStaticConfigurationConflict> staticConfigurationConflicts,
    bool unsafeAssumeAllocationGroups,
    llvm::SmallVectorImpl<DFBAssumedAllocationGroup> &assumedAllocationGroups);

} // namespace mlir::tt::ttl

#endif
