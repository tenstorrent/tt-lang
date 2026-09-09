// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATIONREPORT_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATIONREPORT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace mlir::tt::ttl {
struct SRAMAllocationPlan;
class DFBConcurrentKernelLivenessAnalysis;
class DFBPhysicalConflictModel;

/// Emits one versioned JSON record from a validated placement and its immutable
/// analysis evidence. Byte counts describe the planned per-core arena.
void printSRAMAllocationReport(
    llvm::raw_ostream &output, const SRAMAllocationPlan &plan,
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    const DFBPhysicalConflictModel &conflicts, llvm::StringRef strategy,
    bool reuseEnabled, uint64_t alignmentBytes, uint64_t controlBytes,
    uint64_t budgetBytes);
} // namespace mlir::tt::ttl
#endif
