// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_COMPILERL1ALLOCATION_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_COMPILERL1ALLOCATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir::tt::ttl {

class DFBConcurrentKernelLivenessAnalysis;
class DFBLogicalIdentityAnalysis;
class CompilerL1Allocator;

/// Rejects lifecycle operations unsupported by compiler-managed SRAM.
LogicalResult validateCompilerSRAMLifecycle(ModuleOp module);

/// Plans and materializes compiler-managed L1 offsets. Failure leaves IR
/// unchanged.
LogicalResult
allocateCompilerL1(ModuleOp module,
                   const DFBLogicalIdentityAnalysis &identities,
                   uint64_t budgetOverride, bool reuseStorage,
                   const CompilerL1Allocator &allocator,
                   const DFBConcurrentKernelLivenessAnalysis &liveness);

} // namespace mlir::tt::ttl

#endif
