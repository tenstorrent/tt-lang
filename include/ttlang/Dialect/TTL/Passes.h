// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_PASSES_H
#define TTLANG_DIALECT_TTL_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class RewritePatternSet;
} // namespace mlir

namespace mlir::tt::ttl {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "ttlang/Dialect/TTL/Passes.h.inc"

/// Populate patterns for lowering TTL elementwise tensor ops to ttl.compute.
void populateTTLToComputePatterns(RewritePatternSet &patterns);

/// Populate patterns for lowering ttl.tile_* ops to TTKernel (tile-only pass).
void populateTTLTileOpsToTTKernelPatterns(RewritePatternSet &patterns);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_PASSES_H
