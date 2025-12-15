// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTL_H
#define TTLANG_DIALECT_TTL_IR_TTL_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

#include "ttlang/Dialect/TTL/IR/TTLOpsDialect.h.inc"

namespace mlir::tt::ttl {

/// Default tile dimensions used for TTL tensors.
inline constexpr int kDefaultTileHeight = 32;
inline constexpr int kDefaultTileWidth = 32;

/// Marker for the tile loops generated during TTL to TTKernel lowering.
inline constexpr llvm::StringLiteral kTileLoopMarker = "ttkernel.tile_loop";

/// Check if an operation is marked as a tile loop from TTL lowering.
inline bool isTileLoop(Operation *op) { return op->hasAttr(kTileLoopMarker); }

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_IR_TTL_H
