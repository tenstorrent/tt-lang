// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTL_H
#define TTLANG_DIALECT_TTL_IR_TTL_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

#include "ttlang/Dialect/TTL/IR/TTLOpsDialect.h.inc"

namespace mlir::tt::ttl {

/// Default tile dimensions used for TTL tensors.
inline constexpr int32_t kDefaultTileHeight = 32;
inline constexpr int32_t kDefaultTileWidth = 32;
inline constexpr int32_t kMaxCircularBuffers = 32;

/// Attribute names.
inline constexpr llvm::StringRef kDstIdxAttrName = "dst_idx";
inline constexpr llvm::StringRef kCBIndexAttrPrefix = "ttl.cb_index.";

/// Trait for tile compute operations (add, mul, exp, etc.).
template <typename ConcreteType>
class TTLTileComputeOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLTileComputeOpTrait> {};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_IR_TTL_H
