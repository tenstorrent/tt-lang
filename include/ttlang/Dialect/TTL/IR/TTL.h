// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTL_H
#define TTLANG_DIALECT_TTL_IR_TTL_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

#include "ttlang/Dialect/TTL/IR/TTLOpsDialect.h.inc"

namespace mlir::tt::ttl {

/// Default tile dimensions used for TTL tensors.
inline constexpr int32_t kDefaultTileHeight = 32;
inline constexpr int32_t kDefaultTileWidth = 32;
inline constexpr int32_t kMaxCircularBuffers = 32;

/// Purpose: Enable tagging of all tile-level operations so we can identify
/// them later as tile-level operations without having to check individual
/// types.
template <typename ConcreteType>
class TTLTileOpTrait : public OpTrait::TraitBase<ConcreteType, TTLTileOpTrait> {
};

/// Attribute names.
inline constexpr llvm::StringRef kDstIdxAttrName = "dst_idx";
inline constexpr llvm::StringRef kCBIndexAttrPrefix = "ttl.cb_index.";

/// Trait for tile compute operations (add, mul, exp, etc.).
template <typename ConcreteType>
class TTLTileComputeOpTrait
    : public OpTrait::TraitBase<ConcreteType, TTLTileComputeOpTrait> {};

//===----------------------------------------------------------------------===//
// CB Index Attribute Helpers
//===----------------------------------------------------------------------===//

/// Get the CB index attribute name for a compute input.
inline std::string getCBIndexAttrName(unsigned inputIdx) {
  return (kCBIndexAttrPrefix + std::to_string(inputIdx)).str();
}

/// Set CB index attribute on a compute op for a specific input.
inline void setCBIndexAttr(Operation *compute, unsigned inputIdx,
                           int64_t cbIndex) {
  OpBuilder builder(compute->getContext());
  compute->setAttr(getCBIndexAttrName(inputIdx),
                   builder.getI64IntegerAttr(cbIndex));
}

/// Get CB index attribute from a compute op for a specific input.
/// Returns std::nullopt if the attribute is not present or invalid.
inline std::optional<int64_t> getCBIndexAttr(Operation *compute,
                                             unsigned inputIdx) {
  if (!compute) {
    return std::nullopt;
  }

  if (auto attr = dyn_cast_or_null<IntegerAttr>(
          compute->getAttr(getCBIndexAttrName(inputIdx)))) {
    return attr.getInt();
  }
  return std::nullopt;
}

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_IR_TTL_H
