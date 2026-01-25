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

/// Purpose: Enable tagging of all tile-level operations so we can identify them
/// later as tile-level operations without having to check individual types.
template <typename ConcreteType>
class TTLTileOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLTileOpTrait> {};

/// Attribute names.
inline constexpr llvm::StringRef kDstIdxAttrName = "dst_idx";
inline constexpr llvm::StringRef kCBIndexAttrPrefix = "ttl.cb_index.";

/// Loop marker attribute names (used by loop lowering and sync insertion).
/// The tile_loop attribute marks the innermost loop from ComputeOp lowering.
/// The tile_loop.outer attribute marks the outermost loop of a compute nest.
/// The CB index attributes store which CBs are used for init_sfpu and stores.
inline constexpr llvm::StringRef kTileLoopAttrName = "ttl.tile_loop";
/// TODO(ttl): Remove kTileLoopOuterAttrName once loop flattening is
/// implemented. With flattened loops, there will only be one loop per compute,
/// making the outer marker unnecessary.
inline constexpr llvm::StringRef kTileLoopOuterAttrName = "ttl.tile_loop.outer";
inline constexpr llvm::StringRef kTileLoopAttrPrefix = "ttl.tile_loop.";
/// Input CB indices (ArrayAttr of I64IntegerAttr).
inline constexpr llvm::StringRef kTileLoopInputCBsAttrName =
    "ttl.tile_loop.input_cbs";
/// Output CB indices (ArrayAttr of I64IntegerAttr).
inline constexpr llvm::StringRef kTileLoopOutputCBsAttrName =
    "ttl.tile_loop.output_cbs";

/// Trait for tile compute operations (add, mul, exp, etc.).
template <typename ConcreteType>
class TTLTileComputeOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLTileComputeOpTrait> {};

/// Trait for unary elementwise tensor operations (exp, sqrt, etc.).
template <typename ConcreteType>
class TTLUnaryElementwiseOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      TTLUnaryElementwiseOpTrait> {};

/// Trait for binary elementwise tensor operations (add, mul, etc.).
template <typename ConcreteType>
class TTLBinaryElementwiseOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      TTLBinaryElementwiseOpTrait> {};

/// Trait for tile-level unary operations (execute in-place on DST).
template <typename ConcreteType>
class TTLTileUnaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLTileUnaryOpTrait> {};

/// Trait for tile-level binary operations (write to fresh DST slot).
template <typename ConcreteType>
class TTLTileBinaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLTileBinaryOpTrait> {};

//===----------------------------------------------------------------------===//
// CB Index Attribute Helpers
//===----------------------------------------------------------------------===//

/// Get the CB index attribute name for a compute input.
inline std::string getCBIndexAttrName(unsigned inputIdx) {
  return (kCBIndexAttrPrefix + std::to_string(inputIdx)).str();
}

/// Set CB index attribute on a compute op for a specific input.
inline void setCBIndexAttr(mlir::Operation *compute, unsigned inputIdx,
                           int64_t cbIndex) {
  auto attr = mlir::IntegerAttr::get(
      mlir::IntegerType::get(compute->getContext(), 64), cbIndex);
  compute->setAttr(getCBIndexAttrName(inputIdx), attr);
}

/// Get CB index attribute from a compute op for a specific input.
/// Returns std::nullopt if the attribute is not present.
inline std::optional<int64_t> getCBIndexAttr(mlir::Operation *compute,
                                             unsigned inputIdx) {
  if (auto attr = compute->getAttrOfType<mlir::IntegerAttr>(
          getCBIndexAttrName(inputIdx))) {
    return attr.getInt();
  }
  return std::nullopt;
}

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_IR_TTL_H
