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

/// Runtime configuration attributes
inline constexpr llvm::StringRef kFp32DestAccEnAttrName = "fp32_dest_acc_en";
inline constexpr llvm::StringRef kDstFullSyncEnAttrName = "dst_full_sync_en";

/// FPU binary attribute: marks add/sub/mul ops that should use the FPU
/// execution engine (reads from CB) instead of SFPU (reads from DST).
constexpr llvm::StringLiteral kFPUBinaryAttrName("ttl.fpu_binary");

/// Subblock stride attribute: marks scf.for loops created by the subblock pass
/// with the linearized stride for that loop's dimension. Used by
/// computeCBTileIndexFromLoops to distinguish subblock loops from tile
/// iteration loops and compute correct absolute CB offsets.
constexpr llvm::StringLiteral kSubblockStrideAttrName("ttl.subblock_stride");

/// Tile loop attribute: marks scf.for loops created by lower-to-loops as tile
/// iteration loops. Carries the linearization stride (from the full tensor
/// shape) as an index value. Used by computeCBTileIndexFromLoops to compute
/// correct absolute CB indices — the stride may differ from the loop's upper
/// bound when the compute has been subblocked.
constexpr llvm::StringLiteral kTileLoopAttrName("ttl.tile_loop");

/// Full linearization strides attribute: set on inner ComputeOps created by
/// the subblock pass. Contains the row-major strides from the original (full)
/// tensor shape. Lower-to-loops reads this to annotate tile loops with correct
/// strides for CB indexing.
constexpr llvm::StringLiteral
    kFullLinStridesAttrName("ttl.full_linearization_strides");

/// Per-tile offset attribute: set on ops in unrolled bodies. Carries the
/// linearized tile offset within the subblock. Used by
/// computeCBTileIndexFromLoops to compute correct CB indices without tile
/// loops.
constexpr llvm::StringLiteral kTileOffsetAttrName("ttl.tile_offset");

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

/// Trait for tile-level operations that read from CB rather than DST.
template <typename ConcreteType>
class TTLCBInputTileOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLCBInputTileOpTrait> {};

/// Trait for tile operations with at least one operand consumed from DST.
template <typename ConcreteType>
class TTLDSTInputsTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLDSTInputsTrait> {};

/// Trait for tile operations whose result overwrites the DST input in-place.
template <typename ConcreteType>
class TTLInPlaceOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLInPlaceOpTrait> {};

/// Trait for tile operations that accumulate across multiple invocations.
template <typename ConcreteType>
class TTLAccumulatingOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLAccumulatingOpTrait> {};

/// Trait for tile operations that carry an explicit output CB operand.
/// These operations' init functions configure the PACK thread and require
/// the output CB identifier. Affects init consolidation ordering: full-init
/// ops (PACK-configuring) must precede short-init ops.
template <typename ConcreteType>
class TTLCBOutputTileOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLCBOutputTileOpTrait> {};

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
