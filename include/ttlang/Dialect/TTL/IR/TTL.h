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
constexpr llvm::StringLiteral kDstIdxAttrName("dst_idx");
constexpr llvm::StringLiteral kCBIndexAttrPrefix("ttl.cb_index.");

/// Runtime configuration attributes.
constexpr llvm::StringLiteral kFp32DestAccEnAttrName("fp32_dest_acc_en");
constexpr llvm::StringLiteral kDstFullSyncEnAttrName("dst_full_sync_en");

/// Marks binary ops that use the FPU engine (reads from CB) instead of SFPU.
constexpr llvm::StringLiteral kFPUBinaryAttrName("ttl.fpu_binary");

/// Number of tiles to process per DST sync region (set by TTLAssignDST).
constexpr llvm::StringLiteral kUnrollFactorAttrName("ttl.unroll_factor");

/// Linearized stride for a subblock loop dimension. Distinguishes subblock
/// loops from tile iteration loops for CB index computation.
constexpr llvm::StringLiteral kSubblockStrideAttrName("ttl.subblock_stride");

/// Row-major strides of the CB block iteration domain (before subblocking),
/// carried on subblocked ComputeOps so tile loops get correct CB linearization
/// strides.
constexpr llvm::StringLiteral
    kFullLinStridesAttrName("ttl.full_linearization_strides");

/// Linearization stride on a tile iteration loop. May differ from the loop
/// bound when the compute has been subblocked.
constexpr llvm::StringLiteral kTileLoopAttrName("ttl.tile_loop");

/// Linearized tile offset within a subblock, used for CB index computation
/// in unrolled (loop-free) bodies.
constexpr llvm::StringLiteral kTileOffsetAttrName("ttl.tile_offset");

/// Output CB index on tile_bcast ops, avoiding SSA tracing during lowering.
constexpr llvm::StringLiteral
    kBcastOutputCBIndexAttrName("ttl.bcast_output_cb_index");

/// Trait for data movement operations (copy_tile, copy_dst).
template <typename ConcreteType>
class TTLDataMovementOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLDataMovementOpTrait> {};

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
