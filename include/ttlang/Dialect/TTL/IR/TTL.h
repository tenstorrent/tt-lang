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
constexpr llvm::StringLiteral kCBIndexAttrPrefix("ttl.cb_index.");

/// Runtime configuration attributes.
constexpr llvm::StringLiteral kFp32DestAccEnAttrName("fp32_dest_acc_en");
constexpr llvm::StringLiteral kDstFullSyncEnAttrName("dst_full_sync_en");

/// Marks binary ops that use the FPU engine (reads from CB) instead of SFPU.
constexpr llvm::StringLiteral kFPUBinaryAttrName("ttl.fpu_binary");

/// Number of tiles to process per DST sync region (set by TTLAssignDST).
constexpr llvm::StringLiteral kUnrollFactorAttrName("ttl.unroll_factor");

/// Marks an scf.for as a compiler-generated subblock loop. The integer value
/// is the linearization stride of this dimension, assuming row-major tile
/// ordering in the CB (interleaved layout).
constexpr llvm::StringLiteral
    kSubblockLoopStrideAttrName("ttl.subblock_loop_stride");

/// Iteration domain dimension index on a subblock loop, recording which
/// dimension the loop iterates over.
constexpr llvm::StringLiteral kSubblockDimAttrName("ttl.subblock_dim");

/// Linearization strides of the full iteration domain (before subblocking),
/// carried on subblocked ComputeOps so tile loops get correct CB strides.
constexpr llvm::StringLiteral
    kFullLinStridesAttrName("ttl.full_linearization_strides");

/// Marks an scf.for as a compiler-generated tile loop. The integer value is
/// the linearization stride of this dimension, assuming row-major tile
/// ordering in the CB (interleaved layout).
constexpr llvm::StringLiteral kTileLoopStrideAttrName("ttl.tile_loop_stride");

/// Marks an scf.for loop as iterating over a reduction dimension.
/// Preserves the reduction semantics from iterator_types after the
/// ComputeOp is lowered to loops.
constexpr llvm::StringLiteral kReductionLoopAttrName("ttl.reduction_loop");

/// Output CB index on tile ops that need it for init insertion.
constexpr llvm::StringLiteral
    kBcastOutputCBIndexAttrName("ttl.bcast_output_cb_index");
constexpr llvm::StringLiteral
    kReduceOutputCBIndexAttrName("ttl.reduce_output_cb_index");
constexpr llvm::StringLiteral
    kTransposeOutputCBIndexAttrName("ttl.transpose_output_cb_index");

/// Marks a copy_tile as a placeholder inserted during DST assignment Phase 1.
/// Replaced with a proper copy in Phase 2b.
constexpr llvm::StringLiteral kPlaceholderCopyAttrName("ttl.placeholder_copy");

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

/// Trait for tile operations that write to a DST register.
template <typename ConcreteType>
class TTLDstResultOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLDstResultOpTrait> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    if (op->getNumOperands() == 0) {
      return op->emitOpError("expected at least one operand (dst_index)");
    }
    mlir::Value lastOperand = op->getOperand(op->getNumOperands() - 1);
    if (!lastOperand.getType().isIndex()) {
      return op->emitOpError("last operand (dst_index) must be index type, "
                             "got ")
             << lastOperand.getType();
    }
    return mlir::success();
  }
};

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
