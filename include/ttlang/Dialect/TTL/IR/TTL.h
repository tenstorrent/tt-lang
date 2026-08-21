// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTL_H
#define TTLANG_DIALECT_TTL_IR_TTL_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Target/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

#include "ttlang/Dialect/TTL/IR/TTLOpsDialect.h.inc"

namespace mlir::tt::ttl {

/// Default tile dimensions used for TTL tensors.
inline constexpr int32_t kDefaultTileHeight = 32;
inline constexpr int32_t kDefaultTileWidth = 32;
/// TT kernel hardware semaphore id capacity. Mirrored by
/// python/ttl/constants.py for simulator-side resource checks.
inline constexpr int64_t kMaxHardwareSemaphoreIds = 16;

/// Tag for tile-level operations to enable identity checks without type
/// inspection.
template <typename ConcreteType>
class TTLTileOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTLTileOpTrait> {};

/// Attribute names.
constexpr llvm::StringLiteral kCBIndexAttrPrefix("ttl.cb_index.");

/// Runtime configuration attributes.
constexpr llvm::StringLiteral kFp32DestAccEnAttrName("fp32_dest_acc_en");
constexpr llvm::StringLiteral kDstFullSyncEnAttrName("dst_full_sync_en");
constexpr llvm::StringLiteral
    kUnpackToDestFp32AttrName("ttl.unpack_to_dest_fp32");

/// Selected strategy on tile operations with execution alternatives.
constexpr llvm::StringLiteral
    kTileExecutionStrategyAttrName("ttl.tile_execution_strategy");
/// PipeNet role exposed by `is_src` / `is_dst` / `is_active` predicate ops
/// and by `pipenet_scope` declarations.
enum class PipeRole : int64_t {
  Source = 0,
  Destination = 1,
  Active = 2,
};

/// Target-independent compute primitive implemented by a TTL operation.
enum class ComputePrimitive {
  Add,
  Subtract,
  Multiply,
  ElementwiseBinary,
  ElementwiseUnary,
  Broadcast,
  Reduce,
  Transpose,
  Fill,
  Matmul,
  Typecast,
  MultiplyByConstant,
  Passthrough,
};

/// A contiguous set of DST slots starting at `baseIndex`.
struct DstFootprint {
  mlir::Value baseIndex;
  int64_t tileCount = 1;
};

mlir::FailureOr<llvm::SmallVector<DstFootprint, 2>>
getDefaultDstReadFootprints(mlir::Operation *op);
llvm::SmallVector<DstFootprint, 2>
getDefaultDstWriteFootprints(mlir::Operation *op);
mlir::FailureOr<DstFootprint> getDefaultResultDstFootprint(mlir::Operation *op,
                                                           mlir::Value result);

/// Function-level policy for selecting FPU add, subtract, and multiply.
constexpr llvm::StringLiteral
    kEnableFPUBinaryOpsAttrName("ttl.enable_fpu_binary_ops");

/// Func-level: tags a func.func as a kernel thread (compute / dataflow);
/// the attribute value is a `ttkernel.thread` enum.
constexpr llvm::StringLiteral kKernelThreadAttrName("ttl.kernel_thread");

/// Func-level target-independent logical-kernel identity.
constexpr llvm::StringLiteral kLogicalKernelAttrName("ttl.logical_kernel");

/// Number of tiles per DST sync region.
constexpr llvm::StringLiteral kUnrollFactorAttrName("ttl.unroll_factor");

/// Func-level: NOC index (0 = reader/NCRISC, 1 = writer/BRISC) of a
/// datamovement kernel; set by the frontend, read via getNocIndex during
/// TTL->TTKernel lowering and by the ttnn runtime bridge for reader/writer
/// config assignment. Mirrored in python/ttl/ttl_api.py.
constexpr llvm::StringLiteral kNocIndexAttrName("ttl.noc_index");

/// Marks an scf.for as a compiler-generated subblock loop. Integer value is
/// the linearization stride for this dimension.
constexpr llvm::StringLiteral
    kSubblockLoopStrideAttrName("ttl.subblock_loop_stride");

/// Iteration domain dimension index on a subblock loop.
constexpr llvm::StringLiteral kSubblockDimAttrName("ttl.subblock_dim");

/// Linearization strides of the full iteration domain (pre-subblocking).
constexpr llvm::StringLiteral
    kFullLinStridesAttrName("ttl.full_linearization_strides");

/// Marks an scf.for as a compiler-generated tile loop. Integer value is the
/// linearization stride for this dimension.
constexpr llvm::StringLiteral kTileLoopStrideAttrName("ttl.tile_loop_stride");

/// Marks an scf.for loop as iterating over a reduction dimension.
constexpr llvm::StringLiteral kReductionLoopAttrName("ttl.reduction_loop");

/// Marks a user-written scf.for as an L1 accumulation loop. Distinct from
/// kReductionLoopAttrName which marks compiler-generated reduction loops.
constexpr llvm::StringLiteral kL1AccLoopAttrName("ttl.l1_acc_loop");

/// Output CB index for tile ops.
constexpr llvm::StringLiteral
    kBcastOutputCBIndexAttrName("ttl.bcast_output_cb_index");
constexpr llvm::StringLiteral
    kReduceOutputCBIndexAttrName("ttl.reduce_output_cb_index");
constexpr llvm::StringLiteral
    kTransposeOutputCBIndexAttrName("ttl.transpose_output_cb_index");

/// Placeholder marker on copy_tile (replaced during DST assignment).
constexpr llvm::StringLiteral kPlaceholderCopyAttrName("ttl.placeholder_copy");

/// Module attribute containing one runtime descriptor per physical DFB index.
constexpr llvm::StringLiteral kDFBAllocationsAttrName("ttl.dfb_allocations");

/// Module attributes carrying compiler-owned pipe resource allocation.
constexpr llvm::StringLiteral
    kPipeSyncSemaphoreCountAttrName("ttl.pipe_sync_semaphore_count");
constexpr llvm::StringLiteral
    kPipeGlobalSemaphoreCountAttrName("ttl.pipe_global_semaphore_count");
constexpr llvm::StringLiteral
    kPipeSramScratchBytesAttrName("ttl.pipe_sram_scratch_bytes");
constexpr llvm::StringLiteral
    kPipeConservativeL1BytesAttrName("ttl.pipe_conservative_l1_bytes");

/// Module attribute carrying the number of synchronized DFB resets.
constexpr llvm::StringLiteral kDFBResetCountAttrName("ttl.dfb_reset_count");

/// Function attribute listing receiver DFB indices whose L1 base addresses are
/// passed after tensor buffer addresses as common runtime arguments.
constexpr llvm::StringLiteral kPipeComputedAddressDFBIndicesAttrName(
    "ttl.pipe_computed_address_dfb_indices");

/// Marker on BindCBOp to distinguish compiler-allocated DFBs from user-declared
/// ones.
constexpr llvm::StringLiteral
    kCompilerAllocatedAttrName("ttl.compiler_allocated");

/// Function attribute recording the base compile-time argument index.
/// CTA layout is [CBs, TAs], so this equals the number of CBs.
constexpr llvm::StringLiteral kBaseCTAIndexAttrName("ttl.base_cta_index");

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

/// Marks binary tile ops that support both FPU and SFPU execution strategies.
template <typename ConcreteType>
class TTLStrategyDependentBinaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      TTLStrategyDependentBinaryOpTrait> {};

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

/// Trait for tile operations with an explicit output CB operand.
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
