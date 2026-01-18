// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Tile Ops to TTKernel Lowering
//===----------------------------------------------------------------------===//
//
// This file lowers TTL tile operations (ttl.tile_* and ttl.copy_tile) to
// TTKernel operations using DialectConversion.
// Future work (TODO #124):
// - DST lifecycle wrapper (acquire/commit/wait/release) around loop iterations
// - copy_tile (CB → DST) before compute, pack_tile (DST → CB) after
//
// Following LLVM/MLIR best practices:
// - Generic template patterns for tile op categories
// - Type aliases for op-to-op mappings
// - Batch pattern registration via patterns.add<...>
// - Explicit state passing for copy_tile (CB → DST) to avoid multipleIR walks
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#define DEBUG_TYPE "ttl-tile-ops-to-ttkernel"

namespace mlir::tt::ttl {

namespace ttk = mlir::tt::ttkernel;

namespace {

/// Look up a CB for a copy_tile source.
/// After loop lowering, src is typically a tensor.extract result.
/// We trace back to find the tensor, then use getAttachedCB to find the CB.
// TODO(#161): Cache cb_index → BindCBOp mapping to avoid O(n×m) complexity
// where n = copy_tile ops, m = bind_cb ops.
static Value lookupCBByIndex(Value src, Operation *funcOp) {
  // Check if src is a block argument (before loop lowering).
  if (auto barg = llvm::dyn_cast<BlockArgument>(src)) {
    // Find the parent compute op and read the cb_index attribute.
    auto computeOp = llvm::dyn_cast<ComputeOp>(barg.getOwner()->getParentOp());
    if (computeOp) {
      unsigned argIdx = barg.getArgNumber();
      if (auto cbIndex = getCBIndexAttr(computeOp, argIdx)) {
        // Validate cb_index is in valid range.
        assert(*cbIndex >= 0 && *cbIndex < kMaxCircularBuffers &&
               "cb_index must be in range [0, 31]");

        // Find the bind_cb op with matching cb_index in the function.
        Value result;
        funcOp->walk([&](BindCBOp bindOp) {
          if (bindOp.getCbIndexAttr().getInt() == *cbIndex) {
            result = bindOp.getResult();
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        return result;
      }
    }
  }

  // After loop lowering: src is a tile from tensor.extract.
  // Trace back to the tensor and use getAttachedCB.
  Value tensor = src;
  if (auto extract = src.getDefiningOp<tensor::ExtractOp>()) {
    tensor = extract.getTensor();
  }

  // Trace through unrealized conversion casts.
  // After cb_wait lowering, the tensor is an unrealized_cast(ttkernel.cb).
  tensor = traceUnrealizedCasts(tensor);

  // If we traced to a ttkernel.cb, return it directly.
  if (llvm::isa<ttkernel::CBType>(tensor.getType())) {
    return tensor;
  }

  // Otherwise, try to find the attached CB.
  if (Value attached = getAttachedCB(tensor)) {
    return attached;
  }

  return Value();
}

//===----------------------------------------------------------------------===//
// DST lifecycle ops
//===----------------------------------------------------------------------===//

struct TTLInitSFPUToTTKernel : OpConversionPattern<InitSFPUOp> {
  using OpConversionPattern<InitSFPUOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InitSFPUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto icb = utils::convertTTLCBToTTKernel(adaptor.getIcb(), rewriter, loc,
                                             getTypeConverter());
    auto ocb = utils::convertTTLCBToTTKernel(adaptor.getOcb(), rewriter, loc,
                                             getTypeConverter());
    if (failed(icb) || failed(ocb)) {
      return rewriter.notifyMatchFailure(op, "failed to convert CB types");
    }

    rewriter.replaceOpWithNewOp<ttk::InitSFPUOp>(op, *icb, *ocb);
    return success();
  }
};

struct TTLTileRegsAcquireToTTKernel : OpConversionPattern<TileRegsAcquireOp> {
  using OpConversionPattern<TileRegsAcquireOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileRegsAcquireOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttk::TileRegsAcquireOp>(op);
    return success();
  }
};

struct TTLTileRegsCommitToTTKernel : OpConversionPattern<TileRegsCommitOp> {
  using OpConversionPattern<TileRegsCommitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileRegsCommitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttk::TileRegsCommitOp>(op);
    return success();
  }
};

struct TTLTileRegsWaitToTTKernel : OpConversionPattern<TileRegsWaitOp> {
  using OpConversionPattern<TileRegsWaitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileRegsWaitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttk::TileRegsWaitOp>(op);
    return success();
  }
};

struct TTLTileRegsReleaseToTTKernel : OpConversionPattern<TileRegsReleaseOp> {
  using OpConversionPattern<TileRegsReleaseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileRegsReleaseOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttk::TileRegsReleaseOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Extract the DST register index from a tile value. The index is obtained
/// from either the copy_tile op that placed the tile in DST, or from the
/// dst_idx attribute on the producing tile operation.
///
/// For block arguments (function parameters), returns the argument number as
/// a fallback. This supports testing tile ops in isolation without copy_tile.
static std::optional<int64_t> getDstIndexFromValue(Value v) {
  // Handle block arguments (function parameters) - use arg number as dst_idx
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    return blockArg.getArgNumber();
  }

  auto opRes = dyn_cast<OpResult>(v);
  if (!opRes) {
    return std::nullopt;
  }
  Operation *owner = opRes.getOwner();
  if (auto copy = dyn_cast<CopyTileOp>(owner)) {
    if (auto constIdx = dyn_cast_or_null<arith::ConstantIndexOp>(
            copy.getDstIndex().getDefiningOp())) {
      return constIdx.value();
    }
  }
  if (auto attr = owner->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
    return attr.getInt();
  }
  return std::nullopt;
}

/// Determines if a value comes directly from a circular buffer (CB) and is
/// suitable for FPU operation optimization.
/// Returns true if the value is:
/// - The result of a copy_tile operation (CB → DST copy that FPU can replace)
/// - A block argument in simple test cases (fallback for isolated tests)
/// Returns false if the value is:
/// - From a tile operation (unary or binary), which produces DST intermediates
[[maybe_unused]] static bool isFromCircularBuffer(Value v) {
  auto opRes = dyn_cast<OpResult>(v);
  if (!opRes) {
    // Block arguments: return true for simple test cases without copy_tile
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      return true;
    }
    return false;
  }

  Operation *owner = opRes.getOwner();

  // If from copy_tile, it's a CB → DST copy that FPU can optimize
  // FPU can replace: copy_tile(CB) + add_binary_tile → add_tiles(CB)
  if (isa<CopyTileOp>(owner)) {
    return true;
  }

  // If from another tile operation, it's a DST intermediate (not from CB)
  // Check for tile operations (unary and binary)
  if (isa<ExpTileOp, SqrtTileOp, RsqrtTileOp, ReluTileOp, SigmoidTileOp,
          TanhTileOp, LogTileOp, AbsTileOp, NegTileOp, AddTileOp, SubTileOp,
          MulTileOp, MaxTileOp>(owner)) {
    return false;
  }

  // Default: assume not from CB (conservative)
  return false;
}

/// Extracts the circular buffer (CB) value for operands that come from CBs.
/// This is used by FPU operations to get the CB for direct access.
/// Returns the CB value if the operand is a block argument, nullopt otherwise.
[[maybe_unused]] static std::optional<Value>
getCircularBufferSource(Value v, Operation *funcOp) {
  // Only block arguments directly reference CBs
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    // Find the parent compute op and read the cb_index attribute
    auto computeOp = dyn_cast<ComputeOp>(blockArg.getOwner()->getParentOp());
    if (computeOp) {
      unsigned argIdx = blockArg.getArgNumber();
      if (auto cbIndex = getCBIndexAttr(computeOp, argIdx)) {
        // Validate cb_index is in valid range
        assert(*cbIndex >= 0 && *cbIndex < kMaxCircularBuffers &&
               "cb_index must be in range [0, 31]");

        // Find the bind_cb op with matching cb_index in the function
        Value result;
        funcOp->walk([&](BindCBOp bindOp) {
          if (bindOp.getCbIndexAttr().getInt() == *cbIndex) {
            result = bindOp.getResult();
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        return result;
      }
    }
  }

  // For non-block-argument values, cannot determine CB source
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Generic Tile Op Lowering Templates (using ConversionPattern)
//===----------------------------------------------------------------------===//

/// Generic pattern for lowering TTL unary tile ops to TTKernel SFPU ops.
/// Unary SFPU ops: DST[dst_idx] = op(DST[dst_idx]) - operates in-place.
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileUnaryToTTKernel : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dstIdxAttr = op->template getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    int64_t dstIdx = dstIdxAttr.getInt();
    Value dstIdxVal = rewriter.create<arith::ConstantIndexOp>(loc, dstIdx);

    // Emit init + compute ops
    rewriter.create<InitOp>(loc);
    rewriter.create<TTKernelComputeOp>(loc, dstIdxVal);

    // Replace all uses with a placeholder (the value is now in DST register)
    // For tile ops, we pass through the input since the result is implicit
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

/// Generic pattern for lowering TTL binary tile ops to TTKernel SFPU ops.
/// Binary SFPU ops: DST[odst] = DST[src0] op DST[src1]
///
/// DST indices are extracted from operand-defining ops (copy_tile or tile ops
/// with dst_idx attributes). The output index comes from this op's dst_idx.
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileBinaryToTTKernel : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dstIdxAttr = op->template getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    int64_t odstIdx = dstIdxAttr.getInt();

    auto src0IdxOpt = getDstIndexFromValue(op.getLhs());
    auto src1IdxOpt = getDstIndexFromValue(op.getRhs());

    if (!src0IdxOpt) {
      return rewriter.notifyMatchFailure(
          op, "failed to extract dst_idx from lhs operand");
    }
    if (!src1IdxOpt) {
      return rewriter.notifyMatchFailure(
          op, "failed to extract dst_idx from rhs operand");
    }

    int64_t src0Idx = *src0IdxOpt;
    int64_t src1Idx = *src1IdxOpt;

    Value src0 = rewriter.create<arith::ConstantIndexOp>(loc, src0Idx);
    Value src1 = rewriter.create<arith::ConstantIndexOp>(loc, src1Idx);
    Value odst = rewriter.create<arith::ConstantIndexOp>(loc, odstIdx);

    rewriter.create<InitOp>(loc);
    rewriter.create<TTKernelComputeOp>(loc, src0, src1, odst);

    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }
};

/// Pattern for lowering TTL binary tile ops to TTKernel FPU ops.
/// FPU binary ops: DST[odst] = op(CB[in0_cb][idx0], CB[in1_cb][idx1])
///
/// This pattern matches binary operations where BOTH operands come from
/// circular buffers (block arguments). FPU operations read directly from
/// CBs without requiring copy_tile, making them ~2x faster than SFPU.
///
/// Pattern priority: HIGH (benefit=2) - matches before SFPU pattern
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileBinaryToTTKernelFPU : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Check if both operands are from circular buffers
    if (!isFromCircularBuffer(op.getLhs()) ||
        !isFromCircularBuffer(op.getRhs())) {
      return failure(); // Let lower-priority pattern handle it
    }

    // Get the function context for CB lookup
    auto funcOp = op->template getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(op, "operation not in function");
    }

    // Extract CB sources for both operands
    auto lhsCBOpt = getCircularBufferSource(op.getLhs(), funcOp);
    auto rhsCBOpt = getCircularBufferSource(op.getRhs(), funcOp);

    if (!lhsCBOpt || !rhsCBOpt) {
      return failure(); // Cannot extract CB - let SFPU pattern handle
    }

    Value lhsCB = *lhsCBOpt;
    Value rhsCB = *rhsCBOpt;

    // Convert CB types to !ttkernel.cb
    auto *typeConverter = this->getTypeConverter();
    if (!typeConverter) {
      return rewriter.notifyMatchFailure(op, "no type converter available");
    }

    // Convert LHS CB
    Type targetLhsCbTy;
    if (auto ttkCb = mlir::dyn_cast<ttk::CBType>(lhsCB.getType())) {
      targetLhsCbTy = ttkCb;
    } else if (auto ttlCb =
                   mlir::dyn_cast<CircularBufferType>(lhsCB.getType())) {
      targetLhsCbTy = ttk::CBType::get(lhsCB.getContext(),
                                       ttlCb.getTotalElements(),
                                       ttlCb.getElementType());
    }
    if (!targetLhsCbTy) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to determine lhs cb target type");
    }
    lhsCB = typeConverter->materializeTargetConversion(rewriter, loc,
                                                       targetLhsCbTy, lhsCB);
    if (!lhsCB || lhsCB.getType() != targetLhsCbTy) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize lhs ttkernel.cb");
    }

    // Convert RHS CB
    Type targetRhsCbTy;
    if (auto ttkCb = mlir::dyn_cast<ttk::CBType>(rhsCB.getType())) {
      targetRhsCbTy = ttkCb;
    } else if (auto ttlCb =
                   mlir::dyn_cast<CircularBufferType>(rhsCB.getType())) {
      targetRhsCbTy = ttk::CBType::get(rhsCB.getContext(),
                                       ttlCb.getTotalElements(),
                                       ttlCb.getElementType());
    }
    if (!targetRhsCbTy) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to determine rhs cb target type");
    }
    rhsCB = typeConverter->materializeTargetConversion(rewriter, loc,
                                                       targetRhsCbTy, rhsCB);
    if (!rhsCB || rhsCB.getType() != targetRhsCbTy) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize rhs ttkernel.cb");
    }

    // Extract output DST index
    auto dstIdxAttr = op->template getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    int64_t odstIdx = dstIdxAttr.getInt();

    // Create index constants
    // For block arguments, tile index is 0 (could be extended for tensor.extract)
    Value idx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value idx1 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value odst = rewriter.create<arith::ConstantIndexOp>(loc, odstIdx);

    // Emit FPU operations
    rewriter.create<InitOp>(loc, lhsCB, rhsCB);
    rewriter.create<TTKernelComputeOp>(loc, lhsCB, rhsCB, idx0, idx1, odst);

    // Replace op with LHS operand (result is now in DST register)
    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }
};

/// Special pattern for MaxTileOp which uses 2-arg in-place form:
/// DST[dst0] = max(DST[dst0], DST[dst1])
/// TODO: Remove this special pattern once TTKernel adds a 3-arg max_binary_tile
/// op that matches the add/sub/mul signature: max(src0, src1, odst).
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileMaxToTTKernel : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Extract dst_idx from original (unconverted) operands, not adaptor
    // operands
    auto dst0IdxOpt = getDstIndexFromValue(op.getLhs());
    auto dst1IdxOpt = getDstIndexFromValue(op.getRhs());

    if (!dst0IdxOpt) {
      return rewriter.notifyMatchFailure(
          op, "failed to extract dst_idx from lhs operand");
    }
    if (!dst1IdxOpt) {
      return rewriter.notifyMatchFailure(
          op, "failed to extract dst_idx from rhs operand");
    }

    int64_t dst0Idx = *dst0IdxOpt;
    int64_t dst1Idx = *dst1IdxOpt;

    Value dst0 = rewriter.create<arith::ConstantIndexOp>(loc, dst0Idx);
    Value dst1 = rewriter.create<arith::ConstantIndexOp>(loc, dst1Idx);

    rewriter.create<InitOp>(loc);
    rewriter.create<TTKernelComputeOp>(loc, dst0, dst1, dst0);

    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }
};

/// Lower ttl.copy_tile to TTKernel copy_tile_init + copy_tile.
struct TTLTileCopyToTTKernel : OpConversionPattern<CopyTileOp> {
  using OpConversionPattern<CopyTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CopyTileOp op, CopyTileOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Look up the CB by reading cb_index annotation from the compute op.
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(op, "copy_tile not in function");
    }

    Value cb = lookupCBByIndex(op.getSrc(), funcOp);
    if (!cb) {
      return rewriter.notifyMatchFailure(op, "cannot find attached cb for src");
    }

    // Convert !ttl.cb to !ttkernel.cb.
    auto *typeConverter = this->getTypeConverter();
    Type targetCbTy;
    if (auto ttkCb = mlir::dyn_cast<ttk::CBType>(cb.getType())) {
      targetCbTy = ttkCb;
    } else if (auto ttlCb = mlir::dyn_cast<CircularBufferType>(cb.getType())) {
      targetCbTy = ttk::CBType::get(cb.getContext(), ttlCb.getTotalElements(),
                                    ttlCb.getElementType());
    }
    if (!targetCbTy) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to determine cb target type");
    }
    if (!typeConverter) {
      return rewriter.notifyMatchFailure(op, "no type converter available");
    }
    cb = typeConverter->materializeTargetConversion(rewriter, loc, targetCbTy,
                                                    cb);
    if (!cb || cb.getType() != targetCbTy) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize ttkernel.cb");
    }

    // Initialize the copy for the given CB (matches TTKernel contract).
    rewriter.create<ttk::CopyTileInitOp>(loc, cb);
    // Emit the copy from CB[src_index] to DST[dst_index].
    rewriter.create<ttk::CopyTileOp>(loc, cb, adaptor.getSrcIndex(),
                                     adaptor.getDstIndex());

    // Materialize results: dst token from dst_index, and a tile value
    // passthrough (the tile remains the same logical value for downstream tile
    // ops).
    auto token = rewriter
                     .create<mlir::UnrealizedConversionCastOp>(
                         loc, TypeRange{op.getResult(0).getType()},
                         ValueRange{adaptor.getDstIndex()})
                     .getResult(0);
    auto tile = rewriter
                    .create<mlir::UnrealizedConversionCastOp>(
                        loc, TypeRange{op.getResult(1).getType()},
                        ValueRange{adaptor.getSrc()})
                    .getResult(0);
    rewriter.replaceOp(op, ValueRange{token, tile});
    return success();
  }
};

/// Lower ttl.copy_dst to TTKernel copy_dest_values_init + copy_dest_values.
/// This copies a tile from one DST register to another.
struct TTLCopyDstToTTKernel : OpConversionPattern<CopyDstOp> {
  using OpConversionPattern<CopyDstOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CopyDstOp op, CopyDstOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the source DST index from the input tile's producing operation.
    auto srcDstIdx = getDstIndexFromValue(op.getSrcTile());
    if (!srcDstIdx) {
      return rewriter.notifyMatchFailure(
          op, "cannot determine src DST index from input tile");
    }

    // Get the destination DST index from this op's dst_idx attribute.
    auto dstIdxAttr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    int64_t dstDstIdx = dstIdxAttr.getInt();

    // Create index constants for src and dst DST registers.
    Value srcIdx = rewriter.create<arith::ConstantIndexOp>(loc, *srcDstIdx);
    Value dstIdx = rewriter.create<arith::ConstantIndexOp>(loc, dstDstIdx);

    // Emit copy_dest_values_init + copy_dest_values.
    // copy_dest_values(dst0, dst1) copies DST[dst1] → DST[dst0].
    rewriter.create<ttk::CopyDestValuesInitOp>(loc);
    rewriter.create<ttk::CopyDestValuesOp>(loc, dstIdx, srcIdx);

    // Replace with an unrealized conversion cast to preserve the tile value.
    // The tile is now in DST[dstIdx].
    auto tile = rewriter
                    .create<mlir::UnrealizedConversionCastOp>(
                        loc, TypeRange{op.getResult().getType()},
                        ValueRange{adaptor.getSrcTile()})
                    .getResult(0);
    rewriter.replaceOp(op, tile);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tile Op Lowerings - Generated from TTLElementwiseOps.def
//===----------------------------------------------------------------------===//

// Generate type aliases for unary tile op lowerings
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  using TTL_OP##TileLowering =                                                 \
      TTLTileUnaryToTTKernel<TILE_OP, ttk::TTK_INIT, ttk::TTK_COMPUTE>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

// Generate type aliases for binary tile op lowerings (standard 3-arg form)
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  using TTL_OP##TileLowering =                                                 \
      TTLTileBinaryToTTKernel<TILE_OP, ttk::TTK_INIT, ttk::TTK_COMPUTE>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

// Generate type aliases for special binary tile op lowerings (2-arg in-place)
#define TTL_BINARY_TILE_OP_SPECIAL(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)     \
  using TTL_OP##TileLowering =                                                 \
      TTLTileMaxToTTKernel<TILE_OP, ttk::TTK_INIT, ttk::TTK_COMPUTE>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

// Generate type aliases for FPU binary tile op lowerings
// FPU ops use direct CB access: DST[odst] = op(CB[in0][idx0], CB[in1][idx1])
#define TTL_BINARY_TILE_OP_FPU(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)         \
  using TTL_OP##TileFPULowering =                                              \
      TTLTileBinaryToTTKernelFPU<TILE_OP, ttk::TTK_INIT, ttk::TTK_COMPUTE>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTLTileOpsToTTKernelPatterns(TypeConverter *typeConverter,
                                          RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Control ops (init_sfpu needs type converter for CB conversion).
  patterns.add<TTLInitSFPUToTTKernel>(*typeConverter, ctx);
  patterns.add<TTLTileRegsAcquireToTTKernel, TTLTileRegsCommitToTTKernel,
               TTLTileRegsWaitToTTKernel, TTLTileRegsReleaseToTTKernel>(ctx);

  // Tile op lowerings - generated from TTLElementwiseOps.def
  // Unary ops (ttl.tile_* → ttkernel.*_tile)
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  patterns.add<TTL_OP##TileLowering>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // FPU binary ops (HIGH PRIORITY: benefit=2)
  // Use FPU direct CB access when both operands are from CBs
#define TTL_BINARY_TILE_OP_FPU(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)         \
  patterns.add<TTL_OP##TileFPULowering>(*typeConverter, ctx, 2);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // SFPU binary ops (DEFAULT PRIORITY: benefit=0)
  // Fallback when FPU pattern doesn't match (operands from DST)
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  patterns.add<TTL_OP##TileLowering>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Special binary ops (non-standard lowering template)
#define TTL_BINARY_TILE_OP_SPECIAL(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)     \
  patterns.add<TTL_OP##TileLowering>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Copy ops need the type converter.
  patterns.add<TTLTileCopyToTTKernel>(*typeConverter, ctx);
  patterns.add<TTLCopyDstToTTKernel>(ctx);

  // TODO(#124): Add DST lifecycle wrapper pattern for loop iterations
  // (acquire/commit/wait/release + copy_tile/pack_tile)
}

} // namespace mlir::tt::ttl
