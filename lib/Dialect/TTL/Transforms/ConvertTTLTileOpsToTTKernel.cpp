// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Tile Ops to TTKernel Lowering
//===----------------------------------------------------------------------===//
//
// Lowers TTL tile-level operations to TTKernel using DialectConversion.
// This file covers compute ops (unary SFPU, binary SFPU, FPU binary,
// broadcast), data movement ops (copy_tile, copy_dst), and DST register
// lifecycle ops (tile_regs_acquire/commit/wait/release).
//
// Unary, binary SFPU, and FPU binary compute ops are lowered via generic
// template patterns instantiated from TTLElementwiseOps.def.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
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

  // Trace through tensor.extract_slice (from compute subblocking).
  while (auto slice = tensor.getDefiningOp<tensor::ExtractSliceOp>()) {
    tensor = slice.getSource();
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

/// Compute the CB tile index for an operand by tracing back to its defining
/// tensor.extract and linearizing the extract indices in the operand's
/// row-major layout. The extract indices are the source of truth for CB tile
/// positions -- they encode the indexing map application performed during
/// loop lowering (generateTileProcessing), so this works correctly for any
/// indexing map (identity, transpose, broadcast, reduction) without needing
/// the map itself.
///
/// For unrolled ops, the extract indices are constants and the linearization
/// folds to a constant. For loop-based ops, the indices are loop IVs and
/// the result is an arith expression.
///
/// Precondition: the operand must trace back to a tensor.extract. This holds
/// for all CB-input tile ops after loop lowering, since generateTileProcessing
/// always creates tensor.extract ops for each input. The extract survives
/// unrolling (clone preserves it) and cannot fold away (the source tensor is
/// from attach_cb, which is opaque to canonicalization).
///
/// Returns failure if the precondition is violated.
static FailureOr<Value> computeCBTileIndex(Value operand, OpBuilder &builder,
                                           Location loc) {
  auto extractOp = operand.getDefiningOp<tensor::ExtractOp>();
  if (!extractOp) {
    return failure();
  }

  auto tensorTy =
      mlir::dyn_cast<RankedTensorType>(extractOp.getTensor().getType());
  if (!tensorTy) {
    return failure();
  }

  // Linearize the extract indices within the immediate tensor's shape.
  Value localIndex = affine::AffineLinearizeIndexOp::create(
      builder, loc, extractOp.getIndices(), tensorTy.getShape());

  // If the tensor comes from an extract_slice (subblocking), convert
  // the local index to a global CB index by adding the slice offset.
  return utils::addSliceOffset(extractOp.getTensor(), localIndex, builder, loc);
}

/// Look up and convert a CB for an operand.
/// Combines lookupCBByIndex with type conversion to TTKernel CB type.
static FailureOr<Value> lookupAndConvertCB(Value operand, func::FuncOp funcOp,
                                           const TypeConverter *typeConverter,
                                           ConversionPatternRewriter &rewriter,
                                           Location loc) {
  Value cb = lookupCBByIndex(operand, funcOp);
  if (!cb) {
    return failure();
  }

  Type targetCbTy;
  if (auto ttkCb = mlir::dyn_cast<ttk::CBType>(cb.getType())) {
    targetCbTy = ttkCb;
  } else if (auto ttlCb = mlir::dyn_cast<CircularBufferType>(cb.getType())) {
    targetCbTy = ttk::CBType::get(cb.getContext(), ttlCb.getTotalElements(),
                                  ttlCb.getElementType());
  }
  if (!targetCbTy || !typeConverter) {
    return failure();
  }

  Value converted =
      typeConverter->materializeTargetConversion(rewriter, loc, targetCbTy, cb);
  if (!converted || converted.getType() != targetCbTy) {
    return failure();
  }
  return converted;
}

//===----------------------------------------------------------------------===//
// DST lifecycle ops
//===----------------------------------------------------------------------===//

/// Trivial 1:1 op conversion: replaces SourceOp with TargetOp (no operands,
/// no results, no type conversion). Modeled after upstream
/// OneToOneConvertToLLVMPattern.
template <typename SourceOp, typename TargetOp>
struct TTLSimpleOneToOne : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op);
    return success();
  }
};

using TTLTileRegsAcquireToTTKernel =
    TTLSimpleOneToOne<TileRegsAcquireOp, ttk::TileRegsAcquireOp>;
using TTLTileRegsCommitToTTKernel =
    TTLSimpleOneToOne<TileRegsCommitOp, ttk::TileRegsCommitOp>;
using TTLTileRegsWaitToTTKernel =
    TTLSimpleOneToOne<TileRegsWaitOp, ttk::TileRegsWaitOp>;
using TTLTileRegsReleaseToTTKernel =
    TTLSimpleOneToOne<TileRegsReleaseOp, ttk::TileRegsReleaseOp>;

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
    return getConstantIntValue(copy.getDstIndex());
  }
  if (auto attr = owner->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
    return attr.getInt();
  }
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
    Value dstIdxVal = arith::ConstantIndexOp::create(rewriter, loc, dstIdx);

    TTKernelComputeOp::create(rewriter, loc, dstIdxVal);

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
/// Ops marked with kFPUBinaryAttrName are skipped (handled by FPU pattern).
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileBinaryToTTKernel : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // FPU-marked ops are handled by TTLTileBinaryFPUToTTKernel.
    if (op->hasAttr(kFPUBinaryAttrName)) {
      return failure();
    }

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

    Value src0 = arith::ConstantIndexOp::create(rewriter, loc, src0Idx);
    Value src1 = arith::ConstantIndexOp::create(rewriter, loc, src1Idx);
    Value odst = arith::ConstantIndexOp::create(rewriter, loc, odstIdx);

    TTKernelComputeOp::create(rewriter, loc, src0, src1, odst);

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

    Value dst0 = arith::ConstantIndexOp::create(rewriter, loc, dst0Idx);
    Value dst1 = arith::ConstantIndexOp::create(rewriter, loc, dst1Idx);

    TTKernelComputeOp::create(rewriter, loc, dst0, dst1, dst0);

    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }
};

/// Generic pattern for lowering TTL binary tile ops to TTKernel FPU ops.
/// FPU binary ops: read both operands from CBs, write result to DST.
/// add_tiles(in0_cb, in1_cb, in0_tile_index, in1_tile_index, dst_index)
///
/// Only matches ops marked with kFPUBinaryAttrName (set by TTLAssignDST).
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileBinaryFPUToTTKernel : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match FPU-marked ops.
    if (!op->hasAttr(kFPUBinaryAttrName)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto funcOp = op->template getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(op, "op not in function");
    }
    auto *typeConverter = this->getTypeConverter();

    // Look up CBs for lhs and rhs.
    auto lhsCB =
        lookupAndConvertCB(op.getLhs(), funcOp, typeConverter, rewriter, loc);
    auto rhsCB =
        lookupAndConvertCB(op.getRhs(), funcOp, typeConverter, rewriter, loc);
    if (failed(lhsCB) || failed(rhsCB)) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot find/convert input CBs for "
                                         "FPU binary");
    }

    // DST output index from attribute (assigned by TTLAssignDST).
    auto dstIdxAttr = op->template getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    Value dstIdx =
        arith::ConstantIndexOp::create(rewriter, loc, dstIdxAttr.getInt());

    // Verify matching per-block tile counts (via tensor shape, not
    // ttk::CBType::getNumTiles() which includes buffer_factor).
    auto lhsExtract = op.getLhs().template getDefiningOp<tensor::ExtractOp>();
    auto rhsExtract = op.getRhs().template getDefiningOp<tensor::ExtractOp>();
    assert(lhsExtract && rhsExtract &&
           "FPU binary operands must originate from tensor.extract");
    auto lhsTensorTy =
        mlir::cast<RankedTensorType>(lhsExtract.getTensor().getType());
    auto rhsTensorTy =
        mlir::cast<RankedTensorType>(rhsExtract.getTensor().getType());
    if (lhsTensorTy.getShape() != rhsTensorTy.getShape()) {
      return rewriter.notifyMatchFailure(
          op,
          llvm::Twine("FPU binary requires operands with matching per-block "
                      "shapes; lhs has ") +
              llvm::Twine(lhsTensorTy.getNumElements()) + " tiles, rhs has " +
              llvm::Twine(rhsTensorTy.getNumElements()));
    }

    // CB tile index: both operands share the same index because
    // TTLAssignDST only marks ops as FPU-eligible when both have
    // identical indexing maps.
    auto cbIdx = computeCBTileIndex(op.getLhs(), rewriter, loc);
    if (failed(cbIdx)) {
      return rewriter.notifyMatchFailure(
          op, "cannot compute CB tile index from tensor.extract");
    }

    // Emit compute op (init inserted by ttkernel-insert-inits pass).
    TTKernelComputeOp::create(rewriter, loc, *lhsCB, *rhsCB, *cbIdx, *cbIdx,
                              dstIdx);

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

    auto cbResult = lookupAndConvertCB(op.getSrc(), funcOp,
                                       this->getTypeConverter(), rewriter, loc);
    if (failed(cbResult)) {
      return rewriter.notifyMatchFailure(
          op, "cannot find/convert attached cb for src");
    }
    Value cb = *cbResult;

    // Linearize multi-dimensional src_indices to a flat CB tile index.
    // Use the immediate tensor shape (which may be a subblock slice), not
    // the full root shape. addSliceOffset converts from local to global.
    ValueRange srcIndices = adaptor.getSrcIndices();
    if (srcIndices.empty()) {
      return op.emitError("copy_tile has no src_indices; "
                          "ttl-lower-to-loops must run first");
    }
    Value srcTensor = op.getSrc();
    if (auto extract = srcTensor.getDefiningOp<tensor::ExtractOp>()) {
      srcTensor = extract.getTensor();
    }
    auto srcTensorTy = mlir::dyn_cast<RankedTensorType>(srcTensor.getType());
    if (!srcTensorTy) {
      return rewriter.notifyMatchFailure(
          op, "cannot determine source tensor shape for linearization");
    }
    Value flatSrcIndex = affine::AffineLinearizeIndexOp::create(
        rewriter, loc, srcIndices, srcTensorTy.getShape());

    // If the source is a subblock slice, convert local to global DFB index.
    flatSrcIndex =
        utils::addSliceOffset(op.getSrc(), flatSrcIndex, rewriter, loc);

    // Emit the copy from CB[flat_index] to DST[dst_index]
    // (init inserted by ttkernel-insert-inits pass).
    ttk::CopyTileOp::create(rewriter, loc, cb, flatSrcIndex,
                            adaptor.getDstIndex());

    // Materialize results: dst token from dst_index, and a tile value
    // passthrough (the tile remains the same logical value for downstream tile
    // ops).
    auto token = mlir::UnrealizedConversionCastOp::create(
                     rewriter, loc, TypeRange{op.getResult(0).getType()},
                     ValueRange{adaptor.getDstIndex()})
                     .getResult(0);
    auto tile = mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, TypeRange{op.getResult(1).getType()},
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
    Value srcIdx = arith::ConstantIndexOp::create(rewriter, loc, *srcDstIdx);
    Value dstIdx = arith::ConstantIndexOp::create(rewriter, loc, dstDstIdx);

    // Emit copy_dest_values(idst_in, idst_out): copies DST[idst_in] ->
    // DST[idst_out].
    ttk::CopyDestValuesOp::create(rewriter, loc, srcIdx, dstIdx);

    // Replace with an unrealized conversion cast to preserve the tile value.
    // The tile is now in DST[dstIdx].
    auto tile = mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, TypeRange{op.getResult().getType()},
                    ValueRange{adaptor.getSrcTile()})
                    .getResult(0);
    rewriter.replaceOp(op, tile);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Bcast Tile Op Lowering
//===----------------------------------------------------------------------===//

/// Convert TTL BcastType to TTKernel BcastType.
static ttk::BcastType convertBcastType(ttl::BcastType ttlType) {
  switch (ttlType) {
  case ttl::BcastType::Col:
    return ttk::BcastType::Col;
  case ttl::BcastType::Row:
    return ttk::BcastType::Row;
  case ttl::BcastType::Scalar:
    return ttk::BcastType::Scalar;
  }
  llvm_unreachable("unknown BcastType");
}

/// Lower ttl.tile_bcast to TTKernel unary_bcast_init + unary_bcast.
/// Supports shape expansion where input CB has different shape than output CB.
struct TTLTileBcastToTTKernel : OpConversionPattern<TileBcastOp> {
  using OpConversionPattern<TileBcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileBcastOp op, TileBcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(op, "op not in function");
    }

    auto *typeConverter = this->getTypeConverter();
    auto inCB =
        lookupAndConvertCB(op.getInput(), funcOp, typeConverter, rewriter, loc);
    if (failed(inCB)) {
      return rewriter.notifyMatchFailure(op, "cannot find/convert input CB");
    }

    auto outCB = lookupAndConvertCB(op.getOutput(), funcOp, typeConverter,
                                    rewriter, loc);
    if (failed(outCB)) {
      // Use the output CB index annotation from ttl-annotate-cb-associations.
      if (auto cbIdx =
              op->getAttrOfType<IntegerAttr>(kBcastOutputCBIndexAttrName)) {
        Value cb;
        funcOp->walk([&](BindCBOp bindOp) {
          if (bindOp.getCbIndexAttr().getInt() == cbIdx.getInt()) {
            cb = bindOp.getResult();
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (cb) {
          outCB =
              utils::convertTTLCBToTTKernel(cb, rewriter, loc, typeConverter);
        }
      }
      if (failed(outCB)) {
        return rewriter.notifyMatchFailure(op, "cannot find/convert output CB");
      }
    }

    // Get DST index from attribute (assigned by TTLAssignDST pass).
    auto dstIdxAttr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    int64_t dstIdxVal = dstIdxAttr.getInt();
    Value dstIdx = arith::ConstantIndexOp::create(rewriter, loc, dstIdxVal);

    // Get input CB tile index.
    // For shape expansion (input CB smaller than output), use broadcast-aware
    // CB tile index from the tensor.extract that produced this operand.
    // Works for both shape-expansion and same-shape bcast -- the extract
    // indices encode the bcast map application from loop lowering.
    auto inCBIdxResult = computeCBTileIndex(op.getInput(), rewriter, loc);
    if (failed(inCBIdxResult)) {
      return rewriter.notifyMatchFailure(
          op, "cannot compute input CB tile index from tensor.extract");
    }
    Value inCBIdx = *inCBIdxResult;

    auto ttkAttr = convertBcastType(op.getBcastType());

    // Emit compute op (init inserted by ttkernel-insert-inits pass).
    auto bcastOp = ttk::UnaryBcastTileOp::create(rewriter, loc, *inCB, inCBIdx,
                                                 dstIdx, ttkAttr);

    // Propagate output CB index for per-op init insertion.
    if (auto cbIdxAttr =
            op->getAttrOfType<IntegerAttr>(kBcastOutputCBIndexAttrName)) {
      bcastOp->setAttr(kBcastOutputCBIndexAttrName, cbIdxAttr);
    }

    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CB-input tile op lowering helper
//===----------------------------------------------------------------------===//

/// Common setup for tile ops that read from a CB: resolves the function,
/// input CB, DST index, and CB tile index. Returns failure if any step fails.
struct CBInputTileOpSetup {
  Value inCB;
  Value dstIdx;
  Value inCBIdx;

  static FailureOr<CBInputTileOpSetup>
  create(Operation *op, Value input, ConversionPatternRewriter &rewriter,
         const TypeConverter *typeConverter) {
    Location loc = op->getLoc();
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(op, "op not in function");
    }

    auto cb = lookupAndConvertCB(input, funcOp, typeConverter, rewriter, loc);
    if (failed(cb)) {
      return rewriter.notifyMatchFailure(op, "cannot find/convert input CB");
    }

    auto dstIdxAttr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    Value dstIdx =
        arith::ConstantIndexOp::create(rewriter, loc, dstIdxAttr.getInt());

    // Compute CB tile index by tracing back to the tensor.extract that
    // produced this operand. The extract indices encode the indexing map
    // application from loop lowering -- no map annotations needed.
    auto cbIdx = computeCBTileIndex(input, rewriter, loc);
    if (failed(cbIdx)) {
      return rewriter.notifyMatchFailure(
          op, "cannot compute CB tile index: operand does not trace to "
              "tensor.extract (loop lowering must run first)");
    }

    return CBInputTileOpSetup{*cb, dstIdx, *cbIdx};
  }
};

//===----------------------------------------------------------------------===//
// Reduce Tile Lowering
//===----------------------------------------------------------------------===//

/// Lower ttl.tile_reduce to ttkernel.reduce_tile.
struct TTLTileReduceToTTKernel : OpConversionPattern<TileReduceOp> {
  bool fullFp32;

  TTLTileReduceToTTKernel(TypeConverter &converter, MLIRContext *ctx,
                          bool fullFp32)
      : OpConversionPattern<TileReduceOp>(converter, ctx), fullFp32(fullFp32) {}

  LogicalResult
  matchAndRewrite(TileReduceOp op, TileReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto setup = CBInputTileOpSetup::create(op, op.getInput(), rewriter,
                                            this->getTypeConverter());
    if (failed(setup)) {
      return failure();
    }

    // Scaler CB lookup.
    auto funcOp = op->getParentOfType<func::FuncOp>();
    auto scalerCB =
        lookupAndConvertCB(op.getScaler(), funcOp, this->getTypeConverter(),
                           rewriter, op.getLoc());
    if (failed(scalerCB)) {
      return rewriter.notifyMatchFailure(op, "cannot find/convert scaler CB");
    }

    // Scaler tile index is always 0.
    // TODO: Support scalar constants via fill in a follow-up PR.
    Value scalerIdx = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);

    // TTL and TTKernel ReduceType share the same underlying values.
    auto ttkReduceType =
        static_cast<ttk::ReduceType>(static_cast<uint32_t>(op.getReduceType()));
    static_assert(static_cast<int>(ttl::ReduceType::Sum) ==
                          static_cast<int>(ttk::ReduceType::Sum) &&
                      static_cast<int>(ttl::ReduceType::Max) ==
                          static_cast<int>(ttk::ReduceType::Max),
                  "TTL and TTKernel ReduceType enum values must match");
    auto reduceOp = ttk::ReduceTileOp::create(
        rewriter, op.getLoc(), setup->inCB, *scalerCB, setup->inCBIdx,
        scalerIdx, setup->dstIdx,
        ttk::ReduceTypeAttr::get(op.getContext(), ttkReduceType),
        ttk::ReduceDimAttr::get(op.getContext(), op.getReduceDim()));
    if (fullFp32) {
      reduceOp->setAttr("full_fp32", rewriter.getUnitAttr());
    }

    // Propagate output CB index for per-op init insertion.
    if (auto cbIdxAttr =
            op->getAttrOfType<IntegerAttr>(kReduceOutputCBIndexAttrName)) {
      reduceOp->setAttr(kReduceOutputCBIndexAttrName, cbIdxAttr);
    }

    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Transpose Tile Lowering
//===----------------------------------------------------------------------===//

/// Lower ttl.tile_transpose to ttkernel.transpose_wh_tile.
struct TTLTileTransposeToTTKernel : OpConversionPattern<TileTransposeOp> {
  using OpConversionPattern<TileTransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileTransposeOp op, TileTransposeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto setup = CBInputTileOpSetup::create(op, op.getInput(), rewriter,
                                            this->getTypeConverter());
    if (failed(setup)) {
      return failure();
    }

    auto transposeOp = ttk::TransposeTileOp::create(
        rewriter, op.getLoc(), setup->inCB, setup->inCBIdx, setup->dstIdx);

    // Propagate output CB index for per-op init insertion.
    if (auto cbIdxAttr =
            op->getAttrOfType<IntegerAttr>(kTransposeOutputCBIndexAttrName)) {
      transposeOp->setAttr(kTransposeOutputCBIndexAttrName, cbIdxAttr);
    }

    rewriter.replaceOp(op, adaptor.getInput());
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
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  using TTL_OP##TileLowering =                                                 \
      TTLTileMaxToTTKernel<TILE_OP, ttk::TTK_INIT, ttk::TTK_COMPUTE>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

// Generate type aliases for FPU binary tile op lowerings
#define TTL_FPU_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)         \
  using TTL_OP##FPUTileLowering =                                              \
      TTLTileBinaryFPUToTTKernel<TILE_OP, ttk::TTK_INIT, ttk::TTK_COMPUTE>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

//===----------------------------------------------------------------------===//
// Matmul Block Lowering
//===----------------------------------------------------------------------===//

/// Lower ttl.tile_matmul_block to ttkernel.matmul_block.
/// Block dimensions (rt, ct, kt) are derived from the enclosing
/// ttl.compute's operand tensor shapes.
struct TTLTileMatmulBlockToTTKernel : OpConversionPattern<TileMatmulBlockOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileMatmulBlockOp op, TileMatmulBlockOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(op, "op not in function");
    }
    auto *typeConverter = this->getTypeConverter();

    // Look up CBs for lhs (A) and rhs (B).
    auto lhsCB =
        lookupAndConvertCB(op.getLhs(), funcOp, typeConverter, rewriter, loc);
    auto rhsCB =
        lookupAndConvertCB(op.getRhs(), funcOp, typeConverter, rewriter, loc);
    if (failed(lhsCB) || failed(rhsCB)) {
      return rewriter.notifyMatchFailure(
          op, "cannot find/convert input CBs for matmul_block");
    }

    // DST output index from attribute (assigned by TTLAssignDST).
    auto dstIdxAttr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    Value dstIdx =
        arith::ConstantIndexOp::create(rewriter, loc, dstIdxAttr.getInt());

    // Derive block dimensions from the immediate operand shapes. For
    // subblocked computes, these are the subblock dimensions (e.g., 1x1
    // for a subblocked 3x1 lhs), not the full block dimensions.
    auto lhsTy = mlir::dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsTy = mlir::dyn_cast<RankedTensorType>(op.getRhs().getType());
    if (!lhsTy || !rhsTy) {
      return rewriter.notifyMatchFailure(
          op, "cannot determine operand tensor shapes for block dimensions");
    }
    // Assumes non-transposed: lhs is [M, K], rhs is [K, N].
    // TODO(#420): support transpose.
    int32_t rt = lhsTy.getDimSize(0); // M
    int32_t ct = rhsTy.getDimSize(1); // N
    int32_t kt = lhsTy.getDimSize(1); // K

    // Starting DFB tile index: 0 when not subblocked (DFB refilled each
    // K-step), or the slice offset when subblocked.
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value in0TileIndex =
        utils::addSliceOffset(op.getLhs(), zero, rewriter, loc);
    Value in1TileIndex =
        utils::addSliceOffset(op.getRhs(), zero, rewriter, loc);

    Value transpose =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value ctVal = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getI32IntegerAttr(ct));
    Value rtVal = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getI32IntegerAttr(rt));
    Value ktVal = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getI32IntegerAttr(kt));

    // Accumulator: emit individual copy_tile ops to load DST before matmul.
    // copy_tile_init is inserted later by ttkernel-insert-inits.
    if (op.getAccumulator()) {
      auto accDFB = lookupAndConvertCB(op.getAccumulator(), funcOp,
                                       typeConverter, rewriter, loc);
      if (failed(accDFB)) {
        return rewriter.notifyMatchFailure(
            op, "cannot find/convert accumulator DFB for matmul_block");
      }

      // Load accumulator tiles from DFB to DST. addSliceOffset converts
      // each local tile index to the global DFB position.
      int32_t ntiles = rt * ct;
      for (int32_t i = 0; i < ntiles; ++i) {
        Value localIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
        Value cbIdx =
            utils::addSliceOffset(op.getAccumulator(), localIdx, rewriter, loc);
        Value dstTileIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
        ttk::CopyTileOp::create(rewriter, loc, *accDFB, cbIdx, dstTileIdx);
      }
    }

    // B stride per K step is the full CB N dimension (not subblock ct).
    // B is [K, N] row-major in the CB; stride between K rows is N.
    int32_t fullN = ct; // default when not subblocked
    Value rhsCBVal = lookupCBByIndex(op.getRhs(), funcOp);
    assert(rhsCBVal && "rhs CB lookup failed after prior successful lookup");
    if (auto ttlCb = mlir::dyn_cast<CircularBufferType>(rhsCBVal.getType())) {
      auto cbShape = ttlCb.getShape();
      if (cbShape.size() == 2) {
        fullN = cbShape[1];
      }
    }

    // Emit matmul_block K loop. Each matmul_block call processes one K step;
    // the caller iterates over K. kt_dim is a configuration parameter that
    // tells the hardware the full K dimension (used by init for stride
    // setup), not the number of tiles processed per call. Each step advances
    // A's tile index by 1 (row-major [M,K]) and B's by fullN (row-major
    // [K,N]).
    if (kt == 1) {
      ttk::MatmulBlockOp::create(rewriter, loc, *lhsCB, *rhsCB, in0TileIndex,
                                 in1TileIndex, dstIdx, transpose, ctVal, rtVal,
                                 ktVal);
    } else {
      Value ub = arith::ConstantIndexOp::create(rewriter, loc, kt);
      Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value fullNIndex = arith::ConstantIndexOp::create(rewriter, loc, fullN);
      auto forOp = scf::ForOp::create(rewriter, loc, zero, ub, step);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());
        Value kIdx = forOp.getInductionVar();
        // A tile index: base + k.
        Value in0Idx = arith::AddIOp::create(rewriter, loc, in0TileIndex, kIdx);
        // B tile index: base + k * fullN.
        Value kTimesFullN =
            arith::MulIOp::create(rewriter, loc, kIdx, fullNIndex);
        Value in1Idx =
            arith::AddIOp::create(rewriter, loc, in1TileIndex, kTimesFullN);
        ttk::MatmulBlockOp::create(rewriter, loc, *lhsCB, *rhsCB, in0Idx,
                                   in1Idx, dstIdx, transpose, ctVal, rtVal,
                                   ktVal);
      }
    }

    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTLTileOpsToTTKernelPatterns(TypeConverter *typeConverter,
                                          RewritePatternSet &patterns,
                                          bool reduceFullFp32) {
  MLIRContext *ctx = patterns.getContext();

  // DST lifecycle ops (1:1 conversion, no operands/results).
  patterns.add<TTLTileRegsAcquireToTTKernel, TTLTileRegsCommitToTTKernel,
               TTLTileRegsWaitToTTKernel, TTLTileRegsReleaseToTTKernel>(ctx);

  // Tile op lowerings - generated from TTLElementwiseOps.def
  // Unary ops (ttl.tile_* → ttkernel.*_tile)
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  patterns.add<TTL_OP##TileLowering>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Binary ops (ttl.tile_* → ttkernel.*_tiles)
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  patterns.add<TTL_OP##TileLowering>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Special binary ops (non-standard lowering template)
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  patterns.add<TTL_OP##TileLowering>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // FPU binary ops (CB -> DST, needs type converter for CB lookup)
#define TTL_FPU_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)         \
  patterns.add<TTL_OP##FPUTileLowering>(*typeConverter, ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Copy ops need the type converter.
  patterns.add<TTLTileCopyToTTKernel>(*typeConverter, ctx);
  patterns.add<TTLCopyDstToTTKernel>(ctx);

  // Bcast ops need the type converter for CB lookup.
  patterns.add<TTLTileBcastToTTKernel>(*typeConverter, ctx);

  // Reduce and transpose ops need the type converter for CB lookup.
  patterns.add<TTLTileReduceToTTKernel>(*typeConverter, ctx, reduceFullFp32);
  patterns.add<TTLTileTransposeToTTKernel>(*typeConverter, ctx);

  // Matmul block needs the type converter for CB lookup.
  patterns.add<TTLTileMatmulBlockToTTKernel>(*typeConverter, ctx);
}

} // namespace mlir::tt::ttl
