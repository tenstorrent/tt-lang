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

/// Trace a tile-level op operand back through tensor.extract, extract_slice,
/// and unrealized casts to find the source tensor's shape.
/// Returns std::nullopt if a ranked tensor cannot be reached.
static std::optional<SmallVector<int64_t>>
getOperandTensorShape(Value operand) {
  Value tensor = operand;
  if (auto extract = operand.getDefiningOp<tensor::ExtractOp>()) {
    tensor = extract.getTensor();
  }
  while (auto slice = tensor.getDefiningOp<tensor::ExtractSliceOp>()) {
    tensor = slice.getSource();
  }
  while (auto cast = tensor.getDefiningOp<UnrealizedConversionCastOp>()) {
    // Stop if we already have a tensor type -- don't follow casts past it.
    if (mlir::isa<RankedTensorType>(tensor.getType())) {
      break;
    }
    if (cast.getInputs().size() == 1) {
      tensor = cast.getInputs().front();
    } else {
      break;
    }
  }
  if (auto tensorTy = mlir::dyn_cast<RankedTensorType>(tensor.getType())) {
    return SmallVector<int64_t>(tensorTy.getShape());
  }
  return std::nullopt;
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

    // Verify both CBs have the same number of tiles, which is required
    // for using the same linearized tile index for both operands.
    auto lhsCBTy = mlir::cast<ttk::CBType>(lhsCB->getType());
    auto rhsCBTy = mlir::cast<ttk::CBType>(rhsCB->getType());
    if (lhsCBTy.getNumTiles() != rhsCBTy.getNumTiles()) {
      return rewriter.notifyMatchFailure(
          op, llvm::Twine("FPU binary requires CBs with matching tile counts; "
                          "lhs has ") +
                  llvm::Twine(lhsCBTy.getNumTiles()) + " tiles, rhs has " +
                  llvm::Twine(rhsCBTy.getNumTiles()));
    }

    // CB tile index from enclosing loops. The same index is used for
    // lhs and rhs because TTLAssignDST only marks ops as FPU-eligible
    // when both operands have identical indexing maps (identity).
    auto operandShape = getOperandTensorShape(op.getLhs());
    if (!operandShape) {
      return rewriter.notifyMatchFailure(
          op, "cannot determine operand tensor shape for CB indexing");
    }
    AffineMap identity = AffineMap::getMultiDimIdentityMap(
        operandShape->size(), rewriter.getContext());
    auto cbIdx =
        utils::computeCBTileIndex(op, rewriter, identity, *operandShape,
                                  *operandShape, operandShape->size());
    if (failed(cbIdx)) {
      return failure();
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
    ValueRange srcIndices = adaptor.getSrcIndices();
    if (srcIndices.empty()) {
      return op.emitError("copy_tile has no src_indices; "
                          "ttl-lower-to-loops must run first");
    }
    auto srcShape = getOperandTensorShape(op.getSrc());
    if (!srcShape) {
      return rewriter.notifyMatchFailure(
          op, "cannot determine source tensor shape for linearization");
    }
    Value flatSrcIndex = affine::AffineLinearizeIndexOp::create(
        rewriter, loc, srcIndices, *srcShape);

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

    // Emit copy_dest_values(dst0, dst1): copies DST[dst1] → DST[dst0].
    ttk::CopyDestValuesOp::create(rewriter, loc, dstIdx, srcIdx);

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

/// Get the CB tile grid shape from an operand by tracing to the tensor type.
/// After loop lowering, operands come from tensor.extract, so we trace back
/// to find the tensor and extract its shape (in tiles).
/// Returns std::nullopt if the shape cannot be determined.
static std::optional<std::pair<int64_t, int64_t>>
getCBTileGridShape(Value operand, func::FuncOp funcOp) {
  // First, try to get shape from TTL CB type.
  Value cb = lookupCBByIndex(operand, funcOp);
  if (cb) {
    if (auto ttlCb = mlir::dyn_cast<CircularBufferType>(cb.getType())) {
      auto shape = ttlCb.getShape();
      if (shape.size() == 2) {
        return std::make_pair(shape[0], shape[1]);
      }
    }
  }

  // If that fails, try to extract shape from the tensor type.
  // After loop lowering, the operand comes from tensor.extract.
  Value tensor = operand;
  if (auto extract = operand.getDefiningOp<tensor::ExtractOp>()) {
    tensor = extract.getTensor();
  }

  // Trace through unrealized conversion casts.
  tensor = traceUnrealizedCasts(tensor);

  // Check if we have a ranked tensor with a 2D tile shape.
  if (auto tensorTy = mlir::dyn_cast<RankedTensorType>(tensor.getType())) {
    if (tensorTy.getRank() == 2) {
      return std::make_pair(tensorTy.getDimSize(0), tensorTy.getDimSize(1));
    }
  }

  return std::nullopt;
}

/// Check if broadcast has shape expansion (input CB smaller than output CB).
/// Returns true if the input CB is reduced on the broadcast dimension(s).
static bool hasBcastShapeExpansion(Value input, Value output,
                                   ttl::BcastType bcastType,
                                   func::FuncOp funcOp) {
  // TTLAnnotateCBAssociations guarantees bcast operands have attached CBs
  // with 2D tile grid shapes.
  auto inShape = getCBTileGridShape(input, funcOp);
  auto outShape = getCBTileGridShape(output, funcOp);
  assert(inShape && outShape &&
         "expected 2D tile grid shapes for broadcast operands");

  int64_t inRows = inShape->first;
  int64_t inCols = inShape->second;
  int64_t outRows = outShape->first;
  int64_t outCols = outShape->second;

  switch (bcastType) {
  case ttl::BcastType::Col:
    // Col broadcast: input has fewer cols than output.
    return inCols < outCols;
  case ttl::BcastType::Row:
    // Row broadcast: input has fewer rows than output.
    return inRows < outRows;
  case ttl::BcastType::Scalar:
    // Scalar broadcast: input is smaller in both dimensions.
    return (inRows < outRows) || (inCols < outCols);
  }
  return false;
}

/// Compute input CB tile index for broadcast with shape expansion.
///
/// Uses computeCBTileIndex with a broadcast-derived indexing map:
///   - Col broadcast (Nx1 input): map (i,j) -> (i), operand shape [N]
///   - Row broadcast (1xM input): map (i,j) -> (j), operand shape [M]
///   - Scalar broadcast: constant 0
///
/// The iteration domain is 2D (output tile grid). The indexing map projects
/// out the broadcast dimension(s). computeCBTileIndex handles tile loops,
/// subblock loops, and tile offsets.
static FailureOr<Value> computeBcastShapeExpansionIndex(ttl::TileBcastOp op,
                                                        func::FuncOp funcOp,
                                                        OpBuilder &builder,
                                                        Location loc) {
  auto bcastType = op.getBcastType();
  if (bcastType == ttl::BcastType::Scalar) {
    return arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  }

  auto inShape = getCBTileGridShape(op.getInput(), funcOp);
  auto outShape = getCBTileGridShape(op.getOutput(), funcOp);
  assert(inShape && outShape &&
         "expected 2D tile grid shapes for broadcast operands");

  SmallVector<int64_t> iterDomain = {outShape->first, outShape->second};

  // Build the broadcast indexing map and 1D operand shape.
  MLIRContext *ctx = builder.getContext();
  AffineMap bcastMap;
  SmallVector<int64_t> operandShape;
  if (bcastType == ttl::BcastType::Col) {
    // Input is Nx1: only row dimension varies.
    bcastMap = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                              getAffineDimExpr(0, ctx));
    operandShape.push_back(inShape->first);
  } else {
    assert(bcastType == ttl::BcastType::Row);
    // Input is 1xM: only col dimension varies.
    bcastMap = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                              getAffineDimExpr(1, ctx));
    operandShape.push_back(inShape->second);
  }

  // Broadcast iteration is always 2D (row x col tile grid).
  return utils::computeCBTileIndex(op, builder, bcastMap, iterDomain,
                                   operandShape, /*cbShapeRank=*/2);
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
    // indexing. Otherwise, use linearized index for same-shape CB iteration.
    Value inCBIdx;
    if (hasBcastShapeExpansion(op.getInput(), op.getOutput(), op.getBcastType(),
                               funcOp)) {
      auto bcastIdx =
          computeBcastShapeExpansionIndex(op, funcOp, rewriter, loc);
      if (failed(bcastIdx)) {
        return failure();
      }
      inCBIdx = *bcastIdx;
    } else {
      auto inTensorShape = getOperandTensorShape(op.getInput());
      if (!inTensorShape) {
        return rewriter.notifyMatchFailure(
            op, "cannot determine input tensor shape for CB indexing");
      }
      AffineMap identity = AffineMap::getMultiDimIdentityMap(
          inTensorShape->size(), rewriter.getContext());
      auto cbIdx =
          utils::computeCBTileIndex(op, rewriter, identity, *inTensorShape,
                                    *inTensorShape, inTensorShape->size());
      if (failed(cbIdx)) {
        return failure();
      }
      inCBIdx = *cbIdx;
    }

    auto ttkAttr = convertBcastType(op.getBcastType());

    // Emit compute op (init inserted by ttkernel-insert-inits pass).
    auto bcastOp = ttk::UnaryBcastTileOp::create(rewriter, loc, *inCB, inCBIdx,
                                                 dstIdx, ttkAttr);

    // Propagate output CB index so ttkernel-insert-inits can derive the
    // output CB for unary_bcast_init without walking the function.
    if (auto cbIdx =
            op->getAttrOfType<IntegerAttr>(kBcastOutputCBIndexAttrName)) {
      bcastOp->setAttr(kBcastOutputCBIndexAttrName, cbIdx);
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

/// Lower ttl.tile_matmul_block to ttkernel.experimental::matmul_block.
/// Block dimensions (rt, ct, kt, nt) are derived from the enclosing
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

    // Derive block dimensions from operand shapes.
    auto lhsShape = getOperandTensorShape(op.getLhs());
    auto rhsShape = getOperandTensorShape(op.getRhs());
    if (!lhsShape || !rhsShape) {
      return rewriter.notifyMatchFailure(
          op, "cannot determine operand tensor shapes for block dimensions");
    }
    // lhs is [M, K] per K-step (K=1 for block matmul pattern).
    // rhs is [K, N] per K-step (K=1 for block matmul pattern).
    int32_t rt = (*lhsShape)[0]; // M (A row count in tiles)
    int32_t ct = (*rhsShape)[1]; // N (B column count in tiles)
    // nt_dim is the B column stride: the experimental::matmul_block wrapper
    // advances in1_tile_index by nt_dim per K-step. For non-transposed B
    // laid out row-major, one K-step moves past N columns, so nt == ct.
    int32_t nt = ct;

    // CB tile indices are always 0: the CB is popped and refilled each
    // K-step, so the read pointer resets. kt_dim=1 (one K-step per call,
    // matching the proven tt-metal pattern).
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    Value transpose =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value ctVal = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getI32IntegerAttr(ct));
    Value rtVal = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getI32IntegerAttr(rt));
    Value ktVal =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
    Value ntVal = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getI32IntegerAttr(nt));

    // Accumulator: emit rt*ct copy_tile ops to load DST before matmul_block.
    // TODO: Replace with copy_block_matmul_partials(cb, 0, 0, rt*ct) once a
    // TTKernel op exists (tt_metal/hw/inc/api/compute/tile_move_copy.h).
    // Similarly, pack_tile ops should use pack_tile_block (pack.h).
    if (op.getAccumulator()) {
      auto accCB = lookupAndConvertCB(op.getAccumulator(), funcOp,
                                      typeConverter, rewriter, loc);
      if (failed(accCB)) {
        return rewriter.notifyMatchFailure(
            op, "cannot find/convert accumulator CB for matmul_block");
      }

      // Emit rt*ct copy_tile ops: CB tile [i] -> DST[i].
      // copy_tile_init is inserted later by ttkernel-insert-inits.
      int32_t ntiles = rt * ct;
      for (int32_t i = 0; i < ntiles; ++i) {
        Value cbIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
        Value dstTileIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
        ttk::CopyTileOp::create(rewriter, loc, *accCB, cbIdx, dstTileIdx);
      }
    }

    // Emit matmul_block with kt_dim=1 (init inserted by ttkernel-insert-inits).
    ttk::ExperimentalMatmulBlockOp::create(rewriter, loc, *lhsCB, *rhsCB, zero,
                                           zero, dstIdx, transpose, ctVal,
                                           rtVal, ktVal, ntVal);

    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTLTileOpsToTTKernelPatterns(TypeConverter *typeConverter,
                                          RewritePatternSet &patterns) {
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

  // Matmul block needs the type converter for CB lookup.
  patterns.add<TTLTileMatmulBlockToTTKernel>(*typeConverter, ctx);
}

} // namespace mlir::tt::ttl
