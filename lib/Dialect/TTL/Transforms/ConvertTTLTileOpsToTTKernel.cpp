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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
  auto inShape = getCBTileGridShape(input, funcOp);
  auto outShape = getCBTileGridShape(output, funcOp);
  if (!inShape || !outShape) {
    return false;
  }

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
/// For broadcast ops where input CB is smaller than output CB:
///   - Col broadcast (dims=[1]): input has 1 col, index = row_idx
///   - Row broadcast (dims=[0]): input has 1 row, index = col_idx
///   - Scalar broadcast (dims=[0,1]): input is (1,1), index = 0
static Value computeBcastShapeExpansionIndex(ttl::TileBcastOp op,
                                             OpBuilder &builder, Location loc) {
  SmallVector<scf::ForOp> loops = utils::collectEnclosingLoops(op);

  // Expect at least 2 loops for 2D tile iteration.
  // Loops are collected innermost-first: loops[0]=col, loops[1]=row.
  if (loops.size() < 2) {
    return builder.create<arith::ConstantIndexOp>(loc, 0);
  }

  Value colIdx = loops[0].getInductionVar();
  Value rowIdx = loops[1].getInductionVar();

  // Determine index based on broadcast type.
  auto bcastType = op.getBcastType();
  switch (bcastType) {
  case ttl::BcastType::Col:
    // Input has shape (N, 1): index = row_idx.
    return rowIdx;
  case ttl::BcastType::Row:
    // Input has shape (1, M): index = col_idx.
    return colIdx;
  case ttl::BcastType::Scalar:
    // Input has shape (1, 1): index = 0.
    return builder.create<arith::ConstantIndexOp>(loc, 0);
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
      // After loop lowering in fused blocks, the output operand traces to
      // iter_args. Find the output CB from the init_sfpu op in the function.
      funcOp->walk([&](InitSFPUOp initOp) {
        outCB = utils::convertTTLCBToTTKernel(initOp.getOcb(), rewriter, loc,
                                              typeConverter);
        return WalkResult::interrupt();
      });
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
    Value dstIdx = rewriter.create<arith::ConstantIndexOp>(loc, dstIdxVal);

    // Get input CB tile index.
    // For shape expansion (input CB smaller than output), use broadcast-aware
    // indexing. Otherwise, use linearized index for same-shape CB iteration.
    Value inCBIdx;
    if (hasBcastShapeExpansion(op.getInput(), op.getOutput(), op.getBcastType(),
                               funcOp)) {
      inCBIdx = computeBcastShapeExpansionIndex(op, rewriter, loc);
    } else {
      inCBIdx =
          utils::computeCBTileIndexFromLoops(op, rewriter, /*cbShapeRank=*/2);
    }

    auto ttkAttr = convertBcastType(op.getBcastType());

    rewriter.create<ttk::UnaryBcastInitOp>(loc, *inCB, *outCB, ttkAttr);
    rewriter.create<ttk::UnaryBcastTileOp>(loc, *inCB, inCBIdx, dstIdx,
                                           ttkAttr);

    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Matmul Tile Op Lowering
//===----------------------------------------------------------------------===//

/// Lower ttl.tile_matmul to TTKernel mm_init_short + matmul_tiles.
/// Reads A and B from CBs, accumulates into DST.
/// Handles K-dimension accumulation by emitting a loop over K tiles.
struct TTLTileMatmulToTTKernel : OpConversionPattern<TileMatmulOp> {
  using OpConversionPattern<TileMatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileMatmulOp op, TileMatmulOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(op, "op not in function");
    }

    auto *typeConverter = this->getTypeConverter();
    auto aCB =
        lookupAndConvertCB(op.getA(), funcOp, typeConverter, rewriter, loc);
    if (failed(aCB)) {
      return rewriter.notifyMatchFailure(op, "cannot find/convert A CB");
    }

    auto bCB =
        lookupAndConvertCB(op.getB(), funcOp, typeConverter, rewriter, loc);
    if (failed(bCB)) {
      return rewriter.notifyMatchFailure(op, "cannot find/convert B CB");
    }

    // Get DST index from attribute (assigned by TTLAssignDST pass).
    auto dstIdxAttr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    int64_t dstIdxVal = dstIdxAttr.getInt();
    Value dstIdx = rewriter.create<arith::ConstantIndexOp>(loc, dstIdxVal);

    // Get CB shapes to determine K dimension.
    // A has shape [M, K], B has shape [K, N].
    auto aShape = getCBTileGridShape(op.getA(), funcOp);
    auto bShape = getCBTileGridShape(op.getB(), funcOp);
    if (!aShape || !bShape) {
      return rewriter.notifyMatchFailure(op, "cannot determine CB shapes");
    }

    int64_t aK = aShape->second; // A is [M, K]
    int64_t bK = bShape->first;  // B is [K, N]
    int64_t bN = bShape->second; // B is [K, N]

    if (aK != bK) {
      return rewriter.notifyMatchFailure(
          op, "K dimension mismatch between A and B");
    }
    int64_t kDim = aK;

    // Get M, N indices from enclosing loops (if any).
    SmallVector<scf::ForOp> loops = utils::collectEnclosingLoops(op);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value mIdx = zero;
    Value nIdx = zero;

    if (loops.size() >= 2) {
      // loops[0] = n (innermost), loops[1] = m (outer)
      nIdx = loops[0].getInductionVar();
      mIdx = loops[1].getInductionVar();
    } else if (loops.size() == 1) {
      nIdx = loops[0].getInductionVar();
    }

    // Emit mm_init_short before the K loop.
    Value transpose =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    rewriter.create<ttk::MatmulInitShortOp>(loc, *aCB, *bCB, transpose);

    // Create K loop if K > 1, otherwise just emit single matmul_tiles.
    if (kDim > 1) {
      Value kStart = zero;
      Value kEnd = rewriter.create<arith::ConstantIndexOp>(loc, kDim);
      Value kStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      auto kLoop = rewriter.create<scf::ForOp>(loc, kStart, kEnd, kStep);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kIdx = kLoop.getInductionVar();

      // A index = m * K + k
      Value aKVal = rewriter.create<arith::ConstantIndexOp>(loc, kDim);
      Value aIdxMulK = rewriter.create<arith::MulIOp>(loc, mIdx, aKVal);
      Value aIdx = rewriter.create<arith::AddIOp>(loc, aIdxMulK, kIdx);

      // B index = k * N + n
      Value bNVal = rewriter.create<arith::ConstantIndexOp>(loc, bN);
      Value bIdxMulN = rewriter.create<arith::MulIOp>(loc, kIdx, bNVal);
      Value bIdx = rewriter.create<arith::AddIOp>(loc, bIdxMulN, nIdx);

      rewriter.create<ttk::MatmulTilesOp>(loc, *aCB, *bCB, aIdx, bIdx, dstIdx);
      rewriter.setInsertionPointAfter(kLoop);
    } else {
      // K=1: simple case, just compute indices directly.
      // A index = m * 1 + 0 = m
      // B index = 0 * N + n = n
      Value aIdx = mIdx;
      Value bIdx = nIdx;
      rewriter.create<ttk::MatmulTilesOp>(loc, *aCB, *bCB, aIdx, bIdx, dstIdx);
    }

    rewriter.replaceOp(op, adaptor.getA());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reduce Tile Op Lowering
//===----------------------------------------------------------------------===//

/// Convert TTL ReduceType to TTKernel ReduceType.
static ttk::ReduceType convertReduceType(ttl::ReduceType ttlType) {
  switch (ttlType) {
  case ttl::ReduceType::Sum:
    return ttk::ReduceType::Sum;
  case ttl::ReduceType::Max:
    return ttk::ReduceType::Max;
  }
  llvm_unreachable("unknown ReduceType");
}

/// Convert TTL ReduceDim to TTKernel ReduceDim.
static ttk::ReduceDim convertReduceDim(ttl::ReduceDim ttlDim) {
  switch (ttlDim) {
  case ttl::ReduceDim::Row:
    return ttk::ReduceDim::Row;
  case ttl::ReduceDim::Col:
    return ttk::ReduceDim::Col;
  case ttl::ReduceDim::Scalar:
    return ttk::ReduceDim::Scalar;
  }
  llvm_unreachable("unknown ReduceDim");
}

/// Lower ttl.tile_reduce to TTKernel reduce_init + reduce_tile.
/// Reads from input CB and scaler CB, writes to DST.
/// Handles multi-tile accumulation by emitting loops over tiles to reduce.
struct TTLTileReduceToTTKernel : OpConversionPattern<TileReduceOp> {
  using OpConversionPattern<TileReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileReduceOp op, TileReduceOp::Adaptor adaptor,
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

    auto scalerCB = lookupAndConvertCB(op.getScaler(), funcOp, typeConverter,
                                       rewriter, loc);
    if (failed(scalerCB)) {
      return rewriter.notifyMatchFailure(op, "cannot find/convert scaler CB");
    }

    auto outCB = lookupAndConvertCB(op.getOutput(), funcOp, typeConverter,
                                    rewriter, loc);
    if (failed(outCB)) {
      return rewriter.notifyMatchFailure(op, "cannot find/convert output CB");
    }

    // Get DST index from attribute (assigned by TTLAssignDST pass).
    auto dstIdxAttr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    int64_t dstIdxVal = dstIdxAttr.getInt();
    Value dstIdx = rewriter.create<arith::ConstantIndexOp>(loc, dstIdxVal);

    // Get input CB shape to determine reduction dimensions.
    auto inShape = getCBTileGridShape(op.getInput(), funcOp);
    if (!inShape) {
      return rewriter.notifyMatchFailure(op, "cannot determine input CB shape");
    }
    int64_t inRows = inShape->first;
    int64_t inCols = inShape->second;

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Get M, N indices from enclosing loops (current output tile position).
    SmallVector<scf::ForOp> loops = utils::collectEnclosingLoops(op);
    Value mIdx = zero;
    Value nIdx = zero;
    if (loops.size() >= 2) {
      nIdx = loops[0].getInductionVar();
      mIdx = loops[1].getInductionVar();
    } else if (loops.size() == 1) {
      nIdx = loops[0].getInductionVar();
    }

    auto ttkReduceType = convertReduceType(op.getReduceType());
    auto ttkReduceDim = convertReduceDim(op.getReduceDim());

    // Emit reduce_init before accumulation loops.
    rewriter.create<ttk::ReduceInitOp>(loc, *inCB, *scalerCB, *outCB,
                                       ttkReduceType, ttkReduceDim);

    auto reduceDim = op.getReduceDim();

    if (reduceDim == ttl::ReduceDim::Scalar) {
      // Scalar reduce: loop over all M*N tiles and accumulate.
      if (inRows > 1 || inCols > 1) {
        Value rowEnd = rewriter.create<arith::ConstantIndexOp>(loc, inRows);
        Value colEnd = rewriter.create<arith::ConstantIndexOp>(loc, inCols);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto rowLoop = rewriter.create<scf::ForOp>(loc, zero, rowEnd, one);
        rewriter.setInsertionPointToStart(rowLoop.getBody());
        Value rowIdx = rowLoop.getInductionVar();

        auto colLoop = rewriter.create<scf::ForOp>(loc, zero, colEnd, one);
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value colIdx = colLoop.getInductionVar();

        // inCBIdx = rowIdx * inCols + colIdx
        Value inColsVal = rewriter.create<arith::ConstantIndexOp>(loc, inCols);
        Value rowMulCols = rewriter.create<arith::MulIOp>(loc, rowIdx, inColsVal);
        Value inCBIdx = rewriter.create<arith::AddIOp>(loc, rowMulCols, colIdx);

        rewriter.create<ttk::ReduceTileOp>(loc, *inCB, *scalerCB, inCBIdx, zero,
                                           dstIdx, ttkReduceType, ttkReduceDim);
        rewriter.setInsertionPointAfter(rowLoop);
      } else {
        // Single tile: no loop needed.
        rewriter.create<ttk::ReduceTileOp>(loc, *inCB, *scalerCB, zero, zero,
                                           dstIdx, ttkReduceType, ttkReduceDim);
      }
    } else if (reduceDim == ttl::ReduceDim::Row) {
      // Row reduce (sum across columns): for current row, loop over columns.
      if (inCols > 1) {
        Value colEnd = rewriter.create<arith::ConstantIndexOp>(loc, inCols);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto colLoop = rewriter.create<scf::ForOp>(loc, zero, colEnd, one);
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value colIdx = colLoop.getInductionVar();

        // inCBIdx = mIdx * inCols + colIdx
        Value inColsVal = rewriter.create<arith::ConstantIndexOp>(loc, inCols);
        Value rowMulCols = rewriter.create<arith::MulIOp>(loc, mIdx, inColsVal);
        Value inCBIdx = rewriter.create<arith::AddIOp>(loc, rowMulCols, colIdx);

        rewriter.create<ttk::ReduceTileOp>(loc, *inCB, *scalerCB, inCBIdx, zero,
                                           dstIdx, ttkReduceType, ttkReduceDim);
        rewriter.setInsertionPointAfter(colLoop);
      } else {
        // Single column: no loop needed.
        Value inCBIdx = mIdx;
        rewriter.create<ttk::ReduceTileOp>(loc, *inCB, *scalerCB, inCBIdx, zero,
                                           dstIdx, ttkReduceType, ttkReduceDim);
      }
    } else if (reduceDim == ttl::ReduceDim::Col) {
      // Col reduce (sum across rows): for current column, loop over rows.
      if (inRows > 1) {
        Value rowEnd = rewriter.create<arith::ConstantIndexOp>(loc, inRows);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto rowLoop = rewriter.create<scf::ForOp>(loc, zero, rowEnd, one);
        rewriter.setInsertionPointToStart(rowLoop.getBody());
        Value rowIdx = rowLoop.getInductionVar();

        // inCBIdx = rowIdx * inCols + nIdx
        Value inColsVal = rewriter.create<arith::ConstantIndexOp>(loc, inCols);
        Value rowMulCols = rewriter.create<arith::MulIOp>(loc, rowIdx, inColsVal);
        Value inCBIdx = rewriter.create<arith::AddIOp>(loc, rowMulCols, nIdx);

        rewriter.create<ttk::ReduceTileOp>(loc, *inCB, *scalerCB, inCBIdx, zero,
                                           dstIdx, ttkReduceType, ttkReduceDim);
        rewriter.setInsertionPointAfter(rowLoop);
      } else {
        // Single row: no loop needed.
        Value inCBIdx = nIdx;
        rewriter.create<ttk::ReduceTileOp>(loc, *inCB, *scalerCB, inCBIdx, zero,
                                           dstIdx, ttkReduceType, ttkReduceDim);
      }
    }

    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Transpose Tile Op Lowering
//===----------------------------------------------------------------------===//

/// Lower ttl.tile_transpose to TTKernel transpose_wh_init + transpose_wh_tile.
/// Reads from input CB, writes to DST.
/// For multi-tile transpose, computes transposed input CB index:
/// Output position (i, j) reads from input position (j, i).
struct TTLTileTransposeToTTKernel : OpConversionPattern<TileTransposeOp> {
  using OpConversionPattern<TileTransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileTransposeOp op, TileTransposeOp::Adaptor adaptor,
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
      funcOp->walk([&](InitSFPUOp initOp) {
        outCB = utils::convertTTLCBToTTKernel(initOp.getOcb(), rewriter, loc,
                                              typeConverter);
        return WalkResult::interrupt();
      });
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
    Value dstIdx = rewriter.create<arith::ConstantIndexOp>(loc, dstIdxVal);

    // Get input CB shape to compute transposed index.
    auto inShape = getCBTileGridShape(op.getInput(), funcOp);
    if (!inShape) {
      return rewriter.notifyMatchFailure(op, "cannot determine input CB shape");
    }
    int64_t inCols = inShape->second; // Input is [M, N], so N columns

    // Compute transposed input CB index.
    // Loop iterates over output shape [N, M]. For output position (i, j),
    // we read from input position (j, i).
    // Input CB index = j * N + i (linearized for input shape [M, N]).
    SmallVector<scf::ForOp> loops = utils::collectEnclosingLoops(op);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value inCBIdx = zero;

    if (loops.size() >= 2) {
      // loops[0] = j (innermost, cols), loops[1] = i (outer, rows)
      Value colIdx = loops[0].getInductionVar(); // j
      Value rowIdx = loops[1].getInductionVar(); // i
      // inCBIdx = colIdx * inCols + rowIdx (transposed indexing)
      Value inColsVal = rewriter.create<arith::ConstantIndexOp>(loc, inCols);
      Value colMulN = rewriter.create<arith::MulIOp>(loc, colIdx, inColsVal);
      inCBIdx = rewriter.create<arith::AddIOp>(loc, colMulN, rowIdx);
    } else if (loops.size() == 1) {
      inCBIdx = loops[0].getInductionVar();
    }

    rewriter.create<ttk::TransposeInitOp>(loc, *inCB, *outCB);
    rewriter.create<ttk::TransposeTileOp>(loc, *inCB, inCBIdx, dstIdx);

    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Power Tile Op Lowering
//===----------------------------------------------------------------------===//

/// Lower ttl.tile_power to TTKernel power_tile_init + power_tile.
/// Operates in-place in DST with an integer exponent.
struct TTLTilePowerToTTKernel : OpConversionPattern<TilePowerOp> {
  using OpConversionPattern<TilePowerOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TilePowerOp op, TilePowerOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dstIdxAttr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    int64_t dstIdx = dstIdxAttr.getInt();
    Value dstIdxVal = rewriter.create<arith::ConstantIndexOp>(loc, dstIdx);

    // Get the exponent from the op's attribute.
    Value exponent = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(op.getExponent()));

    rewriter.create<ttk::PowerTileInitOp>(loc);
    rewriter.create<ttk::PowUnaryTileOp>(loc, dstIdxVal, exponent);

    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Where Tile Op Lowering
//===----------------------------------------------------------------------===//

/// Lower ttl.tile_where to TTKernel where_tile_init + where_tile.
/// Ternary DST-based op: cond ? true : false.
struct TTLTileWhereToTTKernel : OpConversionPattern<TileWhereOp> {
  using OpConversionPattern<TileWhereOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileWhereOp op, TileWhereOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dstIdxAttr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    int64_t odstIdx = dstIdxAttr.getInt();

    auto condIdxOpt = getDstIndexFromValue(op.getCondition());
    auto trueIdxOpt = getDstIndexFromValue(op.getTrueValue());
    auto falseIdxOpt = getDstIndexFromValue(op.getFalseValue());

    if (!condIdxOpt) {
      return rewriter.notifyMatchFailure(
          op, "failed to extract dst_idx from condition operand");
    }
    if (!trueIdxOpt) {
      return rewriter.notifyMatchFailure(
          op, "failed to extract dst_idx from true_value operand");
    }
    if (!falseIdxOpt) {
      return rewriter.notifyMatchFailure(
          op, "failed to extract dst_idx from false_value operand");
    }

    Value condIdx = rewriter.create<arith::ConstantIndexOp>(loc, *condIdxOpt);
    Value trueIdx = rewriter.create<arith::ConstantIndexOp>(loc, *trueIdxOpt);
    Value falseIdx = rewriter.create<arith::ConstantIndexOp>(loc, *falseIdxOpt);
    Value odst = rewriter.create<arith::ConstantIndexOp>(loc, odstIdx);

    rewriter.create<ttk::WhereTileInitOp>(loc);
    rewriter.create<ttk::WhereTileOp>(loc, condIdx, trueIdx, falseIdx, odst);

    rewriter.replaceOp(op, adaptor.getCondition());
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

  // Binary ops (ttl.tile_* → ttkernel.*_tiles)
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  patterns.add<TTL_OP##TileLowering>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Special binary ops (non-standard lowering template)
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  patterns.add<TTL_OP##TileLowering>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Copy ops need the type converter.
  patterns.add<TTLTileCopyToTTKernel>(*typeConverter, ctx);
  patterns.add<TTLCopyDstToTTKernel>(ctx);

  // CB -> DST ops with attribute need the type converter.
  patterns.add<TTLTileBcastToTTKernel>(*typeConverter, ctx);
  patterns.add<TTLTileMatmulToTTKernel>(*typeConverter, ctx);
  patterns.add<TTLTileReduceToTTKernel>(*typeConverter, ctx);
  patterns.add<TTLTileTransposeToTTKernel>(*typeConverter, ctx);

  // DST-based ops.
  patterns.add<TTLTilePowerToTTKernel>(ctx);
  patterns.add<TTLTileWhereToTTKernel>(ctx);

  // TODO(#124): Add DST lifecycle wrapper pattern for loop iterations
  // (acquire/commit/wait/release + copy_tile/pack_tile)
}

} // namespace mlir::tt::ttl
