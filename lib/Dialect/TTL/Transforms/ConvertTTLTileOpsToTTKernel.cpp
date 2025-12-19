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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#define DEBUG_TYPE "ttl-tile-ops-to-ttkernel"

namespace mlir::tt::ttl {

namespace ttk = mlir::tt::ttkernel;

namespace {

/// Look up a CB using precomputed analysis state. Handles block arguments and
/// tensors (including tensor.extract bases).
static Value lookupCB(Value src, const CopyTileCBState *state) {
  if (!state) {
    return Value();
  }

  // Block argument path.
  if (auto barg = llvm::dyn_cast<BlockArgument>(src)) {
    if (auto it = state->blockArgToCb.find(barg);
        it != state->blockArgToCb.end()) {
      return it->second;
    }
  }

  // Tensor path (including tensor.extract bases).
  Value tensor = src;
  if (auto extract = tensor.getDefiningOp<tensor::ExtractOp>()) {
    tensor = extract.getTensor();
  }
  if (auto it = state->tensorToCb.find(tensor); it != state->tensorToCb.end()) {
    return it->second;
  }

  return Value();
}

//===----------------------------------------------------------------------===//
// DST lifecycle ops
//===----------------------------------------------------------------------===//

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
/// DST index assignment: The ttl-tile-and-assign-dst pass assigns dst_idx
/// attributes to operands. For now, we use simple defaults: src0=0, src1=1,
/// odst=0 (in-place on src0).
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileBinaryToTTKernel : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Use op dst_idx for output; operands default to 0/1 for now.
    auto dstIdxAttr = op->template getAttrOfType<IntegerAttr>(kDstIdxAttrName);
    if (!dstIdxAttr) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }
    int64_t odstIdx = dstIdxAttr.getInt();

    FailureOr<int64_t> src0IdxOrErr =
        getDstIdxRequired(adaptor.getLhs(), rewriter, loc);
    FailureOr<int64_t> src1IdxOrErr =
        getDstIdxRequired(adaptor.getRhs(), rewriter, loc);
    if (failed(src0IdxOrErr) || failed(src1IdxOrErr)) {
      return failure();
    }
    int64_t src0Idx = *src0IdxOrErr;
    int64_t src1Idx = *src1IdxOrErr;

    Value src0 = rewriter.create<arith::ConstantIndexOp>(loc, src0Idx);
    Value src1 = rewriter.create<arith::ConstantIndexOp>(loc, src1Idx);
    Value odst = rewriter.create<arith::ConstantIndexOp>(loc, odstIdx);

    // Emit init + compute ops
    rewriter.create<InitOp>(loc);
    rewriter.create<TTKernelComputeOp>(loc, src0, src1, odst);

    // Replace with lhs (result is in DST[odst], which is typically src0)
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

    FailureOr<int64_t> dst0IdxOrErr =
        getDstIdxRequired(adaptor.getLhs(), rewriter, loc);
    FailureOr<int64_t> dst1IdxOrErr =
        getDstIdxRequired(adaptor.getRhs(), rewriter, loc);
    if (failed(dst0IdxOrErr) || failed(dst1IdxOrErr)) {
      return failure();
    }

    Value dst0 = rewriter.create<arith::ConstantIndexOp>(loc, *dst0IdxOrErr);
    Value dst1 = rewriter.create<arith::ConstantIndexOp>(loc, *dst1IdxOrErr);

    rewriter.create<InitOp>(loc);
    rewriter.create<TTKernelComputeOp>(loc, dst0, dst1);

    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }
};

/// Lower ttl.copy_tile to TTKernel copy_tile_init + copy_tile.
struct TTLTileCopyToTTKernel : OpConversionPattern<CopyTileOp> {
  TTLTileCopyToTTKernel(TypeConverter &tc, MLIRContext *ctx,
                        const CopyTileCBState *state)
      : OpConversionPattern<CopyTileOp>(tc, ctx), cbState(state) {}

  LogicalResult
  matchAndRewrite(CopyTileOp op, CopyTileOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Look up the CB via analysis state.
    Value cb = lookupCB(op.getSrc(), cbState);
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
                        ValueRange{op.getSrc()})
                    .getResult(0);
    rewriter.replaceOp(op, ValueRange{token, tile});
    return success();
  }

  // Analysis state carrying precomputed CB attachments:
  // - blockArgToCb maps ttl.compute block args to their CB
  // - tensorToCb maps tensors (including attach_cb results) to their CB
  const CopyTileCBState *cbState;
};

//===----------------------------------------------------------------------===//
// Unary Tile Op Lowerings (LLVM-style type aliases)
//===----------------------------------------------------------------------===//

using ExpTileLowering =
    TTLTileUnaryToTTKernel<ExpTileOp, ttk::ExpTileInitOp, ttk::ExpTileOp>;
using LogTileLowering =
    TTLTileUnaryToTTKernel<LogTileOp, ttk::LogTileInitOp, ttk::LogTileOp>;
using SqrtTileLowering =
    TTLTileUnaryToTTKernel<SqrtTileOp, ttk::SqrtTileInitOp, ttk::SqrtTileOp>;
using RsqrtTileLowering =
    TTLTileUnaryToTTKernel<RsqrtTileOp, ttk::RsqrtTileInitOp, ttk::RsqrtTileOp>;
using TanhTileLowering =
    TTLTileUnaryToTTKernel<TanhTileOp, ttk::TanhTileInitOp, ttk::TanhTileOp>;
using SigmoidTileLowering =
    TTLTileUnaryToTTKernel<SigmoidTileOp, ttk::SigmoidTileInitOp,
                           ttk::SigmoidTileOp>;
using AbsTileLowering =
    TTLTileUnaryToTTKernel<AbsTileOp, ttk::AbsTileInitOp, ttk::AbsTileOp>;
using NegTileLowering =
    TTLTileUnaryToTTKernel<NegTileOp, ttk::NegativeTileInitOp,
                           ttk::NegativeTileOp>;
using ReluTileLowering =
    TTLTileUnaryToTTKernel<ReluTileOp, ttk::ReluTileInitOp, ttk::ReluTileOp>;

//===----------------------------------------------------------------------===//
// Binary Tile Op Lowerings
//===----------------------------------------------------------------------===//

using AddTileLowering =
    TTLTileBinaryToTTKernel<AddTileOp, ttk::AddBinaryTilesInitOp,
                            ttk::AddBinaryTilesOp>;
using SubTileLowering =
    TTLTileBinaryToTTKernel<SubTileOp, ttk::SubBinaryTilesInitOp,
                            ttk::SubBinaryTilesOp>;
using MulTileLowering =
    TTLTileBinaryToTTKernel<MulTileOp, ttk::MulBinaryTilesInitOp,
                            ttk::MulBinaryTilesOp>;
using MaxTileLowering =
    TTLTileMaxToTTKernel<MaxTileOp, ttk::MaxTilesInitOp, ttk::MaxTilesOp>;

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTLTileOpsToTTKernelPatterns(TypeConverter *typeConverter,
                                          const CopyTileCBState *cbState,
                                          RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Control ops.
  patterns.add<TTLTileRegsAcquireToTTKernel, TTLTileRegsCommitToTTKernel,
               TTLTileRegsWaitToTTKernel, TTLTileRegsReleaseToTTKernel>(ctx);

  // Tile op lowerings (ttl.tile_* → ttkernel.*_tile)
  patterns.add<
      // Unary ops
      ExpTileLowering, LogTileLowering, SqrtTileLowering, RsqrtTileLowering,
      TanhTileLowering, SigmoidTileLowering, AbsTileLowering, NegTileLowering,
      ReluTileLowering,
      // Binary ops
      AddTileLowering, SubTileLowering, MulTileLowering, MaxTileLowering>(ctx);

  // Copy op needs the type converter and CB map.
  patterns.add<TTLTileCopyToTTKernel>(*typeConverter, ctx, cbState);

  // TODO(#124): Add DST lifecycle wrapper pattern for loop iterations
  // (acquire/commit/wait/release + copy_tile/pack_tile)
}

} // namespace mlir::tt::ttl
