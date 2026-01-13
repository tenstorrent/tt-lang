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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
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

/// Compute dynamic DST index for multi-tile operations.
/// Returns: numInputs + tile_linear_index for output tiles.
/// If dst_footprint attribute is not found or not in loops, returns static
/// baseDstIdx.
///
/// The dst_footprint attribute marks the outermost tile loop. Only loops at or
/// below this level contribute to the tile linear index. Outer block loops
/// (above the tile loops) are excluded from linearization.
static Value computeDynamicDstIndex(Operation *op, OpBuilder &builder,
                                    int64_t baseDstIdx) {
  Location loc = op->getLoc();

  // Collect enclosing loops (innermost first) and find the tile loop boundary.
  SmallVector<scf::ForOp> allLoops;
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(p)) {
      allLoops.push_back(forOp);
    }
  }

  if (allLoops.empty()) {
    return builder.create<arith::ConstantIndexOp>(loc, baseDstIdx);
  }

  // Find dst_footprint attribute (stores numInputs) - check ComputeOp
  // (pre-loop lowering) or any enclosing SCF loop (post-loop lowering).
  // The attribute is placed on the outermost tile loop.
  IntegerAttr numInputsAttr;
  size_t tileLoopBoundary = allLoops.size(); // Index of outermost tile loop

  if (auto computeOp = op->getParentOfType<ComputeOp>()) {
    numInputsAttr =
        computeOp->getAttrOfType<IntegerAttr>(kDstFootprintAttrName);
    // In pre-loop lowering, all loops are tile loops.
    tileLoopBoundary = allLoops.size();
  } else {
    // Search enclosing loops for the attribute (innermost first).
    // The loop with dst_footprint is the outermost tile loop.
    for (size_t i = 0; i < allLoops.size(); ++i) {
      numInputsAttr =
          allLoops[i]->getAttrOfType<IntegerAttr>(kDstFootprintAttrName);
      if (numInputsAttr) {
        // Found: loops[0..i] are tile loops, loops[i+1..] are block loops.
        tileLoopBoundary = i + 1;
        break;
      }
    }
  }

  if (!numInputsAttr) {
    return builder.create<arith::ConstantIndexOp>(loc, baseDstIdx);
  }

  int64_t numInputs = numInputsAttr.getInt();

  // Multi-tile DST optimization: inputs reuse the same DST slots across
  // iterations, outputs get unique slots per tile.
  // - If baseDstIdx < numInputs: this is an input, return constant (no offset)
  // - If baseDstIdx >= numInputs: this is an output, add tile_linear_index
  if (baseDstIdx < numInputs) {
    // Input: constant DST index, no tile offset
    return builder.create<arith::ConstantIndexOp>(loc, baseDstIdx);
  }

  // Extract only tile loops (up to and including the loop with dst_footprint).
  SmallVector<scf::ForOp> tileLoops(allLoops.begin(),
                                    allLoops.begin() + tileLoopBoundary);

  // Output: compute numInputs + tile_linear_index
  // Collect IVs and bounds from tile loops only (outermost first for strides).
  SmallVector<Value> ivs;
  SmallVector<int64_t> ubs;
  for (auto loop : llvm::reverse(tileLoops)) {
    ivs.push_back(loop.getInductionVar());
    auto ub = getConstantIntValue(loop.getUpperBound());
    ubs.push_back(ub ? *ub : 1);
  }

  // Compute strides and build linearization: numInputs + linearize(ivs, ubs)
  SmallVector<int64_t> strides = mlir::computeStrides(ubs);
  AffineExpr linearExpr = builder.getAffineConstantExpr(numInputs);
  for (size_t i = 0; i < ivs.size(); ++i) {
    linearExpr = linearExpr + builder.getAffineDimExpr(i) *
                                  builder.getAffineConstantExpr(strides[i]);
  }

  AffineMap map = AffineMap::get(ivs.size(), 0, linearExpr);
  return builder.create<affine::AffineApplyOp>(loc, map, ivs);
}

/// Extract the DST register index from a tile value. The index is obtained
/// from either the copy_tile op that placed the tile in DST, or from the
/// dst_idx attribute on the producing tile operation.
static std::optional<int64_t> getDstIndexFromValue(Value v) {
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
    int64_t baseDstIdx = dstIdxAttr.getInt();
    Value dstIdxVal = computeDynamicDstIndex(op, rewriter, baseDstIdx);

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
    int64_t baseOdstIdx = dstIdxAttr.getInt();

    // Use original operands (not adapted) to get DST indices, since the adapted
    // operands may have lost their dst_idx attributes after prior lowerings
    // (binary ops replace themselves with adaptor.getLhs(), losing the
    // intermediate dst_idx tracking).
    int64_t src0Idx = getDstIndexFromValue(op.getLhs()).value_or(0);
    int64_t src1Idx = getDstIndexFromValue(op.getRhs()).value_or(1);

    Value src0 = computeDynamicDstIndex(op, rewriter, src0Idx);
    Value src1 = computeDynamicDstIndex(op, rewriter, src1Idx);
    Value odst = computeDynamicDstIndex(op, rewriter, baseOdstIdx);

    rewriter.create<InitOp>(loc);
    rewriter.create<TTKernelComputeOp>(loc, src0, src1, odst);

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

    // Compute dynamic DST index: base + (tileIndex * footprint).
    // The base index comes from the static dstIndex operand.
    Value baseDstIdx = adaptor.getDstIndex();
    int64_t baseDstIdxVal = 0;
    if (auto constOp = dyn_cast_or_null<arith::ConstantIndexOp>(
            baseDstIdx.getDefiningOp())) {
      baseDstIdxVal = constOp.value();
    }
    Value dynamicDstIdx = computeDynamicDstIndex(op, rewriter, baseDstIdxVal);

    // Emit the copy from CB[src_index] to DST[dynamic_dst_index].
    rewriter.create<ttk::CopyTileOp>(loc, cb, adaptor.getSrcIndex(),
                                     dynamicDstIdx);

    // Materialize results: dst token from dynamic dst_index, and a tile value
    // passthrough (the tile remains the same logical value for downstream tile
    // ops).
    auto token = rewriter
                     .create<mlir::UnrealizedConversionCastOp>(
                         loc, TypeRange{op.getResult(0).getType()},
                         ValueRange{dynamicDstIdx})
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

  // Copy op needs the type converter.
  patterns.add<TTLTileCopyToTTKernel>(*typeConverter, ctx);

  // TODO(#124): Add DST lifecycle wrapper pattern for loop iterations
  // (acquire/commit/wait/release + copy_tile/pack_tile)
}

} // namespace mlir::tt::ttl
