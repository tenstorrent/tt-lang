// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Tile Ops to TTKernel Lowering
//===----------------------------------------------------------------------===//
//
// This file contains patterns for lowering TTL tile operations (ttl.tile_add,
// ttl.tile_exp, etc.) to TTKernel SFPU/FPU operations.
//
// Current scope (this PR):
// - Tile op patterns only (ttl.tile_* → ttkernel.*_tile)
// - These patterns work on tensor types during development/testing
//
// Future work (TODO(#124)):
// - Full pipeline integration where this pass runs AFTER bufferization
// - DST lifecycle wrapper (acquire/commit/wait/release) around loop iterations
// - copy_tile (CB → DST) before compute, pack_tile (DST → CB) after
// - ttl.compute is lowered to scf.for loops BEFORE bufferization by
//   ttl-lower-to-loops pass; this pass only converts the ttl.tile_* ops
//
// Following LLVM/MLIR best practices:
// - Generic template patterns for tile op categories (like ArithToLLVM)
// - Type aliases for specific op-to-op mappings
// - Batch pattern registration via patterns.add<...>
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir::tt::ttl {

namespace ttk = mlir::tt::ttkernel;

namespace {

/// Get dst_idx for a value (op result or block argument). Defaults to 0.
/// TODO(#120): Remove after implementing proper DST assignment pass.
static int64_t getDstIdxForValue(Value v) {
  if (Operation *defOp = v.getDefiningOp()) {
    if (auto dstIdxAttr = defOp->getAttrOfType<IntegerAttr>("dst_idx")) {
      return dstIdxAttr.getInt();
    }
  }
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    return blockArg.getArgNumber();
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Generic Tile Op Lowering Templates (using ConversionPattern)
//===----------------------------------------------------------------------===//

/// Generic pattern for lowering TTL unary tile ops to TTKernel SFPU ops.
/// Unary SFPU ops: DST[dst_idx] = op(DST[dst_idx]) - operates in-place
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileUnaryToTTKernel : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get dst_idx from attribute (assigned by ttl-assign-dst-registers pass)
    // Default to 0 if not assigned yet (for testing/development)
    int64_t dstIdx = 0;
    if (auto dstIdxAttr = op->template getAttrOfType<IntegerAttr>("dst_idx")) {
      dstIdx = dstIdxAttr.getInt();
    }

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
/// DST index assignment: The ttl-assign-dst-registers pass assigns dst_idx
/// attributes to operands. For now, we use simple defaults: src0=0, src1=1,
/// odst=0 (in-place on src0).
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileBinaryToTTKernel : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    int64_t src0Idx = getDstIdxForValue(adaptor.getLhs());
    int64_t src1Idx = getDstIdxForValue(adaptor.getRhs());
    int64_t odstIdx = src0Idx;
    if (auto dstIdxAttr = op->template getAttrOfType<IntegerAttr>("dst_idx")) {
      odstIdx = dstIdxAttr.getInt();
    }

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
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileMaxToTTKernel : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    int64_t dst0Idx = getDstIdxForValue(adaptor.getLhs());
    int64_t dst1Idx = getDstIdxForValue(adaptor.getRhs());

    Value dst0 = rewriter.create<arith::ConstantIndexOp>(loc, dst0Idx);
    Value dst1 = rewriter.create<arith::ConstantIndexOp>(loc, dst1Idx);

    rewriter.create<InitOp>(loc);
    rewriter.create<TTKernelComputeOp>(loc, dst0, dst1);

    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }
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

void populateTTLTileOpsToTTKernelPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Tile op lowerings (ttl.tile_* → ttkernel.*_tile)
  patterns.add<
      // Unary ops
      ExpTileLowering, LogTileLowering, SqrtTileLowering, RsqrtTileLowering,
      TanhTileLowering, SigmoidTileLowering, AbsTileLowering, NegTileLowering,
      ReluTileLowering,
      // Binary ops
      AddTileLowering, SubTileLowering, MulTileLowering, MaxTileLowering>(ctx);

  // TODO(#124): Add DST lifecycle wrapper pattern for loop iterations
  // (acquire/commit/wait/release + copy_tile/pack_tile)
}

} // namespace mlir::tt::ttl
