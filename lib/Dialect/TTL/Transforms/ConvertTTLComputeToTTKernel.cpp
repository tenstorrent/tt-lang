// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Compute to TTKernel Lowering
//===----------------------------------------------------------------------===//
//
// This file implements lowering of ttl.compute and ttl.tile_* operations to
// TTKernel ops using DialectConversion framework (best practice for nested
// regions). The approach:
//
// 1. Use applyPartialConversion with ConversionTarget
// 2. Process tile ops first (inside compute body), converting them in-place
// 3. Then process ComputeOp, inlining the converted body between DST lifecycle
//
// Following LLVM/MLIR best practices:
// - Generic template patterns for tile op categories (like ArithToLLVM)
// - Type aliases for specific op-to-op mappings
// - Batch pattern registration via patterns.add<...>
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir::tt::ttl {

namespace ttk = mlir::tt::ttkernel;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get DST index for a value (block argument or op result).
/// Returns the dst_idx attribute from the defining op, or looks up from the
/// compute op's dst_idx_map for block arguments.
static int64_t getDstIdxForValue(Value v) {
  // For op results, get dst_idx from the defining op's attribute
  if (Operation *defOp = v.getDefiningOp()) {
    if (auto dstIdxAttr = defOp->getAttrOfType<IntegerAttr>("dst_idx")) {
      return dstIdxAttr.getInt();
    }
    // Fallback: assume index 0 if not assigned yet
    return 0;
  }

  // For block arguments, look up in parent compute op's dst_idx_map
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto dstIdxMap =
            parentOp->getAttrOfType<DenseI64ArrayAttr>("dst_idx_map")) {
      return dstIdxMap[blockArg.getArgNumber()];
    }
    // Fallback: use block argument number as DST index
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
    int64_t dstIdx = 0;
    if (auto dstIdxAttr = op->template getAttrOfType<IntegerAttr>("dst_idx")) {
      dstIdx = dstIdxAttr.getInt();
    } else {
      // Unary ops operate in-place, so use input's DST index
      dstIdx = getDstIdxForValue(op.getInput());
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
template <typename SourceOp, typename InitOp, typename TTKernelComputeOp>
struct TTLTileBinaryToTTKernel : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get DST indices for operands and result
    int64_t src0Idx = getDstIdxForValue(op.getLhs());
    int64_t src1Idx = getDstIdxForValue(op.getRhs());

    // Result DST index from attribute, or default to src0 (in-place)
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

    // MaxTilesOp is in-place: DST[dst0] = max(DST[dst0], DST[dst1])
    int64_t dst0Idx = getDstIdxForValue(op.getLhs());
    int64_t dst1Idx = getDstIdxForValue(op.getRhs());

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

//===----------------------------------------------------------------------===//
// ComputeOp Lowering
//===----------------------------------------------------------------------===//

/// Lower ttl.compute to TTKernel DST register lifecycle operations.
/// The body operations (tile ops) should already be converted to TTKernel
/// by the tile op patterns before this pattern runs.
struct LowerComputeOp : OpConversionPattern<ComputeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Block &bodyBlock = op.getBody().front();

    // Check that body has a terminator
    if (!bodyBlock.mightHaveTerminator()) {
      return rewriter.notifyMatchFailure(op, "body block has no terminator");
    }

    auto yieldOp = dyn_cast<YieldOp>(bodyBlock.getTerminator());
    if (!yieldOp) {
      return rewriter.notifyMatchFailure(op, "terminator is not YieldOp");
    }

    // 1. Acquire DST registers
    rewriter.create<ttk::TileRegsAcquireOp>(loc);

    // 2. Load input tiles from CBs to DST registers
    // Only emit copy_tile ops for CB-typed inputs; skip for tensor inputs.
    // TODO: When tensors are converted to CBs earlier in pipeline, remove check.
    for (auto [idx, input] : llvm::enumerate(adaptor.getInputs())) {
      // Skip non-CB inputs (tensors) - copy_tile requires CB type
      if (!llvm::isa<ttk::CBType>(input.getType())) {
        continue;
      }

      int64_t dstIdx = idx;
      if (auto dstIdxMap =
              op->getAttrOfType<DenseI64ArrayAttr>("dst_idx_map")) {
        if (idx < static_cast<size_t>(dstIdxMap.size())) {
          dstIdx = dstIdxMap[idx];
        }
      }

      Value dstIdxVal = rewriter.create<arith::ConstantIndexOp>(loc, dstIdx);
      Value cbIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      rewriter.create<ttk::CopyTileInitOp>(loc, input);
      rewriter.create<ttk::CopyTileOp>(loc, input, cbIdx, dstIdxVal);
    }

    // 3. Inline the body operations (which should now be TTKernel ops)
    // Create mapping from block args to adapted inputs
    IRMapping mapping;
    for (auto [blockArg, input] :
         llvm::zip(bodyBlock.getArguments().take_front(adaptor.getInputs().size()),
                   adaptor.getInputs())) {
      mapping.map(blockArg, input);
    }
    // Map output block args
    for (auto [blockArg, output] :
         llvm::zip(bodyBlock.getArguments().drop_front(adaptor.getInputs().size()),
                   adaptor.getOutputs())) {
      mapping.map(blockArg, output);
    }

    // Clone body ops (excluding terminator) with mapping
    for (Operation &bodyOp : bodyBlock.without_terminator()) {
      rewriter.clone(bodyOp, mapping);
    }

    // 4. Commit and wait for compute
    rewriter.create<ttk::TileRegsCommitOp>(loc);
    rewriter.create<ttk::TileRegsWaitOp>(loc);

    // 5. Pack results from DST to output CBs
    // Only emit pack_tile ops for CB-typed outputs; skip for tensor outputs.
    for (auto [idx, yieldVal] : llvm::enumerate(yieldOp.getOperands())) {
      Value output = adaptor.getOutputs()[idx];
      // Skip non-CB outputs (tensors) - pack_tile requires CB type
      if (!llvm::isa<ttk::CBType>(output.getType())) {
        continue;
      }

      Value remappedResult = mapping.lookupOrDefault(yieldVal);
      int64_t dstIdx = getDstIdxForValue(remappedResult);
      Value dstIdxVal = rewriter.create<arith::ConstantIndexOp>(loc, dstIdx);
      Value cbIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      rewriter.create<ttk::PackTileOp>(loc, dstIdxVal, output, cbIdx,
                                       rewriter.getBoolAttr(true));
    }

    // 6. Release DST registers
    rewriter.create<ttk::TileRegsReleaseOp>(loc);

    // 7. Replace compute op results with output values
    rewriter.replaceOp(op, adaptor.getOutputs());
    return success();
  }
};

/// Lower YieldOp - should only be encountered after ComputeOp lowering has
/// already processed it (orphaned yields in invalid IR).
struct LowerYieldOp : OpConversionPattern<YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // YieldOp should have been handled during ComputeOp lowering.
    // If we encounter one here, it's orphaned and should be erased.
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTLComputeToTTKernelPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Tile op lowerings first (higher benefit ensures they run before ComputeOp)
  // This ensures the body operations are converted before ComputeOp inlines them
  patterns.add<
      // Unary ops
      ExpTileLowering, LogTileLowering, SqrtTileLowering, RsqrtTileLowering,
      TanhTileLowering, SigmoidTileLowering, AbsTileLowering, NegTileLowering,
      ReluTileLowering,
      // Binary ops
      AddTileLowering, SubTileLowering, MulTileLowering, MaxTileLowering>(
      ctx, /*benefit=*/20);

  // ComputeOp lowering (processes after tile ops are converted)
  patterns.add<LowerComputeOp>(ctx, /*benefit=*/10);

  // YieldOp cleanup (lowest priority)
  patterns.add<LowerYieldOp>(ctx, /*benefit=*/1);
}

} // namespace mlir::tt::ttl
