// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Broadcast to EmitC Lowering
//===----------------------------------------------------------------------===//
//
// This pass converts TTL tile binary ops with bcast_dim attribute directly
// to EmitC call_opaque ops that emit tt-metal broadcast intrinsics.
//
// For each tile op with bcast_dim attribute (e.g., ttl.tile_add with bcast_dim=col):
// 1. Emit the init function: add_bcast_cols_init_short() (or rows/scalar variant)
// 2. Emit the compute function: add_tiles_bcast_cols(cb0, cb1, i0, i1, dst)
//
// This pass runs AFTER ttl-tile-ops-to-ttkernel (which skips broadcast ops)
// and BEFORE ttkernel-to-emitc.
//
// NOTE: EmitC dialect is NOT listed in dependentDialects to avoid double
// registration when running with other passes that also use EmitC.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-convert-bcast-to-emitc"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLCONVERTBCASTTOEMITC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Get the broadcast dimension string suffix for function names.
/// Returns "cols", "rows", or "scalar" based on the BcastDim enum.
static StringRef getBcastDimSuffix(BcastDim dim) {
  switch (dim) {
  case BcastDim::row:
    return "rows";
  case BcastDim::col:
    return "cols";
  case BcastDim::scalar:
    return "scalar";
  }
  llvm_unreachable("Unknown BcastDim");
}

/// Get the operation name prefix from the tile op name.
/// e.g., "ttl.tile_add" -> "add", "ttl.tile_mul" -> "mul"
static StringRef getOpPrefix(StringRef opName) {
  // Strip "ttl.tile_" prefix
  if (opName.starts_with("ttl.tile_")) {
    return opName.drop_front(9);
  }
  return opName;
}

/// Build the init function name for a broadcast operation.
/// e.g., "add" + "cols" -> "add_bcast_cols_init_short"
static std::string buildInitFuncName(StringRef opPrefix, BcastDim dim) {
  std::string result;
  result += opPrefix;
  result += "_bcast_";
  result += getBcastDimSuffix(dim);
  result += "_init_short";
  return result;
}

/// Build the compute function name for a broadcast operation.
/// e.g., "add" + "cols" -> "add_tiles_bcast_cols"
static std::string buildComputeFuncName(StringRef opPrefix, BcastDim dim) {
  std::string result;
  result += opPrefix;
  result += "_tiles_bcast_";
  result += getBcastDimSuffix(dim);
  return result;
}

/// Extract the dst_idx attribute from a tile operation.
static std::optional<int64_t> getDstIdx(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
    return attr.getInt();
  }
  return std::nullopt;
}

/// Get CB index from a tensor.extract operand by tracing back to the attached CB.
/// Returns the CB index (0-31) or nullopt if not found.
static std::optional<int64_t> getCBIndexFromOperand(Value operand) {
  auto extractOp = operand.getDefiningOp<tensor::ExtractOp>();
  if (!extractOp) {
    return std::nullopt;
  }

  Value cb = getAttachedCB(extractOp.getTensor());
  if (!cb) {
    return std::nullopt;
  }

  if (auto bindOp = cb.getDefiningOp<BindCBOp>()) {
    return bindOp.getCbIndex().getSExtValue();
  }
  return std::nullopt;
}

/// Get the tile index value from a tensor.extract operand as i32.
/// For 2D tensors with shape [R, C], linearizes as: row * C + col.
static Value getTileIndexFromExtract(tensor::ExtractOp extractOp,
                                     PatternRewriter &rewriter, Location loc) {
  auto indices = extractOp.getIndices();
  Value indexVal;
  if (indices.size() == 1) {
    indexVal = indices[0];
  } else if (indices.size() == 2) {
    auto tensorType = cast<RankedTensorType>(extractOp.getTensor().getType());
    int64_t numCols = tensorType.getShape()[1];
    Value numColsVal = rewriter.create<arith::ConstantIndexOp>(loc, numCols);
    Value rowTimesNumCols =
        rewriter.create<arith::MulIOp>(loc, indices[0], numColsVal);
    indexVal = rewriter.create<arith::AddIOp>(loc, rowTimesNumCols, indices[1]);
  } else {
    indexVal = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  }
  // Convert index to i32 for tt-metal API compatibility
  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                             indexVal);
}

/// Create emitc.literal for get_compile_time_arg_val(cbIndex).
/// tt-metal uses compile-time arguments for CB indices.
static Value createCBIndexLiteral(OpBuilder &rewriter, Location loc,
                                  int64_t cbIndex) {
  auto i32Type = rewriter.getI32Type();
  std::string literalStr =
      "get_compile_time_arg_val(" + std::to_string(cbIndex) + ")";

  OperationState state(loc, "emitc.literal");
  state.addTypes(i32Type);
  state.addAttribute("value", rewriter.getStringAttr(literalStr));

  return rewriter.create(state)->getResult(0);
}

/// Create an i32 constant value.
static Value createI32Constant(OpBuilder &rewriter, Location loc,
                               int64_t value) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, 32);
}

/// Pattern to convert TTL tile binary ops with bcast_dim to EmitC.
/// Emits tt-metal broadcast intrinsics that read directly from CBs.
template <typename SourceOp>
struct TTLTileBcastToEmitC : OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "TTLTileBcastToEmitC: checking op: " << op << "\n");

    // Only match ops with bcast_dim attribute set.
    auto bcastDim = op.getBcastDim();
    if (!bcastDim) {
      return rewriter.notifyMatchFailure(op, "no bcast_dim attribute");
    }

    Location loc = op.getLoc();
    StringRef opPrefix =
        getOpPrefix(op.getOperation()->getName().getStringRef());

    // Get dst_idx for the result.
    auto dstIdxOpt = getDstIdx(op.getOperation());
    if (!dstIdxOpt) {
      return rewriter.notifyMatchFailure(op, "missing dst_idx attribute");
    }

    // Get operands - should be from tensor.extract after loop lowering.
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    auto lhsExtract = lhs.getDefiningOp<tensor::ExtractOp>();
    auto rhsExtract = rhs.getDefiningOp<tensor::ExtractOp>();

    if (!lhsExtract || !rhsExtract) {
      return rewriter.notifyMatchFailure(
          op, "operands must be from tensor.extract");
    }

    // Get CB indices for both operands.
    auto lhsCBIdx = getCBIndexFromOperand(lhs);
    auto rhsCBIdx = getCBIndexFromOperand(rhs);

    if (!lhsCBIdx || !rhsCBIdx) {
      return rewriter.notifyMatchFailure(op, "could not find CB indices");
    }

    LLVM_DEBUG(llvm::dbgs() << "  lhs CB=" << *lhsCBIdx << ", rhs CB=" << *rhsCBIdx
                            << ", dst=" << *dstIdxOpt << "\n");

    // Get tile indices from tensor.extract indices (as i32).
    Value lhsTileIdx = getTileIndexFromExtract(lhsExtract, rewriter, loc);
    Value rhsTileIdx = getTileIndexFromExtract(rhsExtract, rewriter, loc);

    // Create emitc.literal for CB indices using get_compile_time_arg_val().
    Value lhsCBVal = createCBIndexLiteral(rewriter, loc, *lhsCBIdx);
    Value rhsCBVal = createCBIndexLiteral(rewriter, loc, *rhsCBIdx);
    // Create i32 constant for DST index.
    Value dstIdxVal = createI32Constant(rewriter, loc, *dstIdxOpt);

    // Build function names.
    std::string initFuncName = buildInitFuncName(opPrefix, *bcastDim);
    std::string computeFuncName = buildComputeFuncName(opPrefix, *bcastDim);

    // Emit init function: <op>_bcast_<dim>_init_short(bcast_cb, regular_cb)
    // tt-metal pattern: bcast CB comes first, regular CB comes second.
    // Use OperationState to build the op generically to avoid symbol conflicts
    {
      OperationState state(loc, "emitc.call_opaque");
      state.addAttribute("callee", rewriter.getStringAttr(initFuncName));
      // Init function takes CB indices: (bcast_cb, regular_cb)
      state.addOperands({rhsCBVal, lhsCBVal});
      rewriter.create(state);
    }

    // Emit compute function:
    //   <op>_tiles_bcast_<dim>(bcast_cb, regular_cb, bcast_tile, regular_tile, dst)
    // tt-metal pattern: bcast CB/tile come first, regular CB/tile come second.
    {
      OperationState state(loc, "emitc.call_opaque");
      state.addAttribute("callee", rewriter.getStringAttr(computeFuncName));
      state.addOperands({rhsCBVal, lhsCBVal, rhsTileIdx, lhsTileIdx, dstIdxVal});
      rewriter.create(state);
    }

    // Replace the op with lhs (result goes to DST, but we need a value for SSA).
    // The actual result is in DST register dstIdxOpt.
    rewriter.replaceOp(op, lhs);
    return success();
  }
};

struct TTLConvertBcastToEmitCPass
    : impl::TTLConvertBcastToEmitCBase<TTLConvertBcastToEmitCPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();


    LLVM_DEBUG(llvm::dbgs() << "=== TTLConvertBcastToEmitCPass running on: "
                            << funcOp.getName() << " ===\n");

    RewritePatternSet patterns(ctx);

    // Add patterns for all tile binary ops that can have bcast_dim.
    patterns.add<TTLTileBcastToEmitC<AddTileOp>>(ctx);
    patterns.add<TTLTileBcastToEmitC<SubTileOp>>(ctx);
    patterns.add<TTLTileBcastToEmitC<MulTileOp>>(ctx);
    patterns.add<TTLTileBcastToEmitC<MaxTileOp>>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
