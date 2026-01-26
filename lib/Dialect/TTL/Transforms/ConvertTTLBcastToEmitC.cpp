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
// For each tile op with bcast_dim attribute (e.g., ttl.tile_add with
// bcast_dim=col):
// 1. Emit the init function: add_bcast_cols_init_short() (or rows/scalar
// variant)
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

/// Get BroadcastType enum name for code generation.
static StringRef getBcastTypeEnumName(BcastDim dim) {
  switch (dim) {
  case BcastDim::row:
    return "BroadcastType::ROW";
  case BcastDim::col:
    return "BroadcastType::COL";
  case BcastDim::scalar:
    return "BroadcastType::SCALAR";
  }
  llvm_unreachable("Unknown BcastDim");
}

/// Get EltwiseBinaryType enum name for code generation.
static StringRef getEltwiseBinaryTypeEnumName(StringRef opPrefix) {
  if (opPrefix == "add") {
    return "EltwiseBinaryType::ELWADD";
  }
  if (opPrefix == "sub") {
    return "EltwiseBinaryType::ELWSUB";
  }
  if (opPrefix == "mul") {
    return "EltwiseBinaryType::ELWMUL";
  }
  // Default to add for unknown ops
  return "EltwiseBinaryType::ELWADD";
}

/// Build the full init_bcast template function name.
/// e.g., "add" + "col" -> "init_bcast<EltwiseBinaryType::ELWADD,
/// BroadcastType::COL>"
static std::string buildFullInitFuncName(StringRef opPrefix, BcastDim dim) {
  std::string result = "init_bcast<";
  result += getEltwiseBinaryTypeEnumName(opPrefix);
  result += ", ";
  result += getBcastTypeEnumName(dim);
  result += ">";
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

/// Get CB index from a tensor.extract operand by tracing back to the attached
/// CB. Returns the CB index (0-31) or nullopt if not found.
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

/// Find the output CB index by looking at the StoreOp that uses this result.
/// Returns the CB index (0-31) or nullopt if not found.
static std::optional<int64_t> findOutputCBIndex(Value tileResult) {
  // Look for a StoreOp that uses this tile result
  for (Operation *user : tileResult.getUsers()) {
    if (auto storeOp = dyn_cast<StoreOp>(user)) {
      // The store's second operand is the output tensor
      Value outputTensor = storeOp.getOperand(1);
      // Trace back to find the CB
      Value cb = getAttachedCB(outputTensor);
      if (!cb) {
        // Try tracing through the tensor directly
        if (auto cbReserve = outputTensor.getDefiningOp<CBReserveOp>()) {
          cb = cbReserve.getCb();
        }
      }
      if (cb) {
        if (auto bindOp = cb.getDefiningOp<BindCBOp>()) {
          return bindOp.getCbIndex().getSExtValue();
        }
      }
    }
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
/// tt-metal bcast functions take uint32_t for CB indices.
static Value createCBIndexLiteral(OpBuilder &rewriter, Location loc,
                                  int64_t cbIndex) {
  // Use i32 since tt-metal API takes uint32_t
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
    LLVM_DEBUG(llvm::dbgs()
               << "TTLTileBcastToEmitC: checking op: " << op << "\n");

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

    LLVM_DEBUG(llvm::dbgs() << "  lhs CB=" << *lhsCBIdx << ", rhs CB="
                            << *rhsCBIdx << ", dst=" << *dstIdxOpt << "\n");

    // Get tile indices from tensor.extract indices (as i32).
    Value lhsTileIdx = getTileIndexFromExtract(lhsExtract, rewriter, loc);
    Value rhsTileIdx = getTileIndexFromExtract(rhsExtract, rewriter, loc);

    // Find output CB index from the store operation that uses this result
    auto outCBIdx = findOutputCBIndex(op.getResult());
    if (!outCBIdx) {
      // Fallback: try common output CB indices
      outCBIdx = 2; // Default to CB2 if we can't find it
      LLVM_DEBUG(llvm::dbgs()
                 << "  Warning: could not find output CB, using default CB2\n");
    }

    LLVM_DEBUG(llvm::dbgs() << "  output CB=" << *outCBIdx << "\n");

    // Create emitc.literal for CB indices using get_compile_time_arg_val().
    Value lhsCBVal = createCBIndexLiteral(rewriter, loc, *lhsCBIdx);
    Value rhsCBVal = createCBIndexLiteral(rewriter, loc, *rhsCBIdx);
    Value outCBVal = createCBIndexLiteral(rewriter, loc, *outCBIdx);
    // Create i32 constant for DST index.
    Value dstIdxVal = createI32Constant(rewriter, loc, *dstIdxOpt);

    // Build function names.
    std::string fullInitFuncName = buildFullInitFuncName(opPrefix, *bcastDim);
    std::string computeFuncName = buildComputeFuncName(opPrefix, *bcastDim);

    // Emit full init_bcast: init_bcast<ELWADD, COL>(regular_cb, bcast_cb,
    // out_cb) This does full hardware initialization including UNPACK, MATH,
    // and PACK. tt-metal API (from
    // _examples/ttnn-bcast/kernels/compute/bcast_add.cpp):
    //   init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::COL>(cb_in0,
    //   cb_in1, cb_out)
    {
      OperationState state(loc, "emitc.call_opaque");
      state.addAttribute("callee", rewriter.getStringAttr(fullInitFuncName));
      // Init function takes CB indices: (regular_cb, bcast_cb, out_cb)
      state.addOperands({lhsCBVal, rhsCBVal, outCBVal});
      rewriter.create(state);
    }

    // Emit compute function:
    //   <op>_tiles_bcast_<dim>(regular_cb, bcast_cb, regular_tile, bcast_tile,
    //   dst)
    // tt-metal API (from bcast_add.cpp line 72):
    //   add_tiles_bcast_cols(cb_in0, cb_in1, base_t + i, 0, i)
    //   where cb_in0=regular (A), cb_in1=bcast (B), first index is A's tile,
    //   second is B's tile
    {
      OperationState state(loc, "emitc.call_opaque");
      state.addAttribute("callee", rewriter.getStringAttr(computeFuncName));
      state.addOperands(
          {lhsCBVal, rhsCBVal, lhsTileIdx, rhsTileIdx, dstIdxVal});
      rewriter.create(state);
    }

    // Replace the op with lhs (result goes to DST, but we need a value for
    // SSA). The actual result is in DST register dstIdxOpt.
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
