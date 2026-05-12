// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLLOWERELEMENTACCESSTOEMITC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

template <typename BuilderT>
static void emitVerbatim(Location loc, StringRef value, BuilderT &builder) {
  OperationState state(loc, "emitc.verbatim");
  state.addAttribute("value", builder.getStringAttr(value));
  builder.create(state);
}

/// Resolve the CB index from a CB value. Returns failure without emitting
/// diagnostics; callers inside pattern rewriters must use
/// notifyMatchFailure to report the error.
static FailureOr<int64_t> resolveCBIndex(Value cbValue) {
  cbValue = traceUnrealizedCasts(cbValue);

  if (auto bindOp = cbValue.getDefiningOp<BindCBOp>()) {
    return bindOp.getCbIndex().getSExtValue();
  }

  if (auto blockArg = dyn_cast<BlockArgument>(cbValue)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();
    if (auto computeOp = dyn_cast<ComputeOp>(parentOp)) {
      unsigned argIdx = blockArg.getArgNumber();
      auto cbIdx = getCBIndexAttr(computeOp, argIdx);
      if (cbIdx) {
        return *cbIdx;
      }
      return failure();
    }
  }

  return failure();
}

/// Determine whether a block tensor originates from cb_wait (read) or
/// cb_reserve (write). Returns "read" or "write", or failure without
/// emitting diagnostics.
static FailureOr<std::string> resolveBlockDirection(Value block) {
  Value traced = traceUnrealizedCasts(block);

  if (auto attach = traced.getDefiningOp<AttachCBOp>()) {
    Value tensor = traceUnrealizedCasts(attach.getTensor());
    if (tensor.getDefiningOp<CBWaitOp>()) {
      return std::string("read");
    }
    if (tensor.getDefiningOp<CBReserveOp>()) {
      return std::string("write");
    }
  }

  // Fallback: CBWaitOp and CBReserveOp implement ViewLikeOpInterface, so when
  // the block reaches here without an AttachCBOp wrapper (e.g., future IR
  // patterns or canonicalization), we can still resolve direction directly.
  if (auto viewLike = traced.getDefiningOp<ViewLikeOpInterface>()) {
    if (isa<CBWaitOp>(viewLike.getOperation())) {
      return std::string("read");
    }
    if (isa<CBReserveOp>(viewLike.getOperation())) {
      return std::string("write");
    }
  }

  return failure();
}

/// Get the tile element type from a block tensor. Returns failure without
/// emitting diagnostics.
static FailureOr<Type> getBlockElementType(Value block) {
  auto tensorType = dyn_cast<RankedTensorType>(block.getType());
  if (!tensorType) {
    return failure();
  }
  auto tileType = dyn_cast<ttcore::TileType>(tensorType.getElementType());
  if (!tileType) {
    return failure();
  }
  return tileType.getElementType();
}

/// Create an emitc.call_opaque op via OperationState (avoids EmitC C++
/// header dependency). Returns the created operation.
static Operation *createCallOpaque(Location loc, StringRef callee,
                                   Type resultType, ValueRange operands,
                                   OpBuilder &builder) {
  OperationState state(loc, "emitc.call_opaque");
  state.addAttribute("callee", builder.getStringAttr(callee));
  if (resultType) {
    state.addTypes(resultType);
  }
  state.addOperands(operands);
  return builder.create(state);
}

/// Create an emitc.literal op that produces a C++ expression as-is.
static Operation *createLiteral(Location loc, Type resultType, StringRef value,
                                OpBuilder &builder) {
  OperationState state(loc, "emitc.literal");
  state.addAttribute("value", builder.getStringAttr(value));
  state.addTypes(resultType);
  return builder.create(state);
}

/// Cast an Index value to i32 using arith.index_cast.
static Value indexToI32(Location loc, Value indexVal, OpBuilder &builder) {
  auto i32Type = builder.getI32Type();
  return arith::IndexCastOp::create(builder, loc, i32Type, indexVal);
}

//===----------------------------------------------------------------------===//
// Lowering patterns
//===----------------------------------------------------------------------===//

struct ElementReadLowering : OpConversionPattern<ElementReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ElementReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value block = op.getBlock();

    Value cb = getAttachedCB(block);
    if (!cb) {
      return rewriter.notifyMatchFailure(
          op, "cannot find attached CB for element_read block");
    }
    auto cbIdx = resolveCBIndex(cb);
    if (failed(cbIdx)) {
      return rewriter.notifyMatchFailure(
          op, "cannot resolve CB index: value must trace to ttl.bind_cb "
              "or be a compute block argument with CB annotation");
    }

    auto dir = resolveBlockDirection(block);
    if (failed(dir)) {
      return rewriter.notifyMatchFailure(
          op, "cannot determine block direction: block must trace "
              "to cb_wait (read) or cb_reserve (write)");
    }
    std::string ptrFn = (*dir == "write") ? "get_write_ptr" : "get_read_ptr";

    auto elemType = getBlockElementType(block);
    if (failed(elemType)) {
      return rewriter.notifyMatchFailure(
          op, "element access block must be a ranked tensor of tiles");
    }
    std::string helperName =
        elemType->isBF16() ? "_ttl_elem_read_bf16" : "_ttl_elem_read_f32";

    auto i32Type = rewriter.getI32Type();

    auto *ctaOp = createLiteral(
        loc, i32Type,
        "get_compile_time_arg_val(" + std::to_string(*cbIdx) + ")", rewriter);
    Value ctaVal = ctaOp->getResult(0);

    auto *ptrOp =
        createCallOpaque(loc, ptrFn, i32Type, ValueRange{ctaVal}, rewriter);
    Value l1Addr = ptrOp->getResult(0);

    Value rowI32 = indexToI32(loc, op.getRow(), rewriter);
    Value colI32 = indexToI32(loc, op.getCol(), rewriter);

    auto *readOp = createCallOpaque(
        loc, helperName, i32Type, ValueRange{l1Addr, rowI32, colI32}, rewriter);

    rewriter.replaceOp(op, readOp->getResults());
    return success();
  }
};

struct ElementWriteLowering : OpConversionPattern<ElementWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ElementWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value block = op.getBlock();

    Value cb = getAttachedCB(block);
    if (!cb) {
      return rewriter.notifyMatchFailure(
          op, "cannot find attached CB for element_write block");
    }
    auto cbIdx = resolveCBIndex(cb);
    if (failed(cbIdx)) {
      return rewriter.notifyMatchFailure(
          op, "cannot resolve CB index: value must trace to ttl.bind_cb "
              "or be a compute block argument with CB annotation");
    }

    auto dir = resolveBlockDirection(block);
    if (failed(dir)) {
      return rewriter.notifyMatchFailure(
          op, "cannot determine block direction: block must trace "
              "to cb_wait (read) or cb_reserve (write)");
    }
    std::string ptrFn = (*dir == "write") ? "get_write_ptr" : "get_read_ptr";

    auto elemType = getBlockElementType(block);
    if (failed(elemType)) {
      return rewriter.notifyMatchFailure(
          op, "element access block must be a ranked tensor of tiles");
    }
    std::string helperName =
        elemType->isBF16() ? "_ttl_elem_write_bf16" : "_ttl_elem_write_f32";

    auto i32Type = rewriter.getI32Type();

    auto *ctaOp = createLiteral(
        loc, i32Type,
        "get_compile_time_arg_val(" + std::to_string(*cbIdx) + ")", rewriter);
    Value ctaVal = ctaOp->getResult(0);

    auto *ptrOp =
        createCallOpaque(loc, ptrFn, i32Type, ValueRange{ctaVal}, rewriter);
    Value l1Addr = ptrOp->getResult(0);

    Value rowI32 = indexToI32(loc, op.getRow(), rewriter);
    Value colI32 = indexToI32(loc, op.getCol(), rewriter);

    createCallOpaque(loc, helperName, /*resultType=*/nullptr,
                     ValueRange{l1Addr, rowI32, colI32, op.getValue()},
                     rewriter);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper function emission
//===----------------------------------------------------------------------===//

/// Emit inline C++ helper functions for element access at the start of
/// a function. These handle face-based tile layout for 32x32 tiles.
///
/// Tile layout (bf16 and f32 both use this face-based arrangement):
///   A 32x32 tile is divided into 4 contiguous 16x16 "faces" in memory:
///     Face 0: rows  0-15, cols  0-15  (offset    0..255)
///     Face 1: rows  0-15, cols 16-31  (offset  256..511)
///     Face 2: rows 16-31, cols  0-15  (offset  512..767)
///     Face 3: rows 16-31, cols 16-31  (offset  768..1023)
///
///   Formula:  face = (row >= 16 ? 2 : 0) + (col >= 16 ? 1 : 0)
///             offset = face * 256 + (row % 16) * 16 + (col % 16)
///
///   Constants:
///     16  = kFaceDim (face is 16x16 elements)
///     256 = kFaceDim * kFaceDim (elements per face)
///
///   For bf16, each element is uint16_t (2 bytes); for f32, each element
///   is uint32_t (4 bytes). The face indexing formula is identical for both.
///
///   Reference: tt-metal tile layout described in
///   tt-metal/tt_metal/hw/inc/dataflow_api.h (tilized data format).
///   TODO(#572): validate f32 face ordering on hardware with a runtime test.
static void addElementAccessHelpers(func::FuncOp func, OpBuilder &builder,
                                    bool needsBF16, bool needsF32) {
  builder.setInsertionPointToStart(&func.getBody().front());
  auto loc = func.getLoc();

  // Use lambdas (valid inside function bodies in C++17) instead of nested
  // function definitions, which are not allowed in C++.
  if (needsBF16) {
    emitVerbatim(loc,
                 "auto _ttl_elem_read_bf16 = [](uint32_t l1_addr, uint32_t row,"
                 " uint32_t col) -> uint32_t {"
                 " ASSERT(row < 32 && col < 32);"
                 " volatile tt_l1_ptr uint16_t* base ="
                 " reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);"
                 " uint32_t face = (row >= 16 ? 2 : 0) + (col >= 16 ? 1 : 0);"
                 " uint32_t offset = face * 256 + (row % 16) * 16 + (col % 16);"
                 " return (uint32_t)base[offset];"
                 " };",
                 builder);
    emitVerbatim(
        loc,
        "auto _ttl_elem_write_bf16 = [](uint32_t l1_addr, uint32_t row,"
        " uint32_t col, uint32_t val) {"
        " ASSERT(row < 32 && col < 32);"
        " volatile tt_l1_ptr uint16_t* base ="
        " reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);"
        " uint32_t face = (row >= 16 ? 2 : 0) + (col >= 16 ? 1 : 0);"
        " uint32_t offset = face * 256 + (row % 16) * 16 + (col % 16);"
        " base[offset] = (uint16_t)val;"
        " };",
        builder);
  }

  if (needsF32) {
    emitVerbatim(loc,
                 "auto _ttl_elem_read_f32 = [](uint32_t l1_addr, uint32_t row,"
                 " uint32_t col) -> uint32_t {"
                 " ASSERT(row < 32 && col < 32);"
                 " volatile tt_l1_ptr uint32_t* base ="
                 " reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);"
                 " uint32_t face = (row >= 16 ? 2 : 0) + (col >= 16 ? 1 : 0);"
                 " uint32_t offset = face * 256 + (row % 16) * 16 + (col % 16);"
                 " return base[offset];"
                 " };",
                 builder);
    emitVerbatim(loc,
                 "auto _ttl_elem_write_f32 = [](uint32_t l1_addr, uint32_t row,"
                 " uint32_t col, uint32_t val) {"
                 " ASSERT(row < 32 && col < 32);"
                 " volatile tt_l1_ptr uint32_t* base ="
                 " reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);"
                 " uint32_t face = (row >= 16 ? 2 : 0) + (col >= 16 ? 1 : 0);"
                 " uint32_t offset = face * 256 + (row % 16) * 16 + (col % 16);"
                 " base[offset] = val;"
                 " };",
                 builder);
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TTLLowerElementAccessToEmitCPass
    : impl::TTLLowerElementAccessToEmitCBase<TTLLowerElementAccessToEmitCPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ModuleOp mod = getOperation();

    // Record which functions have element ops and what types they need.
    struct FuncInfo {
      bool needsBF16 = false;
      bool needsF32 = false;
    };
    llvm::DenseMap<func::FuncOp, FuncInfo> funcsWithElementOps;

    mod.walk([&](Operation *op) {
      Value block;
      if (auto readOp = dyn_cast<ElementReadOp>(op)) {
        block = readOp.getBlock();
      } else if (auto writeOp = dyn_cast<ElementWriteOp>(op)) {
        block = writeOp.getBlock();
      } else {
        return;
      }

      auto func = op->getParentOfType<func::FuncOp>();
      if (!func) {
        return;
      }

      auto &info = funcsWithElementOps[func];
      auto elemType = getBlockElementType(block);
      if (succeeded(elemType)) {
        if (elemType->isBF16()) {
          info.needsBF16 = true;
        } else {
          info.needsF32 = true;
        }
      }
    });

    if (funcsWithElementOps.empty()) {
      return;
    }

    ConversionTarget target(ctx);
    target.addIllegalOp<ElementReadOp, ElementWriteOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&ctx);
    patterns.add<ElementReadLowering, ElementWriteLowering>(&ctx);

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Add helper functions at the start of each affected function.
    OpBuilder builder(&ctx);
    for (auto &[func, info] : funcsWithElementOps) {
      addElementAccessHelpers(func, builder, info.needsBF16, info.needsF32);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
