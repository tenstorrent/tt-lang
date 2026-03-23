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

/// Resolve the CB index from a CB value.
static FailureOr<int64_t> resolveCBIndex(Value cbValue, Operation *op) {
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
      return op->emitError("CB index annotation missing for compute input ")
             << argIdx;
    }
  }

  return op->emitError(
      "cannot resolve CB index: value must trace to ttl.bind_cb "
      "or be a compute block argument with CB annotation");
}

/// Determine whether a block tensor originates from cb_wait (read) or
/// cb_reserve (write). Returns "read" or "write".
static std::string resolveBlockDirection(Value block) {
  Value traced = traceUnrealizedCasts(block);

  if (auto attach = traced.getDefiningOp<AttachCBOp>()) {
    Value tensor = traceUnrealizedCasts(attach.getTensor());
    if (tensor.getDefiningOp<CBWaitOp>()) {
      return "read";
    }
    if (tensor.getDefiningOp<CBReserveOp>()) {
      return "write";
    }
  }

  if (auto viewLike = traced.getDefiningOp<ViewLikeOpInterface>()) {
    if (isa<CBWaitOp>(viewLike.getOperation())) {
      return "read";
    }
    if (isa<CBReserveOp>(viewLike.getOperation())) {
      return "write";
    }
  }

  return "read";
}

/// Get the tile element type from a block tensor.
static FailureOr<Type> getBlockElementType(Value block, Operation *op) {
  auto tensorType = dyn_cast<RankedTensorType>(block.getType());
  if (!tensorType) {
    return op->emitError("element access block must be a ranked tensor");
  }
  auto tileType = dyn_cast<ttcore::TileType>(tensorType.getElementType());
  if (!tileType) {
    return op->emitError("element access block element type is not a tile");
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
static Operation *createLiteral(Location loc, Type resultType,
                                 StringRef value, OpBuilder &builder) {
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

    // Trace block -> CB
    Value cb = getAttachedCB(block);
    if (!cb) {
      return op.emitError(
          "cannot find attached CB for element_read block");
    }
    auto cbIdx = resolveCBIndex(cb, op);
    if (failed(cbIdx)) {
      return failure();
    }

    // Determine read_ptr vs write_ptr
    std::string dir = resolveBlockDirection(block);
    std::string ptrFn = (dir == "write") ? "get_write_ptr" : "get_read_ptr";

    // Determine bf16 vs f32
    auto elemType = getBlockElementType(block, op);
    if (failed(elemType)) {
      return failure();
    }
    std::string helperName =
        elemType->isBF16() ? "_ttl_elem_read_bf16" : "_ttl_elem_read_f32";

    auto i32Type = rewriter.getI32Type();

    // Step 1: Get CB compile-time arg val as a literal expression
    auto *ctaOp = createLiteral(
        loc, i32Type,
        "get_compile_time_arg_val(" + std::to_string(*cbIdx) + ")",
        rewriter);
    Value ctaVal = ctaOp->getResult(0);

    // Step 2: Get read/write pointer
    auto *ptrOp = createCallOpaque(loc, ptrFn, i32Type,
                                    ValueRange{ctaVal}, rewriter);
    Value l1Addr = ptrOp->getResult(0);

    // Step 3: Cast row/col from Index to i32
    Value rowI32 = indexToI32(loc, op.getRow(), rewriter);
    Value colI32 = indexToI32(loc, op.getCol(), rewriter);

    // Step 4: Call helper function
    // emitc.call_opaque "helperName"(l1_addr, row, col) -> i32
    auto *readOp = createCallOpaque(
        loc, helperName, i32Type,
        ValueRange{l1Addr, rowI32, colI32}, rewriter);

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

    // Trace block -> CB
    Value cb = getAttachedCB(block);
    if (!cb) {
      return op.emitError(
          "cannot find attached CB for element_write block");
    }
    auto cbIdx = resolveCBIndex(cb, op);
    if (failed(cbIdx)) {
      return failure();
    }

    // Determine read_ptr vs write_ptr
    std::string dir = resolveBlockDirection(block);
    std::string ptrFn = (dir == "write") ? "get_write_ptr" : "get_read_ptr";

    // Determine bf16 vs f32
    auto elemType = getBlockElementType(block, op);
    if (failed(elemType)) {
      return failure();
    }
    std::string helperName =
        elemType->isBF16() ? "_ttl_elem_write_bf16" : "_ttl_elem_write_f32";

    auto i32Type = rewriter.getI32Type();

    // Step 1: Get CB compile-time arg val as a literal expression
    auto *ctaOp = createLiteral(
        loc, i32Type,
        "get_compile_time_arg_val(" + std::to_string(*cbIdx) + ")",
        rewriter);
    Value ctaVal = ctaOp->getResult(0);

    // Step 2: Get read/write pointer
    auto *ptrOp = createCallOpaque(loc, ptrFn, i32Type,
                                    ValueRange{ctaVal}, rewriter);
    Value l1Addr = ptrOp->getResult(0);

    // Step 3: Cast row/col from Index to i32
    Value rowI32 = indexToI32(loc, op.getRow(), rewriter);
    Value colI32 = indexToI32(loc, op.getCol(), rewriter);

    // Step 4: Call helper function (void return)
    createCallOpaque(
        loc, helperName, /*resultType=*/nullptr,
        ValueRange{l1Addr, rowI32, colI32, op.getValue()}, rewriter);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper function emission
//===----------------------------------------------------------------------===//

/// Emit inline C++ helper functions for element access at the start of
/// a function. These handle face-based tile layout for 32x32 tiles.
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
        " volatile tt_l1_ptr uint16_t* base ="
        " reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);"
        " uint32_t face = (row >= 16 ? 2 : 0) + (col >= 16 ? 1 : 0);"
        " uint32_t offset = face * 256 + (row % 16) * 16 + (col % 16);"
        " return (uint32_t)base[offset];"
        " };",
        builder);
    emitVerbatim(loc,
        "auto _ttl_elem_write_bf16 = [](uint32_t l1_addr, uint32_t row,"
        " uint32_t col, uint32_t val) {"
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
    : impl::TTLLowerElementAccessToEmitCBase<
          TTLLowerElementAccessToEmitCPass> {
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
      auto elemType = getBlockElementType(block, op);
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
