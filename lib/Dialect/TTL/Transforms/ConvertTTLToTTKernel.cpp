// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/Support/Casting.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLCONVERTTTLTOTTKERNEL
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

using mlir::LogicalResult;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::RewritePatternSet;
using mlir::UnrealizedConversionCastOp;
using mlir::ValueRange;

namespace ttk = mlir::tt::ttkernel;

struct CreateCBLowering : OpRewritePattern<CreateCBOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CreateCBOp op,
                                PatternRewriter &rewriter) const override {
    auto cbTy = llvm::cast<CircularBufferType>(op.getResult().getType());
    int64_t totalElems = cbTy.getTotalElements();
    auto elementTy = cbTy.getElementType();

    auto tkCbTy =
        ttk::CBType::get(rewriter.getContext(), totalElems, elementTy);
    auto zero = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(0));
    auto cast = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), tkCbTy,
                                                            ValueRange{zero});
    rewriter.replaceOp(op, cast.getResult(0));
    return mlir::success();
  }
};

struct CopyLowering : OpRewritePattern<CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto zeroI32 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));

    auto args =
        rewriter.create<ttk::TensorAccessorArgsOp>(loc, zeroI32, zeroI32);
    auto accessor = rewriter.create<ttk::TensorAccessorOp>(
        loc, args.getResult(), zeroI32, zeroI32);

    rewriter.create<ttk::NocAsyncReadTileOp>(
        loc, zeroI32.getResult(), accessor.getResult(), zeroI32.getResult());

    auto handleTy = TransferHandleType::get(rewriter.getContext());
    auto handle = rewriter.create<UnrealizedConversionCastOp>(
        loc, handleTy, ValueRange{zeroI32.getResult()});
    rewriter.replaceOp(op, handle.getResult(0));
    return mlir::success();
  }
};

struct WaitLowering : OpRewritePattern<WaitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WaitOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<ttk::NocAsyncReadBarrierOp>(op.getLoc());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct TTLConvertTTLToTTKernelPass
    : impl::TTLConvertTTLToTTKernelBase<TTLConvertTTLToTTKernelPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CreateCBLowering, CopyLowering, WaitLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
