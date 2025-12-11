// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
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
using mlir::PatternRewriter;
using mlir::RewritePatternSet;
using mlir::TypeConverter;
using mlir::UnrealizedConversionCastOp;
using mlir::ValueRange;

namespace ttk = mlir::tt::ttkernel;

struct CreateCBLowering : OpConversionPattern<CreateCBOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CreateCBOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = this->getTypeConverter();
    if (!typeConverter) {
      return rewriter.notifyMatchFailure(op, "no type converter");
    }

    auto converted = typeConverter->convertType(op.getResult().getType());
    if (!converted) {
      return rewriter.notifyMatchFailure(op, "failed to convert CB type");
    }

    // For now, fabricate a CB value via unrealized cast; real lowering would
    // supply a concrete CB handle.
    auto zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), converted, ValueRange{zero});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

enum class CopySourceKind { TensorAccessor, CircularBuffer, Pipe, Unknown };
enum class CopyDestKind { TensorAccessor, CircularBuffer, Pipe, Unknown };

static CopySourceKind classifySrc(Value v) {
  if (llvm::isa<CircularBufferType>(v.getType())) {
    return CopySourceKind::CircularBuffer;
  }
  // TODO(ttl): Detect tensor accessor type when added; for now, assume tensor.
  return CopySourceKind::TensorAccessor;
}

static CopyDestKind classifyDst(Value v) {
  if (llvm::isa<CircularBufferType>(v.getType())) {
    return CopyDestKind::CircularBuffer;
  }
  return CopyDestKind::TensorAccessor;
}

static Value emitPlaceholderCB(ValueRange inputs,
                               ConversionPatternRewriter &rewriter,
                               Location loc, Type targetType) {
  // TODO(ttl): Emit real CB handle; this is a placeholder to keep the pipeline
  // alive.
  auto cast =
      rewriter.create<UnrealizedConversionCastOp>(loc, targetType, inputs);
  return cast.getResult(0);
}

static LogicalResult lowerTensorToCB(CopyOp op,
                                     ConversionPatternRewriter &rewriter,
                                     const TypeConverter &typeConverter) {
  auto loc = op.getLoc();
  auto zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

  auto args = rewriter.create<ttk::TensorAccessorArgsOp>(loc, zeroI32, zeroI32);
  auto accessor = rewriter.create<ttk::TensorAccessorOp>(loc, args.getResult(),
                                                         zeroI32, zeroI32);

  rewriter.create<ttk::NocAsyncReadTileOp>(
      loc, zeroI32.getResult(), accessor.getResult(), zeroI32.getResult());
  // TODO(ttl): Use TRID-specific read barrier when available.
  rewriter.create<ttk::NocAsyncReadBarrierOp>(loc);

  // TODO(ttl): Real CB handle; this uses a placeholder cast.
  auto tkCbTy = typeConverter.convertType(op.getDst().getType());
  auto cbVal = emitPlaceholderCB(ValueRange{zeroI32}, rewriter, loc, tkCbTy);

  rewriter.create<ttk::NocAsyncWriteTileOp>(
      loc, zeroI32.getResult(), accessor.getResult(), zeroI32.getResult());
  // TODO(ttl): Use TRID-specific write barrier when available.
  rewriter.create<ttk::NocAsyncWriteBarrierOp>(loc);

  auto handleTy =
      typeConverter.convertType(TransferHandleType::get(rewriter.getContext()));
  auto handle = rewriter.create<UnrealizedConversionCastOp>(
      loc, handleTy, ValueRange{zeroI32.getResult()});
  (void)cbVal;
  rewriter.replaceOp(op, handle.getResult(0));
  return success();
}

static LogicalResult lowerCBToTensor(CopyOp op,
                                     ConversionPatternRewriter &rewriter,
                                     const TypeConverter &typeConverter) {
  auto loc = op.getLoc();
  auto zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

  // TODO(ttl): Real CB handle and noc addresses for the source CB.
  auto tkCbTy = typeConverter.convertType(op.getSrc().getType());
  auto cbVal = emitPlaceholderCB(ValueRange{zeroI32}, rewriter, loc, tkCbTy);
  (void)cbVal;

  auto args = rewriter.create<ttk::TensorAccessorArgsOp>(loc, zeroI32, zeroI32);
  auto accessor = rewriter.create<ttk::TensorAccessorOp>(loc, args.getResult(),
                                                         zeroI32, zeroI32);

  rewriter.create<ttk::NocAsyncWriteTileOp>(
      loc, zeroI32.getResult(), accessor.getResult(), zeroI32.getResult());
  // TODO(ttl): Use TRID-specific write barrier when available.
  rewriter.create<ttk::NocAsyncWriteBarrierOp>(loc);

  auto handleTy =
      typeConverter.convertType(TransferHandleType::get(rewriter.getContext()));
  auto handle = rewriter.create<UnrealizedConversionCastOp>(
      loc, handleTy, ValueRange{zeroI32.getResult()});
  rewriter.replaceOp(op, handle.getResult(0));
  return success();
}

struct CopyLowering : OpConversionPattern<CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CopyOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = this->getTypeConverter();
    if (!typeConverter) {
      return rewriter.notifyMatchFailure(op, "no type converter");
    }

    auto srcKind = classifySrc(op.getSrc());
    auto dstKind = classifyDst(op.getDst());

    // Tensor accessor -> CB
    if (srcKind == CopySourceKind::TensorAccessor &&
        dstKind == CopyDestKind::CircularBuffer) {
      return lowerTensorToCB(op, rewriter, *typeConverter);
    }

    // CB -> Tensor accessor
    if (srcKind == CopySourceKind::CircularBuffer &&
        dstKind == CopyDestKind::TensorAccessor) {
      return lowerCBToTensor(op, rewriter, *typeConverter);
    }

    // Unsupported pairs for now.
    return rewriter.notifyMatchFailure(op, "unsupported src/dst copy pair");
  }
};

struct WaitLowering : OpConversionPattern<WaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(ttl): Use TRID-specific barrier keyed by the transfer handle.
    rewriter.create<ttk::NocAsyncReadBarrierOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

struct TTLConvertTTLToTTKernelPass
    : impl::TTLConvertTTLToTTKernelBase<TTLConvertTTLToTTKernelPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ModuleOp mod = getOperation();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });
    typeConverter.addConversion([](CircularBufferType t) -> Type {
      return ttk::CBType::get(t.getContext(), t.getTotalElements(),
                              t.getElementType());
    });
    typeConverter.addConversion([](TransferHandleType t) -> Type {
      // TODO(ttl): Use a real handle representation once available.
      return IntegerType::get(t.getContext(), 32);
    });
    typeConverter.addTargetMaterialization(
        [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
          return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
              .getResult(0);
        });

    ConversionTarget target(ctx);
    target.addIllegalDialect<tt::ttl::TTLDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<ttk::TTKernelDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addDynamicallyLegalOp<ModuleOp>([&](ModuleOp op) {
      return typeConverter.isLegal(&op.getBodyRegion());
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    // WaitOp will be lowered; do not mark it legal.

    RewritePatternSet patterns(&ctx);
    patterns.add<CreateCBLowering, CopyLowering, WaitLowering>(typeConverter,
                                                               &ctx);
    populateFunctionOpInterfaceTypeConversionPattern(
        func::FuncOp::getOperationName(), patterns, typeConverter);

    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPartialConversion(mod, target, frozen))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
