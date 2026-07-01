// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLLOWERSCALARFPTYPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

namespace ttk = mlir::tt::ttkernel;

struct CmpFToSoftFloat : OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type origFloatTy = op.getLhs().getType();
    unsigned bitWidth = origFloatTy.getIntOrFloatBitWidth();

    if (!origFloatTy.isF32() && !origFloatTy.isBF16()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported float type for scalar comparison");
    }

    Value lhsInt = adaptor.getLhs();
    Value rhsInt = adaptor.getRhs();
    Location loc = op.getLoc();

    Value result;
    switch (op.getPredicate()) {
    case arith::CmpFPredicate::OGT:
      if (bitWidth == 32) {
        result = ttk::Float32GreaterOp::create(
            rewriter, loc, rewriter.getI1Type(), lhsInt, rhsInt);
      } else {
        result = ttk::Bfloat16GreaterOp::create(
            rewriter, loc, rewriter.getI1Type(), lhsInt, rhsInt);
      }
      break;
    case arith::CmpFPredicate::OLT:
      if (bitWidth == 32) {
        result = ttk::Float32GreaterOp::create(
            rewriter, loc, rewriter.getI1Type(), rhsInt, lhsInt);
      } else {
        result = ttk::Bfloat16GreaterOp::create(
            rewriter, loc, rewriter.getI1Type(), rhsInt, lhsInt);
      }
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported cmpf predicate for soft-float lowering; only ogt "
              "and olt are currently supported");
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert arith.truncf (e.g. f32 -> bf16) into integer bit extraction.
/// bf16 is the upper 16 bits of the f32 IEEE-754 encoding, so a truncf
/// becomes a right shift by (srcWidth - dstWidth) followed by an integer
/// truncation.
struct TruncFToBitExtract : OpConversionPattern<arith::TruncFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned srcWidth = op.getOperand().getType().getIntOrFloatBitWidth();
    unsigned dstWidth = op.getResult().getType().getIntOrFloatBitWidth();
    Location loc = op.getLoc();

    Value src = adaptor.getIn();
    Value shift = arith::ConstantIntOp::create(rewriter, loc,
                                               srcWidth - dstWidth, srcWidth);
    Value shifted = arith::ShRUIOp::create(rewriter, loc, src, shift);
    auto dstIntTy = IntegerType::get(rewriter.getContext(), dstWidth);
    Value truncated = arith::TruncIOp::create(rewriter, loc, dstIntTy, shifted);
    rewriter.replaceOp(op, truncated);
    return success();
  }
};

/// Convert arith.constant with a FloatAttr into an integer constant holding
/// the IEEE-754 bit pattern.  Skip constants consumed by TTKernel ops, which
/// legitimately operate on scalar floats (e.g. ttkernel.fill_tile).
struct ConstantOpConversion : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto floatAttr = mlir::dyn_cast<FloatAttr>(op.getValue());
    if (!floatAttr) {
      return rewriter.notifyMatchFailure(op, "not a float constant");
    }

    for (Operation *user : op.getResult().getUsers()) {
      if (isa<ttk::TTKernelDialect>(user->getDialect())) {
        return rewriter.notifyMatchFailure(
            op, "float constant consumed by TTKernel op");
      }
    }

    APInt bits = floatAttr.getValue().bitcastToAPInt();
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, bits.getZExtValue(),
                                                      bits.getBitWidth());
    return success();
  }
};

struct TTLLowerScalarFpTypesPass
    : impl::TTLLowerScalarFpTypesBase<TTLLowerScalarFpTypesPass> {
  using TTLLowerScalarFpTypesBase::TTLLowerScalarFpTypesBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext &ctx = getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });
    typeConverter.addConversion([](FloatType t) -> Type {
      return IntegerType::get(t.getContext(), t.getWidth());
    });
    auto materializeCast = [](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);

    ConversionTarget target(ctx);
    target.addLegalDialect<ttk::TTKernelDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<arith::CmpFOp>();
    target.addIllegalOp<arith::TruncFOp>();
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      if (!mlir::isa<FloatAttr>(op.getValue())) {
        return true;
      }
      for (Operation *user : op.getResult().getUsers()) {
        if (isa<ttk::TTKernelDialect>(user->getDialect())) {
          return true;
        }
      }
      return false;
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(&ctx);
    patterns.add<CmpFToSoftFloat, TruncFToBitExtract, ConstantOpConversion>(
        typeConverter, &ctx);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
