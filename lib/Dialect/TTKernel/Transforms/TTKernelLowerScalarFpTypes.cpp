// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTKERNELLOWERSCALARFPTYPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

namespace ttk = mlir::tt::ttkernel;

static Value createGreaterOp(ConversionPatternRewriter &rewriter, Location loc,
                             unsigned bitWidth, Value lhs, Value rhs) {
  if (bitWidth == 32) {
    return ttk::Float32GreaterOp::create(rewriter, loc, rewriter.getI1Type(),
                                         lhs, rhs);
  }
  return ttk::Bfloat16GreaterOp::create(rewriter, loc, rewriter.getI1Type(),
                                        lhs, rhs);
}

static bool isSupportedCmpFPredicate(arith::CmpFPredicate predicate) {
  return predicate == arith::CmpFPredicate::OGT ||
         predicate == arith::CmpFPredicate::OLT ||
         predicate == arith::CmpFPredicate::OEQ ||
         predicate == arith::CmpFPredicate::UNE;
}

static Value promoteBf16BitsToF32(ConversionPatternRewriter &rewriter,
                                  Location loc, Value bf16Bits) {
  Type i32Type = rewriter.getI32Type();
  Value extended = arith::ExtUIOp::create(rewriter, loc, i32Type, bf16Bits);
  Value shift = arith::ConstantIntOp::create(rewriter, loc, 16, 32);
  return arith::ShLIOp::create(rewriter, loc, extended, shift);
}

static Value createOrderedEqualOp(ConversionPatternRewriter &rewriter,
                                  Location loc, unsigned bitWidth, Value lhs,
                                  Value rhs) {
  uint64_t magnitudeMask = bitWidth == 32 ? 0x7FFFFFFF : 0x7FFF;
  uint64_t exponentMask = bitWidth == 32 ? 0x7F800000 : 0x7F80;
  uint64_t mantissaMask = bitWidth == 32 ? 0x007FFFFF : 0x007F;

  auto constant = [&](uint64_t value) {
    return Value(arith::ConstantIntOp::create(rewriter, loc, value, bitWidth));
  };
  Value zero = constant(0);
  Value magnitude = constant(magnitudeMask);
  Value exponent = constant(exponentMask);
  Value mantissa = constant(mantissaMask);

  Value bitPatternsEqual =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, lhs, rhs);
  Value lhsMagnitude = arith::AndIOp::create(rewriter, loc, lhs, magnitude);
  Value rhsMagnitude = arith::AndIOp::create(rewriter, loc, rhs, magnitude);
  Value lhsIsZero = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, lhsMagnitude, zero);
  Value rhsIsZero = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, rhsMagnitude, zero);
  Value bothZero = arith::AndIOp::create(rewriter, loc, lhsIsZero, rhsIsZero);
  Value numericallyEqual =
      arith::OrIOp::create(rewriter, loc, bitPatternsEqual, bothZero);

  auto createIsNaN = [&](Value bits) {
    Value exponentBits = arith::AndIOp::create(rewriter, loc, bits, exponent);
    Value hasMaxExponent = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, exponentBits, exponent);
    Value mantissaBits = arith::AndIOp::create(rewriter, loc, bits, mantissa);
    Value hasMantissa = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, mantissaBits, zero);
    return Value(
        arith::AndIOp::create(rewriter, loc, hasMaxExponent, hasMantissa));
  };

  Value eitherIsNaN =
      arith::OrIOp::create(rewriter, loc, createIsNaN(lhs), createIsNaN(rhs));
  Value falseValue =
      arith::ConstantIntOp::create(rewriter, loc, rewriter.getI1Type(), 0);
  Value neitherIsNaN = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, eitherIsNaN, falseValue);
  return arith::AndIOp::create(rewriter, loc, numericallyEqual, neitherIsNaN);
}

struct CmpFToSoftFloat : OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto floatTy = dyn_cast<FloatType>(op.getLhs().getType());
    if (!floatTy) {
      return rewriter.notifyMatchFailure(op, "expected scalar float cmpf");
    }

    if (!floatTy.isF32() && !floatTy.isBF16()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported float type for scalar comparison");
    }

    unsigned bitWidth = floatTy.getWidth();
    Value lhsInt = adaptor.getLhs();
    Value rhsInt = adaptor.getRhs();
    Location loc = op.getLoc();

    Value result;
    switch (op.getPredicate()) {
    case arith::CmpFPredicate::OGT:
      result = createGreaterOp(rewriter, loc, bitWidth, lhsInt, rhsInt);
      break;
    case arith::CmpFPredicate::OLT:
      result = createGreaterOp(rewriter, loc, bitWidth, rhsInt, lhsInt);
      break;
    case arith::CmpFPredicate::OEQ:
      result = createOrderedEqualOp(rewriter, loc, bitWidth, lhsInt, rhsInt);
      break;
    case arith::CmpFPredicate::UNE: {
      Value orderedEqual =
          createOrderedEqualOp(rewriter, loc, bitWidth, lhsInt, rhsInt);
      Value falseValue =
          arith::ConstantIntOp::create(rewriter, loc, rewriter.getI1Type(), 0);
      result = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                     orderedEqual, falseValue);
      break;
    }
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported cmpf predicate for soft-float lowering");
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert arith.truncf f32 -> bf16 into integer bit extraction.
/// bf16 is the upper 16 bits of the f32 IEEE-754 encoding, so the truncation
/// is a right shift by 16 followed by an integer truncation. This is a
/// truncation toward zero (no rounding bias). Only f32 -> bf16 without an
/// explicit rounding mode is supported; other type combinations or explicit
/// rounding should use upstream arith-expand instead.
struct TruncFToBitExtract : OpConversionPattern<arith::TruncFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<FloatType>(op.getOperand().getType());
    auto dstTy = dyn_cast<FloatType>(op.getResult().getType());

    if (!srcTy || !dstTy || !srcTy.isF32() || !dstTy.isBF16()) {
      return rewriter.notifyMatchFailure(
          op, "only f32 -> bf16 truncation is supported");
    }

    if (op.getRoundingmodeAttr()) {
      return rewriter.notifyMatchFailure(
          op, "explicit rounding mode not supported");
    }

    Location loc = op.getLoc();
    Value src = adaptor.getIn();
    unsigned srcWidth = 32;
    unsigned dstWidth = 16;

    Value shift = arith::ConstantIntOp::create(rewriter, loc,
                                               srcWidth - dstWidth, srcWidth);
    Value shifted = arith::ShRUIOp::create(rewriter, loc, src, shift);
    auto dstIntTy = IntegerType::get(rewriter.getContext(), dstWidth);
    Value truncated = arith::TruncIOp::create(rewriter, loc, dstIntTy, shifted);
    rewriter.replaceOp(op, truncated);
    return success();
  }
};

struct ExtFToBitPromotion : OpConversionPattern<arith::ExtFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<FloatType>(op.getIn().getType());
    auto resultType = dyn_cast<FloatType>(op.getOut().getType());
    if (!sourceType || !resultType || !sourceType.isBF16() ||
        !resultType.isF32()) {
      return rewriter.notifyMatchFailure(
          op, "only bf16 -> f32 scalar promotion is supported");
    }
    rewriter.replaceOp(
        op, promoteBf16BitsToF32(rewriter, op.getLoc(), adaptor.getIn()));
    return success();
  }
};

template <typename ArithOp, typename TTKernelOp>
struct F32BinaryToTTKernel : OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<FloatType>(op.getResult().getType());
    if (!resultType || !resultType.isF32()) {
      return rewriter.notifyMatchFailure(op, "expected scalar f32 arithmetic");
    }
    rewriter.replaceOpWithNewOp<TTKernelOp>(op, adaptor.getLhs(),
                                            adaptor.getRhs());
    return success();
  }
};

/// Convert arith.constant with a FloatAttr into an integer constant holding
/// the IEEE-754 bit pattern.  Skip constants whose users are ALL TTKernel ops,
/// which legitimately operate on scalar floats (e.g. ttkernel.fill_tile).
struct ConstantOpConversion : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto floatAttr = mlir::dyn_cast<FloatAttr>(op.getValue());
    if (!floatAttr) {
      return rewriter.notifyMatchFailure(op, "not a float constant");
    }

    if (llvm::all_of(op.getResult().getUsers(), [](Operation *user) {
          return isa<ttk::TTKernelDialect>(user->getDialect());
        })) {
      return rewriter.notifyMatchFailure(
          op, "float constant consumed exclusively by TTKernel ops");
    }

    APInt bits = floatAttr.getValue().bitcastToAPInt();
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, bits.getZExtValue(),
                                                      bits.getBitWidth());
    return success();
  }
};

struct TTKernelLowerScalarFpTypesPass
    : impl::TTKernelLowerScalarFpTypesBase<TTKernelLowerScalarFpTypesPass> {
  using TTKernelLowerScalarFpTypesBase::TTKernelLowerScalarFpTypesBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext &ctx = getContext();

    TypeConverter typeConverter;
    // Identity conversion: marks all non-float types as legal so that
    // typeConverter.isLegal(op) does not reject operands/results that this
    // pass does not intend to convert.
    typeConverter.addConversion([](Type t) { return t; });
    typeConverter.addConversion([](FloatType t) -> std::optional<Type> {
      if (!t.isF32() && !t.isBF16()) {
        return std::nullopt;
      }
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

    // Reject unsupported arith.cmpf ops before running the conversion so that
    // diagnostics are specific (rather than generic "failed to legalize").
    bool hasUnsupportedCmpF = false;
    func.walk([&](arith::CmpFOp op) {
      auto floatTy = dyn_cast<FloatType>(op.getLhs().getType());
      if (!floatTy) {
        op.emitOpError("non-scalar float comparison not supported");
        hasUnsupportedCmpF = true;
        return;
      }
      if (!floatTy.isF32() && !floatTy.isBF16()) {
        op.emitOpError("unsupported float type for scalar comparison; "
                       "only f32 and bf16 are supported");
        hasUnsupportedCmpF = true;
        return;
      }
      if (!isSupportedCmpFPredicate(op.getPredicate())) {
        op.emitOpError("unsupported scalar float comparison predicate; "
                       "supported predicates are ogt, olt, oeq, and une");
        hasUnsupportedCmpF = true;
      }
    });
    if (hasUnsupportedCmpF) {
      signalPassFailure();
      return;
    }

    ConversionTarget target(ctx);
    target.addLegalDialect<ttk::TTKernelDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<arith::CmpFOp>();
    target.addIllegalOp<arith::AddFOp>();
    target.addIllegalOp<arith::MulFOp>();
    target.addIllegalOp<arith::SubFOp>();
    target.addIllegalOp<arith::ExtFOp>();
    target.addIllegalOp<arith::TruncFOp>();
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      if (!mlir::isa<FloatAttr>(op.getValue())) {
        return true;
      }
      return llvm::all_of(op.getResult().getUsers(), [](Operation *user) {
        return isa<ttk::TTKernelDialect>(user->getDialect());
      });
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
    patterns.add<CmpFToSoftFloat, ExtFToBitPromotion,
                 F32BinaryToTTKernel<arith::AddFOp, ttk::Float32AddOp>,
                 F32BinaryToTTKernel<arith::MulFOp, ttk::Float32MulOp>,
                 F32BinaryToTTKernel<arith::SubFOp, ttk::Float32SubOp>,
                 TruncFToBitExtract, ConstantOpConversion>(typeConverter, &ctx);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Reconcile cast chains (e.g., iN->fN->iN collapses to identity), erase
    // dead remainders, and verify no live iN<->fN bridges survive.
    SmallVector<UnrealizedConversionCastOp> allCasts;
    func.walk([&](UnrealizedConversionCastOp op) { allCasts.push_back(op); });

    SmallVector<UnrealizedConversionCastOp> remainingCasts;
    reconcileUnrealizedCasts(allCasts, &remainingCasts);

    auto isHandledFloat = [](Type t) { return t.isF32() || t.isBF16(); };
    bool hasBridgeCast = false;
    for (auto castOp : remainingCasts) {
      if (castOp.use_empty()) {
        castOp.erase();
        continue;
      }
      if (castOp.getNumOperands() != 1 || castOp.getNumResults() != 1) {
        continue;
      }
      Type inputTy = castOp.getOperand(0).getType();
      Type outputTy = castOp.getResult(0).getType();
      if ((isa<IntegerType>(inputTy) && isHandledFloat(outputTy)) ||
          (isHandledFloat(inputTy) && isa<IntegerType>(outputTy))) {
        castOp.emitOpError("leftover scalar float bridge cast not eliminated");
        hasBridgeCast = true;
      }
    }
    if (hasBridgeCast) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
