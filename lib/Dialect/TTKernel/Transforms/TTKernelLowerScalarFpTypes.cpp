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

/// Resolve a float-typed SSA value to its underlying signless integer bit
/// pattern. Two sources are handled:
///   1. unrealized_conversion_cast(iN -> fN) from RawElementReadLowering.
///   2. arith.constant <float> -- materializes the bit pattern as an integer.
/// Returns a null Value on failure.
static Value resolveIntBits(Value floatVal, unsigned bitWidth,
                            OpBuilder &builder, Location loc) {
  if (auto cast = floatVal.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() == 1) {
      Value src = cast.getInputs()[0];
      if (auto intTy = dyn_cast<IntegerType>(src.getType())) {
        if (intTy.getWidth() == bitWidth && intTy.isSignless()) {
          return src;
        }
      }
    }
  }

  if (auto constOp = floatVal.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
      APInt bits = floatAttr.getValue().bitcastToAPInt();
      return arith::ConstantIntOp::create(builder, loc, bits.getZExtValue(),
                                          bits.getBitWidth());
    }
  }

  return Value();
}

/// Lower arith.fptosi on a raw-element float to integer bit twiddling.
/// Only non-negative integer-valued floats are supported (index reads); the
/// value is reconstructed from the exponent and mantissa with shifts.
static Value lowerFPToSIBits(Value bits, unsigned mantBits, OpBuilder &b,
                             Location loc) {
  Type i32 = b.getI32Type();
  if (bits.getType() != i32) {
    bits = arith::ExtUIOp::create(b, loc, i32, bits);
  }
  auto cst = [&](int32_t v) {
    return arith::ConstantIntOp::create(b, loc, v, 32).getResult();
  };
  Value exp = arith::AndIOp::create(
      b, loc, arith::ShRUIOp::create(b, loc, bits, cst(mantBits)), cst(0xFF));
  Value ePrime = arith::SubIOp::create(b, loc, exp, cst(127));
  Value mant = arith::OrIOp::create(
      b, loc, arith::AndIOp::create(b, loc, bits, cst((1 << mantBits) - 1)),
      cst(1 << mantBits));
  Value shl = arith::SubIOp::create(b, loc, ePrime, cst(mantBits));
  Value shr = arith::SubIOp::create(b, loc, cst(mantBits), ePrime);
  Value shrC = arith::MinSIOp::create(b, loc, shr, cst(31));
  Value left = arith::ShLIOp::create(b, loc, mant, shl);
  Value right = arith::ShRUIOp::create(b, loc, mant, shrC);
  Value ge = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, ePrime,
                                   cst(mantBits));
  Value val = arith::SelectOp::create(b, loc, ge, left, right);
  Value isZero =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, bits, cst(0));
  return arith::SelectOp::create(b, loc, isZero, cst(0), val);
}

static Value createGreaterOp(ConversionPatternRewriter &rewriter, Location loc,
                             unsigned bitWidth, Value lhs, Value rhs) {
  if (bitWidth == 32) {
    return ttk::Float32GreaterOp::create(rewriter, loc, rewriter.getI1Type(),
                                         lhs, rhs);
  }
  return ttk::Bfloat16GreaterOp::create(rewriter, loc, rewriter.getI1Type(),
                                        lhs, rhs);
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
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported cmpf predicate for soft-float lowering; only ogt "
              "and olt are currently supported");
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

    // Lower raw-element scalar fp casts to integer bit twiddling ahead of the
    // dialect conversion. Both feed tensor-driven indices (ttl.read_index).
    func.walk([&](arith::ExtFOp extOp) {
      // bf16 -> f32 widening between two raw-element bit casts: the bf16 bit
      // pattern shifts left 16 to become the f32 bit pattern.
      if (!extOp.getType().isF32() || !extOp.getIn().getType().isBF16()) {
        return;
      }
      OpBuilder builder(extOp);
      Location loc = extOp.getLoc();
      Value bits = resolveIntBits(extOp.getIn(), 16, builder, loc);
      if (!bits) {
        return;
      }
      Value wide =
          arith::ExtUIOp::create(builder, loc, builder.getI32Type(), bits);
      Value sh = arith::ConstantIntOp::create(builder, loc, 16, 32);
      Value f32bits = arith::ShLIOp::create(builder, loc, wide, sh);
      auto cast = UnrealizedConversionCastOp::create(
          builder, loc, builder.getF32Type(), f32bits);
      extOp.replaceAllUsesWith(cast.getResult(0));
      extOp.erase();
    });

    func.walk([&](arith::FPToSIOp op) {
      Type floatTy = op.getIn().getType();
      unsigned bitWidth = floatTy.isF32() ? 32 : floatTy.isBF16() ? 16 : 0;
      if (!bitWidth || !op.getType().isInteger(32)) {
        return;
      }
      OpBuilder builder(op);
      Location loc = op.getLoc();
      Value bits = resolveIntBits(op.getIn(), bitWidth, builder, loc);
      if (!bits) {
        return;
      }
      unsigned mantBits = bitWidth == 32 ? 23 : 7;
      Value val = lowerFPToSIBits(bits, mantBits, builder, loc);
      op.replaceAllUsesWith(val);
      op.erase();
    });

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
    patterns.add<CmpFToSoftFloat, TruncFToBitExtract, ConstantOpConversion>(
        typeConverter, &ctx);
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
