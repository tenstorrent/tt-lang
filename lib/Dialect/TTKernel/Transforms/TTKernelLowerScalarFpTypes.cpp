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
