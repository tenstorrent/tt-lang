// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLLOWERSCALARCMPF
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

struct TTLLowerScalarCmpFPass
    : impl::TTLLowerScalarCmpFBase<TTLLowerScalarCmpFPass> {
  using TTLLowerScalarCmpFBase::TTLLowerScalarCmpFBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool hadError = false;

    mod.walk([&](arith::ExtFOp extOp) {
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
      Value wide = arith::ExtUIOp::create(builder, loc, builder.getI32Type(), bits);
      Value sh = arith::ConstantIntOp::create(builder, loc, 16, 32);
      Value f32bits = arith::ShLIOp::create(builder, loc, wide, sh);
      auto cast = UnrealizedConversionCastOp::create(
          builder, loc, builder.getF32Type(), f32bits);
      extOp.replaceAllUsesWith(cast.getResult(0));
      extOp.erase();
    });

    mod.walk([&](arith::FPToSIOp op) {
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

    mod.walk([&](arith::CmpFOp cmpOp) {
      Type floatTy = cmpOp.getLhs().getType();

      unsigned bitWidth;
      if (floatTy.isF32()) {
        bitWidth = 32;
      } else if (floatTy.isBF16()) {
        bitWidth = 16;
      } else {
        cmpOp.emitOpError("unsupported float type for scalar comparison: ")
            << floatTy;
        hadError = true;
        return;
      }

      OpBuilder builder(cmpOp);
      Location loc = cmpOp.getLoc();

      Value lhsInt = resolveIntBits(cmpOp.getLhs(), bitWidth, builder, loc);
      Value rhsInt = resolveIntBits(cmpOp.getRhs(), bitWidth, builder, loc);

      if (!lhsInt || !rhsInt) {
        cmpOp.emitOpError(
            "could not resolve float operand to integer bit pattern; "
            "operands must come from raw_element_read or float constants");
        hadError = true;
        return;
      }

      Value result;
      auto pred = cmpOp.getPredicate();

      switch (pred) {
      case arith::CmpFPredicate::OGT: {
        if (bitWidth == 32) {
          result = ttk::Float32GreaterOp::create(
              builder, loc, builder.getI1Type(), lhsInt, rhsInt);
        } else {
          result = ttk::Bfloat16GreaterOp::create(
              builder, loc, builder.getI1Type(), lhsInt, rhsInt);
        }
        break;
      }
      case arith::CmpFPredicate::OLT: {
        if (bitWidth == 32) {
          result = ttk::Float32GreaterOp::create(
              builder, loc, builder.getI1Type(), rhsInt, lhsInt);
        } else {
          result = ttk::Bfloat16GreaterOp::create(
              builder, loc, builder.getI1Type(), rhsInt, lhsInt);
        }
        break;
      }
      case arith::CmpFPredicate::OEQ: {
        result = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       lhsInt, rhsInt);
        break;
      }
      case arith::CmpFPredicate::ONE: {
        result = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                       lhsInt, rhsInt);
        break;
      }
      default:
        cmpOp.emitOpError("unsupported cmpf predicate for soft-float "
                          "lowering; only ogt, olt, oeq, and one are "
                          "currently supported");
        hadError = true;
        return;
      }

      cmpOp.replaceAllUsesWith(result);
      cmpOp.erase();
    });

    if (hadError) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
