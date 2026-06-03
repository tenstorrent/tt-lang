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

/// Cast a signless iN value to unsigned uiN for TTKernel numeric ops.
static Value toUnsigned(Value signlessVal, OpBuilder &builder, Location loc) {
  auto intTy = cast<IntegerType>(signlessVal.getType());
  auto unsignedTy = IntegerType::get(builder.getContext(), intTy.getWidth(),
                                     IntegerType::Unsigned);
  return UnrealizedConversionCastOp::create(builder, loc, unsignedTy,
                                            signlessVal)
      .getResult(0);
}

struct TTLLowerScalarCmpFPass
    : impl::TTLLowerScalarCmpFBase<TTLLowerScalarCmpFPass> {
  using TTLLowerScalarCmpFBase::TTLLowerScalarCmpFBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool hadError = false;

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
        Value lhsU = toUnsigned(lhsInt, builder, loc);
        Value rhsU = toUnsigned(rhsInt, builder, loc);
        if (bitWidth == 32) {
          result = ttk::Float32GreaterOp::create(
              builder, loc, builder.getI1Type(), lhsU, rhsU);
        } else {
          result = ttk::Bfloat16GreaterOp::create(
              builder, loc, builder.getI1Type(), lhsU, rhsU);
        }
        break;
      }
      case arith::CmpFPredicate::OLT: {
        Value lhsU = toUnsigned(lhsInt, builder, loc);
        Value rhsU = toUnsigned(rhsInt, builder, loc);
        if (bitWidth == 32) {
          result = ttk::Float32GreaterOp::create(
              builder, loc, builder.getI1Type(), rhsU, lhsU);
        } else {
          result = ttk::Bfloat16GreaterOp::create(
              builder, loc, builder.getI1Type(), rhsU, lhsU);
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
