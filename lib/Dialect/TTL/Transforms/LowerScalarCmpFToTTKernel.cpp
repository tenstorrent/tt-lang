// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "ttlang/Dialect/Utils/ConversionUtils.h"

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
      auto intTy = IntegerType::get(builder.getContext(), bitWidth);

      auto lhsInt =
          utils::materializeIntBits(cmpOp.getLhs(), intTy, builder, loc);
      auto rhsInt =
          utils::materializeIntBits(cmpOp.getRhs(), intTy, builder, loc);

      if (failed(lhsInt) || failed(rhsInt)) {
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
              builder, loc, builder.getI1Type(), *lhsInt, *rhsInt);
        } else {
          result = ttk::Bfloat16GreaterOp::create(
              builder, loc, builder.getI1Type(), *lhsInt, *rhsInt);
        }
        break;
      }
      case arith::CmpFPredicate::OLT: {
        if (bitWidth == 32) {
          result = ttk::Float32GreaterOp::create(
              builder, loc, builder.getI1Type(), *rhsInt, *lhsInt);
        } else {
          result = ttk::Bfloat16GreaterOp::create(
              builder, loc, builder.getI1Type(), *rhsInt, *lhsInt);
        }
        break;
      }
      default:
        cmpOp.emitOpError("unsupported cmpf predicate for soft-float "
                          "lowering; only ogt and olt are "
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
