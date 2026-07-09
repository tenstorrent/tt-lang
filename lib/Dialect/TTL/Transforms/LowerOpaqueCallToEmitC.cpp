// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLLOWEROPAQUECALLTOEMITC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Lower ttl.opaque_call -> emitc.call_opaque, inserting
/// unrealized_conversion_cast for scalar float operands (f32 -> i32,
/// bf16 -> i16) so that ttkernel-lower-scalar-fp-types can resolve them.
struct OpaqueCallLowering : OpConversionPattern<OpaqueCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpaqueCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> convertedArgs;
    for (Value arg : op.getArgOperands()) {
      Type ty = arg.getType();
      if (ty.isF32()) {
        auto cast = UnrealizedConversionCastOp::create(
            rewriter, op.getLoc(), rewriter.getI32Type(), arg);
        convertedArgs.push_back(cast.getResult(0));
      } else if (ty.isBF16()) {
        auto cast = UnrealizedConversionCastOp::create(
            rewriter, op.getLoc(), rewriter.getI16Type(), arg);
        convertedArgs.push_back(cast.getResult(0));
      } else {
        convertedArgs.push_back(arg);
      }
    }

    // Map result types: float results also need integer bit-pattern types
    // so the emitc.call_opaque produces EmitC-legal types.
    SmallVector<Type> resultTypes;
    for (Type resTy : op.getResultTypes()) {
      if (resTy.isF32()) {
        resultTypes.push_back(rewriter.getI32Type());
      } else if (resTy.isBF16() || resTy.isF16()) {
        resultTypes.push_back(rewriter.getI16Type());
      } else {
        resultTypes.push_back(resTy);
      }
    }

    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, op.getLoc(), resultTypes, op.getCallee(),
        /*args=*/nullptr, /*templateArgs=*/nullptr, convertedArgs);

    // If the original op had float results, cast them back.
    SmallVector<Value> results;
    for (auto [idx, resTy] : llvm::enumerate(op.getResultTypes())) {
      if (resTy.isF32() || resTy.isBF16() || resTy.isF16()) {
        auto cast = UnrealizedConversionCastOp::create(
            rewriter, op.getLoc(), resTy, callOp.getResult(idx));
        results.push_back(cast.getResult(0));
      } else {
        results.push_back(callOp.getResult(idx));
      }
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

struct TTLLowerOpaqueCallToEmitCPass
    : impl::TTLLowerOpaqueCallToEmitCBase<TTLLowerOpaqueCallToEmitCPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ModuleOp mod = getOperation();

    // Collect unique headers referenced by opaque_call ops.
    llvm::SetVector<StringRef> headers;
    mod.walk([&](OpaqueCallOp op) { headers.insert(op.getHeader()); });

    if (headers.empty()) {
      return;
    }

    // Emit #include directives at module scope, before the first function.
    {
      OpBuilder builder(&ctx);
      builder.setInsertionPointToStart(mod.getBody());
      for (StringRef header : headers) {
        emitc::IncludeOp::create(builder, mod.getLoc(), header,
                                 /*is_standard_include=*/false);
      }
    }

    ConversionTarget target(ctx);
    target.addIllegalOp<OpaqueCallOp>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&ctx);
    patterns.add<OpaqueCallLowering>(&ctx);

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
