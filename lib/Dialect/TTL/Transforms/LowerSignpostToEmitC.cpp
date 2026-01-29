// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLLOWERSIGNPOSTTOEMITC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static void createEmitCVerbatim(Location loc, StringRef value,
                                ConversionPatternRewriter &rewriter) {
  OperationState state(loc, "emitc.verbatim");
  state.addAttribute("value", rewriter.getStringAttr(value));
  rewriter.create(state);
}

struct SignpostLowering : OpConversionPattern<SignpostOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SignpostOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Emit DeviceZoneScopedN wrapped in braces to avoid variable conflicts.
    auto loc = op.getLoc();
    createEmitCVerbatim(loc, "{", rewriter);
    createEmitCVerbatim(
        loc, "DeviceZoneScopedN(\"" + op.getName().str() + "\");", rewriter);
    createEmitCVerbatim(loc, "}", rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TTLLowerSignpostToEmitCPass
    : impl::TTLLowerSignpostToEmitCBase<TTLLowerSignpostToEmitCPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ModuleOp mod = getOperation();

    ConversionTarget target(ctx);
    target.addIllegalOp<SignpostOp>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&ctx);
    patterns.add<SignpostLowering>(&ctx);

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
