// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Rewrite Reduce Scalers
//===----------------------------------------------------------------------===//
//
// Splits any `ttl.reduce` whose scaler is not `fill(1.0)` into a reduce
// (with scaler `fill(1.0)`) followed by an explicit post-reduce multiply
// op. The LLK's REDUCE_SCALAR mode applies the scaler tile twice
// internally (see tt-metal's reduce_op.cpp); feeding `fill(1.0)` makes
// that double-apply a no-op, and the extracted multiply applies the
// actual scaler once. The pass is purely a semantic rewrite; DFB
// materialization of the new ops happens later in
// `ttl-insert-intermediate-dfbs`.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-rewrite-reduce-scalers"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLREWRITEREDUCESCALERS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Returns true if `dimsAttr` reduces every axis of a rank-`inputRank`
/// input (REDUCE_SCALAR mode in the LLK). Negative dims are normalized.
bool reducesAllDims(ArrayRef<int64_t> dimsAttr, int64_t inputRank) {
  llvm::SmallSet<int64_t, 2> normDims;
  for (int64_t dim : dimsAttr) {
    normDims.insert((dim % inputRank + inputRank) % inputRank);
  }
  return static_cast<int64_t>(normDims.size()) == inputRank;
}

/// Rewrites `ttl.reduce(x, scaler)` so the scaler operand of the reduce
/// itself is always `fill(1.0)`, and the actual scaler is applied via a
/// separate post-reduce multiply.
///
/// Two post-multiply forms are generated:
///   - Statically-known `fill(c)` scaler: emit
///     `ttl.mul_unary_const(result, c)`. Lowers to LLK `mul_unary_tile`
///     with an i32 bit pattern of `c`; no extra DFB needed.
///   - Any other scaler tile (user tile-form, value produced by a prior
///     op, etc.) AND `dims=[0,1]`: emit `ttl.mul(result, original_scaler)`.
///     Element-wise tile multiply; correct at position [0, 0] of the
///     REDUCE_SCALAR output (`result[0,0] = intermediate[0,0] * scaler[0,0]`),
///     and only [0, 0] is read.
///
/// Non-fill scalers on single-dim reduces (REDUCE_COL / REDUCE_ROW) are
/// not rewritten — the LLK applies the scaler correctly there.
///
/// One neutral fill is cached per source FillOp so multiple reduces
/// sharing the same compile-time scaler converge on the same SSA value.
void rewriteReduceScalers(func::FuncOp funcOp, OpBuilder &builder) {
  SmallVector<ReduceOp> reducesToRewrite;
  funcOp.walk([&](ReduceOp reduceOp) {
    Value scaler = reduceOp.getScaler();
    if (auto fillOp = scaler.getDefiningOp<FillOp>()) {
      float c = fillOp.getValueAttr().getValue().convertToFloat();
      if (c != 1.0f) {
        reducesToRewrite.push_back(reduceOp);
      }
      return;
    }
    auto inputType = cast<RankedTensorType>(reduceOp.getInput().getType());
    if (reducesAllDims(reduceOp.getDims(), inputType.getRank())) {
      reducesToRewrite.push_back(reduceOp);
    }
  });

  llvm::DenseMap<FillOp, Value> neutralFillFor;
  for (ReduceOp reduceOp : reducesToRewrite) {
    Location loc = reduceOp.getLoc();
    Value scaler = reduceOp.getScaler();
    auto fillOp = scaler.getDefiningOp<FillOp>();

    if (fillOp) {
      float c = fillOp.getValueAttr().getValue().convertToFloat();
      auto [iter, inserted] = neutralFillFor.try_emplace(fillOp, Value{});
      if (inserted) {
        builder.setInsertionPointAfter(fillOp);
        auto neutral =
            FillOp::create(builder, fillOp.getLoc(), fillOp.getType(),
                           builder.getF32FloatAttr(1.0));
        iter->second = neutral.getResult();
      }
      reduceOp.getScalerMutable().assign(iter->second);
      builder.setInsertionPointAfter(reduceOp);
      auto mulOp = MulUnaryConstOp::create(
          builder, loc, reduceOp.getResult().getType(), reduceOp.getResult(),
          builder.getF32FloatAttr(c));
      reduceOp.getResult().replaceAllUsesExcept(mulOp.getResult(), mulOp);
    } else {
      builder.setInsertionPoint(reduceOp);
      auto neutral =
          FillOp::create(builder, loc, cast<RankedTensorType>(scaler.getType()),
                         builder.getF32FloatAttr(1.0));
      reduceOp.getScalerMutable().assign(neutral.getResult());
      builder.setInsertionPointAfter(reduceOp);
      auto mulOp = MulOp::create(builder, loc, reduceOp.getResult().getType(),
                                 reduceOp.getResult(), scaler);
      reduceOp.getResult().replaceAllUsesExcept(mulOp.getResult(), mulOp);
    }
  }

  // Erase FillOps with no remaining uses after scaler substitution.
  funcOp.walk([](FillOp fillOp) {
    if (fillOp->use_empty()) {
      fillOp.erase();
    }
  });
}

struct TTLRewriteReduceScalersPass
    : public impl::TTLRewriteReduceScalersBase<TTLRewriteReduceScalersPass> {
  using TTLRewriteReduceScalersBase::TTLRewriteReduceScalersBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());
    rewriteReduceScalers(funcOp, builder);
  }
};

} // namespace

} // namespace mlir::tt::ttl
