// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTLLowerMatmulBlock Pass
//===----------------------------------------------------------------------===//
//
// Replaces ttl.compute ops containing tile_matmul_block with a flat sequence:
// sync acquire, matmul_block, M*N tile_stores, sync release.
//
// CB lifecycle (wait/pop for inputs, reserve/push for output) is NOT emitted
// here — it comes from the user's DFB operations outside the compute.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "ttl-lower-matmul-block"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLLOWERMATMULBLOCK
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static TileMatmulBlockOp findMatmulBlock(ComputeOp computeOp) {
  TileMatmulBlockOp result;
  computeOp.getBody().walk([&](TileMatmulBlockOp op) {
    result = op;
    return WalkResult::interrupt();
  });
  return result;
}

/// Validate that all matmul computes fit within DST capacity.
/// Returns failure and emits diagnostics for any that exceed the limit.
static LogicalResult validateMatmulDSTCapacity(func::FuncOp func) {
  bool hadError = false;
  func.walk([&](ComputeOp computeOp) {
    if (!findMatmulBlock(computeOp)) {
      return;
    }
    auto capacityOrErr = computeDSTCapacity(computeOp);
    if (failed(capacityOrErr)) {
      hadError = true;
      return;
    }
    auto outType = cast<RankedTensorType>(computeOp.getOutputs()[0].getType());
    int64_t M = outType.getDimSize(0);
    int64_t N = outType.getDimSize(1);
    int64_t dstCapacity = static_cast<int64_t>(*capacityOrErr);
    if (M * N > dstCapacity) {
      computeOp.emitOpError()
          << "matmul output " << M << "x" << N << " = " << M * N
          << " tiles exceeds DST capacity of " << dstCapacity
          << "; automatic subblocking is not yet implemented";
      hadError = true;
    }
  });
  return hadError ? failure() : success();
}

/// Replace a matmul compute with flat ops: sync + matmul_block + stores.
struct LowerMatmulBlockCompute : OpRewritePattern<ComputeOp> {
  using OpRewritePattern<ComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComputeOp computeOp,
                                PatternRewriter &rewriter) const override {
    auto mmOp = findMatmulBlock(computeOp);
    if (!mmOp) {
      return failure();
    }

    assert(computeOp.getInputs().size() >= 2 &&
           "matmul compute must have at least 2 inputs (lhs, rhs)");

    auto outType = cast<RankedTensorType>(computeOp.getOutputs()[0].getType());
    int64_t M = outType.getDimSize(0);
    int64_t N = outType.getDimSize(1);

    // Find the store view for tile_stores.
    SmallVector<TileStoreOp> stores;
    computeOp.getBody().walk(
        [&](TileStoreOp store) { stores.push_back(store); });
    if (stores.empty()) {
      return rewriter.notifyMatchFailure(computeOp, "no tile_store in body");
    }
    Value outView = stores[0].getView();

    Location loc = computeOp.getLoc();
    Type tileType = mmOp.getResult().getType();

    rewriter.setInsertionPoint(computeOp);

    // Sync acquire.
    TileRegsAcquireOp::create(rewriter, loc);

    // Single matmul_block call. Operands are the CB-attached input tensors.
    auto mmResult = TileMatmulBlockOp::create(rewriter, loc, tileType,
                                              computeOp.getInputs()[0],
                                              computeOp.getInputs()[1]);
    mmResult->setAttr(kDstIdxAttrName, rewriter.getI32IntegerAttr(0));

    // Sync commit + wait.
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);

    // M*N tile_stores. TODO: replace with pack_tile_block.
    Value mmTile = UnrealizedConversionCastOp::create(rewriter, loc, tileType,
                                                      ValueRange{})
                       .getResult(0);
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        Value mIdx = arith::ConstantIndexOp::create(rewriter, loc, m);
        Value nIdx = arith::ConstantIndexOp::create(rewriter, loc, n);
        auto store = TileStoreOp::create(rewriter, loc, mmTile, outView,
                                         ValueRange{mIdx, nIdx});
        store->setAttr(kDstIdxAttrName, rewriter.getI32IntegerAttr(m * N + n));
      }
    }

    // Sync release.
    TileRegsReleaseOp::create(rewriter, loc);

    // Replace compute with placeholder tensor.
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outType.getShape(), outType.getElementType());
    rewriter.replaceOp(computeOp, emptyTensor);
    return success();
  }
};

struct TTLLowerMatmulBlockPass
    : public impl::TTLLowerMatmulBlockBase<TTLLowerMatmulBlockPass> {
  using TTLLowerMatmulBlockBase::TTLLowerMatmulBlockBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(validateMatmulDSTCapacity(func))) {
      return signalPassFailure();
    }
    RewritePatternSet patterns(func.getContext());
    patterns.add<LowerMatmulBlockCompute>(func.getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
