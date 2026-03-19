// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTLLowerMatmulBlock Pass
//===----------------------------------------------------------------------===//
//
// Replaces ttl.compute ops containing tile_matmul_block with the flat
// tt-metal matmul_block pattern: sync acquire, K-loop with per-step CB
// wait/pop and matmul_block(kt_dim=1), M*N tile_stores, sync release.
//
// This pass fully lowers matmul computes. The resulting IR contains no
// ttl.compute — only flat TTL ops (sync, CB lifecycle, matmul_block,
// tile_store) and scf.for for the K loop. lower-to-loops does not need
// to handle these computes.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

/// Replace a matmul compute with the flat tt-metal matmul_block pattern.
struct LowerMatmulBlockCompute : OpRewritePattern<ComputeOp> {
  using OpRewritePattern<ComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComputeOp computeOp,
                                PatternRewriter &rewriter) const override {
    if (!findMatmulBlock(computeOp)) {
      return failure();
    }

    auto lhsType = cast<RankedTensorType>(computeOp.getInputs()[0].getType());
    auto rhsType = cast<RankedTensorType>(computeOp.getInputs()[1].getType());
    auto outType = cast<RankedTensorType>(computeOp.getOutputs()[0].getType());
    int64_t M = lhsType.getDimSize(0);
    int64_t K = lhsType.getDimSize(1);
    int64_t N = rhsType.getDimSize(1);

    // DST capacity check. TODO: subblocking.
    int64_t dstCapacity = 8;
    if (M * N > dstCapacity) {
      return rewriter.notifyMatchFailure(
          computeOp, "matmul output " + llvm::Twine(M) + "x" + llvm::Twine(N) +
                         " exceeds DST capacity; subblocking not implemented");
    }

    // Find CBs from the compute's operands.
    Value lhsCb = getAttachedCB(computeOp.getInputs()[0]);
    Value rhsCb = getAttachedCB(computeOp.getInputs()[1]);
    Value outCb = getAttachedCB(computeOp.getOutputs()[0]);
    if (!lhsCb || !rhsCb || !outCb) {
      return rewriter.notifyMatchFailure(computeOp, "missing attached CBs");
    }

    // Find the output view (from cb_reserve) for tile_stores.
    SmallVector<TileStoreOp> stores;
    computeOp.getBody().walk(
        [&](TileStoreOp store) { stores.push_back(store); });
    if (stores.empty()) {
      return rewriter.notifyMatchFailure(computeOp, "no tile_store in body");
    }
    Value outView = stores[0].getView();

    Location loc = computeOp.getLoc();
    Type tileType = findMatmulBlock(computeOp).getResult().getType();

    rewriter.setInsertionPoint(computeOp);

    // Output CB reserve.
    CBReserveOp::create(rewriter, loc, outType, outCb);

    // Sync acquire — DST persists across all K iterations.
    TileRegsAcquireOp::create(rewriter, loc);

    // K loop.
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value kBound = arith::ConstantIndexOp::create(rewriter, loc, K);

    scf::ForOp kLoop = scf::ForOp::create(rewriter, loc, c0, kBound, c1);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(kLoop.getBody());

      // CB wait for per-K-step input tiles.
      CBWaitOp::create(rewriter, loc, lhsType, lhsCb);
      CBWaitOp::create(rewriter, loc, rhsType, rhsCb);

      // matmul_block: tile_matmul_block reads from CBs directly.
      // The TTKernel lowering derives block dims from operand shapes.
      auto mmResult = TileMatmulBlockOp::create(
          rewriter, loc, tileType,
          // Operands are the CB-attached input tensors. The TTKernel
          // lowering traces through tensor.extract/unrealized_cast to
          // find the CB. Pass the compute's input values directly.
          computeOp.getInputs()[0], computeOp.getInputs()[1]);
      mmResult->setAttr(kDstIdxAttrName, rewriter.getI32IntegerAttr(0));

      // CB pop per K-step.
      CBPopOp::create(rewriter, loc, lhsCb);
      CBPopOp::create(rewriter, loc, rhsCb);
    }

    // Sync commit + wait (between math and pack phases).
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);

    // M*N tile_stores (pack DST registers to output CB).
    // TODO: replace with pack_tile_block when available.
    // The tile_store's dst_idx attribute determines which DST register to pack.
    // The tile SSA value is a placeholder (the real DST index is in the attr).
    Value mmTile = UnrealizedConversionCastOp::create(rewriter, loc, tileType,
                                                      ValueRange{})
                       .getResult(0);
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        Value mIdx = arith::ConstantIndexOp::create(rewriter, loc, m);
        Value nIdx = arith::ConstantIndexOp::create(rewriter, loc, n);
        int64_t dstIdx = m * N + n;
        auto store = TileStoreOp::create(rewriter, loc, mmTile, outView,
                                         ValueRange{mIdx, nIdx});
        store->setAttr(kDstIdxAttrName, rewriter.getI32IntegerAttr(dstIdx));
      }
    }

    // Sync release.
    TileRegsReleaseOp::create(rewriter, loc);

    // Output CB push.
    CBPushOp::create(rewriter, loc, outCb);

    // Replace compute result with empty tensor placeholder.
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
    RewritePatternSet patterns(func.getContext());
    patterns.add<LowerMatmulBlockCompute>(func.getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
