// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTLLowerMatmulBlock Pass
//===----------------------------------------------------------------------===//
//
// Marks matmul computes for block-level lowering by collapsing the 3D
// iteration space to a single point. The tile_matmul_block op stays inside
// the compute, but the iteration domain becomes [1, 1] so lower-to-loops
// generates no per-tile iteration.
//
// The block dimensions (M, N, K) are preserved in the matmul_block op's
// enclosing CB shapes and are derived by the TTKernel lowering.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "ttl-lower-matmul-block"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLLOWERMATMULBLOCK
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Attribute name marking a compute as block-matmul-lowered. When set,
/// lower-to-loops generates a single-iteration loop nest (trip count 1 on
/// each dimension) rather than iterating per tile.
constexpr llvm::StringLiteral kBlockMatmulAttrName("ttl.block_matmul");

/// Find the TileMatmulBlockOp inside a compute body, or return nullptr.
static TileMatmulBlockOp findMatmulBlock(ComputeOp computeOp) {
  TileMatmulBlockOp result;
  computeOp.getBody().walk([&](TileMatmulBlockOp op) {
    result = op;
    return WalkResult::interrupt();
  });
  return result;
}

/// Collapse a matmul compute's iteration space for block-level lowering.
///
/// Rewrites the compute op to use [1, 1] output tensors with identity maps,
/// so lower-to-loops sees a trivial iteration domain (one point). The real
/// tensor shapes are preserved on the inputs (for CB sizing) and in the
/// store view (for pack index computation). Sets kBlockMatmulAttrName so
/// downstream passes know this is a block matmul.
struct CollapseMatmulIteration : OpRewritePattern<ComputeOp> {
  using OpRewritePattern<ComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComputeOp computeOp,
                                PatternRewriter &rewriter) const override {
    // Skip non-matmul computes and already-lowered ones.
    if (!findMatmulBlock(computeOp)) {
      return failure();
    }
    if (computeOp->hasAttr(kBlockMatmulAttrName)) {
      return failure();
    }

    auto outType = cast<RankedTensorType>(computeOp.getOutputs()[0].getType());
    int64_t M = outType.getDimSize(0);
    int64_t N = outType.getDimSize(1);

    // DST capacity check. TODO: subblocking.
    // TODO: query from device config.
    int64_t dstCapacity = 8;
    if (M * N > dstCapacity) {
      return rewriter.notifyMatchFailure(
          computeOp,
          "matmul output " + llvm::Twine(M) + "x" + llvm::Twine(N) +
              " exceeds DST capacity; subblocking not yet implemented");
    }

    Location loc = computeOp.getLoc();

    // Expand the single tile_store into M*N stores with explicit indices.
    // TODO: replace with pack_tile_block when available.
    Block &body = computeOp.getBody().front();
    SmallVector<TileStoreOp> stores;
    body.walk([&](TileStoreOp store) { stores.push_back(store); });

    if (stores.size() == 1 && M * N > 1) {
      TileStoreOp origStore = stores[0];
      Value tileResult = origStore.getTile();
      Value view = origStore.getView();

      rewriter.setInsertionPoint(origStore);
      int64_t dstIdx = 0;
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
          Value mIdx = arith::ConstantIndexOp::create(rewriter, loc, m);
          Value nIdx = arith::ConstantIndexOp::create(rewriter, loc, n);
          auto store = TileStoreOp::create(rewriter, loc, tileResult, view,
                                           ValueRange{mIdx, nIdx});
          // Each output tile occupies a distinct DST register [0..M*N-1].
          store->setAttr(kDstIdxAttrName,
                         rewriter.getI32IntegerAttr(dstIdx++));
        }
      }
      rewriter.eraseOp(origStore);
    }

    // Mark the compute as block-matmul.
    rewriter.modifyOpInPlace(computeOp, [&] {
      computeOp->setAttr(kBlockMatmulAttrName, rewriter.getUnitAttr());
    });

    return success();
  }
};

struct TTLLowerMatmulBlockPass
    : public impl::TTLLowerMatmulBlockBase<TTLLowerMatmulBlockPass> {
  using TTLLowerMatmulBlockBase::TTLLowerMatmulBlockBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    patterns.add<CollapseMatmulIteration>(func.getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
