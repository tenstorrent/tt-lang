// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTLLowerMatmulBlock Pass
//===----------------------------------------------------------------------===//
//
// Replaces ttl.compute ops containing tile_matmul_block with a linear
// sequence: sync acquire, matmul_block, M*N tile_stores, sync release.
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
    if (!computeOp.containsOp<TileMatmulBlockOp>()) {
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
          << "; enable maximize_dst to auto-subblock";
      hadError = true;
    }
  });
  return hadError ? failure() : success();
}

/// Trace a value through copy_tile (inserted by assign-dst) to its source
/// block argument. Returns the block arg index, or std::nullopt if the value
/// does not trace to a block argument.
static std::optional<unsigned> traceToBlockArgIndex(Value v) {
  if (auto copyOp = v.getDefiningOp<CopyTileOp>()) {
    v = copyOp.getSrc();
  }
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    return blockArg.getArgNumber();
  }
  return std::nullopt;
}

/// Emit M*N copies of an in-place unary tile op (relu, exp, etc.) by cloning
/// the body op. Each copy operates on DST[m*N + n].
static void emitPerTileUnaryOps(OpBuilder &rewriter, Location loc,
                                Operation *bodyOp, Value placeholder, int64_t M,
                                int64_t N) {
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      int64_t dstIdx = m * N + n;
      auto *cloned = rewriter.clone(*bodyOp);
      // The body is erased after expansion; use placeholder for DST reference.
      cloned->setOperand(0, placeholder);
      cloned->setAttr(kDstIdxAttrName, rewriter.getI32IntegerAttr(dstIdx));
    }
  }
}

/// Replace a matmul compute with a linear sequence of tile-level ops:
/// sync acquire, copy_tiles (accumulator), matmul_block, unary post-ops,
/// sync commit/wait, tile_stores, sync release.
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

    // Get the store view from the body's tile_store. For non-subblocked
    // computes, this is the cb_reserve result. For subblocked computes,
    // this is the per-subblock cb_reserve result (inside the subblock
    // loop), which has the subblock shape for correct local linearization.
    SmallVector<TileStoreOp> stores;
    computeOp.getBody().walk(
        [&](TileStoreOp store) { stores.push_back(store); });
    if (stores.empty()) {
      return rewriter.notifyMatchFailure(computeOp, "no tile_store in body");
    }
    Value outView = stores[0].getView();

    // Collect in-place unary ops between matmul and store for M*N expansion.
    SmallVector<Operation *> postMatmulUnaryOps;
    bool foundMatmul = false;
    for (Operation &bodyOp : computeOp.getBody().front()) {
      if (isa<TileMatmulBlockOp>(&bodyOp)) {
        foundMatmul = true;
        continue;
      }
      if (foundMatmul && bodyOp.hasTrait<TTLInPlaceOpTrait>()) {
        postMatmulUnaryOps.push_back(&bodyOp);
      }
    }

    // Map matmul body operands to compute input tensors via block arg indices.
    // The body's block arg order matches the compute's input order, but the
    // matmul's lhs/rhs may not be at indices 0/1 (e.g., fused computes place
    // the accumulator operand first).
    auto getInputForBodyOperand = [&](Value bodyVal) -> Value {
      auto idx = traceToBlockArgIndex(bodyVal);
      return idx ? computeOp.getInputs()[*idx] : Value();
    };

    Value lhsTensor = getInputForBodyOperand(mmOp.getLhs());
    Value rhsTensor = getInputForBodyOperand(mmOp.getRhs());
    assert(lhsTensor && rhsTensor && "matmul operands must trace to inputs");

    // Accumulator (3rd operand) maps to a compute input if present.
    Value accTensor;
    if (Value acc = mmOp.getAccumulator()) {
      auto accIdx = traceToBlockArgIndex(acc);
      assert(accIdx && *accIdx < computeOp.getInputs().size() &&
             "accumulator must trace to a compute input");
      accTensor = computeOp.getInputs()[*accIdx];
    }

    Location loc = computeOp.getLoc();
    Type tileType = mmOp.getResult().getType();

    rewriter.setInsertionPoint(computeOp);

    // Sync acquire.
    TileRegsAcquireOp::create(rewriter, loc);

    // Matmul_block with optional accumulator. TTKernel lowering emits
    // individual copy_tile ops for the accumulator load.
    auto mmResult = TileMatmulBlockOp::create(rewriter, loc, tileType,
                                              lhsTensor, rhsTensor, accTensor);
    mmResult->setAttr(kDstIdxAttrName, rewriter.getI32IntegerAttr(0));

    // Per-tile unary post-ops (relu, exp, etc.).
    Value placeholder = UnrealizedConversionCastOp::create(
                            rewriter, loc, tileType, ValueRange{})
                            .getResult(0);
    for (Operation *unaryOp : postMatmulUnaryOps) {
      emitPerTileUnaryOps(rewriter, loc, unaryOp, placeholder, M, N);
    }

    // Sync commit + wait (math -> pack boundary).
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);

    // M*N individual tile_store ops. The combine-pack-tiles pass can
    // optionally consolidate these into pack_tile_block downstream.
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        Value mIdx = arith::ConstantIndexOp::create(rewriter, loc, m);
        Value nIdx = arith::ConstantIndexOp::create(rewriter, loc, n);
        auto store = TileStoreOp::create(rewriter, loc, placeholder, outView,
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
