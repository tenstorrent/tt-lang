// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttkernel {

namespace {

/// Deduplicate consecutive barriers of the same type. Global barriers wait for
/// all outstanding transactions, so multiple consecutive barriers are
/// redundant.
template <typename BarrierOp>
struct DeduplicateConsecutiveBarriers : OpRewritePattern<BarrierOp> {
  using OpRewritePattern<BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter &rewriter) const override {
    if (auto *prev = op->getPrevNode()) {
      if (isa<BarrierOp>(prev)) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

/// Deduplicate consecutive TRID barriers of the same type *only* when they
/// target the same TRID (and optional NOC). Unlike global barriers, barriers
/// with different TRIDs are not redundant and must not be removed.
template <typename BarrierWithTridOp>
struct DeduplicateConsecutiveTridBarriers
    : OpRewritePattern<BarrierWithTridOp> {
  using OpRewritePattern<BarrierWithTridOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierWithTridOp op,
                                PatternRewriter &rewriter) const override {
    auto *prev = op->getPrevNode();
    if (!prev) {
      return failure();
    }
    auto prevBarrier = dyn_cast<BarrierWithTridOp>(prev);
    if (!prevBarrier) {
      return failure();
    }

    if (op->getNumOperands() != prevBarrier->getNumOperands()) {
      return failure();
    }

    for (auto [a, b] :
         llvm::zip_equal(op->getOperands(), prevBarrier->getOperands())) {
      if (a != b) {
        return failure();
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTTKernelCleanupPatterns(RewritePatternSet &patterns) {
  patterns.add<DeduplicateConsecutiveBarriers<NocAsyncReadBarrierOp>>(
      patterns.getContext());
  patterns.add<DeduplicateConsecutiveBarriers<NocAsyncWriteBarrierOp>>(
      patterns.getContext());
  patterns
      .add<DeduplicateConsecutiveTridBarriers<NocAsyncReadBarrierWithTridOp>>(
          patterns.getContext());
  patterns
      .add<DeduplicateConsecutiveTridBarriers<NocAsyncWriteBarrierWithTridOp>>(
          patterns.getContext());
}

} // namespace mlir::tt::ttkernel
