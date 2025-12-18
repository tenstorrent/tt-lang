// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

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

} // namespace

void populateTTKernelCleanupPatterns(RewritePatternSet &patterns) {
  patterns.add<DeduplicateConsecutiveBarriers<NocAsyncReadBarrierOp>>(
      patterns.getContext());
  patterns.add<DeduplicateConsecutiveBarriers<NocAsyncWriteBarrierOp>>(
      patterns.getContext());
}

} // namespace mlir::tt::ttkernel
