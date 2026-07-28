// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir::tt::ttkernel {

namespace {

/// Return whether `op` is a side-effect-free TTKernel value computation that
/// can be moved out of loops and single-region conditionals.
static bool isHoistableTTKernelValueComputation(Operation *op) {
  Dialect *dialect = op->getDialect();
  bool supportedDialect = isa<arith::ArithDialect, TTKernelDialect>(dialect);
  return supportedDialect && isPure(op);
}

/// Return whether `value` is defined outside `region`.
static bool isDefinedOutsideRegion(Value value, Region *region) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return true;
  }
  return !region->isAncestor(definingOp->getParentRegion());
}

/// Deduplicate consecutive barriers of the same type and NoC. Barriers only
/// wait for transactions issued on the selected NoC.
template <typename BarrierOp>
struct DeduplicateConsecutiveBarriers : OpRewritePattern<BarrierOp> {
  using OpRewritePattern<BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter &rewriter) const override {
    if (auto *prev = op->getPrevNode()) {
      if (auto previousBarrier = dyn_cast<BarrierOp>(prev)) {
        if (previousBarrier.getNoc() == op.getNoc()) {
          rewriter.eraseOp(op);
          return success();
        }
      }
    }
    return failure();
  }
};

/// Hoist pure value computations from `scf.if` regions when all operands are
/// already available before the conditional.
struct HoistIfRegionInvariantValueOps : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Region *> regions{&op.getThenRegion()};
    if (!op.getElseRegion().empty()) {
      regions.push_back(&op.getElseRegion());
    }
    size_t numMoved = moveLoopInvariantCode(
        regions,
        [](Value value, Region *region) {
          return isDefinedOutsideRegion(value, region);
        },
        [](Operation *candidate, Region *) {
          return isHoistableTTKernelValueComputation(candidate);
        },
        [&](Operation *candidate, Region *region) {
          rewriter.moveOpBefore(candidate, region->getParentOp());
        });
    return success(numMoved != 0);
  }
};

/// Hoist pure value computations from loops when their operands are not defined
/// by the loop body.
struct HoistLoopInvariantValueOps : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    LoopLikeOpInterface loop = cast<LoopLikeOpInterface>(op.getOperation());
    size_t numMoved = moveLoopInvariantCode(
        loop.getLoopRegions(),
        [&](Value value, Region *) {
          return loop.isDefinedOutsideOfLoop(value);
        },
        [](Operation *candidate, Region *) {
          return isHoistableTTKernelValueComputation(candidate);
        },
        [&](Operation *candidate, Region *) {
          rewriter.moveOpBefore(candidate, op);
        });
    return success(numMoved != 0);
  }
};

} // namespace

void populateTTKernelCleanupPatterns(RewritePatternSet &patterns) {
  patterns.add<DeduplicateConsecutiveBarriers<NocAsyncReadBarrierOp>>(
      patterns.getContext());
  patterns.add<DeduplicateConsecutiveBarriers<NocAsyncWriteBarrierOp>>(
      patterns.getContext());
  patterns.add<HoistIfRegionInvariantValueOps, HoistLoopInvariantValueOps>(
      patterns.getContext());
}

} // namespace mlir::tt::ttkernel
