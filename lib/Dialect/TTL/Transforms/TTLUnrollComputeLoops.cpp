// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#define DEBUG_TYPE "ttl-unroll-compute-loops"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLUNROLLCOMPUTELOOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static uint32_t countDSTSlotsInLoop(scf::ForOp forOp) {
  uint32_t maxDstIdx = 0;
  forOp.getBody()->walk([&](Operation *op) {
    if (auto dstIdx = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
      maxDstIdx = std::max(maxDstIdx, static_cast<uint32_t>(dstIdx.getInt()));
    }
    if (auto copyOp = dyn_cast<CopyTileOp>(op)) {
      Value dstIdxOperand = copyOp.getDstIndex();
      if (auto constOp = dstIdxOperand.getDefiningOp<arith::ConstantIndexOp>()) {
        maxDstIdx = std::max(maxDstIdx, static_cast<uint32_t>(constOp.value()));
      }
    }
  });
  return maxDstIdx + 1;
}

static void annotateDSTIndices(Operation *op, unsigned unrollIdx,
                               uint32_t slotsPerIteration, OpBuilder &b) {
  auto baseIdxAttr = op->getAttrOfType<IntegerAttr>("base_dst_idx");
  if (!baseIdxAttr)
    return;

  int64_t baseIdx = baseIdxAttr.getInt();
  int64_t newIdx = baseIdx + unrollIdx * slotsPerIteration;

  if (op->hasAttr(kDstIdxAttrName)) {
    op->setAttr(kDstIdxAttrName, b.getI32IntegerAttr(newIdx));
  }

  if (auto copyOp = dyn_cast<CopyTileOp>(op)) {
    OpBuilder localBuilder(copyOp);
    Value newConst = localBuilder.create<arith::ConstantIndexOp>(copyOp.getLoc(), newIdx);
    copyOp.getDstIndexMutable().assign(newConst);
  }
}

struct TTLUnrollComputeLoopsPass
    : public impl::TTLUnrollComputeLoopsBase<TTLUnrollComputeLoopsPass> {
  using Base = impl::TTLUnrollComputeLoopsBase<TTLUnrollComputeLoopsPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<scf::ForOp> loopsToUnroll;
    func.walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(kUnrollFactorAttrName))
        loopsToUnroll.push_back(forOp);
    });

    for (scf::ForOp forOp : loopsToUnroll) {
      auto unrollAttr = forOp->getAttrOfType<IntegerAttr>(kUnrollFactorAttrName);
      uint64_t factor = unrollAttr.getInt();

      uint32_t slotsPerIteration = countDSTSlotsInLoop(forOp);

      // Pre-annotate ops with their base DST indices before unrolling.
      OpBuilder builder(forOp.getContext());
      forOp.getBody()->walk([&](Operation *op) {
        if (auto dstIdx = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
          op->setAttr("base_dst_idx", dstIdx);
        } else if (auto copyOp = dyn_cast<CopyTileOp>(op)) {
          Value dstIdx = copyOp.getDstIndex();
          if (auto constOp = dstIdx.getDefiningOp<arith::ConstantIndexOp>()) {
            copyOp->setAttr("base_dst_idx",
                            builder.getI64IntegerAttr(constOp.value()));
          }
        }
      });

      auto annotateFn = [slotsPerIteration](unsigned unrollIdx, Operation *op,
                                             OpBuilder b) {
        annotateDSTIndices(op, unrollIdx, slotsPerIteration, b);
      };

      (void)loopUnrollByFactor(forOp, factor, annotateFn);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
