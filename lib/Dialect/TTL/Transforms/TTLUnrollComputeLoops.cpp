// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#define DEBUG_TYPE "ttl-unroll-compute-loops"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLUNROLLCOMPUTELOOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Count DST slots used in one loop iteration by scanning dst_idx attrs.
static uint32_t countDSTSlotsInLoop(scf::ForOp forOp) {
  uint32_t maxDstIdx = 0;
  forOp.getBody()->walk([&](Operation *op) {
    if (auto dstIdx = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
      maxDstIdx = std::max<uint32_t>(maxDstIdx, dstIdx.getInt() + 1);
    }
  });
  return maxDstIdx;
}

/// Update dst_idx attribute and copy_tile dst_index operand for an unrolled
/// copy. Input DST registers (idx < numInputs) stay fixed across iterations.
/// Output DST registers (idx >= numInputs) increment per iteration.
static void updateDSTIndices(Operation *op, unsigned unrollIdx,
                             uint32_t numInputs, uint32_t numOutputs,
                             OpBuilder &b) {
  Location loc = op->getLoc();

  if (auto dstIdx = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
    int32_t oldIdx = dstIdx.getInt();
    int32_t newIdx;

    if (oldIdx < static_cast<int32_t>(numInputs)) {
      // Input DST: keep fixed across iterations (A always uses DST[0], B uses
      // DST[1])
      newIdx = oldIdx;
    } else {
      // Output DST: increment by iteration (iteration k uses DST[numInputs +
      // k])
      newIdx = numInputs + (oldIdx - numInputs) + unrollIdx * numOutputs;
    }

    op->setAttr(kDstIdxAttrName, b.getI32IntegerAttr(newIdx));
  }

  if (auto copyOp = dyn_cast<CopyTileOp>(op)) {
    Value oldDstIdx = copyOp.getDstIndex();
    if (auto constOp = oldDstIdx.getDefiningOp<arith::ConstantIndexOp>()) {
      int64_t oldIdx = constOp.value();
      int64_t newIdx;

      if (oldIdx < static_cast<int64_t>(numInputs)) {
        // Input DST: keep fixed
        newIdx = oldIdx;
      } else {
        // Output DST: increment per iteration
        newIdx = numInputs + (oldIdx - numInputs) + unrollIdx * numOutputs;
      }

      // Set insertion point right before the copy_tile op to ensure domination
      b.setInsertionPoint(op);
      Value newDstIdxVal = b.create<arith::ConstantIndexOp>(loc, newIdx);
      op->setOperand(2, newDstIdxVal);
    }
  }
}

struct TTLUnrollComputeLoopsPass
    : public impl::TTLUnrollComputeLoopsBase<TTLUnrollComputeLoopsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<scf::ForOp> loopsToUnroll;

    func.walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(kUnrollFactorAttrName)) {
        loopsToUnroll.push_back(forOp);
      }
    });

    for (scf::ForOp forOp : loopsToUnroll) {
      auto unrollAttr =
          forOp->getAttrOfType<IntegerAttr>(kUnrollFactorAttrName);
      if (!unrollAttr) {
        continue;
      }

      uint64_t factor = unrollAttr.getInt();
      if (factor <= 1) {
        continue;
      }

      // Get numInputs from the loop attribute (set by DST assignment pass).
      // Input DST registers stay fixed across unrolled iterations.
      auto numInputsAttr = forOp->getAttrOfType<IntegerAttr>("ttl.num_inputs");
      uint32_t numInputs = numInputsAttr ? numInputsAttr.getInt() : 0;

      // Compute total slots and outputs.
      uint32_t slotsPerIteration = countDSTSlotsInLoop(forOp);
      uint32_t numOutputs =
          (slotsPerIteration > numInputs) ? (slotsPerIteration - numInputs) : 0;

      auto annotateFn = [numInputs, numOutputs](unsigned unrollIdx,
                                                Operation *op, OpBuilder b) {
        updateDSTIndices(op, unrollIdx, numInputs, numOutputs, b);
      };

      LogicalResult unrollResult =
          loopUnrollByFactor(forOp, factor, annotateFn);

      // After unrolling, group all tensor.insert operations at the end
      // (before the terminator) so sync ops can be inserted before them.
      // Only do this if the loop still exists (wasn't fully eliminated).
      // Check if the loop operation still has a parent block (not erased).
      if (succeeded(unrollResult) && forOp->getBlock() != nullptr) {
        Block &body = forOp.getRegion().front();
        auto *terminator = body.getTerminator();

        // Collect all tensor.insert operations that aren't already at the end
        SmallVector<Operation *> tensorInsertOps;
        for (Operation &op : body.without_terminator()) {
          if (op.getName().getStringRef() == "tensor.insert") {
            tensorInsertOps.push_back(&op);
          }
        }

        // Move them to just before the yield, preserving their relative order
        for (Operation *insertOp : tensorInsertOps) {
          insertOp->moveBefore(terminator);
        }
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
