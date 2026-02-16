// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Subblock Compute For DST Pass
//===----------------------------------------------------------------------===//
//
// Partitions ttl.compute operations into DST-sized subblocks. Uses the
// ttl.unroll_factor attribute (set by ttl-assign-dst) to partition the
// iteration space into subblocks. Each subblock becomes an inner ttl.compute
// that processes unroll_factor tiles per DST sync cycle.
//
// The outer loop is side-effect-only (no iter_args) because stores are
// explicit side effects (ttl.tile_store) referencing external reserve views.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-subblock-compute-for-dst"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLSUBBLOCKCOMPUTEFORDST
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLSubblockComputeForDSTPass
    : public impl::TTLSubblockComputeForDSTBase<TTLSubblockComputeForDSTPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Collect compute ops to tile (avoid modifying while walking).
    SmallVector<ComputeOp> opsToTile;
    funcOp.walk([&](ComputeOp computeOp) {
      auto unrollAttr =
          computeOp->getAttrOfType<IntegerAttr>("ttl.unroll_factor");
      if (unrollAttr && unrollAttr.getInt() > 1) {
        opsToTile.push_back(computeOp);
      }
    });

    for (ComputeOp computeOp : opsToTile) {
      if (failed(tileComputeOp(computeOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult tileComputeOp(ComputeOp computeOp) {
    auto unrollAttr =
        computeOp->getAttrOfType<IntegerAttr>("ttl.unroll_factor");
    int64_t unrollFactor = unrollAttr.getInt();
    Location loc = computeOp.getLoc();

    OpBuilder b(computeOp);
    SmallVector<Range> iterDomain = computeOp.getIterationDomain(b);
    if (iterDomain.empty()) {
      return computeOp.emitOpError("empty iteration domain");
    }

    // Tile the innermost dimension by unroll_factor.
    int64_t tilingDim = iterDomain.size() - 1;
    auto innerSizeAttr =
        dyn_cast<IntegerAttr>(iterDomain[tilingDim].size.dyn_cast<Attribute>());
    if (!innerSizeAttr) {
      return computeOp.emitOpError(
          "dynamic innermost dimension not supported for DST tiling");
    }
    int64_t innerSize = innerSizeAttr.getInt();

    // When unroll_factor == total tiles, no outer loop is needed -- the compute
    // op already fits in one DST sync cycle.
    int64_t totalTiles = 1;
    for (auto &range : iterDomain) {
      if (auto sizeAttr =
              dyn_cast<IntegerAttr>(range.size.dyn_cast<Attribute>())) {
        totalTiles *= sizeAttr.getInt();
      }
    }
    if (unrollFactor >= totalTiles) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping tiling: unroll_factor (" << unrollFactor
                 << ") >= total tiles (" << totalTiles << ")\n");
      return success();
    }

    // Create outer scf.for loop: side-effect-only (no iter_args).
    Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
    Value ub = b.create<arith::ConstantIndexOp>(loc, innerSize);
    Value step = b.create<arith::ConstantIndexOp>(loc, unrollFactor);

    auto outerLoop = b.create<scf::ForOp>(
        loc, lb, ub, step, ValueRange{},
        [&](OpBuilder &loopBuilder, Location loopLoc, Value iv,
            ValueRange /*iterArgs*/) {
          // Compute offsets and sizes for this tile block.
          SmallVector<OpFoldResult> offsets(iterDomain.size(),
                                           loopBuilder.getIndexAttr(0));
          SmallVector<OpFoldResult> sizes;
          for (auto &range : iterDomain) {
            sizes.push_back(range.size);
          }

          // Override the tiling dimension with the current block.
          offsets[tilingDim] = iv;

          // Handle remainder: min(unrollFactor, innerSize - iv).
          if (innerSize % unrollFactor != 0) {
            Value remaining = loopBuilder.create<arith::SubIOp>(
                loopLoc,
                loopBuilder.create<arith::ConstantIndexOp>(loopLoc, innerSize),
                iv);
            Value ufVal = loopBuilder.create<arith::ConstantIndexOp>(
                loopLoc, unrollFactor);
            Value blockSize =
                loopBuilder.create<arith::MinSIOp>(loopLoc, ufVal, remaining);
            sizes[tilingDim] = blockSize;
          } else {
            sizes[tilingDim] = loopBuilder.getIndexAttr(unrollFactor);
          }

          // Use TilingInterface to create the tiled compute op.
          auto tiledResult =
              computeOp.getTiledImplementation(loopBuilder, offsets, sizes);
          if (failed(tiledResult)) {
            return;
          }

          // Remove the unroll_factor attribute from the tiled inner compute.
          for (Operation *tiledOp : tiledResult->tiledOps) {
            tiledOp->removeAttr("ttl.unroll_factor");
          }

          loopBuilder.create<scf::YieldOp>(loopLoc);
        });

    // Replace the original compute op with its output operands.
    // The outer loop is side-effect-only; results flow through tile_store.
    computeOp.replaceAllUsesWith(computeOp.getOutputs());
    computeOp.erase();

    LLVM_DEBUG({
      llvm::dbgs() << "Tiled compute op: innermost dim " << innerSize
                   << " -> blocks of " << unrollFactor << "\n";
      llvm::dbgs() << "Outer loop: ";
      outerLoop.dump();
    });

    return success();
  }
};

} // namespace

} // namespace mlir::tt::ttl
