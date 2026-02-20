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
// Multi-dimensional iteration spaces are partitioned across multiple dimensions
// to fill DST. For each dimension, the subblock size is chosen as a divisor of
// the dimension size, maximizing the total subblock size (product of per-dim
// subblock sizes) while staying within unroll_factor. This approach handles
// non-identity indexing maps (broadcast, reduction) because
// getTiledImplementation maps iteration domain offsets/sizes to per-operand
// slices via the indexing maps.
//
// The outer loop(s) are side-effect-only (no iter_args) because stores are
// explicit side effects (ttl.tile_store) referencing external reserve views.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-subblock-compute-for-dst"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLSUBBLOCKCOMPUTEFORDST
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Find subblock sizes [t0, t1, ...] such that each ti divides dimSizes[i],
/// product(ti) <= unrollFactor, and the product is maximized.
/// Ties are broken by preferring larger inner (higher-index) dimensions.
static SmallVector<int64_t>
computeMultiDimSubblockSizes(ArrayRef<int64_t> dimSizes, int64_t unrollFactor) {
  int64_t rank = dimSizes.size();

  // Collect divisors per dimension (sorted descending for early pruning).
  SmallVector<SmallVector<int64_t>> allDivisors(rank);
  for (int64_t d = 0; d < rank; ++d) {
    for (int64_t i = dimSizes[d]; i >= 1; --i) {
      if (dimSizes[d] % i == 0) {
        allDivisors[d].push_back(i);
      }
    }
  }

  SmallVector<int64_t> bestSizes(rank, 1);
  int64_t bestProduct = 1;
  SmallVector<int64_t> current(rank);

  // Return true if `a` should be preferred over `b` when products are equal.
  // Prefers larger inner (higher-index) dimensions to minimize outer loops.
  auto prefersInner = [&](ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
    for (int64_t d = rank - 1; d >= 0; --d) {
      if (a[d] != b[d]) {
        return a[d] > b[d];
      }
    }
    return false;
  };

  // Recursive brute-force search with pruning.
  std::function<void(int64_t, int64_t)> search;
  search = [&](int64_t dim, int64_t currentProduct) {
    if (dim == rank) {
      if (currentProduct > bestProduct ||
          (currentProduct == bestProduct && prefersInner(current, bestSizes))) {
        bestProduct = currentProduct;
        bestSizes.assign(current.begin(), current.end());
      }
      return;
    }
    for (int64_t divisor : allDivisors[dim]) {
      int64_t newProduct = currentProduct * divisor;
      if (newProduct > unrollFactor) {
        continue;
      }
      current[dim] = divisor;
      search(dim + 1, newProduct);
    }
  };

  search(0, 1);
  return bestSizes;
}

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

    int64_t rank = iterDomain.size();

    // Collect dim sizes and compute total tiles.
    SmallVector<int64_t> dimSizes(rank);
    int64_t totalTiles = 1;
    for (int64_t d = 0; d < rank; ++d) {
      auto sizeAttr =
          dyn_cast<IntegerAttr>(iterDomain[d].size.dyn_cast<Attribute>());
      if (!sizeAttr) {
        return computeOp.emitOpError(
            "dynamic dimension not supported for DST tiling");
      }
      dimSizes[d] = sizeAttr.getInt();
      totalTiles *= dimSizes[d];
    }

    // When unroll_factor >= total tiles, no outer loop is needed -- the compute
    // op already fits in one DST sync cycle. Set strides so lower-to-loops
    // can annotate tile loops with correct CB linearization strides.
    if (unrollFactor >= totalTiles) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping tiling: unroll_factor (" << unrollFactor
                 << ") >= total tiles (" << totalTiles << ")\n");

      // Compute full-tensor row-major strides for tile offset computation.
      SmallVector<int64_t> fullStrides = computeStrides(dimSizes);
      computeOp->setAttr(kFullLinStridesAttrName,
                         b.getDenseI64ArrayAttr(fullStrides));
      return success();
    }

    // Compute multi-dimensional subblock sizes that maximize DST utilization.
    SmallVector<int64_t> subblockSizes =
        computeMultiDimSubblockSizes(dimSizes, unrollFactor);

    int64_t subblockProduct = 1;
    for (int64_t ss : subblockSizes) {
      subblockProduct *= ss;
    }

    // If subblock product is 1, no subblocking benefit -- skip.
    if (subblockProduct <= 1) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping tiling: subblock product is "
                              << subblockProduct << "\n");
      return success();
    }

    // Create loop bounds before entering loop nesting.
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps;
    SmallVector<int64_t> tiledDims;
    for (int64_t d = 0; d < rank; ++d) {
      if (subblockSizes[d] < dimSizes[d]) {
        upperBounds.push_back(
            b.create<arith::ConstantIndexOp>(loc, dimSizes[d]));
        steps.push_back(
            b.create<arith::ConstantIndexOp>(loc, subblockSizes[d]));
        tiledDims.push_back(d);
      }
    }

    // Create nested scf.for loops for tiled dimensions.
    // Annotate each with ttl.subblock_stride so downstream passes
    // (computeCBTileIndexFromLoops) can distinguish subblock loops from tile
    // iteration loops and compute correct linearized CB offsets.
    SmallVector<Value> loopIVs;
    for (size_t i = 0; i < tiledDims.size(); ++i) {
      auto forOp = b.create<scf::ForOp>(loc, c0, upperBounds[i], steps[i]);

      int64_t d = tiledDims[i];
      int64_t stride = 1;
      for (int64_t j = d + 1; j < rank; ++j) {
        stride *= dimSizes[j];
      }
      forOp->setAttr(kSubblockStrideAttrName, b.getIndexAttr(stride));

      loopIVs.push_back(forOp.getInductionVar());
      b.setInsertionPointToStart(forOp.getBody());
    }

    // Build offsets and sizes for getTiledImplementation.
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (int64_t d = 0; d < rank; ++d) {
      sizes.push_back(b.getIndexAttr(subblockSizes[d]));
    }

    // Set offsets for tiled dimensions to their loop IVs.
    for (size_t i = 0; i < tiledDims.size(); ++i) {
      offsets[tiledDims[i]] = loopIVs[i];
    }

    // Use TilingInterface to create the tiled compute op.
    auto tiledResult = computeOp.getTiledImplementation(b, offsets, sizes);
    if (failed(tiledResult)) {
      return failure();
    }

    // Compute full-tensor row-major strides for tile loop annotation.
    // These are needed by lower-to-loops to annotate tile loops with correct
    // CB linearization strides (which differ from the subblock shape bounds).
    SmallVector<int64_t> fullStrides(rank);
    for (int64_t d = 0; d < rank; ++d) {
      int64_t stride = 1;
      for (int64_t j = d + 1; j < rank; ++j) {
        stride *= dimSizes[j];
      }
      fullStrides[d] = stride;
    }

    // Remove the unroll_factor attribute from the tiled inner compute,
    // set full linearization strides, and offset linearized indices.
    for (Operation *tiledOp : tiledResult->tiledOps) {
      tiledOp->removeAttr("ttl.unroll_factor");
      tiledOp->setAttr(kFullLinStridesAttrName,
                       b.getDenseI64ArrayAttr(fullStrides));

      if (auto innerCompute = dyn_cast<ComputeOp>(tiledOp)) {
        SmallVector<LinearizedIndexOp> linIdxOps;
        innerCompute.getBody().front().walk(
            [&](LinearizedIndexOp op) { linIdxOps.push_back(op); });
        for (LinearizedIndexOp linIdx : linIdxOps) {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointAfter(linIdx);

          // Compute offset = sum(loopIV[d] * stride[d]) where
          // stride[d] = product of dimSizes[j] for j > d.
          Value offset;
          for (size_t i = 0; i < tiledDims.size(); ++i) {
            int64_t d = tiledDims[i];
            int64_t stride = 1;
            for (int64_t j = d + 1; j < rank; ++j) {
              stride *= dimSizes[j];
            }

            Value contribution;
            if (stride == 1) {
              contribution = loopIVs[i];
            } else {
              Value strideVal = b.create<arith::ConstantIndexOp>(loc, stride);
              contribution =
                  b.create<arith::MulIOp>(loc, loopIVs[i], strideVal);
            }

            if (!offset) {
              offset = contribution;
            } else {
              offset = b.create<arith::AddIOp>(loc, offset, contribution);
            }
          }

          if (offset) {
            Value adjusted =
                b.create<arith::AddIOp>(loc, linIdx.getResult(), offset);
            linIdx.getResult().replaceAllUsesExcept(adjusted,
                                                    adjusted.getDefiningOp());
          }
        }
      }
    }

    // Replace the original compute op with its output operands.
    // The outer loop(s) are side-effect-only; results flow through tile_store.
    computeOp.replaceAllUsesWith(computeOp.getOutputs());
    computeOp.erase();

    LLVM_DEBUG({
      llvm::dbgs() << "Tiled compute op: " << rank << "D [";
      for (int64_t d = 0; d < rank; ++d) {
        llvm::dbgs() << dimSizes[d];
        if (d < rank - 1) {
          llvm::dbgs() << "x";
        }
      }
      llvm::dbgs() << "] -> subblocks of [";
      for (int64_t d = 0; d < rank; ++d) {
        llvm::dbgs() << subblockSizes[d];
        if (d < rank - 1) {
          llvm::dbgs() << "x";
        }
      }
      llvm::dbgs() << "] (" << subblockProduct << " tiles)\n";
    });

    return success();
  }
};

} // namespace

} // namespace mlir::tt::ttl
