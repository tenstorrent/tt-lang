// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Subblock Compute For DST Pass
//===----------------------------------------------------------------------===//
//
// This file partitions ttl.compute into DST-sized subblocks via
// TilingInterface. See the ttl-subblock-compute-for-dst pass description in
// Passes.td.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
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
      auto sizeVal = getConstantIntValue(iterDomain[d].size);
      if (!sizeVal) {
        return computeOp.emitOpError(
            "dynamic dimension not supported for DST tiling");
      }
      dimSizes[d] = *sizeVal;
      totalTiles *= dimSizes[d];
    }

    // Compute full-tensor row-major strides for tile offset computation.
    // Used for loop annotation, full linearization strides attribute, and
    // linearized index offset adjustment.
    SmallVector<int64_t> fullStrides = computeStrides(dimSizes);

    // When unroll_factor >= total tiles, no outer loop is needed -- the compute
    // op already fits in one DST sync region. Set strides so lower-to-loops
    // can annotate tile loops with correct CB linearization strides.
    if (unrollFactor >= totalTiles) {
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
    // TODO: consider supporting peeling/remainder loops for dimensions whose
    // only divisor <= unrollFactor is 1 (e.g. primes larger than unrollFactor).
    // Currently these fall back to processing one tile at a time, wasting DST
    // capacity.
    if (subblockProduct <= 1) {
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

      forOp->setAttr(kSubblockStrideAttrName,
                     b.getIndexAttr(fullStrides[tiledDims[i]]));

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

    // Use TilingInterface to create the subblocked compute op.
    auto tiledResult = computeOp.getTiledImplementation(b, offsets, sizes);
    if (failed(tiledResult)) {
      return failure();
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

          // Compute offset = sum(loopIV[d] * fullStrides[d]) for tiled dims.
          Value offset;
          for (size_t i = 0; i < tiledDims.size(); ++i) {
            int64_t stride = fullStrides[tiledDims[i]];

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

    return success();
  }
};

} // namespace

} // namespace mlir::tt::ttl
