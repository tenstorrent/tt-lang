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
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

#include <numeric>

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
  SmallVector<int64_t> current(rank, 1);

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
      // All dimensions have been assigned. Update best if this candidate
      // has a larger product, or the same product but larger inner dimensions.
      if (currentProduct > bestProduct ||
          (currentProduct == bestProduct && prefersInner(current, bestSizes))) {
        bestProduct = currentProduct;
        bestSizes = current;
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

    // Collect compute ops to subblock (avoid modifying while walking).
    SmallVector<ComputeOp> opsToSubblock;
    funcOp.walk([&](ComputeOp computeOp) {
      auto unrollAttr =
          computeOp->getAttrOfType<IntegerAttr>(kUnrollFactorAttrName);
      if (unrollAttr && unrollAttr.getInt() > 1 &&
          !computeOp.containsOp<TileMatmulBlockOp>()) {
        opsToSubblock.push_back(computeOp);
      }
    });

    for (ComputeOp computeOp : opsToSubblock) {
      if (failed(subblockComputeOp(computeOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult subblockComputeOp(ComputeOp computeOp) {
    auto unrollAttr =
        computeOp->getAttrOfType<IntegerAttr>(kUnrollFactorAttrName);
    int64_t unrollFactor = unrollAttr.getInt();
    Location loc = computeOp.getLoc();
    OpBuilder b(computeOp);

    // Collect dim sizes and compute total tile count.
    SmallVector<int64_t> dimSizes = computeOp.getStaticIterationDomainSizes();
    int64_t rank = dimSizes.size();
    int64_t totalTiles = computeOp.getTotalIterationTiles();
    SmallVector<utils::IteratorType> iterTypes =
        computeOp.getIteratorTypesArray();

    // Compute row-major strides over the CB block iteration domain for tile
    // offset computation. Used for loop annotation, CB linearization strides
    // attribute, and linearized index offset adjustment.
    SmallVector<int64_t> blockStrides = computeStrides(dimSizes);

    // When unroll_factor >= total tiles, no outer loop is needed -- the compute
    // op already fits in one DST sync region. Set strides so lower-to-loops
    // can annotate tile loops with correct CB linearization strides.
    if (unrollFactor >= totalTiles) {
      computeOp->setAttr(kFullLinStridesAttrName,
                         b.getDenseI64ArrayAttr(blockStrides));
      return success();
    }

    // Only parallel dimensions are candidates for subblocking; reduction
    // dimensions must be fully included in each subblock in this
    // implementation.
    SmallVector<int64_t> parallelDimSizes;
    int64_t reductionProduct = 1;
    for (int64_t d = 0; d < rank; ++d) {
      if (iterTypes[d] == utils::IteratorType::parallel) {
        parallelDimSizes.push_back(dimSizes[d]);
      } else {
        reductionProduct *= dimSizes[d];
      }
    }

    // If reduction dims alone exceed the DST capacity, no subblocking is
    // possible with this pass.
    if (reductionProduct > unrollFactor) {
      return success();
    }

    // Budget remaining for parallel dimensions after accounting for reductions.
    int64_t parallelBudget = unrollFactor / reductionProduct;

    // Compute subblock sizes for parallel dimensions only.
    SmallVector<int64_t> parallelSubblockSizes =
        computeMultiDimSubblockSizes(parallelDimSizes, parallelBudget);

    // Expand back to full-rank subblock sizes: reduction dims get their full
    // size, parallel dims get the computed subblock size.
    SmallVector<int64_t> subblockSizes(rank);
    int64_t parallelIdx = 0;
    for (int64_t d = 0; d < rank; ++d) {
      if (iterTypes[d] == utils::IteratorType::parallel) {
        subblockSizes[d] = parallelSubblockSizes[parallelIdx++];
      } else {
        subblockSizes[d] = dimSizes[d];
      }
    }

    int64_t subblockProduct =
        std::accumulate(subblockSizes.begin(), subblockSizes.end(), int64_t{1},
                        std::multiplies<>());

    // If subblock product is 1, no subblocking benefit -- skip.
    // TODO: consider supporting peeling/remainder loops for dimensions whose
    // only divisor <= unrollFactor is 1 (e.g. primes larger than unrollFactor).
    // Currently these fall back to processing one tile at a time, wasting DST
    // capacity. Examples: a 7x1 block with unrollFactor=4 could process 4
    // tiles then 3 via a remainder loop, but currently processes 1 at a time;
    // a 5x3 block with unrollFactor=8 has no exact 2D subblock and also
    // falls back to single-tile. In practice, users can avoid this by choosing
    // block sizes with non-prime dimensions (e.g. 8x1 instead of 7x1).
    if (subblockProduct <= 1) {
      return success();
    }

    // Collect loop bounds for subblocked dimensions.
    SmallVector<Value> lowerBounds, upperBounds, steps;
    SmallVector<int64_t> subblockedDims;
    for (int64_t d = 0; d < rank; ++d) {
      if (subblockSizes[d] < dimSizes[d]) {
        lowerBounds.push_back(arith::ConstantIndexOp::create(b, loc, 0));
        upperBounds.push_back(
            arith::ConstantIndexOp::create(b, loc, dimSizes[d]));
        steps.push_back(
            arith::ConstantIndexOp::create(b, loc, subblockSizes[d]));
        subblockedDims.push_back(d);
      }
    }

    // Build nested scf.for loops via buildLoopNest and subblock the compute
    // inside the innermost loop body. The loops have no iter_args; results
    // flow through tile_store side effects.
    bool subblockingFailed = false;
    scf::LoopNest loopNest = scf::buildLoopNest(
        b, loc, lowerBounds, upperBounds, steps, ValueRange{},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs,
            ValueRange /*iterArgs*/) -> scf::ValueVector {
          // Build offsets and sizes for getTiledImplementation.
          SmallVector<OpFoldResult> offsets(rank,
                                            nestedBuilder.getIndexAttr(0));
          SmallVector<OpFoldResult> sizes;
          for (int64_t d = 0; d < rank; ++d) {
            sizes.push_back(nestedBuilder.getIndexAttr(subblockSizes[d]));
          }
          for (size_t i = 0; i < subblockedDims.size(); ++i) {
            offsets[subblockedDims[i]] = ivs[i];
          }

          // Use TilingInterface to create the subblocked compute op.
          auto tiledResult =
              computeOp.getTiledImplementation(nestedBuilder, offsets, sizes);
          if (failed(tiledResult)) {
            subblockingFailed = true;
            return {};
          }

          // Remove the unroll_factor attribute from the subblocked inner
          // compute, set full linearization strides, and offset linearized
          // indices.
          for (Operation *tiledOp : tiledResult->tiledOps) {
            tiledOp->removeAttr(kUnrollFactorAttrName);
            tiledOp->setAttr(kFullLinStridesAttrName,
                             nestedBuilder.getDenseI64ArrayAttr(blockStrides));

            if (auto innerCompute = dyn_cast<ComputeOp>(tiledOp)) {
              // Add subblock IV offset to iter_index results for subblocked
              // dimensions (local -> global coordinates for CB indexing).
              // Same pattern as linalg.index adjustment during tiling.
              SmallVector<IterIndexOp> iterIdxOps;
              innerCompute.getBody().front().walk(
                  [&](IterIndexOp op) { iterIdxOps.push_back(op); });
              for (IterIndexOp iterIdx : iterIdxOps) {
                int64_t dim = iterIdx.getDim();
                // Find if this dimension is being subblocked.
                for (size_t i = 0; i < subblockedDims.size(); ++i) {
                  if (static_cast<int64_t>(subblockedDims[i]) == dim) {
                    OpBuilder::InsertionGuard guard(nestedBuilder);
                    nestedBuilder.setInsertionPointAfter(iterIdx);
                    Value adjusted = arith::AddIOp::create(
                        nestedBuilder, nestedLoc, iterIdx.getResult(), ivs[i]);
                    iterIdx.getResult().replaceAllUsesExcept(
                        adjusted, adjusted.getDefiningOp());
                    break;
                  }
                }
              }
            }
          }
          // The loops have no iter_args (results use tile_store)
          return {};
        });

    if (subblockingFailed) {
      return failure();
    }

    // Annotate each loop with stride and dimension index so downstream passes
    // can distinguish subblock loops from tile loops and compute correct
    // CB offsets (both linearized and per-dimension).
    for (size_t i = 0; i < subblockedDims.size(); ++i) {
      loopNest.loops[i]->setAttr(
          kSubblockLoopStrideAttrName,
          b.getIndexAttr(blockStrides[subblockedDims[i]]));
      loopNest.loops[i]->setAttr(kSubblockDimAttrName,
                                 b.getIndexAttr(subblockedDims[i]));
    }

    // Replace the original compute op with its output operands.
    // The outer loop(s) are side-effect-only; results flow through tile_store.
    assert(computeOp.getResults().size() == computeOp.getOutputs().size() &&
           "result count must match output count for RAUW");
    computeOp.replaceAllUsesWith(computeOp.getOutputs());
    computeOp.erase();

    return success();
  }
};

} // namespace

} // namespace mlir::tt::ttl
