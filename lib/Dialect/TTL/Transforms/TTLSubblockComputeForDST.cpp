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
// Multi-dimensional iteration spaces are flattened to 1D before partitioning.
// This ensures subblock sizes are correct regardless of tensor shape.
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

/// Return true if all indexing maps on the ComputeOp are identity maps
/// of the given rank.
static bool allMapsAreIdentity(ComputeOp computeOp, int64_t rank) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(rank, computeOp.getContext());
  for (Attribute attr : computeOp.getIndexingMaps()) {
    if (cast<AffineMapAttr>(attr).getValue() != identityMap) {
      return false;
    }
  }
  return true;
}

/// Flatten a multi-dimensional ComputeOp to 1D by inserting
/// tensor.collapse_shape on all operands and creating a new ComputeOp with
/// 1D identity indexing maps. Returns the new 1D ComputeOp, or the original
/// if no flattening is needed. Returns failure for unsupported cases.
///
/// Flattening is needed when the iteration space has multiple dimensions
/// with outer dims > 1 (i.e., totalTiles > innerDimSize). Without flattening,
/// innermost-dim-only tiling would produce subblocks containing all outer
/// tiles, exceeding unroll_factor.
static FailureOr<ComputeOp> flattenComputeOp(ComputeOp computeOp) {
  OpBuilder b(computeOp);
  SmallVector<Range> iterDomain = computeOp.getIterationDomain(b);
  int64_t rank = iterDomain.size();

  // Already 1D or scalar: no flattening needed.
  if (rank <= 1) {
    return computeOp;
  }

  // Compute total tiles and innermost dim size.
  int64_t totalTiles = 1;
  for (auto &range : iterDomain) {
    auto sizeAttr = dyn_cast<IntegerAttr>(range.size.dyn_cast<Attribute>());
    if (!sizeAttr) {
      return computeOp.emitOpError(
          "dynamic dimensions not supported for flattening");
    }
    totalTiles *= sizeAttr.getInt();
  }

  auto innerSizeAttr =
      dyn_cast<IntegerAttr>(iterDomain.back().size.dyn_cast<Attribute>());
  int64_t innerDimSize = innerSizeAttr.getInt();

  // If all outer dims are 1, the iteration space is effectively 1D and the
  // existing innermost-dim tiling produces correct subblock sizes.
  if (totalTiles == innerDimSize) {
    return computeOp;
  }

  // If all tiles fit in one subblock, no tiling will happen, so flattening
  // is unnecessary.
  auto unrollAttr = computeOp->getAttrOfType<IntegerAttr>("ttl.unroll_factor");
  if (unrollAttr && unrollAttr.getInt() >= totalTiles) {
    return computeOp;
  }

  // Require all identity maps. Broadcast maps need per-operand reassociation
  // (not yet implemented).
  if (!allMapsAreIdentity(computeOp, rank)) {
    return computeOp.emitOpError(
        "non-identity indexing maps not supported for flattening");
  }

  Location loc = computeOp.getLoc();

  // Reassociation: merge all dims into one.
  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices allDims;
  for (int64_t i = 0; i < rank; ++i) {
    allDims.push_back(i);
  }
  reassociation.push_back(allDims);

  // Flatten each input operand.
  SmallVector<Value> flatInputs;
  for (Value input : computeOp.getInputs()) {
    flatInputs.push_back(
        b.create<tensor::CollapseShapeOp>(loc, input, reassociation));
  }

  // Flatten each output operand.
  SmallVector<Value> flatOutputs;
  for (Value output : computeOp.getOutputs()) {
    flatOutputs.push_back(
        b.create<tensor::CollapseShapeOp>(loc, output, reassociation));
  }

  // 1D identity maps and parallel iterator type.
  AffineMap id1D = AffineMap::getMultiDimIdentityMap(1, b.getContext());
  size_t numOperands = flatInputs.size() + flatOutputs.size();
  SmallVector<Attribute> flatMaps(numOperands, AffineMapAttr::get(id1D));
  SmallVector<Attribute> flatIterTypes = {
      StringAttr::get(b.getContext(), "parallel")};

  SmallVector<Type> resultTypes;
  for (Value out : flatOutputs) {
    resultTypes.push_back(out.getType());
  }

  auto newOp = b.create<ComputeOp>(loc, resultTypes, flatInputs, flatOutputs,
                                   b.getArrayAttr(flatMaps),
                                   b.getArrayAttr(flatIterTypes));

  // Clone the body region. Block arguments are tile types (unchanged by
  // flattening). External references (e.g., reserve views from cb_reserve)
  // are preserved.
  IRMapping mapping;
  computeOp.getBody().cloneInto(&newOp.getBody(), mapping);

  // After flattening to 1D, update LinearizedIndexOp maps to 1D identity.
  // The original multi-dim map (e.g., (d0,d1) -> d0*4+d1) no longer matches
  // the 1D iteration domain. The 1D identity (d0) -> (d0) is correct because
  // the flattened iteration space IS the linearized index.
  AffineMapAttr id1DAttr = AffineMapAttr::get(id1D);
  newOp.getBody().front().walk(
      [&](LinearizedIndexOp linIdx) { linIdx.setIndexMapAttr(id1DAttr); });

  // Copy custom attributes (ttl.unroll_factor, ttl.dst_allocation, etc.).
  // Skip builder-managed attributes.
  for (NamedAttribute attr : computeOp->getAttrs()) {
    StringRef name = attr.getName();
    if (name == "indexing_maps" || name == "iterator_types" ||
        name == "operandSegmentSizes") {
      continue;
    }
    newOp->setAttr(name, attr.getValue());
  }

  // Replace original compute (side-effect-only: results → output operands).
  computeOp.replaceAllUsesWith(computeOp.getOutputs());
  computeOp.erase();

  LLVM_DEBUG(llvm::dbgs() << "Flattened " << rank << "D compute to 1D ("
                          << totalTiles << " tiles)\n");

  return newOp;
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
      // Flatten multi-dimensional computes to 1D if needed.
      auto flatResult = flattenComputeOp(computeOp);
      if (failed(flatResult)) {
        signalPassFailure();
        return;
      }

      if (failed(tileComputeOp(*flatResult))) {
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

    // After flattening, the iteration domain is 1D. Tile the single
    // (innermost) dimension by unroll_factor.
    int64_t tilingDim = iterDomain.size() - 1;
    auto innerSizeAttr =
        dyn_cast<IntegerAttr>(iterDomain[tilingDim].size.dyn_cast<Attribute>());
    if (!innerSizeAttr) {
      return computeOp.emitOpError(
          "dynamic innermost dimension not supported for DST tiling");
    }
    int64_t innerSize = innerSizeAttr.getInt();

    // When unroll_factor >= total tiles, no outer loop is needed -- the compute
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

    // Adjust unroll_factor to evenly divide innerSize. The downstream pipeline
    // requires constant loop bounds (no dynamic remainder blocks). Find the
    // largest divisor of innerSize that is <= unrollFactor.
    if (innerSize % unrollFactor != 0) {
      int64_t adjusted = unrollFactor;
      while (adjusted > 1 && innerSize % adjusted != 0) {
        --adjusted;
      }
      LLVM_DEBUG(llvm::dbgs() << "Adjusted unroll_factor from " << unrollFactor
                              << " to " << adjusted << " (innerSize="
                              << innerSize << " must be evenly divisible)\n");
      unrollFactor = adjusted;

      // If adjusted to 1, no subblocking benefit -- skip.
      if (unrollFactor <= 1) {
        return success();
      }
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
          sizes[tilingDim] = loopBuilder.getIndexAttr(unrollFactor);

          // Use TilingInterface to create the tiled compute op.
          auto tiledResult =
              computeOp.getTiledImplementation(loopBuilder, offsets, sizes);
          if (failed(tiledResult)) {
            return;
          }

          // Remove the unroll_factor attribute from the tiled inner compute
          // and offset linearized indices by the outer loop IV (follows LLVM
          // offsetIndices pattern). Inner IVs are local to the subblock
          // (0..unroll_factor-1); copy_tile needs absolute CB position.
          for (Operation *tiledOp : tiledResult->tiledOps) {
            tiledOp->removeAttr("ttl.unroll_factor");

            if (auto innerCompute = dyn_cast<ComputeOp>(tiledOp)) {
              SmallVector<LinearizedIndexOp> linIdxOps;
              innerCompute.getBody().front().walk(
                  [&](LinearizedIndexOp op) { linIdxOps.push_back(op); });
              for (LinearizedIndexOp linIdx : linIdxOps) {
                OpBuilder::InsertionGuard guard(loopBuilder);
                loopBuilder.setInsertionPointAfter(linIdx);
                Value adjusted = loopBuilder.create<arith::AddIOp>(
                    loopLoc, linIdx.getResult(), iv);
                linIdx.getResult().replaceAllUsesExcept(
                    adjusted, adjusted.getDefiningOp());
              }
            }
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
