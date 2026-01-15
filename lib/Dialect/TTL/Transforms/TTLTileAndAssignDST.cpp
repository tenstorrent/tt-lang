// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL DST Register Assignment Pass
//===----------------------------------------------------------------------===//
//
// This pass performs DST (destination) register assignment for ttl.compute
// operations by inserting ttl.copy_tile operations that explicitly move tiles
// from circular buffers into DST registers.
//
// Algorithm:
// 1. For each ttl.compute operation:
//    - Compute peak DST register usage using liveness analysis
//    - Verify capacity is not exceeded (default: 8 registers for f16/bf16)
//    - Insert copy_tile operations for each block argument at first use
//    - Assign DST indices with register reuse: freed registers are recycled
//    - Replace block argument uses with copied tile values
//
// 2. Register allocation with reuse (similar to LLVM's RegAllocFast):
//    - Maintain a free pool of DST register indices
//    - On first use of a value: allocate from free pool, or new index if empty
//    - On last use of a value: return its register to the free pool
//    - This minimizes the number of registers needed for any given IR
//
// 3. Liveness analysis:
//    - Block arguments start live at entry
//    - Values become live when added as operands
//    - Values die at last use (within current block)
//    - Peak usage determines if capacity is exceeded
//
// Current limitations/future work:
// - Hardcoded capacity (doesn't account for f32 vs f16 differences)
// - Basic last-use analysis (only checks current block)
// - No spill/reload handling
// - Enable choosing among several register allocation strategies (linear,
//   graph-coloring, etc.)
//
// Pass pipeline position: After convert-ttl-to-compute, before
// ttl-insert-tile-regs-sync.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "llvm/ADT/SmallBitVector.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "ttl-tile-and-assign-dst"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLTILEANDASSIGNDST
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Default DST capacity (16-bit, double-buffered).
constexpr std::uint32_t kDefaultDSTCapacity = 8;

/// TODO(#150): Compute capacity from datatype and device configuration.
/// - f16/bf16: 16 tiles (8 with double-buffering)
/// - f32: 8 tiles (4 with double-buffering)
/// Pull from device/ComputeKernelConfig (fp32_dest_acc_en, fullSyncEn).
static std::uint32_t computeDefaultCapacity() { return kDefaultDSTCapacity; }

static bool isTileValue(Value v) { return isa<ttcore::TileType>(v.getType()); }

// NOTE: isLastUse is block-local only. This is safe because ttl.compute bodies
// are single-block (enforced by SizedRegion<1> in the op definition). If nested
// regions are added to compute bodies, this analysis must be enhanced.
static bool isLastUse(Operation &op, Value v) {
  for (Operation *user : v.getUsers()) {
    if (user != &op && op.isBeforeInBlock(user)) {
      return false;
    }
  }
  return true;
}

static bool isOnlyUsedByUnaryTileOps(Value v) {
  for (Operation *user : v.getUsers()) {
    if (!isTileUnaryOp(user)) {
      return false;
    }
  }
  return true;
}

/// Estimate peak DST usage for a compute body using a simple live-set walk.
static std::uint32_t estimatePeakDSTUsage(Block *body) {
  llvm::SmallPtrSet<Value, 16> live;
  for (BlockArgument arg : body->getArguments()) {
    if (isTileValue(arg) && !arg.use_empty()) {
      live.insert(arg);
    }
  }

  std::uint32_t peakUsage = static_cast<std::uint32_t>(live.size());

  for (Operation &op : *body) {
    if (!tt::ttl::isTileComputeOp(&op)) {
      continue;
    }

    // Add operands to live set
    for (Value operand : op.getOperands()) {
      if (isTileValue(operand)) {
        live.insert(operand);
      }
    }
    peakUsage = std::max<std::uint32_t>(
        peakUsage, static_cast<std::uint32_t>(live.size()));

    // Remove operands at last use
    for (Value operand : op.getOperands()) {
      if (isTileValue(operand) && isLastUse(op, operand)) {
        live.erase(operand);
      }
    }

    // Add results to live set
    for (Value result : op.getResults()) {
      if (isTileValue(result)) {
        live.insert(result);
      }
    }
    peakUsage = std::max<std::uint32_t>(
        peakUsage, static_cast<std::uint32_t>(live.size()));
  }

  return peakUsage;
}

struct TTLTileAndAssignDSTPass
    : public impl::TTLTileAndAssignDSTBase<TTLTileAndAssignDSTPass> {
  using Base = impl::TTLTileAndAssignDSTBase<TTLTileAndAssignDSTPass>;
  using Base::Base;
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    funcOp.walk([&](ComputeOp computeOp) {
      Block *body = &computeOp.getRegion().front();
      if (!body) {
        return;
      }

      std::uint32_t peakUsage = estimatePeakDSTUsage(body);
      std::uint32_t capacity =
          dstCapacity == 0 ? computeDefaultCapacity() : dstCapacity;

      if (peakUsage > capacity) {
        computeOp.emitOpError()
            << "operation chain requires " << peakUsage
            << " DST registers but capacity is only " << capacity
            << "\nnote: consider splitting complex operation chains or "
               "enabling register spilling";
        signalPassFailure();
        return;
      }

      std::uint32_t numTiles = 1;
      if (!computeOp.getOutputs().empty()) {
        auto outputType =
            dyn_cast<RankedTensorType>(computeOp.getOutputs()[0].getType());
        if (outputType) {
          for (int64_t dim : outputType.getShape()) {
            numTiles *= dim;
          }
        }
      }

      // DST footprint for a single tile loop iteration.
      // inputs_footprint excludes unary-only inputs (they reuse output DST).
      std::uint32_t numInputs = 0;
      for (unsigned i = 0; i < computeOp.getInputs().size(); ++i) {
        BlockArgument arg = body->getArgument(i);
        if (isTileValue(arg) && !isOnlyUsedByUnaryTileOps(arg)) {
          numInputs++;
        }
      }

      std::uint32_t numOutputs =
          static_cast<std::uint32_t>(computeOp.getOutputs().size());

      std::uint32_t footprintPerIteration = numInputs + numOutputs;

      std::uint32_t unrollFactor = 1;
      if (numTiles > 1 && footprintPerIteration > 0) {
        unrollFactor = std::min(capacity / footprintPerIteration, numTiles);
      }

      if (unrollFactor > 1) {
        OpBuilder attrBuilder(computeOp.getContext());
        computeOp->setAttr(kUnrollFactorAttrName,
                           attrBuilder.getI32IntegerAttr(unrollFactor));
        // Store numInputs for use by the unroll pass to determine which DST
        // indices should stay fixed (inputs) vs increment (outputs).
        computeOp->setAttr("ttl.num_inputs",
                           attrBuilder.getI32IntegerAttr(numInputs));
      }

      // Insert copy_tile immediately before the first use of each block
      // argument. Track register usage for capacity validation. Bit set =
      // register in use, bit clear = register free.
      llvm::SmallBitVector inUse(capacity);
      DenseMap<Value, std::uint32_t> dstIndexForValue;

      for (Operation &op : *body) {
        // First pass: allocate registers for new block arguments used by this
        // op. We replace all uses to ensure the copy happens only once at first
        // use.
        for (OpOperand &operand : op.getOpOperands()) {
          auto arg = dyn_cast<BlockArgument>(operand.get());
          if (!arg || !isTileValue(arg)) {
            continue;
          }

          // Skip if already copied
          if (dstIndexForValue.count(arg)) {
            continue;
          }

          // Allocate: find first free register
          int freeReg = inUse.find_first_unset();
          assert(freeReg >= 0 && "no free DST register (should have been "
                                 "caught by capacity check)");
          auto assignedDstIndex = static_cast<std::uint32_t>(freeReg);
          inUse.set(assignedDstIndex);

          OpBuilder builder(&op);
          Location loc = op.getLoc();

          // Compute index_map for CB linearization from iteration dimensions.
          // Find which input this block argument corresponds to.
          AffineMapAttr indexMapAttr;
          for (auto [idx, input] : llvm::enumerate(computeOp.getInputs())) {
            if (arg == body->getArgument(idx)) {
              auto tensorType = cast<RankedTensorType>(input.getType());
              int64_t rank = tensorType.getRank();

              // Build row-major linearization affine expression.
              // TODO: Support dynamic shapes using symbols and getMixedSizes().
              SmallVector<int64_t> staticShape(tensorType.getShape().begin(),
                                               tensorType.getShape().end());
              SmallVector<int64_t> strides = mlir::computeStrides(staticShape);

              // Build linearization: sum of (dim_i * stride_i)
              AffineExpr linearExpr = builder.getAffineConstantExpr(0);
              for (int64_t i = 0; i < rank; ++i) {
                linearExpr =
                    linearExpr + builder.getAffineDimExpr(i) *
                                     builder.getAffineConstantExpr(strides[i]);
              }

              AffineMap indexMap =
                  AffineMap::get(rank, /*numSymbols=*/0, linearExpr);
              indexMapAttr = AffineMapAttr::get(indexMap);
              break;
            }
          }

          // Create linearized_index op (will be materialized during loop
          // lowering)
          Value srcIndex = builder.create<LinearizedIndexOp>(loc, indexMapAttr);
          Value dstIndex =
              builder.create<arith::ConstantIndexOp>(loc, assignedDstIndex);
          auto copy = builder.create<CopyTileOp>(
              loc,
              TypeRange{DSTRegisterType::get(arg.getContext()), arg.getType()},
              ValueRange{arg, srcIndex, dstIndex});
          dstIndexForValue[copy.getDstTile()] = assignedDstIndex;

          arg.replaceUsesWithIf(copy.getDstTile(), [&](OpOperand &use) {
            return use.getOwner() != copy;
          });
        }

        // Allocate DST registers for tile compute operations.
        // ALWAYS allocate outputs BEFORE freeing inputs, even when not unrolling.
        // This ensures output indices are >= numInputs, which is required for:
        // 1. Correct unrolling: inputs stay fixed, outputs increment
        // 2. Correct non-unrolled binary ops: hardware seems to require separate output slots
        if (tt::ttl::isTileComputeOp(&op)) {
          // First: allocate registers for results
          for (Value res : op.getResults()) {
            if (!isTileValue(res)) {
              continue;
            }
            int freeReg = inUse.find_first_unset();
            if (freeReg < 0) {
              op.emitOpError("insufficient DST registers for results");
              signalPassFailure();
              return;
            }
            dstIndexForValue[res] = static_cast<std::uint32_t>(freeReg);
            inUse.set(freeReg);
            // NOTE: Sets single dst_idx on the operation. All tile ops
            // currently produce exactly one tile result, so this is safe. If
            // multi-result tile ops are added, this will need per-result
            // attributes.
            OpBuilder attrBuilder(res.getContext());
            op.setAttr(kDstIdxAttrName, attrBuilder.getI32IntegerAttr(
                                            static_cast<int32_t>(freeReg)));
          }

          // Second: free operands at their last use (after allocating outputs)
          for (Value operand : op.getOperands()) {
            if (!isTileValue(operand)) {
              continue;
            }
            if (isLastUse(op, operand)) {
              auto it = dstIndexForValue.find(operand);
              if (it != dstIndexForValue.end()) {
                inUse.reset(it->second);
              }
            }
          }
        } else {
          // For non-compute ops, still free operands at last use
          for (Value operand : op.getOperands()) {
            if (!isTileValue(operand)) {
              continue;
            }
            if (isLastUse(op, operand)) {
              auto it = dstIndexForValue.find(operand);
              if (it != dstIndexForValue.end()) {
                inUse.reset(it->second);
              }
            }
          }
        }
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
