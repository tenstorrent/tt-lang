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
// Current limitations:
// - Hardcoded capacity (doesn't account for f32 vs f16 differences)
// - Basic last-use analysis (only checks current block)
// - No spill/reload handling
//
// Pass pipeline position: After convert-ttl-to-compute, before
// ttl-insert-tile-regs-sync.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallBitVector.h"
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLTILEANDASSIGNDST
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static_assert(std::is_class_v<TTLDialect>);

/// Default DST capacity (16-bit, double-buffered).
constexpr std::uint32_t kDefaultDSTCapacity = 8;

/// TODO(#XXX): Pull from device/ComputeKernelConfig (fp32_dest_acc_en,
/// double-buffer). Mirrors tt-mlir graph-coloring allocator defaults.
static std::uint32_t computeDefaultCapacity() { return kDefaultDSTCapacity; }

static bool isTileOp(Operation *op) {
  return isa<AddTileOp, SubTileOp, MulTileOp, MaxTileOp, ExpTileOp, LogTileOp,
             SqrtTileOp, RsqrtTileOp, TanhTileOp, SigmoidTileOp, AbsTileOp,
             NegTileOp, ReluTileOp, CopyTileOp>(op);
}

static bool isTileValue(Value v) { return isa<ttcore::TileType>(v.getType()); }

static bool isLastUse(Operation &op, Value v) {
  for (Operation *user : v.getUsers()) {
    if (user != &op && op.isBeforeInBlock(user)) {
      return false;
    }
  }
  return true;
}

/// Estimate peak DST usage for a compute body using a simple live-set walk.
static std::uint32_t estimatePeakDSTUsage(Block *body) {
  llvm::SmallPtrSet<Value, 16> live;
  for (BlockArgument arg : body->getArguments()) {
    if (isTileValue(arg)) {
      live.insert(arg);
    }
  }

  std::uint32_t peakUsage = static_cast<std::uint32_t>(live.size());

  for (Operation &op : *body) {
    if (!isTileOp(&op)) {
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

      // Insert copy_tile immediately before the first use of each block argument.
      // Use SmallBitVector for register allocation with reuse.
      // Bit set = register in use, bit clear = register free.
      llvm::SmallBitVector inUse(capacity);
      DenseMap<BlockArgument, std::uint32_t> dstIndexForArg;
      DenseMap<BlockArgument, Value> copiedTileForArg;

      for (Operation &op : *body) {
        // First pass: allocate registers for new block arguments used by this op
        for (OpOperand &operand : op.getOpOperands()) {
          auto arg = dyn_cast<BlockArgument>(operand.get());
          if (!arg || !isTileValue(arg)) {
            continue;
          }
          if (auto it = copiedTileForArg.find(arg); it != copiedTileForArg.end()) {
            operand.set(it->second);
            continue;
          }
          // Allocate: find first free register
          int freeReg = inUse.find_first_unset();
          assert(freeReg >= 0 &&
                 "no free DST register (should have been caught by capacity check)");
          auto assignedDstIndex = static_cast<std::uint32_t>(freeReg);
          inUse.set(assignedDstIndex);

          OpBuilder builder(&op);
          Location loc = op.getLoc();
          // src_index is 0 (the tile index within the circular buffer)
          Value srcIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value dstIndex =
              builder.create<arith::ConstantIndexOp>(loc, assignedDstIndex);
          auto copy = builder.create<CopyTileOp>(
              loc,
              TypeRange{DSTRegisterType::get(arg.getContext()), arg.getType()},
              ValueRange{arg, srcIndex, dstIndex});
          dstIndexForArg[arg] = assignedDstIndex;
          copiedTileForArg[arg] = copy.getDstTile();
          operand.set(copy.getDstTile());
        }

        // Second pass: free registers for block arguments at their last use
        for (OpOperand &operand : op.getOpOperands()) {
          auto arg = dyn_cast<BlockArgument>(operand.get());
          if (!arg || !isTileValue(arg)) {
            continue;
          }
          if (isLastUse(op, arg)) {
            auto it = dstIndexForArg.find(arg);
            if (it != dstIndexForArg.end()) {
              inUse.reset(it->second);
            }
          }
        }
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
