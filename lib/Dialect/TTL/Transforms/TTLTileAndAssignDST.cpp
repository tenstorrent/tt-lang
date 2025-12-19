// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL DST Register Assignment and Tiling Pass
//===----------------------------------------------------------------------===//
//
// This pass analyzes DST register requirements for ttl.compute operations,
// emits diagnostics when capacity is exceeded, and uses ttl.copy_tile tokens
// to order tile execution without attaching dst_idx attributes.
//
// Pass pipeline position: Stage 2 (between convert-ttl-to-compute and
// ttl-lower-to-loops).
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

    std::uint32_t currentUsage =
        static_cast<std::uint32_t>(live.size() + op.getNumOperands());
    if (op.getNumResults() > 0) {
      currentUsage += static_cast<std::uint32_t>(op.getNumResults());
    }
    peakUsage = std::max<std::uint32_t>(peakUsage, currentUsage);

    for (Value operand : op.getOperands()) {
      if (isTileValue(operand)) {
        live.insert(operand);
      }
    }
    peakUsage = std::max<std::uint32_t>(
        peakUsage, static_cast<std::uint32_t>(live.size()));

    for (Value operand : op.getOperands()) {
      if (isTileValue(operand) && isLastUse(op, operand)) {
        live.erase(operand);
      }
    }

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
      std::uint32_t capacity = computeDefaultCapacity();

      if (peakUsage > capacity) {
        computeOp.emitOpError()
            << "operation chain requires " << peakUsage
            << " DST registers but capacity is only " << capacity
            << "\nnote: consider splitting complex operation chains or "
               "enabling register spilling";
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
