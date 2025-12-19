// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL DST Register Assignment and Tiling Pass
//===----------------------------------------------------------------------===//
//
// This pass analyzes DST register requirements for ttl.compute operations,
// performs DST assignment, and emits diagnostics when capacity is exceeded.
// Batching and spilling are TODOs; current implementation assigns dst_idx
// attributes and inserts ttl.copy_tile for block arguments.
//
// Pass pipeline position: Stage 2 (between convert-ttl-to-compute and
// ttl-lower-to-loops).
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLTILEANDASSIGNDST
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Default DST capacity (16-bit, double-buffered).
constexpr unsigned kDefaultDSTCapacity = 8;

static bool isTileOp(Operation *op) {
  return isa<AddTileOp, SubTileOp, MulTileOp, MaxTileOp, ExpTileOp, LogTileOp,
             SqrtTileOp, RsqrtTileOp, TanhTileOp, SigmoidTileOp, AbsTileOp,
             NegTileOp, ReluTileOp, CopyTileOp>(op);
}

static bool isUnaryTileOp(Operation *op) {
  return isa<ExpTileOp, LogTileOp, SqrtTileOp, RsqrtTileOp, TanhTileOp,
             SigmoidTileOp, AbsTileOp, NegTileOp, ReluTileOp>(op);
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
static unsigned estimatePeakDSTUsage(Block *body) {
  llvm::SmallPtrSet<Value, 16> live;
  for (BlockArgument arg : body->getArguments()) {
    if (isTileValue(arg)) {
      live.insert(arg);
    }
  }

  unsigned peakUsage = live.size();

  for (Operation &op : *body) {
    if (!isTileOp(&op)) {
      continue;
    }

    unsigned currentUsage = live.size() + op.getNumOperands();
    if (op.getNumResults() > 0) {
      currentUsage += op.getNumResults();
    }
    peakUsage = std::max<unsigned>(peakUsage, currentUsage);

    for (Value operand : op.getOperands()) {
      if (isTileValue(operand)) {
        live.insert(operand);
      }
    }
    peakUsage = std::max<unsigned>(peakUsage, live.size());

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
    peakUsage = std::max<unsigned>(peakUsage, live.size());
  }

  return peakUsage;
}

/// Assign DST indices to tile operations using linear scan allocation.
static DenseMap<Value, unsigned> assignDSTIndices(Block *body,
                                                  unsigned capacity) {
  DenseMap<Value, unsigned> allocations;
  SmallVector<unsigned, 8> freeRegs;
  for (unsigned i = capacity; i-- > 0;) {
    freeRegs.push_back(i);
  }

  auto allocateReg = [&]() -> unsigned {
    assert(!freeRegs.empty() && "No free DST registers.");
    return freeRegs.pop_back_val();
  };
  auto freeReg = [&](unsigned reg) { freeRegs.push_back(reg); };

  unsigned nextBlockArgDst = 0;
  for (BlockArgument arg : body->getArguments()) {
    if (isa<ttcore::TileType>(arg.getType())) {
      allocations[arg] = nextBlockArgDst++;
    }
  }

  for (Operation &op : *body) {
    if (!isTileOp(&op)) {
      continue;
    }

    if (isUnaryTileOp(&op) && op.getNumOperands() == 1) {
      Value input = op.getOperand(0);
      if (allocations.count(input)) {
        allocations[op.getResult(0)] = allocations[input];
        continue;
      }
    }

    if (op.getNumOperands() == 2 && op.getNumResults() == 1) {
      Value lhs = op.getOperand(0);
      Value rhs = op.getOperand(1);
      if (allocations.count(lhs) && isLastUse(op, lhs)) {
        allocations[op.getResult(0)] = allocations[lhs];
        if (allocations.count(rhs) && isLastUse(op, rhs) &&
            !isa<BlockArgument>(rhs)) {
          freeReg(allocations[rhs]);
        }
        continue;
      }
    }

    unsigned dstIdx = allocateReg();
    allocations[op.getResult(0)] = dstIdx;

    for (Value operand : op.getOperands()) {
      if (!isa<BlockArgument>(operand) && allocations.count(operand) &&
          isLastUse(op, operand)) {
        freeReg(allocations[operand]);
      }
    }
  }

  return allocations;
}

struct TTLTileAndAssignDSTPass
    : public impl::TTLTileAndAssignDSTBase<TTLTileAndAssignDSTPass> {
  using Base = impl::TTLTileAndAssignDSTBase<TTLTileAndAssignDSTPass>;
  using Base::Base;
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    funcOp.walk([&](ComputeOp computeOp) {
      Block *body = &computeOp.getRegion().front();
      if (!body) {
        return;
      }

      unsigned peakUsage = estimatePeakDSTUsage(body);
      unsigned capacity = dstCapacity ? dstCapacity : kDefaultDSTCapacity;

      if (peakUsage > capacity) {
        computeOp.emitOpError()
            << "operation chain requires " << peakUsage
            << " DST registers but capacity is only " << capacity
            << "\nnote: consider splitting complex operation chains or "
               "enabling register spilling";
        signalPassFailure();
        return;
      }

      DenseMap<Value, unsigned> allocations = assignDSTIndices(body, capacity);

      OpBuilder builder(ctx);
      for (Operation &op : *body) {
        if (!isTileOp(&op) || isa<CopyTileOp>(&op)) {
          continue;
        }
        Value result = op.getResult(0);
        if (!allocations.count(result)) {
          continue;
        }
        unsigned dstIdx = allocations[result];
        if (dstIdx >= capacity) {
          op.emitError() << "dst_idx " << dstIdx << " exceeds capacity of "
                         << capacity << " registers";
          signalPassFailure();
          return;
        }
        op.setAttr("dst_idx", builder.getI32IntegerAttr(dstIdx));
      }

      // TODO(#XXX): Add dst_idx_map attribute to ttl.compute.
      // TODO(#XXX): Implement batching and spilling when peak usage exceeds
      // capacity.
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
