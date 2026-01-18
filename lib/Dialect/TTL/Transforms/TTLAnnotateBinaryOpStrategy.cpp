// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Annotate Binary Operation Strategy Pass
//===----------------------------------------------------------------------===//
//
// This analysis pass annotates binary tile operations (add, sub, mul) with
// their optimal execution strategy based on operand provenance.
//
// Execution target labeling strategies:
//
// | Annotation   | Condition           | TTKernel Operation      | HW   |
// |--------------|---------------------|-------------------------|------|
// | "fpu"        | Both block args     | add_tiles               | FPU  |
// | "dest_reuse" | One block arg       | binary_dest_reuse_tiles | FPU  |
// | "sfpu"       | Both DST (tile ops) | add_binary_tile         | SFPU |
//
// This means FPU optimization applies to most cases:
// - Simple binary ops: a + b -> FPU
// - Chained ops: (a + b) + c -> FPU + dest_reuse (both FPU hardware!)
// - Fused with unary: exp(a) + b -> dest_reuse (FPU)
// - Only DST + DST falls back to SFPU (rare in practice if fusion considers
//   FPU utilization).
// Note that in some cases, such as for a * b + c * d, maximal fusion is not
// the optimal strategy since it will prevent the use of the FPU for the sum
// of a*b and c*d.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#define GEN_PASS_DEF_TTLANNOTATEBINARYOPSTRATEGY
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace mlir::tt::ttl {

namespace {

/// Checks if a value is a block argument (operand from circular buffer)
static bool isBlockArgument(Value v) { return isa<BlockArgument>(v); }

struct TTLAnnotateBinaryOpStrategyPass
    : public ::impl::TTLAnnotateBinaryOpStrategyBase<
          TTLAnnotateBinaryOpStrategyPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    funcOp.walk([&](Operation *op) {
      if (!op->hasTrait<TTLFPUElementwiseOpTrait>()) {
        return;
      }

      // Get operands (FPU ops are all binary elementwise ops)
      Value lhs = op->getOperand(0);
      Value rhs = op->getOperand(1);

      // Determine execution strategy based on operand provenance
      bool lhsFromCB = isBlockArgument(lhs);
      bool rhsFromCB = isBlockArgument(rhs);

      StringRef strategy;
      if (lhsFromCB && rhsFromCB) {
        strategy = kExecutionTargetFPU; // Both from CB -> FPU
      } else if (lhsFromCB || rhsFromCB) {
        strategy = kExecutionTargetDestReuse; // Mixed -> Dest-reuse
      } else {
        strategy = kExecutionTargetSFPU; // Both from DST -> SFPU
      }

      op->setAttr(kExecutionTargetAttrName, StringAttr::get(ctx, strategy));
    });
  }
};

} // namespace

// Forward to the generated ::impl::create function
std::unique_ptr<Pass> createTTLAnnotateBinaryOpStrategy() {
  return ::impl::createTTLAnnotateBinaryOpStrategy();
}

} // namespace mlir::tt::ttl
