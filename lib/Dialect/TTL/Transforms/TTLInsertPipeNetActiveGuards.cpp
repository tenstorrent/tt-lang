// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert PipeNet Active Guards Pass
//===----------------------------------------------------------------------===//
//
// Wraps each `ttl.kernel_thread` function body in an `scf.if` over
// `ttl.core_x` / `ttl.core_y` so only cores that participate in some pipe
// (as a source or a destination) execute the body. Inactive cores fall
// through directly to the function terminator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-pipenet-active-guards"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTPIPENETACTIVEGUARDS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

// Half-open rectangle [xLo, xHi) x [yLo, yHi).
struct ActiveRect {
  int64_t xLo;
  int64_t xHi;
  int64_t yLo;
  int64_t yHi;
};

constexpr llvm::StringLiteral kActiveGuardAttrName = "ttl.pipenet_active_guard";
constexpr llvm::StringLiteral kKernelThreadAttrName = "ttl.kernel_thread";

// Collect every active rectangle implied by `ttl.create_pipe` ops in the
// module. Each pipe contributes:
//   - A unit rectangle for its source coordinate.
//   - A rectangle covering its destination range (inclusive bounds, expanded
//     to half-open form).
SmallVector<ActiveRect> collectActiveRects(ModuleOp module) {
  SmallVector<ActiveRect> rects;
  module.walk([&](CreatePipeOp pipe) {
    int64_t srcX = pipe.getSrcX();
    int64_t srcY = pipe.getSrcY();
    rects.push_back({srcX, srcX + 1, srcY, srcY + 1});

    int64_t startX = pipe.getDstStartX();
    int64_t endX = pipe.getDstEndX();
    int64_t startY = pipe.getDstStartY();
    int64_t endY = pipe.getDstEndY();
    int64_t xLo = std::min(startX, endX);
    int64_t xHi = std::max(startX, endX) + 1;
    int64_t yLo = std::min(startY, endY);
    int64_t yHi = std::max(startY, endY) + 1;
    rects.push_back({xLo, xHi, yLo, yHi});
  });
  return rects;
}

// Build a single i1 predicate `(x, y) ∈ ⋃ rects` using arith ops.
Value buildActivePredicate(OpBuilder &b, Location loc, Value coreX, Value coreY,
                           ArrayRef<ActiveRect> rects) {
  assert(!rects.empty() && "predicate requires at least one rectangle");

  auto idxConst = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };

  Value any;
  for (const ActiveRect &r : rects) {
    Value xLo = idxConst(r.xLo);
    Value xHi = idxConst(r.xHi);
    Value yLo = idxConst(r.yLo);
    Value yHi = idxConst(r.yHi);
    Value xGe = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, coreX,
                                      xLo);
    Value xLt = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, coreX,
                                      xHi);
    Value yGe = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, coreY,
                                      yLo);
    Value yLt = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, coreY,
                                      yHi);
    Value xIn = arith::AndIOp::create(b, loc, xGe, xLt);
    Value yIn = arith::AndIOp::create(b, loc, yGe, yLt);
    Value inRect = arith::AndIOp::create(b, loc, xIn, yIn);
    any = any ? arith::OrIOp::create(b, loc, any, inRect).getResult() : inRect;
  }
  return any;
}

// Returns true if `func` already contains an scf.if marked with the active
// guard attribute (idempotency check).
bool hasExistingGuard(func::FuncOp func) {
  bool found = false;
  func.walk([&](scf::IfOp ifOp) {
    if (ifOp->hasAttr(kActiveGuardAttrName)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// Wrap the body of a single-block kernel-thread function in an scf.if guard.
LogicalResult wrapFunctionBody(func::FuncOp func,
                               ArrayRef<ActiveRect> rects) {
  if (!func.getBody().hasOneBlock()) {
    return func.emitOpError(
        "ttl-insert-pipenet-active-guards requires single-block functions");
  }

  Block &block = func.getBody().front();
  if (block.empty()) {
    return func.emitOpError("kernel-thread function has no terminator");
  }
  Operation *terminator = block.getTerminator();
  if (!isa<func::ReturnOp>(terminator)) {
    return func.emitOpError(
        "ttl-insert-pipenet-active-guards expects func.return terminator");
  }

  // Empty body (only terminator): nothing to guard.
  if (&block.front() == terminator) {
    return success();
  }

  if (hasExistingGuard(func)) {
    return success();
  }

  OpBuilder builder(terminator);
  Location loc = func.getLoc();

  Value coreX = CoreXOp::create(builder, loc, builder.getIndexType());
  Value coreY = CoreYOp::create(builder, loc, builder.getIndexType());
  Value pred = buildActivePredicate(builder, loc, coreX, coreY, rects);

  auto ifOp = scf::IfOp::create(builder, loc, /*resultTypes=*/TypeRange{},
                                pred, /*withElseRegion=*/false);
  ifOp->setAttr(kActiveGuardAttrName, builder.getUnitAttr());

  // Move every operation that preceded the inserted ops (which now sit just
  // before the terminator) into the then block, except the newly inserted
  // ops themselves and the terminator.
  Block *thenBlock = ifOp.thenBlock();
  // The then block has an scf.yield inserted by the builder; preserve it.
  Operation *thenTerminator = thenBlock->getTerminator();

  // Collect ops to move: everything strictly before coreX in the original
  // block. coreX is the first newly inserted op.
  Operation *coreXOp = coreX.getDefiningOp();
  SmallVector<Operation *> toMove;
  for (Operation &op : block) {
    if (&op == coreXOp) {
      break;
    }
    toMove.push_back(&op);
  }
  for (Operation *op : toMove) {
    op->moveBefore(thenTerminator);
  }

  return success();
}

struct TTLInsertPipeNetActiveGuardsPass
    : impl::TTLInsertPipeNetActiveGuardsBase<TTLInsertPipeNetActiveGuardsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<ActiveRect> rects = collectActiveRects(module);
    if (rects.empty()) {
      // No pipes in the module: nothing to guard.
      return;
    }

    SmallVector<func::FuncOp> threads;
    module.walk([&](func::FuncOp func) {
      if (func->hasAttr(kKernelThreadAttrName)) {
        threads.push_back(func);
      }
    });

    for (func::FuncOp func : threads) {
      if (failed(wrapFunctionBody(func, rects))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
