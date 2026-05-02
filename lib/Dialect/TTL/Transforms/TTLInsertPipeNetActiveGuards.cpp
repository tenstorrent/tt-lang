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

// Rank-agnostic axis-aligned half-open rectangle. `lo[d]` and `hi[d]` give
// the inclusive lower bound and exclusive upper bound along dimension d.
// `lo.size() == hi.size()` defines the rectangle's rank.
//
// Rank is set by the dialect's coordinate accessors (currently 2D — see
// readPipeSourceBounds / readPipeDstBounds below). Predicate construction
// (buildActivePredicate) and rectangle collection iterate over dimensions
// generically, so n-D support reduces to extending those accessors when
// the dialect grows beyond `core_x` / `core_y`.
struct ActiveRect {
  SmallVector<int64_t> lo;
  SmallVector<int64_t> hi;
};

constexpr llvm::StringLiteral kActiveGuardAttrName = "ttl.pipenet_active_guard";
constexpr llvm::StringLiteral kKernelThreadAttrName = "ttl.kernel_thread";

// 2D-specific accessor for a pipe's source unit rectangle. The only place
// the active-set pass reaches into named-X/Y attributes; generalize here
// when CreatePipeOp moves to an n-D coordinate attribute.
ActiveRect readPipeSourceRect(CreatePipeOp pipe) {
  // CreatePipeOp accessors return uint64_t; match TTLOps.cpp's verifier
  // pattern of explicit casts to the int64_t storage type.
  int64_t srcX = static_cast<int64_t>(pipe.getSrcX());
  int64_t srcY = static_cast<int64_t>(pipe.getSrcY());
  return {{srcX, srcY}, {srcX + 1, srcY + 1}};
}

// 2D-specific accessor for a pipe's destination rectangle, normalized to
// half-open form. Tolerates inverted ranges via min/max because
// CreatePipeOp's verifier does not currently enforce dstStart <= dstEnd
// ordering.
ActiveRect readPipeDstRect(CreatePipeOp pipe) {
  int64_t startX = static_cast<int64_t>(pipe.getDstStartX());
  int64_t endX = static_cast<int64_t>(pipe.getDstEndX());
  int64_t startY = static_cast<int64_t>(pipe.getDstStartY());
  int64_t endY = static_cast<int64_t>(pipe.getDstEndY());
  return {
      {std::min(startX, endX), std::min(startY, endY)},
      {std::max(startX, endX) + 1, std::max(startY, endY) + 1},
  };
}

// Collect every active rectangle implied by `ttl.create_pipe` ops in the
// module. Each pipe contributes a unit rectangle for its source coordinate
// and a rectangle covering its destination range.
SmallVector<ActiveRect> collectActiveRects(ModuleOp module) {
  SmallVector<ActiveRect> rects;
  module.walk([&](CreatePipeOp pipe) {
    rects.push_back(readPipeSourceRect(pipe));
    rects.push_back(readPipeDstRect(pipe));
  });
  return rects;
}

// True if every coordinate of `inner` lies within `outer` along every
// dimension. Both rectangles must have the same rank.
bool rectContains(const ActiveRect &outer, const ActiveRect &inner) {
  if (outer.lo.size() != inner.lo.size()) {
    return false;
  }
  for (size_t d = 0; d < outer.lo.size(); ++d) {
    if (outer.lo[d] > inner.lo[d] || outer.hi[d] < inner.hi[d]) {
      return false;
    }
  }
  return true;
}

// Drop rectangles fully contained in another rectangle. The most common
// source of redundancy is loopback multicast pipes, where the source
// unit cell sits inside the destination range; the pass would otherwise
// emit two predicates whose union is just the destination rectangle.
//
// Equal rectangles each contain each other; tie-break by keeping the
// earliest occurrence so deduplication is deterministic. O(N^2) in the
// number of rectangles, which is fine because N is at most twice the
// number of pipes in the module.
SmallVector<ActiveRect> coalesceContainedRects(ArrayRef<ActiveRect> rects) {
  SmallVector<ActiveRect> kept;
  for (size_t i = 0; i < rects.size(); ++i) {
    bool absorbed = false;
    for (size_t j = 0; j < rects.size() && !absorbed; ++j) {
      if (i == j || !rectContains(rects[j], rects[i])) {
        continue;
      }
      // rects[j] contains rects[i]. Drop rects[i] unless they are equal
      // and j is later — keep the first occurrence.
      bool equal = rectContains(rects[i], rects[j]);
      absorbed = !equal || j < i;
    }
    if (!absorbed) {
      kept.push_back(rects[i]);
    }
  }
  return kept;
}

// Build a single i1 predicate "coords lie in the union of rects" using
// arith ops. `coords[d]` is the runtime coordinate along dimension d; all
// rectangles must have the same rank as `coords`.
Value buildActivePredicate(OpBuilder &b, Location loc, ValueRange coords,
                           ArrayRef<ActiveRect> rects) {
  assert(!rects.empty() && "predicate requires at least one rectangle");
  const size_t rank = coords.size();
  for (const ActiveRect &r : rects) {
    (void)r;
    assert(r.lo.size() == rank && r.hi.size() == rank &&
           "rectangle rank must match coordinate rank");
  }

  auto idxConst = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };

  Value any;
  for (const ActiveRect &r : rects) {
    Value inRect;
    for (size_t d = 0; d < rank; ++d) {
      Value lo = idxConst(r.lo[d]);
      Value hi = idxConst(r.hi[d]);
      Value ge = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                       coords[d], lo);
      Value lt = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                       coords[d], hi);
      Value dimIn = arith::AndIOp::create(b, loc, ge, lt);
      inRect = inRect ? arith::AndIOp::create(b, loc, inRect, dimIn).getResult()
                      : dimIn;
    }
    any = any ? arith::OrIOp::create(b, loc, any, inRect).getResult() : inRect;
  }
  return any;
}

// 2D-specific node-coordinate emitter. Returns one Value per dimension,
// matching the rank of every ActiveRect produced by collectActiveRects.
// Generalize here when the dialect grows beyond `core_x` / `core_y`.
SmallVector<Value> emitNodeCoords(OpBuilder &b, Location loc) {
  Type idx = b.getIndexType();
  return {CoreXOp::create(b, loc, idx), CoreYOp::create(b, loc, idx)};
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
LogicalResult wrapFunctionBody(func::FuncOp func, ArrayRef<ActiveRect> rects) {
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

  SmallVector<Value> coords = emitNodeCoords(builder, loc);
  Value pred = buildActivePredicate(builder, loc, coords, rects);

  auto ifOp = scf::IfOp::create(builder, loc, /*resultTypes=*/TypeRange{}, pred,
                                /*withElseRegion=*/false);
  ifOp->setAttr(kActiveGuardAttrName, builder.getUnitAttr());

  // Move every operation that preceded the inserted ops (which now sit just
  // before the terminator) into the then block, except the newly inserted
  // ops themselves and the terminator.
  Block *thenBlock = ifOp.thenBlock();
  // The then block has an scf.yield inserted by the builder; preserve it.
  Operation *thenTerminator = thenBlock->getTerminator();

  // Collect ops to move: everything strictly before the first inserted
  // coordinate op in the original block.
  Operation *firstNewOp = coords.front().getDefiningOp();
  SmallVector<Operation *> toMove;
  for (Operation &op : block) {
    if (&op == firstNewOp) {
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
    rects = coalesceContainedRects(rects);

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
