// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::ttkernel {
#define GEN_PASS_DEF_TTKERNELFUSESIBLINGTILELOOPS
#include "ttlang/Dialect/TTKernel/Passes.h.inc"

namespace {

static constexpr llvm::StringLiteral kTileLoopMarker = "ttkernel.tile_loop";

/// Check if two values are equivalent (same SSA value or same constant).
static bool areValuesEquivalent(Value v1, Value v2) {
  if (v1 == v2) {
    return true;
  }

  IntegerAttr c1, c2;
  if (matchPattern(v1, m_Constant(&c1)) && matchPattern(v2, m_Constant(&c2))) {
    return c1.getInt() == c2.getInt();
  }
  return false;
}

/// Check if two loops have identical bounds (lb, ub, step).
static bool haveSameBounds(scf::ForOp a, scf::ForOp b) {
  return areValuesEquivalent(a.getLowerBound(), b.getLowerBound()) &&
         areValuesEquivalent(a.getUpperBound(), b.getUpperBound()) &&
         areValuesEquivalent(a.getStep(), b.getStep());
}

/// Check if two loops are adjacent in the same block with only constants
/// between them.
static bool areAdjacent(scf::ForOp first, scf::ForOp second) {
  if (first->getBlock() != second->getBlock()) {
    return false;
  }
  if (!first->isBeforeInBlock(second)) {
    return false;
  }

  for (Operation *op = first->getNextNode(); op != second;
       op = op->getNextNode()) {
    if (!isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp>(
            op)) {
      return false;
    }
  }
  return true;
}

// Forward declaration for mutual recursion.
static void fuseInnerLoops(scf::ForOp fusedOuter, IRRewriter &rewriter);

/// Fuse two loops and recursively fuse their inner loops.
static scf::ForOp fuseLoopPair(scf::ForOp target, scf::ForOp source,
                               IRRewriter &rewriter) {
  scf::ForOp fused = fuseIndependentSiblingForLoops(target, source, rewriter);
  if (fused) {
    fused->setAttr(kTileLoopMarker, rewriter.getUnitAttr());
    fuseInnerLoops(fused, rewriter);
  }
  return fused;
}

/// Find and fuse adjacent inner loops within a fused outer loop.
static void fuseInnerLoops(scf::ForOp outer, IRRewriter &rewriter) {
  SmallVector<scf::ForOp> innerLoops;
  for (Operation &op : *outer.getBody()) {
    if (auto inner = dyn_cast<scf::ForOp>(&op)) {
      if (!inner->hasAttr(kTileLoopMarker)) {
        continue;
      }
      innerLoops.push_back(inner);
    }
  }

  // Fuse adjacent pairs with matching bounds.
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i + 1 < innerLoops.size(); ++i) {
      if (areAdjacent(innerLoops[i], innerLoops[i + 1]) &&
          haveSameBounds(innerLoops[i], innerLoops[i + 1])) {
        auto fused = fuseLoopPair(innerLoops[i], innerLoops[i + 1], rewriter);
        if (fused) {
          innerLoops.erase(innerLoops.begin() + i, innerLoops.begin() + i + 2);
          innerLoops.insert(innerLoops.begin() + i, fused);
          changed = true;
          break;
        }
      }
    }
  }
}

struct TTKernelFuseSiblingTileLoopsPass
    : impl::TTKernelFuseSiblingTileLoopsBase<TTKernelFuseSiblingTileLoopsPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    // Process each block looking for adjacent fusable loops.
    func.walk([&](Block *block) {
      bool changed = true;
      while (changed) {
        changed = false;

        SmallVector<scf::ForOp> loops;
        for (Operation &op : *block) {
          if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
            if (!forOp->hasAttr(kTileLoopMarker)) {
              continue;
            }
            loops.push_back(forOp);
          }
        }

        for (size_t i = 0; i + 1 < loops.size(); ++i) {
          if (areAdjacent(loops[i], loops[i + 1]) &&
              haveSameBounds(loops[i], loops[i + 1])) {
            if (fuseLoopPair(loops[i], loops[i + 1], rewriter)) {
              changed = true;
              break;
            }
          }
        }
      }
    });

    // Strip the tile-loop marker after fusion (proper cleanup)
    func.walk([](scf::ForOp loop) {
      if (loop->hasAttr(kTileLoopMarker)) {
        loop->removeAttr(kTileLoopMarker);
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttkernel
