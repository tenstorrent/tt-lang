// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTLSpecializeCores
//
// SKETCH / WORK-IN-PROGRESS (see the per-core-dispatch epic).
//
// Clones every kernel function that carries a `ttl.operation_grid` attribute
// once per launch coordinate, tags each clone with `ttl.core_coord`, and
// const-folds `ttl.core_x` / `ttl.core_y` inside the clone to the concrete
// coordinate. Downstream `sccp`/`canonicalize`/`cse`/DCE then specialize each
// clone by folding coordinate-dependent predicates and deleting dead branches.
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLSPECIALIZECORES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Module attribute (per function) giving the `[gridX, gridY]` extent the
/// function is launched over.
constexpr llvm::StringLiteral kOperationGridAttrName = "ttl.operation_grid";

/// Attribute set on each specialized clone recording its `[x, y]` coordinate.
constexpr llvm::StringLiteral kCoreCoordAttrName = "ttl.core_coord";

/// Read a length-2 i64 array attribute into (x, y). Returns false when the
/// attribute is missing or malformed.
static bool readGrid(ArrayAttr attr, int64_t &gridX, int64_t &gridY) {
  if (!attr || attr.size() != 2)
    return false;
  auto x = llvm::dyn_cast<IntegerAttr>(attr[0]);
  auto y = llvm::dyn_cast<IntegerAttr>(attr[1]);
  if (!x || !y)
    return false;
  gridX = x.getInt();
  gridY = y.getInt();
  return gridX > 0 && gridY > 0;
}

/// Replace every `ttl.core_x` / `ttl.core_y` result in `func` with the
/// constant coordinate (`x`, `y`).
static void constFoldCoreOps(func::FuncOp func, int64_t x, int64_t y) {
  SmallVector<Operation *> toErase;
  func.walk([&](Operation *op) {
    int64_t value;
    if (isa<CoreXOp>(op)) {
      value = x;
    } else if (isa<CoreYOp>(op)) {
      value = y;
    } else {
      return;
    }
    OpBuilder builder(op);
    Value cst = arith::ConstantIndexOp::create(builder, op->getLoc(), value);
    op->getResult(0).replaceAllUsesWith(cst);
    toErase.push_back(op);
  });
  for (Operation *op : toErase)
    op->erase();
}

struct TTLSpecializeCoresPass
    : impl::TTLSpecializeCoresBase<TTLSpecializeCoresPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<func::FuncOp> templates;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func->hasAttr(kOperationGridAttrName))
        templates.push_back(func);
    }

    Builder builder(&getContext());
    for (func::FuncOp func : templates) {
      int64_t gridX, gridY;
      auto gridAttr =
          func->getAttrOfType<ArrayAttr>(kOperationGridAttrName);
      if (!readGrid(gridAttr, gridX, gridY)) {
        func->emitOpError()
            << "invalid " << kOperationGridAttrName
            << " attribute (expected [gridX, gridY] with positive entries)";
        signalPassFailure();
        return;
      }

      OpBuilder moduleBuilder(func);
      // TODO(per-core-dispatch): group coordinates whose control flow is
      // identical (e.g. via LaunchNodeDomainAnalysis) and emit one clone per
      // group instead of one clone per coordinate.
      for (int64_t y = 0; y < gridY; ++y) {
        for (int64_t x = 0; x < gridX; ++x) {
          func::FuncOp clone = func.clone();
          clone.setSymName(
              (func.getSymName() + "_c" + Twine(x) + "_" + Twine(y)).str());
          clone->removeAttr(kOperationGridAttrName);
          clone->setAttr(
              kCoreCoordAttrName,
              builder.getI64ArrayAttr({x, y}));
          moduleBuilder.insert(clone);
          constFoldCoreOps(clone, x, y);
        }
      }

      func.erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
