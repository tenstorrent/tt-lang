// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernelSpecializeCores (per-core specialization)
//
// A single module pass that runs at the TTKernel level, right before EmitC
// conversion. For every kernel function whose control flow branches on a core
// coordinate (i.e. an `scf.if` whose condition is derived from
// `ttkernel.my_logical_x_` / `ttkernel.my_logical_y_`), the pass clones the
// function once per launch coordinate. In each clone, the coordinate reads are
// replaced by `arith.constant`s for that core, so the following
// `canonicalize` / `cse` fold the now-constant branch conditions and delete the
// untaken regions. Each clone is tagged with a `ttl.core_coord` attribute (the
// coordinate it serves) and the runtime bridge (ttl_api.py) turns that into a
// per-kernel core range for dispatch.
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace ttk = mlir::tt::ttkernel;

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELSPECIALIZECORES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

// Attribute names. These are part of the frontend / runtime contract and keep
// the `ttl.` prefix even though this pass runs at the TTKernel level:
// `ttl.launch_grid` (the launch extent) is set on the module by the Python
// frontend, and `ttl.core_coord` is read back by the ttnn runtime bridge for
// dispatch.
constexpr llvm::StringLiteral LaunchGridAttrName = "ttl.launch_grid";
constexpr llvm::StringLiteral CoreCoordAttrName = "ttl.core_coord";

/// Parse the launch extent from an i64 array attribute into (gridX, gridY).
///
/// NOTE: operations.py specifies that only dims=2 is supported for now.
///       this should be updated once operations.py is updated
static bool readGrid(ArrayAttr attr, int64_t &gridX, int64_t &gridY) {
  if (!attr || attr.size() != 2) {
    return false;
  }
  auto x = llvm::dyn_cast<IntegerAttr>(attr[0]);
  auto y = llvm::dyn_cast<IntegerAttr>(attr[1]);
  if (!x || !y) {
    return false;
  }
  gridX = x.getInt();
  gridY = y.getInt();
  return gridX > 0 && gridY > 0;
}

/// Return true when `condition` is derived from a core
/// coordinate reads (`ttkernel.my_logical_x_` / `my_logical_y_`).
static bool conditionDependsOnCore(Value condition) {
  llvm::DenseSet<Value> visited;
  SmallVector<Value> worklist{condition};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second) {
      continue;
    }
    Operation *op = value.getDefiningOp();
    if (!op) {
      continue;
    }
    if (isa<ttk::MyLogicalXOp, ttk::MyLogicalYOp>(op)) {
      return true;
    }
    worklist.append(op->operand_begin(), op->operand_end());
  }
  return false;
}

/// Return true when `func` has any `scf.if` whose condition branches on a core
/// coordinate. Only such functions need per-core clones.
static bool funcBranchesOnCore(func::FuncOp func) {
  bool found = false;
  func.walk([&](scf::IfOp ifOp) {
    if (conditionDependsOnCore(ifOp.getCondition())) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Emit one clone of `func` for core `(x, y)`, replacing every coordinate read
/// with the matching constant and tagging the clone with `ttl.core_coord`.
/// TODO: See if we can leverage LaunchDomainAnalysis in an earlier pass
/// to further minimize clones.
static void emitCoreClone(func::FuncOp func, int64_t x, int64_t y,
                          OpBuilder &moduleBuilder, Builder &builder) {
  func::FuncOp clone = func.clone();
  clone.setSymName(
      (func.getSymName() + "_c" + Twine(x) + "_" + Twine(y)).str());

  SmallVector<ttk::MyLogicalXOp> xReads;
  SmallVector<ttk::MyLogicalYOp> yReads;
  clone.walk([&](Operation *op) {
    if (auto xr = dyn_cast<ttk::MyLogicalXOp>(op)) {
      xReads.push_back(xr);
    } else if (auto yr = dyn_cast<ttk::MyLogicalYOp>(op)) {
      yReads.push_back(yr);
    }
  });
  for (ttk::MyLogicalXOp xr : xReads) {
    OpBuilder b(xr);
    Value cst = arith::ConstantOp::create(b, xr.getLoc(), b.getIndexAttr(x));
    xr.getResult().replaceAllUsesWith(cst);
    xr.erase();
  }
  for (ttk::MyLogicalYOp yr : yReads) {
    OpBuilder b(yr);
    Value cst = arith::ConstantOp::create(b, yr.getLoc(), b.getIndexAttr(y));
    yr.getResult().replaceAllUsesWith(cst);
    yr.erase();
  }

  clone->setAttr(CoreCoordAttrName,
                 builder.getArrayAttr({builder.getI64ArrayAttr({x, y})}));
  moduleBuilder.insert(clone);
}

struct TTKernelSpecializeCoresPass
    : impl::TTKernelSpecializeCoresBase<TTKernelSpecializeCoresPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto gridAttr = module->getAttrOfType<ArrayAttr>(LaunchGridAttrName);
    if (!gridAttr) {
      module.emitOpError() << "requires a `" << LaunchGridAttrName
                           << "` module attribute";
      signalPassFailure();
      return;
    }
    int64_t gridX = 0, gridY = 0;
    if (!readGrid(gridAttr, gridX, gridY)) {
      module.emitOpError() << "`" << LaunchGridAttrName
                           << "` must be a length-2 array of positive i64 "
                              "extents";
      signalPassFailure();
      return;
    }

    if (gridX * gridY <= 1) {
      return;
    }

    // Cloning renames a target and erases the original; inter-function
    // SymbolRefAttr fixups are not performed. A referenced function is left
    // un-specialized (still a correct whole-grid binary via its runtime
    // coordinate reads) rather than failing the whole pass, so unrelated
    // functions still get specialized.
    SmallVector<func::FuncOp> targets;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (!funcBranchesOnCore(func)) {
        continue;
      }
      if (auto uses = SymbolTable::getSymbolUses(func, module);
          uses && !uses->empty()) {
        func.emitWarning() << "not specializing '" << func.getSymName()
                           << "': function has symbol uses";
        continue;
      }
      targets.push_back(func);
    }

    Builder builder(&getContext());
    for (func::FuncOp func : targets) {
      OpBuilder moduleBuilder(func);
      for (int64_t y = 0; y < gridY; ++y) {
        for (int64_t x = 0; x < gridX; ++x) {
          emitCoreClone(func, x, y, moduleBuilder, builder);
        }
      }
      func.erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
