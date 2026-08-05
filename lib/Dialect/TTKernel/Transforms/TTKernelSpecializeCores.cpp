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
// function once per launch coordinate. In each clone, coordinate reads in the
// backward slice of branch predicates are replaced by `arith.constant`s for
// that core. Addressing and data uses remain dynamic by default. Full
// specialization instead replaces every coordinate read in each clone. The following
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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
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
static FailureOr<std::pair<int64_t, int64_t>> readGrid(ArrayAttr attr) {
  if (!attr || attr.size() != 2) {
    return failure();
  }
  auto x = llvm::dyn_cast<IntegerAttr>(attr[0]);
  auto y = llvm::dyn_cast<IntegerAttr>(attr[1]);
  if (!x || !y) {
    return failure();
  }
  int64_t gridX = x.getInt();
  int64_t gridY = y.getInt();
  if (gridX <= 0 || gridY <= 0) {
    return failure();
  }
  return std::pair<int64_t, int64_t>{gridX, gridY};
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

/// Return true when `func` reads either logical core coordinate anywhere.
static bool funcReadsCore(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<ttk::MyLogicalXOp, ttk::MyLogicalYOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Clone the pure backward slice for `value`, replacing coordinate leaves with
/// constants. The original slice remains available to non-control-flow users
/// such as bank, shard, and NoC address calculations.
static FailureOr<Value>
cloneSpecializedPredicateValue(Value value, int64_t x, int64_t y,
                               OpBuilder &builder,
                               llvm::DenseMap<Value, Value> &specialized) {
  if (auto mapped = specialized.find(value); mapped != specialized.end()) {
    return mapped->second;
  }

  Operation *op = value.getDefiningOp();
  if (!op) {
    specialized[value] = value;
    return value;
  }

  if (isa<ttk::MyLogicalXOp>(op) || isa<ttk::MyLogicalYOp>(op)) {
    int64_t coord = isa<ttk::MyLogicalXOp>(op) ? x : y;
    Value constant = arith::ConstantOp::create(
        builder, op->getLoc(), builder.getIndexAttr(coord));
    specialized[value] = constant;
    return constant;
  }

  if (!conditionDependsOnCore(value)) {
    specialized[value] = value;
    return value;
  }
  if (!isPure(op) || op->getNumRegions() != 0) {
    return failure();
  }

  IRMapping mapping;
  for (Value operand : op->getOperands()) {
    FailureOr<Value> specializedOperand =
        cloneSpecializedPredicateValue(operand, x, y, builder, specialized);
    if (failed(specializedOperand)) {
      return failure();
    }
    mapping.map(operand, *specializedOperand);
  }
  Operation *cloned = builder.clone(*op, mapping);
  for (auto [originalResult, clonedResult] :
       llvm::zip(op->getResults(), cloned->getResults())) {
    specialized[originalResult] = clonedResult;
  }
  return specialized.lookup(value);
}

/// Specialize only coordinate-dependent branch predicates. Coordinate reads
/// used by the selected branch body or by addressing remain dynamic.
static LogicalResult specializeBranchPredicates(func::FuncOp clone, int64_t x,
                                                int64_t y) {
  SmallVector<scf::IfOp> coordinateIfs;
  clone.walk([&](scf::IfOp ifOp) {
    if (conditionDependsOnCore(ifOp.getCondition())) {
      coordinateIfs.push_back(ifOp);
    }
  });
  for (scf::IfOp ifOp : coordinateIfs) {
    OpBuilder builder(ifOp);
    llvm::DenseMap<Value, Value> specialized;
    FailureOr<Value> condition = cloneSpecializedPredicateValue(
        ifOp.getCondition(), x, y, builder, specialized);
    if (failed(condition)) {
      return failure();
    }
    ifOp.getConditionMutable().assign(*condition);
  }
  return success();
}

/// Replace every coordinate read of `CoordOp` in `clone` with `coord`.
template <typename CoordOp>
static void specializeAllCoordinateReads(func::FuncOp clone, int64_t coord) {
  SmallVector<CoordOp> reads;
  clone.walk([&](CoordOp op) { reads.push_back(op); });
  for (CoordOp op : reads) {
    OpBuilder builder(op);
    Value constant = arith::ConstantOp::create(
        builder, op.getLoc(), builder.getIndexAttr(coord));
    op.getResult().replaceAllUsesWith(constant);
    op.erase();
  }
}

/// Emit one clone of func for core (x, y), specializing its branch predicates
/// and tagging the clone with ttl.core_coord.
/// TODO: See if we can leverage LaunchDomainAnalysis in an earlier pass
/// to further minimize clones.
static FailureOr<func::FuncOp> emitCoreClone(func::FuncOp func, int64_t x,
                                            int64_t y, bool fullSpecialization,
                                            OpBuilder &moduleBuilder) {
  func::FuncOp clone = func.clone();
  clone.setSymName(
      (func.getSymName() + "_c" + Twine(x) + "_" + Twine(y)).str());

  if (fullSpecialization) {
    specializeAllCoordinateReads<ttk::MyLogicalXOp>(clone, x);
    specializeAllCoordinateReads<ttk::MyLogicalYOp>(clone, y);
  } else if (failed(specializeBranchPredicates(clone, x, y))) {
    clone.erase();
    return failure();
  }

  clone->setAttr(
      CoreCoordAttrName,
      moduleBuilder.getArrayAttr({moduleBuilder.getI64ArrayAttr({x, y})}));
  moduleBuilder.insert(clone);
  return clone;
}

struct TTKernelSpecializeCoresPass
    : impl::TTKernelSpecializeCoresBase<TTKernelSpecializeCoresPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto gridAttr = module->getAttrOfType<ArrayAttr>(LaunchGridAttrName);
    if (!gridAttr) {
      module.emitOpError() << "requires a `" << LaunchGridAttrName
                           << "` module attribute";
      signalPassFailure();
      return;
    }
    FailureOr<std::pair<int64_t, int64_t>> grid = readGrid(gridAttr);
    if (failed(grid)) {
      module.emitOpError() << "`" << LaunchGridAttrName
                           << "` must be a length-2 array of positive i64 "
                              "extents";
      signalPassFailure();
      return;
    }
    auto [gridX, gridY] = *grid;

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
      bool shouldSpecialize = fullSpecialization ? funcReadsCore(func)
                                                 : funcBranchesOnCore(func);
      if (!shouldSpecialize) {
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
    for (func::FuncOp func : targets) {
      OpBuilder moduleBuilder(func);
      for (int64_t y = 0; y < gridY; ++y) {
        for (int64_t x = 0; x < gridX; ++x) {
          if (failed(emitCoreClone(func, x, y, fullSpecialization,
                                   moduleBuilder))) {
            func.emitOpError()
                << "cannot specialize a coordinate-dependent predicate "
                   "containing non-pure or region operations";
            signalPassFailure();
            return;
          }
        }
      }
      func.erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
