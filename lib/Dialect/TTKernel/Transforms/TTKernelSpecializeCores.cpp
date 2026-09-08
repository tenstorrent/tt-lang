// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//
// Per-core specialization of TTKernel functions. Coordinate-dependent control
// flow is identified through MLIR backward slices and control-flow value
// origins. Each clone substitutes its assigned coordinates so subsequent
// canonicalization can simplify branches and loop bounds.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
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

// Determine whether `rootValue` depends on a logical core-coordinate read,
// following block arguments through `originAnalysis`. `visitedValues` prevents
// recursion through loop-carried values; an unavailable backward slice returns
// true so the caller conservatively specializes the function.
static bool valueDependsOnCore(Value rootValue,
                               const ValueOriginAnalysis &originAnalysis,
                               llvm::DenseSet<Value> &visitedValues) {
  if (!visitedValues.insert(rootValue).second) {
    return false;
  }

  BackwardSliceOptions options;
  options.inclusive = true;
  options.omitBlockArguments = false;
  options.omitUsesFromAbove = false;

  llvm::SetVector<Operation *> backwardSlice;
  if (failed(getBackwardSlice(rootValue, &backwardSlice, options))) {
    return true;
  }
  for (Operation *operation : backwardSlice) {
    if (isa<ttk::MyLogicalXOp, ttk::MyLogicalYOp>(operation)) {
      return true;
    }
  }

  // Backward slices stop at multi-region block arguments. Resolve their entry
  // and backedge values with the shared control-flow origin analysis.
  SmallVector<BlockArgument> blockArguments;
  if (auto rootArgument = dyn_cast<BlockArgument>(rootValue)) {
    blockArguments.push_back(rootArgument);
  }
  for (Operation *operation : backwardSlice) {
    for (Value operand : operation->getOperands()) {
      if (auto blockArgument = dyn_cast<BlockArgument>(operand)) {
        blockArguments.push_back(blockArgument);
      }
    }
  }
  for (BlockArgument blockArgument : blockArguments) {
    for (Value origin : originAnalysis.getOrigins(blockArgument)) {
      if (valueDependsOnCore(origin, originAnalysis, visitedValues)) {
        return true;
      }
    }
  }
  return false;
}

// Check whether `branch` uses a core-dependent value to choose a successor or
// control repetition. Values forwarded unchanged into regions are not
// selectors.
static bool
regionBranchDependsOnCore(RegionBranchOpInterface branch,
                          const ValueOriginAnalysis &originAnalysis) {
  RegionBranchSuccessorMapping forwardedOperands;
  branch.getSuccessorOperandInputMapping(forwardedOperands);

  // Forwarded values are inspected when they reach a later branch point;
  // inspect the remaining operands where they determine region control.
  for (RegionBranchPoint branchPoint : branch.getAllRegionBranchPoints()) {
    Operation *branchOperation =
        branchPoint.isParent()
            ? branch.getOperation()
            : branchPoint.getTerminatorPredecessorOrNull().getOperation();
    for (OpOperand &operand : branchOperation->getOpOperands()) {
      if (forwardedOperands.contains(&operand)) {
        continue;
      }
      llvm::DenseSet<Value> visitedValues;
      if (valueDependsOnCore(operand.get(), originAnalysis, visitedValues)) {
        return true;
      }
    }
  }
  return false;
}

// Return whether any structured branch or loop in `function` requires per-core
// specialization, sharing one origin analysis across its control-flow checks.
static bool functionControlFlowDependsOnCore(func::FuncOp function) {
  ValueOriginAnalysis originAnalysis(function);
  auto result = function.walk([&](RegionBranchOpInterface branch) {
    return regionBranchDependsOnCore(branch, originAnalysis)
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Replace every CoordOp in func clone with an arith.constant of coord.
template <typename CoordOp>
static void replaceCoordReads(func::FuncOp clone, int64_t coord) {
  SmallVector<CoordOp> reads;
  clone.walk([&](CoordOp op) { reads.push_back(op); });
  for (CoordOp op : reads) {
    OpBuilder b(op);
    Value cst =
        arith::ConstantOp::create(b, op.getLoc(), b.getIndexAttr(coord));
    op.getResult().replaceAllUsesWith(cst);
    op.erase();
  }
}

/// Emit one clone of func for core (x, y), replacing every coordinate read
/// with the matching constant and tagging the clone with ttl.core_coord.
/// TODO: See if we can leverage LaunchDomainAnalysis in an earlier pass
/// to further minimize clones.
static void emitCoreClone(func::FuncOp func, int64_t x, int64_t y,
                          OpBuilder &moduleBuilder) {
  func::FuncOp clone = func.clone();
  clone.setSymName(
      (func.getSymName() + "_c" + Twine(x) + "_" + Twine(y)).str());

  replaceCoordReads<ttk::MyLogicalXOp>(clone, x);
  replaceCoordReads<ttk::MyLogicalYOp>(clone, y);

  clone->setAttr(
      CoreCoordAttrName,
      moduleBuilder.getArrayAttr({moduleBuilder.getI64ArrayAttr({x, y})}));
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
      if (!functionControlFlowDependsOnCore(func)) {
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
          emitCoreClone(func, x, y, moduleBuilder);
        }
      }
      func.erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
