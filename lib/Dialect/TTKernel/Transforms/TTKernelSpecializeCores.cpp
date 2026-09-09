// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//
// Per-core specialization of TTKernel functions. Coordinate-dependent control
// flow is identified through MLIR backward slices and control-flow value
// origins. Each clone substitutes its assigned coordinates so subsequent
// exact evaluation and canonicalization can simplify branches and loop bounds.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Analysis/IntegerExpressionEvaluator.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
  auto gridXAttr = llvm::dyn_cast<IntegerAttr>(attr[0]);
  auto gridYAttr = llvm::dyn_cast<IntegerAttr>(attr[1]);
  if (!gridXAttr || !gridYAttr) {
    return failure();
  }
  int64_t gridX = gridXAttr.getInt();
  int64_t gridY = gridYAttr.getInt();
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
    OpBuilder builder(op);
    Value coordinateConstant = arith::ConstantOp::create(
        builder, op.getLoc(), builder.getIndexAttr(coord));
    op.getResult().replaceAllUsesWith(coordinateConstant);
    op.erase();
  }
}

/// Expose exact scf.if conditions to the standard SCF canonicalization
/// patterns. Re-evaluating after each rewrite handles conditions whose values
/// become provable only after an earlier scf.if is removed.
struct MaterializeExactIfCondition : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    Attribute constant;
    if (matchPattern(ifOp.getCondition(), m_Constant(&constant))) {
      return failure();
    }

    IntegerExpressionEvaluator evaluator;
    std::optional<llvm::APInt> exactValue =
        evaluator.evaluate(ifOp.getCondition());
    if (!exactValue) {
      return failure();
    }

    Value conditionConstant = arith::ConstantOp::create(
        rewriter, ifOp.getLoc(),
        rewriter.getBoolAttr(exactValue->getBoolValue()));
    rewriter.modifyOpInPlace(ifOp,
                             [&] { ifOp->setOperand(0, conditionConstant); });
    return success();
  }
};

static LogicalResult foldExactBranchConditions(func::FuncOp clone) {
  RewritePatternSet patterns(clone.getContext());
  patterns.add<MaterializeExactIfCondition>(clone.getContext());
  scf::IfOp::getCanonicalizationPatterns(patterns, clone.getContext());
  return applyPatternsGreedily(clone, std::move(patterns));
}

/// Build one detached clone for a core, replacing every coordinate read with
/// the matching constant and tagging the clone with ttl.core_coord.
/// TODO: See if we can leverage LaunchDomainAnalysis in an earlier pass
/// to further minimize clones.
static FailureOr<func::FuncOp> buildCoreClone(func::FuncOp func, int64_t coreX,
                                              int64_t coreY,
                                              Builder &attributeBuilder) {
  func::FuncOp clone = func.clone();
  clone.setSymName(
      (func.getSymName() + "_c" + Twine(coreX) + "_" + Twine(coreY)).str());

  replaceCoordReads<ttk::MyLogicalXOp>(clone, coreX);
  replaceCoordReads<ttk::MyLogicalYOp>(clone, coreY);
  if (failed(foldExactBranchConditions(clone))) {
    clone->destroy();
    return failure();
  }

  clone->setAttr(CoreCoordAttrName,
                 attributeBuilder.getArrayAttr(
                     {attributeBuilder.getI64ArrayAttr({coreX, coreY})}));
  return clone;
}

struct FunctionSpecialization {
  func::FuncOp original;
  SmallVector<func::FuncOp> clones;
};

static void
destroyDetachedClones(ArrayRef<FunctionSpecialization> specializations,
                      ArrayRef<func::FuncOp> pendingClones = {}) {
  for (func::FuncOp clone : pendingClones) {
    clone->destroy();
  }
  for (const FunctionSpecialization &specialization : specializations) {
    for (func::FuncOp clone : specialization.clones) {
      clone->destroy();
    }
  }
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

    auto allocationMode =
        module->getAttrOfType<StringAttr>("ttl.sram_allocation_mode");
    bool independentStorage =
        allocationMode && allocationMode.getValue() == "per-core";
    SmallVector<func::FuncOp> targets;
    for (auto func : module.getOps<func::FuncOp>()) {
      bool requiresStorageBinding =
          independentStorage && func->hasAttr(ttk::ThreadTypeAttr::name);
      if (!requiresStorageBinding && !functionControlFlowDependsOnCore(func)) {
        continue;
      }
      if (auto uses = SymbolTable::getSymbolUses(func, module);
          uses && !uses->empty()) {
        if (requiresStorageBinding) {
          func.emitOpError(
              "per-core SRAM kernel cannot have symbol references");
          signalPassFailure();
          return;
        }
        func.emitWarning() << "not specializing '" << func.getSymName()
                           << "': function has symbol uses";
        continue;
      }
      targets.push_back(func);
    }

    Builder attributeBuilder(module.getContext());
    SmallVector<FunctionSpecialization> specializations;
    for (func::FuncOp func : targets) {
      SmallVector<func::FuncOp> clones;
      for (int64_t coreY = 0; coreY < gridY; ++coreY) {
        for (int64_t coreX = 0; coreX < gridX; ++coreX) {
          FailureOr<func::FuncOp> clone =
              buildCoreClone(func, coreX, coreY, attributeBuilder);
          if (failed(clone)) {
            func.emitOpError("failed to simplify exact per-core conditions");
            destroyDetachedClones(specializations, clones);
            signalPassFailure();
            return;
          }
          clones.push_back(*clone);
        }
      }
      specializations.push_back({func, std::move(clones)});
    }

    for (FunctionSpecialization &specialization : specializations) {
      OpBuilder moduleBuilder(specialization.original);
      for (func::FuncOp clone : specialization.clones) {
        moduleBuilder.insert(clone);
      }
      specialization.original.erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
