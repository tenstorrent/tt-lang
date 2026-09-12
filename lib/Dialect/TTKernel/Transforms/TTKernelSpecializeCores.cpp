// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//
// Per-core specialization of TTKernel functions. Each clone substitutes its
// assigned coordinates so exact evaluation and canonicalization can simplify
// coordinate-dependent branches.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Analysis/IntegerExpressionEvaluator.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

/// Expose exact conditions to the standard `scf.if` canonicalization patterns.
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
