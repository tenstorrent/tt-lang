// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTLSpecializeCores
//
// Specializes every kernel function that carries a `ttl.operation_grid`
// attribute for the launch grid it runs on. Each emitted clone is tagged with
// `ttl.core_coord` (a list of `[x, y]` launch coordinates the clone serves) and
// has `ttl.core_x` / `ttl.core_y` const-folded to a representative coordinate.
// Downstream `sccp`/`canonicalize`/`cse`/DCE then specialize each clone by
// folding coordinate-dependent predicates and deleting dead branches.
//
// To avoid emitting one near-identical binary per core, the pass groups
// coordinates whose control flow is identical (computed with
// `LaunchNodeDomainAnalysis`) and emits a single clone per group. This
// de-duplication is only sound when the coordinate is used purely to drive
// control flow; if `ttl.core_x` / `ttl.core_y` feed a data value (e.g. a tensor
// address), folding a group to one representative would bake that coordinate's
// constants into every core, so the pass falls back to one clone per
// coordinate.
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <map>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLSPECIALIZECORES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Module attribute (per function) giving the `[gridX, gridY]` extent the
/// function is launched over.
constexpr llvm::StringLiteral kOperationGridAttrName = "ttl.operation_grid";

/// Attribute set on each specialized clone recording the `[x, y]` coordinates
/// it serves, as an array of length-2 `[x, y]` arrays.
constexpr llvm::StringLiteral kCoreCoordAttrName = "ttl.core_coord";

/// Read a length-2 i64 array attribute into (x, y). Returns false when the
/// attribute is missing or malformed.
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
  for (Operation *op : toErase) {
    op->erase();
  }
}

/// Return true when a `ttl.core_x` / `ttl.core_y` result reaches a data value
/// (anything other than a branch predicate). De-duplicating coordinates is only
/// sound when this returns false, because a merged clone folds the coordinate
/// to a single representative.
static bool coordUsedAsData(func::FuncOp func) {
  llvm::DenseMap<Value, bool> memo;
  std::function<bool(Value)> isDataUsed = [&](Value v) -> bool {
    if (auto it = memo.find(v); it != memo.end()) {
      return it->second;
    }
    // Tentatively break potential cycles; coordinate arith graphs are acyclic.
    memo[v] = false;
    bool dataUsed = false;
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      // `scf.if`'s only operand is its condition, and `affine.if` operands are
      // all inputs to the integer set: both are predicate uses.
      if (isa<scf::IfOp, affine::AffineIfOp>(user)) {
        continue;
      }
      // Pure integer/index arithmetic can compose a predicate; recurse into the
      // result to see whether it too stays confined to predicates.
      if (isa<arith::CmpIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
              arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::IndexCastOp,
              arith::ExtUIOp, arith::ExtSIOp>(user)) {
        for (Value res : user->getResults()) {
          if (isDataUsed(res)) {
            dataUsed = true;
            break;
          }
        }
        if (dataUsed) {
          break;
        }
        continue;
      }
      // Any other consumer treats the coordinate as a data value.
      dataUsed = true;
      break;
    }
    memo[v] = dataUsed;
    return dataUsed;
  };

  bool result = false;
  func.walk([&](Operation *op) {
    if (isa<CoreXOp, CoreYOp>(op) && isDataUsed(op->getResult(0))) {
      result = true;
    }
  });
  return result;
}

/// Partition the launch grid into groups of coordinates that follow identical
/// control flow. Returns nullopt when reachability could not be proven
/// statically (an unknown domain or a solver failure), signalling the caller to
/// fall back to one clone per coordinate.
static std::optional<SmallVector<SmallVector<LaunchNodeCoord>>>
groupCoordsByControlFlow(func::FuncOp func, ModuleOp module, int64_t gridX,
                         int64_t gridY) {
  LaunchNodeDomainState state;
  state.initialize(module);
  // The specialize pass keys off the per-function `ttl.operation_grid`, so
  // override the module-level launch grid with this function's extent.
  state.baseDomain = getFullLaunchNodeDomain(gridX, gridY);
  state.hasLaunchGrid = true;

  llvm::DenseMap<Operation *, LaunchNodeDomain> opDomains;
  bool sawUnknown = false;
  LaunchNodeDomainAnalysisOptions options;
  options.operationCallback = [&](Operation *op, const LaunchNodeDomain &domain,
                                  Operation *) {
    opDomains[op] = domain;
    if (!domain.known) {
      sawUnknown = true;
    }
  };

  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<LaunchNodeDomainAnalysis>(state, options);
  if (failed(solver.initializeAndRun(func)) || sawUnknown) {
    return std::nullopt;
  }

  const size_t fullSize = state.baseDomain.nodes.size();

  // Only ops that are not reached by the whole grid distinguish one
  // coordinate's control flow from another's.
  SmallVector<Operation *> discriminators;
  func.walk([&](Operation *op) {
    auto it = opDomains.find(op);
    if (it == opDomains.end() || !it->second.known) {
      return;
    }
    if (it->second.nodes.size() != fullSize) {
      discriminators.push_back(op);
    }
  });

  // Each coordinate's signature is its membership bitset across the
  // discriminating ops. Coordinates with equal signatures reach exactly the
  // same ops, i.e. execute identical control flow.
  std::map<std::string, SmallVector<LaunchNodeCoord>> groups;
  for (int64_t y = 0; y < gridY; ++y) {
    for (int64_t x = 0; x < gridX; ++x) {
      LaunchNodeCoord coord{x, y};
      std::string signature;
      signature.reserve(discriminators.size());
      for (Operation *op : discriminators) {
        signature.push_back(opDomains[op].nodes.count(coord) ? '1' : '0');
      }
      groups[signature].push_back(coord);
    }
  }

  SmallVector<SmallVector<LaunchNodeCoord>> result;
  result.reserve(groups.size());
  for (auto &entry : groups) {
    result.push_back(std::move(entry.second));
  }
  return result;
}

/// Clone `func` for one group of coordinates, tag it with the group's
/// `ttl.core_coord` list, and const-fold the coordinate ops to the group's
/// representative (its first coordinate).
static void emitSpecializedClone(func::FuncOp func, OpBuilder &moduleBuilder,
                                 Builder &builder,
                                 ArrayRef<LaunchNodeCoord> coords) {
  assert(!coords.empty() && "expected at least one coordinate per clone");
  LaunchNodeCoord rep = coords.front();

  func::FuncOp clone = func.clone();
  clone.setSymName(
      (func.getSymName() + "_c" + Twine(rep.x) + "_" + Twine(rep.y)).str());
  clone->removeAttr(kOperationGridAttrName);

  SmallVector<Attribute> coordAttrs;
  coordAttrs.reserve(coords.size());
  for (LaunchNodeCoord coord : coords) {
    coordAttrs.push_back(builder.getI64ArrayAttr({coord.x, coord.y}));
  }
  clone->setAttr(kCoreCoordAttrName, builder.getArrayAttr(coordAttrs));

  moduleBuilder.insert(clone);
  constFoldCoreOps(clone, rep.x, rep.y);
}

struct TTLSpecializeCoresPass
    : impl::TTLSpecializeCoresBase<TTLSpecializeCoresPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<func::FuncOp> templates;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func->hasAttr(kOperationGridAttrName)) {
        templates.push_back(func);
      }
    }

    Builder builder(&getContext());
    for (func::FuncOp func : templates) {
      int64_t gridX, gridY;
      auto gridAttr = func->getAttrOfType<ArrayAttr>(kOperationGridAttrName);
      if (!readGrid(gridAttr, gridX, gridY)) {
        func->emitOpError()
            << "invalid " << kOperationGridAttrName
            << " attribute (expected [gridX, gridY] with positive entries)";
        signalPassFailure();
        return;
      }

      OpBuilder moduleBuilder(func);

      // De-duplicate coordinates by control flow when the coordinate only
      // drives predicates; otherwise emit one clone per coordinate.
      std::optional<SmallVector<SmallVector<LaunchNodeCoord>>> groups;
      if (!coordUsedAsData(func)) {
        groups = groupCoordsByControlFlow(func, module, gridX, gridY);
      }

      if (groups) {
        for (const SmallVector<LaunchNodeCoord> &group : *groups) {
          emitSpecializedClone(func, moduleBuilder, builder, group);
        }
      } else {
        for (int64_t y = 0; y < gridY; ++y) {
          for (int64_t x = 0; x < gridX; ++x) {
            LaunchNodeCoord coord{x, y};
            emitSpecializedClone(func, moduleBuilder, builder, coord);
          }
        }
      }

      func.erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
