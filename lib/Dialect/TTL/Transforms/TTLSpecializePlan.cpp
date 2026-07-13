// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTLSpecializePlan (Phase A of per-core specialization)
//
// Annotates each kernel `func.func` with a `ttl.specialize_plan` describing how
// it should be cloned per launch coordinate. This pass only reads its function
// (plus read-only facts from the parent module) and writes attributes, so it is
// safe to run in parallel across functions.
//
// It uses `LaunchNodeDomainAnalysis` to learn which launch coordinates reach
// each program point, then considers only `scf.if` branches whose condition
// depends on `ttl.core_x` / `ttl.core_y`. Such branches are stamped with a
// `ttl.specialize_branch` id, coordinates are partitioned into groups that take
// the same branch outcomes, and the grouping (plus per-branch outcomes) is
// recorded on the function. Phase B (`ttl-specialize-cores`) materializes the
// clones from this plan at the TTKernel level.
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/SpecializeCoresAttrs.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <string>
#include <utility>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLSPECIALIZEPLAN
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

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

/// Return true when any function in `module` participates in a PipeNet.
///
/// Cloning a kernel per core interacts badly with PipeNet lowering: a pipe
/// handshake couples several kernels (readers, writers, compute) through
/// per-core semaphores and whole-grid dispatch, so specializing even a
/// non-pipe kernel in such a module changes the per-core kernel layout and
/// deadlocks at runtime. The minimum version is therefore conservative and
/// refuses to specialize *any* function in a module that uses pipes, leaving
/// every kernel as a single whole-grid binary where the runtime guards handle
/// roles.
///
/// Alternative (deferred): a finer, per-function gate that specializes non-pipe
/// kernels while proving the clone does not perturb the pipe protocol.
static bool moduleUsesPipes(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    if (isa<PipeNetPredicateOpInterface, IfSrcOp, IfDstOp>(op)) {
      return WalkResult::interrupt();
    }
    if (op->getName().getStringRef().contains("pipe")) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Return true when `condition` is (transitively) derived from a
/// `ttl.core_x` / `ttl.core_y` op. Pure PipeNet predicates (`is_active` etc.)
/// have no operands, so a role-only condition returns false here.
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
    if (isa<CoreXOp, CoreYOp>(op)) {
      return true;
    }
    worklist.append(op->operand_begin(), op->operand_end());
  }
  return false;
}

struct TTLSpecializePlanPass
    : impl::TTLSpecializePlanBase<TTLSpecializePlanPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto gridAttr = func->getAttrOfType<ArrayAttr>(kOperationGridAttrName);
    int64_t gridX = 0, gridY = 0;
    if (!readGrid(gridAttr, gridX, gridY)) {
      return;
    }

    auto module = func->getParentOfType<ModuleOp>();
    if (!module) {
      return;
    }

    // Conservatively skip specialization for any module that uses pipes;
    // cloning perturbs the pipe handshake and deadlocks at runtime.
    if (moduleUsesPipes(module)) {
      return;
    }

    LaunchNodeDomainState state;
    state.initialize(module);
    // Key off the per-function operation grid, not the module launch grid.
    state.baseDomain = getFullLaunchNodeDomain(gridX, gridY);
    state.hasLaunchGrid = true;

    llvm::DenseMap<Operation *, LaunchNodeDomain> opDomains;
    bool sawUnknown = false;
    LaunchNodeDomainAnalysisOptions options;
    options.operationCallback =
        [&](Operation *op, const LaunchNodeDomain &domain, Operation *) {
          opDomains[op] = domain;
          if (!domain.known) {
            sawUnknown = true;
          }
        };

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<LaunchNodeDomainAnalysis>(state, options);
    if (failed(solver.initializeAndRun(func)) || sawUnknown) {
      return;
    }

    // Collect, in deterministic walk order, every scf.if that branches on a
    // core coordinate and record the launch coordinates that enter its then
    // region.
    SmallVector<scf::IfOp> coreBranches;
    SmallVector<LaunchNodeDomain> thenDomains;
    func.walk([&](scf::IfOp ifOp) {
      if (!conditionDependsOnCore(ifOp.getCondition())) {
        return;
      }
      Region &thenRegion = ifOp.getThenRegion();
      if (thenRegion.empty() || thenRegion.front().empty()) {
        return;
      }
      Operation *thenFront = &thenRegion.front().front();
      auto it = opDomains.find(thenFront);
      if (it == opDomains.end() || !it->second.known) {
        return;
      }
      coreBranches.push_back(ifOp);
      thenDomains.push_back(it->second);
    });

    if (coreBranches.empty()) {
      return;
    }

    // Group coordinates by their branch-outcome signature. The signature is the
    // membership bitset across all core branches: coordinates with equal
    // signatures execute identical control flow.
    std::map<std::string,
             std::pair<SmallVector<LaunchNodeCoord>, SmallVector<bool>>>
        groups;
    for (int64_t y = 0; y < gridY; ++y) {
      for (int64_t x = 0; x < gridX; ++x) {
        LaunchNodeCoord coord{x, y};
        std::string signature;
        signature.reserve(coreBranches.size());
        SmallVector<bool> taken;
        taken.reserve(coreBranches.size());
        for (const LaunchNodeDomain &thenDomain : thenDomains) {
          bool takesThen = thenDomain.nodes.count(coord) != 0;
          signature.push_back(takesThen ? '1' : '0');
          taken.push_back(takesThen);
        }
        auto &entry = groups[signature];
        entry.first.push_back(coord);
        entry.second = std::move(taken);
      }
    }

    // A single group means every coordinate follows the same control flow, so
    // cloning would not specialize anything.
    if (groups.size() <= 1) {
      return;
    }

    Builder builder(&getContext());

    // Stamp each core branch with its id (its index in `coreBranches`).
    for (size_t id = 0; id < coreBranches.size(); ++id) {
      coreBranches[id]->setAttr(
          kSpecializeBranchAttrName,
          builder.getI64IntegerAttr(static_cast<int64_t>(id)));
    }

    SmallVector<Attribute> groupAttrs;
    groupAttrs.reserve(groups.size());
    for (auto &entry : groups) {
      const SmallVector<LaunchNodeCoord> &coords = entry.second.first;
      const SmallVector<bool> &taken = entry.second.second;
      SmallVector<int64_t> coordsFlat;
      coordsFlat.reserve(coords.size() * 2);
      for (LaunchNodeCoord coord : coords) {
        coordsFlat.push_back(coord.x);
        coordsFlat.push_back(coord.y);
      }
      SmallVector<NamedAttribute> fields{
          builder.getNamedAttr("coords",
                               builder.getDenseI64ArrayAttr(coordsFlat)),
          builder.getNamedAttr("taken",
                               DenseBoolArrayAttr::get(&getContext(), taken))};
      groupAttrs.push_back(builder.getDictionaryAttr(fields));
    }
    func->setAttr(kSpecializePlanAttrName, builder.getArrayAttr(groupAttrs));
  }
};

} // namespace

} // namespace mlir::tt::ttl
