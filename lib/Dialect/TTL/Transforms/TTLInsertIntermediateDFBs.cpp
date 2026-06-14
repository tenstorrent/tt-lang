// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Intermediate DFBs
//===----------------------------------------------------------------------===//
//
// Resolves DFB attachment and cross-block store fanout before
// convert-ttl-to-compute. Cheap backward slices stored from mutually exclusive
// control-flow regions are rematerialized in those regions. Values whose
// consumers require DFB-attached inputs, or whose control-flow stores cannot be
// safely rematerialized, are materialized through compiler-allocated
// intermediate dataflow buffers.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-intermediate-dfbs"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTINTERMEDIATEDFBS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct StoreBlockGroup {
  Block *block = nullptr;
  SmallVector<StoreOp> stores;
};

enum class CrossRegionStoreAction {
  Rematerialize,
  MaterializeToDFB,
};

struct CrossRegionStorePlan {
  Value value;
  SmallVector<StoreOp> stores;
  CrossRegionStoreAction action = CrossRegionStoreAction::MaterializeToDFB;
  FusionTraceResult backwardSlice;
};

static SmallVector<StoreOp> getCrossRegionStores(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return {};
  }

  Block *definingBlock = definingOp->getBlock();
  SmallVector<StoreOp> stores;
  for (OpOperand &use : value.getUses()) {
    auto storeOp = dyn_cast<StoreOp>(use.getOwner());
    if (!storeOp || storeOp.getTensor() != value) {
      continue;
    }
    if (storeOp->getBlock() != definingBlock) {
      stores.push_back(storeOp);
    }
  }
  return stores;
}

static bool hasStoresInMultipleBlocks(Value value) {
  llvm::SmallPtrSet<Block *, 2> storeBlocks;
  for (OpOperand &use : value.getUses()) {
    auto storeOp = dyn_cast<StoreOp>(use.getOwner());
    if (!storeOp || storeOp.getTensor() != value) {
      continue;
    }
    storeBlocks.insert(storeOp->getBlock());
    if (storeBlocks.size() > 1) {
      return true;
    }
  }
  return false;
}

static bool isCheapRematerializableOp(Operation *op) {
  return isa<FillOp>(op) || isElementwiseOp(op);
}

static bool hasLoopBetween(Operation *ancestor, Operation *descendant) {
  for (Operation *parent = descendant->getParentOp();
       parent && parent != ancestor; parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::ForallOp, scf::ParallelOp, scf::WhileOp>(parent)) {
      return true;
    }
  }
  return false;
}

static bool containsAllStores(Operation *ancestor, ArrayRef<StoreOp> stores) {
  return llvm::all_of(stores, [&](StoreOp storeOp) {
    return ancestor->isProperAncestor(storeOp);
  });
}

static scf::IfOp findExclusiveStoreIf(Value value, ArrayRef<StoreOp> stores) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return nullptr;
  }
  Block *definingBlock = definingOp->getBlock();

  for (StoreOp storeOp : stores) {
    for (Operation *parent = storeOp->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto ifOp = dyn_cast<scf::IfOp>(parent);
      if (!ifOp || ifOp->getBlock() != definingBlock) {
        continue;
      }
      if (!containsAllStores(ifOp, stores)) {
        continue;
      }
      if (llvm::any_of(stores, [&](StoreOp candidateStore) {
            return hasLoopBetween(ifOp, candidateStore);
          })) {
        continue;
      }
      return ifOp;
    }
  }
  return nullptr;
}

static bool getRematerializableBackwardSlice(Value value,
                                             ArrayRef<StoreOp> stores,
                                             FusionTraceResult &backwardSlice) {
  if (!findExclusiveStoreIf(value, stores)) {
    return false;
  }

  backwardSlice = traceFusionToRoots(value);
  if (backwardSlice.failureReason != TraceFailureReason::Success ||
      backwardSlice.opsInOrder.empty()) {
    return false;
  }

  return llvm::all_of(backwardSlice.opsInOrder, isCheapRematerializableOp);
}

static CrossRegionStorePlan
buildCrossRegionStorePlan(Value value, SmallVector<StoreOp> stores) {
  CrossRegionStorePlan plan;
  plan.value = value;
  plan.stores = std::move(stores);

  FusionTraceResult backwardSlice;
  if (getRematerializableBackwardSlice(value, plan.stores, backwardSlice)) {
    plan.action = CrossRegionStoreAction::Rematerialize;
    plan.backwardSlice = std::move(backwardSlice);
  }
  return plan;
}

static SmallVector<CrossRegionStorePlan>
collectCrossRegionStorePlans(func::FuncOp funcOp) {
  SmallVector<CrossRegionStorePlan> plans;
  funcOp.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      if (!isa<RankedTensorType>(result.getType()) || getAttachedCB(result)) {
        continue;
      }
      if (!hasStoresInMultipleBlocks(result)) {
        continue;
      }
      SmallVector<StoreOp> stores = getCrossRegionStores(result);
      if (stores.empty()) {
        continue;
      }
      plans.push_back(buildCrossRegionStorePlan(result, std::move(stores)));
    }
  });
  return plans;
}

static SmallVector<StoreBlockGroup>
groupStoresByBlock(ArrayRef<StoreOp> stores) {
  SmallVector<StoreBlockGroup> groups;
  for (StoreOp storeOp : stores) {
    Block *block = storeOp->getBlock();
    auto iter = llvm::find_if(groups, [&](const StoreBlockGroup &group) {
      return group.block == block;
    });
    if (iter == groups.end()) {
      groups.push_back(StoreBlockGroup{block, {}});
      iter = std::prev(groups.end());
    }
    iter->stores.push_back(storeOp);
  }
  return groups;
}

static Value
cloneBackwardSliceForStoreBlock(Value value,
                                const FusionTraceResult &backwardSlice,
                                ArrayRef<StoreOp> stores, OpBuilder &builder) {
  assert(!stores.empty() && "cloneBackwardSliceForStoreBlock requires stores");
  OpBuilder::InsertionGuard guard(builder);

  StoreOp firstStore = stores.front();
  for (StoreOp storeOp : stores.drop_front()) {
    assert(storeOp->getBlock() == firstStore->getBlock() &&
           "stores must be grouped by block");
    if (storeOp->isBeforeInBlock(firstStore)) {
      firstStore = storeOp;
    }
  }
  builder.setInsertionPoint(firstStore);

  IRMapping mapping;
  for (Value rootInput : backwardSlice.rootInputs) {
    mapping.map(rootInput, rootInput);
  }

  for (Operation *op : backwardSlice.opsInOrder) {
    builder.clone(*op, mapping);
  }
  return mapping.lookup(value);
}

static void
eraseUnusedBackwardSliceOps(const FusionTraceResult &backwardSlice) {
  for (Operation *op : llvm::reverse(backwardSlice.opsInOrder)) {
    if (op->use_empty()) {
      op->erase();
    }
  }
}

static void rematerializeStores(Value value, ArrayRef<StoreOp> stores,
                                const FusionTraceResult &backwardSlice,
                                OpBuilder &builder) {
  // The backward slice contains only pure cheap ops, and the root inputs
  // already dominate the stores. Rewriting store operands preserves branch
  // control.
  for (StoreBlockGroup &group : groupStoresByBlock(stores)) {
    Value replacement = cloneBackwardSliceForStoreBlock(value, backwardSlice,
                                                        group.stores, builder);
    for (StoreOp storeOp : group.stores) {
      storeOp->setOperand(0, replacement);
    }
  }
  eraseUnusedBackwardSliceOps(backwardSlice);
}

static Value getOrCreateMaterializedDFB(Value value, ModuleOp moduleOp,
                                        OpBuilder &builder,
                                        llvm::DenseMap<Value, Value> &cache);

static LogicalResult
verifyCompilerDFBEnabledForPlans(ArrayRef<CrossRegionStorePlan> plans,
                                 bool enableCompilerDFBs) {
  if (enableCompilerDFBs) {
    return success();
  }

  for (const CrossRegionStorePlan &plan : plans) {
    if (plan.action == CrossRegionStoreAction::Rematerialize) {
      continue;
    }

    Operation *definingOp = plan.value.getDefiningOp();
    definingOp->emitOpError()
        << "result is stored from a different block and cannot be "
           "rematerialized without changing the producer placement; enable "
           "compiler DFBs or store the intermediate to a user-declared DFB "
           "before the control-flow split";
    return failure();
  }
  return success();
}

static LogicalResult applyCrossRegionStorePlans(
    ArrayRef<CrossRegionStorePlan> plans, ModuleOp moduleOp, OpBuilder &builder,
    llvm::DenseMap<Value, Value> &materialized, bool enableCompilerDFBs) {
  if (failed(verifyCompilerDFBEnabledForPlans(plans, enableCompilerDFBs))) {
    return failure();
  }

  for (const CrossRegionStorePlan &plan : plans) {
    if (plan.action == CrossRegionStoreAction::Rematerialize) {
      rematerializeStores(plan.value, plan.stores, plan.backwardSlice, builder);
      continue;
    }

    assert(enableCompilerDFBs &&
           "DFB materialization plans require compiler DFBs");

    // This follows the storage-boundary model used by upstream MLIR
    // bufferization (`mlir/docs/Bufferization.md` and SCF's
    // BufferizableOpInterfaceImpl): tensor SSA values crossing control-flow
    // regions get concrete storage, and branch-local uses read that storage.
    Value replacement =
        getOrCreateMaterializedDFB(plan.value, moduleOp, builder, materialized);
    // Redirect only the cross-block stores to the DFB read-back; same-block and
    // non-store users keep the original SSA value.
    for (StoreOp storeOp : plan.stores) {
      storeOp->setOperand(0, replacement);
    }
  }
  return success();
}

static LogicalResult
verifyCompilerDFBInputs(ArrayRef<DFBInputOpInterface> candidates) {
  for (DFBInputOpInterface dfbInputOp : candidates) {
    Operation *op = dfbInputOp.getOperation();
    auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

    for (unsigned operandIndex : requiredIndices) {
      Value operand = op->getOperand(operandIndex);
      if (getAttachedCB(operand)) {
        continue;
      }
      op->emitOpError("operand #")
          << operandIndex
          << " requires a DFB-attached value but compiler-allocated DFBs "
             "are disabled (--no-ttl-compiler-dfbs); either enable compiler "
             "DFBs or store the intermediate to a user-declared DFB before "
             "this operation";
      return failure();
    }
  }
  return success();
}

static Value getOrCreateMaterializedDFB(Value value, ModuleOp moduleOp,
                                        OpBuilder &builder,
                                        llvm::DenseMap<Value, Value> &cache) {
  if (auto iter = cache.find(value); iter != cache.end()) {
    return iter->second;
  }

  OpBuilder::InsertionGuard guard(builder);
  Value replacement = materializeToDFB(value, moduleOp, builder);
  cache[value] = replacement;
  return replacement;
}

struct TTLInsertIntermediateDFBsPass
    : public impl::TTLInsertIntermediateDFBsBase<
          TTLInsertIntermediateDFBsPass> {
  using TTLInsertIntermediateDFBsBase::TTLInsertIntermediateDFBsBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return;
    }

    SmallVector<DFBInputOpInterface> candidates;
    funcOp.walk([&](DFBInputOpInterface op) { candidates.push_back(op); });
    // Snapshot all cross-region store decisions before rewriting. A
    // rematerialization can erase the original backward slice and rewrite its
    // uses.
    SmallVector<CrossRegionStorePlan> crossRegionStorePlans =
        collectCrossRegionStorePlans(funcOp);

    OpBuilder builder(funcOp.getContext());
    llvm::DenseMap<Value, Value> materialized;

    if (failed(applyCrossRegionStorePlans(crossRegionStorePlans, moduleOp,
                                          builder, materialized, enable))) {
      signalPassFailure();
      return;
    }

    // When compiler DFBs are disabled, verify that no operations require
    // them and emit an actionable error if any do.
    if (!enable) {
      if (failed(verifyCompilerDFBInputs(candidates))) {
        signalPassFailure();
      }
      return;
    }

    for (DFBInputOpInterface dfbInputOp : candidates) {
      Operation *op = dfbInputOp.getOperation();
      auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

      for (unsigned operandIndex : requiredIndices) {
        Value operand = op->getOperand(operandIndex);

        if (getAttachedCB(operand)) {
          continue;
        }

        Value replacement = getOrCreateMaterializedDFB(operand, moduleOp,
                                                       builder, materialized);

        // Replace only this specific operand. Elementwise consumers of
        // the same value retain the original SSA value and fuse with
        // the producer in a single compute block.
        op->setOperand(operandIndex, replacement);
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
