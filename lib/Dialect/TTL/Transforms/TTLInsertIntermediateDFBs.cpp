// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Intermediate DFBs
//===----------------------------------------------------------------------===//
//
// Resolves DFB attachment and values stored from multiple blocks before
// convert-ttl-to-compute. Cloneable backward slices feeding mutually exclusive
// stores from multiple blocks are relocated into those store blocks. Values
// whose consumers require DFB-attached inputs, or whose stores cannot be proven
// mutually exclusive, are materialized through compiler-allocated intermediate
// dataflow buffers.
//
//===----------------------------------------------------------------------===//

#include "DFBAcquireReleaseAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

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

enum class MultiBlockStoreAction {
  CloneBackwardSlice,
  MaterializeToDFB,
};

struct MultiBlockStorePlan {
  Value value;
  SmallVector<StoreOp> storesOutsideDefiningBlock;
  SmallVector<StoreOp> directStores;
  MultiBlockStoreAction action = MultiBlockStoreAction::MaterializeToDFB;
  FusionTraceResult backwardSlice;
};

struct DFBLifecycleOps {
  SmallVector<Operation *> reserves;
  SmallVector<Operation *> waits;
  SmallVector<Operation *> pushes;
  SmallVector<Operation *> pops;
};

static DFBLifecycleOps collectDFBLifecycleOps(func::FuncOp funcOp) {
  DFBLifecycleOps lifecycle;
  collectDFBAcquireReleaseOps(funcOp, lifecycle.reserves, lifecycle.waits,
                              lifecycle.pushes, lifecycle.pops);
  return lifecycle;
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

static SmallVector<StoreOp> getDirectStores(Value value) {
  SmallVector<StoreOp> stores;
  for (OpOperand &use : value.getUses()) {
    auto storeOp = dyn_cast<StoreOp>(use.getOwner());
    if (storeOp && storeOp.getTensor() == value) {
      stores.push_back(storeOp);
    }
  }
  return stores;
}

// Returns `value`'s stores that live outside its defining block, but only when
// its stores span at least two distinct blocks -- the condition under which
// convert-ttl-to-compute orders stores across blocks and asserts. Returns {}
// for single-block store sets, which lower without help.
static SmallVector<StoreOp> getStoresOutsideDefiningBlock(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return {};
  }

  Block *definingBlock = definingOp->getBlock();
  SmallVector<StoreOp> storesOutsideDefiningBlock;
  llvm::SmallPtrSet<Block *, 2> storeBlocks;
  for (StoreOp storeOp : getDirectStores(value)) {
    storeBlocks.insert(storeOp->getBlock());
    if (storeOp->getBlock() != definingBlock) {
      storesOutsideDefiningBlock.push_back(storeOp);
    }
  }
  if (storeBlocks.size() < 2) {
    return {};
  }
  return storesOutsideDefiningBlock;
}

// Distinct blocks that contain a direct `ttl.store` of `value`.
static unsigned distinctStoreBlockCount(Value value) {
  llvm::SmallPtrSet<Block *, 2> storeBlocks;
  for (StoreOp storeOp : getDirectStores(value)) {
    storeBlocks.insert(storeOp->getBlock());
  }
  return storeBlocks.size();
}

static bool hasLoopBetween(Operation *ancestor, Operation *descendant) {
  for (Operation *parent = descendant->getParentOp();
       parent && parent != ancestor; parent = parent->getParentOp()) {
    if (isa<LoopLikeOpInterface>(parent)) {
      return true;
    }
  }
  return false;
}

static bool areStoreBlocksPairwiseExclusive(ArrayRef<StoreOp> stores) {
  SmallVector<Operation *> representatives;
  for (StoreBlockGroup group : groupStoresByBlock(stores)) {
    representatives.push_back(group.stores.front().getOperation());
  }
  // TODO(#685): Add predicate-based proof for analyzable sibling `scf.if`
  // chains that upstream structural region analysis cannot prove.
  for (unsigned lhsIndex = 0; lhsIndex < representatives.size(); ++lhsIndex) {
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < representatives.size();
         ++rhsIndex) {
      if (!mlir::insideMutuallyExclusiveRegions(representatives[lhsIndex],
                                                representatives[rhsIndex])) {
        return false;
      }
    }
  }
  return true;
}

static bool sliceExternalUsesAreStores(Value value,
                                       const FusionTraceResult &backwardSlice,
                                       ArrayRef<StoreOp> stores) {
  llvm::SmallPtrSet<Operation *, 8> storeOps;
  for (StoreOp storeOp : stores) {
    storeOps.insert(storeOp.getOperation());
  }

  for (Operation *op : backwardSlice.opsInOrder) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (backwardSlice.opsInOrder.contains(user)) {
          continue;
        }
        if (result == value && storeOps.contains(user)) {
          continue;
        }
        return false;
      }
    }
  }
  return true;
}

static SmallVector<StoreOp> getEarliestStorePerBlock(ArrayRef<StoreOp> stores) {
  SmallVector<StoreOp> earliestStores;
  for (StoreBlockGroup group : groupStoresByBlock(stores)) {
    StoreOp earliestStore = group.stores.front();
    for (StoreOp storeOp : ArrayRef<StoreOp>(group.stores).drop_front()) {
      if (storeOp->isBeforeInBlock(earliestStore)) {
        earliestStore = storeOp;
      }
    }
    earliestStores.push_back(earliestStore);
  }
  return earliestStores;
}

static ArrayRef<Operation *>
getSameKindAcquires(Operation *acquire, const DFBLifecycleOps &lifecycle) {
  if (isa<CBReserveOp>(acquire)) {
    return lifecycle.reserves;
  }
  if (isa<CBWaitOp>(acquire)) {
    return lifecycle.waits;
  }
  return {};
}

static ArrayRef<Operation *>
getSameKindReleases(Operation *acquire, const DFBLifecycleOps &lifecycle) {
  if (isa<CBReserveOp>(acquire)) {
    return lifecycle.pushes;
  }
  if (isa<CBWaitOp>(acquire)) {
    return lifecycle.pops;
  }
  return {};
}

static Operation *projectToBlock(Operation *op, Block *block) {
  return op->getBlock() == block ? op : block->findAncestorOpInBlock(*op);
}

static bool releaseCanExecuteBeforeStore(Operation *release, StoreOp storeOp,
                                         Block *orderingBlock) {
  Operation *store = storeOp.getOperation();
  if (release->getBlock() == store->getBlock()) {
    return release->isBeforeInBlock(store);
  }

  Operation *projectedRelease = projectToBlock(release, orderingBlock);
  Operation *projectedStore = projectToBlock(store, orderingBlock);
  if (!projectedRelease || !projectedStore) {
    return true;
  }
  if (projectedRelease == projectedStore) {
    return true;
  }
  return projectedRelease->isBeforeInBlock(projectedStore);
}

static bool
rootInputReleaseCanExecuteBeforeStore(Value rootInput, ArrayRef<StoreOp> stores,
                                      const DFBLifecycleOps &lifecycle) {
  Operation *acquire = findCBAcquireOp(rootInput);
  if (!acquire) {
    return true;
  }

  ArrayRef<Operation *> acquires = getSameKindAcquires(acquire, lifecycle);
  ArrayRef<Operation *> releases = getSameKindReleases(acquire, lifecycle);
  if (acquires.empty()) {
    return true;
  }
  if (releases.empty()) {
    return false;
  }

  DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);
  DFBReleaseSearch releaseSearch =
      findOwnedDFBReleases(interval, /*lastOwnedUse=*/nullptr, releases);

  Block *orderingBlock = acquire->getBlock();
  auto releaseCanExecuteBeforeAnyStore = [&](Operation *release) {
    return llvm::any_of(stores, [&](StoreOp storeOp) {
      return releaseCanExecuteBeforeStore(release, storeOp, orderingBlock);
    });
  };
  if (llvm::any_of(releaseSearch.sameLevelReleases,
                   releaseCanExecuteBeforeAnyStore)) {
    return true;
  }
  return llvm::any_of(releaseSearch.nestedReleases,
                      releaseCanExecuteBeforeAnyStore);
}

static bool rootInputsLiveAtStoreSites(const FusionTraceResult &backwardSlice,
                                       ArrayRef<StoreOp> stores,
                                       const DFBLifecycleOps &lifecycle) {
  SmallVector<StoreOp> cloneSites = getEarliestStorePerBlock(stores);
  return llvm::none_of(backwardSlice.rootInputs, [&](Value rootInput) {
    return rootInputReleaseCanExecuteBeforeStore(rootInput, cloneSites,
                                                 lifecycle);
  });
}

// Cloning is selected only when the original producer slice is completely
// relocated into mutually exclusive store blocks. Otherwise materialization
// preserves single producer execution without depending on predicate analysis.
static bool getCloneableBackwardSlice(Value value, ArrayRef<StoreOp> stores,
                                      const DFBLifecycleOps &lifecycle,
                                      FusionTraceResult &backwardSlice) {
  if (!areStoreBlocksPairwiseExclusive(stores)) {
    return false;
  }

  backwardSlice = traceFusionToRoots(value);
  if (backwardSlice.failureReason != TraceFailureReason::Success ||
      backwardSlice.opsInOrder.empty()) {
    return false;
  }
  // TODO(#686): Add explicit exclusions here if a producer recognized by
  // `traceFusionToRoots` has a concrete clone-safety issue.
  if (!sliceExternalUsesAreStores(value, backwardSlice, stores)) {
    return false;
  }
  if (!rootInputsLiveAtStoreSites(backwardSlice, stores, lifecycle)) {
    return false;
  }

  Operation *producerScope = value.getDefiningOp()->getParentOp();
  return llvm::none_of(stores, [&](StoreOp storeOp) {
    return hasLoopBetween(producerScope, storeOp);
  });
}

static MultiBlockStorePlan
buildMultiBlockStorePlan(Value value, SmallVector<StoreOp> stores,
                         const DFBLifecycleOps &lifecycle) {
  MultiBlockStorePlan plan;
  plan.value = value;
  plan.storesOutsideDefiningBlock = std::move(stores);
  plan.directStores = getDirectStores(value);

  FusionTraceResult backwardSlice;
  if (getCloneableBackwardSlice(value, plan.storesOutsideDefiningBlock,
                                lifecycle, backwardSlice)) {
    plan.action = MultiBlockStoreAction::CloneBackwardSlice;
    plan.backwardSlice = std::move(backwardSlice);
  }
  return plan;
}

static SmallVector<MultiBlockStorePlan, 4>
collectMultiBlockStorePlans(func::FuncOp funcOp) {
  SmallVector<MultiBlockStorePlan, 4> plans;
  DFBLifecycleOps lifecycle = collectDFBLifecycleOps(funcOp);
  funcOp.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      if (!isa<RankedTensorType>(result.getType()) || getAttachedCB(result)) {
        continue;
      }
      SmallVector<StoreOp> stores = getStoresOutsideDefiningBlock(result);
      if (stores.empty()) {
        continue;
      }
      plans.push_back(
          buildMultiBlockStorePlan(result, std::move(stores), lifecycle));
    }
  });
  return plans;
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

static void cloneStores(Value value, ArrayRef<StoreOp> stores,
                        const FusionTraceResult &backwardSlice,
                        OpBuilder &builder) {
  // Plans use cloning only when the root DFB slots are not explicitly released
  // before the cloned use sites.
  for (StoreBlockGroup group : groupStoresByBlock(stores)) {
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
                                        llvm::DenseMap<Value, Value> &cache) {
  if (auto iter = cache.find(value); iter != cache.end()) {
    return iter->second;
  }

  OpBuilder::InsertionGuard guard(builder);
  Value replacement = materializeToDFB(value, moduleOp, builder);
  cache[value] = replacement;
  return replacement;
}

static LogicalResult
verifyCompilerDFBEnabledForPlans(ArrayRef<MultiBlockStorePlan> plans,
                                 bool enableCompilerDFBs) {
  if (enableCompilerDFBs) {
    return success();
  }

  for (const MultiBlockStorePlan &plan : plans) {
    if (plan.action == MultiBlockStoreAction::CloneBackwardSlice) {
      continue;
    }

    Operation *definingOp = plan.value.getDefiningOp();
    definingOp->emitOpError()
        << "result is stored from a different block and cannot be "
           "cloned into mutually exclusive store blocks; enable compiler DFBs "
           "or store the intermediate to a user-declared DFB before the "
           "control-flow split";
    return failure();
  }
  return success();
}

static LogicalResult applyMultiBlockStorePlans(
    ArrayRef<MultiBlockStorePlan> plans, ModuleOp moduleOp, OpBuilder &builder,
    llvm::DenseMap<Value, Value> &materialized, bool enableCompilerDFBs) {
  if (failed(verifyCompilerDFBEnabledForPlans(plans, enableCompilerDFBs))) {
    return failure();
  }

  for (const MultiBlockStorePlan &plan : plans) {
    if (plan.action == MultiBlockStoreAction::CloneBackwardSlice) {
      cloneStores(plan.value, plan.storesOutsideDefiningBlock,
                  plan.backwardSlice, builder);
      continue;
    }

    assert(enableCompilerDFBs &&
           "DFB materialization plans require compiler DFBs");

    // This follows the storage-boundary model used by upstream MLIR
    // bufferization (`mlir/docs/Bufferization.md` and SCF's
    // BufferizableOpInterfaceImpl): tensor SSA values crossing control-flow
    // regions get concrete storage, and store users read that storage.
    Value replacement =
        getOrCreateMaterializedDFB(plan.value, moduleOp, builder, materialized);
    // Redirect all snapshotted direct stores to the DFB read-back. Leaving a
    // defining-block store on the original value can make
    // convert-ttl-to-compute schedule the producer compute after the DFB wait
    // that reads this value.
    for (StoreOp storeOp : plan.directStores) {
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

// A tensor block argument -- today only an `scf.for` iter_arg -- stored from
// multiple blocks has no producer slice to clone and no defining op to
// materialize, so this pass cannot normalize it. The frontend does not yet emit
// loop-carried tensor recurrence (#540); emit an actionable error instead of
// leaving the stores for convert-ttl-to-compute to drop silently.
static LogicalResult diagnoseUnsupportedBlockArgStores(func::FuncOp funcOp) {
  Block *entryBlock = &funcOp.getBody().front();
  WalkResult walk = funcOp.walk([&](Block *block) {
    if (block == entryBlock) {
      return WalkResult::advance();
    }

    for (BlockArgument arg : block->getArguments()) {
      if (!isa<RankedTensorType>(arg.getType()) || getAttachedCB(arg) ||
          distinctStoreBlockCount(arg) < 2) {
        continue;
      }
      block->getParentOp()->emitOpError()
          << "carries a tensor block argument stored from multiple "
             "control-flow blocks, which is not supported; store the value to "
             "a "
             "user-declared DFB before the control-flow split";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
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

    if (failed(diagnoseUnsupportedBlockArgStores(funcOp))) {
      signalPassFailure();
      return;
    }

    // Snapshot all multi-block store decisions before rewriting. A clone
    // rewrite can erase the original backward slice and rewrite its uses;
    // eraseUnusedBackwardSliceOps only removes ops with no remaining users.
    SmallVector<MultiBlockStorePlan, 4> multiBlockStorePlans =
        collectMultiBlockStorePlans(funcOp);

    OpBuilder builder(funcOp.getContext());
    llvm::DenseMap<Value, Value> materialized;

    if (failed(applyMultiBlockStorePlans(multiBlockStorePlans, moduleOp,
                                         builder, materialized, enable))) {
      signalPassFailure();
      return;
    }

    SmallVector<DFBInputOpInterface> candidates;
    funcOp.walk([&](DFBInputOpInterface op) { candidates.push_back(op); });

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
