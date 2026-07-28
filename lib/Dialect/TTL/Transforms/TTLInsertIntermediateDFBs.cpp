// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Intermediate DFBs
//===----------------------------------------------------------------------===//
//
// Resolves DFB attachment and cross-block store fanout before
// convert-ttl-to-compute. Cloneable backward slices feeding mutually exclusive
// cross-block stores are relocated into those store blocks. Values whose
// consumers require DFB-attached inputs, or whose store fanout cannot be proven
// exclusive, are materialized through compiler-allocated intermediate dataflow
// buffers.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/ControlFlowUtils.h"
#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
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

enum class CrossRegionStoreAction {
  CloneBackwardSlice,
  MaterializeToDFB,
};

struct CrossRegionStorePlan {
  Value value;
  SmallVector<StoreOp> crossRegionStores;
  SmallVector<StoreOp> directStores;
  CrossRegionStoreAction action = CrossRegionStoreAction::MaterializeToDFB;
  FusionTraceResult backwardSlice;
};

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

// Returns the stores of `value` that live outside its defining block, but only
// when stores span at least two distinct blocks. That is the condition that can
// make convert-ttl-to-compute compare operation order across blocks.
static SmallVector<StoreOp> getCrossRegionStores(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return {};
  }

  Block *definingBlock = definingOp->getBlock();
  SmallVector<StoreOp> crossRegionStores;
  llvm::SmallPtrSet<Block *, 2> storeBlocks;
  for (StoreOp storeOp : getDirectStores(value)) {
    storeBlocks.insert(storeOp->getBlock());
    if (storeOp->getBlock() != definingBlock) {
      crossRegionStores.push_back(storeOp);
    }
  }
  if (storeBlocks.size() < 2) {
    return {};
  }
  return crossRegionStores;
}

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
  for (StoreBlockGroup &group : groupStoresByBlock(stores)) {
    representatives.push_back(group.stores.front().getOperation());
  }
  return arePairwiseInsideMutuallyExclusiveRegions(representatives);
}

static bool sliceExternalUsesAreStores(Value value,
                                       const FusionTraceResult &backwardSlice,
                                       ArrayRef<StoreOp> stores) {
  llvm::SmallPtrSet<Operation *, 8> sliceOps;
  llvm::SmallPtrSet<Operation *, 8> storeOps;
  for (Operation *op : backwardSlice.opsInOrder) {
    sliceOps.insert(op);
  }
  for (StoreOp storeOp : stores) {
    storeOps.insert(storeOp.getOperation());
  }

  for (Operation *op : backwardSlice.opsInOrder) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (sliceOps.contains(user)) {
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

static bool getCloneableBackwardSlice(Value value, ArrayRef<StoreOp> stores,
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

  Operation *producerScope = value.getDefiningOp()->getParentOp();
  return llvm::none_of(stores, [&](StoreOp storeOp) {
    return hasLoopBetween(producerScope, storeOp);
  });
}

static CrossRegionStorePlan
buildCrossRegionStorePlan(Value value, SmallVector<StoreOp> stores) {
  CrossRegionStorePlan plan;
  plan.value = value;
  plan.crossRegionStores = std::move(stores);
  plan.directStores = getDirectStores(value);

  FusionTraceResult backwardSlice;
  if (getCloneableBackwardSlice(value, plan.crossRegionStores, backwardSlice)) {
    plan.action = CrossRegionStoreAction::CloneBackwardSlice;
    plan.backwardSlice = std::move(backwardSlice);
  }
  return plan;
}

static SmallVector<CrossRegionStorePlan, 4>
collectCrossRegionStorePlans(func::FuncOp funcOp) {
  SmallVector<CrossRegionStorePlan, 4> plans;
  funcOp.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      if (!isa<RankedTensorType>(result.getType()) || getAttachedCB(result)) {
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
  // Root inputs already dominate the stores. Rewriting store operands preserves
  // branch control. Root-input DFB slots stay live at the clone sites because
  // ttl-auto-sync runs after this pass and releases after the cloned uses.
  for (StoreBlockGroup &group : groupStoresByBlock(stores)) {
    Value replacement = cloneBackwardSliceForStoreBlock(value, backwardSlice,
                                                        group.stores, builder);
    for (StoreOp storeOp : group.stores) {
      storeOp->setOperand(0, replacement);
    }
  }
  eraseUnusedBackwardSliceOps(backwardSlice);
}

static bool materializedValueDominates(Value materialized,
                                       Operation *consumerOp,
                                       DominanceInfo &dominanceInfo) {
  Operation *defOp = materialized.getDefiningOp();
  return defOp && dominanceInfo.dominates(defOp, consumerOp);
}

struct ConsumerUse {
  Operation *consumer;
  unsigned operandIndex;
};

struct ResultMaterializationPlan {
  unsigned resultIndex;
  SmallVector<ConsumerUse> uses;
};

struct ComputeMaterializationPlan {
  ComputeOp producer;
  SmallVector<ResultMaterializationPlan> results;
};

struct StandaloneTensorMaterializationUse {
  Operation *consumer;
  unsigned operandIndex;
};

struct MaterializedOutput {
  unsigned sourceResultIndex;
  RankedTensorType tensorType;
  Value sourceDFB;
  BindCBOp bind;
  CBReserveOp reserve;
  AttachCBOp attach;
  unsigned storeCount = 0;
};

struct ComputeResult {
  ComputeOp producer;
  unsigned resultIndex;
};

static std::optional<ComputeResult> getComputeResult(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return std::nullopt;
  }

  auto producer = dyn_cast<ComputeOp>(result.getOwner());
  if (!producer) {
    return std::nullopt;
  }

  return ComputeResult{producer, result.getResultNumber()};
}

static ComputeMaterializationPlan &
getOrCreateComputePlan(SmallVectorImpl<ComputeMaterializationPlan> &plans,
                       ComputeOp producer) {
  for (ComputeMaterializationPlan &plan : plans) {
    if (plan.producer == producer) {
      return plan;
    }
  }
  plans.push_back({producer, {}});
  return plans.back();
}

static ResultMaterializationPlan &
getOrCreateResultPlan(ComputeMaterializationPlan &plan, unsigned resultIndex) {
  for (ResultMaterializationPlan &resultPlan : plan.results) {
    if (resultPlan.resultIndex == resultIndex) {
      return resultPlan;
    }
  }
  plan.results.push_back({resultIndex, {}});
  return plan.results.back();
}

static void addMaterializationUse(
    Value operand, Operation *consumer, unsigned operandIndex,
    SmallVectorImpl<ComputeMaterializationPlan> &computePlans,
    SmallVectorImpl<StandaloneTensorMaterializationUse> &standaloneTensorUses) {
  // Compute results are materialized by rebuilding the producer once, even when
  // several results or consumers require DFB-attached values.
  if (std::optional<ComputeResult> computeResult = getComputeResult(operand)) {
    ComputeMaterializationPlan &computePlan =
        getOrCreateComputePlan(computePlans, computeResult->producer);
    ResultMaterializationPlan &resultPlan =
        getOrCreateResultPlan(computePlan, computeResult->resultIndex);
    resultPlan.uses.push_back({consumer, operandIndex});
    return;
  }

  standaloneTensorUses.push_back({consumer, operandIndex});
}

static LogicalResult
verifyCompilerDFBEnabledForPlans(ArrayRef<CrossRegionStorePlan> plans,
                                 bool enableCompilerDFBs) {
  if (enableCompilerDFBs) {
    return success();
  }

  for (const CrossRegionStorePlan &plan : plans) {
    if (plan.action == CrossRegionStoreAction::CloneBackwardSlice) {
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

static LogicalResult applyCrossRegionStorePlans(
    ArrayRef<CrossRegionStorePlan> plans, bool enableCompilerDFBs,
    OpBuilder &builder,
    SmallVectorImpl<ComputeMaterializationPlan> &computePlans,
    SmallVectorImpl<StandaloneTensorMaterializationUse> &standaloneTensorUses) {
  if (failed(verifyCompilerDFBEnabledForPlans(plans, enableCompilerDFBs))) {
    return failure();
  }

  for (const CrossRegionStorePlan &plan : plans) {
    if (plan.action == CrossRegionStoreAction::CloneBackwardSlice) {
      cloneStores(plan.value, plan.crossRegionStores, plan.backwardSlice,
                  builder);
      continue;
    }

    assert(enableCompilerDFBs &&
           "DFB materialization plans require compiler DFBs");

    // All direct stores read the private DFB, including same-block stores.
    // Otherwise convert-ttl-to-compute can schedule the producer compute after
    // the wait that reads this materialized value.
    for (StoreOp storeOp : plan.directStores) {
      addMaterializationUse(plan.value, storeOp.getOperation(), 0, computePlans,
                            standaloneTensorUses);
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

// A tensor block argument stored from multiple blocks is cross-block store
// fanout with no producer slice to clone and no defining op to materialize.
static LogicalResult
diagnoseUnsupportedBlockArgStoreFanout(func::FuncOp funcOp) {
  WalkResult walk = funcOp.walk([](Block *block) {
    for (BlockArgument arg : block->getArguments()) {
      if (!isa<RankedTensorType>(arg.getType()) || getAttachedCB(arg) ||
          distinctStoreBlockCount(arg) < 2) {
        continue;
      }
      block->getParentOp()->emitOpError()
          << "carries a tensor block argument stored from multiple "
             "control-flow blocks, which is not supported; store the value to "
             "a user-declared DFB before the control-flow split";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

// Non-mutating: validates one result and captures its type and source DFB. Bind
// allocation is deferred to materializeComputePlan so a rejected plan leaves
// the IR unchanged.
static FailureOr<MaterializedOutput>
planMaterializedOutput(ComputeOp computeOp, unsigned resultIndex) {
  if (resultIndex >= computeOp.getNumOutputs()) {
    return computeOp.emitOpError("materialization requested for result ")
           << resultIndex << ", but compute has only "
           << computeOp.getNumOutputs() << " outputs";
  }

  auto tensorType =
      dyn_cast<RankedTensorType>(computeOp.getResult(resultIndex).getType());
  if (!tensorType || !tensorType.hasStaticShape()) {
    return computeOp.emitOpError("result ")
           << resultIndex
           << " must have a statically shaped ranked tensor type";
  }

  Value sourceDFB = getAttachedCB(computeOp.getOutputs()[resultIndex]);
  if (!sourceDFB) {
    return computeOp.emitOpError("output ")
           << resultIndex << " must be attached to a dataflow buffer";
  }

  MaterializedOutput output;
  output.sourceResultIndex = resultIndex;
  output.tensorType = tensorType;
  output.sourceDFB = sourceDFB;
  return output;
}

static LogicalResult cloneComputeBodyWithMaterializedStores(
    ComputeOp sourceCompute, ComputeOp rebuiltCompute,
    MutableArrayRef<MaterializedOutput> materializedOutputs,
    OpBuilder &builder) {
  Block &sourceBody = sourceCompute.getBody().front();
  Block *rebuiltBody = builder.createBlock(&rebuiltCompute.getBody());
  Location loc = sourceCompute.getLoc();

  IRMapping mapper;
  for (Value operand : llvm::concat<Value>(rebuiltCompute.getInputs(),
                                           rebuiltCompute.getOutputs())) {
    auto tensorType = cast<RankedTensorType>(operand.getType());
    rebuiltBody->addArgument(tensorType.getElementType(), loc);
  }
  for (BlockArgument sourceArgument : sourceBody.getArguments()) {
    mapper.map(sourceArgument,
               rebuiltBody->getArgument(sourceArgument.getArgNumber()));
  }

  builder.setInsertionPointToStart(rebuiltBody);
  for (Operation &bodyOp : sourceBody.without_terminator()) {
    Operation *clonedOp = builder.clone(bodyOp, mapper);
    auto clonedStore = dyn_cast<TileStoreOp>(clonedOp);
    if (!clonedStore) {
      continue;
    }

    Value storeDFB = getAttachedCB(clonedStore.getView());
    for (MaterializedOutput &output : materializedOutputs) {
      if (storeDFB != output.sourceDFB) {
        continue;
      }
      auto materializedStore = TileStoreOp::create(
          builder, clonedStore.getLoc(), clonedStore.getTile(),
          output.reserve.getResult(), clonedStore.getIndices(),
          clonedStore.getDstIndex());
      materializedStore->setAttrs(clonedStore->getAttrs());
      ++output.storeCount;
    }
  }

  YieldOp::create(builder, sourceBody.getTerminator()->getLoc());

  for (MaterializedOutput &output : materializedOutputs) {
    assert(output.storeCount > 0 &&
           "verified compute output must have a tile_store");
  }
  return success();
}

static LogicalResult materializeComputePlan(ComputeMaterializationPlan &plan,
                                            OpBuilder &builder) {
  llvm::sort(plan.results, [](const ResultMaterializationPlan &lhs,
                              const ResultMaterializationPlan &rhs) {
    return lhs.resultIndex < rhs.resultIndex;
  });

  ComputeOp producerCompute = plan.producer;

  // Validate every result before allocating any DFB, so a rejected plan leaves
  // the IR unmutated.
  SmallVector<MaterializedOutput> materializedOutputs;
  materializedOutputs.reserve(plan.results.size());
  for (ResultMaterializationPlan &resultPlan : plan.results) {
    FailureOr<MaterializedOutput> output =
        planMaterializedOutput(producerCompute, resultPlan.resultIndex);
    if (failed(output)) {
      return failure();
    }
    materializedOutputs.push_back(*output);
  }

  auto funcOp = producerCompute->getParentOfType<func::FuncOp>();
  assert(funcOp && "ttl.compute must be inside a func::FuncOp");
  {
    OpBuilder::InsertionGuard guard(builder);
    for (MaterializedOutput &output : materializedOutputs) {
      output.bind = createCompilerAllocatedDFB(
          output.tensorType, producerCompute.getLoc(), funcOp, builder);
    }
  }

  SmallVector<Type> resultTypes(producerCompute.getResultTypes().begin(),
                                producerCompute.getResultTypes().end());
  SmallVector<Value> outputs(producerCompute.getOutputs().begin(),
                             producerCompute.getOutputs().end());
  SmallVector<Attribute> indexingMaps(producerCompute.getIndexingMaps().begin(),
                                      producerCompute.getIndexingMaps().end());

  builder.setInsertionPoint(producerCompute);
  for (MaterializedOutput &output : materializedOutputs) {
    output.reserve =
        CBReserveOp::create(builder, producerCompute.getLoc(),
                            output.tensorType, output.bind.getResult());
    Value init = tensor::EmptyOp::create(builder, producerCompute.getLoc(),
                                         output.tensorType.getShape(),
                                         output.tensorType.getElementType());
    Value initAttached =
        AttachCBOp::create(builder, producerCompute.getLoc(), output.tensorType,
                           init, output.bind.getResult());

    resultTypes.push_back(output.tensorType);
    outputs.push_back(initAttached);
    indexingMaps.push_back(
        producerCompute.getIndexingMaps()[producerCompute.getNumInputs() +
                                          output.sourceResultIndex]);
  }

  // Extra DFB outputs change the compute result list, output operands,
  // indexing maps, and tile block arguments; rebuild them as one consistent op.
  auto rebuiltCompute =
      ComputeOp::create(builder, producerCompute.getLoc(),
                        TypeRange(resultTypes), producerCompute.getInputs(),
                        ValueRange(outputs), builder.getArrayAttr(indexingMaps),
                        producerCompute.getIteratorTypesAttr());

  if (failed(cloneComputeBodyWithMaterializedStores(
          producerCompute, rebuiltCompute, materializedOutputs, builder))) {
    return failure();
  }

  SmallVector<Value> originalReplacements;
  originalReplacements.reserve(producerCompute.getNumResults());
  for (unsigned resultIndex = 0; resultIndex < producerCompute.getNumResults();
       ++resultIndex) {
    originalReplacements.push_back(rebuiltCompute.getResult(resultIndex));
  }
  producerCompute->replaceAllUsesWith(originalReplacements);
  producerCompute->erase();

  // Emit each DFB's push and wait/attach in the compute's own block, right
  // after the rebuilt compute, so the push stays unconditional and paired with
  // its acquire. Placing them at the consumer could leave an unconditional push
  // without a matching pop for branch-local consumers.
  // TODO(#724): relax once trace-balance analysis can prove balanced DFB
  // occupancy across structured control flow.
  Operation *insertAfter = rebuiltCompute;
  for (MaterializedOutput &output : materializedOutputs) {
    builder.setInsertionPointAfter(insertAfter);
    auto push = CBPushOp::create(builder, rebuiltCompute.getLoc(),
                                 output.bind.getResult(), IntegerAttr());
    insertAfter = push;

    builder.setInsertionPointAfter(insertAfter);
    output.attach =
        createDFBWaitAndAttach(output.bind.getResult(), output.tensorType,
                               rebuiltCompute.getLoc(), builder);
    insertAfter = output.attach;
  }

  for (auto [resultPlan, output] :
       llvm::zip_equal(plan.results, materializedOutputs)) {
    for (ConsumerUse use : resultPlan.uses) {
      use.consumer->setOperand(use.operandIndex, output.attach.getResult());
    }
  }

  return success();
}

static LogicalResult materializeStandaloneTensorUses(
    ArrayRef<StandaloneTensorMaterializationUse> standaloneTensorUses,
    func::FuncOp funcOp, OpBuilder &builder, DominanceInfo &dominanceInfo) {
  // A shared consumer-side acquire is valid only when its attach dominates
  // the next consumer. Incomparable control-flow regions need separate
  // compiler DFB outputs so each dynamic execution consumes exactly one
  // pushed slot. TODO(#724): Relax this with an explicit DFB occupancy
  // dataflow proof.
  llvm::DenseMap<Value, SmallVector<Value>> materialized;

  for (StandaloneTensorMaterializationUse use : standaloneTensorUses) {
    Operation *op = use.consumer;
    Value operand = op->getOperand(use.operandIndex);

    if (getAttachedCB(operand)) {
      continue;
    }

    // Reuse an existing attached value only when it is valid SSA for this
    // consumer. Branch-incomparable consumers need separate materializations.
    auto existingMaterializations = materialized.find(operand);
    if (existingMaterializations != materialized.end()) {
      SmallVector<Value> &candidateReplacements =
          existingMaterializations->second;
      auto dominatingReplacement =
          llvm::find_if(candidateReplacements, [&](Value candidateReplacement) {
            return materializedValueDominates(candidateReplacement, op,
                                              dominanceInfo);
          });
      if (dominatingReplacement != candidateReplacements.end()) {
        op->setOperand(use.operandIndex, *dominatingReplacement);
        continue;
      }
    }

    // No existing materialization dominates this consumer.
    FailureOr<DFBMaterializedValue> materialization =
        materializeToDFB(operand, funcOp, builder);
    if (failed(materialization)) {
      return failure();
    }

    op->setOperand(use.operandIndex, materialization->materialized);
    materialized[materialization->source].push_back(
        materialization->materialized);
  }
  return success();
}

struct TTLInsertIntermediateDFBsPass
    : public impl::TTLInsertIntermediateDFBsBase<
          TTLInsertIntermediateDFBsPass> {
  using TTLInsertIntermediateDFBsBase::TTLInsertIntermediateDFBsBase;

  void runOnOperation() override {
    auto funcOp = getOperation();

    if (failed(diagnoseUnsupportedBlockArgStoreFanout(funcOp))) {
      signalPassFailure();
      return;
    }

    SmallVector<DFBInputOpInterface> candidates;
    funcOp.walk([&](DFBInputOpInterface op) { candidates.push_back(op); });

    // Snapshot all cross-region store decisions before rewriting. Clone
    // rewrites can erase the original backward slice after redirecting stores.
    SmallVector<CrossRegionStorePlan, 4> crossRegionStorePlans =
        collectCrossRegionStorePlans(funcOp);

    OpBuilder builder(funcOp.getContext());
    SmallVector<ComputeMaterializationPlan> computePlans;
    SmallVector<StandaloneTensorMaterializationUse> standaloneTensorUses;

    if (failed(applyCrossRegionStorePlans(crossRegionStorePlans, enable,
                                          builder, computePlans,
                                          standaloneTensorUses))) {
      signalPassFailure();
      return;
    }

    // When compiler DFBs are disabled, verify that no operations require them.
    // Cloneable store fanout has already been rewritten because it allocates no
    // compiler DFB.
    if (!enable) {
      if (failed(verifyCompilerDFBInputs(candidates))) {
        signalPassFailure();
      }
      return;
    }

    // Elementwise values that depend on a released producer DFB must be stored
    // before the pop, because later consumers cannot legally reread that DFB
    // slot.
    funcOp.walk([&](Operation *op) {
      if (!isElementwiseOp(op)) {
        return;
      }
      for (OpOperand &operand : op->getOpOperands()) {
        Value value = operand.get();
        if (getAttachedCB(value)) {
          continue;
        }
        if (fusableValueCrossesDFBRelease(value, op)) {
          standaloneTensorUses.push_back({op, operand.getOperandNumber()});
        }
      }
    });

    for (DFBInputOpInterface dfbInputOp : candidates) {
      Operation *op = dfbInputOp.getOperation();
      auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

      for (unsigned idx : requiredIndices) {
        Value operand = op->getOperand(idx);

        if (getAttachedCB(operand)) {
          continue;
        }

        addMaterializationUse(operand, op, idx, computePlans,
                              standaloneTensorUses);
      }
    }

    for (ComputeMaterializationPlan &computePlan : computePlans) {
      if (failed(materializeComputePlan(computePlan, builder))) {
        signalPassFailure();
        return;
      }
    }

    DominanceInfo dominanceInfo(funcOp);
    if (failed(materializeStandaloneTensorUses(standaloneTensorUses, funcOp,
                                               builder, dominanceInfo))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
