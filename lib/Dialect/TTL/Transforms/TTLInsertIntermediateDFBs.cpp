// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Intermediate DFBs
//===----------------------------------------------------------------------===//
//
// Inserts compiler-allocated intermediate dataflow buffers at fusion split
// points. Tensor-level ops whose tile-level lowerings require DFB inputs may
// receive operands from ttl.compute results that are not DFB-attached. This
// pass materializes those compute results as extra compute outputs so the
// remaining consumers can be converted after attachment.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-intermediate-dfbs"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTINTERMEDIATEDFBS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

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

struct TensorFallbackUse {
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
    ComputeOp oldCompute, ComputeOp newCompute,
    MutableArrayRef<MaterializedOutput> materializedOutputs,
    OpBuilder &builder) {
  Block &oldBody = oldCompute.getBody().front();
  Block *newBody = builder.createBlock(&newCompute.getBody());
  Location loc = oldCompute.getLoc();

  IRMapping mapper;
  for (Value operand :
       llvm::concat<Value>(newCompute.getInputs(), newCompute.getOutputs())) {
    auto tensorType = cast<RankedTensorType>(operand.getType());
    newBody->addArgument(tensorType.getElementType(), loc);
  }
  for (BlockArgument oldArgument : oldBody.getArguments()) {
    mapper.map(oldArgument, newBody->getArgument(oldArgument.getArgNumber()));
  }

  builder.setInsertionPointToStart(newBody);
  for (Operation &bodyOp : oldBody.without_terminator()) {
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

  YieldOp::create(builder, oldBody.getTerminator()->getLoc());

  for (MaterializedOutput &output : materializedOutputs) {
    assert(output.storeCount > 0 &&
           "verified compute output must have a tile_store");
  }
  return success();
}

static LogicalResult materializeComputePlan(ComputeMaterializationPlan &plan,
                                            ModuleOp moduleOp,
                                            OpBuilder &builder) {
  llvm::sort(plan.results, [](const ResultMaterializationPlan &lhs,
                              const ResultMaterializationPlan &rhs) {
    return lhs.resultIndex < rhs.resultIndex;
  });

  ComputeOp oldCompute = plan.producer;

  // Validate every result before allocating any DFB, so a rejected plan leaves
  // the IR unmutated.
  SmallVector<MaterializedOutput> materializedOutputs;
  materializedOutputs.reserve(plan.results.size());
  for (ResultMaterializationPlan &resultPlan : plan.results) {
    FailureOr<MaterializedOutput> output =
        planMaterializedOutput(oldCompute, resultPlan.resultIndex);
    if (failed(output)) {
      return failure();
    }
    materializedOutputs.push_back(*output);
  }

  auto funcOp = oldCompute->getParentOfType<func::FuncOp>();
  assert(funcOp && "ttl.compute must be inside a func::FuncOp");
  {
    OpBuilder::InsertionGuard guard(builder);
    for (MaterializedOutput &output : materializedOutputs) {
      output.bind = createCompilerAllocatedDFB(
          output.tensorType, oldCompute.getLoc(), funcOp, moduleOp, builder);
    }
  }

  SmallVector<Type> resultTypes(oldCompute.getResultTypes().begin(),
                                oldCompute.getResultTypes().end());
  SmallVector<Value> outputs(oldCompute.getOutputs().begin(),
                             oldCompute.getOutputs().end());
  SmallVector<Attribute> indexingMaps(oldCompute.getIndexingMaps().begin(),
                                      oldCompute.getIndexingMaps().end());

  builder.setInsertionPoint(oldCompute);
  for (MaterializedOutput &output : materializedOutputs) {
    output.reserve =
        CBReserveOp::create(builder, oldCompute.getLoc(), output.tensorType,
                            output.bind.getResult());
    Value init = tensor::EmptyOp::create(builder, oldCompute.getLoc(),
                                         output.tensorType.getShape(),
                                         output.tensorType.getElementType());
    Value initAttached =
        AttachCBOp::create(builder, oldCompute.getLoc(), output.tensorType,
                           init, output.bind.getResult());

    resultTypes.push_back(output.tensorType);
    outputs.push_back(initAttached);
    indexingMaps.push_back(
        oldCompute.getIndexingMaps()[oldCompute.getNumInputs() +
                                     output.sourceResultIndex]);
  }

  auto newCompute = ComputeOp::create(
      builder, oldCompute.getLoc(), TypeRange(resultTypes),
      oldCompute.getInputs(), ValueRange(outputs),
      builder.getArrayAttr(indexingMaps), oldCompute.getIteratorTypesAttr());

  if (failed(cloneComputeBodyWithMaterializedStores(
          oldCompute, newCompute, materializedOutputs, builder))) {
    return failure();
  }

  SmallVector<Value> originalReplacements;
  originalReplacements.reserve(oldCompute.getNumResults());
  for (unsigned resultIndex = 0; resultIndex < oldCompute.getNumResults();
       ++resultIndex) {
    originalReplacements.push_back(newCompute.getResult(resultIndex));
  }
  oldCompute->replaceAllUsesWith(originalReplacements);
  oldCompute->erase();

  // Emit each DFB's push and wait/attach in the compute's own block, right
  // after the rebuilt compute, so the push stays unconditional and paired with
  // its acquire. Placing them at the consumer could leave an unconditional push
  // without a matching pop for branch-local consumers.
  // TODO(#724): relax once trace-balance analysis can prove balanced DFB
  // occupancy across structured control flow.
  Operation *insertAfter = newCompute;
  for (MaterializedOutput &output : materializedOutputs) {
    builder.setInsertionPointAfter(insertAfter);
    auto push = CBPushOp::create(builder, newCompute.getLoc(),
                                 output.bind.getResult(), IntegerAttr());
    insertAfter = push;

    builder.setInsertionPointAfter(insertAfter);
    output.attach =
        createDFBWaitAndAttach(output.bind.getResult(), output.tensorType,
                               newCompute.getLoc(), builder);
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

static LogicalResult
materializeTensorFallbackUses(ArrayRef<TensorFallbackUse> fallbackUses,
                              ModuleOp moduleOp, OpBuilder &builder,
                              DominanceInfo &dominanceInfo) {
  // A shared consumer-side acquire is valid only when its attach dominates
  // the next consumer. Incomparable control-flow regions need separate
  // compiler DFB outputs so each dynamic execution consumes exactly one
  // pushed slot. TODO(#724): Relax this with an explicit DFB occupancy
  // dataflow proof.
  llvm::DenseMap<Value, SmallVector<Value>> materialized;

  for (TensorFallbackUse use : fallbackUses) {
    Operation *op = use.consumer;
    Value operand = op->getOperand(use.operandIndex);

    if (getAttachedCB(operand)) {
      continue;
    }

    if (auto iter = materialized.find(operand); iter != materialized.end()) {
      auto replacement = llvm::find_if(iter->second, [&](Value value) {
        return materializedValueDominates(value, op, dominanceInfo);
      });
      if (replacement != iter->second.end()) {
        op->setOperand(use.operandIndex, *replacement);
        continue;
      }
    }

    FailureOr<DFBMaterializedValue> materialization =
        materializeToDFB(operand, moduleOp, builder);
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
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return;
    }

    SmallVector<DFBInputOpInterface> candidates;
    funcOp.walk([&](DFBInputOpInterface op) { candidates.push_back(op); });

    // When compiler DFBs are disabled, verify that no operations require
    // them and emit an actionable error if any do.
    if (!enable) {
      for (DFBInputOpInterface dfbInputOp : candidates) {
        Operation *op = dfbInputOp.getOperation();
        auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

        for (unsigned idx : requiredIndices) {
          Value operand = op->getOperand(idx);
          if (getAttachedCB(operand)) {
            continue;
          }
          op->emitOpError("operand #")
              << idx
              << " requires a DFB-attached value but compiler-allocated DFBs "
                 "are disabled (--no-ttl-compiler-dfbs); either enable "
                 "compiler DFBs or store the intermediate to a user-declared "
                 "DFB before this operation";
          signalPassFailure();
          return;
        }
      }
      return;
    }

    OpBuilder builder(funcOp.getContext());
    SmallVector<ComputeMaterializationPlan> computePlans;
    SmallVector<TensorFallbackUse> fallbackUses;

    for (DFBInputOpInterface dfbInputOp : candidates) {
      Operation *op = dfbInputOp.getOperation();
      auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

      for (unsigned idx : requiredIndices) {
        Value operand = op->getOperand(idx);

        if (getAttachedCB(operand)) {
          continue;
        }

        if (auto result = dyn_cast<OpResult>(operand)) {
          if (auto computeOp = dyn_cast<ComputeOp>(result.getOwner())) {
            ComputeMaterializationPlan &computePlan =
                getOrCreateComputePlan(computePlans, computeOp);
            ResultMaterializationPlan &resultPlan =
                getOrCreateResultPlan(computePlan, result.getResultNumber());
            resultPlan.uses.push_back({op, idx});
            continue;
          }
        }

        fallbackUses.push_back({op, idx});
      }
    }

    for (ComputeMaterializationPlan &computePlan : computePlans) {
      if (failed(materializeComputePlan(computePlan, moduleOp, builder))) {
        signalPassFailure();
        return;
      }
    }

    DominanceInfo dominanceInfo(funcOp);
    if (failed(materializeTensorFallbackUses(fallbackUses, moduleOp, builder,
                                             dominanceInfo))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
