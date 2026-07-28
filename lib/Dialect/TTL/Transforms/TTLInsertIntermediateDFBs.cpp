// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Intermediate DFBs
//===----------------------------------------------------------------------===//
//
// Inserts compiler-allocated intermediate dataflow buffers at fusion split
// points. When a tensor-level op requires a DFB-attached operand produced by
// ttl.compute, this pass adds a DFB-backed output to the producer compute and
// rewrites the consumer operand to the waited, attached tensor.
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
    SmallVector<StandaloneTensorMaterializationUse> standaloneTensorUses;

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

        // Compute results are materialized by rebuilding the producer once,
        // even when several results or consumers require DFB-attached values.
        if (std::optional<ComputeResult> computeResult =
                getComputeResult(operand)) {
          ComputeMaterializationPlan &computePlan =
              getOrCreateComputePlan(computePlans, computeResult->producer);
          ResultMaterializationPlan &resultPlan =
              getOrCreateResultPlan(computePlan, computeResult->resultIndex);
          resultPlan.uses.push_back({op, idx});
          continue;
        }

        // Other tensor producers use standalone reserve/store/wait/attach.
        standaloneTensorUses.push_back({op, idx});
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
