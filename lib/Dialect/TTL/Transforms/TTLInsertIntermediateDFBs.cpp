// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Intermediate DFBs
//===----------------------------------------------------------------------===//
//
// Inserts compiler-allocated intermediate dataflow buffers for tensor SSA
// values that need concrete storage before final compute creation. The pass
// first builds an immutable operand-rewrite plan, then applies standalone
// materializations and atomic compute rebuilds.
//
//===----------------------------------------------------------------------===//

#include "DFBValueLifetimeAnalysis.h"
#include "IntermediateDFBPlanning.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-intermediate-dfbs"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTINTERMEDIATEDFBS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Tracks one validated source result and its generated DFB lifecycle.
struct MaterializedOutput {
  /// Result of the original compute replicated to this DFB.
  unsigned sourceResultIndex;

  /// Tensor type shared by the result and generated DFB view.
  RankedTensorType tensorType;

  /// Existing output DFB whose tile stores are replicated.
  Value sourceDFB;

  /// Compiler-created DFB declaration.
  BindCBOp bind;

  /// Producer acquisition used by replicated tile stores.
  CBReserveOp reserve;

  /// Consumer association used to replace planned operands.
  AttachCBOp attach;

  /// Number of stores replicated while rebuilding the compute body.
  unsigned storeCount = 0;
};

static void cloneComputeBodyWithMaterializedStores(
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
      // Preserve the original output store and replicate its tile into the
      // compiler DFB required by downstream DFB-only consumers.
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
}

static void applyComputeMaterializationPlan(
    const ComputeDFBMaterializationPlan &plan,
    ArrayRef<IntermediateDFBRequirement> requirements, OpBuilder &builder) {
  ComputeOp producerCompute = plan.producer;
  SmallVector<MaterializedOutput> materializedOutputs;
  materializedOutputs.reserve(plan.results.size());
  for (const ComputeResultDFBMaterializationPlan &resultPlan : plan.results) {
    MaterializedOutput output;
    output.sourceResultIndex = resultPlan.resultIndex;
    output.tensorType = resultPlan.tensorType;
    output.sourceDFB = resultPlan.sourceDFB;
    materializedOutputs.push_back(output);
  }

  auto kernel = producerCompute->getParentOfType<func::FuncOp>();
  assert(kernel && "ttl.compute must be inside a kernel");
  {
    OpBuilder::InsertionGuard guard(builder);
    for (MaterializedOutput &output : materializedOutputs) {
      output.bind = createCompilerAllocatedDFB(
          output.tensorType, producerCompute.getLoc(), kernel, builder);
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

  cloneComputeBodyWithMaterializedStores(producerCompute, rebuiltCompute,
                                         materializedOutputs, builder);

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
  // TODO(#724): Relax this restriction when trace-balance analysis can prove
  // DFB occupancy across structured control flow.
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
    for (unsigned requirementIndex : resultPlan.requirementIndices) {
      const IntermediateDFBRequirement &use = requirements[requirementIndex];
      use.consumer->setOperand(use.operandIndex, output.attach.getResult());
    }
  }
}

static void applyStandaloneMaterializationPlan(
    const StandaloneDFBMaterializationPlan &plan,
    ArrayRef<IntermediateDFBRequirement> requirements, func::FuncOp kernel,
    OpBuilder &builder) {
  // The planner proves that the anchor follows the source definition and that
  // the attached result dominates every rewritten consumer. A later output
  // store anchor keeps the compiler wait after any compute formed for the
  // source. Without an applicable output plan, evaluation remains at the
  // definition. Otherwise, a definition anchor requires every source use to
  // be rewritten, which leaves the compiler DFB store as the source's only use.
  Value materializedValue =
      materializeToDFB(plan.source, plan.insertionAnchor, kernel, builder);
  for (unsigned requirementIndex : plan.requirementIndices) {
    const IntermediateDFBRequirement &use = requirements[requirementIndex];
    use.consumer->setOperand(use.operandIndex, materializedValue);
  }
}

static LogicalResult verifyPlanSources(const IntermediateDFBPlan &plan) {
  for (const IntermediateDFBRequirement &requirement : plan.getRequirements()) {
    if (requirement.operandIndex >= requirement.consumer->getNumOperands() ||
        requirement.consumer->getOperand(requirement.operandIndex) !=
            requirement.value) {
      return requirement.consumer->emitOpError(
          "intermediate DFB plan was invalidated before application");
    }
  }
  return success();
}

struct TTLInsertIntermediateDFBsPass
    : public impl::TTLInsertIntermediateDFBsBase<
          TTLInsertIntermediateDFBsPass> {
  using TTLInsertIntermediateDFBsBase::TTLInsertIntermediateDFBsBase;

  void runOnOperation() override {
    auto kernel = getOperation();
    if (kernel.isExternal()) {
      return;
    }

    PlanningResult<std::unique_ptr<DFBValueLifetimeAnalysis>> plannedLifetimes =
        DFBValueLifetimeAnalysis::create(kernel);
    if (plannedLifetimes.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = plannedLifetimes.getInvalidIR();
      diagnostic.operation->emitOpError(diagnostic.message);
      signalPassFailure();
      return;
    }
    assert(plannedLifetimes.isPlanned() &&
           "lifetime analysis has no recoverable rejection");
    std::unique_ptr<DFBValueLifetimeAnalysis> lifetimes =
        std::move(plannedLifetimes).takePlan();

    IntermediateDFBPlanner materializationPlanner(kernel, *lifetimes);
    PlanningResult<IntermediateDFBPlan> plannedMaterializations =
        materializationPlanner.build();
    if (plannedMaterializations.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic =
          plannedMaterializations.getInvalidIR();
      diagnostic.operation->emitOpError(diagnostic.message);
      signalPassFailure();
      return;
    }
    assert(plannedMaterializations.isPlanned() &&
           "intermediate DFB planning has no recoverable rejection");
    IntermediateDFBPlan materializationPlan =
        std::move(plannedMaterializations).takePlan();
    ArrayRef<IntermediateDFBRequirement> requiredUses =
        materializationPlan.getRequirements();

    // When compiler DFBs are disabled, verify that no operations require
    // them and emit an actionable error if any do.
    if (!enable) {
      if (!requiredUses.empty()) {
        const IntermediateDFBRequirement &requiredUse = requiredUses.front();
        requiredUse.consumer->emitOpError("operand #")
            << requiredUse.operandIndex
            << " requires compiler-created DFB materialization, but "
               "compiler DFBs are disabled (--no-ttl-compiler-dfbs); either "
               "enable compiler DFBs or store the intermediate to a "
               "user-declared DFB before this operation";
        signalPassFailure();
      }
      return;
    }

    if (failed(verifyPlanSources(materializationPlan))) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(kernel.getContext());
    // Standalone materialization only inserts operations and rewrites uses, so
    // applying it first preserves every compute operation recorded for an
    // atomic rebuild. Compute rebuilds then follow definition order; valid SSA
    // ensures a producer rewrites a consumer before that consumer is rebuilt.
    for (const StandaloneDFBMaterializationPlan &standalonePlan :
         materializationPlan.getStandaloneMaterializations()) {
      applyStandaloneMaterializationPlan(standalonePlan, requiredUses, kernel,
                                         builder);
    }
    for (const ComputeDFBMaterializationPlan &computePlan :
         materializationPlan.getComputeMaterializations()) {
      applyComputeMaterializationPlan(computePlan, requiredUses, builder);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
