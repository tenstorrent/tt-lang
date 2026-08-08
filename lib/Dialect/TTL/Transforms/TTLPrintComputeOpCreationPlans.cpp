// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ComputeOpCreationPlanning.h"
#include "DFBValueLifetimeAnalysis.h"
#include "IntermediateDFBPlanning.h"

#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/ComputeTarget.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLPRINTCOMPUTEOPCREATIONPLANS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static StringRef stringifyKind(ComputeOpCreationKind kind) {
  switch (kind) {
  case ComputeOpCreationKind::Direct:
    return "direct";
  case ComputeOpCreationKind::Fused:
    return "fused";
  case ComputeOpCreationKind::Elide:
    return "elide";
  }
  llvm_unreachable("unknown ComputeOp creation kind");
}

static StringRef stringifyRecipe(ComputeOpCreationRecipe recipe) {
  switch (recipe) {
  case ComputeOpCreationRecipe::Elementwise:
    return "elementwise";
  case ComputeOpCreationRecipe::BlockBroadcast:
    return "block-broadcast";
  case ComputeOpCreationRecipe::Matmul:
    return "matmul";
  case ComputeOpCreationRecipe::Reduce:
    return "reduce";
  case ComputeOpCreationRecipe::MulUnaryConst:
    return "mul-unary-const";
  case ComputeOpCreationRecipe::Fill:
    return "fill";
  case ComputeOpCreationRecipe::Typecast:
    return "typecast";
  case ComputeOpCreationRecipe::Transpose:
    return "transpose";
  case ComputeOpCreationRecipe::Fused:
    return "fused";
  case ComputeOpCreationRecipe::Elide:
    return "elide";
  }
  llvm_unreachable("unknown ComputeOp creation recipe");
}

static StringRef stringifyFusedRecipe(FusedOperationRecipe recipe) {
  switch (recipe) {
  case FusedOperationRecipe::TileOperation:
    return "tile-operation";
  case FusedOperationRecipe::InterTileBroadcast:
    return "inter-tile-broadcast";
  case FusedOperationRecipe::TileBroadcast:
    return "tile-broadcast";
  case FusedOperationRecipe::Matmul:
    return "matmul";
  case FusedOperationRecipe::DeferredMatmul:
    return "deferred-matmul";
  case FusedOperationRecipe::MatmulAccumulator:
    return "matmul-accumulator";
  case FusedOperationRecipe::DeferredExpScale:
    return "deferred-exp-scale";
  }
  llvm_unreachable("unknown fused operation recipe");
}

static StringRef stringifyReason(IntermediateDFBReason reason) {
  switch (reason) {
  case IntermediateDFBReason::RequiredDFBOperand:
    return "required-dfb-operand";
  case IntermediateDFBReason::DFBInputMayBeReleased:
    return "dfb-input-may-be-released";
  case IntermediateDFBReason::ExpressionInputMayBeReleased:
    return "expression-input-may-be-released";
  case IntermediateDFBReason::ComputeOpInputMayBeReleased:
    return "compute-op-input-may-be-released";
  case IntermediateDFBReason::OutputStoresInDifferentBlocks:
    return "output-stores-in-different-blocks";
  case IntermediateDFBReason::MultipleOutputTransactions:
    return "multiple-output-transactions";
  case IntermediateDFBReason::ComputeOpWouldNotDominateUse:
    return "compute-op-would-not-dominate-use";
  case IntermediateDFBReason::ComputeOpInstrumentationWouldBeReordered:
    return "compute-op-instrumentation-would-be-reordered";
  case IntermediateDFBReason::ComputeResultHasMaterializedUse:
    return "compute-result-has-materialized-use";
  case IntermediateDFBReason::ComputeOpRequiresMaterializedInput:
    return "compute-op-requires-materialized-input";
  }
  llvm_unreachable("unknown intermediate DFB reason");
}

static void
printOperation(raw_ostream &output, Operation *operation,
               const DenseMap<Operation *, unsigned> &operationIds) {
  output << "O" << operationIds.at(operation);
}

static void printValue(raw_ostream &output, Value value,
                       const DenseMap<Operation *, unsigned> &operationIds,
                       const DenseMap<Block *, unsigned> &blockIds) {
  if (auto result = dyn_cast<OpResult>(value)) {
    printOperation(output, result.getOwner(), operationIds);
    output << "R" << result.getResultNumber();
    return;
  }
  auto argument = cast<BlockArgument>(value);
  auto blockId = blockIds.find(argument.getOwner());
  assert(blockId != blockIds.end() && "every printed block must have an ID");
  output << "B" << blockId->second << "A" << argument.getArgNumber();
}

struct TTLPrintComputeOpCreationPlansPass
    : public impl::TTLPrintComputeOpCreationPlansBase<
          TTLPrintComputeOpCreationPlansPass> {
  using TTLPrintComputeOpCreationPlansBase::TTLPrintComputeOpCreationPlansBase;

  void runOnOperation() override {
    func::FuncOp kernel = getOperation();
    if (kernel.isExternal()) {
      return;
    }

    std::string targetFailureReason;
    FailureOr<std::unique_ptr<ComputeTargetEnvironment>> target =
        ComputeTargetEnvironment::get(kernel, targetFailureReason);
    if (failed(target)) {
      kernel.emitOpError(targetFailureReason);
      signalPassFailure();
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

    ComputeOpCreationPlanner creationPlanner(kernel, *lifetimes, **target);
    PlanningResult<KernelComputeOpCreationPlan> plannedCreations =
        creationPlanner.build();
    if (plannedCreations.isInvalidIR()) {
      const PlanningDiagnostic &planningDiagnostic =
          plannedCreations.getInvalidIR();
      planningDiagnostic.operation->emitOpError(planningDiagnostic.message);
      signalPassFailure();
      return;
    }
    assert(plannedCreations.isPlanned() &&
           "kernel ComputeOp creation planning has no recoverable rejection");
    KernelComputeOpCreationPlan creations =
        std::move(plannedCreations).takePlan();
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
    IntermediateDFBPlan materializations =
        std::move(plannedMaterializations).takePlan();

    DenseMap<Operation *, unsigned> operationIds;
    DenseMap<Block *, unsigned> blockIds;
    unsigned nextOperationId = 0;
    unsigned nextBlockId = 0;
    auto assignBlockId = [&](Block &block) {
      if (!blockIds.contains(&block)) {
        blockIds.try_emplace(&block, nextBlockId++);
      }
    };
    for (Block &block : kernel.getBody()) {
      assignBlockId(block);
    }
    kernel.walk([&](Operation *operation) {
      if (operation != kernel.getOperation()) {
        operationIds.try_emplace(operation, nextOperationId++);
      }
      for (Region &region : operation->getRegions()) {
        for (Block &block : region) {
          assignBlockId(block);
        }
      }
    });

    DenseMap<Operation *, unsigned> creationIds;
    for (auto [creationId, source] :
         llvm::enumerate(creations.getAnalyzedSources())) {
      creationIds.try_emplace(source, creationId);
    }

    raw_ostream &output = llvm::errs();
    output << "ComputeOp creation plan @" << kernel.getSymName() << "\n";
    for (Operation *source : creations.getAnalyzedSources()) {
      const ComputeOpCreationPlan &creation =
          creations.getAnalyzedCreation(source);
      output << "  C" << creationIds.at(source) << " ";
      printOperation(output, source, operationIds);
      output << " " << source->getName()
             << " kind=" << stringifyKind(creation.kind)
             << " recipe=" << stringifyRecipe(creation.recipe)
             << " legal=" << (creation.isLegal() ? "true" : "false")
             << " inputs=" << creation.inputs.size()
             << " outputs=" << creation.outputs.dfbs.size()
             << " transactions=" << creation.outputs.transactions.size()
             << "\n";

      output << "    iterators=[";
      llvm::interleaveComma(creation.iteration.iteratorTypes, output,
                            [&](utils::IteratorType iteratorType) {
                              output
                                  << utils::stringifyIteratorType(iteratorType);
                            });
      output << "] input-maps=" << creation.iteration.inputMaps.size() << " [";
      llvm::interleaveComma(creation.iteration.inputMaps, output,
                            [&](AffineMap inputMap) { output << inputMap; });
      output << "] output-map=" << creation.iteration.outputMap << "\n";

      for (const FusedOperationPlan &operationPlan : creation.fusedOperations) {
        output << "    fused ";
        printOperation(output, operationPlan.source, operationIds);
        output << " " << stringifyFusedRecipe(operationPlan.recipe)
               << " operands=" << operationPlan.operands.size() << "\n";
      }
      for (const ComputeOpCreationWarning &warning : creation.warnings) {
        output << "    warning="
               << getComputeOpCreationWarningMessage(warning.kind) << " at=";
        printOperation(output, warning.operation, operationIds);
        output << "\n";
      }
      for (const ComputeOpCreationUse &preservingUse :
           creation.preservingUses) {
        output << "    preserved-by ";
        printOperation(output, preservingUse.owner, operationIds);
        output << " operand=" << preservingUse.operandIndex << "\n";
      }
      for (const ComputeOpCreationUse &removedUse :
           creation.preCreationRemovedUses) {
        output << "    removed-before ";
        printOperation(output, removedUse.owner, operationIds);
        output << " operand=" << removedUse.operandIndex << "\n";
      }
      if (!creation.isLegal()) {
        output << "    rejected=" << creation.rejectionReason << "\n";
      }
    }

    DenseSet<Operation *> printedRejectedSources;
    for (StoreOp store : creations.getUnassignedStores()) {
      std::optional<Operation *> sourceOperation =
          creations.getUnassignedStoreSource(store);
      if (!sourceOperation) {
        continue;
      }
      Operation *source = *sourceOperation;
      if (creations.hasCreationRecord(source) ||
          !printedRejectedSources.insert(source).second) {
        continue;
      }
      output << "  rejected-source ";
      printOperation(output, source, operationIds);
      output << " " << source->getName()
             << " reason=" << creations.getRejectionReason(source) << "\n";
    }

    output << "  order=[";
    llvm::interleaveComma(
        creations.getCreationOrder(), output,
        [&](Operation *source) { output << "C" << creationIds.at(source); });
    output << "]\n";

    for (StoreOp store : creations.getUnassignedStores()) {
      PlanningDiagnostic diagnostic =
          creations.getUnassignedStoreDiagnostic(store);
      output << "  unassigned-store ";
      printOperation(output, store, operationIds);
      output << " reason=" << diagnostic.message << "\n";
    }

    for (auto [requirementId, requirement] :
         llvm::enumerate(materializations.getRequirements())) {
      output << "  M" << requirementId << " ";
      printOperation(output, requirement.consumer, operationIds);
      output << " operand=" << requirement.operandIndex << "\n";
      for (const IntermediateDFBEvidence &evidence : requirement.evidence) {
        output << "    reason=" << stringifyReason(evidence.reason) << " at=";
        printOperation(output, evidence.observation, operationIds);
        output << " inputs=[";
        llvm::interleaveComma(evidence.inputs, output, [&](Value input) {
          printValue(output, input, operationIds, blockIds);
        });
        output << "]";
        if (evidence.outputDFB) {
          output << " output-dfb=";
          printValue(output, *evidence.outputDFB, operationIds, blockIds);
        }
        output << "\n";
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
