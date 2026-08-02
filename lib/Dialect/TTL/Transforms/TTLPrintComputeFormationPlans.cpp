// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ComputeFormationPlanning.h"
#include "DFBValueLifetimeAnalysis.h"
#include "IntermediateDFBPlanning.h"

#include "ttlang/Dialect/TTL/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLPRINTCOMPUTEFORMATIONPLANS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static StringRef stringifyKind(ComputeFormationKind kind) {
  switch (kind) {
  case ComputeFormationKind::Direct:
    return "direct";
  case ComputeFormationKind::Fused:
    return "fused";
  case ComputeFormationKind::Elide:
    return "elide";
  }
  llvm_unreachable("unknown compute formation kind");
}

static StringRef stringifyRecipe(ComputeFormationRecipe recipe) {
  switch (recipe) {
  case ComputeFormationRecipe::Elementwise:
    return "elementwise";
  case ComputeFormationRecipe::BlockBroadcast:
    return "block-broadcast";
  case ComputeFormationRecipe::Matmul:
    return "matmul";
  case ComputeFormationRecipe::Reduce:
    return "reduce";
  case ComputeFormationRecipe::MulUnaryConst:
    return "mul-unary-const";
  case ComputeFormationRecipe::Fill:
    return "fill";
  case ComputeFormationRecipe::Typecast:
    return "typecast";
  case ComputeFormationRecipe::Transpose:
    return "transpose";
  case ComputeFormationRecipe::Fused:
    return "fused";
  case ComputeFormationRecipe::Elide:
    return "elide";
  }
  llvm_unreachable("unknown compute formation recipe");
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
  case IntermediateDFBReason::FormationInputMayBeReleased:
    return "formation-input-may-be-released";
  case IntermediateDFBReason::MultipleOutputTransactions:
    return "multiple-output-transactions";
  case IntermediateDFBReason::FormationWouldNotDominateUse:
    return "formation-would-not-dominate-use";
  case IntermediateDFBReason::ComputeResultHasMaterializedUse:
    return "compute-result-has-materialized-use";
  case IntermediateDFBReason::FormationRequiresMaterializedInput:
    return "formation-requires-materialized-input";
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

struct TTLPrintComputeFormationPlansPass
    : public impl::TTLPrintComputeFormationPlansBase<
          TTLPrintComputeFormationPlansPass> {
  using TTLPrintComputeFormationPlansBase::TTLPrintComputeFormationPlansBase;

  void runOnOperation() override {
    func::FuncOp kernel = getOperation();
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

    ComputeFormationPlanner formationPlanner(kernel, *lifetimes);
    PlanningResult<KernelComputeFormationPlan> plannedFormations =
        formationPlanner.build();
    if (plannedFormations.isInvalidIR()) {
      const PlanningDiagnostic &planningDiagnostic =
          plannedFormations.getInvalidIR();
      planningDiagnostic.operation->emitOpError(planningDiagnostic.message);
      signalPassFailure();
      return;
    }
    assert(plannedFormations.isPlanned() &&
           "kernel formation planning has no recoverable rejection");
    KernelComputeFormationPlan formations =
        std::move(plannedFormations).takePlan();
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

    DenseMap<Operation *, unsigned> formationIds;
    for (auto [formationId, source] :
         llvm::enumerate(formations.getAnalyzedSources())) {
      formationIds.try_emplace(source, formationId);
    }

    raw_ostream &output = llvm::errs();
    output << "Compute formation plan @" << kernel.getSymName() << "\n";
    for (Operation *source : formations.getAnalyzedSources()) {
      const ComputeFormationPlan &formation =
          formations.getAnalyzedFormation(source);
      output << "  F" << formationIds.at(source) << " ";
      printOperation(output, source, operationIds);
      output << " " << source->getName()
             << " kind=" << stringifyKind(formation.kind)
             << " recipe=" << stringifyRecipe(formation.recipe)
             << " legal=" << (formation.isLegal() ? "true" : "false")
             << " inputs=" << formation.inputs.size()
             << " outputs=" << formation.outputs.dfbs.size()
             << " transactions=" << formation.outputs.transactions.size()
             << "\n";

      output << "    iterators=[";
      llvm::interleaveComma(formation.iteration.iteratorTypes, output,
                            [&](utils::IteratorType iteratorType) {
                              output
                                  << utils::stringifyIteratorType(iteratorType);
                            });
      output << "] input-maps=" << formation.iteration.inputMaps.size() << " [";
      llvm::interleaveComma(formation.iteration.inputMaps, output,
                            [&](AffineMap inputMap) { output << inputMap; });
      output << "] output-map=" << formation.iteration.outputMap << "\n";

      for (const FusedOperationPlan &operationPlan :
           formation.fusedOperations) {
        output << "    fused ";
        printOperation(output, operationPlan.source, operationIds);
        output << " " << stringifyFusedRecipe(operationPlan.recipe)
               << " operands=" << operationPlan.operands.size() << "\n";
      }
      for (const ComputeFormationWarning &warning : formation.warnings) {
        output << "    warning="
               << getComputeFormationWarningMessage(warning.kind) << " at=";
        printOperation(output, warning.operation, operationIds);
        output << "\n";
      }
      for (const ComputeFormationUse &preservingUse :
           formation.preservingUses) {
        output << "    preserved-by ";
        printOperation(output, preservingUse.owner, operationIds);
        output << " operand=" << preservingUse.operandIndex << "\n";
      }
      for (const ComputeFormationUse &removedUse :
           formation.preFormationRemovedUses) {
        output << "    removed-before ";
        printOperation(output, removedUse.owner, operationIds);
        output << " operand=" << removedUse.operandIndex << "\n";
      }
      if (!formation.isLegal()) {
        output << "    rejected=" << formation.rejectionReason << "\n";
      }
    }

    DenseSet<Operation *> printedRejectedSources;
    for (StoreOp store : formations.getUnassignedStores()) {
      std::optional<Operation *> formationSource =
          formations.getUnassignedStoreFormationSource(store);
      if (!formationSource) {
        continue;
      }
      Operation *source = *formationSource;
      if (formations.hasFormationRecord(source) ||
          !printedRejectedSources.insert(source).second) {
        continue;
      }
      output << "  rejected-source ";
      printOperation(output, source, operationIds);
      output << " " << source->getName()
             << " reason=" << formations.getRejectionReason(source) << "\n";
    }

    output << "  order=[";
    llvm::interleaveComma(
        formations.getFormationOrder(), output,
        [&](Operation *source) { output << "F" << formationIds.at(source); });
    output << "]\n";

    for (StoreOp store : formations.getUnassignedStores()) {
      PlanningDiagnostic diagnostic =
          formations.getUnassignedStoreDiagnostic(store);
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
