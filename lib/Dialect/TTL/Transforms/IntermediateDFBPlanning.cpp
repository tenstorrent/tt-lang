// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "IntermediateDFBPlanning.h"

#include "ComputeOpCreationPlanning.h"
#include "DFBValueLifetimeAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <tuple>

namespace mlir::tt::ttl {

IntermediateDFBPlan::IntermediateDFBPlan(
    SmallVector<IntermediateDFBRequirement> requirements,
    SmallVector<ComputeDFBMaterializationPlan> computeMaterializations,
    SmallVector<StandaloneDFBMaterializationPlan> standaloneMaterializations)
    : requirements(std::move(requirements)),
      computeMaterializations(std::move(computeMaterializations)),
      standaloneMaterializations(std::move(standaloneMaterializations)) {}

bool DFBMaterializationAnalysisState::requiresMaterialization(
    const OpOperand &operand) const {
  return llvm::any_of(
      requirements, [&](const IntermediateDFBRequirement &requirement) {
        return requirement.consumer == operand.getOwner() &&
               requirement.operandIndex == operand.getOperandNumber();
      });
}

bool DFBMaterializationAnalysisState::requireMaterialization(
    OpOperand &operand, IntermediateDFBEvidence evidence) {
  Value value = operand.get();
  auto requirement = llvm::find_if(
      requirements, [&](const IntermediateDFBRequirement &candidate) {
        return candidate.consumer == operand.getOwner() &&
               candidate.operandIndex == operand.getOperandNumber();
      });
  if (requirement == requirements.end()) {
    requirements.push_back({operand.getOwner(),
                            operand.getOperandNumber(),
                            value,
                            {std::move(evidence)}});
    return true;
  }
  assert(requirement->value == value &&
         "immutable consumer operand changed during planning");
  if (llvm::none_of(requirement->evidence,
                    [&](const IntermediateDFBEvidence &candidate) {
                      return candidate.reason == evidence.reason &&
                             candidate.observation == evidence.observation &&
                             candidate.outputDFB == evidence.outputDFB &&
                             llvm::equal(candidate.inputs, evidence.inputs);
                    })) {
    requirement->evidence.push_back(std::move(evidence));
  }
  return false;
}

namespace {

static void
addComputeResultUseRequirements(DFBMaterializationAnalysisState &state) {
  llvm::SetVector<Value> materializedComputeResults;
  for (const IntermediateDFBRequirement &requirement :
       state.getRequirements()) {
    auto result = dyn_cast<OpResult>(requirement.value);
    if (result && isa<ComputeOp>(result.getOwner())) {
      materializedComputeResults.insert(result);
    }
  }

  // ComputeOp results preserve SSA dependencies during this stage, but
  // lowering communicates computed tiles only through output DFB stores. If
  // `%sum` feeds both a reduce and a multiply, materializing only the reduce
  // would leave the multiply reading a result with no readable storage after
  // lowering. Rewrite both operands to one stored, pushed, and waited value so
  // every surviving use observes the same publication of `%sum`.
  for (Value result : materializedComputeResults) {
    for (OpOperand &use : result.getUses()) {
      if (state.requiresMaterialization(use)) {
        continue;
      }
      IntermediateDFBEvidence evidence;
      evidence.reason = IntermediateDFBReason::ComputeResultHasMaterializedUse;
      evidence.inputs = {result};
      evidence.observation = use.getOwner();
      state.requireMaterialization(use, std::move(evidence));
    }
  }
}

static void
addExpressionReleaseRequirements(Operation *elementwiseOp,
                                 const DFBValueLifetimeAnalysis &lifetimes,
                                 DFBMaterializationAnalysisState &state) {
  for (OpOperand &operand : elementwiseOp->getOpOperands()) {
    Value value = operand.get();
    // DFB attachment proves where the value resides, not that its slot remains
    // live at this consumer. Tracing attached operands lets the lifetime query
    // materialize values used after their original DFB release.
    FusionTraceResult trace =
        traceFusionToRoots(value, [&](OpOperand &tracedOperand) {
          return state.requiresMaterialization(tracedOperand);
        });
    if (trace.failureReason != TraceFailureReason::Success ||
        !lifetimes.anyValueMayBeReleased(trace.lifetimeRootInputs.getArrayRef(),
                                         elementwiseOp)) {
      continue;
    }

    IntermediateDFBEvidence evidence;
    evidence.reason = IntermediateDFBReason::ExpressionInputMayBeReleased;
    llvm::append_range(evidence.inputs, trace.lifetimeRootInputs);
    evidence.observation = elementwiseOp;
    state.requireMaterialization(operand, std::move(evidence));
  }
}

static void addStoreRequirement(DFBMaterializationAnalysisState &state,
                                OpOperand &storedValueOperand,
                                IntermediateDFBEvidence evidence) {
  state.requireMaterialization(storedValueOperand, std::move(evidence));
}

static SmallVector<OpOperand *> getDirectStoreOperands(Value value) {
  SmallVector<OpOperand *> storeOperands;
  for (OpOperand &use : value.getUses()) {
    auto store = dyn_cast<StoreOp>(use.getOwner());
    if (store && &store.getTensorMutable() == &use) {
      storeOperands.push_back(&use);
    }
  }
  return storeOperands;
}

static bool storeOperandsUseDifferentBlocks(ArrayRef<OpOperand *> operands) {
  if (operands.empty()) {
    return false;
  }
  Block *firstBlock = operands.front()->getOwner()->getBlock();
  return llvm::any_of(operands.drop_front(), [&](OpOperand *operand) {
    return operand->getOwner()->getBlock() != firstBlock;
  });
}

static void
addMultiBlockStoreRequirements(Operation *source,
                               DFBMaterializationAnalysisState &state) {
  for (Value result : source->getResults()) {
    if (!isa<RankedTensorType>(result.getType()) || getAttachedCB(result)) {
      continue;
    }

    SmallVector<OpOperand *> storeOperands = getDirectStoreOperands(result);
    if (!storeOperandsUseDifferentBlocks(storeOperands)) {
      continue;
    }

    for (OpOperand *storeOperand : storeOperands) {
      IntermediateDFBEvidence evidence;
      evidence.reason = IntermediateDFBReason::OutputStoresInDifferentBlocks;
      evidence.inputs = {result};
      evidence.observation = storeOperand->getOwner();
      addStoreRequirement(state, *storeOperand, std::move(evidence));
    }
  }
}

static void
addCreationOrderingRequirements(Operation *source,
                                const OutputPublicationPlan &outputs,
                                const DominanceInfo &dominanceInfo,
                                DFBMaterializationAnalysisState &state) {
  if (isComputeOpCreationElision(source)) {
    return;
  }

  Value result = source->getResult(0);
  auto unsafeUse = llvm::find_if(result.getUses(), [&](OpOperand &use) {
    return !isComputeOpCreationUsePreserved(outputs, use, dominanceInfo);
  });
  if (unsafeUse == result.getUses().end()) {
    return;
  }

  IntermediateDFBEvidence evidence;
  evidence.reason = IntermediateDFBReason::ComputeOpWouldNotDominateUse;
  evidence.inputs = {result};
  evidence.observation = unsafeUse->getOwner();

  // Rewriting every use makes the compiler DFB store the source's only output
  // publication. Final creation must then execute at that store instead of
  // combining it with a later user-DFB store and recreating the bad ordering.
  for (OpOperand &use : result.getUses()) {
    state.requireMaterialization(use, evidence);
  }
}

static LogicalResult addCreationRequirements(
    Operation *source, const DFBValueLifetimeAnalysis &lifetimes,
    const DominanceInfo &dominanceInfo, DFBMaterializationAnalysisState &state,
    std::optional<PlanningDiagnostic> &diagnostic) {
  addMultiBlockStoreRequirements(source, state);

  PlanningResult<OutputPublicationPlan, OutputPublicationRejection> outputs =
      buildOutputPublicationPlan(source);
  if (outputs.isInvalidIR()) {
    diagnostic.emplace(outputs.getInvalidIR());
    return failure();
  }
  if (outputs.isRejected()) {
    return success();
  }
  const OutputPublicationPlan &outputPlan = outputs.getPlan();

  FailureOr<SmallVector<Value>> inputs =
      collectComputeOpCreationLifetimeInputs(source, [&](OpOperand &operand) {
        return state.requiresMaterialization(operand);
      });
  if (failed(inputs)) {
    FusionTraceResult trace =
        traceFusionToRoots(source->getResult(0), [&](OpOperand &operand) {
          return state.requiresMaterialization(operand);
        });
    OpOperand *failedOperand = trace.failedOperand;
    Operation *producer =
        failedOperand ? failedOperand->get().getDefiningOp() : nullptr;
    if (!producer || (!isa<ComputeOp>(producer) &&
                      !hasStandaloneComputeOpCreationRecipe(producer))) {
      return success();
    }

    IntermediateDFBEvidence evidence;
    evidence.reason = IntermediateDFBReason::ComputeOpRequiresMaterializedInput;
    evidence.inputs = {failedOperand->get()};
    evidence.observation = source;
    state.requireMaterialization(*failedOperand, std::move(evidence));
    return success();
  }

  FailureOr<SmallVector<ComputeOpCreationInstrumentationBoundary>> boundaries =
      collectComputeOpCreationInstrumentationBoundaries(
          source, outputPlan, [&](OpOperand &operand) {
            return state.requiresMaterialization(operand);
          });
  if (succeeded(boundaries) && !boundaries->empty()) {
    for (const ComputeOpCreationInstrumentationBoundary &boundary :
         *boundaries) {
      for (const ComputeOpCreationUse &use : boundary.crossingUses) {
        assert(use.operandIndex < use.owner->getNumOperands() &&
               "creation ordering analysis recorded an invalid operand");
        OpOperand &operand = use.owner->getOpOperand(use.operandIndex);
        IntermediateDFBEvidence evidence;
        evidence.reason =
            IntermediateDFBReason::ComputeOpInstrumentationWouldBeReordered;
        evidence.inputs = {operand.get()};
        evidence.observation = boundary.operation;
        state.requireMaterialization(operand, std::move(evidence));
      }
    }
    return success();
  }

  addCreationOrderingRequirements(source, outputPlan, dominanceInfo, state);

  Value result = source->getResult(0);
  for (const OutputDFBTransaction &transaction : outputPlan.transactions) {
    bool splitsTransactions =
        outputPlan.hasMultipleTransactions(transaction.dfb);
    if (splitsTransactions) {
      // No creation can publish the result through these transactions. Route
      // every use through one compiler DFB so each original store and consumer
      // becomes an independent creation after conflict resolution.
      for (OpOperand &use : result.getUses()) {
        IntermediateDFBEvidence evidence;
        evidence.reason = IntermediateDFBReason::MultipleOutputTransactions;
        evidence.observation = use.getOwner();
        evidence.outputDFB = transaction.dfb;
        state.requireMaterialization(use, std::move(evidence));
      }
    }
    for (StoreOp store : transaction.stores) {
      if (lifetimes.anyValueMayBeReleased(*inputs, store)) {
        IntermediateDFBEvidence evidence;
        evidence.reason = IntermediateDFBReason::ComputeOpInputMayBeReleased;
        evidence.inputs = *inputs;
        evidence.observation = store;
        addStoreRequirement(state, store.getTensorMutable(),
                            std::move(evidence));
      }
    }
  }
  return success();
}

static ComputeDFBMaterializationPlan &getOrCreateComputeMaterialization(
    SmallVectorImpl<ComputeDFBMaterializationPlan> &plans, ComputeOp producer) {
  auto existing = llvm::find_if(
      plans, [&](const auto &plan) { return plan.producer == producer; });
  if (existing != plans.end()) {
    return *existing;
  }
  plans.push_back({producer, {}});
  return plans.back();
}

static ComputeResultDFBMaterializationPlan &
getOrCreateResultMaterialization(ComputeDFBMaterializationPlan &plan,
                                 unsigned resultIndex,
                                 RankedTensorType tensorType, Value sourceDFB) {
  auto existing = llvm::find_if(plan.results, [&](const auto &result) {
    return result.resultIndex == resultIndex;
  });
  if (existing != plan.results.end()) {
    assert(existing->tensorType == tensorType &&
           existing->sourceDFB == sourceDFB &&
           "one compute result must have consistent output metadata");
    return *existing;
  }
  plan.results.push_back({resultIndex, tensorType, sourceDFB, {}});
  return plan.results.back();
}

static StandaloneDFBMaterializationPlan &getOrCreateStandaloneMaterialization(
    SmallVectorImpl<StandaloneDFBMaterializationPlan> &plans, Value source,
    RankedTensorType tensorType) {
  auto existing = llvm::find_if(
      plans, [&](const auto &plan) { return plan.source == source; });
  if (existing != plans.end()) {
    assert(existing->tensorType == tensorType &&
           "one tensor value must have a consistent type");
    return *existing;
  }
  plans.push_back({source, tensorType, source.getDefiningOp(), {}});
  return plans.back();
}

} // namespace

PlanningResult<IntermediateDFBPlan>
IntermediateDFBPlanner::buildMaterializationRecords(
    SmallVector<IntermediateDFBRequirement> requirements) const {
  SmallVector<ComputeDFBMaterializationPlan> computeMaterializations;
  SmallVector<StandaloneDFBMaterializationPlan> standaloneMaterializations;

  // Group every result from one existing compute into one atomic rebuild and
  // every other source value into one standalone materialization. These source
  // sets are disjoint, and result replacement updates consumers through SSA,
  // so requirement discovery order does not constrain plan application.
  for (auto [requirementIndex, requirement] : llvm::enumerate(requirements)) {
    if (requirement.consumer->getParentOfType<func::FuncOp>() != kernel ||
        requirement.operandIndex >= requirement.consumer->getNumOperands() ||
        requirement.consumer->getOperand(requirement.operandIndex) !=
            requirement.value) {
      return PlanningResult<IntermediateDFBPlan>::invalidIR(
          requirement.consumer,
          "intermediate DFB requirement does not match its source operand");
    }

    auto tensorType = dyn_cast<RankedTensorType>(requirement.value.getType());
    if (!tensorType) {
      return PlanningResult<IntermediateDFBPlan>::invalidIR(
          requirement.consumer,
          "intermediate DFB materialization requires a ranked tensor value");
    }

    auto result = dyn_cast<OpResult>(requirement.value);
    auto producer =
        result ? dyn_cast<ComputeOp>(result.getOwner()) : ComputeOp();
    if (producer) {
      unsigned resultIndex = result.getResultNumber();
      if (resultIndex >= producer.getNumOutputs()) {
        return PlanningResult<IntermediateDFBPlan>::invalidIR(
            producer, "compute result has no corresponding output operand");
      }
      Value sourceDFB = getAttachedCB(producer.getOutputs()[resultIndex]);
      if (!sourceDFB) {
        return PlanningResult<IntermediateDFBPlan>::invalidIR(
            producer, "compute output is not attached to a dataflow buffer");
      }
      ComputeDFBMaterializationPlan &computePlan =
          getOrCreateComputeMaterialization(computeMaterializations, producer);
      ComputeResultDFBMaterializationPlan &resultPlan =
          getOrCreateResultMaterialization(computePlan, resultIndex, tensorType,
                                           sourceDFB);
      resultPlan.requirementIndices.push_back(requirementIndex);
      continue;
    }

    if (!requirement.value.getDefiningOp()) {
      return PlanningResult<IntermediateDFBPlan>::invalidIR(
          requirement.consumer,
          "intermediate DFB materialization requires a tensor definition");
    }
    StandaloneDFBMaterializationPlan &standalonePlan =
        getOrCreateStandaloneMaterialization(standaloneMaterializations,
                                             requirement.value, tensorType);
    standalonePlan.requirementIndices.push_back(requirementIndex);
  }

  for (ComputeDFBMaterializationPlan &computePlan : computeMaterializations) {
    llvm::sort(computePlan.results,
               [](const ComputeResultDFBMaterializationPlan &lhs,
                  const ComputeResultDFBMaterializationPlan &rhs) {
                 return lhs.resultIndex < rhs.resultIndex;
               });
  }

  DominanceInfo dominanceInfo(kernel);
  for (StandaloneDFBMaterializationPlan &standalonePlan :
       standaloneMaterializations) {
    Operation *source = standalonePlan.source.getDefiningOp();
    assert(source && "standalone materialization requires a definition");
    PlanningResult<OutputPublicationPlan, OutputPublicationRejection> outputs =
        buildOutputPublicationPlan(source);
    if (outputs.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = outputs.getInvalidIR();
      return PlanningResult<IntermediateDFBPlan>::invalidIR(
          diagnostic.operation, diagnostic.message);
    }
    if (outputs.isRejected()) {
      continue;
    }

    const OutputPublicationPlan &outputPlan = outputs.getPlan();
    auto isRequiredUse = [&](OpOperand &use) {
      return llvm::any_of(
          standalonePlan.requirementIndices, [&](unsigned requirementIndex) {
            const IntermediateDFBRequirement &requirement =
                requirements[requirementIndex];
            return requirement.consumer == use.getOwner() &&
                   requirement.operandIndex == use.getOperandNumber();
          });
    };
    auto preservesInstrumentationOrder = [&](Operation *publicationAnchor) {
      return llvm::all_of(
          standalonePlan.requirementIndices, [&](unsigned requirementIndex) {
            const IntermediateDFBRequirement &requirement =
                requirements[requirementIndex];
            return llvm::all_of(
                requirement.evidence,
                [&](const IntermediateDFBEvidence &evidence) {
                  return evidence.reason !=
                             IntermediateDFBReason::
                                 ComputeOpInstrumentationWouldBeReordered ||
                         dominanceInfo.properlyDominates(publicationAnchor,
                                                         evidence.observation);
                });
          });
    };

    // Requirements rewrite selected stores before final creation. Select the
    // last publication that will remain, not the last publication in the
    // analyzed IR. The chosen position must precede recorded instrumentation
    // boundaries and preserve every surviving non-store use.
    for (StoreOp store : llvm::reverse(outputPlan.stores)) {
      OpOperand &storedValue = store.getTensorMutable();
      if (isRequiredUse(storedValue) ||
          !preservesInstrumentationOrder(store.getOperation())) {
        continue;
      }
      bool dominatesRequiredConsumers = llvm::all_of(
          standalonePlan.requirementIndices, [&](unsigned requirementIndex) {
            return dominanceInfo.properlyDominates(
                store, requirements[requirementIndex].consumer);
          });
      bool preservesSurvivingUses =
          llvm::all_of(standalonePlan.source.getUses(), [&](OpOperand &use) {
            if (isRequiredUse(use)) {
              return true;
            }
            auto survivingStore = dyn_cast<StoreOp>(use.getOwner());
            if (survivingStore && &survivingStore.getTensorMutable() == &use &&
                llvm::is_contained(outputPlan.stores, survivingStore)) {
              return !store->isBeforeInBlock(survivingStore);
            }
            return dominanceInfo.properlyDominates(store, use.getOwner());
          });
      if (dominatesRequiredConsumers && preservesSurvivingUses) {
        standalonePlan.insertionAnchor = store;
        break;
      }
    }
    if (standalonePlan.insertionAnchor != source) {
      continue;
    }

    // Creation may relocate the source to its final publication store. When
    // that store cannot dominate every materialized consumer, rewriting every
    // source use removes the original publications and leaves the compiler DFB
    // store at the definition as the sole creation anchor. Any surviving use
    // would permit final creation to execute after the compiler DFB wait.
    bool rewritesEveryUse =
        llvm::all_of(standalonePlan.source.getUses(), isRequiredUse);
    if (!rewritesEveryUse) {
      return PlanningResult<IntermediateDFBPlan>::invalidIR(
          source, "intermediate DFB plan leaves a source use not dominated by "
                  "its planned output publication");
    }
  }

  // A producer must be rebuilt before a planned compute that consumes its
  // result; otherwise rebuilding the consumer invalidates its recorded operand
  // before the producer can replace it. MLIR's region-aware topological sort
  // derives this order from valid SSA dominance across nested regions and CFG
  // blocks instead of relying on region or block list order.
  llvm::SetVector<Operation *> computeProducers;
  DenseMap<Operation *, unsigned> materializationIndices;
  for (auto [materializationIndex, plan] :
       llvm::enumerate(computeMaterializations)) {
    computeProducers.insert(plan.producer);
    materializationIndices.try_emplace(plan.producer, materializationIndex);
  }
  SmallVector<ComputeDFBMaterializationPlan> orderedMaterializations;
  orderedMaterializations.reserve(computeMaterializations.size());
  for (Operation *producer : topologicalSort(computeProducers)) {
    orderedMaterializations.push_back(std::move(
        computeMaterializations[materializationIndices.at(producer)]));
  }
  computeMaterializations = std::move(orderedMaterializations);

  return PlanningResult<IntermediateDFBPlan>::planned(IntermediateDFBPlan(
      std::move(requirements), std::move(computeMaterializations),
      std::move(standaloneMaterializations)));
}

PlanningResult<IntermediateDFBPlan> IntermediateDFBPlanner::build() const {
  DFBMaterializationAnalysisState state;
  DominanceInfo dominanceInfo(kernel);
  kernel->walk([&](DFBInputOpInterface dfbInputOp) {
    Operation *operation = dfbInputOp.getOperation();
    for (unsigned operandIndex : dfbInputOp.getDFBInputOperandIndices()) {
      OpOperand &operand = operation->getOpOperand(operandIndex);
      Value value = operand.get();
      bool isDFBBacked = static_cast<bool>(getAttachedCB(value));
      bool mayBeReleased =
          isDFBBacked && lifetimes.getAvailability(value, operation) ==
                             DFBValueAvailability::MayBeReleased;
      if (isDFBBacked && !mayBeReleased) {
        continue;
      }
      IntermediateDFBEvidence evidence;
      evidence.reason = mayBeReleased
                            ? IntermediateDFBReason::DFBInputMayBeReleased
                            : IntermediateDFBReason::RequiredDFBOperand;
      evidence.inputs = {value};
      evidence.observation = operation;
      state.requireMaterialization(operand, std::move(evidence));
    }
  });

  // A new materialization requirement changes the roots of dependent
  // expressions. Repeating both queries removes kernel walk-order dependence.
  // Each iteration adds only an existing operand, so the finite operand set
  // proves termination.
  size_t previousRequirementCount;
  do {
    previousRequirementCount = state.getRequirements().size();
    addComputeResultUseRequirements(state);
    kernel->walk([&](Operation *operation) {
      if (isElementwiseOp(operation)) {
        addExpressionReleaseRequirements(operation, lifetimes, state);
      }
    });
    std::optional<PlanningDiagnostic> diagnostic;
    WalkResult creationWalk = kernel->walk([&](Operation *operation) {
      if (failed(addCreationRequirements(operation, lifetimes, dominanceInfo,
                                         state, diagnostic))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (creationWalk.wasInterrupted()) {
      assert(diagnostic &&
             "failed creation requirement planning requires a diagnostic");
      return PlanningResult<IntermediateDFBPlan>::invalidIR(
          diagnostic->operation, std::move(diagnostic->message));
    }
  } while (state.getRequirements().size() != previousRequirementCount);

  SmallVector<IntermediateDFBRequirement> requirements =
      std::move(state).takeRequirements();
  DenseMap<Operation *, unsigned> operationOrder;
  unsigned nextOperationIndex = 0;
  kernel->walk([&](Operation *operation) {
    operationOrder.try_emplace(operation, nextOperationIndex++);
  });
  // Analysis iterations and SSA use lists do not define a semantic order.
  // Canonicalizing by kernel IR order makes diagnostics and compiler-created
  // DFB assignments stable when independent requirements are discovered in a
  // different order.
  llvm::stable_sort(requirements, [&](const IntermediateDFBRequirement &lhs,
                                      const IntermediateDFBRequirement &rhs) {
    unsigned lhsOrder = operationOrder.at(lhs.consumer);
    unsigned rhsOrder = operationOrder.at(rhs.consumer);
    return std::tie(lhsOrder, lhs.operandIndex) <
           std::tie(rhsOrder, rhs.operandIndex);
  });

  return buildMaterializationRecords(std::move(requirements));
}

} // namespace mlir::tt::ttl
