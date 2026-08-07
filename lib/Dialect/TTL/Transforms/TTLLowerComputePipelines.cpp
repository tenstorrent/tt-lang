// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Lower Compute Pipelines
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>
#include <utility>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLLOWERCOMPUTEPIPELINES
#define GEN_PASS_DEF_TTLLOWERSOURCESCALARSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct ComputeStageInliningPlan {
  SmallVector<BlockArgument> arguments;
  SmallVector<Value> inputs;
  SmallVector<Operation *> operations;
  SmallVector<OpResult> results;
  SmallVector<Value> yieldedValues;
};

struct ComputePipelineInliningPlan {
  ComputePipelineOp pipeline;
  std::optional<ComputePipelineSchedule> selectedSchedule;
  SmallVector<BlockArgument> arguments;
  SmallVector<Value> inputs;
  SmallVector<ComputeStageInliningPlan, 1> stages;
  SmallVector<Value> yieldedValues;
};

struct SourceScalarScopeInliningPlan {
  SourceScalarScopeOp scope;
  SmallVector<BlockArgument> producerArguments;
  SmallVector<Value> inputs;
  SmallVector<ComputeStageInliningPlan, 1> producerStages;
  Value retainedScalar;
  BlockArgument consumerScalarArgument;
  SmallVector<BlockArgument> consumerInputArguments;
  SmallVector<ComputeStageInliningPlan, 1> consumerStages;
  SmallVector<Value> yieldedValues;
};

static FailureOr<ComputeStageInliningPlan> analyzeStage(ComputeStageOp stage,
                                                        std::string &reason) {
  if (stage.getBody().getBlocks().size() != 1) {
    reason = "stage body must contain exactly one block";
    return failure();
  }
  Block &body = stage.getBody().front();
  auto stageYield = dyn_cast<ComputeStageYieldOp>(body.getTerminator());
  if (!stageYield || body.getNumArguments() != stage.getInputs().size() ||
      stageYield.getValues().size() != stage.getResults().size()) {
    reason = "stage body no longer matches its verified inputs and results";
    return failure();
  }

  ComputeStageInliningPlan plan;
  llvm::append_range(plan.arguments, body.getArguments());
  llvm::append_range(plan.inputs, stage.getInputs());
  for (Operation &operation : body.without_terminator()) {
    if (operation.getNumRegions() != 0) {
      reason = "stage body contains an operation with a nested region";
      return failure();
    }
    plan.operations.push_back(&operation);
  }
  llvm::append_range(plan.results, stage.getResults());
  llvm::append_range(plan.yieldedValues, stageYield.getValues());
  return plan;
}

static FailureOr<ComputePipelineInliningPlan>
analyzePipeline(ComputePipelineOp pipeline, std::string &reason) {
  if (pipeline.getBody().getBlocks().size() != 1) {
    reason = "pipeline body must contain exactly one block";
    return failure();
  }
  Block &body = pipeline.getBody().front();
  auto pipelineYield = dyn_cast<ComputePipelineYieldOp>(body.getTerminator());
  if (!pipelineYield || body.getNumArguments() != pipeline.getInputs().size() ||
      pipelineYield.getValues().size() != pipeline.getResults().size()) {
    reason = "pipeline body no longer matches its verified inputs and results";
    return failure();
  }

  ComputePipelineInliningPlan plan;
  plan.pipeline = pipeline;
  if (ComputePipelineScheduleAttr selected =
          pipeline.getSelectedScheduleAttr()) {
    plan.selectedSchedule = selected.getValue();
  }
  if (pipeline.getPipelineKindAttr() && !plan.selectedSchedule) {
    reason = "recognized pipeline has no selected schedule";
    return failure();
  }
  llvm::append_range(plan.arguments, body.getArguments());
  llvm::append_range(plan.inputs, pipeline.getInputs());
  DenseSet<Value> availableValues(body.getArguments().begin(),
                                  body.getArguments().end());
  for (Operation &operation : body.without_terminator()) {
    auto stage = dyn_cast<ComputeStageOp>(&operation);
    if (!stage) {
      reason = "pipeline body contains a non-stage operation";
      return failure();
    }
    if (llvm::any_of(stage.getInputs(), [&](Value input) {
          return !availableValues.contains(input);
        })) {
      reason =
          "stage input does not come from a pipeline input or preceding stage";
      return failure();
    }
    FailureOr<ComputeStageInliningPlan> stagePlan = analyzeStage(stage, reason);
    if (failed(stagePlan)) {
      return failure();
    }
    plan.stages.push_back(std::move(*stagePlan));
    availableValues.insert(stage.getResults().begin(),
                           stage.getResults().end());
  }
  if (llvm::any_of(pipelineYield.getValues(), [&](Value yieldedValue) {
        return !availableValues.contains(yieldedValue);
      })) {
    reason = "pipeline yield does not come from a planned stage";
    return failure();
  }
  llvm::append_range(plan.yieldedValues, pipelineYield.getValues());
  if (pipeline.getPipelineKindAttr() && plan.yieldedValues.size() != 1) {
    reason = "recognized pipeline must yield exactly one result";
    return failure();
  }
  return plan;
}

static FailureOr<SourceScalarScopeInliningPlan>
analyzeSourceScalarScope(SourceScalarScopeOp scope, std::string &reason) {
  if (scope.getProducer().getBlocks().size() != 1 ||
      scope.getConsumer().getBlocks().size() != 1) {
    reason = "source-scalar regions must each contain exactly one block";
    return failure();
  }
  Block &producerBody = scope.getProducer().front();
  Block &consumerBody = scope.getConsumer().front();
  auto producerYield =
      dyn_cast<SourceScalarYieldOp>(producerBody.getTerminator());
  auto consumerYield =
      dyn_cast<SourceScalarYieldOp>(consumerBody.getTerminator());
  if (!producerYield || !consumerYield ||
      producerBody.getNumArguments() != scope.getInputs().size() ||
      producerYield.getValues().size() != 1 ||
      consumerBody.getNumArguments() != scope.getInputs().size() + 1 ||
      consumerYield.getValues().size() != scope.getResults().size()) {
    reason = "source-scalar scope no longer matches its verified signature";
    return failure();
  }

  SourceScalarScopeInliningPlan plan;
  plan.scope = scope;
  llvm::append_range(plan.producerArguments, producerBody.getArguments());
  llvm::append_range(plan.inputs, scope.getInputs());
  plan.retainedScalar = producerYield.getValues().front();
  plan.consumerScalarArgument = consumerBody.getArgument(0);
  llvm::append_range(plan.consumerInputArguments,
                     consumerBody.getArguments().drop_front());
  for (Operation &operation : producerBody.without_terminator()) {
    auto stage = dyn_cast<ComputeStageOp>(&operation);
    if (!stage) {
      reason = "source-scalar producer contains a non-stage operation";
      return failure();
    }
    FailureOr<ComputeStageInliningPlan> stagePlan = analyzeStage(stage, reason);
    if (failed(stagePlan)) {
      return failure();
    }
    plan.producerStages.push_back(std::move(*stagePlan));
  }
  for (Operation &operation : consumerBody.without_terminator()) {
    auto stage = dyn_cast<ComputeStageOp>(&operation);
    if (!stage) {
      reason = "source-scalar consumer contains a non-stage operation";
      return failure();
    }
    FailureOr<ComputeStageInliningPlan> stagePlan = analyzeStage(stage, reason);
    if (failed(stagePlan)) {
      return failure();
    }
    plan.consumerStages.push_back(std::move(*stagePlan));
  }
  llvm::append_range(plan.yieldedValues, consumerYield.getValues());
  return plan;
}

static void inlineStages(ArrayRef<ComputeStageInliningPlan> stagePlans,
                         IRMapping &mapping, IRRewriter &rewriter) {
  for (const ComputeStageInliningPlan &stagePlan : stagePlans) {
    for (auto [argument, input] :
         llvm::zip_equal(stagePlan.arguments, stagePlan.inputs)) {
      mapping.map(argument, mapping.lookup(input));
    }
    for (Operation *operation : stagePlan.operations) {
      rewriter.clone(*operation, mapping);
    }
    for (auto [result, yieldedValue] :
         llvm::zip_equal(stagePlan.results, stagePlan.yieldedValues)) {
      mapping.map(result, mapping.lookup(yieldedValue));
    }
  }
}

static void applyPipelinePlan(const ComputePipelineInliningPlan &plan,
                              IRRewriter &rewriter) {
  IRMapping mapping;
  for (auto [argument, input] : llvm::zip_equal(plan.arguments, plan.inputs)) {
    mapping.map(argument, input);
  }

  rewriter.setInsertionPoint(plan.pipeline);
  inlineStages(plan.stages, mapping, rewriter);

  SmallVector<Value> replacements;
  replacements.reserve(plan.yieldedValues.size());
  for (Value yieldedValue : plan.yieldedValues) {
    replacements.push_back(mapping.lookup(yieldedValue));
  }
  if (plan.selectedSchedule) {
    Operation *selectedSource = replacements.front().getDefiningOp();
    assert(selectedSource &&
           "recognized pipeline result must have a defining operation");
    selectedSource->setAttr(kSelectedComputePipelineScheduleAttrName,
                            ComputePipelineScheduleAttr::get(
                                rewriter.getContext(), *plan.selectedSchedule));
  }
  rewriter.replaceOp(plan.pipeline, replacements);
}

static void
applySourceScalarScopePlan(const SourceScalarScopeInliningPlan &plan,
                           IRRewriter &rewriter) {
  IRMapping mapping;
  for (auto [argument, input] :
       llvm::zip_equal(plan.producerArguments, plan.inputs)) {
    mapping.map(argument, input);
  }

  rewriter.setInsertionPoint(plan.scope);
  inlineStages(plan.producerStages, mapping, rewriter);
  Value retainedScalar = mapping.lookup(plan.retainedScalar);
  mapping.map(plan.consumerScalarArgument, retainedScalar);
  for (auto [argument, input] :
       llvm::zip_equal(plan.consumerInputArguments, plan.inputs)) {
    mapping.map(argument, input);
  }
  inlineStages(plan.consumerStages, mapping, rewriter);

  SmallVector<Value> replacements;
  replacements.reserve(plan.yieldedValues.size());
  for (Value yieldedValue : plan.yieldedValues) {
    Value replacement = mapping.lookup(yieldedValue);
    Operation *producer = replacement.getDefiningOp();
    assert(producer && "source-scalar result must have a defining operation");
    producer->setAttr(
        kSelectedComputePipelineScheduleAttrName,
        ComputePipelineScheduleAttr::get(
            rewriter.getContext(), ComputePipelineSchedule::RetainedScalar));
    replacements.push_back(replacement);
  }
  rewriter.replaceOp(plan.scope, replacements);
}

class TTLLowerComputePipelinesPass
    : public impl::TTLLowerComputePipelinesBase<TTLLowerComputePipelinesPass> {
public:
  using impl::TTLLowerComputePipelinesBase<
      TTLLowerComputePipelinesPass>::TTLLowerComputePipelinesBase;

  void runOnOperation() override {
    SmallVector<ComputePipelineInliningPlan, 1> plans;
    std::optional<std::pair<ComputePipelineOp, std::string>> invalidPipeline;
    getOperation().walk([&](ComputePipelineOp pipeline) {
      std::string reason;
      FailureOr<ComputePipelineInliningPlan> plan =
          analyzePipeline(pipeline, reason);
      if (failed(plan)) {
        invalidPipeline = std::make_pair(pipeline, std::move(reason));
        return WalkResult::interrupt();
      }
      plans.push_back(std::move(*plan));
      return WalkResult::advance();
    });
    if (invalidPipeline) {
      invalidPipeline->first.emitOpError(invalidPipeline->second);
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    for (const ComputePipelineInliningPlan &plan : llvm::reverse(plans)) {
      applyPipelinePlan(plan, rewriter);
    }
  }
};

class TTLLowerSourceScalarScopesPass
    : public impl::TTLLowerSourceScalarScopesBase<
          TTLLowerSourceScalarScopesPass> {
public:
  using impl::TTLLowerSourceScalarScopesBase<
      TTLLowerSourceScalarScopesPass>::TTLLowerSourceScalarScopesBase;

  void runOnOperation() override {
    SmallVector<SourceScalarScopeInliningPlan, 1> plans;
    std::optional<std::pair<SourceScalarScopeOp, std::string>> invalidScope;
    getOperation().walk([&](SourceScalarScopeOp scope) {
      std::string reason;
      FailureOr<SourceScalarScopeInliningPlan> plan =
          analyzeSourceScalarScope(scope, reason);
      if (failed(plan)) {
        invalidScope = std::make_pair(scope, std::move(reason));
        return WalkResult::interrupt();
      }
      plans.push_back(std::move(*plan));
      return WalkResult::advance();
    });
    if (invalidScope) {
      invalidScope->first.emitOpError(invalidScope->second);
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    for (const SourceScalarScopeInliningPlan &plan : llvm::reverse(plans)) {
      applySourceScalarScopePlan(plan, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
