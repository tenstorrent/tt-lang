// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/ComputeKernelConfigAnalysis.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/ComputeTarget.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Remarks.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <variant>

namespace mlir::tt::ttl {
namespace {

/// Parse a three-state pass option and its accepted boolean spellings.
FailureOr<ConfigSelection> parseConfigSelection(Operation *diagnosticOp,
                                                StringRef optionName,
                                                StringRef value) {
  if (value == "auto") {
    return ConfigSelection::Auto;
  }
  if (value == "enabled" || value == "true" || value == "1") {
    return ConfigSelection::Enabled;
  }
  if (value == "disabled" || value == "false" || value == "0") {
    return ConfigSelection::Disabled;
  }
  diagnosticOp->emitError() << "invalid " << optionName << " value '" << value
                            << "'; expected auto, enabled, or disabled";
  return failure();
}

/// Read and type-check an optional function-level boolean constraint.
FailureOr<std::optional<bool>>
getOptionalBoolConstraint(func::FuncOp function, StringRef attributeName) {
  Attribute rawAttribute = function->getAttr(attributeName);
  if (!rawAttribute) {
    return std::optional<bool>();
  }
  auto boolAttribute = dyn_cast<BoolAttr>(rawAttribute);
  if (!boolAttribute) {
    function.emitOpError() << attributeName << " must be a boolean attribute";
    return failure();
  }
  return std::optional<bool>(boolAttribute.getValue());
}

/// Read an exact per-DFB unpack configuration from the function.
FailureOr<std::optional<SmallVector<int32_t>>>
getOptionalUnpackConstraint(func::FuncOp function) {
  Attribute rawAttribute = function->getAttr(kUnpackToDestFp32AttrName);
  if (!rawAttribute) {
    return std::optional<SmallVector<int32_t>>();
  }
  auto unpackAttribute = dyn_cast<DenseI32ArrayAttr>(rawAttribute);
  if (!unpackAttribute) {
    function.emitOpError() << kUnpackToDestFp32AttrName
                           << " must be a dense i32 array attribute";
    return failure();
  }

  SmallVector<int32_t> dataflowBufferIndices(unpackAttribute.asArrayRef());
  if (llvm::any_of(dataflowBufferIndices, [](int32_t index) {
        return index < 0 || index >= kMaxCircularBuffers;
      })) {
    function.emitOpError()
        << kUnpackToDestFp32AttrName
        << " must contain dataflow buffer indices in range [0, "
        << kMaxCircularBuffers - 1 << "]";
    return failure();
  }
  llvm::sort(dataflowBufferIndices);
  dataflowBufferIndices.erase(
      std::unique(dataflowBufferIndices.begin(), dataflowBufferIndices.end()),
      dataflowBufferIndices.end());
  return std::optional<SmallVector<int32_t>>(std::move(dataflowBufferIndices));
}

/// Return the attached DFB index, or `std::nullopt` when no DFB is attached.
FailureOr<std::optional<int64_t>>
resolveDataflowBufferIndex(Value value, Operation *consumer) {
  if (auto blockArgument = dyn_cast<BlockArgument>(value)) {
    auto computeOp =
        dyn_cast_or_null<ComputeOp>(blockArgument.getOwner()->getParentOp());
    if (!computeOp) {
      return std::optional<int64_t>();
    }
    unsigned argumentNumber = blockArgument.getArgNumber();
    if (argumentNumber < computeOp.getNumInputs()) {
      value = computeOp.getInputs()[argumentNumber];
    } else {
      unsigned outputNumber = argumentNumber - computeOp.getNumInputs();
      if (outputNumber >= computeOp.getNumOutputs()) {
        return std::optional<int64_t>();
      }
      value = computeOp.getOutputs()[outputNumber];
    }
  }

  Value dfb = getAttachedCB(value);
  if (dfb) {
    std::optional<int64_t> dfbIndex = getCBIndex(dfb);
    if (!dfbIndex) {
      consumer->emitOpError("uses a dataflow buffer without a finalized index");
      return failure();
    }
    if (*dfbIndex < 0 || *dfbIndex >= kMaxCircularBuffers) {
      consumer->emitOpError() << "uses dataflow buffer index " << *dfbIndex
                              << " outside the supported range [0, "
                              << kMaxCircularBuffers - 1 << "]";
      return failure();
    }
    return dfbIndex;
  }

  return std::optional<int64_t>();
}

/// Return the scalar element type of a tile or tensor of tiles.
FailureOr<Type> getRequiredTileElementType(Value value,
                                           Operation *diagnosticOp) {
  Type valueType = value.getType();
  auto tileType = dyn_cast<ttcore::TileType>(valueType);
  if (!tileType) {
    if (auto tensorType = dyn_cast<TensorType>(valueType)) {
      tileType = dyn_cast<ttcore::TileType>(tensorType.getElementType());
    }
  }
  if (!tileType) {
    diagnosticOp->emitOpError()
        << "expected a tile or tensor-of-tiles operand, got "
        << value.getType();
    return failure();
  }
  return tileType.getElementType();
}

/// Return whether a tile element occupies a full 32-bit DST slot.
bool requires32BitDestination(Type elementType) {
  return ttcore::getNumberOfBits(ttcore::elementTypeToDataType(elementType)) ==
         32;
}

/// Return a dataflow-buffer operand that the strategy cannot address.
FailureOr<std::optional<unsigned>>
findUnavailableDataflowBufferOperand(Operation *operation,
                                     const TileExecutionInfo &info) {
  for (OpOperand &operand : operation->getOpOperands()) {
    if (info.operandRoutes[operand.getOperandNumber()] !=
        TileOperandRoute::DataflowBuffer) {
      continue;
    }
    FailureOr<std::optional<int64_t>> dataflowBufferIndex =
        resolveDataflowBufferIndex(operand.get(), operation);
    if (failed(dataflowBufferIndex)) {
      return failure();
    }
    if (!*dataflowBufferIndex) {
      return std::optional<unsigned>(operand.getOperandNumber());
    }
  }
  return std::optional<unsigned>();
}

/// Append the configuration requirements imposed by one execution option.
LogicalResult
appendExecutionRequirements(Operation *operation, const TileExecutionInfo &info,
                            SmallVectorImpl<DFBInputUse> &dfbInputUses,
                            SmallVectorImpl<DestinationUse> &destinationUses) {
  for (OpOperand &operand : operation->getOpOperands()) {
    TileOperandRoute route = info.operandRoutes[operand.getOperandNumber()];
    if (route == TileOperandRoute::None) {
      continue;
    }
    FailureOr<Type> elementType =
        getRequiredTileElementType(operand.get(), operation);
    if (failed(elementType)) {
      return failure();
    }
    FailureOr<std::optional<int64_t>> dataflowBufferIndex =
        resolveDataflowBufferIndex(operand.get(), operation);
    if (failed(dataflowBufferIndex)) {
      return failure();
    }
    if (route == TileOperandRoute::DataflowBuffer && !*dataflowBufferIndex) {
      operation->emitOpError()
          << "operand " << operand.getOperandNumber()
          << " must resolve to a dataflow buffer for the selected strategy";
      return failure();
    }
    if (*dataflowBufferIndex) {
      dfbInputUses.push_back({**dataflowBufferIndex, operation,
                              operand.getOperandNumber(), info.primitive, route,
                              *elementType});
    }
    if (route == TileOperandRoute::Dst) {
      destinationUses.push_back({operation, info.primitive, *elementType});
    }
  }

  if (!info.resultInDst) {
    return success();
  }
  for (Value result : operation->getResults()) {
    Type resultType = result.getType();
    auto tileType = dyn_cast<ttcore::TileType>(resultType);
    if (!tileType) {
      if (auto tensorType = dyn_cast<TensorType>(resultType)) {
        tileType = dyn_cast<ttcore::TileType>(tensorType.getElementType());
      }
    }
    if (!tileType) {
      continue;
    }
    destinationUses.push_back(
        {operation, info.primitive, tileType.getElementType()});
  }
  return success();
}

/// Return strategy alternatives and their complete configuration requirements.
FailureOr<TileExecutionChoice>
getTileExecutionChoice(TileExecutionOpInterface executionOp) {
  Operation *operation = executionOp.getOperation();
  SmallVector<TileExecutionStrategy, 2> legalStrategies =
      executionOp.getLegalExecutionStrategies();
  assert(!legalStrategies.empty() &&
         "strategy choice requires at least one legal alternative");

  if (failed(verifyTileExecutionStrategy(operation, legalStrategies))) {
    return failure();
  }
  auto selectedAttr = operation->getAttrOfType<TileExecutionStrategyAttr>(
      kTileExecutionStrategyAttrName);

  if (selectedAttr) {
    legalStrategies.assign(1, selectedAttr.getValue());
  }

  SmallVector<TileExecutionOption, 2> options;
  for (TileExecutionStrategy strategy : legalStrategies) {
    FailureOr<TileExecutionInfo> info =
        executionOp.getTileExecutionInfo(strategy);
    if (failed(info)) {
      operation->emitOpError("has no semantics for a legal tile strategy");
      return failure();
    }
    if (failed(verifyTileExecutionInfo(operation, *info))) {
      return failure();
    }
    FailureOr<std::optional<unsigned>> unavailableOperand =
        findUnavailableDataflowBufferOperand(operation, *info);
    if (failed(unavailableOperand)) {
      return failure();
    }
    if (*unavailableOperand) {
      if (selectedAttr) {
        operation->emitOpError()
            << "operand " << **unavailableOperand
            << " does not resolve to a dataflow buffer required by explicit "
            << kTileExecutionStrategyAttrName;
        return failure();
      }
      continue;
    }

    TileExecutionOption option{strategy, {}, {}};
    if (failed(appendExecutionRequirements(
            operation, *info, option.dfbInputUses, option.destinationUses))) {
      return failure();
    }
    options.push_back(std::move(option));
  }
  if (options.empty()) {
    operation->emitOpError("has no legal tile execution strategy");
    return failure();
  }
  return TileExecutionChoice{operation, std::move(options),
                             static_cast<bool>(selectedAttr)};
}

static bool isFullScalarType(RankedTensorType type) {
  return type.hasStaticShape() &&
         llvm::all_of(type.getShape(),
                      [](int64_t dimension) { return dimension == 1; });
}

static bool isFullScalarMap(AffineMap map) {
  return llvm::all_of(map.getResults(), [](AffineExpr expression) {
    auto constant = dyn_cast<AffineConstantExpr>(expression);
    return constant && constant.getValue() == 0;
  });
}

/// Find a producer-consumer split whose only cross-boundary stage result is a
/// full scalar consumed through zero-indexed operand maps.
static std::optional<SourceScalarRetentionPlan>
buildSourceScalarRetentionPlan(ComputePipelineOp pipeline) {
  Block &body = pipeline.getBody().front();
  SmallVector<ComputeStageOp> stages(body.getOps<ComputeStageOp>().begin(),
                                     body.getOps<ComputeStageOp>().end());
  auto getStageIndex = [&](Operation *operation) -> std::optional<unsigned> {
    auto stage = dyn_cast_or_null<ComputeStageOp>(operation);
    if (!stage) {
      return std::nullopt;
    }
    auto iterator = llvm::find(stages, stage);
    if (iterator == stages.end()) {
      return std::nullopt;
    }
    return static_cast<unsigned>(std::distance(stages.begin(), iterator));
  };

  auto pipelineYield = cast<ComputePipelineYieldOp>(body.getTerminator());
  for (unsigned producerIndex = static_cast<unsigned>(stages.size());
       producerIndex > 0; --producerIndex) {
    ComputeStageOp producerStage = stages[producerIndex - 1];
    for (OpResult scalarResult : llvm::reverse(producerStage.getResults())) {
      auto scalarType = dyn_cast<RankedTensorType>(scalarResult.getType());
      if (!scalarType || !isFullScalarType(scalarType)) {
        continue;
      }

      SourceScalarRetentionPlan plan;
      plan.producerStage = producerStage;
      plan.producerResult = scalarResult.getResultNumber();
      bool validCandidate = true;
      for (OpOperand &use : scalarResult.getUses()) {
        auto consumerStage = dyn_cast<ComputeStageOp>(use.getOwner());
        std::optional<unsigned> consumerIndex = getStageIndex(use.getOwner());
        if (!consumerStage || !consumerIndex ||
            *consumerIndex < producerIndex ||
            use.getOperandNumber() >= consumerStage.getInputs().size()) {
          validCandidate = false;
          break;
        }
        auto map = cast<AffineMapAttr>(
                       consumerStage.getIndexingMaps()[use.getOperandNumber()])
                       .getValue();
        if (!isFullScalarMap(map)) {
          validCandidate = false;
          break;
        }
        plan.consumers.push_back({consumerStage, use.getOperandNumber()});
      }
      if (!validCandidate || plan.consumers.empty()) {
        continue;
      }

      for (unsigned consumerIndex = producerIndex;
           consumerIndex < stages.size() && validCandidate; ++consumerIndex) {
        for (Value input : stages[consumerIndex].getInputs()) {
          if (input == scalarResult || isa<BlockArgument>(input)) {
            continue;
          }
          auto result = dyn_cast<OpResult>(input);
          std::optional<unsigned> definingStageIndex =
              result ? getStageIndex(result.getOwner()) : std::nullopt;
          if (!definingStageIndex || *definingStageIndex < producerIndex) {
            validCandidate = false;
            break;
          }
        }
      }
      for (Value yieldedValue : pipelineYield.getValues()) {
        auto result = dyn_cast<OpResult>(yieldedValue);
        std::optional<unsigned> definingStageIndex =
            result ? getStageIndex(result.getOwner()) : std::nullopt;
        if (!definingStageIndex || *definingStageIndex < producerIndex) {
          validCandidate = false;
          break;
        }
      }
      if (!validCandidate) {
        continue;
      }

      for (unsigned stageIndex = 0; stageIndex < producerIndex; ++stageIndex) {
        plan.producerStages.push_back(
            {stages[stageIndex],
             llvm::to_vector(stages[stageIndex].getInputs())});
      }
      for (unsigned stageIndex = producerIndex; stageIndex < stages.size();
           ++stageIndex) {
        plan.consumerStages.push_back(
            {stages[stageIndex],
             llvm::to_vector(stages[stageIndex].getInputs())});
      }
      return plan;
    }
  }
  return std::nullopt;
}

/// Collect target-independent alternatives for a recognized semantic
/// pipeline. The retained-scalar option names only resources that exist before
/// ordinary materialization creates its internal DFBs.
FailureOr<ComputePipelineScheduleChoice>
getComputePipelineScheduleChoice(ComputePipelineOp pipeline) {
  ComputePipelineKindAttr pipelineKind = pipeline.getPipelineKindAttr();
  if (!pipelineKind) {
    pipeline.emitOpError("has no configuration semantics for pipeline_kind");
    return failure();
  }
  if (pipeline.getInputs().empty()) {
    pipeline.emitOpError("recognized pipeline requires at least one input");
    return failure();
  }

  auto inputType =
      dyn_cast<RankedTensorType>(pipeline.getInputs().front().getType());
  if (!inputType || !inputType.hasStaticShape() || inputType.getRank() != 2 ||
      inputType.getNumElements() < 1 ||
      static_cast<std::uint64_t>(inputType.getNumElements()) >
          std::numeric_limits<std::uint32_t>::max()) {
    pipeline.emitOpError(
        "recognized pipeline requires a non-empty static rank-2 input");
    return failure();
  }
  FailureOr<Type> elementType =
      getRequiredTileElementType(pipeline.getInputs().front(), pipeline);
  if (failed(elementType)) {
    return failure();
  }

  TilePrimitive retainedPrimitive;
  switch (pipelineKind.getValue()) {
  case ComputePipelineKind::MultiplyFullScalarReduction:
    retainedPrimitive = TilePrimitive::MultiplyFullScalarReduction;
    break;
  case ComputePipelineKind::RowNormalization:
    retainedPrimitive = TilePrimitive::RowNormalization;
    break;
  }

  ComputePipelineScheduleOption retained{
      pipelineKind.getValue(),
      ComputePipelineSchedule::RetainedScalar,
      {},
      {},
      static_cast<std::uint32_t>(inputType.getNumElements()),
      nullptr};
  std::optional<SourceScalarRetentionPlan> sourceScalar =
      buildSourceScalarRetentionPlan(pipeline);
  if (sourceScalar) {
    retained.sourceScalar = std::make_shared<const SourceScalarRetentionPlan>(
        std::move(*sourceScalar));
  }
  if (pipelineKind.getValue() == ComputePipelineKind::RowNormalization &&
      !retained.sourceScalar) {
    pipeline.emitOpError(
        "row_normalization retained schedule requires one full-scalar "
        "producer-consumer lifetime");
    return failure();
  }
  for (auto [operandIndex, input] : llvm::enumerate(pipeline.getInputs())) {
    auto currentType = dyn_cast<RankedTensorType>(input.getType());
    FailureOr<Type> currentElementType =
        getRequiredTileElementType(input, pipeline);
    if (!currentType || currentType != inputType ||
        failed(currentElementType) || *currentElementType != *elementType) {
      pipeline.emitOpError(
          "recognized pipeline inputs must have one tensor type");
      return failure();
    }
    FailureOr<std::optional<int64_t>> dfbIndex =
        resolveDataflowBufferIndex(input, pipeline);
    if (failed(dfbIndex)) {
      return failure();
    }
    if (!*dfbIndex) {
      pipeline.emitOpError("recognized pipeline input must be DFB-backed");
      return failure();
    }
    retained.dfbInputUses.push_back(
        {**dfbIndex, pipeline, static_cast<unsigned>(operandIndex),
         retainedPrimitive, TileOperandRoute::DataflowBuffer, *elementType});
  }
  retained.destinationUses.push_back(
      {pipeline, retainedPrimitive, *elementType});

  ComputePipelineScheduleOption materialized{
      pipelineKind.getValue(),
      ComputePipelineSchedule::Materialized,
      {},
      {},
      0,
      nullptr};

  SmallVector<ComputePipelineScheduleOption, 2> options;
  ComputePipelineScheduleAttr selected = pipeline.getSelectedScheduleAttr();
  if (!selected ||
      selected.getValue() == ComputePipelineSchedule::RetainedScalar) {
    options.push_back(std::move(retained));
  }
  if (!selected ||
      selected.getValue() == ComputePipelineSchedule::Materialized) {
    options.push_back(std::move(materialized));
  }
  return ComputePipelineScheduleChoice{pipeline, std::move(options)};
}

struct DestinationWidthEvidence {
  Operation *operation;
  std::optional<DestinationUse> use;
};

/// Diagnose the two constraints that eliminated every destination width.
void emitDestinationWidthConflict(
    func::FuncOp function, const DestinationWidthEvidence &requires32Bits,
    const DestinationWidthEvidence &requires16Bits) {
  if (!requires16Bits.use) {
    requires32Bits.operation->emitOpError(
        "requires 32-bit destination elements, but fp32 destination "
        "accumulation is explicitly disabled");
    return;
  }

  InFlightDiagnostic diagnostic =
      requires32Bits.use
          ? requires32Bits.operation->emitOpError(
                "requires 32-bit destination elements, but no kernel-wide "
                "destination width supports all tile operations")
          : function.emitOpError(
                "explicit 32-bit destination elements are unsupported by the "
                "kernel's tile operations");
  diagnostic.attachNote(requires16Bits.operation->getLoc())
      << "the target does not support 32-bit destination elements for "
      << requires16Bits.operation->getName() << " with "
      << requires16Bits.use->elementType << " elements";
}

struct DestinationWidthConflict {
  DestinationWidthEvidence requires32Bits;
  DestinationWidthEvidence requires16Bits;
};

struct ExplicitUnpackConflict {
  DFBInputUse use;
  bool configuresUnpackToDestination;
};

struct DFBUnpackConflict {
  DFBInputUse firstUse;
  DFBInputUse secondUse;
};

struct UnsupportedDFBConfiguration {
  DFBInputUse use;
};

struct UnsupportedDestinationConfiguration {
  DestinationUse use;
};

struct UnsupportedComputePipelineSchedule {
  Operation *pipeline;
};

struct ComputePipelineCapacityConflict {
  Operation *pipeline;
  std::uint32_t requiredDstSlots;
  std::uint32_t availableDstSlots;
};

using ConfigConstraintConflict = std::variant<
    DestinationWidthConflict, ExplicitUnpackConflict, DFBUnpackConflict,
    UnsupportedDFBConfiguration, UnsupportedDestinationConfiguration,
    UnsupportedComputePipelineSchedule, ComputePipelineCapacityConflict>;

struct ConfigurationCandidate {
  ConfigurationCandidate(DestinationElementWidth width, DstSyncMode syncMode)
      : destinationElementWidth(width), dstSyncMode(syncMode) {}

  DestinationElementWidth destinationElementWidth;
  DstSyncMode dstSyncMode;
  llvm::MapVector<int64_t, llvm::SmallSet<DFBUnpackMode, 2>> unpackModes;
  llvm::DenseMap<int64_t, DFBInputUse> firstUnpackUses;
};

struct ConfigConstraintState {
  ConfigConstraintState() {
    for (DestinationElementWidth width :
         {DestinationElementWidth::Bits16, DestinationElementWidth::Bits32}) {
      candidates.emplace_back(width, DstSyncMode::DoubleBuffered);
      candidates.emplace_back(width, DstSyncMode::Full);
    }
  }

  llvm::SmallVector<ConfigurationCandidate, 2> candidates;
  DestinationWidthEvidence requires32Bits{nullptr, std::nullopt};
  DestinationWidthEvidence requires16Bits{nullptr, std::nullopt};
};

using ConfigConstraintResult =
    std::variant<ConfigConstraintState, ConfigConstraintConflict>;

/// Intersect one execution option's requirements with the current domains.
ConfigConstraintResult applyConfigConstraints(
    ConfigConstraintState state, const KernelTargetEnvironment &target,
    const KernelConfigPolicy &policy, ArrayRef<DFBInputUse> dfbInputUses,
    ArrayRef<DestinationUse> destinationUses) {
  for (const DestinationUse &use : destinationUses) {
    bool supportsBits16 = target.supportsDestinationElementWidth(
        use.primitive, use.elementType, DestinationElementWidth::Bits16);
    bool supportsBits32 = target.supportsDestinationElementWidth(
        use.primitive, use.elementType, DestinationElementWidth::Bits32);
    if (!supportsBits16 && !supportsBits32) {
      return ConfigConstraintConflict(UnsupportedDestinationConfiguration{use});
    }
    if (requires32BitDestination(use.elementType)) {
      if (!state.requires32Bits.operation) {
        state.requires32Bits = {use.operation, use};
      }
    }
    if (!supportsBits32) {
      if (!state.requires16Bits.operation) {
        state.requires16Bits = {use.operation, use};
      }
    }
    llvm::erase_if(state.candidates,
                   [&](const ConfigurationCandidate &candidate) {
                     return (requires32BitDestination(use.elementType) &&
                             candidate.destinationElementWidth ==
                                 DestinationElementWidth::Bits16) ||
                            (candidate.destinationElementWidth ==
                                     DestinationElementWidth::Bits16
                                 ? !supportsBits16
                                 : !supportsBits32);
                   });
    if (state.candidates.empty()) {
      return ConfigConstraintConflict(
          DestinationWidthConflict{state.requires32Bits, state.requires16Bits});
    }
  }

  for (const DFBInputUse &use : dfbInputUses) {
    llvm::SmallVector<DFBHardwareConfiguration, 4> supportedConfigurations =
        target.getSupportedDFBConfigurations(use.primitive, use.route,
                                             use.elementType);
    bool hasExplicitConfiguration = policy.unpackToDestFp32.has_value();
    bool configuresUnpackToDestination =
        hasExplicitConfiguration &&
        llvm::is_contained(*policy.unpackToDestFp32, use.dfbIndex);
    std::optional<ConfigConstraintConflict> firstConflict;

    for (auto candidateIterator = state.candidates.begin();
         candidateIterator != state.candidates.end();) {
      ConfigurationCandidate &candidate = *candidateIterator;
      llvm::SmallSet<DFBUnpackMode, 2> allowedModes;
      for (const DFBHardwareConfiguration &configuration :
           supportedConfigurations) {
        if (configuration.destinationElementWidth ==
            candidate.destinationElementWidth) {
          allowedModes.insert(configuration.unpackMode);
        }
      }

      // ComputeConfigDescriptor::unpack_to_dest_mode changes the route only
      // for f32. For other formats, tt-metal ignores an enabled entry.
      if (hasExplicitConfiguration && use.elementType.isF32()) {
        DFBUnpackMode configuredMode = configuresUnpackToDestination
                                           ? DFBUnpackMode::UnpackToDestination
                                           : DFBUnpackMode::Default;
        bool supportsConfiguredMode = allowedModes.contains(configuredMode);
        allowedModes.clear();
        if (supportsConfiguredMode) {
          allowedModes.insert(configuredMode);
        }
      }

      auto modeIterator = candidate.unpackModes.find(use.dfbIndex);
      if (modeIterator != candidate.unpackModes.end()) {
        llvm::SmallSet<DFBUnpackMode, 2> intersection;
        for (DFBUnpackMode mode : modeIterator->second) {
          if (allowedModes.contains(mode)) {
            intersection.insert(mode);
          }
        }
        allowedModes = std::move(intersection);
      }

      if (allowedModes.empty()) {
        if (!firstConflict) {
          if (hasExplicitConfiguration && use.elementType.isF32()) {
            firstConflict =
                ExplicitUnpackConflict{use, configuresUnpackToDestination};
          } else if (modeIterator != candidate.unpackModes.end()) {
            firstConflict = DFBUnpackConflict{
                candidate.firstUnpackUses.lookup(use.dfbIndex), use};
          } else {
            firstConflict = UnsupportedDFBConfiguration{use};
          }
        }
        candidateIterator = state.candidates.erase(candidateIterator);
        continue;
      }

      if (modeIterator == candidate.unpackModes.end()) {
        candidate.unpackModes.insert({use.dfbIndex, std::move(allowedModes)});
        candidate.firstUnpackUses.insert({use.dfbIndex, use});
      } else {
        modeIterator->second = std::move(allowedModes);
      }
      ++candidateIterator;
    }

    if (state.candidates.empty()) {
      assert(firstConflict && "an eliminated candidate must retain evidence");
      return *firstConflict;
    }
  }
  return state;
}

/// Emit the constraint evidence retained by an unsuccessful resolution.
void emitConfigConstraintConflict(func::FuncOp function,
                                  const ConfigConstraintConflict &conflict) {
  std::visit(
      [&](const auto &typedConflict) {
        using ConflictType = std::decay_t<decltype(typedConflict)>;
        if constexpr (std::is_same_v<ConflictType, DestinationWidthConflict>) {
          emitDestinationWidthConflict(function, typedConflict.requires32Bits,
                                       typedConflict.requires16Bits);
        } else if constexpr (std::is_same_v<ConflictType,
                                            ExplicitUnpackConflict>) {
          const DFBInputUse &use = typedConflict.use;
          use.consumer->emitOpError()
              << "dataflow buffer " << use.dfbIndex << " requires "
              << (typedConflict.configuresUnpackToDestination
                      ? "default unpack mode, but "
                      : "unpack-to-DST-f32 mode, but ")
              << kUnpackToDestFp32AttrName
              << (typedConflict.configuresUnpackToDestination
                      ? " includes this index"
                      : " excludes this index");
        } else if constexpr (std::is_same_v<ConflictType, DFBUnpackConflict>) {
          InFlightDiagnostic diagnostic =
              typedConflict.secondUse.consumer->emitOpError()
              << "dataflow buffer " << typedConflict.secondUse.dfbIndex
              << " requires incompatible unpack modes in one kernel";
          diagnostic.attachNote(typedConflict.firstUse.consumer->getLoc())
              << "operand " << typedConflict.firstUse.operandIndex
              << " establishes the conflicting unpack mode";
        } else if constexpr (std::is_same_v<ConflictType,
                                            UnsupportedDFBConfiguration>) {
          typedConflict.use.consumer->emitOpError()
              << "dataflow buffer " << typedConflict.use.dfbIndex
              << " has no target-supported destination-width and unpack-route "
                 "configuration";
        } else if constexpr (std::is_same_v<
                                 ConflictType,
                                 UnsupportedDestinationConfiguration>) {
          typedConflict.use.operation->emitOpError()
              << "has no target-supported destination element width for "
              << typedConflict.use.elementType << " elements";
        } else if constexpr (std::is_same_v<
                                 ConflictType,
                                 UnsupportedComputePipelineSchedule>) {
          typedConflict.pipeline->emitOpError(
              "selected compute-pipeline schedule is unsupported by the "
              "target");
        } else {
          typedConflict.pipeline->emitOpError()
              << "selected compute-pipeline schedule requires "
              << typedConflict.requiredDstSlots << " DST slots, but at most "
              << typedConflict.availableDstSlots
              << " are available under the kernel configuration";
        }
      },
      conflict);
}

enum class ExecutionChoiceKind {
  TileStrategy,
  ComputePipelineSchedule,
};

struct KernelExecutionOption {
  llvm::SmallVector<DFBInputUse> dfbInputUses;
  llvm::SmallVector<DestinationUse> destinationUses;
  std::optional<TileExecutionStrategy> tileStrategy;
  std::optional<ComputePipelineKind> pipelineKind;
  std::optional<ComputePipelineSchedule> pipelineSchedule;
  Type pipelineElementType;
  std::uint32_t requiredDstSlots = 0;
  SourceScalarRetentionPlanPtr sourceScalar;
};

struct KernelExecutionChoiceOptions {
  Operation *operation;
  ExecutionChoiceKind kind;
  SmallVector<KernelExecutionOption, 2> options;
};

using KernelExecutionOptions = SmallVector<KernelExecutionChoiceOptions, 0>;

/// Apply execution policy without changing the collected requirements.
FailureOr<KernelExecutionOptions>
getKernelExecutionOptions(const KernelRequirements &requirements,
                          const KernelConfigPolicy &policy) {
  KernelExecutionOptions allOptions;
  allOptions.reserve(requirements.tileStrategyChoices.size() +
                     requirements.pipelineScheduleChoices.size());
  for (const TileExecutionChoice &choice : requirements.tileStrategyChoices) {
    SmallVector<KernelExecutionOption, 2> choiceOptions;
    for (const TileExecutionOption &option : choice.options) {
      if (option.strategy == TileExecutionStrategy::FPU &&
          !policy.allowFPUBinary) {
        continue;
      }
      choiceOptions.push_back({option.dfbInputUses,
                               option.destinationUses,
                               option.strategy,
                               std::nullopt,
                               std::nullopt,
                               {},
                               0,
                               nullptr});
    }
    if (choiceOptions.empty()) {
      if (choice.hasExplicitStrategy) {
        choice.operation->emitOpError(
            "explicit FPU strategy conflicts with disabled FPU binary policy");
      } else {
        choice.operation->emitOpError(
            "has no tile execution strategy allowed by kernel policy");
      }
      return failure();
    }
    llvm::stable_sort(choiceOptions, [](const KernelExecutionOption &lhs,
                                        const KernelExecutionOption &rhs) {
      return lhs.tileStrategy == TileExecutionStrategy::FPU &&
             rhs.tileStrategy != TileExecutionStrategy::FPU;
    });
    allOptions.push_back({choice.operation, ExecutionChoiceKind::TileStrategy,
                          std::move(choiceOptions)});
  }
  for (const ComputePipelineScheduleChoice &choice :
       requirements.pipelineScheduleChoices) {
    SmallVector<KernelExecutionOption, 2> choiceOptions;
    for (const ComputePipelineScheduleOption &option : choice.options) {
      Type elementType = option.destinationUses.empty()
                             ? Type()
                             : option.destinationUses.front().elementType;
      choiceOptions.push_back({option.dfbInputUses, option.destinationUses,
                               std::nullopt, option.kind, option.schedule,
                               elementType, option.requiredDstSlots,
                               option.sourceScalar});
    }
    allOptions.push_back({choice.pipeline,
                          ExecutionChoiceKind::ComputePipelineSchedule,
                          std::move(choiceOptions)});
  }
  return allOptions;
}

struct ExecutionSearchState {
  ConfigConstraintState constraints;
  SmallVector<std::optional<unsigned>> selections;
};

using ExecutionSearchResult =
    std::variant<ExecutionSearchState, ConfigConstraintConflict>;

ConfigConstraintResult applyExecutionConstraints(
    ConfigConstraintState state, const KernelExecutionOption &option,
    Operation *operation, const KernelTargetEnvironment &target,
    const KernelConfigPolicy &policy) {
  ConfigConstraintResult result =
      applyConfigConstraints(std::move(state), target, policy,
                             option.dfbInputUses, option.destinationUses);
  if (std::holds_alternative<ConfigConstraintConflict>(result) ||
      option.pipelineSchedule != ComputePipelineSchedule::RetainedScalar) {
    return result;
  }

  assert(option.pipelineKind && "pipeline schedule requires semantic kind");
  std::optional<std::uint32_t> targetMaximum =
      target.getMaxComputePipelineTiles(*option.pipelineKind,
                                        *option.pipelineSchedule,
                                        option.pipelineElementType);
  if (!targetMaximum) {
    return ConfigConstraintConflict(
        UnsupportedComputePipelineSchedule{operation});
  }
  ConfigConstraintState constrained =
      std::get<ConfigConstraintState>(std::move(result));
  std::uint32_t maximumAvailable = 0;
  for (const ConfigurationCandidate &candidate : constrained.candidates) {
    std::uint32_t registerCapacity = getDstCapacity(
        candidate.destinationElementWidth == DestinationElementWidth::Bits32,
        candidate.dstSyncMode == DstSyncMode::Full);
    maximumAvailable =
        std::max(maximumAvailable, std::min(*targetMaximum, registerCapacity));
  }
  llvm::erase_if(
      constrained.candidates, [&](const ConfigurationCandidate &candidate) {
        std::uint32_t registerCapacity =
            getDstCapacity(candidate.destinationElementWidth ==
                               DestinationElementWidth::Bits32,
                           candidate.dstSyncMode == DstSyncMode::Full);
        return std::min(*targetMaximum, registerCapacity) <
               option.requiredDstSlots;
      });
  if (constrained.candidates.empty()) {
    return ConfigConstraintConflict(ComputePipelineCapacityConflict{
        operation, option.requiredDstSlots, maximumAvailable});
  }
  return constrained;
}

ComputePipelineScheduleRejection getComputePipelineScheduleRejection(
    const ConfigConstraintConflict &conflict,
    const KernelTargetEnvironment &target,
    const KernelExecutionOption &retainedOption) {
  return std::visit(
      [&](const auto &typedConflict) -> ComputePipelineScheduleRejection {
        using ConflictType = std::decay_t<decltype(typedConflict)>;
        if constexpr (std::is_same_v<ConflictType,
                                     ComputePipelineCapacityConflict>) {
          return {ComputePipelineScheduleRejectionKind::DSTCapacity,
                  typedConflict.requiredDstSlots,
                  typedConflict.availableDstSlots};
        }
        if constexpr (std::is_same_v<ConflictType,
                                     UnsupportedComputePipelineSchedule>) {
          bool supportedTarget = target.getArch() == ttcore::Arch::Blackhole;
          return {
              supportedTarget
                  ? ComputePipelineScheduleRejectionKind::UnsupportedElementType
                  : ComputePipelineScheduleRejectionKind::UnsupportedTarget,
              retainedOption.requiredDstSlots, 0};
        }
        return {
            ComputePipelineScheduleRejectionKind::KernelConfigurationConflict,
            retainedOption.requiredDstSlots, 0};
      },
      conflict);
}

/// Select tile strategies and pipeline schedules with shared constraints.
ExecutionSearchResult
resolveExecutionChoices(ArrayRef<KernelExecutionChoiceOptions> allOptions,
                        const KernelTargetEnvironment &target,
                        const KernelConfigPolicy &policy,
                        ExecutionSearchState state) {
  std::optional<size_t> selectedChoice;
  size_t fewestCompatibleOptions = std::numeric_limits<size_t>::max();

  for (auto [choiceIndex, options] : llvm::enumerate(allOptions)) {
    if (state.selections[choiceIndex]) {
      continue;
    }
    size_t compatibleOptions = 0;
    std::optional<ConfigConstraintConflict> choiceConflict;
    for (const KernelExecutionOption &option : options.options) {
      ConfigConstraintResult result = applyExecutionConstraints(
          state.constraints, option, options.operation, target, policy);
      if (std::holds_alternative<ConfigConstraintState>(result)) {
        ++compatibleOptions;
      } else if (!choiceConflict) {
        choiceConflict = std::get<ConfigConstraintConflict>(std::move(result));
      }
    }
    if (compatibleOptions == 0) {
      assert(choiceConflict && "an incompatible option must retain evidence");
      return *choiceConflict;
    }
    if (compatibleOptions < fewestCompatibleOptions) {
      selectedChoice = choiceIndex;
      fewestCompatibleOptions = compatibleOptions;
    }
  }

  if (!selectedChoice) {
    return state;
  }

  // Option order preserves target-schedule and FPU preferences after shared
  // constraints are satisfied. The most constrained choice limits search.
  std::optional<ConfigConstraintConflict> immediateConflict;
  std::optional<ConfigConstraintConflict> branchConflict;
  const KernelExecutionChoiceOptions &choice = allOptions[*selectedChoice];
  for (auto [optionIndex, option] : llvm::enumerate(choice.options)) {
    ConfigConstraintResult result = applyExecutionConstraints(
        state.constraints, option, choice.operation, target, policy);
    if (std::holds_alternative<ConfigConstraintConflict>(result)) {
      if (!immediateConflict) {
        immediateConflict =
            std::get<ConfigConstraintConflict>(std::move(result));
      }
      continue;
    }
    ExecutionSearchState nextState = state;
    nextState.constraints = std::get<ConfigConstraintState>(std::move(result));
    nextState.selections[*selectedChoice] = optionIndex;
    ExecutionSearchResult searchResult = resolveExecutionChoices(
        allOptions, target, policy, std::move(nextState));
    if (std::holds_alternative<ExecutionSearchState>(searchResult)) {
      return searchResult;
    }
    if (!branchConflict) {
      branchConflict =
          std::get<ConfigConstraintConflict>(std::move(searchResult));
    }
  }

  if (branchConflict) {
    return *branchConflict;
  }
  assert(immediateConflict && "failed strategy search must retain evidence");
  return *immediateConflict;
}

SmallVector<DFBHardwareConfiguration, 4>
getWormholeBlackholeSupportedDFBConfigurations(TilePrimitive primitive,
                                               TileOperandRoute route,
                                               Type elementType) {
  bool requiresDestinationRoute =
      elementType.isF32() &&
      (route == TileOperandRoute::Dst || primitive == TilePrimitive::Copy);
  if (requiresDestinationRoute) {
    return {
        {DestinationElementWidth::Bits32, DFBUnpackMode::UnpackToDestination}};
  }
  return {{DestinationElementWidth::Bits16, DFBUnpackMode::Default},
          {DestinationElementWidth::Bits32, DFBUnpackMode::Default}};
}

class UnspecifiedKernelTargetEnvironment final
    : public KernelTargetEnvironment {
public:
  bool supportsDestinationElementWidth(
      TilePrimitive, Type elementType,
      DestinationElementWidth destinationElementWidth) const override {
    return destinationElementWidth == DestinationElementWidth::Bits32 ||
           !requires32BitDestination(elementType);
  }

  FullFp32AccumulationSupport
  getFullFp32AccumulationSupport(FullFp32AccumulationKind) const override {
    return {true, std::nullopt};
  }

  std::optional<std::uint32_t>
  getMaxComputePipelineTiles(ComputePipelineKind, ComputePipelineSchedule,
                             Type) const override {
    return std::nullopt;
  }

  SmallVector<DFBHardwareConfiguration, 4>
  getSupportedDFBConfigurations(TilePrimitive primitive, TileOperandRoute route,
                                Type elementType) const override {
    return getWormholeBlackholeSupportedDFBConfigurations(primitive, route,
                                                          elementType);
  }
};

class WormholeBlackholeKernelTargetEnvironment
    : public KernelTargetEnvironment {
public:
  bool supportsDestinationElementWidth(
      TilePrimitive primitive, Type elementType,
      DestinationElementWidth destinationElementWidth) const final {
    if (destinationElementWidth == DestinationElementWidth::Bits16) {
      return !requires32BitDestination(elementType);
    }
    if (requires32BitDestination(elementType)) {
      return true;
    }
    switch (primitive) {
    case TilePrimitive::BroadcastColumn:
    case TilePrimitive::BroadcastRow:
    case TilePrimitive::BroadcastScalar:
      // These LLKs interpret non-32-bit inputs using the default DST format.
      return false;
    default:
      return true;
    }
  }

  SmallVector<DFBHardwareConfiguration, 4>
  getSupportedDFBConfigurations(TilePrimitive primitive, TileOperandRoute route,
                                Type elementType) const final {
    return getWormholeBlackholeSupportedDFBConfigurations(primitive, route,
                                                          elementType);
  }
};

class WormholeKernelTargetEnvironment final
    : public WormholeBlackholeKernelTargetEnvironment {
public:
  FullFp32AccumulationSupport
  getFullFp32AccumulationSupport(FullFp32AccumulationKind kind) const override {
    return {kind == FullFp32AccumulationKind::Matmul, std::nullopt};
  }

  std::optional<std::uint32_t>
  getMaxComputePipelineTiles(ComputePipelineKind, ComputePipelineSchedule,
                             Type) const override {
    return std::nullopt;
  }
};

class BlackholeKernelTargetEnvironment final
    : public WormholeBlackholeKernelTargetEnvironment {
public:
  FullFp32AccumulationSupport
  getFullFp32AccumulationSupport(FullFp32AccumulationKind kind) const override {
    if (kind == FullFp32AccumulationKind::ReduceRow) {
      return {false,
              "full-fp32 row reduce is unavailable on Blackhole (tt-metal "
              "#47311); using non-full-fp32 reduce lowering"};
    }
    return {true, std::nullopt};
  }

  std::optional<std::uint32_t>
  getMaxComputePipelineTiles(ComputePipelineKind kind,
                             ComputePipelineSchedule schedule,
                             Type elementType) const override {
    if (schedule == ComputePipelineSchedule::RetainedScalar &&
        elementType.isBF16() &&
        (kind == ComputePipelineKind::MultiplyFullScalarReduction ||
         kind == ComputePipelineKind::RowNormalization)) {
      return 8;
    }
    return std::nullopt;
  }
};

} // namespace

FailureOr<std::unique_ptr<KernelTargetEnvironment>>
KernelTargetEnvironment::get(func::FuncOp function) {
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module) {
    function.emitOpError("is not nested in a module");
    return failure();
  }

  std::string targetFailureReason;
  FailureOr<std::optional<ttcore::Arch>> targetArch =
      resolveComputeTargetArch(function, targetFailureReason);
  if (failed(targetArch)) {
    module.emitOpError(targetFailureReason);
    return failure();
  }
  if (!*targetArch) {
    return std::unique_ptr<KernelTargetEnvironment>(
        std::make_unique<UnspecifiedKernelTargetEnvironment>());
  }

  switch (**targetArch) {
  case ttcore::Arch::WormholeB0:
    return std::unique_ptr<KernelTargetEnvironment>(
        std::make_unique<WormholeKernelTargetEnvironment>());
  case ttcore::Arch::Blackhole:
    return std::unique_ptr<KernelTargetEnvironment>(
        std::make_unique<BlackholeKernelTargetEnvironment>());
  case ttcore::Arch::Quasar:
    module.emitOpError(
        "Quasar compute kernels require the Gen2 configuration and launch "
        "APIs, which are not supported by the current TT-Lang runtime");
    return failure();
  }
  module.emitOpError()
      << "compute-kernel configuration is not implemented for target "
         "architecture "
      << ttcore::ArchAttr::get(module.getContext(), **targetArch);
  return failure();
}

FailureOr<KernelConfigPolicy>
KernelConfigPolicy::get(func::FuncOp function, StringRef fp32Selection,
                        StringRef syncSelection, bool reduceFullFp32,
                        bool matmulFullFp32, bool enableFPUBinary) {
  FailureOr<ConfigSelection> fp32 =
      parseConfigSelection(function, "fp32-dest-acc-en", fp32Selection);
  FailureOr<ConfigSelection> sync =
      parseConfigSelection(function, "dst-full-sync-en", syncSelection);
  if (failed(fp32) || failed(sync)) {
    return failure();
  }

  FailureOr<std::optional<bool>> fp32Constraint =
      getOptionalBoolConstraint(function, kFp32DestAccEnAttrName);
  FailureOr<std::optional<bool>> syncConstraint =
      getOptionalBoolConstraint(function, kDstFullSyncEnAttrName);
  FailureOr<std::optional<bool>> fpuConstraint =
      getOptionalBoolConstraint(function, kEnableFPUBinaryOpsAttrName);
  FailureOr<std::optional<SmallVector<int32_t>>> unpackConstraint =
      getOptionalUnpackConstraint(function);
  if (failed(fp32Constraint) || failed(syncConstraint) ||
      failed(fpuConstraint) || failed(unpackConstraint)) {
    return failure();
  }

  KernelConfigPolicy policy;
  policy.fp32DestAccumulation = *fp32;
  policy.dstSynchronization = *sync;
  policy.preferFullFp32Reduce = reduceFullFp32;
  policy.preferFullFp32Matmul = matmulFullFp32;
  policy.allowFPUBinary = enableFPUBinary;
  policy.unpackToDestFp32 = std::move(*unpackConstraint);
  if (*fp32Constraint) {
    policy.fp32DestAccumulation =
        **fp32Constraint ? ConfigSelection::Enabled : ConfigSelection::Disabled;
  }
  if (*syncConstraint) {
    policy.dstSynchronization =
        **syncConstraint ? ConfigSelection::Enabled : ConfigSelection::Disabled;
  }
  if (*fpuConstraint) {
    policy.allowFPUBinary = **fpuConstraint;
  }
  return policy;
}

FailureOr<KernelRequirements> collectKernelRequirements(func::FuncOp function) {
  KernelRequirements requirements;
  WalkResult result = function.walk([&](Operation *operation) {
    if (auto pipeline = dyn_cast<ComputePipelineOp>(operation)) {
      if (!pipeline.getPipelineKind()) {
        pipeline.emitOpError(
            "must be lowered before compute-kernel configuration");
        return WalkResult::interrupt();
      }
      FailureOr<ComputePipelineScheduleChoice> choice =
          getComputePipelineScheduleChoice(pipeline);
      if (failed(choice)) {
        return WalkResult::interrupt();
      }
      requirements.pipelineScheduleChoices.push_back(std::move(*choice));
      requirements.fullFp32AccumulationUses.push_back(
          {pipeline, FullFp32AccumulationKind::ReduceScalar});
      return WalkResult::skip();
    }
    auto executionOp = dyn_cast<TileExecutionOpInterface>(operation);
    if (!executionOp) {
      if (isTileComputeOp(operation)) {
        operation->emitOpError("does not implement TileExecutionOpInterface");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (!executionOp.getLegalExecutionStrategies().empty()) {
      FailureOr<TileExecutionChoice> choice =
          getTileExecutionChoice(executionOp);
      if (failed(choice)) {
        return WalkResult::interrupt();
      }
      requirements.tileStrategyChoices.push_back(std::move(*choice));
      return WalkResult::advance();
    }

    if (failed(verifyTileExecutionStrategy(operation, {}))) {
      return WalkResult::interrupt();
    }
    FailureOr<TileExecutionInfo> info =
        executionOp.getTileExecutionInfo(std::nullopt);
    if (failed(info)) {
      operation->emitOpError("has no tile execution semantics");
      return WalkResult::interrupt();
    }
    if (failed(verifyTileExecutionInfo(operation, *info)) ||
        failed(appendExecutionRequirements(operation, *info,
                                           requirements.dfbInputUses,
                                           requirements.destinationUses))) {
      return WalkResult::interrupt();
    }
    if (info->fullFp32Accumulation) {
      requirements.fullFp32AccumulationUses.push_back(
          {operation, *info->fullFp32Accumulation});
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }
  return requirements;
}

FailureOr<KernelConfigPlan> resolveKernelConfig(
    func::FuncOp function, const KernelTargetEnvironment &target,
    const KernelConfigPolicy &policy, const KernelRequirements &requirements) {
  ConfigConstraintState initialState;
  if (policy.fp32DestAccumulation == ConfigSelection::Enabled) {
    llvm::erase_if(initialState.candidates,
                   [](const ConfigurationCandidate &candidate) {
                     return candidate.destinationElementWidth ==
                            DestinationElementWidth::Bits16;
                   });
    initialState.requires32Bits = {function.getOperation(), std::nullopt};
  } else if (policy.fp32DestAccumulation == ConfigSelection::Disabled) {
    llvm::erase_if(initialState.candidates,
                   [](const ConfigurationCandidate &candidate) {
                     return candidate.destinationElementWidth ==
                            DestinationElementWidth::Bits32;
                   });
    initialState.requires16Bits = {function.getOperation(), std::nullopt};
  }
  if (policy.dstSynchronization == ConfigSelection::Enabled) {
    llvm::erase_if(initialState.candidates,
                   [](const ConfigurationCandidate &candidate) {
                     return candidate.dstSyncMode != DstSyncMode::Full;
                   });
  } else if (policy.dstSynchronization == ConfigSelection::Disabled) {
    llvm::erase_if(
        initialState.candidates, [](const ConfigurationCandidate &candidate) {
          return candidate.dstSyncMode != DstSyncMode::DoubleBuffered;
        });
  }
  ConfigConstraintResult fixedResult = applyConfigConstraints(
      initialState, target, policy, requirements.dfbInputUses,
      requirements.destinationUses);
  if (std::holds_alternative<ConfigConstraintConflict>(fixedResult)) {
    emitConfigConstraintConflict(
        function, std::get<ConfigConstraintConflict>(std::move(fixedResult)));
    return failure();
  }

  bool preferFp32 = false;
  SmallVector<Operation *> supportedFullFp32Preferences;
  for (const FullFp32AccumulationUse &use :
       requirements.fullFp32AccumulationUses) {
    bool isMatmul = use.kind == FullFp32AccumulationKind::Matmul;
    bool preferred =
        isMatmul ? policy.preferFullFp32Matmul : policy.preferFullFp32Reduce;
    if (!preferred) {
      continue;
    }
    FullFp32AccumulationSupport support =
        target.getFullFp32AccumulationSupport(use.kind);
    if (support.supported) {
      preferFp32 = true;
      supportedFullFp32Preferences.push_back(use.operation);
      continue;
    }
    if (support.fallbackWarning) {
      use.operation->emitWarning() << *support.fallbackWarning;
    }
  }

  FailureOr<KernelExecutionOptions> allOptions =
      getKernelExecutionOptions(requirements, policy);
  if (failed(allOptions)) {
    return failure();
  }
  ConfigConstraintState fixedState =
      std::get<ConfigConstraintState>(std::move(fixedResult));
  auto search = [&](ConfigConstraintState constraints) {
    return resolveExecutionChoices(
        *allOptions, target, policy,
        ExecutionSearchState{
            std::move(constraints),
            SmallVector<std::optional<unsigned>>(allOptions->size())});
  };

  std::optional<ExecutionSearchResult> searchResult;
  if (preferFp32 && llvm::any_of(fixedState.candidates,
                                 [](const ConfigurationCandidate &candidate) {
                                   return candidate.destinationElementWidth ==
                                          DestinationElementWidth::Bits32;
                                 })) {
    ConfigConstraintState preferredState = fixedState;
    llvm::erase_if(preferredState.candidates,
                   [](const ConfigurationCandidate &candidate) {
                     return candidate.destinationElementWidth !=
                            DestinationElementWidth::Bits32;
                   });
    ExecutionSearchResult preferredResult = search(std::move(preferredState));
    if (std::holds_alternative<ExecutionSearchState>(preferredResult)) {
      searchResult = std::move(preferredResult);
    }
  }
  if (!searchResult) {
    searchResult = search(std::move(fixedState));
  }
  if (std::holds_alternative<ConfigConstraintConflict>(*searchResult)) {
    emitConfigConstraintConflict(
        function, std::get<ConfigConstraintConflict>(std::move(*searchResult)));
    return failure();
  }
  ExecutionSearchState resolvedState =
      std::get<ExecutionSearchState>(std::move(*searchResult));

  SmallVector<TileExecutionDecision> tileStrategies;
  tileStrategies.reserve(requirements.tileStrategyChoices.size());
  SmallVector<ComputePipelineScheduleDecision> pipelineSchedules;
  pipelineSchedules.reserve(requirements.pipelineScheduleChoices.size());
  for (auto [choiceIndex, choice] : llvm::enumerate(*allOptions)) {
    std::optional<unsigned> selectedOption =
        resolvedState.selections[choiceIndex];
    assert(selectedOption && *selectedOption < choice.options.size() &&
           "successful execution search must select every choice");
    const KernelExecutionOption &option = choice.options[*selectedOption];
    switch (choice.kind) {
    case ExecutionChoiceKind::TileStrategy:
      assert(option.tileStrategy && !option.pipelineKind &&
             !option.pipelineSchedule &&
             "tile choice must select a tile strategy");
      tileStrategies.push_back({choice.operation, *option.tileStrategy});
      break;
    case ExecutionChoiceKind::ComputePipelineSchedule:
      assert(option.pipelineKind && option.pipelineSchedule &&
             !option.tileStrategy &&
             "pipeline choice must select a pipeline schedule");
      std::optional<ComputePipelineScheduleRejection> rejection;
      if (*option.pipelineSchedule == ComputePipelineSchedule::Materialized) {
        auto retained = llvm::find_if(
            choice.options, [](const KernelExecutionOption &candidate) {
              return candidate.pipelineSchedule ==
                     ComputePipelineSchedule::RetainedScalar;
            });
        if (retained != choice.options.end()) {
          ConfigConstraintResult retainedResult =
              applyExecutionConstraints(resolvedState.constraints, *retained,
                                        choice.operation, target, policy);
          if (std::holds_alternative<ConfigConstraintConflict>(
                  retainedResult)) {
            rejection = getComputePipelineScheduleRejection(
                std::get<ConfigConstraintConflict>(std::move(retainedResult)),
                target, *retained);
          }
        }
      }
      pipelineSchedules.push_back({choice.operation, *option.pipelineSchedule,
                                   rejection, option.sourceScalar});
      break;
    }
  }
  auto hasDestinationWidth = [&](DestinationElementWidth width) {
    return llvm::any_of(resolvedState.constraints.candidates,
                        [&](const ConfigurationCandidate &candidate) {
                          return candidate.destinationElementWidth == width;
                        });
  };
  DestinationElementWidth destinationElementWidth =
      DestinationElementWidth::Bits16;
  if ((preferFp32 || !hasDestinationWidth(DestinationElementWidth::Bits16)) &&
      hasDestinationWidth(DestinationElementWidth::Bits32)) {
    destinationElementWidth = DestinationElementWidth::Bits32;
  }
  if (preferFp32 && policy.fp32DestAccumulation != ConfigSelection::Disabled &&
      destinationElementWidth == DestinationElementWidth::Bits16) {
    for (Operation *operation : supportedFullFp32Preferences) {
      operation->emitWarning()
          << "preferred full-fp32 accumulation is unavailable in the resolved "
             "kernel configuration; using non-full-fp32 lowering";
    }
  }
  auto selectedCandidate = llvm::find_if(
      resolvedState.constraints.candidates,
      [&](const ConfigurationCandidate &candidate) {
        return candidate.destinationElementWidth == destinationElementWidth &&
               candidate.dstSyncMode == DstSyncMode::DoubleBuffered;
      });
  if (selectedCandidate == resolvedState.constraints.candidates.end()) {
    selectedCandidate = llvm::find_if(
        resolvedState.constraints.candidates,
        [&](const ConfigurationCandidate &candidate) {
          return candidate.destinationElementWidth == destinationElementWidth;
        });
  }
  assert(selectedCandidate != resolvedState.constraints.candidates.end() &&
         "resolved destination width must have a candidate");

  DstSyncMode syncMode = selectedCandidate->dstSyncMode;

  if (policy.unpackToDestFp32) {
    return KernelConfigPlan(destinationElementWidth, syncMode,
                            *policy.unpackToDestFp32, std::move(tileStrategies),
                            std::move(pipelineSchedules));
  }

  SmallVector<int32_t> unpackToDestFp32;
  for (const auto &[dfbIndex, modes] : selectedCandidate->unpackModes) {
    if (!modes.contains(DFBUnpackMode::Default) &&
        modes.contains(DFBUnpackMode::UnpackToDestination)) {
      unpackToDestFp32.push_back(static_cast<int32_t>(dfbIndex));
    }
  }
  llvm::sort(unpackToDestFp32);
  return KernelConfigPlan(
      destinationElementWidth, syncMode, std::move(unpackToDestFp32),
      std::move(tileStrategies), std::move(pipelineSchedules));
}

static LogicalResult
validateSourceScalarRetentionPlan(ComputePipelineOp pipeline,
                                  const SourceScalarRetentionPlan &plan) {
  Block &body = pipeline.getBody().front();
  SmallVector<Operation *> stages;
  for (ComputeStageOp stage : body.getOps<ComputeStageOp>()) {
    stages.push_back(stage);
  }
  SmallVector<Operation *> plannedStages;
  for (const SourceScalarStagePlan &stage : plan.producerStages) {
    plannedStages.push_back(stage.stage);
  }
  for (const SourceScalarStagePlan &stage : plan.consumerStages) {
    plannedStages.push_back(stage.stage);
  }
  if (!llvm::equal(stages, plannedStages) || plan.producerStages.empty() ||
      plan.consumerStages.empty() ||
      plan.producerStage != plan.producerStages.back().stage ||
      plan.producerResult >= plan.producerStage->getNumResults()) {
    return pipeline.emitOpError(
        "source-scalar resource plan no longer matches the pipeline stages");
  }
  auto stageInputsChanged = [](const SourceScalarStagePlan &stage) {
    return !llvm::equal(stage.stage->getOperands(), stage.inputs);
  };
  if (llvm::any_of(plan.producerStages, stageInputsChanged) ||
      llvm::any_of(plan.consumerStages, stageInputsChanged)) {
    return pipeline.emitOpError(
        "source-scalar resource plan no longer matches stage inputs");
  }

  Value scalar = plan.producerStage->getResult(plan.producerResult);
  SmallVector<std::pair<Operation *, unsigned>> actualConsumers;
  for (OpOperand &use : scalar.getUses()) {
    actualConsumers.emplace_back(use.getOwner(), use.getOperandNumber());
  }
  if (actualConsumers.size() != plan.consumers.size()) {
    return pipeline.emitOpError(
        "source-scalar resource plan no longer matches scalar uses");
  }
  for (const SourceScalarConsumerPlan &consumer : plan.consumers) {
    if (!llvm::is_contained(actualConsumers,
                            std::pair<Operation *, unsigned>{
                                consumer.stage, consumer.operandIndex})) {
      return pipeline.emitOpError(
          "source-scalar resource plan no longer matches scalar consumers");
    }
  }
  return success();
}

static void formSourceScalarScope(ComputePipelineOp pipeline,
                                  const SourceScalarRetentionPlan &plan,
                                  IRRewriter &rewriter) {
  Block &pipelineBody = pipeline.getBody().front();
  auto pipelineYield =
      cast<ComputePipelineYieldOp>(pipelineBody.getTerminator());
  Location location = pipeline.getLoc();

  rewriter.setInsertionPoint(pipeline);
  auto scope = SourceScalarScopeOp::create(
      rewriter, location, pipeline.getResultTypes(), pipeline.getInputs());

  Block *producerBody = rewriter.createBlock(&scope.getProducer());
  for (Value input : pipeline.getInputs()) {
    producerBody->addArgument(input.getType(), location);
  }
  IRMapping producerMapping;
  for (auto [pipelineArgument, producerArgument] : llvm::zip_equal(
           pipelineBody.getArguments(), producerBody->getArguments())) {
    producerMapping.map(pipelineArgument, producerArgument);
  }
  rewriter.setInsertionPointToStart(producerBody);
  for (const SourceScalarStagePlan &stage : plan.producerStages) {
    rewriter.clone(*stage.stage, producerMapping);
  }
  Value scalar = plan.producerStage->getResult(plan.producerResult);
  Value retainedScalar = producerMapping.lookup(scalar);
  SourceScalarYieldOp::create(rewriter, location, retainedScalar);

  Block *consumerBody = rewriter.createBlock(&scope.getConsumer());
  consumerBody->addArgument(retainedScalar.getType(), location);
  for (Value input : pipeline.getInputs()) {
    consumerBody->addArgument(input.getType(), location);
  }
  IRMapping consumerMapping;
  consumerMapping.map(scalar, consumerBody->getArgument(0));
  for (auto [pipelineArgument, consumerArgument] :
       llvm::zip_equal(pipelineBody.getArguments(),
                       consumerBody->getArguments().drop_front())) {
    consumerMapping.map(pipelineArgument, consumerArgument);
  }
  rewriter.setInsertionPointToStart(consumerBody);
  for (const SourceScalarStagePlan &stage : plan.consumerStages) {
    rewriter.clone(*stage.stage, consumerMapping);
  }
  SmallVector<Value> yieldedValues;
  yieldedValues.reserve(pipelineYield.getValues().size());
  for (Value yieldedValue : pipelineYield.getValues()) {
    yieldedValues.push_back(consumerMapping.lookup(yieldedValue));
  }
  SourceScalarYieldOp::create(rewriter, location, yieldedValues);
  rewriter.replaceOp(pipeline, scope.getResults());
}

LogicalResult applyComputePipelineSchedulePlan(func::FuncOp function,
                                               const KernelConfigPlan &plan) {
  for (const ComputePipelineScheduleDecision &decision :
       plan.getComputePipelineSchedules()) {
    auto pipeline = dyn_cast<ComputePipelineOp>(decision.pipeline);
    if (!pipeline) {
      return function.emitOpError(
          "compute-pipeline schedule plan references an invalid operation");
    }
    if (decision.sourceScalar && failed(validateSourceScalarRetentionPlan(
                                     pipeline, *decision.sourceScalar))) {
      return failure();
    }
  }

  MLIRContext *context = function.getContext();
  IRRewriter rewriter(context);
  for (const ComputePipelineScheduleDecision &decision :
       plan.getComputePipelineSchedules()) {
    auto pipeline = cast<ComputePipelineOp>(decision.pipeline);
    if (decision.rejection) {
      ComputePipelineKind kind = pipeline.getPipelineKindAttr().getValue();
      std::string message;
      llvm::raw_string_ostream output(message);
      output << stringifyComputePipelineKind(kind) << " fusion not selected: ";
      switch (decision.rejection->kind) {
      case ComputePipelineScheduleRejectionKind::UnsupportedTarget:
        output << "the target does not provide the retained-scalar schedule";
        break;
      case ComputePipelineScheduleRejectionKind::UnsupportedElementType:
        output << "the retained-scalar schedule does not support the pipeline "
                  "element type";
        break;
      case ComputePipelineScheduleRejectionKind::DSTCapacity:
        output << "the reduction requires "
               << decision.rejection->requiredDstSlots << " DST slots, but "
               << decision.rejection->availableDstSlots << " are available";
        break;
      case ComputePipelineScheduleRejectionKind::KernelConfigurationConflict:
        output << "the retained-scalar schedule conflicts with another "
                  "kernel-wide compute requirement";
        break;
      }
      output << "; ordinary materialized lowering remains selected";
      remark::missed(pipeline.getLoc(),
                     remark::RemarkOpts::name("ReductionFusion")
                         .category("ttl-reduction-fusion")
                         .function(function.getSymName()))
          << message;
    }

    if (decision.sourceScalar) {
      assert(decision.schedule == ComputePipelineSchedule::RetainedScalar &&
             "source-scalar plan requires the retained schedule");
      formSourceScalarScope(pipeline, *decision.sourceScalar, rewriter);
      continue;
    }
    pipeline.setSelectedScheduleAttr(
        ComputePipelineScheduleAttr::get(context, decision.schedule));
  }
  return success();
}

LogicalResult applyKernelConfigPlan(func::FuncOp function,
                                    const KernelConfigPlan &plan) {
  if (failed(applyComputePipelineSchedulePlan(function, plan))) {
    return failure();
  }
  MLIRContext *context = function.getContext();
  for (const TileExecutionDecision &decision : plan.getTileStrategies()) {
    decision.operation->setAttr(
        kTileExecutionStrategyAttrName,
        TileExecutionStrategyAttr::get(context, decision.strategy));
  }
  function->setAttr(
      kFp32DestAccEnAttrName,
      BoolAttr::get(context, plan.getDestinationElementWidth() ==
                                 DestinationElementWidth::Bits32));
  function->setAttr(
      kDstFullSyncEnAttrName,
      BoolAttr::get(context, plan.getDstSyncMode() == DstSyncMode::Full));
  function->setAttr(
      kUnpackToDestFp32AttrName,
      DenseI32ArrayAttr::get(context, plan.getUnpackToDestFp32()));
  function->removeAttr(kEnableFPUBinaryOpsAttrName);
  return success();
}

} // namespace mlir::tt::ttl
