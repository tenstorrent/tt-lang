// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/ComputeKernelConfigAnalysis.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/ComputeTarget.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallSet.h"

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
  int32_t targetMaxDFBIndices = getTargetMaxDFBIndices(function);
  if (llvm::any_of(dataflowBufferIndices, [&](int32_t index) {
        return index < 0 || index >= targetMaxDFBIndices;
      })) {
    function.emitOpError()
        << kUnpackToDestFp32AttrName
        << " must contain dataflow buffer indices in range [0, "
        << targetMaxDFBIndices - 1 << "] for "
        << getTargetDFBIndexCapacityDescription(function);
    return failure();
  }
  llvm::sort(dataflowBufferIndices);
  dataflowBufferIndices.erase(
      std::unique(dataflowBufferIndices.begin(), dataflowBufferIndices.end()),
      dataflowBufferIndices.end());
  return std::optional<SmallVector<int32_t>>(std::move(dataflowBufferIndices));
}

struct ResolvedDataflowBuffer {
  Value dfb;
  int64_t dfbIndex;
};

/// Return the attached DFB and index after resolving a compute-region argument.
FailureOr<std::optional<ResolvedDataflowBuffer>>
resolveDataflowBuffer(Value value, Operation *consumer) {
  if (auto blockArgument = dyn_cast<BlockArgument>(value)) {
    auto computeOp =
        dyn_cast_or_null<ComputeOp>(blockArgument.getOwner()->getParentOp());
    if (!computeOp) {
      return std::optional<ResolvedDataflowBuffer>();
    }
    unsigned argumentNumber = blockArgument.getArgNumber();
    if (argumentNumber < computeOp.getNumInputs()) {
      value = computeOp.getInputs()[argumentNumber];
    } else {
      unsigned outputNumber = argumentNumber - computeOp.getNumInputs();
      assert(outputNumber < computeOp.getNumOutputs() &&
             "compute region argument must map to an input or output");
      value = computeOp.getOutputs()[outputNumber];
    }
  }

  Value dfb = getAttachedCB(value);
  if (!dfb) {
    return std::optional<ResolvedDataflowBuffer>();
  }
  std::optional<int64_t> dfbIndex = getCBIndex(dfb);
  if (!dfbIndex) {
    consumer->emitOpError("uses a dataflow buffer without a finalized index");
    return failure();
  }
  return std::make_optional(ResolvedDataflowBuffer{dfb, *dfbIndex});
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
    FailureOr<std::optional<ResolvedDataflowBuffer>> dataflowBuffer =
        resolveDataflowBuffer(operand.get(), operation);
    if (failed(dataflowBuffer)) {
      return failure();
    }
    if (!*dataflowBuffer) {
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
    FailureOr<std::optional<ResolvedDataflowBuffer>> dataflowBuffer =
        resolveDataflowBuffer(operand.get(), operation);
    if (failed(dataflowBuffer)) {
      return failure();
    }
    if (route == TileOperandRoute::DataflowBuffer && !*dataflowBuffer) {
      operation->emitOpError()
          << "operand " << operand.getOperandNumber()
          << " must resolve to a dataflow buffer for the selected strategy";
      return failure();
    }
    if (*dataflowBuffer) {
      dfbInputUses.push_back(
          {(*dataflowBuffer)->dfbIndex, (*dataflowBuffer)->dfb, operation,
           operand.getOperandNumber(), info.primitive, route, *elementType});
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

using ConfigConstraintConflict =
    std::variant<DestinationWidthConflict, ExplicitUnpackConflict,
                 DFBUnpackConflict, UnsupportedDFBConfiguration,
                 UnsupportedDestinationConfiguration>;

struct ConfigurationCandidate {
  explicit ConfigurationCandidate(DestinationElementWidth width)
      : destinationElementWidth(width) {}

  DestinationElementWidth destinationElementWidth;
  llvm::MapVector<int64_t, llvm::SmallSet<DFBUnpackMode, 2>> unpackModes;
  llvm::DenseMap<int64_t, DFBInputUse> firstUnpackUses;
};

struct ConfigConstraintState {
  ConfigConstraintState() {
    candidates.emplace_back(DestinationElementWidth::Bits16);
    candidates.emplace_back(DestinationElementWidth::Bits32);
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
        } else {
          typedConflict.use.operation->emitOpError()
              << "has no target-supported destination element width for "
              << typedConflict.use.elementType << " elements";
        }
      },
      conflict);
}

using TileStrategyOptions = SmallVector<TileExecutionOption, 2>;
using KernelTileStrategyOptions = SmallVector<TileStrategyOptions, 0>;

/// Apply strategy policy without changing the collected requirements.
FailureOr<KernelTileStrategyOptions>
getTileStrategyOptions(const KernelRequirements &requirements,
                       const KernelConfigPolicy &policy) {
  KernelTileStrategyOptions allOptions;
  allOptions.reserve(requirements.tileStrategyChoices.size());
  for (const TileExecutionChoice &choice : requirements.tileStrategyChoices) {
    TileStrategyOptions choiceOptions;
    for (const TileExecutionOption &option : choice.options) {
      if (option.strategy == TileExecutionStrategy::FPU &&
          !policy.allowFPUBinary) {
        continue;
      }
      choiceOptions.push_back(option);
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
    llvm::stable_sort(choiceOptions, [](const TileExecutionOption &lhs,
                                        const TileExecutionOption &rhs) {
      return lhs.strategy == TileExecutionStrategy::FPU &&
             rhs.strategy != TileExecutionStrategy::FPU;
    });
    allOptions.push_back(std::move(choiceOptions));
  }
  return allOptions;
}

struct StrategySearchState {
  ConfigConstraintState constraints;
  SmallVector<std::optional<TileExecutionStrategy>> selections;
};

using StrategySearchResult =
    std::variant<StrategySearchState, ConfigConstraintConflict>;

/// Select strategies jointly with shared per-DFB unpack constraints.
StrategySearchResult
resolveTileStrategies(ArrayRef<TileStrategyOptions> allOptions,
                      const KernelTargetEnvironment &target,
                      const KernelConfigPolicy &policy,
                      StrategySearchState state) {
  std::optional<size_t> selectedChoice;
  size_t fewestCompatibleOptions = std::numeric_limits<size_t>::max();

  for (auto [choiceIndex, options] : llvm::enumerate(allOptions)) {
    if (state.selections[choiceIndex]) {
      continue;
    }
    size_t compatibleOptions = 0;
    std::optional<ConfigConstraintConflict> choiceConflict;
    for (const TileExecutionOption &option : options) {
      ConfigConstraintResult result =
          applyConfigConstraints(state.constraints, target, policy,
                                 option.dfbInputUses, option.destinationUses);
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

  // Option order preserves the FPU preference after shared constraints are
  // satisfied. Resolving the most constrained operation limits backtracking.
  std::optional<ConfigConstraintConflict> immediateConflict;
  std::optional<ConfigConstraintConflict> branchConflict;
  for (const TileExecutionOption &option : allOptions[*selectedChoice]) {
    ConfigConstraintResult result =
        applyConfigConstraints(state.constraints, target, policy,
                               option.dfbInputUses, option.destinationUses);
    if (std::holds_alternative<ConfigConstraintConflict>(result)) {
      if (!immediateConflict) {
        immediateConflict =
            std::get<ConfigConstraintConflict>(std::move(result));
      }
      continue;
    }
    StrategySearchState nextState = state;
    nextState.constraints = std::get<ConfigConstraintState>(std::move(result));
    nextState.selections[*selectedChoice] = option.strategy;
    StrategySearchResult searchResult =
        resolveTileStrategies(allOptions, target, policy, std::move(nextState));
    if (std::holds_alternative<StrategySearchState>(searchResult)) {
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
      resolveTargetArch(function, targetFailureReason);
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

static FailureOr<KernelRequirements> collectKernelRequirementsImpl(
    func::FuncOp function,
    llvm::function_ref<bool(Operation *)> includeOperation) {
  KernelRequirements requirements;
  WalkResult result = function.walk([&](Operation *operation) {
    auto executionOp = dyn_cast<TileExecutionOpInterface>(operation);
    if (!executionOp) {
      if (isTileComputeOp(operation)) {
        operation->emitOpError("does not implement TileExecutionOpInterface");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    bool contributesConfiguration = includeOperation(operation);
    if (!executionOp.getLegalExecutionStrategies().empty()) {
      FailureOr<TileExecutionChoice> choice =
          getTileExecutionChoice(executionOp);
      if (failed(choice)) {
        return WalkResult::interrupt();
      }
      if (!contributesConfiguration) {
        for (TileExecutionOption &option : choice->options) {
          option.dfbInputUses.clear();
          option.destinationUses.clear();
        }
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
    if (failed(verifyTileExecutionInfo(operation, *info))) {
      return WalkResult::interrupt();
    }
    if (!contributesConfiguration) {
      return WalkResult::advance();
    }
    if (failed(appendExecutionRequirements(operation, *info,
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

FailureOr<KernelRequirements>
collectKernelRequirements(func::FuncOp function,
                          const LaunchNodeDomainState &launchDomains) {
  return collectKernelRequirementsImpl(function, [&](Operation *operation) {
    return !hasExactEmptyLaunchDomain(operation, launchDomains);
  });
}

namespace {

using DFBConfigurationMask = std::uint8_t;
constexpr DFBConfigurationMask kAllDFBConfigurations = 0xF;

DFBConfigurationMask
getDFBConfigurationMask(const KernelTargetEnvironment &target,
                        const DFBInputUse &use) {
  DFBConfigurationMask mask = 0;
  for (const DFBHardwareConfiguration &configuration :
       target.getSupportedDFBConfigurations(use.primitive, use.route,
                                            use.elementType)) {
    unsigned widthOffset =
        configuration.destinationElementWidth == DestinationElementWidth::Bits32
            ? 2
            : 0;
    unsigned modeOffset =
        configuration.unpackMode == DFBUnpackMode::UnpackToDestination ? 1 : 0;
    mask |= DFBConfigurationMask{1} << (widthOffset + modeOffset);
  }
  return mask;
}

void appendUniqueMask(SmallVectorImpl<DFBConfigurationMask> &masks,
                      DFBConfigurationMask mask) {
  if (mask != 0 && !llvm::is_contained(masks, mask)) {
    masks.push_back(mask);
  }
}

struct DFBConfigurationProfile {
  Value dfb;
  Operation *evidence = nullptr;
  SmallVector<DFBConfigurationMask, 4> alternatives;
};

DFBConfigurationProfile
buildDFBConfigurationProfile(Value dfb, Operation *evidence,
                             const KernelTargetEnvironment &target,
                             const KernelRequirements &requirements) {
  DFBConfigurationMask fixedMask = kAllDFBConfigurations;
  for (const DFBInputUse &use : requirements.dfbInputUses) {
    if (use.dfb == dfb) {
      fixedMask &= getDFBConfigurationMask(target, use);
    }
  }

  SmallVector<DFBConfigurationMask, 4> alternatives;
  appendUniqueMask(alternatives, fixedMask);
  for (const TileExecutionChoice &choice : requirements.tileStrategyChoices) {
    SmallVector<DFBConfigurationMask, 2> choiceMasks;
    for (const TileExecutionOption &option : choice.options) {
      DFBConfigurationMask optionMask = kAllDFBConfigurations;
      for (const DFBInputUse &use : option.dfbInputUses) {
        if (use.dfb == dfb) {
          optionMask &= getDFBConfigurationMask(target, use);
        }
      }
      appendUniqueMask(choiceMasks, optionMask);
    }

    SmallVector<DFBConfigurationMask, 4> nextAlternatives;
    for (DFBConfigurationMask priorMask : alternatives) {
      for (DFBConfigurationMask choiceMask : choiceMasks) {
        appendUniqueMask(nextAlternatives, priorMask & choiceMask);
      }
    }
    alternatives = std::move(nextAlternatives);
  }
  return {dfb, evidence, std::move(alternatives)};
}

bool canAlwaysShareDFBConfiguration(const DFBConfigurationProfile &lhs,
                                    const DFBConfigurationProfile &rhs) {
  constexpr DFBConfigurationMask kBits16Configurations = 0x3;
  constexpr DFBConfigurationMask kBits32Configurations = 0xC;
  for (DFBConfigurationMask lhsMask : lhs.alternatives) {
    for (DFBConfigurationMask rhsMask : rhs.alternatives) {
      for (DFBConfigurationMask widthMask :
           {kBits16Configurations, kBits32Configurations}) {
        DFBConfigurationMask lhsWidthMask = lhsMask & widthMask;
        DFBConfigurationMask rhsWidthMask = rhsMask & widthMask;
        if (lhsWidthMask != 0 && rhsWidthMask != 0 &&
            (lhsWidthMask & rhsWidthMask) == 0) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace

SmallVector<DFBConfigurationAliasConflict>
collectDFBConfigurationAliasConflicts(const KernelTargetEnvironment &target,
                                      const KernelRequirements &requirements) {
  llvm::MapVector<Value, Operation *> evidenceByDFB;
  auto collectUses = [&](ArrayRef<DFBInputUse> uses) {
    for (const DFBInputUse &use : uses) {
      evidenceByDFB.try_emplace(use.dfb, use.consumer);
    }
  };
  collectUses(requirements.dfbInputUses);
  for (const TileExecutionChoice &choice : requirements.tileStrategyChoices) {
    for (const TileExecutionOption &option : choice.options) {
      collectUses(option.dfbInputUses);
    }
  }

  SmallVector<DFBConfigurationProfile> profiles;
  profiles.reserve(evidenceByDFB.size());
  for (const auto &[dfb, evidence] : evidenceByDFB) {
    DFBConfigurationProfile profile =
        buildDFBConfigurationProfile(dfb, evidence, target, requirements);
    if (!profile.alternatives.empty()) {
      profiles.push_back(std::move(profile));
    }
  }

  SmallVector<DFBConfigurationAliasConflict> conflicts;
  for (unsigned lhsIndex = 0; lhsIndex < profiles.size(); ++lhsIndex) {
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < profiles.size();
         ++rhsIndex) {
      const DFBConfigurationProfile &lhs = profiles[lhsIndex];
      const DFBConfigurationProfile &rhs = profiles[rhsIndex];
      if (!canAlwaysShareDFBConfiguration(lhs, rhs)) {
        conflicts.push_back({lhs.dfb, rhs.dfb, lhs.evidence, rhs.evidence});
      }
    }
  }
  return conflicts;
}

namespace {

LogicalResult
validateFinalizedDFBIndices(const KernelRequirements &requirements) {
  auto validateUses = [](ArrayRef<DFBInputUse> uses) {
    for (const DFBInputUse &use : uses) {
      int32_t targetMaxDFBIndices = getTargetMaxDFBIndices(use.consumer);
      if (use.dfbIndex < 0 || use.dfbIndex >= targetMaxDFBIndices) {
        use.consumer->emitOpError()
            << "uses dataflow buffer index " << use.dfbIndex
            << " outside the supported range [0, " << targetMaxDFBIndices - 1
            << "] for " << getTargetDFBIndexCapacityDescription(use.consumer);
        return failure();
      }
    }
    return success();
  };

  if (failed(validateUses(requirements.dfbInputUses))) {
    return failure();
  }
  for (const TileExecutionChoice &choice : requirements.tileStrategyChoices) {
    for (const TileExecutionOption &option : choice.options) {
      if (failed(validateUses(option.dfbInputUses))) {
        return failure();
      }
    }
  }
  return success();
}

} // namespace

FailureOr<KernelConfigPlan> resolveKernelConfig(
    func::FuncOp function, const KernelTargetEnvironment &target,
    const KernelConfigPolicy &policy, const KernelRequirements &requirements) {
  if (failed(validateFinalizedDFBIndices(requirements))) {
    return failure();
  }
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
  ConfigConstraintResult fixedResult = applyConfigConstraints(
      initialState, target, policy, requirements.dfbInputUses,
      requirements.destinationUses);
  if (std::holds_alternative<ConfigConstraintConflict>(fixedResult)) {
    emitConfigConstraintConflict(
        function, std::get<ConfigConstraintConflict>(std::move(fixedResult)));
    return failure();
  }

  FailureOr<KernelTileStrategyOptions> allOptions =
      getTileStrategyOptions(requirements, policy);
  if (failed(allOptions)) {
    return failure();
  }
  StrategySearchState searchState{
      std::get<ConfigConstraintState>(std::move(fixedResult)),
      SmallVector<std::optional<TileExecutionStrategy>>(
          requirements.tileStrategyChoices.size())};
  StrategySearchResult searchResult = resolveTileStrategies(
      *allOptions, target, policy, std::move(searchState));
  if (std::holds_alternative<ConfigConstraintConflict>(searchResult)) {
    emitConfigConstraintConflict(
        function, std::get<ConfigConstraintConflict>(std::move(searchResult)));
    return failure();
  }
  StrategySearchState resolvedState =
      std::get<StrategySearchState>(std::move(searchResult));

  SmallVector<TileExecutionDecision> tileStrategies;
  tileStrategies.reserve(requirements.tileStrategyChoices.size());
  for (auto [choice, strategy] : llvm::zip_equal(
           requirements.tileStrategyChoices, resolvedState.selections)) {
    assert(strategy && "successful strategy search must select every op");
    tileStrategies.push_back({choice.operation, *strategy});
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
        return candidate.destinationElementWidth == destinationElementWidth;
      });
  assert(selectedCandidate != resolvedState.constraints.candidates.end() &&
         "resolved destination width must have a candidate");

  DstSyncMode syncMode = policy.dstSynchronization == ConfigSelection::Enabled
                             ? DstSyncMode::Full
                             : DstSyncMode::DoubleBuffered;

  if (policy.unpackToDestFp32) {
    return KernelConfigPlan(destinationElementWidth, syncMode,
                            *policy.unpackToDestFp32,
                            std::move(tileStrategies));
  }

  SmallVector<int32_t> unpackToDestFp32;
  for (const auto &[dfbIndex, modes] : selectedCandidate->unpackModes) {
    if (!modes.contains(DFBUnpackMode::Default) &&
        modes.contains(DFBUnpackMode::UnpackToDestination)) {
      unpackToDestFp32.push_back(static_cast<int32_t>(dfbIndex));
    }
  }
  llvm::sort(unpackToDestFp32);
  return KernelConfigPlan(destinationElementWidth, syncMode,
                          std::move(unpackToDestFp32),
                          std::move(tileStrategies));
}

void applyKernelConfigPlan(func::FuncOp function,
                           const KernelConfigPlan &plan) {
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
}

} // namespace mlir::tt::ttl
