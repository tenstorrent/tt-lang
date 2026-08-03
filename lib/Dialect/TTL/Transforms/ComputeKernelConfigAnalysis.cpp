// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/ComputeKernelConfigAnalysis.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTCore/IR/Utils.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
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

  if (value.getDefiningOp() || isa<BlockArgument>(value)) {
    return std::optional<int64_t>();
  }
  consumer->emitOpError("has an unresolved tile operand");
  return failure();
}

/// Return the element type required by configuration queries.
FailureOr<Type> getRequiredTileElementType(Value value, Operation *operation) {
  std::optional<Type> elementType = getTileElementType(value.getType());
  if (!elementType) {
    operation->emitOpError()
        << "expected a tile operand or result, got " << value.getType();
    return failure();
  }
  return *elementType;
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
                            SmallVectorImpl<DstModeUse> &dstModeUses) {
  for (OpOperand &operand : operation->getOpOperands()) {
    TileOperandRoute route = info.operandRoutes[operand.getOperandNumber()];
    if (route == TileOperandRoute::None) {
      continue;
    }
    FailureOr<Type> elementType =
        getRequiredTileElementType(operand.get(), operation);
    FailureOr<std::optional<int64_t>> dataflowBufferIndex =
        resolveDataflowBufferIndex(operand.get(), operation);
    if (failed(elementType) || failed(dataflowBufferIndex)) {
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
      dstModeUses.push_back({operation, info.primitive, *elementType});
    }
  }

  if (!info.resultInDst) {
    return success();
  }
  for (Value result : operation->getResults()) {
    if (!isa<ttcore::TileType>(result.getType())) {
      continue;
    }
    FailureOr<Type> elementType = getRequiredTileElementType(result, operation);
    if (failed(elementType)) {
      return failure();
    }
    dstModeUses.push_back({operation, info.primitive, *elementType});
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

  Attribute rawStrategy = operation->getAttr(kTileExecutionStrategyAttrName);
  auto selectedAttr = dyn_cast_or_null<TileExecutionStrategyAttr>(rawStrategy);
  if (rawStrategy && !selectedAttr) {
    operation->emitOpError()
        << kTileExecutionStrategyAttrName
        << " must be a #ttl.tile_execution_strategy attribute";
    return failure();
  }
  if (selectedAttr &&
      !llvm::is_contained(legalStrategies, selectedAttr.getValue())) {
    operation->emitOpError() << "explicit " << kTileExecutionStrategyAttrName
                             << " is not legal for its operands";
    return failure();
  }

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
            operation, *info, option.dfbInputUses, option.dstModeUses))) {
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

struct DstModeEvidence {
  Operation *operation;
  std::optional<DstModeUse> use;
};

/// Diagnose the two constraints that emptied the kernel-wide DST domain.
void emitDstModeConflict(func::FuncOp function, DstModeEvidence requiresFp32,
                         DstModeEvidence requiresDefault) {
  if (!requiresDefault.use) {
    requiresFp32.operation->emitOpError(
        "requires f32 DST mode, but fp32 destination accumulation is "
        "explicitly disabled");
    return;
  }

  InFlightDiagnostic diagnostic =
      requiresFp32.use
          ? requiresFp32.operation->emitOpError(
                "requires f32 DST mode, but no kernel-wide DST mode supports "
                "all tile operations")
          : function.emitOpError(
                "explicit f32 destination accumulation is unsupported by "
                "the kernel's tile operations");
  diagnostic.attachNote(requiresDefault.operation->getLoc())
      << "the target does not support f32 DST mode for "
      << requiresDefault.operation->getName() << " with "
      << requiresDefault.use->elementType << " elements";
}

struct DstModeConflict {
  DstModeEvidence requiresFp32;
  DstModeEvidence requiresDefault;
};

struct ExplicitUnpackConflict {
  DFBInputUse use;
  DFBUnpackMode configuredMode;
  DFBUnpackMode requiredMode;
};

struct DFBUnpackConflict {
  DFBInputUse firstUse;
  DFBInputUse secondUse;
};

using ConfigConstraintConflict =
    std::variant<DstModeConflict, ExplicitUnpackConflict, DFBUnpackConflict>;

struct ConfigConstraintState {
  llvm::SmallSet<DstMode, 2> dstModes{DstMode::Default, DstMode::Fp32};
  DstModeEvidence requiresFp32{nullptr, std::nullopt};
  DstModeEvidence requiresDefault{nullptr, std::nullopt};
  llvm::MapVector<int64_t, DFBUnpackMode> unpackModes;
  llvm::DenseMap<int64_t, DFBInputUse> firstUnpackUses;
};

using ConfigConstraintResult =
    std::variant<ConfigConstraintState, ConfigConstraintConflict>;

/// Intersect one execution option's requirements with the current domains.
ConfigConstraintResult applyConfigConstraints(
    ConfigConstraintState state, const KernelTargetEnvironment &target,
    const KernelConfigPolicy &policy, ArrayRef<DFBInputUse> dfbInputUses,
    ArrayRef<DstModeUse> dstModeUses) {
  for (const DstModeUse &use : dstModeUses) {
    if (use.elementType.isF32()) {
      state.dstModes.erase(DstMode::Default);
      if (!state.requiresFp32.operation) {
        state.requiresFp32 = {use.operation, use};
      }
    }
    if (!target.supportsDstMode(use.primitive, use.elementType,
                                DstMode::Fp32)) {
      state.dstModes.erase(DstMode::Fp32);
      if (!state.requiresDefault.operation) {
        state.requiresDefault = {use.operation, use};
      }
    }
    if (state.dstModes.empty()) {
      return ConfigConstraintConflict(
          DstModeConflict{state.requiresFp32, state.requiresDefault});
    }
  }

  for (const DFBInputUse &use : dfbInputUses) {
    DFBUnpackMode requiredMode =
        target.getRequiredUnpackMode(use.primitive, use.route, use.elementType);
    if (policy.unpackToDestFp32) {
      bool configuresFp32 =
          llvm::is_contained(*policy.unpackToDestFp32, use.dfbIndex);
      DFBUnpackMode configuredMode = configuresFp32
                                         ? DFBUnpackMode::UnpackToDestFp32
                                         : DFBUnpackMode::Default;
      if (configuredMode != requiredMode) {
        return ConfigConstraintConflict(
            ExplicitUnpackConflict{use, configuredMode, requiredMode});
      }
      continue;
    }

    auto [iterator, inserted] =
        state.unpackModes.insert({use.dfbIndex, requiredMode});
    if (inserted) {
      state.firstUnpackUses.insert({use.dfbIndex, use});
      continue;
    }
    if (iterator->second != requiredMode) {
      return ConfigConstraintConflict(
          DFBUnpackConflict{state.firstUnpackUses.lookup(use.dfbIndex), use});
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
        if constexpr (std::is_same_v<ConflictType, DstModeConflict>) {
          emitDstModeConflict(function, typedConflict.requiresFp32,
                              typedConflict.requiresDefault);
        } else if constexpr (std::is_same_v<ConflictType,
                                            ExplicitUnpackConflict>) {
          const DFBInputUse &use = typedConflict.use;
          use.consumer->emitOpError()
              << "dataflow buffer " << use.dfbIndex << " requires "
              << (typedConflict.requiredMode == DFBUnpackMode::UnpackToDestFp32
                      ? "unpack-to-DST-f32 mode, but "
                      : "default unpack mode, but ")
              << kUnpackToDestFp32AttrName
              << (typedConflict.configuredMode ==
                          DFBUnpackMode::UnpackToDestFp32
                      ? " includes this index"
                      : " excludes this index");
        } else {
          InFlightDiagnostic diagnostic =
              typedConflict.secondUse.consumer->emitOpError()
              << "dataflow buffer " << typedConflict.secondUse.dfbIndex
              << " requires incompatible unpack modes in one kernel";
          diagnostic.attachNote(typedConflict.firstUse.consumer->getLoc())
              << "operand " << typedConflict.firstUse.operandIndex
              << " establishes the conflicting unpack mode";
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
                                 option.dfbInputUses, option.dstModeUses);
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
                               option.dfbInputUses, option.dstModeUses);
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

} // namespace

FailureOr<KernelTargetEnvironment>
KernelTargetEnvironment::get(func::FuncOp function) {
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module) {
    function.emitOpError("is not nested in a module");
    return failure();
  }

  Attribute rawTarget = module->getAttr(kTargetArchAttrName);
  auto targetAttr = dyn_cast_or_null<ttcore::ArchAttr>(rawTarget);
  if (rawTarget && !targetAttr) {
    module.emitOpError() << kTargetArchAttrName
                         << " must be a #ttcore.arch attribute";
    return failure();
  }
  std::optional<ttcore::Arch> targetArch;
  if (targetAttr) {
    targetArch = targetAttr.getValue();
  }

  auto systemDesc = module->getAttrOfType<ttcore::SystemDescAttr>(
      ttcore::SystemDescAttr::name);
  auto device =
      module.lookupSymbol<ttcore::DeviceOp>(ttcore::getDefaultDeviceName());
  if (systemDesc && device) {
    ArrayRef<unsigned> chipIds = device.getDeviceAttr().getChipIds();
    if (chipIds.empty()) {
      device.emitOpError("has no selected chip");
      return failure();
    }
    auto invalidChip = llvm::find_if(chipIds, [&](unsigned chipId) {
      return chipId >= systemDesc.getChipDescIndices().size();
    });
    if (invalidChip != chipIds.end()) {
      device.emitOpError() << "selects chip " << *invalidChip
                           << " outside the system description";
      return failure();
    }
    ttcore::Arch deviceArch =
        systemDesc.getChipDesc(chipIds.front()).getArch().getValue();
    if (llvm::any_of(llvm::drop_begin(chipIds), [&](unsigned chipId) {
          return systemDesc.getChipDesc(chipId).getArch().getValue() !=
                 deviceArch;
        })) {
      device.emitOpError("selects chips with different architectures");
      return failure();
    }
    if (targetArch && *targetArch != deviceArch) {
      module.emitOpError() << kTargetArchAttrName
                           << " does not match the selected device arch";
      return failure();
    }
    targetArch = deviceArch;
  }

  return KernelTargetEnvironment(targetArch);
}

bool KernelTargetEnvironment::supportsDstMode(TilePrimitive primitive,
                                              Type elementType,
                                              DstMode mode) const {
  if (mode == DstMode::Default) {
    return true;
  }
  if (primitive == TilePrimitive::BroadcastRow && !elementType.isF32() &&
      (!arch || *arch == ttcore::Arch::WormholeB0)) {
    // tt-llk #1338: Wormhole row broadcast is incorrect for bf16 input when
    // DST stores f32 values.
    return false;
  }
  return true;
}

bool KernelTargetEnvironment::supportsFullFp32Accumulation(
    FullFp32AccumulationKind kind) const {
  if (kind == FullFp32AccumulationKind::Matmul) {
    return true;
  }
  if (arch == ttcore::Arch::WormholeB0) {
    return false;
  }
  return arch != ttcore::Arch::Blackhole ||
         kind != FullFp32AccumulationKind::ReduceRow;
}

DFBUnpackMode KernelTargetEnvironment::getRequiredUnpackMode(
    TilePrimitive primitive, TileOperandRoute route, Type elementType) const {
  if (elementType.isF32() &&
      (route == TileOperandRoute::Dst || primitive == TilePrimitive::Copy)) {
    return DFBUnpackMode::UnpackToDestFp32;
  }
  return DFBUnpackMode::Default;
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
  WalkResult result = function.walk([&](TileExecutionOpInterface executionOp) {
    Operation *operation = executionOp.getOperation();
    if (!executionOp.getLegalExecutionStrategies().empty()) {
      FailureOr<TileExecutionChoice> choice =
          getTileExecutionChoice(executionOp);
      if (failed(choice)) {
        return WalkResult::interrupt();
      }
      requirements.tileStrategyChoices.push_back(std::move(*choice));
      return WalkResult::advance();
    }

    if (operation->hasAttr(kTileExecutionStrategyAttrName)) {
      operation->emitOpError()
          << kTileExecutionStrategyAttrName
          << " is only valid on tile operations with execution-strategy "
             "alternatives";
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
                                           requirements.dstModeUses))) {
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
    initialState.dstModes.erase(DstMode::Default);
    initialState.requiresFp32 = {function.getOperation(), std::nullopt};
  } else if (policy.fp32DestAccumulation == ConfigSelection::Disabled) {
    initialState.dstModes.erase(DstMode::Fp32);
    initialState.requiresDefault = {function.getOperation(), std::nullopt};
  }

  ConfigConstraintResult fixedResult = applyConfigConstraints(
      initialState, target, policy, requirements.dfbInputUses,
      requirements.dstModeUses);
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
  for (const FullFp32AccumulationUse &use :
       requirements.fullFp32AccumulationUses) {
    bool isMatmul = use.kind == FullFp32AccumulationKind::Matmul;
    bool preferred =
        isMatmul ? policy.preferFullFp32Matmul : policy.preferFullFp32Reduce;
    if (!preferred) {
      continue;
    }
    if (target.supportsFullFp32Accumulation(use.kind)) {
      preferFp32 = true;
      continue;
    }
    if (target.getArch() == ttcore::Arch::Blackhole &&
        use.kind == FullFp32AccumulationKind::ReduceRow) {
      use.operation->emitWarning()
          << "full-fp32 row reduce is unavailable on Blackhole (tt-metal "
             "#47311); using non-full-fp32 reduce lowering";
    }
  }
  DstMode dstMode = DstMode::Default;
  if ((preferFp32 ||
       !resolvedState.constraints.dstModes.contains(DstMode::Default)) &&
      resolvedState.constraints.dstModes.contains(DstMode::Fp32)) {
    dstMode = DstMode::Fp32;
  }

  DstSyncMode syncMode = policy.dstSynchronization == ConfigSelection::Enabled
                             ? DstSyncMode::Full
                             : DstSyncMode::DoubleBuffered;

  if (policy.unpackToDestFp32) {
    return KernelConfigPlan{dstMode, syncMode, *policy.unpackToDestFp32,
                            std::move(tileStrategies)};
  }

  SmallVector<int32_t> unpackToDestFp32;
  for (auto [dfbIndex, mode] : resolvedState.constraints.unpackModes) {
    if (mode == DFBUnpackMode::UnpackToDestFp32) {
      unpackToDestFp32.push_back(static_cast<int32_t>(dfbIndex));
    }
  }
  llvm::sort(unpackToDestFp32);
  return KernelConfigPlan{dstMode, syncMode, std::move(unpackToDestFp32),
                          std::move(tileStrategies)};
}

LogicalResult applyKernelConfigPlan(func::FuncOp function,
                                    const KernelConfigPlan &plan) {
  if (!std::is_sorted(plan.unpackToDestFp32.begin(),
                      plan.unpackToDestFp32.end()) ||
      std::adjacent_find(plan.unpackToDestFp32.begin(),
                         plan.unpackToDestFp32.end()) !=
          plan.unpackToDestFp32.end() ||
      llvm::any_of(plan.unpackToDestFp32, [](int32_t index) {
        return index < 0 || index >= kMaxCircularBuffers;
      })) {
    function.emitOpError(
        "kernel configuration plan contains invalid dataflow buffer indices");
    return failure();
  }

  llvm::SmallPtrSet<Operation *, 8> plannedOperations;
  for (const TileExecutionDecision &decision : plan.tileStrategies) {
    if (!decision.operation || !function->isAncestor(decision.operation)) {
      function.emitOpError(
          "tile execution plan refers to an operation outside the kernel");
      return failure();
    }
    if (!plannedOperations.insert(decision.operation).second) {
      decision.operation->emitOpError(
          "tile execution plan contains duplicate strategy decisions");
      return failure();
    }
    auto executionOp = dyn_cast<TileExecutionOpInterface>(decision.operation);
    if (!executionOp) {
      decision.operation->emitOpError(
          "tile execution plan records a strategy for an unsupported op");
      return failure();
    }
    if (!llvm::is_contained(executionOp.getLegalExecutionStrategies(),
                            decision.strategy)) {
      decision.operation->emitOpError(
          "tile execution plan is inconsistent with the operation operands");
      return failure();
    }
    FailureOr<TileExecutionInfo> info =
        executionOp.getTileExecutionInfo(decision.strategy);
    if (failed(info)) {
      decision.operation->emitOpError(
          "tile execution plan selects incomplete execution semantics");
      return failure();
    }
    if (failed(verifyTileExecutionInfo(decision.operation, *info))) {
      return failure();
    }
  }

  WalkResult completeness =
      function.walk([&](TileExecutionOpInterface executionOp) {
        if (executionOp.getLegalExecutionStrategies().empty() ||
            plannedOperations.contains(executionOp.getOperation())) {
          return WalkResult::advance();
        }
        executionOp->emitOpError(
            "tile execution plan does not select an execution strategy");
        return WalkResult::interrupt();
      });
  if (completeness.wasInterrupted()) {
    return failure();
  }

  MLIRContext *context = function.getContext();
  for (const TileExecutionDecision &decision : plan.tileStrategies) {
    decision.operation->setAttr(
        kTileExecutionStrategyAttrName,
        TileExecutionStrategyAttr::get(context, decision.strategy));
  }
  function->setAttr(kFp32DestAccEnAttrName,
                    BoolAttr::get(context, plan.dstMode == DstMode::Fp32));
  function->setAttr(
      kDstFullSyncEnAttrName,
      BoolAttr::get(context, plan.dstSyncMode == DstSyncMode::Full));
  function->setAttr(kUnpackToDestFp32AttrName,
                    DenseI32ArrayAttr::get(context, plan.unpackToDestFp32));
  function->removeAttr(kEnableFPUBinaryOpsAttrName);
  return success();
}

} // namespace mlir::tt::ttl
