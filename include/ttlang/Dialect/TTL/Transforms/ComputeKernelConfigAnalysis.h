// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEKERNELCONFIGANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEKERNELCONFIGANALYSIS_H

#include "ttlang/Dialect/TTL/IR/TileExecution.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::tt::ttl {

/// User policy for a configuration value that may otherwise be inferred.
enum class ConfigSelection {
  Auto,
  Enabled,
  Disabled,
};

/// Physical element width used by the kernel's destination register file.
/// Tensix exposes only 16-bit and 32-bit destination element configurations.
enum class DestinationElementWidth {
  Bits16,
  Bits32,
};

/// Synchronization protocol used for the kernel's DST register file.
enum class DstSyncMode {
  DoubleBuffered,
  Full,
};

/// Per-DFB unpack selection shared by the complete kernel. A target defines
/// when each selection changes the primitive's physical operand route.
enum class DFBUnpackMode {
  Default,
  UnpackToDestination,
};

/// Target support and fallback warning for a preferred accumulation mode.
struct FullFp32AccumulationSupport {
  bool supported;
  std::optional<StringRef> fallbackWarning;
};

/// One target-supported combination of shared destination width and per-DFB
/// unpack selection.
struct DFBHardwareConfiguration {
  DestinationElementWidth destinationElementWidth;
  DFBUnpackMode unpackMode;
};

/// One compute-local facility required by a pipeline schedule.
enum class ComputePipelineCapability {
  MultiplyFullScalarReduction,
  SourceScalarRetention,
};

/// Target and element-type support for one compute-local facility.
struct ComputePipelineCapabilityLimits {
  bool targetSupported = false;
  std::optional<std::uint32_t> maxTiles;
};

/// Immutable hardware and backend capabilities for one kernel target.
class KernelTargetEnvironment {
public:
  virtual ~KernelTargetEnvironment() = default;

  static FailureOr<std::unique_ptr<KernelTargetEnvironment>>
  get(func::FuncOp function);

  virtual bool supportsDestinationElementWidth(
      TilePrimitive primitive, Type elementType,
      DestinationElementWidth destinationElementWidth) const = 0;
  virtual FullFp32AccumulationSupport
  getFullFp32AccumulationSupport(FullFp32AccumulationKind kind) const = 0;
  virtual ComputePipelineCapabilityLimits
  getComputePipelineCapabilityLimits(ComputePipelineCapability capability,
                                     Type elementType) const = 0;
  virtual llvm::SmallVector<DFBHardwareConfiguration, 4>
  getSupportedDFBConfigurations(TilePrimitive primitive, TileOperandRoute route,
                                Type elementType) const = 0;

protected:
  KernelTargetEnvironment() = default;
};

/// Explicit configuration constraints normalized without inspecting tile ops.
struct KernelConfigPolicy {
  ConfigSelection fp32DestAccumulation = ConfigSelection::Auto;
  ConfigSelection dstSynchronization = ConfigSelection::Auto;
  bool preferFullFp32Reduce = true;
  bool preferFullFp32Matmul = true;
  bool allowFPUBinary = true;
  std::optional<llvm::SmallVector<int32_t>> unpackToDestFp32;

  static FailureOr<KernelConfigPolicy>
  get(func::FuncOp function, StringRef fp32DestAccumulation,
      StringRef dstSynchronization, bool preferFullFp32Reduce,
      bool preferFullFp32Matmul, bool allowFPUBinary);
};

/// Strategy selected for one tile operation with alternatives.
struct TileExecutionDecision {
  Operation *operation;
  TileExecutionStrategy strategy;
};

/// One dataflow-buffer operand and the facts used to determine unpack mode.
struct DFBInputUse {
  int64_t dfbIndex;
  Operation *consumer;
  unsigned operandIndex;
  TilePrimitive primitive;
  TileOperandRoute route;
  Type elementType;
};

/// One operand or result that constrains the shared destination width.
struct DestinationUse {
  Operation *operation;
  TilePrimitive primitive;
  Type elementType;
};

/// Requirements imposed by one legal execution strategy.
struct TileExecutionOption {
  TileExecutionStrategy strategy;
  llvm::SmallVector<DFBInputUse> dfbInputUses;
  llvm::SmallVector<DestinationUse> destinationUses;
};

/// Legal strategy alternatives retained for kernel-wide resolution.
struct TileExecutionChoice {
  Operation *operation;
  llvm::SmallVector<TileExecutionOption, 2> options;
  bool hasExplicitStrategy = false;
};

/// One operation eligible for optional full-fp32 accumulation.
struct FullFp32AccumulationUse {
  Operation *operation;
  FullFp32AccumulationKind kind;
};

/// Fixed operation requirements for target facilities and DST residency.
struct ComputeResourceUse {
  Operation *operation;
  llvm::SmallVector<ComputePipelineCapability, 2> requiredCapabilities;
  Type elementType;
  std::uint64_t requiredDstSlots;
};

/// One exact compute-stage operand that consumes a retained full scalar.
struct SourceScalarConsumerPlan {
  Operation *stage;
  unsigned operandIndex;
};

/// One stage and its exact tensor inputs at planning time.
struct SourceScalarStagePlan {
  Operation *stage;
  llvm::SmallVector<Value> inputs;
};

/// Immutable producer-consumer split for one source-scalar lifetime.
struct SourceScalarRetentionPlan {
  Operation *producerStage;
  unsigned producerResult;
  llvm::SmallVector<SourceScalarStagePlan> producerStages;
  llvm::SmallVector<SourceScalarStagePlan> consumerStages;
  llvm::SmallVector<SourceScalarConsumerPlan> consumers;
};

using SourceScalarRetentionPlanPtr =
    std::shared_ptr<const SourceScalarRetentionPlan>;

/// Requirements for one legal compute-pipeline schedule.
struct ComputePipelineScheduleOption {
  ComputePipelineKind kind;
  ComputePipelineSchedule schedule;
  llvm::SmallVector<ComputePipelineCapability, 2> requiredCapabilities;
  llvm::SmallVector<DFBInputUse> dfbInputUses;
  llvm::SmallVector<DestinationUse> destinationUses;
  std::uint32_t requiredDstSlots = 0;
  SourceScalarRetentionPlanPtr sourceScalar;
};

/// Schedule alternatives retained for kernel-wide resolution.
struct ComputePipelineScheduleChoice {
  Operation *pipeline;
  llvm::SmallVector<ComputePipelineScheduleOption, 2> options;
};

enum class ComputePipelineScheduleRejectionKind {
  UnsupportedTarget,
  UnsupportedElementType,
  DSTCapacity,
  KernelConfigurationConflict,
};

/// Reason the preferred retained-scalar schedule was not selected.
struct ComputePipelineScheduleRejection {
  ComputePipelineScheduleRejectionKind kind;
  std::uint32_t requiredDstSlots = 0;
  std::uint32_t availableDstSlots = 0;
};

/// Schedule selected for one semantic compute pipeline.
struct ComputePipelineScheduleDecision {
  Operation *pipeline;
  ComputePipelineSchedule schedule;
  std::optional<ComputePipelineScheduleRejection> rejection;
  SourceScalarRetentionPlanPtr sourceScalar;
};

/// Target-independent requirements collected from immutable TTL IR.
struct KernelRequirements {
  llvm::SmallVector<DFBInputUse> dfbInputUses;
  llvm::SmallVector<DestinationUse> destinationUses;
  llvm::SmallVector<FullFp32AccumulationUse> fullFp32AccumulationUses;
  llvm::SmallVector<ComputeResourceUse> computeResourceUses;
  llvm::SmallVector<TileExecutionChoice, 0> tileStrategyChoices;
  llvm::SmallVector<ComputePipelineScheduleChoice, 0> pipelineScheduleChoices;
};

/// Complete configuration selected before any IR mutation. Apply the plan
/// before mutating IR because its decisions contain operation pointers.
class KernelConfigPlan {
public:
  DestinationElementWidth getDestinationElementWidth() const {
    return destinationElementWidth;
  }
  DstSyncMode getDstSyncMode() const { return dstSyncMode; }
  llvm::ArrayRef<int32_t> getUnpackToDestFp32() const {
    return unpackToDestFp32;
  }
  llvm::ArrayRef<TileExecutionDecision> getTileStrategies() const {
    return tileStrategies;
  }
  llvm::ArrayRef<ComputePipelineScheduleDecision>
  getComputePipelineSchedules() const {
    return computePipelineSchedules;
  }

private:
  friend FailureOr<KernelConfigPlan> resolveKernelConfig(
      func::FuncOp function, const KernelTargetEnvironment &target,
      const KernelConfigPolicy &policy, const KernelRequirements &requirements);

  KernelConfigPlan(DestinationElementWidth destinationElementWidth,
                   DstSyncMode dstSyncMode,
                   llvm::SmallVector<int32_t> unpackToDestFp32,
                   llvm::SmallVector<TileExecutionDecision> tileStrategies,
                   llvm::SmallVector<ComputePipelineScheduleDecision>
                       computePipelineSchedules)
      : destinationElementWidth(destinationElementWidth),
        dstSyncMode(dstSyncMode), unpackToDestFp32(std::move(unpackToDestFp32)),
        tileStrategies(std::move(tileStrategies)),
        computePipelineSchedules(std::move(computePipelineSchedules)) {}

  DestinationElementWidth destinationElementWidth;
  DstSyncMode dstSyncMode;
  llvm::SmallVector<int32_t> unpackToDestFp32;
  llvm::SmallVector<TileExecutionDecision> tileStrategies;
  llvm::SmallVector<ComputePipelineScheduleDecision> computePipelineSchedules;
};

/// Collect target-independent requirements and legal tile strategies.
FailureOr<KernelRequirements> collectKernelRequirements(func::FuncOp function);

/// Resolve one complete configuration from target, policy, and requirements.
FailureOr<KernelConfigPlan> resolveKernelConfig(
    func::FuncOp function, const KernelTargetEnvironment &target,
    const KernelConfigPolicy &policy, const KernelRequirements &requirements);

/// Apply a resolved plan without deriving additional configuration policy.
LogicalResult applyKernelConfigPlan(func::FuncOp function,
                                    const KernelConfigPlan &plan);

/// Apply only compute pipeline schedule decisions from a resolved plan.
LogicalResult applyComputePipelineSchedulePlan(func::FuncOp function,
                                               const KernelConfigPlan &plan);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEKERNELCONFIGANALYSIS_H
