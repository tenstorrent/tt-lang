// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEKERNELCONFIGANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEKERNELCONFIGANALYSIS_H

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TileExecution.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace mlir::tt::ttl {

/// User policy for a configuration value that may otherwise be inferred.
enum class ConfigSelection {
  Auto,
  Enabled,
  Disabled,
};

/// Element format used by the kernel's DST register file.
enum class DstMode {
  Default,
  Fp32,
};

/// Synchronization protocol used for the kernel's DST register file.
enum class DstSyncMode {
  DoubleBuffered,
  Full,
};

/// Unpack mode assigned to one dataflow buffer for the complete kernel.
enum class DFBUnpackMode {
  Default,
  UnpackToDestFp32,
};

/// Immutable hardware and backend capabilities for one kernel target.
class KernelTargetEnvironment {
public:
  static FailureOr<KernelTargetEnvironment> get(func::FuncOp function);

  std::optional<ttcore::Arch> getArch() const { return arch; }
  bool supportsDstMode(TilePrimitive primitive, Type elementType,
                       DstMode mode) const;
  bool supportsFullFp32Accumulation(FullFp32AccumulationKind kind) const;
  DFBUnpackMode getRequiredUnpackMode(TilePrimitive primitive,
                                      TileOperandRoute route,
                                      Type elementType) const;

private:
  explicit KernelTargetEnvironment(std::optional<ttcore::Arch> arch)
      : arch(arch) {}

  std::optional<ttcore::Arch> arch;
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

/// One operand or result that requires a kernel-wide DST mode.
struct DstModeUse {
  Operation *operation;
  TilePrimitive primitive;
  Type elementType;
};

/// Requirements imposed by one legal execution strategy.
struct TileExecutionOption {
  TileExecutionStrategy strategy;
  llvm::SmallVector<DFBInputUse> dfbInputUses;
  llvm::SmallVector<DstModeUse> dstModeUses;
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

/// Target-independent requirements collected from immutable TTL IR.
struct KernelRequirements {
  llvm::SmallVector<DFBInputUse> dfbInputUses;
  llvm::SmallVector<DstModeUse> dstModeUses;
  llvm::SmallVector<FullFp32AccumulationUse> fullFp32AccumulationUses;
  llvm::SmallVector<TileExecutionChoice, 0> tileStrategyChoices;
};

/// Complete configuration selected before any IR mutation. Apply the plan
/// before mutating IR because its decisions contain operation pointers.
class KernelConfigPlan {
public:
  DstMode getDstMode() const { return dstMode; }
  DstSyncMode getDstSyncMode() const { return dstSyncMode; }
  llvm::ArrayRef<int32_t> getUnpackToDestFp32() const {
    return unpackToDestFp32;
  }
  llvm::ArrayRef<TileExecutionDecision> getTileStrategies() const {
    return tileStrategies;
  }

private:
  friend FailureOr<KernelConfigPlan> resolveKernelConfig(
      func::FuncOp function, const KernelTargetEnvironment &target,
      const KernelConfigPolicy &policy, const KernelRequirements &requirements);

  KernelConfigPlan(DstMode dstMode, DstSyncMode dstSyncMode,
                   llvm::SmallVector<int32_t> unpackToDestFp32,
                   llvm::SmallVector<TileExecutionDecision> tileStrategies)
      : dstMode(dstMode), dstSyncMode(dstSyncMode),
        unpackToDestFp32(std::move(unpackToDestFp32)),
        tileStrategies(std::move(tileStrategies)) {}

  DstMode dstMode;
  DstSyncMode dstSyncMode;
  llvm::SmallVector<int32_t> unpackToDestFp32;
  llvm::SmallVector<TileExecutionDecision> tileStrategies;
};

/// Collect target-independent requirements and legal tile strategies.
FailureOr<KernelRequirements> collectKernelRequirements(func::FuncOp function);

/// Resolve one complete configuration from target, policy, and requirements.
FailureOr<KernelConfigPlan> resolveKernelConfig(
    func::FuncOp function, const KernelTargetEnvironment &target,
    const KernelConfigPolicy &policy, const KernelRequirements &requirements);

/// Apply a resolved plan without deriving additional configuration policy.
void applyKernelConfigPlan(func::FuncOp function, const KernelConfigPlan &plan);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEKERNELCONFIGANALYSIS_H
