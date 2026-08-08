// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H

#include "PipeCounter.h"
#include "PipeGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cstddef>
#include <optional>

namespace mlir::tt {
class ValueOriginAnalysis;
}

namespace mlir::tt::ttl {

class PipeTransferIndex;

inline constexpr llvm::StringLiteral kFabricRoutesAttrName =
    "ttl.fabric_routes";
inline constexpr llvm::StringLiteral kFabricDeviceDomainAttrName =
    "ttl.fabric_device_domain";

/// One logical device route used by `sourceNodes` in a kernel function.
/// `routeIndex` identifies the connection within `localDevice`.
struct FabricRoute {
  DeviceRefAttr localDevice;
  DeviceRefAttr remoteDevice;
  SmallVector<LaunchNodeCoord> sourceNodes;
  std::size_t routeIndex;
};

/// Fabric routes and their logical device domain for one kernel function.
struct FunctionFabricRoutePlan {
  DeviceDomainAttr deviceDomain;
  SmallVector<FabricRoute> routes;
};

/// Fabric routes and transfer associations derived before PipeNet lowering.
struct FabricRoutePlan {
  /// Routes grouped by the kernel function that submits each transfer.
  llvm::MapVector<func::FuncOp, FunctionFabricRoutePlan> routesByFunction;
  /// Connection indices in selected-record order. Static operations have one
  /// entry.
  llvm::MapVector<Operation *, SmallVector<std::size_t>> routeIndices;

  ArrayRef<std::size_t> lookupRouteIndices(Operation *operation) const {
    auto routeIt = routeIndices.find(operation);
    return routeIt == routeIndices.end()
               ? ArrayRef<std::size_t>()
               : ArrayRef<std::size_t>(routeIt->second);
  }
};

/// Per-function routing-plane state materialized before transfer lowering.
struct FabricRuntimeInfo {
  Value manager;
  Value routeId;
  Value connectionCount;
  std::size_t routeCount = 0;
};

/// Routing-plane state indexed by its kernel function.
using FabricRuntimeMap = llvm::DenseMap<Operation *, FabricRuntimeInfo>;

struct PipeInfo {
  PipeType pipeType;
  PipeTransferContract transferContract;
};

struct PipeSramAddressTableInfo {
  int64_t byteOffset;
};

/// Sender-side receiver address formula:
/// `base + slot(i) * blockStrideBytes + staticTileByteOffset`.
struct PipeComputedAddressInfo {
  int64_t receiverDFBIndex = 0;
  int64_t baseRuntimeCommonArgIndex = 0;
  /// Initial physical receiver DFB block assigned to this transfer.
  int64_t initialSlot = 0;
  /// `slot(i + 1) = (slot(i) + repeatStride) % blockCount`.
  int64_t repeatStride = 0;
  int64_t blockCount = 1;
  int64_t blockStrideBytes = 0;
  /// Byte offset for the destination tile within the selected DFB block.
  int64_t staticTileByteOffset = 0;
  std::optional<int64_t> dynamicSlotCounterIndex;

  /// Return whether the sender must track physical slot progress at runtime.
  bool usesDynamicSlotCounter() const {
    return dynamicSlotCounterIndex.has_value();
  }
};

enum class PipeAddressMode {
  ReceiverPublishedAddressTable,
  ComputedReceiverDFB,
};

struct PipeResourcePlan;
class PipeModulePlan;
class PipeTransferPlan;
class PipeCapacityPlan;
class PipeSynchronizationSelection;

/// Receiver-side completion state for one transfer definition.
struct PipeCompletionInfo {
  PipeCounterInfo counter;
};

/// Address storage used by one transfer-allocation unit.
struct PipeAddressStorageInfo {
  static PipeAddressStorageInfo
  receiverPublishedAddressTable(PipeSramAddressTableInfo sramAddressTable) {
    return PipeAddressStorageInfo{
        PipeAddressMode::ReceiverPublishedAddressTable, sramAddressTable,
        std::nullopt};
  }

  static PipeAddressStorageInfo
  computedReceiverDFB(PipeComputedAddressInfo computedAddress) {
    return PipeAddressStorageInfo{PipeAddressMode::ComputedReceiverDFB,
                                  std::nullopt, computedAddress};
  }

  bool usesComputedReceiverDFB() const {
    return mode == PipeAddressMode::ComputedReceiverDFB;
  }

  PipeAddressMode mode = PipeAddressMode::ReceiverPublishedAddressTable;
  std::optional<PipeSramAddressTableInfo> sramAddressTable;
  std::optional<PipeComputedAddressInfo> computedAddress;
};

/// Lowering information shared by one transfer definition's send, receiver
/// posts, and receiver waits.
/// Address storage and readiness synchronization are independent protocol
/// choices: computed addresses do not determine which ready counter is used.
struct PipeResourceInfo {
  PipeKey pipe;
  PipeTransferContract transferContract;
  PipeCompletionInfo completion;
  /// Absent when the transfer does not use receiver-post sender readiness.
  std::optional<PipeCounterInfo> readyCounter;
  PipeAddressStorageInfo addressStorage;
};

/// Kernel-local progress associated with one allocated PipeNet counter.
struct PipeCounterProgress {
  PipeCounterInfo counter;
  Value value;
};

/// Per-function cumulative progress values for PipeNet counters.
using PipeCounterProgressMap =
    llvm::MapVector<func::FuncOp, SmallVector<PipeCounterProgress>>;

struct PipeCounterTable {
  /// Indexed by `counters` so runtime record selection does not require one
  /// control-flow branch per transfer definition.
  Value values;
  SmallVector<PipeCounterInfo> counters;
};

/// Per-function counter tables indexed by a compile-time or selected record.
using PipeCounterTableMap = llvm::MapVector<func::FuncOp, PipeCounterTable>;

/// Initial value for one sender-local computed-address slot counter.
struct PipeComputedAddressCounterInitInfo {
  int64_t counterIndex = 0;
  int64_t initialSlot = 0;
};

/// Per-function table of sender-local computed-address slot counters.
using PipeComputedAddressCounterMap = llvm::MapVector<func::FuncOp, Value>;

/// pipeNetId -> deduplicated list of pipes in that net. Built once
/// before lowering so is_src/is_dst/is_active patterns avoid walking the
/// module per match.
using PipeNetIndex = llvm::MapVector<int64_t, SmallVector<PipeInfo>>;

struct PipeSramScratchInfo {
  int64_t bytes = 0;
};

enum class PipeResourceTableKind { Static, Selected };

/// Static resource allocation used by pipe lowering. Each protocol operation
/// maps to its transfer-specific completion, readiness, and address resources.
struct PipeResourcePlan {
  PipeSramScratchInfo sramScratch;
  /// Maps each pipe send, receiver post, and receiver wait to the resources
  /// shared by that transfer definition.
  llvm::MapVector<Operation *, PipeResourceInfo> resources;
  /// Record order is retained so runtime indices address the corresponding
  /// resources.
  llvm::MapVector<Operation *, SmallVector<PipeResourceInfo>> selectedResources;
  /// Protocol operations proven unreachable at their pipe endpoint. Lowering
  /// removes these operations without allocating synchronization resources.
  llvm::SmallPtrSet<Operation *, 8> staticallyInactiveOps;
  /// Each entry-block counter preserves slot state across repeated sends that
  /// share one computed-address allocation unit.
  llvm::MapVector<func::FuncOp, SmallVector<PipeComputedAddressCounterInitInfo>>
      computedAddressCounterInitializations;
  /// Receiver DFB indices supplied as common runtime arguments to each sender.
  llvm::MapVector<func::FuncOp, SmallVector<int32_t>> computedAddressDFBIndices;

  /// Visit each protocol operation and its complete resource table.
  LogicalResult forEachResourceTable(
      llvm::function_ref<LogicalResult(Operation *, ArrayRef<PipeResourceInfo>,
                                       PipeResourceTableKind)>
          callback) const;
};

/// Resource totals consumed by TTKernel lowering and runtime setup.
struct PipeResourceRequirements {
  int64_t syncSemaphoreCount = 0;
  int64_t globalSemaphoreCount = 0;
  int64_t sramScratchBytes = 0;
};

/// Return all pipe resource totals derived from the selected allocation plan.
PipeResourceRequirements
getPipeResourceRequirements(const PipeResourcePlan &info,
                            const PipeCapacityPlan *pipeCapacityPlan = nullptr);

/// Build and validate the high-level PipeNet declarations used by role
/// predicates. Duplicate records contribute one entry per transfer contract.
LogicalResult buildPipeNetIndex(ModuleOp mod, PipeNetIndex &index);

/// Build per-kernel routing-plane records from transfers validated by
/// PipeGraph.
LogicalResult buildFabricRoutePlan(const PipeTransferIndex &transferIndex,
                                   const PipeGraph &pipeGraph,
                                   FabricRoutePlan &plan);

/// Materialize the function attributes recorded by `plan`.
void applyFabricRoutePlan(ModuleOp module, const FabricRoutePlan &plan);

/// Materialize one routing-plane manager per kernel that uses fabric routes.
void initializeFabricRuntime(const FabricRoutePlan &plan,
                             FabricRuntimeMap &runtime);

/// Build the pipe resource plan used by pipe lowering. Transfer intervals that
/// cannot be bounded by dominance are conservatively treated as conflicting
/// with every other transfer interval from the same source core.
LogicalResult buildPipeResourcePlan(
    ModuleOp mod, const PipeTransferIndex &transferIndex,
    const PipeGraph &pipeGraph, PipeResourcePlan &info,
    bool enableComputedAddresses = true,
    PipeCounterAllocationPolicy counterPolicy =
        PipeCounterAllocationPolicy::LocalThenGlobal,
    const PipeSynchronizationSelection *synchronizationSelection = nullptr);

/// Initialize sender-side capacity counters and allocate one kernel-local
/// progress value per counter. The sender waits for the shared counter to reach
/// its cumulative acquired count, so only receivers increment the shared word.
void initializePipeCapacityCounters(
    const PipeCapacityPlan &pipeCapacityPlan,
    const PipeResourcePlan &pipeResourcePlan,
    PipeCounterProgressMap &senderCapacityCounters);

/// Allocate one kernel-local cumulative readiness value for every fabric
/// sender. Receivers are the only writers of the shared readiness counter.
void initializeFabricReadyCounters(const PipeModulePlan &pipeModulePlan,
                                   const PipeResourcePlan &pipeResourcePlan,
                                   PipeCounterTableMap &fabricReadyCounters);

/// Emit sender-local slot counters for computed receiver addresses whose
/// physical receiver DFB slot advances at runtime.
void initializePipeComputedAddressCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeComputedAddressCounterMap &computedAddressCounters);

/// At each receiver function entry, emit one zero-initialized sequence counter
/// for every completion counter used by that function.
void initializePipePostSequenceCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeCounterTableMap &postSequenceCounters);

/// Remove a sender operation proven unreachable at its pipe endpoint.
void lowerInactivePipeTransferSend(PipeTransferSendOp op,
                                   ConversionPatternRewriter &rewriter);

/// Lower the sender-side pipe transfer and signal receiver completion.
LogicalResult lowerPipeTransferSend(
    PipeTransferSendOp op, Value srcCB, const PipeTransferPlan &transferPlan,
    const PipeResourcePlan &pipeResourcePlan,
    const PipeCapacityPlan &pipeCapacityPlan,
    const PipeCounterProgressMap &senderCapacityCounters,
    const PipeCounterTableMap &fabricReadyCounters,
    const PipeComputedAddressCounterMap &computedAddressCounters,
    const FabricRuntimeMap &fabricRuntime, ConversionPatternRewriter &rewriter);

/// Remove a receiver post proven unreachable at its pipe endpoint.
void lowerInactivePipeTransferPost(PipeTransferPostOp op,
                                   ConversionPatternRewriter &rewriter);

LogicalResult lowerPipeTransferPost(
    PipeTransferPostOp op, Value dst, const PipeTransferPlan &transferPlan,
    const PipeCounterTableMap &postSequenceCounters,
    const PipeResourcePlan &pipeResourcePlan,
    const FabricRuntimeMap &fabricRuntime, ConversionPatternRewriter &rewriter);

/// Lower a dataflow buffer pop and emit any proven pipe capacity releases.
LogicalResult lowerCBPop(CBPopOp op, Value cb,
                         const PipeCapacityPlan &pipeCapacityPlan,
                         const PipeResourcePlan &pipeResourcePlan,
                         ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe receive completion wait.
LogicalResult lowerPipeTransferWait(PipeTransferWaitOp op, Value tokenSequence,
                                    const PipeTransferPlan &transferPlan,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter);

/// Add pipe-specific lowering patterns (IfSrc, IfDst, CreatePipe) to the set.
/// `pipeNetIndex` is borrowed and must outlive `patterns`; the is_src /
/// is_dst / is_active lowerings use it for O(1) net-id lookup.
void populatePipeLoweringPatterns(RewritePatternSet &patterns,
                                  const TypeConverter &typeConverter,
                                  const PipeNetIndex &pipeNetIndex);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H
