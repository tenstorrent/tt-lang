// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H

#include "PipeGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"

#include <optional>

namespace mlir::tt::ttl {

inline constexpr llvm::StringLiteral kFabricRoutesAttrName =
    "ttl.fabric_routes";

struct FabricRoute {
  DeviceRefAttr localDevice;
  DeviceRefAttr remoteDevice;
  SmallVector<SmallVector<int64_t>> sourceNodes;
  unsigned routeIndex;
};

struct FabricRoutePlan {
  llvm::MapVector<func::FuncOp, SmallVector<FabricRoute>> routesByFunction;
  llvm::DenseMap<Operation *, unsigned> sendRouteIndex;
  llvm::DenseMap<Operation *, unsigned> postRouteIndex;
};

struct FabricRouteTarget {
  Value destinationDeviceId;
  Value destinationMeshId;
};

struct FabricRuntimeInfo {
  Value manager;
  Value routeId;
  SmallVector<FabricRouteTarget> routeTargets;
};

using FabricRuntimeMap = llvm::DenseMap<Operation *, FabricRuntimeInfo>;

struct PipeInfo {
  PipeType pipeType;
  PipeTransferContract transferContract;
};

struct PipeSramAddressTableInfo {
  int64_t byteOffset;
};

/// Sender-side recipe for computing a receiver DFB address. The graph assigns
/// `receiverSlotIndex`; a dynamic counter is needed only when repeated
/// executions revisit the same receiver batch at a different physical slot.
struct PipeComputedAddressInfo {
  int64_t receiverDFBIndex = 0;
  int64_t baseCompileTimeArgIndex = 0;
  /// Initial physical receiver DFB block assigned to this transfer.
  int64_t receiverSlotIndex = 0;
  /// Number of receiver DFB blocks reserved by one statically proven batch.
  int64_t receiverBatchSize = 1;
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
class PipeCapacityPlan;

/// Resolved lowering-address form of a pipe counter. A GlobalSemaphore counter
/// resolves to a runtime-arg index because its address is bound at runtime.
enum class PipeCounterAddressStorage {
  LocalSemaphore,
  GlobalSemaphoreRuntimeArg,
};

/// Allocation-time storage kind chosen for a pipe counter during planning,
/// before its address form is resolved.
enum class PipeCounterStorage {
  LocalSemaphore,
  GlobalSemaphore,
};

/// Pipe synchronization counters can live either in local semaphore space or
/// in GlobalSemaphore-backed SRAM. The storage kind disambiguates the index.
struct PipeCounterAddressInfo {
  PipeCounterAddressStorage storage;
  int64_t index;
};

/// Visitor for pipe-counter accounting. Default no-op methods let each
/// accounting pass consume only the counter namespace it owns.
class PipeCounterObserver {
public:
  virtual ~PipeCounterObserver() = default;

  virtual void observeLocalSemaphore(int64_t index) {}
  virtual void observeGlobalSemaphore(int64_t index) {}
};

/// Pipe counter allocation. This translates the stored index into the lowering
/// address form and reports it in its resource namespace for count and limit
/// checks.
class PipeCounterInfo {
public:
  /// Allocate a counter from TTKernel local semaphore ids.
  static PipeCounterInfo localSemaphore(int64_t semaphoreIndex);

  /// Allocate a counter from host-created GlobalSemaphore storage.
  static PipeCounterInfo globalSemaphore(int64_t globalSemaphoreIndex);

  /// Resolve this allocation to the address consumed by TTKernel lowering.
  PipeCounterAddressInfo
  getAddressInfo(Operation *op, const PipeResourcePlan &pipeResourcePlan) const;

  /// Report this allocation to a pass-specific observer.
  void observe(PipeCounterObserver &observer) const;

private:
  PipeCounterInfo(PipeCounterStorage storage, int64_t index)
      : storage(storage), index(index) {}

  PipeCounterStorage storage;
  int64_t index;
};

struct PipeCompletionWaitInfo {
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

/// Lowering information for a set of ttl.pipe_transfer.create ops sharing one
/// PipeKey. This keeps address storage separate from readiness counting so
/// physical allocation can choose local semaphores or GlobalSemaphore-backed
/// counters independently.
struct PipeResourceInfo {
  PipeKey pipe;
  PipeTransferContract transferContract;
  /// Absent when the transfer does not use the sender-ready protocol.
  std::optional<PipeCounterInfo> readyCounter;
  PipeAddressStorageInfo addressStorage;
};

/// Per-function map: pipeNetId -> kernel-local i32 counter for cumulative
/// pipe receive wait_min progress.
using PipeNetCounterMap =
    llvm::MapVector<func::FuncOp, llvm::MapVector<int64_t, Value>>;

/// Initial value for one sender-local computed-address slot counter.
struct PipeComputedAddressCounterInitInfo {
  int64_t counterIndex = 0;
  int64_t initialSlot = 0;
};

/// Per-function map: computed-address slot counter index -> kernel-local i32
/// counter used by senders whose receiver DFB ring position advances at
/// runtime.
using PipeComputedAddressCounterMap =
    llvm::MapVector<func::FuncOp, llvm::MapVector<int64_t, Value>>;

/// pipeNetId -> deduplicated list of pipes in that net. Built once
/// before lowering so is_src/is_dst/is_active patterns avoid walking the
/// module per match.
using PipeNetIndex = llvm::MapVector<int64_t, SmallVector<PipeInfo>>;

struct PipeSramScratchInfo {
  int64_t bytes = 0;
};

/// Static resource allocation used by pipe lowering. Receiver-completion
/// counters are per PipeNet. Sender-ready indices and address-table offsets are
/// per source node and only need to be unique across concurrently live transfer
/// intervals.
struct PipeResourcePlan {
  PipeSramScratchInfo sramScratch;
  llvm::MapVector<int64_t, PipeCompletionWaitInfo> completionWaits;
  llvm::MapVector<Operation *, PipeResourceInfo> resources;
  llvm::MapVector<Operation *, SmallVector<PipeResourceInfo>> selectedResources;
  /// Entry-block counter initializers are part of the resource plan so all
  /// computed-address sends sharing one allocation unit share one slot state.
  llvm::MapVector<func::FuncOp, SmallVector<PipeComputedAddressCounterInitInfo>>
      computedAddressCounterInitializations;
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

/// Diagnose layouts that exceed the hardware semaphore id limit before
/// emitting ttkernel.get_semaphore ops with invalid ids.
LogicalResult
verifyPipeResourcePlanFitsHardware(ModuleOp mod, const PipeResourcePlan &info,
                                   const PipeCapacityPlan *pipeCapacityPlan,
                                   const PipeResourceRequirements &reqs);

/// Walk `mod` once and group every pipe transfer by its net id.
/// Deduplicates by (src, dst start/end) so the same pipe appearing on
/// multiple ops contributes one entry.
void buildPipeNetIndex(ModuleOp mod, PipeNetIndex &index);

/// Build per-kernel routing-plane records from typed device transfers.
LogicalResult buildFabricRoutePlan(ModuleOp mod, FabricRoutePlan &plan);

/// Materialize one routing-plane manager per kernel that uses fabric routes.
void initializeFabricRuntime(const FabricRoutePlan &plan,
                             FabricRuntimeMap &runtime);

/// Build the pipe resource plan used by pipe lowering. Transfer intervals that
/// cannot be bounded by dominance are conservatively treated as conflicting
/// with every other transfer interval from the same source core.
LogicalResult
buildPipeResourcePlan(ModuleOp mod, const PipeGraph &pipeGraph,
                      PipeResourcePlan &info,
                      bool enableComputedAddresses = true,
                      const PipeCapacityPlan *pipeCapacityPlan = nullptr,
                      bool updateComputedAddressAttrs = true);

/// Emit sender-side capacity semaphore initial values at kernel entry, and
/// allocate one zero-initialized `memref<1xi32>` per capacity semaphore that
/// tracks the sender's cumulative acquired count (keyed by capacity semaphore
/// index). The sender waits for the capacity semaphore to reach that count, so
/// the receiver's remote increment stays the only writer of the shared word.
void initializePipeCapacitySemaphores(
    const PipeCapacityPlan &pipeCapacityPlan,
    PipeNetCounterMap &senderCapacityCounters);

/// Emit sender-local slot counters for computed receiver addresses whose
/// physical receiver DFB slot advances at runtime.
void initializePipeComputedAddressCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeComputedAddressCounterMap &computedAddressCounters);

/// At each function entry, emit one zero-initialized `memref<1xi32>` per
/// pipeNetId used by a pipe receive.
void allocatePipeNetReceiveCounters(ModuleOp mod, PipeNetCounterMap &counters);

/// Lower the sender-side pipe transfer and signal receiver completion.
LogicalResult lowerPipeTransferSend(
    PipeTransferSendOp op, Value srcCB, bool isConsumerCB,
    const PipeResourcePlan &pipeResourcePlan,
    const PipeCapacityPlan *pipeCapacityPlan,
    const PipeNetCounterMap *senderCapacityCounters,
    const PipeComputedAddressCounterMap *computedAddressCounters,
    const FabricRoutePlan *fabricRoutePlan,
    const FabricRuntimeMap *fabricRuntime, ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe rendezvous.
LogicalResult lowerPipeTransferPost(PipeTransferPostOp op, Value dst,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    const PipeCapacityPlan *pipeCapacityPlan,
                                    const FabricRoutePlan *fabricRoutePlan,
                                    const FabricRuntimeMap *fabricRuntime,
                                    ConversionPatternRewriter &rewriter);

/// Lower a dataflow buffer pop and emit any proven pipe capacity releases.
LogicalResult lowerCBPop(CBPopOp op, Value cb,
                         const PipeCapacityPlan *pipeCapacityPlan,
                         ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe receive completion wait.
LogicalResult lowerPipeTransferWait(PipeTransferWaitOp op,
                                    const PipeNetCounterMap *counters,
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
