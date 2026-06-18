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
#include "llvm/ADT/MapVector.h"

namespace mlir::tt::ttl {

/// Receiver-completion semaphores are indexed by PipeNet id.
inline int64_t getReceiverCompletionSemIdx(int64_t pipeNetId) {
  return pipeNetId;
}

struct PipeInfo {
  PipeType pipeType;
  PipeTransferContract transferContract;
};

struct PipeSramAddressTableInfo {
  int64_t byteOffset;
};

struct PipeResourcePlan;

/// Resolved lowering-address form of a ready counter. A GlobalSemaphore counter
/// resolves to a runtime-arg index because its address is bound at runtime.
enum class ReadyCounterAddressStorage {
  LocalSemaphore,
  GlobalSemaphoreRuntimeArg,
};

/// Allocation-time storage kind chosen for a ready counter during planning,
/// before its address form is resolved.
enum class PipeReadyCounterStorage {
  LocalSemaphore,
  GlobalSemaphore,
};

/// Sender-ready counters can live either in local semaphore space or in
/// GlobalSemaphore-backed SRAM. The storage kind disambiguates the index value.
struct ReadyCounterAddressInfo {
  ReadyCounterAddressStorage storage;
  int64_t index;
};

/// Visitor for ready-counter accounting. Default no-op methods let each
/// accounting pass consume only the counter namespace it owns.
class PipeReadyCounterObserver {
public:
  virtual ~PipeReadyCounterObserver() = default;

  virtual void observeLocalSemaphore(int64_t index) {}
  virtual void observeGlobalSemaphore(int64_t index) {}
};

/// Sender-ready counter allocation. This translates the stored index into the
/// lowering address form and reports it in its resource namespace for count and
/// limit checks.
class PipeReadyCounterInfo {
public:
  /// Allocate a sender-ready counter from TTKernel local semaphore ids.
  static PipeReadyCounterInfo localSemaphore(int64_t senderReadyCounterSemIdx);

  /// Allocate a sender-ready counter from host-created GlobalSemaphore storage.
  static PipeReadyCounterInfo globalSemaphore(int64_t globalSemaphoreIndex);

  /// Resolve this allocation to the address consumed by TTKernel lowering.
  ReadyCounterAddressInfo
  getAddressInfo(Operation *op, const PipeResourcePlan &pipeResourcePlan) const;

  /// Report this allocation to a pass-specific observer.
  void observe(PipeReadyCounterObserver &observer) const;

private:
  PipeReadyCounterInfo(PipeReadyCounterStorage storage, int64_t index)
      : storage(storage), index(index) {}

  PipeReadyCounterStorage storage;
  int64_t index;
};

struct PipeCompletionWaitInfo {
  int64_t pipeNetId;
  int64_t receiverCompletionSemIdx;
};

/// Address storage used by one transfer-allocation unit. Each receiver
/// publishes its DFB write address into the source core's SRAM table before
/// incrementing the sender-ready counter.
struct PipeAddressStorageInfo {
  PipeSramAddressTableInfo sramAddressTable;
};

/// Lowering information for a set of ttl.pipe_transfer.create ops sharing one
/// PipeKey. This keeps address storage separate from readiness counting so
/// physical allocation can choose local semaphores or GlobalSemaphore-backed
/// counters independently.
struct PipeResourceInfo {
  PipeKey pipe;
  PipeTransferContract transferContract;
  PipeReadyCounterInfo readyCounter;
  PipeAddressStorageInfo addressStorage;
};

/// Per-function map: pipeNetId -> kernel-local i32 counter for cumulative
/// pipe receive wait_min progress.
using PipeNetCounterMap =
    llvm::MapVector<func::FuncOp, llvm::MapVector<int64_t, Value>>;

/// pipeNetId -> deduplicated list of pipes in that net. Built once
/// before lowering so is_src/is_dst/is_active patterns avoid walking the
/// module per match.
using PipeNetIndex = llvm::MapVector<int64_t, SmallVector<PipeInfo>>;

struct PipeSramScratchInfo {
  int64_t bytes = 0;
};

/// Static resource allocation used by pipe lowering. Receiver-completion
/// semaphore indices are per PipeNet. Sender-ready indices and address-table
/// offsets are per source core and only need to be unique across concurrently
/// live transfer intervals.
struct PipeResourcePlan {
  PipeSramScratchInfo sramScratch;
  llvm::MapVector<int64_t, PipeCompletionWaitInfo> completionWaits;
  llvm::MapVector<Operation *, PipeResourceInfo> resources;
};

/// Resource totals consumed by TTKernel lowering and runtime setup.
struct PipeResourceRequirements {
  int64_t syncSemaphoreCount = 0;
  int64_t globalSemaphoreCount = 0;
  int64_t sramScratchBytes = 0;
};

/// Return all pipe resource totals derived from the selected allocation plan.
PipeResourceRequirements
getPipeResourceRequirements(const PipeResourcePlan &info);

/// Diagnose layouts that exceed the hardware semaphore id limit before
/// emitting ttkernel.get_semaphore ops with invalid ids.
LogicalResult
verifyPipeResourcePlanFitsHardware(ModuleOp mod, const PipeResourcePlan &info,
                                   const PipeResourceRequirements &reqs);

/// Walk `mod` once and group every pipe transfer by its net id.
/// Deduplicates by (src, dst start/end) so the same pipe appearing on
/// multiple ops contributes one entry.
void buildPipeNetIndex(ModuleOp mod, PipeNetIndex &index);

/// Build the pipe resource plan used by pipe lowering. Transfer intervals that
/// cannot be bounded by dominance are conservatively treated as conflicting
/// with every other transfer interval from the same source core.
LogicalResult buildPipeResourcePlan(ModuleOp mod, PipeResourcePlan &info);

/// At each function entry, emit one zero-initialized `memref<1xi32>` per
/// pipeNetId used by a pipe receive.
void allocatePipeNetReceiveCounters(ModuleOp mod, PipeNetCounterMap &counters);

/// Lower the sender-side pipe transfer. Uses receiver-published destination
/// addresses and signals receiver completion.
LogicalResult lowerPipeTransferSend(PipeTransferSendOp op, Value srcCB,
                                    bool isConsumerCB,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe destination address publication.
LogicalResult lowerPipeTransferPost(PipeTransferPostOp op, Value dst,
                                    const PipeResourcePlan &pipeResourcePlan,
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
