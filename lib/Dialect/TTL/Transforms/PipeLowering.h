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
#include <variant>

namespace mlir::tt::ttl {

/// Receiver-completion semaphores are indexed by PipeNet id. Sender-ready
/// semaphores and address-table slots are allocated from source-local transfer
/// intervals because non-overlapping transfers can reuse the same state.
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

struct PipeLocalReadyCounterInfo {
  int64_t senderReadySemIdx = 0;
};

struct PipeGlobalReadyCounterInfo {
  int64_t globalSemaphoreIndex = 0;
};

using PipeReadyCounterInfo =
    std::variant<PipeLocalReadyCounterInfo, PipeGlobalReadyCounterInfo>;

struct PipeCompletionWaitInfo {
  int64_t pipeNetId;
  int64_t receiverSemIdx;
};

/// Address storage used by one active pipe transfer. Each receiver publishes
/// its DFB write address into the source core's SRAM table before incrementing
/// the sender-ready counter.
struct PipeAddressStorageInfo {
  PipeSramAddressTableInfo sramAddressTable;
};

/// Lowering information for one active pipe transfer. This keeps address
/// storage separate from readiness counting so physical allocation can choose
/// local semaphores or GlobalSemaphore-backed counters independently.
struct PipeResourceInfo {
  PipeKey pipe;
  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;
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

/// Diagnose layouts that exceed the hardware semaphore id limit before
/// emitting ttkernel.get_semaphore ops with invalid ids.
LogicalResult verifyPipeResourcePlanFitsHardware(ModuleOp mod,
                                                 const PipeResourcePlan &info);

/// Return the number of semaphore ids referenced by the selected pipe lowering.
int64_t getRequiredPipeSyncSemaphoreCount(const PipeResourcePlan &info);

/// Return the number of GlobalSemaphore descriptors referenced by pipe
/// lowering.
int64_t getRequiredPipeGlobalSemaphoreCount(const PipeResourcePlan &info);

/// Return the per-core SRAM scratch bytes required by pipe address storage.
int64_t getRequiredPipeSramScratchBytes(const PipeResourcePlan &info);

/// Walk `mod` once and group every pipe transfer by its net id.
/// Deduplicates by (src, dst start/end) so the same pipe appearing on
/// multiple ops contributes one entry.
void buildPipeNetIndex(ModuleOp mod, PipeNetIndex &index);

/// Build the pipe resource plan used by pipe lowering.
LogicalResult buildPipeResourcePlan(ModuleOp mod, PipeResourcePlan &info);

/// At each function entry, emit one zero-initialized `memref<1xi32>` per
/// pipeNetId used by a pipe receive.
void allocatePipeNetReceiveCounters(ModuleOp mod, PipeNetCounterMap &counters);

/// Lower the sender-side pipe transfer. Uses receiver-published destination
/// addresses and signals destinations via semaphore.
LogicalResult lowerPipeTransferSend(PipeTransferSendOp op, Value srcCB,
                                    bool isConsumerCB,
                                    const PipeResourcePlan *pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe destination address publication.
LogicalResult lowerPipeTransferPost(PipeTransferPostOp op, Value dst,
                                    const PipeResourcePlan *pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe receive completion wait.
LogicalResult lowerPipeTransferWait(PipeTransferWaitOp op,
                                    const PipeNetCounterMap *counters,
                                    const PipeResourcePlan *pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter);

/// Add pipe-specific lowering patterns (IfSrc, IfDst, CreatePipe) to the set.
/// `pipeNetIndex` is borrowed and must outlive `patterns`; the is_src /
/// is_dst / is_active lowerings use it for O(1) net-id lookup.
void populatePipeLoweringPatterns(RewritePatternSet &patterns,
                                  const TypeConverter &typeConverter,
                                  const PipeNetIndex &pipeNetIndex);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H
