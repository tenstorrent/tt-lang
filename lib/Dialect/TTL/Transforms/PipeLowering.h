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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

namespace mlir::tt {
class ValueOriginAnalysis;
}

namespace mlir::tt::ttl {

class PipeTransferIndex;

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
  PipeCounterInfo readyCounter;
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

/// Static resource allocation used by pipe lowering. Each protocol operation
/// maps to its transfer-specific completion, readiness, and address resources.
struct PipeResourcePlan {
  PipeSramScratchInfo sramScratch;
  /// Maps each pipe send, receiver post, and receiver wait to the resources
  /// shared by that transfer definition.
  llvm::MapVector<Operation *, PipeResourceInfo> resources;
  /// Protocol operations proven unreachable at their pipe endpoint. Lowering
  /// removes these operations without allocating rendezvous resources.
  llvm::SmallPtrSet<Operation *, 8> staticallyInactiveOps;
  /// One entry-block counter preserves slot state across repeated sends from
  /// the same transfer definition.
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
LogicalResult buildPipeResourcePlan(ModuleOp mod,
                                    const PipeTransferIndex &transferIndex,
                                    const PipeGraph &pipeGraph,
                                    PipeResourcePlan &info,
                                    bool enableComputedAddresses = true);

/// Emit sender-local slot counters for computed receiver addresses whose
/// physical receiver DFB slot advances at runtime.
void initializePipeComputedAddressCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeComputedAddressCounterMap &computedAddressCounters);

/// At each receiver function entry, emit one zero-initialized sequence counter
/// for every completion counter used by that function.
void initializePipePostSequenceCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeCounterProgressMap &postSequenceCounters);

/// Lower the sender-side pipe transfer and signal receiver completion.
LogicalResult lowerPipeTransferSend(
    PipeTransferSendOp op, Value srcCB, bool isConsumerCB,
    ValueOriginAnalysis &analysis, const PipeResourcePlan &pipeResourcePlan,
    const PipeComputedAddressCounterMap &computedAddressCounters,
    ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe rendezvous.
LogicalResult lowerPipeTransferPost(PipeTransferPostOp op, Value dst,
                                    ValueOriginAnalysis &analysis,
                                    const PipeCounterProgressMap &counters,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe receive completion wait.
LogicalResult lowerPipeTransferWait(PipeTransferWaitOp op, Value tokenSequence,
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
