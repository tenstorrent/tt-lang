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
#include "llvm/ADT/DenseMap.h"

#include <optional>

namespace mlir::tt::ttl {

/// Receiver-arrival semaphores are indexed by PipeNet id. Sender-ready
/// semaphores are per pipe because different pipes in one PipeNet can be posted
/// and sent independently. Posted-mailbox channels also need one source-local
/// mailbox word per pipe.
inline int64_t getReceiverSemIdx(int64_t pipeNetId) { return pipeNetId; }

struct PipeInfo {
  PipeType pipeType;
  bool isMulticast;
};

enum class PipeChannelKind {
  PostedMailbox,
  AggregateRendezvous,
};

struct AggregateRendezvousInfo {
  int64_t receiverCBIndex;
  CircularBufferType receiverCBType;
  int64_t staticTileOffset;
  int64_t receiverReserveSlot;
  int64_t numReceiverReserveSlots;
  bool sourceInDestination;
};

/// Lowering information for one logical pipe channel. The sender-ready counter
/// is always present. Posted-mailbox channels also reserve one source-local
/// mailbox word; aggregate channels instead record the uniform receiver DFB
/// address information needed to avoid per-destination mailboxes.
struct PipeChannelInfo {
  int64_t senderReadySemIdx;
  PipeChannelKind kind;
  std::optional<int64_t> mailboxSemIdxBase;
  std::optional<AggregateRendezvousInfo> aggregateInfo;

  bool usesMailbox() const { return kind == PipeChannelKind::PostedMailbox; }
  bool usesAggregateRendezvous() const {
    return kind == PipeChannelKind::AggregateRendezvous;
  }
};

/// Per-function map: pipeNetId -> kernel-local i32 counter for cumulative
/// pipe receive wait_min progress.
using PipeNetCounterMap =
    llvm::DenseMap<func::FuncOp, llvm::DenseMap<int64_t, Value>>;

/// Per-function map: PipeKey -> kernel-local i32 counter for aggregate
/// non-loopback sends. Source-in-destination multicast uses the source core's
/// receiver DFB write pointer directly because the source also posts.
using AggregateEpochCounterMap =
    llvm::DenseMap<func::FuncOp, llvm::DenseMap<PipeKey, Value>>;

/// pipeNetId -> deduplicated list of pipes in that net. Built once
/// before lowering so is_src/is_dst/is_active patterns avoid walking the
/// module per match.
using PipeNetIndex = llvm::DenseMap<int64_t, SmallVector<PipeInfo>>;

/// Static information used by pipe lowering. Receiver-arrival semaphore indices
/// are global. Receive posts use one local staging semaphore per NOC
/// data-movement thread because remote SRAM writes read from local memory.
/// Sender-ready and mailbox indices only need to be unique among pipes that
/// share a source core. Aggregate rendezvous channels omit the mailbox word.
struct PipeChannelLoweringInfo {
  int64_t mailboxStagingSemIdxBase = 0;
  int64_t numMailboxStagingSems = 0;
  llvm::DenseMap<PipeKey, PipeChannelInfo> channels;
};

/// Diagnose layouts that exceed the hardware semaphore id limit before
/// emitting ttkernel.get_semaphore ops with invalid ids.
LogicalResult
verifyPipeChannelLoweringInfoFitsHardware(ModuleOp mod,
                                          const PipeChannelLoweringInfo &info);

/// Walk `mod` once and group every PipeType result by its net id.
/// Deduplicates by (src, dst start/end) so the same pipe appearing on
/// multiple ops contributes one entry.
void buildPipeNetIndex(ModuleOp mod, PipeNetIndex &index);

/// Build the channel information used by pipe lowering.
void buildPipeChannelLoweringInfo(ModuleOp mod, const PipeNetIndex &index,
                                  const PipeGraph &pipeGraph,
                                  PipeChannelLoweringInfo &info);

/// At each function entry, emit one zero-initialized `memref<1xi32>` per
/// pipeNetId used by a pipe receive.
void allocatePipeNetReceiveCounters(ModuleOp mod, PipeNetCounterMap &counters);

/// At each function entry, emit one zero-initialized `memref<1xi32>` per
/// non-loopback aggregate channel used by a sender.
void allocateAggregateEpochCounters(ModuleOp mod,
                                    const PipeChannelLoweringInfo &info,
                                    AggregateEpochCounterMap &counters);

/// Lower CB -> Pipe copy (sender side). Uses receiver-published destination
/// addresses and signals destinations via semaphore.
LogicalResult lowerCBToPipe(CopyOp op, Value srcCB, Value pipe,
                            bool isConsumerCB,
                            const PipeChannelLoweringInfo *pipeChannelInfo,
                            const AggregateEpochCounterMap *epochCounters,
                            ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe receive address publication.
LogicalResult lowerPipeRecvPost(PipeRecvPostOp op, Value pipe, Value dst,
                                const PipeChannelLoweringInfo *pipeChannelInfo,
                                ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe receive completion wait.
LogicalResult lowerPipeRecvWait(PipeRecvWaitOp op, Value pipe, Value dst,
                                const PipeNetCounterMap *counters,
                                ConversionPatternRewriter &rewriter);

/// Add pipe-specific lowering patterns (IfSrc, IfDst, CreatePipe) to the set.
/// `pipeNetIndex` is borrowed and must outlive `patterns`; the is_src /
/// is_dst / is_active lowerings use it for O(1) net-id lookup.
void populatePipeLoweringPatterns(RewritePatternSet &patterns,
                                  const TypeConverter &typeConverter,
                                  const PipeNetIndex &pipeNetIndex);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H
