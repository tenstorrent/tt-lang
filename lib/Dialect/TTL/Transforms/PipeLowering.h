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

enum class PipeAddressStorageKind {
  PostedMailbox,
  LocalReceiverDFB,
};

struct LocalReceiverDFBAddressInfo {
  int64_t receiverCBIndex;
  CircularBufferType receiverCBType;
  int64_t staticTileOffset;
};

struct PipeReadyCounterInfo {
  int64_t senderReadySemIdx;
};

/// Address storage used by one logical pipe channel. Posted-mailbox channels
/// carry receiver-authored addresses in source-local storage. Local receiver
/// DFB channels are limited to source-in-destination multicast, where the
/// sender also executes the receive post and can read the receiver DFB address
/// directly.
struct PipeAddressStorageInfo {
  PipeAddressStorageKind kind;
  std::optional<int64_t> mailboxSemIdxBase;
  std::optional<LocalReceiverDFBAddressInfo> localReceiverDFB;

  bool usesMailbox() const {
    return kind == PipeAddressStorageKind::PostedMailbox;
  }
  bool usesLocalReceiverDFB() const {
    return kind == PipeAddressStorageKind::LocalReceiverDFB;
  }
};

/// Lowering information for one logical pipe channel. This keeps address
/// storage separate from readiness counting so later channel-table and
/// GlobalSemaphore allocation can replace either resource independently.
struct PipeChannelInfo {
  PipeReadyCounterInfo readyCounter;
  PipeAddressStorageInfo addressStorage;

  bool usesMailbox() const { return addressStorage.usesMailbox(); }
  bool usesAggregateRendezvous() const {
    return addressStorage.usesLocalReceiverDFB();
  }
};

/// Per-function map: pipeNetId -> kernel-local i32 counter for cumulative
/// pipe receive wait_min progress.
using PipeNetCounterMap =
    llvm::DenseMap<func::FuncOp, llvm::DenseMap<int64_t, Value>>;

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

/// Lower CB -> Pipe copy (sender side). Uses receiver-published destination
/// addresses and signals destinations via semaphore.
LogicalResult lowerCBToPipe(CopyOp op, Value srcCB, Value pipe,
                            bool isConsumerCB,
                            const PipeChannelLoweringInfo *pipeChannelInfo,
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
