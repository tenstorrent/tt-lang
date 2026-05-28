// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/// \file
/// Declares helpers used by TTL-to-TTKernel pipe conversion.

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H

#include "PipeGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::ttl {

/// Receiver-arrival semaphores are indexed by PipeNet id. Sender-ready
/// semaphores and posted-address words are per pipe because different pipes in
/// one PipeNet can be posted and sent independently.
inline int64_t getReceiverSemIdx(int64_t pipeNetId) { return pipeNetId; }

struct PipeChannelLayout {
  int64_t senderReadySemIdx;
  int64_t mailboxSemIdxBase;
};

/// Per-function map: pipeNetId -> kernel-local i32 counter for cumulative
/// pipe receive wait_min progress.
using PipeNetCounterMap =
    llvm::DenseMap<func::FuncOp, llvm::DenseMap<int64_t, Value>>;

/// pipeNetId -> deduplicated list of pipe types in that net. Built once
/// before lowering so is_src/is_dst/is_active patterns avoid walking the
/// module per match.
using PipeNetIndex = llvm::DenseMap<int64_t, SmallVector<PipeType>>;

/// Static lookup table used by pipe lowering. Receiver-arrival semaphore
/// indices are global. Receive posts use one local staging semaphore per NOC
/// data-movement thread because remote SRAM writes read from local memory.
/// Sender-ready and posted-address indices only need to be unique among pipes
/// that share a source node.
struct PipeRuntimeLayout {
  int64_t mailboxStagingSemIdxBase = 0;
  int64_t numMailboxStagingSems = 0;
  llvm::DenseMap<PipeKey, PipeChannelLayout> channels;
};

/// Look up the per-pipe channel layout for `pipeType` in `pipeRuntimeLayout`.
/// Emits an internal-compiler-error diagnostic if the layout pointer is null or
/// the pipe is missing.
FailureOr<PipeChannelLayout>
lookupPipeChannelLayout(Operation *op, PipeType pipeType,
                        const PipeRuntimeLayout *pipeRuntimeLayout);

/// Walk `mod` once and group every PipeType result by its net id.
/// Deduplicates by (src, dst start/end) so the same pipe appearing on
/// multiple ops contributes one entry.
void buildPipeNetIndex(ModuleOp mod, PipeNetIndex &index);

/// Build the runtime semaphore layout used by pipe lowering.
void buildPipeRuntimeLayout(ModuleOp mod, const PipeNetIndex &index,
                            PipeRuntimeLayout &layout);

/// Diagnose layouts that exceed the hardware semaphore id limit before
/// emitting ttkernel.get_semaphore ops with invalid ids.
LogicalResult
verifyPipeRuntimeLayoutFitsHardware(ModuleOp mod,
                                    const PipeRuntimeLayout &layout);

/// At each function entry, emit one zero-initialized `memref<1xi32>` per
/// PipeNet id used by a pipe receive wait in that function.
///
/// The counter is function-local because each lowered kernel thread owns its
/// local memory and may execute a different dynamic number of waits. The
/// counter tracks the next receiver completion semaphore value expected by
/// that function for the PipeNet.
void allocatePipeNetReceiveCounters(ModuleOp mod, PipeNetCounterMap &counters);

/// Lower CB -> Pipe copy (sender side). Uses the published destination DFB
/// addresses and signals destinations via semaphore.
LogicalResult lowerCBToPipe(CopyOp op, Value srcCB, Value pipe,
                            bool isConsumerCB,
                            const PipeRuntimeLayout *pipeRuntimeLayout,
                            ConversionPatternRewriter &rewriter);

/// Lower the receiver-side pipe receive address publication.
LogicalResult lowerPipeRecvPost(PipeRecvPostOp op, Value pipe, Value dst,
                                const PipeRuntimeLayout *pipeRuntimeLayout,
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
                                  const PipeNetIndex &pipeNetIndex,
                                  const PipeRuntimeLayout &pipeRuntimeLayout);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H
