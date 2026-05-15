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

namespace mlir::tt::ttl {

/// Each PipeNet allocates two semaphores: one signaled by receivers (sender
/// waits on it before multicasting) and one signaled by the sender (receivers
/// wait on it for data arrival). They are laid out consecutively per net id,
/// `sender` at `id * 2` and `receiver` at `id * 2 + 1`, so kernel-side code
/// and host-side allocators agree without an extra side table.
inline int64_t getSenderSemIdx(int64_t pipeNetId) { return pipeNetId * 2; }
inline int64_t getReceiverSemIdx(int64_t pipeNetId) {
  return pipeNetId * 2 + 1;
}

/// Per-function map: pipeNetId -> kernel-local i32 counter for the
/// multicast cumulative wait_min protocol (issue #505).
using PipeNetCounterMap =
    llvm::DenseMap<func::FuncOp, llvm::DenseMap<int64_t, Value>>;

/// pipeNetId -> deduplicated list of pipe types in that net. Built once
/// before lowering so is_src/is_dst/is_active patterns avoid walking the
/// module per match.
using PipeNetIndex = llvm::DenseMap<int64_t, SmallVector<PipeType>>;

/// Walk `mod` once and group every PipeType result by its net id.
/// Deduplicates by (src, dst start/end) so the same pipe appearing on
/// multiple ops contributes one entry.
void buildPipeNetIndex(ModuleOp mod, PipeNetIndex &index);

/// At each function entry, emit one zero-initialized `memref<1xi32>` per
/// pipeNetId used by a multicast Pipe->CB CopyOp.
void allocatePipeNetCountersForMulticast(ModuleOp mod,
                                         PipeNetCounterMap &counters);

/// Lower CB -> Pipe copy (sender side). Uses receiver's CB address from
/// PipeGraph for gather; signals destinations via semaphore.
LogicalResult lowerCBToPipe(CopyOp op, Value srcCB, Value pipe,
                            const ReceiverCBInfo *receiverInfo,
                            bool isConsumerCB,
                            ConversionPatternRewriter &rewriter);

/// Lower Pipe -> CB copy (receiver side). Unicast gather: cumulative
/// wait_min with static recvProgress from PipeGraph. Multicast:
/// cumulative wait_min via the runtime counter from
/// `allocatePipeNetCountersForMulticast`.
LogicalResult lowerPipeToCB(CopyOp op, Value pipe, Value dstCB,
                            const PipeGraph *pipeGraph,
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
