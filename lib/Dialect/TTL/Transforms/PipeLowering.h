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

/// Per-function map: pipeNetId -> kernel-local i32 counter for the
/// multicast cumulative wait_min protocol (issue #505).
using PipeNetCounterMap =
    llvm::DenseMap<func::FuncOp, llvm::DenseMap<int64_t, Value>>;

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
void populatePipeLoweringPatterns(RewritePatternSet &patterns,
                                  const TypeConverter &typeConverter);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H
