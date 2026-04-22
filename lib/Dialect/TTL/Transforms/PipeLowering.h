// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H

#include "PipeGraph.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

namespace mlir::tt::ttl {

/// Lower CB -> Pipe copy (sender side): multicast tiles from source CB to
/// destination cores. For gather patterns, uses receiver's CB address from
/// PipeGraph. After transfer, signals destinations via semaphore.
LogicalResult lowerCBToPipe(CopyOp op, Value srcCB, Value pipe,
                            const ReceiverCBInfo *receiverInfo,
                            bool isConsumerCB,
                            ConversionPatternRewriter &rewriter);

/// Lower Pipe -> CB copy (receiver side): wait for data from sender via
/// semaphore handshake. For unicast gather, uses cumulative semaphore waits.
/// For multicast, signals sender "ready" then waits for VALID.
LogicalResult lowerPipeToCB(CopyOp op, Value pipe, Value dstCB,
                            const PipeGraph *pipeGraph,
                            ConversionPatternRewriter &rewriter);

/// Add pipe-specific lowering patterns (IfSrc, IfDst, CreatePipe) to the set.
void populatePipeLoweringPatterns(RewritePatternSet &patterns,
                                  const TypeConverter &typeConverter);

/// For each kernel thread function in `mod`, emit a noc_semaphore_set(..., 0)
/// at the top for every pipe semaphore slot the function references. Guards
/// against cross-program state leaks (e.g. multicast recv_sem left at 1 on
/// loopback senders) corrupting the first run of a subsequent kernel.
void emitPipeSemaphoreZeroInitPreamble(ModuleOp mod);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPELOWERING_H
