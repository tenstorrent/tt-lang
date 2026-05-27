// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/// \file
/// Declares lowering support for PipeNet foreach ops.
///
/// PipeNet foreach lowering preserves one invariant from the high-level
/// Python PipeNet API: the callback body executes once for each selected
/// PipeNet record whose role contains the current launch node. Selected pipe
/// values are therefore scoped to the foreach body and must be lowered while
/// the selected record coordinates are still available.

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H

#include "PipeLowering.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir::tt::ttl {

/// Convert one static PipeNet foreach record to the PipeType used by existing
/// pipe resource allocation.
PipeType getPipeTypeFromRecord(MLIRContext *context, PipeRecordAttr record,
                               int64_t pipeNetId);

/// Add pipes referenced only by PipeNet foreach ops to `index`.
void addPipeNetForeachRecordsToIndex(ModuleOp mod, PipeNetIndex &index);

/// Add the PipeNet id required by a selected-pipe receive wait, if any.
///
/// Receive completion waits use a per-function runtime counter keyed by
/// PipeNet id. For selected-pipe waits, the id is defined by the enclosing
/// `ttl.pipenet_foreach_dst` operation rather than by the pipe value type.
void collectPipeNetForeachReceiveWaitCounterIds(
    PipeRecvWaitOp wait, llvm::SmallSet<int64_t, 4> &pipeNetIds);

/// Lower a DFB-to-selected-pipe copy inside `ttl.pipenet_foreach_src`.
LogicalResult lowerDataflowBufferToSelectedPipe(
    CopyOp op, Value sourceDataflowBuffer, SelectPipeSrcOp selectedPipe,
    bool isConsumerDataflowBuffer, ConversionPatternRewriter &rewriter);

/// Lower selected-pipe receive address publication.
LogicalResult lowerSelectedPipeRecvPost(PipeRecvPostOp op, Value pipe,
                                        Value dst,
                                        const PipeRuntimeLayout *layout,
                                        ConversionPatternRewriter &rewriter);

/// Lower selected-pipe receive completion waits.
///
/// `counters` must contain an entry for the enclosing function and PipeNet id.
/// This mirrors static-pipe receive wait lowering: repeated dynamic waits in
/// loops wait for monotonically increasing receiver semaphore values.
LogicalResult lowerSelectedPipeRecvWait(PipeRecvWaitOp op, Value pipe,
                                        const PipeNetCounterMap *counters,
                                        ConversionPatternRewriter &rewriter);

/// Add PipeNet foreach lowering patterns to `patterns`.
void populatePipeNetForeachLoweringPatterns(
    RewritePatternSet &patterns, const TypeConverter &typeConverter,
    const PipeRuntimeLayout &pipeRuntimeLayout);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H
