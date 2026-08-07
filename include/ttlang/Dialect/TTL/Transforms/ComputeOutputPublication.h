// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEOUTPUTPUBLICATION_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEOUTPUTPUBLICATION_H

#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

class CBPushOp;
class ComputeOp;
class StoreOp;
struct OutputPublicationPlan;

LogicalResult
resolveCurrentOutputPublication(Operation *source, PatternRewriter &rewriter,
                                const OutputPublicationPlan &analyzed,
                                OutputPublicationPlan &resolved);

Value createOutputInitTensor(OpBuilder &builder, Location loc,
                             RankedTensorType type, Value exemplar);

void setInsertionPointToOutputPublication(PatternRewriter &rewriter,
                                          const OutputPublicationPlan &outputs);

void createComputeTileStore(PatternRewriter &rewriter, Location loc,
                            Value tileResult, ComputeOp computeOp,
                            StoreOp store);

void relocateOutputPushesAfterCompute(
    PatternRewriter &rewriter, ComputeOp computeOp,
    const OutputPublicationPlan &outputs,
    SmallVectorImpl<CBPushOp> &replacedPushes);

void eraseReplacedOutputPublication(PatternRewriter &rewriter,
                                    const OutputPublicationPlan &outputs,
                                    ComputeOp computeOp,
                                    ArrayRef<CBPushOp> replacedPushes);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEOUTPUTPUBLICATION_H
