// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_LOWERFUSIONCOMPUTE_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_LOWERFUSIONCOMPUTE_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttl {

class ComputeOp;

/// Verify a target-selected fixed fusion schedule before compute lowering
/// mutates the function.
LogicalResult verifyFusionCompute(ComputeOp op);

/// Replace a verified fixed fusion compute with one DST section and its exact
/// output store.
LogicalResult generateFusionCompute(PatternRewriter &rewriter, Location loc,
                                    ComputeOp op);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_LOWERFUSIONCOMPUTE_H
