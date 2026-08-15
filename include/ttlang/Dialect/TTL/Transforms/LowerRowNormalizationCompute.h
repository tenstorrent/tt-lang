// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_LOWERROWNORMALIZATIONCOMPUTE_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_LOWERROWNORMALIZATIONCOMPUTE_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttl {

class ComputeOp;

/// Verify the complete block schedule and its effective DST capacity before
/// any compute lowering mutates the function.
LogicalResult verifyRowNormalizationCompute(ComputeOp op);

/// Replace a verified row-normalization compute with one DST section, one
/// block compute operation, and explicit contiguous output stores.
LogicalResult generateRowNormalizationCompute(PatternRewriter &rewriter,
                                              Location loc, ComputeOp op);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_LOWERROWNORMALIZATIONCOMPUTE_H
