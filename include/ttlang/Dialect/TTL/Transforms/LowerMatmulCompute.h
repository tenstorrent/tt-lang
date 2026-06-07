// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_LOWERMATMULCOMPUTE_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_LOWERMATMULCOMPUTE_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttl {

class ComputeOp;

/// Generate lowering for a ComputeOp containing tile_matmul_block.
///
/// Creates a single DstSectionOp with the matmul_block call, all cloned
/// body ops expanded M*N times, and per-output-view stores. For M=N=1
/// (single output tile), each op is emitted once. For M*N > 1, ops are
/// cloned per tile with remapped DST indices.
LogicalResult generateMatmulCompute(PatternRewriter &rewriter, Location loc,
                                    ComputeOp op,
                                    ArrayRef<AffineMap> indexingMaps,
                                    ArrayRef<StringAttr> iterTypes);

/// Emit a diagnostic and return failure if a block matmul compute's expanded
/// DST usage exceeds capacity. Run as a pass precondition before the rewrite so
/// the error is reported once, not on each greedy rewrite retry. Computes
/// nothing and succeeds for compute ops that are not block matmul candidates.
LogicalResult verifyMatmulComputeCapacity(ComputeOp op);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_LOWERMATMULCOMPUTE_H
