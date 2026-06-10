// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_LOWERMATMULCOMPUTE_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_LOWERMATMULCOMPUTE_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

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

/// Return the number of DST slots required for each logical output tile of a
/// block-matmul compute. This includes the output slot and scratch slots used
/// by non-matmul tile ops in the same compute body.
FailureOr<int64_t> getMatmulComputeDstSlotsPerOutputTile(ComputeOp op);

/// Emit a diagnostic and return failure if a block matmul compute's expanded
/// DST usage exceeds capacity. Run as a pass precondition before the rewrite so
/// the error is reported once, not on each greedy rewrite retry.
LogicalResult verifyMatmulComputeCapacity(ComputeOp op);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_LOWERMATMULCOMPUTE_H
