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
/// Creates a single DstSectionOp with:
///   1. The matmul_block call (block-level, full tensor operands)
///   2. Per-tile post-matmul ops expanded M*N times
///   3. Per-tile stores expanded M*N times
///
/// For M=N=1 (single output tile), the expansion is trivial: each op is
/// emitted once. For M*N > 1, ops are cloned per tile with remapped DST
/// indices. All ops share a single DST register section.
LogicalResult generateMatmulCompute(PatternRewriter &rewriter, Location loc,
                                    ComputeOp op,
                                    ArrayRef<AffineMap> indexingMaps,
                                    ArrayRef<StringAttr> iterTypes);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_LOWERMATMULCOMPUTE_H
