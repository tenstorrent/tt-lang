// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DSTSUBBLOCKUTILS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DSTSUBBLOCKUTILS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt::ttl {

/// Find subblock sizes [t0, t1, ...] such that each ti divides dimSizes[i],
/// product(ti) <= maxTiles, and the product is maximized.
/// Ties are broken by preferring larger inner (higher-index) dimensions.
SmallVector<int64_t> computeMultiDimSubblockSizes(ArrayRef<int64_t> dimSizes,
                                                  int64_t maxTiles);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DSTSUBBLOCKUTILS_H
