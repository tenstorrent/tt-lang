// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_CONTROLFLOWUTILS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_CONTROLFLOWUTILS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::tt::ttl {

/// Return true when every pair of operations is proven mutually exclusive by
/// upstream RegionBranchOpInterface control-flow semantics.
bool arePairwiseInsideMutuallyExclusiveRegions(ArrayRef<Operation *> ops);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_CONTROLFLOWUTILS_H
