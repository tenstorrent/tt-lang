// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTKERNEL_TRANSFORMS_TTKERNELCLEANUPPATTERNS_H
#define TTLANG_DIALECT_TTKERNEL_TRANSFORMS_TTKERNELCLEANUPPATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttkernel {

/// Populate cleanup patterns for TTKernel ops. These patterns optimize
/// TTKernel code by removing redundant operations (e.g., deduplicating
/// consecutive barriers of the same type).
void populateTTKernelCleanupPatterns(RewritePatternSet &patterns);

} // namespace mlir::tt::ttkernel

#endif // TTLANG_DIALECT_TTKERNEL_TRANSFORMS_TTKERNELCLEANUPPATTERNS_H
