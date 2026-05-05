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
/// When useTridBarriers is true, also adds TRID-barrier deduplication patterns.
void populateTTKernelCleanupPatterns(RewritePatternSet &patterns,
                                     bool useTridBarriers = false);

} // namespace mlir::tt::ttkernel

#endif // TTLANG_DIALECT_TTKERNEL_TRANSFORMS_TTKERNELCLEANUPPATTERNS_H
