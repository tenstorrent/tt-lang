// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//
// Reapply TTKernel cleanup after control-flow and endpoint simplification.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELCLEANUP
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTKernelCleanupPass : impl::TTKernelCleanupBase<TTKernelCleanupPass> {
  void runOnOperation() override {
    // Callable-effect queries inspect other functions, so cleanup must not
    // mutate those functions concurrently in a nested function pipeline.
    RewritePatternSet patterns(&getContext());
    ttkernel::populateTTKernelCleanupPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
