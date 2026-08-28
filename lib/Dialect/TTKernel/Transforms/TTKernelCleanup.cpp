// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELCLEANUP
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

class TTKernelCleanupPass
    : public impl::TTKernelCleanupBase<TTKernelCleanupPass> {
public:
  using impl::TTKernelCleanupBase<TTKernelCleanupPass>::TTKernelCleanupBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ttkernel::populateTTKernelCleanupPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
