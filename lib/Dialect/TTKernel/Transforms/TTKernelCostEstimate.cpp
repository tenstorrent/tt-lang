// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernelCostEstimate Pass
//===----------------------------------------------------------------------===//
//
// Runs the CostEstimator analysis and reports the per-core work split. The
// pass is a placement wrapper only: it owns where in the pipeline the estimate
// happens and where the report goes, while the analysis itself stays reusable
// by any other consumer.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Analysis/CostEstimator.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ttkernel-cost-estimate"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELCOSTESTIMATE
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTKernelCostEstimatePass
    : public impl::TTKernelCostEstimateBase<TTKernelCostEstimatePass> {
  using impl::TTKernelCostEstimateBase<
      TTKernelCostEstimatePass>::TTKernelCostEstimateBase;

  void runOnOperation() override {
    CostEstimator estimator(getOperation());
    FailureOr<CostEstimator::Report> report = estimator.estimate();
    if (failed(report)) {
      // estimate() already emitted a diagnostic explaining the stage mismatch.
      return signalPassFailure();
    }

    std::string text = report->render();
    if (outputPath.empty()) {
      llvm::outs() << text;
      return;
    }

    std::error_code error;
    llvm::raw_fd_ostream out(outputPath, error, llvm::sys::fs::OF_Text);
    if (error) {
      getOperation().emitError()
          << "cannot write cost estimate to '" << outputPath
          << "': " << error.message();
      return signalPassFailure();
    }
    out << text;
  }
};

} // namespace
} // namespace mlir::tt::ttl
