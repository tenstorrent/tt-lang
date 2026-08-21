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
    // The detail view is part of the report, so asking for it with the estimate
    // disabled asks for something that cannot exist. Refused rather than
    // ignored: a caller who typed it wants the tables, and silence would look
    // like the kernel had nothing to say.
    if (detail && !enable) {
      getOperation().emitError()
          << "cost estimate detail was requested with the estimate disabled: "
             "pass 'enable' to produce a report for it to add to";
      signalPassFailure();
      return;
    }
    if (!enable) {
      return;
    }

    // Nothing past here signals pass failure. The estimate is opt-in and
    // mutates nothing, so a program the estimator cannot account for is a gap
    // in the estimator, not a reason to fail a compile that would otherwise
    // succeed. The analysis warns at each operation responsible; this pass
    // reports that no estimate is coming and lets the pipeline continue.
    std::string text;

    CostEstimator::Options options;
    // The IR does not carry math fidelity, so the rows keyed on it can only be
    // reached by a caller that knows the value. Left empty they stay unmatched,
    // which the report counts.
    options.mathFidelity = mathFidelity;

    CostEstimator estimator(getOperation(), options);
    FailureOr<CostEstimator::Report> report = estimator.estimate();
    if (failed(report)) {
      text = "cost estimate: unavailable, see the warnings above\n";
    } else {
      // Summary only by default. The per-operation views are opt-in because a
      // kernel whose loops unroll to tens of thousands of operations produces a
      // report far longer than anyone reads.
      text = report->render();
      if (detail) {
        text += report->renderDetail() + "\n" + report->renderTimeline();
      }
    }

    if (outputPath.empty()) {
      llvm::outs() << text;
      return;
    }

    std::error_code error;
    llvm::raw_fd_ostream out(outputPath, error, llvm::sys::fs::OF_Text);
    if (error) {
      // The report is a side output, so losing it does not invalidate the
      // compile either. Say where it went instead of dropping it silently.
      getOperation().emitWarning()
          << "cannot write cost estimate to '" << outputPath
          << "': " << error.message() << "; writing it to stdout instead";
      llvm::outs() << text;
      return;
    }
    out << text;
  }
};

} // namespace
} // namespace mlir::tt::ttl
