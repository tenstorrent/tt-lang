// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBNetworkPlanning.h"

#include "ttlang/Dialect/TTL/Passes.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLPRINTDFBNETWORKPLAN
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLPrintDFBNetworkPlanPass
    : public impl::TTLPrintDFBNetworkPlanBase<TTLPrintDFBNetworkPlanPass> {
  void runOnOperation() override {
    DFBNetworkPlanner planner(getOperation());
    DFBNetworkPlanningResult result = planner.build();

    if (result.isInvalidIR()) {
      emitPlanningDiagnostic(result.getInvalidIR());
      return signalPassFailure();
    }

    raw_ostream &output = llvm::errs();
    if (result.isRejected()) {
      output << "DFB network plan: rejected\n";
      for (const DFBNetworkRejection &rejection : result.getRejection()) {
        output << "  " << rejection.reason << "\n";
      }
      return;
    }
    output << "DFB network plan: planned\n";
  }
};

} // namespace

} // namespace mlir::tt::ttl
