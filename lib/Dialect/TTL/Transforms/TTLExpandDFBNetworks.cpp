// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Expand DFB Networks
//===----------------------------------------------------------------------===//
//
// Rewrites each `ttl.dfb.network` and the kernel ops that use it into
// ordinary DFBs. The pass plans every network on the unchanged module with
// `DFBNetworkPlanner` and changes nothing unless planning succeeds.
//
// Network records declare DFB producers and consumers that no kernel op
// shows, so passes that infer DFB endpoints from kernel ops would see an
// incomplete program. This pass must therefore run before any pass that
// reasons about DFB producers or consumers.
//
//===----------------------------------------------------------------------===//

#include "DFBNetworkPlanning.h"

#include "ttlang/Dialect/TTL/Passes.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLEXPANDDFBNETWORKS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLExpandDFBNetworksPass
    : public impl::TTLExpandDFBNetworksBase<TTLExpandDFBNetworksPass> {
  void runOnOperation() override {
    DFBNetworkPlanner planner(getOperation());
    DFBNetworkPlanningResult result = planner.build();

    if (result.isInvalidIR()) {
      emitPlanningDiagnostic(result.getInvalidIR());
      return signalPassFailure();
    }

    if (result.isRejected()) {
      for (const DFBNetworkRejection &rejection : result.getRejection()) {
        rejection.operation->emitOpError() << rejection.reason;
      }
      return signalPassFailure();
    }

    // Nothing applies a plan yet, so an admitted network is still reported.
    ArrayRef<DFBNetworkExpansion> expansions = result.getPlan().getNetworks();
    for (const DFBNetworkExpansion &expansion : expansions) {
      DFBNetworkOp network = expansion.network;
      network.emitOpError() << "@" << network.getSymName()
                            << ": DFB networks are not supported yet";
    }
    if (!expansions.empty()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
