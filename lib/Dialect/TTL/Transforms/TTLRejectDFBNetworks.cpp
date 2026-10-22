// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Reject DFB Networks
//===----------------------------------------------------------------------===//
//
// Emits an error for each `ttl.dfb.network` in the module, naming the
// network. Nothing lowers networks yet.
//
// Network records declare DFB producers and consumers that no kernel op
// shows, so passes that infer DFB endpoints from kernel ops would see an
// incomplete program. This pass must therefore run before any pass that
// reasons about DFB producers or consumers.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLREJECTDFBNETWORKS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLRejectDFBNetworksPass
    : public impl::TTLRejectDFBNetworksBase<TTLRejectDFBNetworksPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool foundNetwork = false;
    module.walk([&](DFBNetworkOp network) {
      network.emitOpError() << "@" << network.getSymName()
                            << ": DFB networks are not supported yet";
      foundNetwork = true;
    });
    if (foundNetwork) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
