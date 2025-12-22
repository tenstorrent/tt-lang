// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Annotate CB Associations Pass
//===----------------------------------------------------------------------===//
//
// Analysis pass that annotates ttl.compute block arguments with CB index
// associations. This enables subsequent conversion passes to find the correct
// CB without fragile state management across multi-phase lowering.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#define DEBUG_TYPE "ttl-annotate-cb-associations"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLANNOTATECBASSOCIATIONS
#include "ttlang/Dialect/TTL/Passes.h.inc"

struct TTLAnnotateCBAssociationsPass
    : impl::TTLAnnotateCBAssociationsBase<TTLAnnotateCBAssociationsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([&](ComputeOp compute) {
      // For each input, find the associated CB and annotate the corresponding
      // block argument with its cb_index.
      for (auto [idx, input] : llvm::enumerate(compute.getInputs())) {
        Value cb = getAttachedCB(input);
        if (!cb) {
          // Emit a warning if no CB is found. This may cause downstream
          // lowering failures if the input is used in copy_tile operations.
          // The compute verifier should catch this, but we warn here for
          // better diagnostics.
          compute.emitWarning("input ")
              << idx
              << " does not have an attached circular buffer; this may cause "
                 "lowering failures if used in copy_tile operations";
          continue;
        }

        // Extract cb_index from the CB.
        IntegerAttr cbIndexAttr;
        if (auto bindOp = cb.getDefiningOp<BindCBOp>()) {
          cbIndexAttr = bindOp.getCbIndexAttr();
        } else {
          // CB is not from bind_cb (shouldn't happen in well-formed IR).
          compute.emitWarning("input ")
              << idx
              << " has a circular buffer that is not from ttl.bind_cb; "
                 "skipping annotation";
          continue;
        }

        // Store the mapping on the compute op itself using an attribute.
        // Build attribute name: "ttl.cb_index.N" where N is the block arg
        // index.
        std::string attrName = (kCBIndexAttrPrefix + std::to_string(idx)).str();
        compute->setAttr(attrName, cbIndexAttr);
      }
    });
  }
};

} // namespace mlir::tt::ttl
