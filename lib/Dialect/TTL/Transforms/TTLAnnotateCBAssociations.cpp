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
          auto diag = compute.emitWarning()
                      << "input " << idx
                      << " does not have an attached circular buffer";
          diag.attachNote(input.getLoc()) << "input defined here";
          diag.attachNote(compute.getLoc())
              << "to fix: add 'ttl.attach_cb' or 'ttl.cb_wait' before this "
                 "compute operation, e.g.:\n"
              << "  %cb = ttl.bind_cb {cb_index = N, ...}\n"
              << "  %attached = ttl.attach_cb <input>, %cb\n"
              << "  or\n"
              << "  %view = ttl.cb_wait %cb, <num_pages>";
          continue;
        }

        // Extract cb_index from the CB.
        IntegerAttr cbIndexAttr;
        if (auto bindOp = cb.getDefiningOp<BindCBOp>()) {
          cbIndexAttr = bindOp.getCbIndexAttr();
        } else {
          // CB is not from bind_cb (shouldn't happen in well-formed IR).
          auto diag = compute.emitWarning()
                      << "input " << idx
                      << " has a circular buffer that is not from ttl.bind_cb";
          diag.attachNote(cb.getLoc()) << "circular buffer defined here";
          diag.attachNote(compute.getLoc())
              << "all circular buffers must be created with ttl.bind_cb";
          continue;
        }

        // Validate cb_index is in valid range [0, 31].
        int64_t cbIndex = cbIndexAttr.getInt();
        if (cbIndex < 0 || cbIndex >= kMaxCircularBuffers) {
          compute.emitError("input ")
              << idx << " has invalid cb_index " << cbIndex
              << " (must be in range [0, " << (kMaxCircularBuffers - 1) << "])";
          signalPassFailure();
          return;
        }

        // Store the mapping on the compute op itself using an attribute.
        setCBIndexAttr(compute, idx, cbIndex);
      }

      // Also annotate outputs (for ops like transpose that need output CB).
      // Output block arguments start after inputs.
      size_t numInputs = compute.getInputs().size();
      for (auto [idx, output] : llvm::enumerate(compute.getOutputs())) {
        Value cb = getAttachedCB(output);
        if (!cb) {
          // Outputs should always have an attached CB.
          continue;
        }

        IntegerAttr cbIndexAttr;
        if (auto bindOp = cb.getDefiningOp<BindCBOp>()) {
          cbIndexAttr = bindOp.getCbIndexAttr();
        } else {
          continue;
        }

        int64_t cbIndex = cbIndexAttr.getInt();
        if (cbIndex < 0 || cbIndex >= kMaxCircularBuffers) {
          compute.emitError("output ")
              << idx << " has invalid cb_index " << cbIndex
              << " (must be in range [0, " << (kMaxCircularBuffers - 1) << "])";
          signalPassFailure();
          return;
        }

        // Output block args start after input block args
        setCBIndexAttr(compute, numInputs + idx, cbIndex);
      }
    });
  }
};

} // namespace mlir::tt::ttl
