// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Annotate CB Associations Pass
//===----------------------------------------------------------------------===//
//
// Analysis pass that annotates CB index associations on TTL ops. This enables
// subsequent conversion passes to find the correct CB without SSA
// tracing across multi-phase lowering.
//
// Annotations:
// - ttl.compute: each input gets a ttl.cb_index.<N> attribute on the compute
//   op, keyed by the input's positional index
// - ttl.tile_bcast: gets a ttl.bcast_output_cb_index attribute so that
//   downstream passes (after loop lowering) can look up the output CB
//   without SSA tracing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
        // ComputeOp verifier rejects inputs without attached CBs, so this
        // should never be null after verification.
        assert(cb && "ComputeOp input must have attached CB (verifier bug?)");

        // Extract cb_index from the CB. If the CB is not from bind_cb
        // (e.g., a function argument), skip annotation — the cb_index
        // is not locally available.
        auto bindOp = cb.getDefiningOp<BindCBOp>();
        if (!bindOp) {
          continue;
        }

        setCBIndexAttr(compute, idx, bindOp.getCbIndexAttr().getInt());
      }
    });

    // Annotate tile_bcast ops with their output CB index so the
    // conversion pass can look it up without SSA tracing.
    func.walk([&](TileBcastOp bcast) {
      Value output = bcast.getOutput();
      // getAttachedCB traces through tensor.extract automatically.
      Value cb = getAttachedCB(output);
      if (!cb) {
        bcast.emitError("output does not have an attached circular buffer");
        signalPassFailure();
        return;
      }
      auto bindOp = cb.getDefiningOp<BindCBOp>();
      if (!bindOp) {
        auto diag = bcast.emitError()
                    << "output circular buffer is not from ttl.bind_cb; "
                       "cb_index required for broadcast lowering";
        diag.attachNote(cb.getLoc()) << "circular buffer defined here";
        signalPassFailure();
        return;
      }

      bcast->setAttr(kBcastOutputCBIndexAttrName, bindOp.getCbIndexAttr());
    });

    // Annotate tile_reduce and tile_transpose with output CB index.
    auto annotateOutputCB = [&](Operation *tileOp, Value output,
                                StringRef attrName) {
      Value cb = getAttachedCB(output);
      if (!cb) {
        tileOp->emitError("output does not have an attached dataflow buffer");
        signalPassFailure();
        return;
      }
      auto bindOp = cb.getDefiningOp<BindCBOp>();
      if (!bindOp) {
        tileOp->emitError("output dataflow buffer is not from ttl.bind_cb");
        signalPassFailure();
        return;
      }
      tileOp->setAttr(attrName, bindOp.getCbIndexAttr());
    };

    func.walk([&](TileReduceOp reduce) {
      annotateOutputCB(reduce, reduce.getOutput(),
                       kReduceOutputCBIndexAttrName);
    });

    func.walk([&](TileTransposeOp transpose) {
      annotateOutputCB(transpose, transpose.getOutput(),
                       kTransposeOutputCBIndexAttrName);
    });
  }
};

} // namespace mlir::tt::ttl
