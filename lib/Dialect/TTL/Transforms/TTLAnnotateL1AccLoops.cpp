// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Annotate L1 Acc Loops
//===----------------------------------------------------------------------===//
//
// Detects user-written scf.for loops that accumulate into the same CB slot
// (reserve before loop, store inside, push after) and annotates them with
// kL1AccLoopAttrName for L1 accumulation.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "ttl-annotate-l1-acc-loops"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLANNOTATEL1ACCLOOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLAnnotateL1AccLoopsPass
    : public impl::TTLAnnotateL1AccLoopsBase<TTLAnnotateL1AccLoopsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([&](scf::ForOp forOp) {
      // Skip loops already annotated (compiler-generated or prior run).
      if (forOp->hasAttr(kL1AccLoopAttrName) ||
          forOp->hasAttr(kReductionLoopAttrName) ||
          forOp->hasAttr(kTileLoopStrideAttrName) ||
          forOp->hasAttr(kSubblockLoopStrideAttrName)) {
        return;
      }

      // Check if the loop body contains a store (ttl.store) targeting a
      // CB that was reserved (ttl.cb_reserve) before the loop.
      bool hasReductionStore = false;
      forOp.getBody()->walk([&](StoreOp store) {
        auto reserve = store.getView().getDefiningOp<CBReserveOp>();
        if (reserve && !forOp->isAncestor(reserve)) {
          hasReductionStore = true;
        }
      });

      if (hasReductionStore) {
        forOp->setAttr(kL1AccLoopAttrName, OpBuilder(forOp).getUnitAttr());
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
