// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Annotate L1 Acc Loops
//===----------------------------------------------------------------------===//
//
// Detects user-written scf.for loops containing accumulating stores
// (ttl.store with the {accumulate} attribute, emitted by +=) and annotates
// them with kL1AccLoopAttrName for L1 packer accumulation.
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

      // Check if this loop directly contains an accumulating store
      // (ttl.store with the {accumulate} attribute, emitted by +=).
      // Only count stores whose nearest enclosing scf.for is this forOp,
      // so that nested inner loops are not attributed to outer loops.
      bool hasAccumulatingStore = false;
      forOp.getBody()->walk([&](StoreOp store) {
        if (store.getAccumulate() &&
            store->getParentOfType<scf::ForOp>() == forOp) {
          hasAccumulatingStore = true;
        }
      });

      if (hasAccumulatingStore) {
        forOp->setAttr(kL1AccLoopAttrName, OpBuilder(forOp).getUnitAttr());
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
