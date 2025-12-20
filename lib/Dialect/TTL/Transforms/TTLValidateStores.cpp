// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Validate Stores Pass
//===----------------------------------------------------------------------===//
//
// Checks ttl.store usage in ttl.compute regions. Ensures that any provided
// ttl.store:
// - Stores a yielded output tile
// - Uses a view produced by ttl.cb_reserve on the output's attached CB
// - Appears before ttl.yield
//
// Missing ttl.store ops are tolerated here because TTLInsertTileRegsSync will
// insert them when views exist; this pass focuses on validating provided
// stores.
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-validate-stores"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVALIDATESTORES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLValidateStoresPass
    : public impl::TTLValidateStoresBase<TTLValidateStoresPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    WalkResult result = funcOp.walk([&](ComputeOp computeOp) {
      Block &body = computeOp.getRegion().front();
      YieldOp yieldOp = dyn_cast<YieldOp>(body.getTerminator());
      if (!yieldOp) {
        return WalkResult::advance();
      }

      SmallVector<Value> yielded(yieldOp.getOperands().begin(),
                                 yieldOp.getOperands().end());

      DenseMap<size_t, StoreOp> storeForOutput;

      for (Operation &op : body.without_terminator()) {
        auto store = dyn_cast<StoreOp>(&op);
        if (!store) {
          continue;
        }

        auto it = llvm::find(yielded, store.getTile());
        if (it == yielded.end()) {
          store.emitOpError()
              << "tile operand must be one of the ttl.compute yielded outputs";
          return WalkResult::interrupt();
        }
        size_t outputIdx = static_cast<size_t>(it - yielded.begin());

        if (!store->isBeforeInBlock(yieldOp.getOperation())) {
          store.emitOpError() << "must appear before ttl.yield";
          return WalkResult::interrupt();
        }

        Value attachedCb = getAttachedCB(computeOp.getOutputs()[outputIdx]);
        if (!attachedCb) {
          computeOp.emitOpError()
              << "output " << outputIdx
              << " must have an attached circular buffer before storing";
          return WalkResult::interrupt();
        }

        auto reserve = store.getView().getDefiningOp<CBReserveOp>();
        if (!reserve) {
          store.emitOpError()
              << "view must be produced by ttl.cb_reserve for the output CB";
          return WalkResult::interrupt();
        }
        if (reserve.getCb() != attachedCb) {
          store.emitOpError()
              << "view CB does not match the circular buffer attached to "
              << "output " << outputIdx;
          return WalkResult::interrupt();
        }

        if (!storeForOutput.try_emplace(outputIdx, store).second) {
          store.emitOpError()
              << "duplicate ttl.store for output " << outputIdx;
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl

