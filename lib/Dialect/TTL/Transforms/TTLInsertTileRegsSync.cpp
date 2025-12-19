// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <type_traits>

#define DEBUG_TYPE "ttl-insert-tile-regs-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTTILEREGSSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLInsertTileRegsSyncPass
    : public impl::TTLInsertTileRegsSyncBase<TTLInsertTileRegsSyncPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    static_assert(std::is_class_v<TTLDialect>);

    funcOp.walk([&](ComputeOp computeOp) {
      Block *body = &computeOp.getRegion().front();
      if (!body) {
        return;
      }

      // Insert tile_regs_acquire at entry if absent.
      if (body->empty() || !isa<TileRegsAcquireOp>(body->front())) {
        OpBuilder frontBuilder(body, body->begin());
        frontBuilder.create<TileRegsAcquireOp>(computeOp.getLoc());
      }

      // Insert tile_regs_commit immediately before ttl.yield.
      if (Operation *terminator = body->getTerminator()) {
        Operation *prev = terminator->getPrevNode();
        if (!prev || !isa<TileRegsCommitOp>(prev)) {
          OpBuilder commitBuilder(terminator);
          commitBuilder.create<TileRegsCommitOp>(computeOp.getLoc());
        }
      }

      // Insert wait/release after the compute op in the parent block.
      Operation *computeOperation = computeOp.getOperation();
      OpBuilder afterBuilder(computeOperation->getBlock(),
                             ++Block::iterator(computeOperation));
      afterBuilder.create<TileRegsWaitOp>(computeOp.getLoc());
      afterBuilder.create<TileRegsReleaseOp>(computeOp.getLoc());
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
