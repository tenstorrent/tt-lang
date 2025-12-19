// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Tile Regs Sync Pass
//===----------------------------------------------------------------------===//
//
// This pass inserts DST register synchronization operations around ttl.compute
// regions to enforce the MATH/PACK thread synchronization protocol required by
// the hardware DST register bank.
//
// The pass performs the following transformations:
//
// 1. Inside ttl.compute body:
//    - Inserts tile_regs_acquire at the beginning (if not present)
//    - Inserts tile_regs_commit immediately before the ttl.yield terminator
//
// 2. Outside ttl.compute (in parent block):
//    - Inserts tile_regs_wait immediately after the ttl.compute operation
//    - Inserts tile_regs_release immediately after tile_regs_wait
//
// This establishes the correct DST lifecycle:
//   MATH thread:  acquire -> [compute] -> commit
//   PACK thread:  wait -> release
//
// The pass is idempotent - it checks if sync ops are already present before
// inserting them, allowing it to be run multiple times safely.
//
//===----------------------------------------------------------------------===//

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
      assert(!computeOp.getRegion().empty() &&
             "ComputeOp must have a non-empty region");
      Block *body = &computeOp.getRegion().front();

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
      Operation *next = computeOperation->getNextNode();
      OpBuilder afterBuilder(computeOperation->getBlock(),
                             ++Block::iterator(computeOperation));
      if (!isa_and_nonnull<TileRegsWaitOp>(next)) {
        afterBuilder.create<TileRegsWaitOp>(computeOp.getLoc());
        next = computeOperation->getNextNode();
      }
      if (!isa_and_nonnull<TileRegsReleaseOp>(next ? next->getNextNode()
                                                   : nullptr)) {
        afterBuilder.create<TileRegsReleaseOp>(computeOp.getLoc());
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
