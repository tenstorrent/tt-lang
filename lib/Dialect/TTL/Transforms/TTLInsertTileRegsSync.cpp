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
// The pass is designed to run once during lowering; it does not check for
// existing sync ops.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

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

    funcOp.walk([&](ComputeOp computeOp) {
      Operation *computeOperation = computeOp.getOperation();
      Block *parent = computeOperation->getBlock();
      assert(parent && "ComputeOp must have parent block");

      // Acquire: before the compute op in parent block.
      Operation *prev = computeOperation->getPrevNode();
      if (!isa_and_nonnull<TileRegsAcquireOp>(prev)) {
        OpBuilder beforeBuilder(parent, Block::iterator(computeOperation));
        beforeBuilder.create<TileRegsAcquireOp>(computeOp.getLoc());
      }

      // Commit: inside compute body, immediately before ttl.yield.
      Block &body = computeOp.getRegion().front();
      if (Operation *terminator = body.getTerminator()) {
        Operation *beforeTerminator = terminator->getPrevNode();
        if (!beforeTerminator || !isa<TileRegsCommitOp>(beforeTerminator)) {
          OpBuilder inBody(terminator);
          inBody.create<TileRegsCommitOp>(computeOp.getLoc());
        }
      }

      // Wait: place immediately before ttl.yield (after commit), to guard the
      // impending write back (currently tensor.insert; TODO: lower to store).
      if (Operation *terminator = body.getTerminator()) {
        Operation *beforeTerminator = terminator->getPrevNode();
        if (!beforeTerminator || !isa<TileRegsWaitOp>(beforeTerminator)) {
          OpBuilder inBody(terminator);
          inBody.create<TileRegsWaitOp>(computeOp.getLoc());
        }
      }

      // Release: after compute in parent block.
      Operation *next = computeOperation->getNextNode();
      OpBuilder afterBuilder(parent, ++Block::iterator(computeOperation));
      if (!isa_and_nonnull<TileRegsReleaseOp>(next)) {
        afterBuilder.create<TileRegsReleaseOp>(computeOp.getLoc());
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
