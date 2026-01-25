// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Inter-Loop CB Sync Pass
//===----------------------------------------------------------------------===//
//
// This pass inserts circular buffer synchronization operations between
// consecutive scf.for loops (marked with ttl.tile_loop) when the output CB
// of one loop feeds into the input CB of the next loop.
//
// The pass runs AFTER ttl-lower-to-loops and BEFORE ttl-insert-tile-regs-sync.
// It uses the loop marker attributes to identify CB dependencies:
//   - ttl.tile_loop.output_cbs: Array of CB indices written by the loop
//   - ttl.tile_loop.input_cbs: Array of CB indices read by the loop
//
// When loop1.output_cb == loop2.input_cb, this pass inserts:
//   - cb_wait before loop2 to ensure data from loop1 is available
//
// This enables proper synchronization for multi-compute kernels where the
// output of one compute feeds into the input of another.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "ttl-insert-inter-loop-cb-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTINTERLOOPCBSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLInsertInterLoopCBSyncPass
    : public impl::TTLInsertInterLoopCBSyncBase<TTLInsertInterLoopCBSyncPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Build a cache of CB index -> BindCBOp for O(1) lookups.
    llvm::DenseMap<int64_t, BindCBOp> cbIndexCache;
    funcOp.walk([&](BindCBOp bindOp) {
      cbIndexCache[bindOp.getCbIndex().getSExtValue()] = bindOp;
    });

    // Collect all tile loops in program order, along with their outermost
    // compute loops. The tile_loop attribute is on the innermost loop, and
    // tile_loop.outer marks the outermost loop of each compute nest.
    SmallVector<std::pair<scf::ForOp, scf::ForOp>> tileLoopsWithOuter;
    funcOp.walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(kTileLoopAttrName)) {
        scf::ForOp outermost = findOutermostComputeLoop(forOp);
        tileLoopsWithOuter.push_back({forOp, outermost});
      }
    });

    if (tileLoopsWithOuter.size() < 2) {
      return; // Nothing to sync
    }

    // Check consecutive loop pairs for CB dependencies.
    // Compare outermost loops to find consecutive compute operations.
    for (size_t i = 0; i + 1 < tileLoopsWithOuter.size(); ++i) {
      auto [producerInner, producerOuter] = tileLoopsWithOuter[i];
      auto [consumerInner, consumerOuter] = tileLoopsWithOuter[i + 1];

      // Only sync loops whose outermost loops are siblings in the same block.
      if (producerOuter->getBlock() != consumerOuter->getBlock()) {
        continue;
      }

      // Get all output CBs from producer and all input CBs from consumer.
      SmallVector<int64_t> producerOutputCBs =
          getCBIndicesFromLoopAttr(producerInner, kTileLoopOutputCBsAttrName);
      SmallVector<int64_t> consumerInputCBs =
          getCBIndicesFromLoopAttr(consumerInner, kTileLoopInputCBsAttrName);

      if (producerOutputCBs.empty() || consumerInputCBs.empty()) {
        continue;
      }

      // Find CBs that are in both producer's output and consumer's input.
      // These need cb_wait inserted before the consumer.
      // Use sets to avoid duplicate cb_wait insertions when the same CB
      // appears multiple times in input_cbs (e.g., both inputs from same CB).
      llvm::SmallDenseSet<int64_t> outputSet(producerOutputCBs.begin(),
                                             producerOutputCBs.end());
      llvm::SmallDenseSet<int64_t> sharedCBSet;
      for (int64_t cb : consumerInputCBs) {
        if (outputSet.contains(cb)) {
          sharedCBSet.insert(cb);
        }
      }
      SmallVector<int64_t> sharedCBs(sharedCBSet.begin(), sharedCBSet.end());

      if (sharedCBs.empty()) {
        continue;
      }

      // Insert cb_wait for each shared CB before the consumer's outermost loop.
      OpBuilder builder(consumerOuter);
      Location loc = consumerOuter.getLoc();

      for (int64_t cbIndex : sharedCBs) {
        auto it = cbIndexCache.find(cbIndex);
        if (it == cbIndexCache.end()) {
          consumerOuter.emitOpError()
              << "inter-loop CB sync failed: loop attribute references "
              << "cb_index " << cbIndex
              << " but no bind_cb with that index exists";
          return signalPassFailure();
        }
        BindCBOp bindOp = it->second;

        // Insert cb_wait to ensure data is available.
        auto cbType = cast<CircularBufferType>(bindOp.getType());
        auto tensorType =
            RankedTensorType::get(cbType.getShape(), cbType.getElementType());

        builder.create<CBWaitOp>(loc, tensorType, bindOp.getResult());
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
