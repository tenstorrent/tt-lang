// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Inter-Loop CB Sync Pass
//===----------------------------------------------------------------------===//
//
// This pass inserts circular buffer synchronization operations between
// scf.for loops (marked with ttl.tile_loop) when the output CB of a prior
// loop feeds into the input CB of a later loop. Handles non-consecutive
// dependencies and producers in ancestor blocks.
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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Iterators.h"
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
    funcOp.walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
      if (forOp->hasAttr(kTileLoopAttrName)) {
        scf::ForOp outermost = findOutermostComputeLoop(forOp);
        tileLoopsWithOuter.push_back({forOp, outermost});
      }
    });

    if (tileLoopsWithOuter.size() < 2) {
      return; // Nothing to sync
    }

    // Build a set of all CBs written by each loop for efficient lookup.
    // cbWrittenByLoop[i] = set of CB indices written by loop i.
    SmallVector<llvm::SmallDenseSet<int64_t>> cbWrittenByLoop;
    for (auto &[innerLoop, outerLoop] : tileLoopsWithOuter) {
      llvm::SmallDenseSet<int64_t> outputCBs;
      for (int64_t cb :
           getCBIndicesFromLoopAttr(innerLoop, kTileLoopOutputCBsAttrName)) {
        outputCBs.insert(cb);
      }
      cbWrittenByLoop.push_back(std::move(outputCBs));
    }

    // Build dominance info for checking producer-consumer relationships.
    DominanceInfo domInfo(funcOp);

    // For each consumer loop, check ALL prior loops for CB dependencies.
    for (size_t i = 1; i < tileLoopsWithOuter.size(); ++i) {
      auto [consumerInner, consumerOuter] = tileLoopsWithOuter[i];

      SmallVector<int64_t> consumerInputCBs =
          getCBIndicesFromLoopAttr(consumerInner, kTileLoopInputCBsAttrName);
      if (consumerInputCBs.empty()) {
        continue;
      }

      // Find CBs that need sync (CBs produced by prior loops and used here).
      // Use SmallVector with duplicate check for deterministic ordering.
      SmallVector<int64_t> sharedCBs;
      for (int64_t inputCB : consumerInputCBs) {
        // Skip if already added (deduplication).
        if (llvm::is_contained(sharedCBs, inputCB)) {
          continue;
        }
        for (size_t j = 0; j < i; ++j) {
          auto [producerInner, producerOuter] = tileLoopsWithOuter[j];
          // Only consider producers that dominate the consumer.
          if (!domInfo.dominates(producerOuter.getOperation(),
                                 consumerOuter.getOperation())) {
            continue;
          }
          if (cbWrittenByLoop[j].contains(inputCB)) {
            sharedCBs.push_back(inputCB);
            break; // Found a producer for this CB, no need to check more
          }
        }
      }

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
