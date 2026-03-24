// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernelCombinePackTiles Pass
//===----------------------------------------------------------------------===//
//
// Combines consecutive pack_tile ops on the same dataflow buffer with
// contiguous DST and DFB tile indices into a single pack_tile_block call.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "ttkernel-combine-pack-tiles"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELCOMBINEPACKTILES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

namespace ttk = mlir::tt::ttkernel;

/// Check whether a pack_tile op extends the current contiguous run:
/// same DFB, DST index == expected next, DFB tile index == expected next.
static bool extendsRun(ttk::PackTileOp op, Value runDFB, int64_t expectedDst,
                       int64_t expectedDfbIdx) {
  auto dst = getConstantIntValue(op.getDstIndex());
  auto cb = getConstantIntValue(op.getOutIndex());
  return dst && cb && op.getOutCb() == runDFB && *dst == expectedDst &&
         *cb == expectedDfbIdx;
}

/// Replace a run of 2+ consecutive pack_tile ops with a single
/// pack_tile_block.
static void replaceRun(ArrayRef<ttk::PackTileOp> run) {
  ttk::PackTileOp first = run.front();
  OpBuilder builder(first);
  Location loc = first.getLoc();

  Value ntiles = arith::ConstantIndexOp::create(builder, loc, run.size());
  ttk::PackTileBlockOp::create(builder, loc, first.getDstIndex(),
                               first.getOutCb(), ntiles);

  for (ttk::PackTileOp op : run) {
    op->erase();
  }
}

struct TTKernelCombinePackTilesPass
    : public impl::TTKernelCombinePackTilesBase<TTKernelCombinePackTilesPass> {
  using TTKernelCombinePackTilesBase::TTKernelCombinePackTilesBase;

  void runOnOperation() override {
    getOperation().walk([](Block *block) {
      // Collect all combinable runs first, then replace them. Replacing
      // during iteration would invalidate the block's operation list.
      SmallVector<SmallVector<ttk::PackTileOp>> runs;
      SmallVector<ttk::PackTileOp> run;

      // Finalize the current run: save it for replacement if combinable
      // (2+ ops), then clear for the next group.
      auto finalizeRun = [&]() {
        if (run.size() >= 2) {
          runs.push_back(std::move(run));
        }
        run.clear();
      };

      for (Operation &op : *block) {
        if (isa<arith::ConstantOp, arith::ConstantIndexOp,
                arith::ConstantIntOp>(&op)) {
          continue;
        }

        auto packOp = dyn_cast<ttk::PackTileOp>(&op);
        if (!packOp || !getConstantIntValue(packOp.getDstIndex()) ||
            !getConstantIntValue(packOp.getOutIndex())) {
          finalizeRun();
          continue;
        }

        if (!run.empty() &&
            extendsRun(packOp, run.front().getOutCb(),
                       *getConstantIntValue(run.back().getDstIndex()) + 1,
                       *getConstantIntValue(run.back().getOutIndex()) + 1)) {
          run.push_back(packOp);
        } else {
          finalizeRun();
          // pack_tile_block always writes to the CB starting from index 0
          // (per the op definition: the CB write pointer is reset by
          // cb_reserve_back and advanced by ntiles per call). A run can
          // only be combined when the first CB tile index is 0.
          if (*getConstantIntValue(packOp.getOutIndex()) == 0) {
            run.push_back(packOp);
          }
        }
      }

      finalizeRun();

      for (auto &r : runs) {
        replaceRun(r);
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::ttl
