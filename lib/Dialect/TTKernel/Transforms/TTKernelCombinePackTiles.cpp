// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernelCombinePackTiles Pass
//===----------------------------------------------------------------------===//
//
// Combines consecutive pack_tile ops on the same DFB with contiguous DST
// and CB tile indices into a single pack_tile_block call.
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
/// same CB, DST index == expected next, CB index == expected next.
static bool extendRun(ttk::PackTileOp op, Value runCB, int64_t expectedDst,
                      int64_t expectedCb) {
  auto dst = getConstantIntValue(op.getDstIndex());
  auto cb = getConstantIntValue(op.getOutIndex());
  return dst && cb && op.getOutCb() == runCB && *dst == expectedDst &&
         *cb == expectedCb;
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
      SmallVector<ttk::PackTileOp> run;

      auto flush = [&]() {
        if (run.size() >= 2) {
          replaceRun(run);
        }
        run.clear();
      };

      for (Operation &op : *block) {
        // Constant definitions may appear between pack_tile ops (they
        // define the index operands) and do not break a run.
        if (isa<arith::ConstantOp, arith::ConstantIndexOp,
                arith::ConstantIntOp>(&op)) {
          continue;
        }

        auto packOp = dyn_cast<ttk::PackTileOp>(&op);
        if (!packOp || !getConstantIntValue(packOp.getDstIndex()) ||
            !getConstantIntValue(packOp.getOutIndex())) {
          flush();
          continue;
        }

        if (!run.empty() &&
            extendRun(packOp, run.front().getOutCb(),
                      *getConstantIntValue(run.back().getDstIndex()) + 1,
                      *getConstantIntValue(run.back().getOutIndex()) + 1)) {
          run.push_back(packOp);
        } else {
          flush();
          run.push_back(packOp);
        }
      }

      flush();
    });
  }
};

} // namespace
} // namespace mlir::tt::ttl
