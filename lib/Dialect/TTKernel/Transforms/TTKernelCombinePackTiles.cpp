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

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

struct PackRun {
  SmallVector<ttk::PackTileOp> operations;
  bool useContiguousPack = false;
};

static void replaceRun(const PackRun &run) {
  ArrayRef<ttk::PackTileOp> operations = run.operations;
  ttk::PackTileOp first = operations.front();
  OpBuilder builder(first);
  Location loc = first.getLoc();

  Value ntiles =
      arith::ConstantIndexOp::create(builder, loc, operations.size());
  if (run.useContiguousPack) {
    ttk::PackBlockContiguousOp::create(
        builder, loc, first.getDstIndex(), first.getOutCb(), ntiles);
  } else {
    ttk::PackTileBlockOp::create(builder, loc, first.getDstIndex(),
                                 first.getOutCb(), ntiles);
  }

  for (ttk::PackTileOp packOp : operations) {
    packOp->erase();
  }
}

static bool isInsideAccumulationLoop(Block *block) {
  for (Operation *parent = block->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (forOp->hasAttr(kReductionLoopAttrName) ||
          forOp->hasAttr(kL1AccLoopAttrName)) {
        return true;
      }
    }
  }
  return false;
}

struct TTKernelCombinePackTilesPass
    : public impl::TTKernelCombinePackTilesBase<TTKernelCombinePackTilesPass> {
  using TTKernelCombinePackTilesBase::TTKernelCombinePackTilesBase;

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    std::string failureReason;
    FailureOr<std::optional<ttcore::Arch>> targetArch =
        resolveTargetArch(function, failureReason);
    if (failed(targetArch)) {
      function.emitOpError(failureReason);
      signalPassFailure();
      return;
    }
    bool useContiguousPack =
        *targetArch && **targetArch == ttcore::Arch::Blackhole;

    function.walk([&](Block *block) {
      bool insideAccumulationLoop = isInsideAccumulationLoop(block);
      if (insideAccumulationLoop && !useContiguousPack) {
        return;
      }

      // Replacing during iteration would invalidate the block list, so
      // collect all combinable runs first.
      SmallVector<PackRun> runs;
      SmallVector<ttk::PackTileOp> run;

      auto finalizeRun = [&]() {
        bool canUseContiguousPack =
            useContiguousPack && !run.empty() && run.size() <= 8;
        bool canUseGenericPack = !insideAccumulationLoop && run.size() >= 2;
        if (canUseContiguousPack || canUseGenericPack) {
          runs.push_back({std::move(run), canUseContiguousPack});
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
          // pack_tile_block writes from CB tile 0; runs starting elsewhere
          // cannot be combined.
          if (*getConstantIntValue(packOp.getOutIndex()) == 0) {
            run.push_back(packOp);
          }
        }
      }

      finalizeRun();

      for (const PackRun &packRun : runs) {
        replaceRun(packRun);
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::ttl
