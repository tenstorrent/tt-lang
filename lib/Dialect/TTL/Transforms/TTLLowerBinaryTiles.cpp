// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Lower Binary Tiles Pass
//===----------------------------------------------------------------------===//
//
// Lowers polymorphic ttl.tile_add / tile_sub / tile_mul to concrete FPU or
// SFPU tile ops with the appropriate input traits.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLLOWERBINARYTILES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Replace a polymorphic binary tile op with either the FPU or SFPU variant.
template <typename FpuOp, typename SfpuOp, typename PolyOp>
static void replacePolyBinary(PolyOp polyOp, bool useFpu) {
  OpBuilder b(polyOp);
  Location loc = polyOp.getLoc();
  if (useFpu) {
    auto newOp = FpuOp::create(b, loc, polyOp.getLhs(), polyOp.getRhs(),
                               polyOp.getDstIndex());
    polyOp.getResult().replaceAllUsesWith(newOp.getResult());
    polyOp.erase();
    return;
  }
  auto newOp = SfpuOp::create(b, loc, polyOp.getLhs(), polyOp.getRhs(),
                              polyOp.getDstIndex());
  polyOp.getResult().replaceAllUsesWith(newOp.getResult());
  polyOp.erase();
}

struct TTLLowerBinaryTilesPass
    : public impl::TTLLowerBinaryTilesBase<TTLLowerBinaryTilesPass> {
  using Base = impl::TTLLowerBinaryTilesBase<TTLLowerBinaryTilesPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect polymorphic tile binares first so nested-region order does not
    // invalidate Operation pointers before replacement.
    SmallVector<std::pair<Operation *, ComputeOp>> worklist;
    func.walk([&](Operation *op) {
      if (!isa<AddTileOp, SubTileOp, MulTileOp>(op))
        return WalkResult::advance();
      ComputeOp computeOp = op->getParentOfType<ComputeOp>();
      worklist.push_back({op, computeOp});
      return WalkResult::advance();
    });

    for (auto [op, computeOp] : worklist) {
      const bool useFpu =
          computeOp &&
          isFpuBinaryEligible(op, computeOp, enableFPUBinaryOps);
      if (auto addOp = dyn_cast<AddTileOp>(op)) {
        replacePolyBinary<AddFpuTileOp, AddSfpuTileOp>(addOp, useFpu);
        continue;
      }
      if (auto subOp = dyn_cast<SubTileOp>(op)) {
        replacePolyBinary<SubFpuTileOp, SubSfpuTileOp>(subOp, useFpu);
        continue;
      }
      if (auto mulOp = dyn_cast<MulTileOp>(op)) {
        replacePolyBinary<MulFpuTileOp, MulSfpuTileOp>(mulOp, useFpu);
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
