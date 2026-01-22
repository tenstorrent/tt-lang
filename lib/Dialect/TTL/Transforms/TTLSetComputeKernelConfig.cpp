// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Set Compute Kernel Config Pass
//===----------------------------------------------------------------------===//
//
// Sets compute configuration attributes on ttl.compute operations so
// downstream passes can consume stable, explicit settings.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLSETCOMPUTEKERNELCONFIG
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

// TODO(#264): This function returns true if ANY arg is f32, enabling
// fp32_dest_acc_en for the entire compute op. Consider emitting a diagnostic
// when mixed dtypes are detected, or allowing per-operation fp32 control.
static bool hasF32TileArgs(ComputeOp computeOp) {
  Block *body = &computeOp.getRegion().front();
  if (!body) {
    return false;
  }

  for (BlockArgument arg : body->getArguments()) {
    auto tileType = dyn_cast<ttcore::TileType>(arg.getType());
    if (!tileType) {
      continue;
    }
    if (tileType.getElementType().isF32()) {
      return true;
    }
  }

  return false;
}

struct TTLSetComputeKernelConfigPass
    : public impl::TTLSetComputeKernelConfigBase<
          TTLSetComputeKernelConfigPass> {
  using Base =
      impl::TTLSetComputeKernelConfigBase<TTLSetComputeKernelConfigPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    funcOp.walk([&](ComputeOp computeOp) {
      if (!computeOp->hasAttr("fp32_dest_acc_en")) {
        bool enableFp32 = fp32DestAccEn || hasF32TileArgs(computeOp);
        if (enableFp32) {
          computeOp->setAttr("fp32_dest_acc_en",
                             BoolAttr::get(computeOp.getContext(), true));
        }
      }

      if (!computeOp->hasAttr("dst_full_sync_en") && dstFullSyncEn) {
        computeOp->setAttr("dst_full_sync_en",
                           BoolAttr::get(computeOp.getContext(), true));
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::ttl
