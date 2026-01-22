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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "llvm/ADT/STLExtras.h"

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

  return llvm::any_of(body->getArguments(), [](BlockArgument arg) {
    std::optional<mlir::Type> elementType = getTileElementType(arg.getType());
    return elementType && elementType->isF32();
  });
}

static void
setBoolAttrIf(ComputeOp computeOp, llvm::StringRef attrName, bool condition,
              llvm::function_ref<bool(ComputeOp)> extraPredicate = {}) {
  if (!condition || computeOp->hasAttr(attrName)) {
    return;
  }
  if (extraPredicate && !extraPredicate(computeOp)) {
    return;
  }
  computeOp->setAttr(attrName, BoolAttr::get(computeOp.getContext(), true));
}

struct TTLSetComputeKernelConfigPass
    : public impl::TTLSetComputeKernelConfigBase<
          TTLSetComputeKernelConfigPass> {
  using Base =
      impl::TTLSetComputeKernelConfigBase<TTLSetComputeKernelConfigPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    for (ComputeOp computeOp : funcOp.getOps<ComputeOp>()) {

      // Set fp32_dest_acc_en if any tile arg is f32
      setBoolAttrIf(computeOp, kFp32DestAccEnAttrName, true, [&](ComputeOp op) {
        return fp32DestAccEn || hasF32TileArgs(op);
      });

      // Set dst_full_sync_en if not already set
      setBoolAttrIf(computeOp, kDstFullSyncEnAttrName, dstFullSyncEn);

      // Add other runtime configuration attributes as needed below
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
