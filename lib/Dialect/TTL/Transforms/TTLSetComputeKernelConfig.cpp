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

struct TTLSetComputeKernelConfigPass
    : public impl::TTLSetComputeKernelConfigBase<
          TTLSetComputeKernelConfigPass> {
  using Base =
      impl::TTLSetComputeKernelConfigBase<TTLSetComputeKernelConfigPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // fp32_dest_acc_en and dst_full_sync_en are per-kernel compile-time
    // settings. Set them on the function so all compute ops inherit the
    // same value via getKernelBoolAttr().
    bool needsFp32 = fp32DestAccEn;
    bool fp32FromMatmul = false;
    if (!needsFp32) {
      funcOp->walk([&](ComputeOp computeOp) {
        if (needsFp32) {
          return WalkResult::interrupt();
        }
        if (hasF32TileArgs(computeOp)) {
          needsFp32 = true;
          return WalkResult::interrupt();
        }
        if (reduceFullFp32) {
          bool hasReduce = false;
          computeOp->walk([&](TileReduceOp) -> WalkResult {
            hasReduce = true;
            return WalkResult::interrupt();
          });
          if (hasReduce) {
            needsFp32 = true;
            return WalkResult::interrupt();
          }
        }
        if (matmulFullFp32) {
          bool hasMatmul = false;
          computeOp->walk([&](TileMatmulBlockOp) -> WalkResult {
            hasMatmul = true;
            return WalkResult::interrupt();
          });
          if (hasMatmul) {
            needsFp32 = true;
            fp32FromMatmul = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
    }

    // TODO(#454): Remove once tt-llk #1338 is fixed. unary_bcast produces
    // incorrect results with fp32_dest_acc_en and bf16 CBs.
    if (fp32FromMatmul) {
      bool hasBf16Bcast = false;
      funcOp->walk([&](TileBcastOp bcastOp) -> WalkResult {
        auto elemType = getTileElementType(bcastOp.getInput().getType());
        if (elemType && !elemType->isF32()) {
          hasBf16Bcast = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hasBf16Bcast) {
        needsFp32 = false;
      }
    }

    if (needsFp32 && !funcOp->hasAttr(kFp32DestAccEnAttrName)) {
      funcOp->setAttr(kFp32DestAccEnAttrName,
                      BoolAttr::get(funcOp.getContext(), true));
    }
    if (dstFullSyncEn && !funcOp->hasAttr(kDstFullSyncEnAttrName)) {
      funcOp->setAttr(kDstFullSyncEnAttrName,
                      BoolAttr::get(funcOp.getContext(), true));
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
