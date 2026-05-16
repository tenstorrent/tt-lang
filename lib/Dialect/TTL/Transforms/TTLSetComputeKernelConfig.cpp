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
#include "mlir/IR/BuiltinTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
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

static bool isF32CB0InputBlockArgument(Value value, ComputeOp computeOp) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg || arg.getOwner() != &computeOp.getRegion().front()) {
    return false;
  }
  unsigned argNumber = arg.getArgNumber();
  if (argNumber >= computeOp.getNumInputs()) {
    return false;
  }
  std::optional<mlir::Type> elementType = getTileElementType(arg.getType());
  if (!elementType || !elementType->isF32()) {
    return false;
  }
  Value cb = getAttachedCB(computeOp.getInputs()[argNumber]);
  return cb && getCBIndex(cb) == 0;
}

// TODO: Add TTLFPUOp and TTLSFPUOp traits to distinguish FPU and SFPU tile ops.
// Then stop relying on the list of ops in "if (isa<TileReduceOp,
// TileMatmulBlockOp>(op), ...) "
static bool isDstInputTileComputeOp(Operation *op) {
  if (!isTileComputeOp(op)) {
    return false;
  }
  if (isa<TileReduceOp, TileMatmulBlockOp>(op)) {
    return false;
  }
  if (isFPUEligibleBinaryOp(op)) {
    return false;
  }
  return op->hasTrait<TTLDSTInputsTrait>() ||
         isa<TileBcastOp, TileTransposeOp>(op);
}

/// True when a compute body contains an SFPU-strategy tile op that must unpack
/// an f32 input tile from CB0 into DST. FPU consumers (reduce, matmul, and
/// FPU-eligible add/sub/mul) read via SRCA/SRCB and must not enable this mode.
static bool needsUnpackToDestFp32(ComputeOp computeOp) {
  Block &body = computeOp.getRegion().front();
  return llvm::any_of(body.without_terminator(), [&](Operation &op) {
    if (!isDstInputTileComputeOp(&op)) {
      return false;
    }
    return llvm::any_of(op.getOperands(), [&](Value operand) {
      return isF32CB0InputBlockArgument(operand, computeOp);
    });
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
    bool fp32FromReduce = false;
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
          bool hasFullFp32Reduce = false;
          computeOp->walk([&](TileReduceOp reduceOp) -> WalkResult {
            if (shouldUseFullFp32Reduce(reduceOp, reduceFullFp32)) {
              hasFullFp32Reduce = true;
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
          if (hasFullFp32Reduce) {
            needsFp32 = true;
            fp32FromReduce = true;
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
    // incorrect results with fp32_dest_acc_en and bf16 CBs. The same failure
    // mode appears when full-fp32 reduce enables fp32_dest_acc_en and the
    // fused body still feeds a bf16 unary_bcast (e.g. reduce then broadcast).
    if (fp32FromMatmul || fp32FromReduce) {
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
    funcOp->setAttr(kEnableFPUBinaryOpsAttrName,
                    BoolAttr::get(funcOp.getContext(), enableFPUBinaryOps));

    bool needsUnpackFp32 = false;
    funcOp->walk([&](ComputeOp computeOp) {
      if (needsUnpackToDestFp32(computeOp)) {
        needsUnpackFp32 = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (needsUnpackFp32 && !funcOp->hasAttr(kUnpackToDestFp32AttrName)) {
      funcOp->setAttr(kUnpackToDestFp32AttrName,
                      BoolAttr::get(funcOp.getContext(), true));
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
