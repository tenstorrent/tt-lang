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

/// Resolve the CB index of `value` when it is an f32 input block argument of
/// `computeOp` that is consumed directly from a circular buffer.
static std::optional<int64_t>
getF32InputCBIndexForBlockArg(Value value, ComputeOp computeOp) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg || arg.getOwner() != &computeOp.getRegion().front()) {
    return std::nullopt;
  }
  unsigned argNumber = arg.getArgNumber();
  if (argNumber >= computeOp.getNumInputs()) {
    return std::nullopt;
  }
  std::optional<mlir::Type> elementType = getTileElementType(arg.getType());
  if (!elementType || !elementType->isF32()) {
    return std::nullopt;
  }
  Value cb = getAttachedCB(computeOp.getInputs()[argNumber]);
  if (!cb) {
    return std::nullopt;
  }
  return getCBIndex(cb);
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

/// Return true if `op` benefits from `UnpackToDestFp32` when its input is an
/// f32 tile fed directly from a CB. This is the SFPU subset of
/// `isDstInputTileComputeOp`: tile_bcast and tile_transpose are also
/// DST-input ops, but their LLK paths (unary_bcast, transpose_dest) do not
/// support `UnpackToDestFp32` mode and produce incorrect results when it is
/// enabled on their source CB (see tt-llk #1338). They are therefore
/// excluded here so the CB stays in the default unpack mode.
static inline bool wantsUnpackToDestFp32(Operation *op) {
  return isDstInputTileComputeOp(op) && !isa<TileBcastOp, TileTransposeOp>(op);
}

/// Return the CB index when `value` is an f32 input block argument of
/// `computeOp` consumed by an FPU-style tile op (reduce, matmul, or
/// FPU-eligible add/sub/mul). FPU consumers route their operand through
/// SRCA/SRCB, which is incompatible with `UnpackToDestFp32` mode on the CB.
static std::optional<int64_t> getF32FPUSrcCBIndex(Operation *op, Value operand,
                                                  ComputeOp computeOp) {
  if (!isa<TileReduceOp, TileMatmulBlockOp>(op) && !isFPUEligibleBinaryOp(op)) {
    return std::nullopt;
  }
  return getF32InputCBIndexForBlockArg(operand, computeOp);
}

/// Collect the CB indices that must be configured with `UnpackToDestFp32`
/// because at least one SFPU-strategy tile op in the compute body reads an
/// f32 input tile from that CB directly into DST.
///
/// FPU consumers (reduce, matmul, and FPU-eligible add/sub/mul) read via
/// SRCA/SRCB and must remain in `Default` unpack mode; the two modes are
/// mutually exclusive on a given CB. When a CB is consumed by both
/// strategies the SFPU consumer's full-precision request is dropped (the CB
/// is left in `Default` so the FPU consumer keeps working) and a warning is
/// emitted on the FPU op so the user can rework the kernel if precision
/// matters.
static llvm::SmallSetVector<int64_t, 4>
collectF32SFPUInputCBs(ComputeOp computeOp) {
  llvm::SmallSetVector<int64_t, 4> sfpuCBs;
  llvm::SmallDenseMap<int64_t, Operation *> fpuCBs;

  Block &body = computeOp.getRegion().front();
  for (Operation &op : body.without_terminator()) {
    for (Value operand : op.getOperands()) {
      if (auto fpuIdx = getF32FPUSrcCBIndex(&op, operand, computeOp)) {
        fpuCBs.insert({*fpuIdx, &op});
      }
    }
    if (!wantsUnpackToDestFp32(&op)) {
      continue;
    }
    for (Value operand : op.getOperands()) {
      auto cbIdx = getF32InputCBIndexForBlockArg(operand, computeOp);
      if (!cbIdx) {
        continue;
      }
      sfpuCBs.insert(*cbIdx);
    }
  }

  llvm::SmallVector<int64_t, 2> conflicts;
  for (int64_t cb : sfpuCBs) {
    if (fpuCBs.contains(cb)) {
      conflicts.push_back(cb);
    }
  }
  for (int64_t cb : conflicts) {
    fpuCBs[cb]->emitWarning()
        << "f32 input from CB " << cb
        << " is consumed by both FPU and SFPU strategies; leaving the CB in "
           "default unpack mode so the FPU consumer works, but the SFPU "
           "consumer will lose precision";
    sfpuCBs.remove(cb);
  }

  return sfpuCBs;
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

    llvm::SmallSetVector<int64_t, 4> unpackFp32CBs;
    funcOp->walk([&](ComputeOp computeOp) {
      for (int64_t cb : collectF32SFPUInputCBs(computeOp)) {
        unpackFp32CBs.insert(cb);
      }
    });

    if (!unpackFp32CBs.empty() && !funcOp->hasAttr(kUnpackToDestFp32AttrName)) {
      SmallVector<int64_t> sortedCBs(unpackFp32CBs.begin(),
                                     unpackFp32CBs.end());
      llvm::sort(sortedCBs);
      SmallVector<int32_t> sortedCBs32(sortedCBs.begin(), sortedCBs.end());
      funcOp->setAttr(kUnpackToDestFp32AttrName,
                      DenseI32ArrayAttr::get(funcOp.getContext(), sortedCBs32));
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
