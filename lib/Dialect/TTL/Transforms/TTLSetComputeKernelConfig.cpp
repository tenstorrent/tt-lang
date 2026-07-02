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

  return op->hasTrait<TTLStrategyDependentBinaryOpTrait>() ||
         op->hasTrait<TTLDSTInputsTrait>() ||
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
/// `computeOp` consumed by a tile op that must keep its source CB in `Default`
/// unpack mode. FPU-style ops (reduce, matmul, FPU-eligible add/sub/mul) route
/// the operand through SRCA/SRCB; `tile_bcast`/`tile_transpose` lower to
/// `unary_bcast`/`transpose_dest`, which produce incorrect results under
/// `UnpackToDestFp32` on their source CB (tt-llk #1338). Both are incompatible
/// with the mode.
static std::optional<int64_t>
getF32DefaultUnpackCBIndex(Operation *op, Value operand, ComputeOp computeOp) {
  if (!isa<TileReduceOp, TileMatmulBlockOp, TileBcastOp, TileTransposeOp>(op) &&
      !isFPUEligibleBinaryOp(op)) {
    return std::nullopt;
  }
  return getF32InputCBIndexForBlockArg(operand, computeOp);
}

static constexpr unsigned kMaxUnpackFp32CBs = 4;

struct F32InputCBUsage {
  llvm::SmallSetVector<int64_t, kMaxUnpackFp32CBs> sfpuCBs;
  llvm::MapVector<int64_t, Operation *> fpuCBConsumers;
};

/// Collect f32 input CB usage in one compute body.
///
/// FPU consumers (reduce, matmul, and FPU-eligible add/sub/mul) read via
/// SRCA/SRCB and must remain in `Default` unpack mode. SFPU consumers that read
/// f32 directly into DST require `UnpackToDestFp32`. These modes are configured
/// per kernel on the function, so conflicts must be diagnosed after aggregating
/// usage across every ttl.compute in the func.func.
static F32InputCBUsage collectF32InputCBUsage(ComputeOp computeOp) {
  F32InputCBUsage usage;

  Block &body = computeOp.getRegion().front();
  for (Operation &op : body.without_terminator()) {
    for (Value operand : op.getOperands()) {
      if (std::optional<int64_t> fpuIdx =
              getF32DefaultUnpackCBIndex(&op, operand, computeOp)) {
        usage.fpuCBConsumers.insert({*fpuIdx, &op});
      }
    }
    if (!wantsUnpackToDestFp32(&op)) {
      continue;
    }
    for (Value operand : op.getOperands()) {
      if (std::optional<int64_t> cbIdx =
              getF32InputCBIndexForBlockArg(operand, computeOp)) {
        usage.sfpuCBs.insert(*cbIdx);
      }
    }
  }

  return usage;
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

    llvm::SmallSetVector<int64_t, kMaxUnpackFp32CBs> kernelSFPUCBs;
    llvm::SmallDenseMap<int64_t, Operation *> kernelFPUCBConsumers;
    funcOp->walk([&](ComputeOp computeOp) {
      const F32InputCBUsage usage = collectF32InputCBUsage(computeOp);
      kernelSFPUCBs.insert_range(usage.sfpuCBs);
      for (auto [cb, consumer] : usage.fpuCBConsumers) {
        // Keep the first FPU consumer for stable diagnostics when multiple
        // compute regions consume the same CB through the FPU path.
        kernelFPUCBConsumers.insert({cb, consumer});
      }
    });

    bool hasConflict = false;
    for (int64_t cb : kernelSFPUCBs) {
      Operation *fpuConsumer = kernelFPUCBConsumers.lookup(cb);
      if (!fpuConsumer) {
        continue;
      }
      fpuConsumer->emitOpError()
          << "f32 input from CB " << cb
          << " is consumed by both FPU and SFPU strategies in the same "
             "kernel; the FPU consumer requires default unpack mode while "
             "the SFPU consumer needs UnpackToDestFp32, and the two modes "
             "are mutually exclusive on a given CB. Split the source into "
             "separate CBs (one per strategy) so the SFPU consumer keeps "
             "full f32 precision";
      hasConflict = true;
    }
    if (hasConflict) {
      signalPassFailure();
      return;
    }

    if (!kernelSFPUCBs.empty() && !funcOp->hasAttr(kUnpackToDestFp32AttrName)) {
      SmallVector<int64_t> sortedCBs(kernelSFPUCBs.begin(),
                                     kernelSFPUCBs.end());
      llvm::sort(sortedCBs);
      SmallVector<int32_t> sortedCBs32(sortedCBs.begin(), sortedCBs.end());
      funcOp->setAttr(kUnpackToDestFp32AttrName,
                      DenseI32ArrayAttr::get(funcOp.getContext(), sortedCBs32));
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
