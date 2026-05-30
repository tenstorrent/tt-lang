// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//
// TTL Validate CB Budget
//
// Validates that the sum of static circular-buffer backing stores (per unique
// cb_index) does not exceed a per-core L1 budget. Per-slot sizes use
// ttcore::TileType::getSizeBytes() when the CB already carries a tile type, and
// ttcore::TileType::get(elemTy).getSizeBytes() for row-wise / scalar element
// types (same rule as tt-mlir DeviceAttr::getMemrefCBPageSizeBytes). Python
// uses python/ttl/kernel_runner.py:build_cb_descriptors — if those ever
// diverge, align them or share one implementation (see issue #511).
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ttl-validate-cb-budget"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVALIDATECBBUDGET
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Fallback when the module has no `system_desc` / device: same static CB
/// region per core as Wormhole and Blackhole (1464 KiB L1 minus 32 KiB
/// reserved in tt-metal dev_mem_map). Matches
/// `ttl.constants.DEFAULT_L1_CB_BUDGET_BYTES` and
/// `ChipDescAttr::getUsableL1Size()` for those chips when IR carries attrs.
static constexpr uint64_t kFallbackUsableL1Bytes =
    static_cast<uint64_t>(1432 * 1024);

static std::string formatShape(llvm::ArrayRef<int64_t> shape) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "[";
  llvm::interleaveComma(shape, os);
  os << "]";
  return os.str();
}

/// If the module has a system descriptor and default device, return usable L1
/// for chip 0; otherwise return std::nullopt (caller uses the WH/BH fallback).
static std::optional<uint64_t> tryBudgetFromModule(ModuleOp moduleOp) {
  auto systemDesc = moduleOp->getAttrOfType<mlir::tt::ttcore::SystemDescAttr>(
      mlir::tt::ttcore::SystemDescAttr::name);
  if (!systemDesc) {
    return std::nullopt;
  }

  auto deviceOp = mlir::tt::ttcore::lookupDeviceOp(
      moduleOp, mlir::tt::ttcore::getDefaultDeviceName());
  if (!deviceOp) {
    return std::nullopt;
  }

  auto chipIds = deviceOp.getDeviceAttr().getChipIds();
  if (chipIds.empty()) {
    return std::nullopt;
  }

  return *llvm::min_element(llvm::map_range(chipIds, [&](unsigned chipId) {
    return systemDesc.getChipDesc(chipId).getUsableL1Size();
  }));
}

/// Bytes per CB slot: explicit ttcore.tile uses its shape/dtype; row-wise
/// (scalar/builtin) element types use the default tile layout for that dtype,
/// matching tt-mlir CB page sizing.
static uint64_t bytesPerCbElement(mlir::Type elemTy) {
  if (auto tileTy = mlir::dyn_cast<mlir::tt::ttcore::TileType>(elemTy)) {
    return tileTy.getSizeBytes();
  }
  return mlir::tt::ttcore::TileType::get(elemTy).getSizeBytes();
}

static FailureOr<uint64_t> cbBytesForBind(BindCBOp bindOp) {
  auto cbTy = mlir::cast<CircularBufferType>(bindOp.getResult().getType());
  mlir::Type elemTy = cbTy.getElementType();
  const uint64_t slotBytes = bytesPerCbElement(elemTy);
  const int64_t totalEl = cbTy.getTotalElements();
  if (totalEl < 0) {
    bindOp.emitOpError() << "invalid negative total element count for CB";
    return failure();
  }
  return static_cast<uint64_t>(totalEl) * slotBytes;
}

struct TTLValidateCBBudgetPass
    : public impl::TTLValidateCBBudgetBase<TTLValidateCBBudgetPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    uint64_t budgetBytes = kFallbackUsableL1Bytes;
    if (l1BudgetOverride > 0) {
      budgetBytes = l1BudgetOverride;
    } else if (auto fromDevice = tryBudgetFromModule(moduleOp)) {
      budgetBytes = *fromDevice;
    }

    llvm::DenseMap<int64_t, uint64_t> maxBytesByIndex;
    llvm::DenseMap<int64_t, BindCBOp> bindForIndex;

    auto walkResult = moduleOp.walk([&](BindCBOp bindOp) -> WalkResult {
      FailureOr<uint64_t> bytes = cbBytesForBind(bindOp);
      if (failed(bytes)) {
        return WalkResult::interrupt();
      }
      int64_t idx = bindOp.getCbIndex().getSExtValue();
      auto it = maxBytesByIndex.find(idx);
      if (it == maxBytesByIndex.end() || *bytes > it->second) {
        maxBytesByIndex[idx] = *bytes;
        bindForIndex[idx] = bindOp;
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    if (maxBytesByIndex.empty()) {
      return;
    }

    uint64_t totalBytes = 0;
    for (auto &e : maxBytesByIndex) {
      totalBytes += e.second;
    }

    SmallVector<int64_t, 32> sortedIndices;
    sortedIndices.reserve(maxBytesByIndex.size());
    for (auto &e : maxBytesByIndex) {
      sortedIndices.push_back(e.first);
    }
    llvm::sort(sortedIndices);

    auto emitBreakdown = [&](InFlightDiagnostic &diag) {
      for (int64_t idx : sortedIndices) {
        BindCBOp bindOp = bindForIndex[idx];
        auto cbTy =
            mlir::cast<CircularBufferType>(bindOp.getResult().getType());
        diag << "\n  CB[" << idx << "]: shape=" << formatShape(cbTy.getShape())
             << ", element_type=" << cbTy.getElementType()
             << ", block_count=" << cbTy.getBlockCount() << ", "
             << maxBytesByIndex[idx] << " bytes";
        if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
          diag << " (compiler-allocated)";
        }
      }
      uint64_t pct = budgetBytes ? (100 * totalBytes) / budgetBytes : 0;
      diag << "\n  total: " << totalBytes << " / " << budgetBytes << " bytes ("
           << pct << " percent)";
      diag << "\n  hint: reduce DFB block shapes or block_count, or reduce "
              "compiler-inserted buffers (fusion splits)";
    };

    // Anchor diagnostics on the bind for the largest per-index allocation so
    // multi-CB cases (and lit expected-error @below) point at the dominant
    // slot.
    auto bindForLargestAllocation = [&]() -> BindCBOp {
      int64_t reportIdx = sortedIndices.front();
      uint64_t reportMax = maxBytesByIndex[reportIdx];
      for (int64_t idx : sortedIndices) {
        const uint64_t b = maxBytesByIndex[idx];
        if (b > reportMax) {
          reportMax = b;
          reportIdx = idx;
        }
      }
      return bindForIndex[reportIdx];
    };

    if (totalBytes > budgetBytes) {
      BindCBOp reportAt = bindForLargestAllocation();
      auto diag = reportAt.emitOpError()
                  << "total circular buffer allocation (" << totalBytes
                  << " bytes) exceeds L1 budget (" << budgetBytes << " bytes)";
      emitBreakdown(diag);
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
