// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//
// TTL Validate CB Budget
//
// Validates that the sum of static dataflow-buffer backing stores (per unique
// cb_index) does not exceed a per-core L1 budget. Explicit tile elements retain
// their dimensions. Scalar elements map to a ttcore data type and use default
// tile dimensions; unmappable element types are errors. Python uses
// python/ttl/kernel_runner.py:build_cb_descriptors; if those implementations
// diverge, align them or share one implementation (see issue #511).
//
//===----------------------------------------------------------------------===//

#include "DFBAllocationLimits.h"

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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ttl-validate-cb-budget"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVALIDATECBBUDGET
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static std::string formatShape(llvm::ArrayRef<int64_t> shape) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "[";
  llvm::interleaveComma(shape, os);
  os << "]";
  return os.str();
}

struct TTLValidateCBBudgetPass
    : public impl::TTLValidateCBBudgetBase<TTLValidateCBBudgetPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    std::optional<uint64_t> overrideBytes =
        l1BudgetOverride == 0 ? std::nullopt
                              : std::optional<uint64_t>(l1BudgetOverride);
    uint64_t budgetBytes = getUsableDFBL1Bytes(moduleOp, overrideBytes);

    DFBAllocationFootprint footprint;
    llvm::DenseMap<int64_t, BindCBOp> bindForIndex;

    auto walkResult = moduleOp.walk([&](BindCBOp bindOp) -> WalkResult {
      if (bindOp.getTensorBackingAttr()) {
        return WalkResult::advance();
      }
      auto cbType = cast<CircularBufferType>(bindOp.getResult().getType());
      int64_t physicalIndex = bindOp.getCbIndex().getSExtValue();
      std::string failureReason;
      FailureOr<bool> increased =
          footprint.add(physicalIndex, cbType, failureReason);
      if (failed(increased)) {
        bindOp.emitOpError() << failureReason;
        return WalkResult::interrupt();
      }
      if (*increased) {
        bindForIndex[physicalIndex] = bindOp;
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    if (footprint.empty()) {
      return;
    }

    uint64_t totalBytes = footprint.getTotalBytes();
    SmallVector<int64_t, kMaxCircularBuffers> sortedIndices =
        footprint.getSortedPhysicalIndices();

    auto emitBreakdown = [&](InFlightDiagnostic &diag) {
      for (int64_t idx : sortedIndices) {
        BindCBOp bindOp = bindForIndex[idx];
        auto cbTy =
            mlir::cast<CircularBufferType>(bindOp.getResult().getType());
        diag << "\n  CB[" << idx << "]: shape=" << formatShape(cbTy.getShape())
             << ", element_type=" << cbTy.getElementType()
             << ", block_count=" << cbTy.getBlockCount() << ", "
             << footprint.getBytes(idx) << " bytes";
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
      uint64_t reportMax = footprint.getBytes(reportIdx);
      for (int64_t idx : sortedIndices) {
        uint64_t allocationBytes = footprint.getBytes(idx);
        if (allocationBytes > reportMax) {
          reportMax = allocationBytes;
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
