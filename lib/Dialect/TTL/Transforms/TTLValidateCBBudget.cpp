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
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ttl-validate-cb-budget"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVALIDATECBBUDGET
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static std::string formatShape(llvm::ArrayRef<int64_t> shape) {
  std::string formattedShape;
  llvm::raw_string_ostream outputStream(formattedShape);
  outputStream << "[";
  llvm::interleaveComma(shape, outputStream);
  outputStream << "]";
  return outputStream.str();
}

/// Formats the integer DFB budget usage percentage without overflowing.
static std::string formatDFBUsagePercentage(uint64_t allocationBytes,
                                            uint64_t budgetBytes) {
  if (budgetBytes == 0) {
    return "0";
  }

  // Multiplying a 64-bit allocation by 100 requires at most 71 bits.
  llvm::APInt percentageNumerator(/*numBits=*/128, allocationBytes);
  percentageNumerator *= 100;
  llvm::APInt percentage = percentageNumerator.udiv(budgetBytes);
  llvm::SmallString<24> percentageString;
  percentage.toStringUnsigned(percentageString);
  return percentageString.str().str();
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
          footprint.add(moduleOp, physicalIndex, cbType, failureReason);
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

    FailureOr<uint64_t> resetScratchBytes =
        getSynchronizedDFBResetStateAllocationBytes(moduleOp);
    if (failed(resetScratchBytes)) {
      signalPassFailure();
      return;
    }
    FailureOr<uint64_t> reconfigurationStateBytes =
        getDFBReconfigurationStateAllocationBytes(moduleOp);
    if (failed(reconfigurationStateBytes)) {
      signalPassFailure();
      return;
    }

    if (footprint.empty() && *resetScratchBytes == 0 &&
        *reconfigurationStateBytes == 0) {
      return;
    }

    FailureOr<uint64_t> maybeTotalBytes = footprint.getTotalBytes();
    if (failed(maybeTotalBytes)) {
      moduleOp.emitOpError()
          << "total DFB allocation size is not representable as uint64_t";
      signalPassFailure();
      return;
    }
    std::optional<uint64_t> maybeDFBAndResetBytes =
        llvm::checkedAddUnsigned(*maybeTotalBytes, *resetScratchBytes);
    if (!maybeDFBAndResetBytes) {
      moduleOp.emitOpError()
          << "total DFB and fixed-state allocation size is not "
             "representable as uint64_t";
      signalPassFailure();
      return;
    }
    std::optional<uint64_t> maybeCombinedBytes = llvm::checkedAddUnsigned(
        *maybeDFBAndResetBytes, *reconfigurationStateBytes);
    if (!maybeCombinedBytes) {
      moduleOp.emitOpError()
          << "total DFB and fixed-state allocation size is not "
             "representable as uint64_t";
      signalPassFailure();
      return;
    }
    uint64_t totalBytes = *maybeCombinedBytes;
    SmallVector<int64_t> sortedIndices = footprint.getSortedPhysicalIndices();

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
      if (*resetScratchBytes > 0) {
        diag << "\n  synchronized-reset scratch: " << *resetScratchBytes
             << " bytes";
      }
      if (*reconfigurationStateBytes > 0) {
        diag << "\n  reconfiguration state: " << *reconfigurationStateBytes
             << " bytes";
      }
      std::string percentage =
          formatDFBUsagePercentage(totalBytes, budgetBytes);
      diag << "\n  total: " << totalBytes << " / " << budgetBytes << " bytes ("
           << percentage << " percent)";
      diag << "\n  hint: reduce DFB block shapes or block_count, reduce "
              "compiler-inserted buffers (fusion splits)";
      if (*resetScratchBytes > 0 || *reconfigurationStateBytes > 0) {
        diag << ", or reduce synchronized-reset or reconfiguration boundaries";
      }
    };

    // Anchor diagnostics on the bind for the largest per-index allocation so
    // multi-CB cases (and lit expected-error @below) point at the dominant
    // slot.
    auto bindForLargestAllocation = [&]() -> BindCBOp {
      int64_t reportIdx = sortedIndices.front();
      uint64_t reportMax = footprint.getBytes(reportIdx);
      for (int64_t idx : sortedIndices) {
        const uint64_t allocationBytes = footprint.getBytes(idx);
        if (allocationBytes > reportMax) {
          reportMax = allocationBytes;
          reportIdx = idx;
        }
      }
      return bindForIndex[reportIdx];
    };

    if (totalBytes > budgetBytes) {
      InFlightDiagnostic diag = sortedIndices.empty()
                                    ? moduleOp.emitOpError()
                                    : bindForLargestAllocation().emitOpError();
      diag << ((*resetScratchBytes > 0 || *reconfigurationStateBytes > 0)
                   ? "total DFB and fixed-state allocation ("
                   : "total DFB allocation (")
           << totalBytes << " bytes) exceeds L1 budget (" << budgetBytes
           << " bytes)";
      emitBreakdown(diag);
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
