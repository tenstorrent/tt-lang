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
// types. Python uses python/ttl/kernel_runner.py:build_cb_descriptors; if
// those ever diverge, align them or share one implementation (see issue #511).
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBAllocation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
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

    uint64_t budgetBytes = getL1DFBBudgetBytes(moduleOp, l1BudgetOverride);
    FailureOr<DFBAllocationSummary> maybeSummary =
        getDFBAllocationSummary(moduleOp);
    if (failed(maybeSummary)) {
      moduleOp.emitOpError("failed to compute DFB allocation sizes");
      signalPassFailure();
      return;
    }
    const DFBAllocationSummary &summary = *maybeSummary;

    if (summary.allocations.empty()) {
      return;
    }

    SmallVector<int64_t, 32> sortedIndices;
    sortedIndices.reserve(summary.allocations.size());
    for (const auto &e : summary.allocations) {
      sortedIndices.push_back(e.first);
    }
    llvm::sort(sortedIndices);

    auto emitBreakdown = [&](InFlightDiagnostic &diag) {
      for (int64_t idx : sortedIndices) {
        const DFBIndexAllocation &allocation = summary.allocations.at(idx);
        BindCBOp bindOp = allocation.representative;
        auto cbTy =
            mlir::cast<CircularBufferType>(bindOp.getResult().getType());
        diag << "\n  CB[" << idx << "]: shape=" << formatShape(cbTy.getShape())
             << ", element_type=" << cbTy.getElementType()
             << ", block_count=" << cbTy.getBlockCount() << ", "
             << allocation.bytes << " bytes";
        if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
          diag << " (compiler-allocated)";
        }
      }
      std::string percentage =
          formatDFBUsagePercentage(summary.totalBytes, budgetBytes);
      diag << "\n  total: " << summary.totalBytes << " / " << budgetBytes
           << " bytes (" << percentage << " percent)";
      diag << "\n  hint: reduce DFB block shapes or block_count, or reduce "
              "compiler-inserted buffers (fusion splits)";
    };

    // Anchor diagnostics on the bind for the largest per-index allocation so
    // multi-CB cases (and lit expected-error @below) point at the dominant
    // slot.
    auto bindForLargestAllocation = [&]() -> BindCBOp {
      int64_t reportIdx = sortedIndices.front();
      uint64_t reportMax = summary.allocations.at(reportIdx).bytes;
      for (int64_t idx : sortedIndices) {
        const uint64_t allocationBytes = summary.allocations.at(idx).bytes;
        if (allocationBytes > reportMax) {
          reportMax = allocationBytes;
          reportIdx = idx;
        }
      }
      return summary.allocations.at(reportIdx).representative;
    };

    if (summary.totalBytes > budgetBytes) {
      BindCBOp reportAt = bindForLargestAllocation();
      auto diag = reportAt.emitOpError()
                  << "total circular buffer allocation (" << summary.totalBytes
                  << " bytes) exceeds L1 budget (" << budgetBytes << " bytes)";
      emitBreakdown(diag);
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
