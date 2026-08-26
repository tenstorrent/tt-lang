// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//
// TTL Validate CB Budget
//
// Validates that the sum of static dataflow-buffer backing stores (per unique
// compiler-selected storage index) does not exceed a per-core L1 budget.
// Explicit tile elements retain their dimensions. Scalar elements map to a
// ttcore data type and use default tile dimensions; unmappable element types
// are errors. Python uses
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

    FailureOr<FinalizedDFBStorageFootprint> finalizedFootprint =
        getFinalizedDFBStorageFootprint(moduleOp);
    if (failed(finalizedFootprint)) {
      signalPassFailure();
      return;
    }

    DFBStorageFootprint &footprint = finalizedFootprint->globalFootprint;
    DenseMap<int64_t, BindCBOp> bindForStorageIndex;
    DenseMap<int64_t, SmallVector<int64_t>> physicalIndicesByStorageIndex;

    auto walkResult = moduleOp.walk([&](BindCBOp bindOp) -> WalkResult {
      if (bindOp.getTensorBackingAttr()) {
        return WalkResult::advance();
      }
      int64_t physicalIndex = bindOp.getCbIndex().getSExtValue();
      auto storageIndexIt = finalizedFootprint->storageIndexByPhysicalIndex.find(
          physicalIndex);
      int64_t storageIndex =
          storageIndexIt ==
                  finalizedFootprint->storageIndexByPhysicalIndex.end()
              ? physicalIndex
              : storageIndexIt->second;
      SmallVector<int64_t> &physicalIndices =
          physicalIndicesByStorageIndex[storageIndex];
      if (!llvm::is_contained(physicalIndices, physicalIndex)) {
        physicalIndices.push_back(physicalIndex);
      }
      if (!bindForStorageIndex.contains(storageIndex)) {
        bindForStorageIndex[storageIndex] = bindOp;
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

    SmallVector<int64_t> sortedIndices = footprint.getSortedStorageIndices();
    std::optional<LaunchNodeCoord> peakNode;
    FailureOr<uint64_t> allocationBytes =
        finalizedFootprint->getPeakL1AllocationBytes(moduleOp, &peakNode);
    if (failed(allocationBytes)) {
      moduleOp.emitOpError()
          << "total DFB allocation size is not representable as uint64_t";
      signalPassFailure();
      return;
    }
    std::optional<uint64_t> maybeDFBAndResetBytes =
        llvm::checkedAddUnsigned(*allocationBytes, *resetScratchBytes);
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

    auto emitBreakdown = [&](InFlightDiagnostic &diag) {
      for (int64_t idx : sortedIndices) {
        SmallVector<int64_t> &physicalIndices =
            physicalIndicesByStorageIndex[idx];
        llvm::sort(physicalIndices);
        diag << "\n  storage[" << idx << "] DFBs=[";
        for (auto indexedPhysicalIndex : llvm::enumerate(physicalIndices)) {
          if (indexedPhysicalIndex.index() != 0) {
            diag << ", ";
          }
          diag << indexedPhysicalIndex.value();
        }
        FailureOr<uint64_t> allocationBytes =
            getL1AllocationSizeBytes(moduleOp, footprint.getBytes(idx));
        assert(succeeded(allocationBytes) &&
               "validated storage allocation must remain representable");
        diag << "]: " << *allocationBytes << " bytes";
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
      if (peakNode) {
        diag << " on launch node (" << peakNode->x << "," << peakNode->y
             << ")";
      }
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
      FailureOr<uint64_t> initialBytes =
          getL1AllocationSizeBytes(moduleOp, footprint.getBytes(reportIdx));
      assert(succeeded(initialBytes));
      uint64_t reportMax = *initialBytes;
      for (int64_t idx : sortedIndices) {
        FailureOr<uint64_t> maybeAllocationBytes =
            getL1AllocationSizeBytes(moduleOp, footprint.getBytes(idx));
        assert(succeeded(maybeAllocationBytes));
        const uint64_t allocationBytes = *maybeAllocationBytes;
        if (allocationBytes > reportMax) {
          reportMax = allocationBytes;
          reportIdx = idx;
        }
      }
      return bindForStorageIndex[reportIdx];
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
