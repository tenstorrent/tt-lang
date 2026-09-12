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

    DenseMap<int64_t, BindCBOp> bindForPhysicalIndex;
    moduleOp.walk([&](BindCBOp bindOp) {
      if (bindOp.getTensorBackingAttr()) {
        return;
      }
      int64_t physicalIndex = bindOp.getCbIndex().getSExtValue();
      if (!bindForPhysicalIndex.contains(physicalIndex)) {
        bindForPhysicalIndex[physicalIndex] = bindOp;
      }
    });

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

    if (finalizedFootprint->globalFootprint.empty() &&
        *resetScratchBytes == 0 && *reconfigurationStateBytes == 0) {
      return;
    }

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
    const DFBStorageFootprint *diagnosticFootprint =
        &finalizedFootprint->globalFootprint;
    const FinalizedDFBStorageFootprint::MembersByStorageIndex
        *diagnosticMembers = &finalizedFootprint->globalMembers;
    if (finalizedFootprint->usesPerNodeAccounting) {
      diagnosticFootprint = nullptr;
      diagnosticMembers = nullptr;
      if (peakNode) {
        auto nodeIt = llvm::find(finalizedFootprint->launchNodes, *peakNode);
        assert(nodeIt != finalizedFootprint->launchNodes.end());
        size_t nodeIndex = static_cast<size_t>(
            nodeIt - finalizedFootprint->launchNodes.begin());
        diagnosticFootprint = &finalizedFootprint->footprintsByNode[nodeIndex];
        diagnosticMembers = &finalizedFootprint->membersByNode[nodeIndex];
      }
    }
    SmallVector<int64_t> sortedIndices =
        diagnosticFootprint ? diagnosticFootprint->getSortedStorageIndices()
                            : SmallVector<int64_t>{};

    auto emitBreakdown = [&](InFlightDiagnostic &diag) {
      for (int64_t storageIndex : sortedIndices) {
        SmallVector<int64_t> physicalIndices =
            diagnosticMembers->lookup(storageIndex);
        llvm::sort(physicalIndices);
        diag << "\n  storage[" << storageIndex << "] DFBs=[";
        for (auto indexedPhysicalIndex : llvm::enumerate(physicalIndices)) {
          if (indexedPhysicalIndex.index() != 0) {
            diag << ", ";
          }
          diag << indexedPhysicalIndex.value();
        }
        FailureOr<uint64_t> allocationBytes = getL1AllocationSizeBytes(
            moduleOp, diagnosticFootprint->getBytes(storageIndex));
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
        diag << " on launch node (" << peakNode->x << "," << peakNode->y << ")";
      }
      diag << "\n  hint: reduce DFB block shapes or block_count, reduce "
              "compiler-inserted buffers (fusion splits)";
      if (*resetScratchBytes > 0 || *reconfigurationStateBytes > 0) {
        diag << ", or reduce synchronized-reset or reconfiguration boundaries";
      }
    };

    // Anchor diagnostics on one resident DFB from the largest storage
    // allocation so expected diagnostics identify a contributing operation.
    auto bindForLargestAllocation = [&]() -> BindCBOp {
      int64_t reportStorageIndex = sortedIndices.front();
      FailureOr<uint64_t> initialBytes = getL1AllocationSizeBytes(
          moduleOp, diagnosticFootprint->getBytes(reportStorageIndex));
      assert(succeeded(initialBytes));
      uint64_t reportMax = *initialBytes;
      for (int64_t storageIndex : sortedIndices) {
        FailureOr<uint64_t> maybeAllocationBytes = getL1AllocationSizeBytes(
            moduleOp, diagnosticFootprint->getBytes(storageIndex));
        assert(succeeded(maybeAllocationBytes));
        const uint64_t allocationBytes = *maybeAllocationBytes;
        if (allocationBytes > reportMax) {
          reportMax = allocationBytes;
          reportStorageIndex = storageIndex;
        }
      }
      SmallVector<int64_t> physicalIndices =
          diagnosticMembers->lookup(reportStorageIndex);
      assert(!physicalIndices.empty() &&
             "reported storage allocation must contain one physical DFB");
      llvm::sort(physicalIndices);
      auto bindIt = bindForPhysicalIndex.find(physicalIndices.front());
      assert(bindIt != bindForPhysicalIndex.end() &&
             "reported physical DFB must have one declaration");
      return bindIt->second;
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
