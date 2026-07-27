// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Finalize DFB Indices
//===----------------------------------------------------------------------===//
//
// Module-level pass that runs after all DFB-creating passes. Assigns physical
// indices to logical DFBs using module-wide liveness, updates
// ttl.base_cta_index, and emits the physical allocation table consumed by the
// Python runtime.
//
//===----------------------------------------------------------------------===//

#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/LiveIntervalUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <functional>

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFINALIZEDFBINDICES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static int32_t getFirstCompilerDFBIndex(ModuleOp moduleOp) {
  int32_t maxUserIndex = -1;
  moduleOp->walk([&](BindCBOp bindOp) {
    if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
      return;
    }
    maxUserIndex = std::max(
        maxUserIndex, static_cast<int32_t>(bindOp.getCbIndex().getSExtValue()));
  });
  return maxUserIndex + 1;
}

/// Assign physical indices to one function's compiler-allocated DFBs,
/// reusing indices when lifetimes do not overlap.
static int32_t assignPhysicalDFBIndices(func::FuncOp funcOp,
                                        ArrayRef<BindCBOp> dfbOps,
                                        int32_t firstPhysicalIndex) {
  Block &body = funcOp.getBody().front();

  // Assign sequential indices to all operations in the body block.
  DenseMap<Operation *, int64_t> opIndex;
  int64_t idx = 0;
  for (Operation &op : body) {
    opIndex[&op] = idx++;
  }
  int64_t lastOpIdx = idx - 1;

  // Project a nested operation to its ancestor in the body block.
  // After LowerToLoops or SubblockComputeForDST, CBPopOps may end up
  // inside loops or compute regions.
  auto getBodyIndex = [&](Operation *op) -> int64_t {
    if (op->getBlock() == &body) {
      return opIndex[op];
    }
    Operation *ancestor = body.findAncestorOpInBlock(*op);
    assert(ancestor && "operation must be reachable from function body");
    return opIndex[ancestor];
  };

  // Build intervals grouped by CircularBufferType.
  llvm::MapVector<Type, SmallVector<ValueLiveInterval>> typeToIntervals;
  DenseMap<Value, BindCBOp> valueToBindOp;

  for (BindCBOp bindOp : dfbOps) {
    assert(bindOp->getBlock() == &body &&
           "compiler-allocated BindCBOp must be in function body block");

    Value cbVal = bindOp.getResult();
    // Lifetime starts at the first acquire (reserve/wait) on this CB, not
    // at the bind_cb itself: bind_cb is just a declaration, and hoisting
    // it to the function body entry would otherwise collapse all compiler-
    // allocated DFB starts together and defeat reuse. If there is no
    // acquire (synthetic IR, pop-only), fall back to the bind_cb position.
    int64_t start = lastOpIdx;
    int64_t end = opIndex[bindOp];
    bool sawAcquire = false;

    for (OpOperand &use : cbVal.getUses()) {
      Operation *user = use.getOwner();
      int64_t useIdx = getBodyIndex(user);
      if (isa<CBReserveOp, CBWaitOp>(user)) {
        start = std::min(start, useIdx);
        sawAcquire = true;
      }
      if (isa<CBPopOp>(user)) {
        end = std::max(end, useIdx);
      }
    }

    if (!sawAcquire) {
      start = opIndex[bindOp];
    }

    // No cb_pop means the DFB's L1 is never explicitly released --
    // conservatively treat it as live for the entire function.
    if (end <= start) {
      end = lastOpIdx;
    }

    SmallVector<ValueLiveInterval> &intervals =
        typeToIntervals[cbVal.getType()];
    int64_t ordinal = static_cast<int64_t>(intervals.size());
    intervals.push_back({start, end, cbVal, ordinal});
    valueToBindOp[cbVal] = bindOp;
  }

  // Linear scan per type partition. Each partition gets a contiguous
  // block of physical DFB indices starting at firstPhysicalIndex + cumulative
  // offset from prior partitions.
  MLIRContext *ctx = funcOp.getContext();
  int32_t nextSlotOffset = 0;

  for (auto &entry : typeToIntervals) {
    SmallVector<ValueLiveInterval> &intervals = entry.second;

    SmallVector<SmallVector<ValueLiveInterval>> colorUsers =
        assignGreedyIntervalColors<ValueLiveInterval>(
            intervals, std::less<ValueLiveInterval>(),
            [](const ValueLiveInterval &lhs, const ValueLiveInterval &rhs) {
              return intervalsOverlap(lhs, rhs);
            });

    DenseMap<Value, int32_t> slotAssignment;
    int32_t maxSlot = -1;

    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      int32_t slotIndex = static_cast<int32_t>(indexedColor.index());
      maxSlot = std::max(maxSlot, slotIndex);
      for (const ValueLiveInterval &interval : indexedColor.value()) {
        slotAssignment[interval.value] = slotIndex;

        LLVM_DEBUG({
          llvm::dbgs() << "DFB reuse: [" << interval.start << ", "
                       << interval.end << "] -> slot " << slotIndex << "\n";
        });
      }
    }

    // Rewrite BindCBOp indices to the assigned physical slot.
    for (auto &[value, slot] : slotAssignment) {
      int32_t newIndex = firstPhysicalIndex + nextSlotOffset + slot;
      BindCBOp bindOp = valueToBindOp[value];
      bindOp.setCbIndexAttr(IntegerAttr::get(IndexType::get(ctx), newIndex));
    }

    nextSlotOffset += maxSlot + 1;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "DFB reuse: " << dfbOps.size()
                 << " compiler-allocated DFBs -> " << nextSlotOffset
                 << " physical slot(s)\n";
  });
  return nextSlotOffset;
}

/// Emits the complete physical DFB allocation table consumed by the runtime.
///
/// Multiple logical DFBs may share one physical index, but every declaration
/// at that index must have the same type so one runtime descriptor suffices.
static LogicalResult emitDFBMetadata(ModuleOp moduleOp, OpBuilder &builder,
                                     ArrayRef<BindCBOp> bindOps) {
  if (bindOps.empty()) {
    moduleOp->setAttr(kDFBAllocationsAttrName,
                      ArrayAttr::get(moduleOp.getContext(), {}));
    return success();
  }

  llvm::DenseMap<int32_t, BindCBOp> uniqueByIndex;
  for (BindCBOp bindOp : bindOps) {
    int32_t dfbIndex = static_cast<int32_t>(bindOp.getCbIndex().getSExtValue());
    auto [existingIt, inserted] = uniqueByIndex.try_emplace(dfbIndex, bindOp);
    if (!inserted && existingIt->second.getResult().getType() !=
                         bindOp.getResult().getType()) {
      return bindOp.emitOpError()
             << "physical DFB index " << dfbIndex
             << " has inconsistent CircularBufferType values "
             << existingIt->second.getResult().getType() << " and "
             << bindOp.getResult().getType();
    }
  }

  SmallVector<std::pair<int32_t, BindCBOp>> sorted(uniqueByIndex.begin(),
                                                   uniqueByIndex.end());
  llvm::sort(sorted, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  MLIRContext *context = moduleOp.getContext();
  SmallVector<Attribute> entries;
  for (auto &[dfbIndex, bindOp] : sorted) {
    auto dfbType = cast<CircularBufferType>(bindOp.getResult().getType());
    SmallVector<NamedAttribute> entryAttributes;
    entryAttributes.push_back(
        builder.getNamedAttr("dfb_index", builder.getI32IntegerAttr(dfbIndex)));
    entryAttributes.push_back(builder.getNamedAttr(
        "num_tiles", builder.getI32IntegerAttr(
                         static_cast<int32_t>(dfbType.getElementsPerBlock()))));
    entryAttributes.push_back(builder.getNamedAttr(
        "element_type", TypeAttr::get(dfbType.getElementType())));
    entryAttributes.push_back(builder.getNamedAttr(
        "block_count", builder.getI32IntegerAttr(
                           static_cast<int32_t>(dfbType.getBlockCount()))));
    entries.push_back(DictionaryAttr::get(context, entryAttributes));
  }

  moduleOp->setAttr(kDFBAllocationsAttrName, ArrayAttr::get(context, entries));
  return success();
}

struct TTLFinalizeDFBIndicesPass
    : public impl::TTLFinalizeDFBIndicesBase<TTLFinalizeDFBIndicesPass> {
  using Base::Base;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    if (reuseUserDFBs) {
      const auto &analysis = getAnalysis<DFBConcurrentKernelLivenessAnalysis>();
      if (!analysis.succeeded()) {
        Operation *errorOperation = analysis.getErrorOperation();
        if (!errorOperation) {
          errorOperation = moduleOp.getOperation();
        }
        errorOperation->emitOpError() << analysis.getErrorMessage();
        signalPassFailure();
        return;
      }

      MLIRContext *context = moduleOp.getContext();
      // Delaying all mutations until analysis succeeds prevents an error from
      // leaving partially finalized DFB declarations.
      for (const DFBPhysicalIndexAssignment &assignment :
           analysis.getAssignments()) {
        for (BindCBOp bindOp : assignment.declarations) {
          bindOp.setDfbIdAttr(
              IntegerAttr::get(IndexType::get(context), assignment.logicalId));
          bindOp.setCbIndexAttr(IntegerAttr::get(IndexType::get(context),
                                                 assignment.physicalIndex));
        }
        LLVM_DEBUG({
          llvm::dbgs() << "DFB reuse: logical DFB " << assignment.logicalId
                       << " -> physical index " << assignment.physicalIndex
                       << (assignment.bounded ? " (bounded)\n"
                                              : " (unbounded)\n");
        });
      }

      int32_t numDFBs = analysis.getPhysicalSlotCount();
      LLVM_DEBUG(llvm::dbgs() << "Total DFB count: " << numDFBs << "\n");

      SmallVector<BindCBOp> allBindOps;
      moduleOp->walk([&](BindCBOp bindOp) { allBindOps.push_back(bindOp); });
      if (failed(emitDFBMetadata(moduleOp, builder, allBindOps))) {
        signalPassFailure();
        return;
      }

      if (numDFBs <= 0) {
        return;
      }

      moduleOp->walk([&](func::FuncOp funcOp) {
        if (funcOp->hasAttr(kBaseCTAIndexAttrName)) {
          funcOp->setAttr(kBaseCTAIndexAttrName,
                          builder.getI32IntegerAttr(numDFBs));
        }
      });
      return;
    }

    // Validate logical identities before mutating compiler DFB indices so an
    // identity error cannot leave partially finalized IR.
    DFBLogicalIdentityAnalysis identityAnalysis(moduleOp);
    if (!identityAnalysis.succeeded()) {
      Operation *errorOperation = identityAnalysis.getErrorOperation();
      if (!errorOperation) {
        errorOperation = moduleOp.getOperation();
      }
      errorOperation->emitOpError() << identityAnalysis.getErrorMessage();
      signalPassFailure();
      return;
    }

    // Collect compiler-allocated BindCBOps grouped by parent function.
    llvm::MapVector<func::FuncOp, SmallVector<BindCBOp>> funcToDFBs;
    moduleOp->walk([&](BindCBOp bindOp) {
      if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
        auto funcOp = bindOp->getParentOfType<func::FuncOp>();
        funcToDFBs[funcOp].push_back(bindOp);
      }
    });

    // Provisional compiler indices are function-local. Assign disjoint
    // module-wide ranges after the highest user-declared index.
    int32_t nextCompilerDFBIndex = getFirstCompilerDFBIndex(moduleOp);
    for (auto &[funcOp, dfbOps] : funcToDFBs) {
      int32_t physicalSlotCount =
          assignPhysicalDFBIndices(funcOp, dfbOps, nextCompilerDFBIndex);
      nextCompilerDFBIndex += physicalSlotCount;
    }

    // Recompute DFB count after reuse may have changed indices.
    int32_t numDFBs = getNextAvailableDFBIndex(moduleOp.getOperation());
    if (numDFBs > 0) {
      LLVM_DEBUG(llvm::dbgs() << "Total DFB count: " << numDFBs << "\n");
    }

    // Verify the final DFB count does not exceed the hardware limit.
    if (numDFBs > kMaxCircularBuffers) {
      // Count compiler-allocated physical slots (after reuse).
      int32_t compilerSlots = 0;
      for (auto &[funcOp, dfbOps] : funcToDFBs) {
        llvm::SmallDenseSet<int32_t> uniqueIndices;
        for (BindCBOp bindOp : dfbOps) {
          uniqueIndices.insert(
              static_cast<int32_t>(bindOp.getCbIndex().getSExtValue()));
        }
        compilerSlots += static_cast<int32_t>(uniqueIndices.size());
      }
      moduleOp.emitError()
          << "need " << numDFBs << " DFB indices but hardware supports "
          << "at most " << kMaxCircularBuffers << " (" << compilerSlots
          << " compiler-allocated after reuse); reduce the number of "
          << "user-declared dataflow buffers or split the computation "
          << "into multiple kernels";
      signalPassFailure();
      return;
    }

    // Delay identity materialization until allocation and capacity validation
    // succeed so those failures leave the identity attributes unchanged.
    MLIRContext *context = moduleOp.getContext();
    for (const DFBLogicalIdentityAssignment &assignment :
         identityAnalysis.getAssignments()) {
      BindCBOp declaration = assignment.declaration;
      declaration.setDfbIdAttr(
          IntegerAttr::get(IndexType::get(context), assignment.logicalId));
    }

    if (numDFBs > 0) {
      moduleOp->walk([&](func::FuncOp funcOp) {
        if (funcOp->hasAttr(kBaseCTAIndexAttrName)) {
          funcOp->setAttr(kBaseCTAIndexAttrName,
                          builder.getI32IntegerAttr(numDFBs));
        }
      });
    }

    SmallVector<BindCBOp> allBindOps;
    moduleOp->walk([&](BindCBOp bindOp) { allBindOps.push_back(bindOp); });
    if (failed(emitDFBMetadata(moduleOp, builder, allBindOps))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
