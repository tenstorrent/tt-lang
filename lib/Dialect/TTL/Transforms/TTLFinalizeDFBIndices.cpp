// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Finalize DFB Indices
//===----------------------------------------------------------------------===//
//
// Module-level pass that runs after all DFB-creating passes. Reuses
// compiler-allocated DFB indices when lifetimes do not overlap, then
// computes the true DFB count, updates ttl.base_cta_index on every
// function, and collects compiler-allocated DFBs into the
// ttl.compiler_allocated_dfbs module attribute for the Python runtime.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFINALIZEDFBINDICES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Reuse compiler-allocated DFB indices within a function when their
/// lifetimes do not overlap. Groups DFBs by CircularBufferType and runs
/// linear scan allocation per group.
static void reuseDFBIndices(func::FuncOp funcOp, ArrayRef<BindCBOp> dfbOps) {
  if (dfbOps.size() <= 1) {
    return;
  }

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
  llvm::MapVector<Type, SmallVector<Interval>> typeToIntervals;
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

    typeToIntervals[cbVal.getType()].push_back({start, end, cbVal});
    valueToBindOp[cbVal] = bindOp;
  }

  // Find the base index (smallest compiler-allocated index).
  int32_t baseIndex = INT32_MAX;
  for (BindCBOp bindOp : dfbOps) {
    int32_t cbIdx = static_cast<int32_t>(bindOp.getCbIndex().getSExtValue());
    baseIndex = std::min(baseIndex, cbIdx);
  }

  // Linear scan per type partition. Each partition gets a contiguous
  // block of physical DFB indices starting at baseIndex + cumulative
  // offset from prior partitions.
  MLIRContext *ctx = funcOp.getContext();
  int32_t nextSlotOffset = 0;

  for (auto &[type, intervals] : typeToIntervals) {
    llvm::sort(intervals, [](const Interval &lhs, const Interval &rhs) {
      return lhs.start < rhs.start;
    });

    SmallVector<Interval *> active;
    llvm::SmallBitVector freeSlots(intervals.size());
    freeSlots.set();
    DenseMap<Value, int32_t> slotAssignment;
    int32_t maxSlot = -1;

    for (Interval &interval : intervals) {
      // Expire intervals whose lifetime ended before this one starts.
      SmallVector<Interval *> expired;
      for (Interval *act : active) {
        if (act->end <= interval.start) {
          freeSlots.set(slotAssignment[act->value]);
          expired.push_back(act);
        }
      }
      for (Interval *exp : expired) {
        llvm::erase(active, exp);
      }

      int freeSlot = freeSlots.find_first();
      assert(freeSlot >= 0 && "DFB slot allocation always succeeds");
      freeSlots.reset(freeSlot);
      slotAssignment[interval.value] = freeSlot;
      maxSlot = std::max(maxSlot, static_cast<int32_t>(freeSlot));
      active.push_back(&interval);

      LLVM_DEBUG({
        llvm::dbgs() << "DFB reuse: [" << interval.start << ", " << interval.end
                     << "] -> slot " << freeSlot << "\n";
      });
    }

    // Rewrite BindCBOp indices to the assigned physical slot.
    for (auto &[value, slot] : slotAssignment) {
      int32_t newIndex = baseIndex + nextSlotOffset + slot;
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
}

struct TTLFinalizeDFBIndicesPass
    : public impl::TTLFinalizeDFBIndicesBase<TTLFinalizeDFBIndicesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    // Collect compiler-allocated BindCBOps grouped by parent function.
    llvm::MapVector<func::FuncOp, SmallVector<BindCBOp>> funcToDFBs;
    moduleOp->walk([&](BindCBOp bindOp) {
      if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
        auto funcOp = bindOp->getParentOfType<func::FuncOp>();
        funcToDFBs[funcOp].push_back(bindOp);
      }
    });

    // Run DFB index reuse per function.
    for (auto &[funcOp, dfbOps] : funcToDFBs) {
      reuseDFBIndices(funcOp, dfbOps);
    }

    // Recompute DFB count after reuse may have changed indices.
    int32_t numDFBs = getNextAvailableDFBIndex(moduleOp);
    if (numDFBs <= 0) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Total DFB count: " << numDFBs << "\n");

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

    // Update ttl.base_cta_index on every function that has it.
    moduleOp->walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr(kBaseCTAIndexAttrName)) {
        funcOp->setAttr(kBaseCTAIndexAttrName,
                        builder.getI32IntegerAttr(numDFBs));
      }
    });

    // Re-collect compiler-allocated ops (indices may have changed).
    SmallVector<BindCBOp> compilerAllocatedOps;
    moduleOp->walk([&](BindCBOp bindOp) {
      if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
        compilerAllocatedOps.push_back(bindOp);
      }
    });

    if (compilerAllocatedOps.empty()) {
      return;
    }

    // Deduplicate entries by physical index. After reuse, multiple
    // BindCBOps may share the same index. The module attribute needs
    // one entry per unique physical DFB.
    llvm::DenseMap<int32_t, BindCBOp> uniqueByIndex;
    for (BindCBOp bindOp : compilerAllocatedOps) {
      int32_t dfbIdx = static_cast<int32_t>(bindOp.getCbIndex().getSExtValue());
      auto [it, inserted] = uniqueByIndex.try_emplace(dfbIdx, bindOp);
      if (!inserted) {
        assert(it->second.getResult().getType() ==
                   bindOp.getResult().getType() &&
               "compiler-allocated DFBs sharing an index must have the "
               "same CircularBufferType");
      }
    }

    // Sort by index for deterministic output.
    SmallVector<std::pair<int32_t, BindCBOp>> sorted(uniqueByIndex.begin(),
                                                     uniqueByIndex.end());
    llvm::sort(sorted,
               [](auto &lhs, auto &rhs) { return lhs.first < rhs.first; });

    MLIRContext *ctx = moduleOp.getContext();
    SmallVector<Attribute> entries;
    for (auto &[dfbIdx, bindOp] : sorted) {
      auto cbType =
          mlir::cast<CircularBufferType>(bindOp.getResult().getType());
      SmallVector<NamedAttribute> entryAttrs;
      entryAttrs.push_back(
          builder.getNamedAttr("dfb_index", builder.getI32IntegerAttr(dfbIdx)));
      entryAttrs.push_back(builder.getNamedAttr(
          "num_tiles", builder.getI32IntegerAttr(static_cast<int32_t>(
                           cbType.getElementsPerBlock()))));
      entryAttrs.push_back(builder.getNamedAttr(
          "element_type", TypeAttr::get(cbType.getElementType())));
      entryAttrs.push_back(builder.getNamedAttr(
          "block_count", builder.getI32IntegerAttr(
                             static_cast<int32_t>(cbType.getBlockCount()))));
      entries.push_back(DictionaryAttr::get(ctx, entryAttrs));
    }

    moduleOp->setAttr(kCompilerAllocatedDFBsAttrName,
                      ArrayAttr::get(ctx, entries));
  }
};

} // namespace

} // namespace mlir::tt::ttl
