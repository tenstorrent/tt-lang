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

  // Number top-level body ops only, and project a nested op onto its
  // body-block ancestor. This runs after LowerToLoops, so a circular buffer
  // used inside a loop is acquired/released every iteration: across the loop
  // back-edge it is live for the WHOLE loop, not just one static interval
  // within the body. Projecting every in-loop use to the single loop op index
  // collapses all buffers used in one loop to the same position so they
  // overlap and never share a slot; only buffers in different top-level
  // regions (e.g. distinct phases/loops) can reuse, which is sound.
  DenseMap<Operation *, int64_t> opIndex;
  int64_t idx = 0;
  for (Operation &op : body) {
    opIndex[&op] = idx++;
  }
  int64_t lastOpIdx = idx - 1;

  auto getBodyIndex = [&](Operation *op) -> int64_t {
    if (op->getBlock() == &body) {
      return opIndex[op];
    }
    Operation *ancestor = body.findAncestorOpInBlock(*op);
    assert(ancestor && "operation must be reachable from function body");
    return opIndex[ancestor];
  };

  // Build intervals grouped by reuse class. Two compiler-allocated DFBs may
  // share a physical index when they have the same element type and block
  // count: each per-op reserve/wait carries its own tile count (derived from
  // the unchanged bind_cb type), and the L1 slot is sized to the largest
  // member, so a smaller buffer reusing a larger slot only touches a prefix.
  using ReuseClass = std::pair<Type, int64_t>;
  llvm::MapVector<ReuseClass, SmallVector<Interval>> classToIntervals;
  DenseMap<Value, BindCBOp> valueToBindOp;

  for (BindCBOp bindOp : dfbOps) {
    assert(bindOp->getBlock() == &body &&
           "compiler-allocated BindCBOp must be in function body block");

    auto cbType = mlir::cast<CircularBufferType>(bindOp.getResult().getType());
    Value cbVal = bindOp.getResult();
    // Lifetime starts at the first acquire (reserve/wait) on this CB, not
    // at the bind_cb itself: bind_cb is just a declaration, and hoisting
    // it to the function body entry would otherwise collapse all compiler-
    // allocated DFB starts together and defeat reuse. If there is no
    // acquire (synthetic IR, pop-only), fall back to the bind_cb position.
    //
    // The end is the last op that reads the buffer. cb_pop ops do not exist
    // yet here (TTLInsertCBSync runs later), so follow the consumer chain
    // wait -> attach_cb -> compute consumer rather than relying on pops; an
    // intermediate is dead once its consuming op has executed.
    int64_t start = lastOpIdx;
    int64_t end = opIndex[bindOp];
    bool sawAcquire = false;
    auto extendEnd = [&](Operation *op) { end = std::max(end, getBodyIndex(op)); };

    for (OpOperand &use : cbVal.getUses()) {
      Operation *user = use.getOwner();
      if (isa<CBReserveOp, CBWaitOp>(user)) {
        start = std::min(start, getBodyIndex(user));
        sawAcquire = true;
        extendEnd(user);
      }
      if (isa<CBPopOp>(user)) {
        extendEnd(user);
      }
      if (auto waitOp = dyn_cast<CBWaitOp>(user)) {
        for (Operation *reader : waitOp.getResult().getUsers()) {
          extendEnd(reader);
          if (auto attachOp = dyn_cast<AttachCBOp>(reader)) {
            for (Operation *consumer : attachOp.getResult().getUsers()) {
              extendEnd(consumer);
            }
          }
        }
      }
    }

    if (!sawAcquire) {
      start = opIndex[bindOp];
    }

    // Degenerate (no consumer found): treat as live for the whole function.
    if (end < start) {
      end = lastOpIdx;
    }

    ReuseClass key(cbType.getElementType(), cbType.getBlockCount());
    classToIntervals[key].push_back({start, end, cbVal});
    valueToBindOp[cbVal] = bindOp;
  }

  // Find the base index (smallest compiler-allocated index).
  int32_t baseIndex = INT32_MAX;
  for (BindCBOp bindOp : dfbOps) {
    int32_t cbIdx = static_cast<int32_t>(bindOp.getCbIndex().getSExtValue());
    baseIndex = std::min(baseIndex, cbIdx);
  }

  // Linear scan per reuse class. Each class gets a contiguous block of
  // physical DFB indices starting at baseIndex + cumulative offset from
  // prior classes.
  MLIRContext *ctx = funcOp.getContext();
  int32_t nextSlotOffset = 0;

  for (auto &[key, intervals] : classToIntervals) {
    llvm::sort(intervals, [](const Interval &lhs, const Interval &rhs) {
      return lhs.start < rhs.start;
    });

    SmallVector<Interval *> active;
    llvm::SmallBitVector freeSlots(intervals.size());
    freeSlots.set();
    DenseMap<Value, int32_t> slotAssignment;
    int32_t maxSlot = -1;

    for (Interval &interval : intervals) {
      // Expire intervals whose lifetime ended strictly before this one
      // starts. Intervals are closed [start, end] (both endpoints live), so a
      // slot frees only when act->end < interval.start; sharing an endpoint
      // means both are live at that op and must keep distinct slots. Two
      // buffers projected onto the same enclosing loop op share that point and
      // so never reuse, which is what loop liveness requires.
      SmallVector<Interval *> expired;
      for (Interval *act : active) {
        if (act->end < interval.start) {
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
    // BindCBOps may share the same index, possibly with different shapes.
    // The module attribute needs one entry per unique physical DFB, sized
    // to the largest member so every reuser's tiles fit in the slot.
    llvm::DenseMap<int32_t, BindCBOp> uniqueByIndex;
    for (BindCBOp bindOp : compilerAllocatedOps) {
      int32_t dfbIdx = static_cast<int32_t>(bindOp.getCbIndex().getSExtValue());
      auto [it, inserted] = uniqueByIndex.try_emplace(dfbIdx, bindOp);
      if (!inserted) {
        auto existing =
            mlir::cast<CircularBufferType>(it->second.getResult().getType());
        auto current =
            mlir::cast<CircularBufferType>(bindOp.getResult().getType());
        if (current.getElementsPerBlock() > existing.getElementsPerBlock()) {
          it->second = bindOp;
        }
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
