// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Finalize DFB Indices
//===----------------------------------------------------------------------===//
//
// Module-level pass that runs after all DFB-creating passes. Reuses DFB
// indices when lifetimes do not overlap, then computes the true DFB count,
// updates ttl.base_cta_index on every function, and publishes the final
// index assignment for the Python runtime (ttl.dfb_index_map for
// user-declared DFBs, ttl.compiler_allocated_dfbs for compiler-allocated
// ones).
//
// A logical DFB is one cb_index, bound by one bind_cb per kernel thread that
// touches it. The same physical CB index has shared pages_received /
// pages_acked counters and per-RISC read/write pointers, so two logical DFBs
// may share an index only when:
//   - both have the same producer thread and the same consumer thread,
//   - their per-thread lifetimes are disjoint in both threads' program order,
//   - they have the same element type (uniform page size). Capacity is a
//     >= check, not equality: the slot is sized to the member needing the
//     most pages, and every reserve/wait carries its own block size.
// With matching thread identity, per-thread program order plus reserve/wait
// blocking make disjoint program-order lifetimes sound at runtime; sharing
// across different producer or consumer threads is never sound without a
// counter/pointer reset and is therefore not attempted.
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
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFINALIZEDFBINDICES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Closed [start, end] lifetime of a logical DFB within one kernel thread,
/// in that thread's top-level program order.
struct ThreadInterval {
  int64_t start;
  int64_t end;
};

/// One logical DFB: a cb_index with one bind_cb per kernel thread.
struct LogicalDFB {
  int64_t origIndex = -1;
  SmallVector<BindCBOp, 3> binds;
  func::FuncOp producer;
  func::FuncOp consumer;
  llvm::SmallDenseMap<Operation *, ThreadInterval, 4> intervals;
  llvm::SmallDenseMap<Operation *, int64_t, 4> firstAcquire;
  Type elemType;
  int64_t blockCount = 0;
  int64_t elemsPerBlock = 0;
  bool compilerAllocated = false;
  bool eligible = true;
  int64_t finalIndex = -1;

  int64_t totalPages() const { return blockCount * elemsPerBlock; }
};

/// Position of an op in its kernel thread's top-level program order. Nested
/// ops project onto their body-block ancestor: a buffer touched inside a
/// loop is acquired and released every iteration, so across the back-edge it
/// is live for the whole loop, not one static interval within the body.
struct ThreadOrder {
  DenseMap<Operation *, int64_t> topLevel;
  int64_t lastIdx = 0;

  explicit ThreadOrder(func::FuncOp func) {
    int64_t idx = 0;
    for (Operation &op : func.getBody().front()) {
      topLevel[&op] = idx++;
    }
    lastIdx = idx == 0 ? 0 : idx - 1;
  }

  int64_t index(Operation *op, func::FuncOp func) {
    Block &body = func.getBody().front();
    if (op->getBlock() == &body) {
      return topLevel[op];
    }
    Operation *ancestor = body.findAncestorOpInBlock(*op);
    assert(ancestor && "operation must be reachable from function body");
    return topLevel[ancestor];
  }
};

/// Extend the per-thread interval of `dfb` to cover `op`.
void extendInterval(LogicalDFB &dfb, Operation *op, func::FuncOp func,
                    ThreadOrder &order) {
  int64_t pos = order.index(op, func);
  auto [it, inserted] =
      dfb.intervals.try_emplace(func.getOperation(), ThreadInterval{pos, pos});
  if (!inserted) {
    it->second.start = std::min(it->second.start, pos);
    it->second.end = std::max(it->second.end, pos);
  }
}

/// Record producer/consumer thread identity; multiple distinct threads in
/// either role make the DFB ineligible (sharing its index would need a
/// counter/pointer reset, which does not exist).
void recordRole(func::FuncOp &role, func::FuncOp func, bool &eligible) {
  if (!role) {
    role = func;
  } else if (role != func) {
    eligible = false;
  }
}

bool disjoint(const ThreadInterval &a, const ThreadInterval &b) {
  return a.end < b.start || b.end < a.start;
}

/// Two logical DFBs of one reuse class conflict when their lifetimes overlap
/// in the producer thread or in the consumer thread. Intervals are closed:
/// sharing an endpoint means both are live at that op.
bool conflicts(const LogicalDFB &a, const LogicalDFB &b) {
  for (auto &[func, interval] : a.intervals) {
    auto it = b.intervals.find(func);
    if (it != b.intervals.end() && !disjoint(interval, it->second)) {
      return true;
    }
  }
  return false;
}

} // namespace

struct TTLFinalizeDFBIndicesPass
    : public impl::TTLFinalizeDFBIndicesBase<TTLFinalizeDFBIndicesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();
    OpBuilder builder(ctx);

    // Collect logical DFBs by original index across kernel threads.
    llvm::MapVector<int64_t, LogicalDFB> dfbs;
    llvm::SmallDenseMap<Operation *, std::unique_ptr<ThreadOrder>> orders;
    bool sawBind = false;

    moduleOp->walk([&](BindCBOp bindOp) {
      sawBind = true;
      func::FuncOp func = getEnclosingKernelThread(bindOp);
      int64_t idx = bindOp.getCbIndex().getSExtValue();
      LogicalDFB &dfb = dfbs[idx];
      dfb.origIndex = idx;
      dfb.binds.push_back(bindOp);
      auto cbType = mlir::cast<CircularBufferType>(bindOp.getResult().getType());
      dfb.elemType = cbType.getElementType();
      dfb.blockCount = cbType.getBlockCount();
      dfb.elemsPerBlock =
          std::max(dfb.elemsPerBlock, cbType.getElementsPerBlock());
      dfb.compilerAllocated |= bindOp->hasAttr(kCompilerAllocatedAttrName);
      if (!func) {
        dfb.eligible = false;
        return;
      }

      auto &order = orders[func.getOperation()];
      if (!order) {
        order = std::make_unique<ThreadOrder>(func);
      }

      // The bind anchors the interval; binds are hoisted to the function
      // entry, so the start is refined to the first acquire below.
      extendInterval(dfb, bindOp, func, *order);

      for (Operation *user : bindOp.getResult().getUsers()) {
        extendInterval(dfb, user, func, *order);
        if (isa<CBReserveOp, CBWaitOp>(user)) {
          int64_t pos = order->index(user, func);
          auto [it, inserted] =
              dfb.firstAcquire.try_emplace(func.getOperation(), pos);
          if (!inserted) {
            it->second = std::min(it->second, pos);
          }
        }
        if (isa<CBReserveOp, CBPushOp>(user)) {
          recordRole(dfb.producer, func, dfb.eligible);
        }
        if (isa<CBWaitOp, CBPopOp>(user)) {
          recordRole(dfb.consumer, func, dfb.eligible);
        }
        // A consumed block stays live until its last reader: follow
        // wait -> readers -> attach_cb -> consumers one level deep.
        if (auto waitOp = dyn_cast<CBWaitOp>(user)) {
          for (Operation *reader : waitOp.getResult().getUsers()) {
            extendInterval(dfb, reader, func, *order);
            if (auto attachOp = dyn_cast<AttachCBOp>(reader)) {
              for (Operation *consumer : attachOp.getResult().getUsers()) {
                extendInterval(dfb, consumer, func, *order);
              }
            }
          }
        }
      }
    });

    if (!sawBind) {
      return;
    }

    // A DFB with no acquire and no release anywhere (declared but unused)
    // keeps its index. One-sided DFBs reuse within the thread that touches
    // them. Lifetime starts at the first acquire in each thread; without one
    // the hoisted bind position stands, conservatively from function entry.
    for (auto &[idx, dfb] : dfbs) {
      if (!dfb.producer && !dfb.consumer) {
        dfb.eligible = false;
        continue;
      }
      if (!dfb.producer) {
        dfb.producer = dfb.consumer;
      }
      if (!dfb.consumer) {
        dfb.consumer = dfb.producer;
      }
      for (auto &[func, interval] : dfb.intervals) {
        auto it = dfb.firstAcquire.find(func);
        if (it != dfb.firstAcquire.end()) {
          interval.start = it->second;
        }
      }
    }

    // Group eligible DFBs into reuse classes; keep deterministic order by
    // original index. Class members may differ in block count and tiles per
    // block: capacity is a >= constraint, the slot is sized to the member
    // needing the most pages, and per-op tile counts come from each bind_cb's
    // own type.
    // User and compiler DFBs share slots freely; the runtime keeps the
    // larger config when both describe one index.
    using ClassKey = std::tuple<Type, Operation *, Operation *>;
    llvm::MapVector<ClassKey, SmallVector<LogicalDFB *>> classes;
    for (auto &[idx, dfb] : dfbs) {
      if (dfb.eligible) {
        classes[{dfb.elemType, dfb.producer.getOperation(),
                 dfb.consumer.getOperation()}]
            .push_back(&dfb);
      } else {
        dfb.finalIndex = dfb.origIndex;
      }
    }

    // Greedy interval coloring per class: walk by ascending producer start
    // and place on the first slot whose members are all disjoint in both
    // threads. A slot takes the lowest original index of its members, so
    // indices only ever decrease and never collide with ineligible DFBs.
    SmallVector<SmallVector<LogicalDFB *>> slots;
    for (auto &[key, members] : classes) {
      llvm::sort(members, [](LogicalDFB *a, LogicalDFB *b) {
        auto pa = a->intervals.find(a->producer.getOperation())->second;
        auto pb = b->intervals.find(b->producer.getOperation())->second;
        return std::tie(pa.start, pa.end, a->origIndex) <
               std::tie(pb.start, pb.end, b->origIndex);
      });
      size_t classBase = slots.size();
      for (LogicalDFB *dfb : members) {
        size_t placed = slots.size();
        for (size_t s = classBase; s < slots.size(); ++s) {
          if (llvm::none_of(slots[s], [&](LogicalDFB *member) {
                return conflicts(*member, *dfb);
              })) {
            placed = s;
            break;
          }
        }
        if (placed == slots.size()) {
          slots.emplace_back();
        }
        slots[placed].push_back(dfb);
      }
    }

    for (auto &slot : slots) {
      int64_t slotIndex = slot.front()->origIndex;
      for (LogicalDFB *dfb : slot) {
        slotIndex = std::min(slotIndex, dfb->origIndex);
      }
      for (LogicalDFB *dfb : slot) {
        dfb->finalIndex = slotIndex;
      }
    }

    // Compact to a dense index space (the runtime builds one CB descriptor
    // per index), preserving relative order.
    SmallVector<int64_t> usedIndices;
    for (auto &[idx, dfb] : dfbs) {
      usedIndices.push_back(dfb.finalIndex);
    }
    llvm::sort(usedIndices);
    usedIndices.erase(llvm::unique(usedIndices), usedIndices.end());
    DenseMap<int64_t, int64_t> rank;
    for (auto [i, index] : llvm::enumerate(usedIndices)) {
      rank[index] = static_cast<int64_t>(i);
    }
    for (auto &[idx, dfb] : dfbs) {
      dfb.finalIndex = rank[dfb.finalIndex];
    }

    // Rewrite bind_cb indices in every kernel thread.
    for (auto &[idx, dfb] : dfbs) {
      if (dfb.finalIndex == dfb.origIndex) {
        continue;
      }
      for (BindCBOp bindOp : dfb.binds) {
        bindOp.setCbIndexAttr(
            IntegerAttr::get(IndexType::get(ctx), dfb.finalIndex));
      }
      LLVM_DEBUG(llvm::dbgs() << "DFB reuse: cb" << dfb.origIndex << " -> cb"
                              << dfb.finalIndex << "\n");
    }

    int32_t numDFBs = getNextAvailableDFBIndex(moduleOp);
    LLVM_DEBUG(llvm::dbgs() << "Total DFB count: " << numDFBs << "\n");

    if (numDFBs > kMaxCircularBuffers) {
      moduleOp.emitError()
          << "need " << numDFBs << " DFB indices after reuse but hardware "
          << "supports at most " << kMaxCircularBuffers
          << "; reduce the number of dataflow buffers or split the "
          << "computation into multiple kernels";
      signalPassFailure();
      return;
    }

    moduleOp->walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr(kBaseCTAIndexAttrName)) {
        funcOp->setAttr(kBaseCTAIndexAttrName,
                        builder.getI32IntegerAttr(numDFBs));
      }
    });

    // Publish remapped user DFB indices for the Python runtime.
    SmallVector<Attribute> mapEntries;
    for (auto &[idx, dfb] : dfbs) {
      if (dfb.compilerAllocated || dfb.finalIndex == dfb.origIndex) {
        continue;
      }
      mapEntries.push_back(DictionaryAttr::get(
          ctx, {builder.getNamedAttr(
                    "old_index",
                    builder.getI32IntegerAttr(static_cast<int32_t>(idx))),
                builder.getNamedAttr("new_index",
                                     builder.getI32IntegerAttr(
                                         static_cast<int32_t>(dfb.finalIndex)))}));
    }
    if (!mapEntries.empty()) {
      moduleOp->setAttr(kDFBIndexMapAttrName, ArrayAttr::get(ctx, mapEntries));
    }

    // Publish compiler-allocated DFBs, one entry per physical index sized to
    // the member needing the most pages.
    llvm::MapVector<int64_t, LogicalDFB *> compilerByIndex;
    for (auto &[idx, dfb] : dfbs) {
      if (!dfb.compilerAllocated) {
        continue;
      }
      auto [it, inserted] = compilerByIndex.try_emplace(dfb.finalIndex, &dfb);
      if (!inserted && dfb.totalPages() > it->second->totalPages()) {
        it->second = &dfb;
      }
    }
    if (compilerByIndex.empty()) {
      return;
    }

    SmallVector<std::pair<int64_t, LogicalDFB *>> sorted(
        compilerByIndex.begin(), compilerByIndex.end());
    llvm::sort(sorted,
               [](auto &lhs, auto &rhs) { return lhs.first < rhs.first; });

    SmallVector<Attribute> entries;
    for (auto &[dfbIdx, dfb] : sorted) {
      entries.push_back(DictionaryAttr::get(
          ctx,
          {builder.getNamedAttr("dfb_index", builder.getI32IntegerAttr(
                                                 static_cast<int32_t>(dfbIdx))),
           builder.getNamedAttr("num_tiles",
                                builder.getI32IntegerAttr(
                                    static_cast<int32_t>(dfb->elemsPerBlock))),
           builder.getNamedAttr("element_type", TypeAttr::get(dfb->elemType)),
           builder.getNamedAttr("block_count",
                                builder.getI32IntegerAttr(
                                    static_cast<int32_t>(dfb->blockCount)))}));
    }
    moduleOp->setAttr(kCompilerAllocatedDFBsAttrName,
                      ArrayAttr::get(ctx, entries));
  }
};

} // namespace mlir::tt::ttl
