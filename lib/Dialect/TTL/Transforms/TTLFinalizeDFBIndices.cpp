// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Finalize DFB Indices
//===----------------------------------------------------------------------===//
//
// Module-level pass that runs after all DFB-creating passes. Reuses
// compiler-created DFBs within exact CircularBufferType partitions and
// balanced user DFBs local to one kernel thread. It then computes the true
// DFB count, updates ttl.base_cta_index on every function, and publishes the
// final index assignment for the Python runtime.
//
// A logical DFB is one cb_index, bound by one bind_cb per kernel thread that
// touches it. The same physical CB index has shared pages_received /
// pages_acked counters and per-RISC read/write pointers, so two logical DFBs
// may share an index only when:
//   - both have the same producer thread and the same consumer thread;
//     user DFBs additionally require producer == consumer,
//   - their per-thread lifetimes are disjoint in both threads' program order,
//   - compiler DFBs have the same complete CB type; user DFBs have the same
//     page type and use a slot sized to their largest member.
// User and compiler DFBs never share a physical index.
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
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#include <cstdlib>

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFINALIZEDFBINDICES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Closed [start, end] lifetime of a logical DFB within one kernel thread,
/// expressed in the program order of the innermost block that contains every
/// one of its touches. A buffer used only in the early half of a loop body is
/// dead in the late half of every iteration, so measuring it against the whole
/// loop -- the only thing top-level order can say -- throws away most of the
/// reuse in a program whose work is inside loops. Balance is what makes the
/// finer measurement legal: a buffer that ends each iteration drained is dead
/// across the back-edge too.
struct ThreadInterval {
  Block *block = nullptr;
  int64_t start = 0;
  int64_t end = 0;
  /// Any op of this lifetime, kept so the interval can be re-expressed in an
  /// enclosing block when a later touch turns out to live outside `block`.
  Operation *rep = nullptr;
};

/// One logical DFB: a cb_index with one bind_cb per kernel thread.
struct LogicalDFB {
  int64_t origIndex = -1;
  SmallVector<BindCBOp, 3> binds;
  func::FuncOp producer;
  func::FuncOp consumer;
  llvm::SmallDenseMap<Operation *, ThreadInterval, 4> intervals;
  llvm::SmallDenseMap<Operation *, SmallVector<Operation *, 4>, 4> acquires;
  Type cbType;
  Type elemType;
  int64_t blockCount = 0;
  int64_t elemsPerBlock = 0;
  bool compilerAllocated = false;
  bool eligible = true;
  bool crossThread = false;
  bool externUse = false;
  /// Kernel threads that call an extern on this DFB. The call's enclosing
  /// func.func carries ttl.kernel_thread, so the side of the handoff that
  /// happens inside foreign code still has a known thread even when no
  /// TT-Lang flow control names it.
  SmallVector<func::FuncOp, 2> externThreads;
  StringRef refusal;
  int64_t finalIndex = -1;

  /// Per-block tallies of the four flow-control operations. A block that
  /// waits more often than it pops, or reserves more often than it pushes,
  /// ends with pages still held, and a later occupant of the same physical
  /// index would inherit a ring that is neither empty nor aligned.
  struct BlockTally {
    int64_t reserves = 0;
    int64_t pushes = 0;
    int64_t waits = 0;
    int64_t pops = 0;
  };
  llvm::SmallDenseMap<Block *, BlockTally, 4> tallies;

  bool balanced() const {
    return llvm::all_of(tallies, [](const auto &entry) {
      return entry.second.reserves == entry.second.pushes &&
             entry.second.waits == entry.second.pops;
    });
  }

  int64_t totalPages() const { return blockCount * elemsPerBlock; }
};

/// Program order of ops within any block of one kernel thread, computed on
/// demand. `posIn(block, op)` is the position in `block` of `op` itself or of
/// the ancestor of `op` that sits in `block`.
struct ThreadOrder {
  DenseMap<Block *, DenseMap<Operation *, int64_t>> positions;

  explicit ThreadOrder(func::FuncOp func) { (void)func; }

  const DenseMap<Operation *, int64_t> &orderOf(Block *block) {
    auto it = positions.find(block);
    if (it != positions.end()) {
      return it->second;
    }
    DenseMap<Operation *, int64_t> map;
    int64_t idx = 0;
    for (Operation &op : *block) {
      map[&op] = idx++;
    }
    return positions.insert({block, std::move(map)}).first->second;
  }

  int64_t posIn(Block *block, Operation *op) {
    Operation *anchor =
        op->getBlock() == block ? op : block->findAncestorOpInBlock(*op);
    assert(anchor && "operation must be reachable from the block");
    return orderOf(block).lookup(anchor);
  }
};

/// The innermost block containing both `a` and `b`, or null when they belong
/// to different regions entirely.
Block *commonBlock(Block *a, Block *b) {
  llvm::SmallPtrSet<Block *, 8> ancestors;
  for (Block *block = a; block;) {
    ancestors.insert(block);
    Operation *parent = block->getParentOp();
    block = parent ? parent->getBlock() : nullptr;
  }
  for (Block *block = b; block;) {
    if (ancestors.contains(block)) {
      return block;
    }
    Operation *parent = block->getParentOp();
    block = parent ? parent->getBlock() : nullptr;
  }
  return nullptr;
}

/// Extend the per-thread interval of `dfb` to cover `op`.
void extendInterval(LogicalDFB &dfb, Operation *op, func::FuncOp func,
                    ThreadOrder &order) {
  auto [it, inserted] = dfb.intervals.try_emplace(func.getOperation());
  ThreadInterval &interval = it->second;
  if (inserted) {
    interval.block = op->getBlock();
    interval.rep = op;
    interval.start = interval.end = order.posIn(interval.block, op);
    return;
  }
  Block *shared = commonBlock(interval.block, op->getBlock());
  if (!shared) {
    // Nothing sensible to say about order across unrelated regions.
    dfb.eligible = false;
    return;
  }
  if (shared != interval.block) {
    // The lifetime now reaches outside the block it was measured in. Every op
    // of the old, deeper span projects onto the one ancestor that stands for
    // it here, so the whole span collapses to that single position.
    int64_t projected = order.posIn(shared, interval.rep);
    interval.block = shared;
    interval.start = interval.end = projected;
  }
  int64_t pos = order.posIn(interval.block, op);
  interval.start = std::min(interval.start, pos);
  interval.end = std::max(interval.end, pos);
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

bool disjoint(ThreadOrder &order, const ThreadInterval &a,
              const ThreadInterval &b) {
  Block *shared =
      a.block == b.block ? a.block : commonBlock(a.block, b.block);
  if (!shared) {
    return false;
  }
  auto project = [&](const ThreadInterval &interval) {
    if (interval.block == shared) {
      return std::make_pair(interval.start, interval.end);
    }
    int64_t pos = order.posIn(shared, interval.rep);
    return std::make_pair(pos, pos);
  };
  auto [aStart, aEnd] = project(a);
  auto [bStart, bEnd] = project(b);
  return aEnd < bStart || bEnd < aStart;
}

/// Two logical DFBs of one reuse class conflict when their lifetimes overlap
/// in the producer thread or in the consumer thread. Intervals are closed:
/// sharing an endpoint means both are live at that op.
bool conflicts(
    const LogicalDFB &a, const LogicalDFB &b,
    llvm::SmallDenseMap<Operation *, std::unique_ptr<ThreadOrder>> &orders) {
  for (auto &[func, interval] : a.intervals) {
    auto it = b.intervals.find(func);
    if (it == b.intervals.end()) {
      continue;
    }
    auto orderIt = orders.find(func);
    if (orderIt == orders.end() ||
        !disjoint(*orderIt->second, interval, it->second)) {
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

    bool externReuse = std::getenv("TTL_DFB_REUSE_EXTERN") != nullptr;

    moduleOp->walk([&](BindCBOp bindOp) {
      sawBind = true;
      func::FuncOp func = getEnclosingKernelThread(bindOp);
      int64_t idx = bindOp.getCbIndex().getSExtValue();
      LogicalDFB &dfb = dfbs[idx];
      dfb.origIndex = idx;
      dfb.binds.push_back(bindOp);
      auto cbType =
          mlir::cast<CircularBufferType>(bindOp.getResult().getType());
      if (!dfb.cbType) {
        dfb.cbType = cbType;
      } else if (dfb.cbType != cbType) {
        // One logical DFB must have one physical ring geometry on every
        // thread that binds it.
        dfb.eligible = false;
      }
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

      // Pipe destinations are slot-addressed (block_count == sender count)
      // and pipe sends read asynchronously, so pipe-attached DFBs keep
      // dedicated indices.
      // A DFB participates in a pipe through a ttl.copy whose other operand
      // is a pipe, or through pipe_recv guards.
      auto isPipeOp = [](Operation *op) {
        if (op->getName().getStringRef().contains("pipe")) {
          return true;
        }
        return llvm::any_of(op->getOperands(), [](Value operand) {
          return mlir::isa<PipeType>(operand.getType());
        });
      };
      // Pipe sends/receives reach a block through view chains (extract_slice,
      // attach_cb, casts); walk a few hops of users.
      auto markPipeUsers = [&](Value view) {
        SmallVector<Value> worklist{view};
        for (int depth = 0; depth < 3 && !worklist.empty(); ++depth) {
          SmallVector<Value> next;
          for (Value v : worklist) {
            for (Operation *user : v.getUsers()) {
              if (isPipeOp(user)) {
                dfb.eligible = false;
                return;
              }
              for (Value result : user->getResults()) {
                next.push_back(result);
              }
            }
          }
          worklist = std::move(next);
        }
      };

      for (Operation *user : bindOp.getResult().getUsers()) {
        extendInterval(dfb, user, func, *order);
        // A direct DFB operand to foreign code may reserve, wait, push, pop,
        // or retain pages, and OpaqueCallOp carries no memory effects or
        // producer/consumer role metadata of its own. It is still not opaque
        // in the way that matters here. The call sits inside a func.func whose
        // ttl.kernel_thread names the thread it runs on -- the same attribute
        // every role is derived from -- and the reserve/push and wait/pop
        // pairs that bracket it in that thread still tally. Under
        // TTL_DFB_REUSE_EXTERN the call is therefore recorded rather than
        // treated as a veto: the eligibility loop below admits it only when
        // TT-Lang flow control on both sides is present and balanced, which
        // pins the ring pointers regardless of what the callee did to the
        // pages in between. An extern that advances the pointers itself
        // breaks that tally and is refused, and an extern on a DFB with no
        // TT-Lang flow control at all has nothing to prove balance with and
        // is refused too.
        if (isa<OpaqueCallOp>(user)) {
          dfb.externUse = true;
          if (func && !llvm::is_contained(dfb.externThreads, func)) {
            dfb.externThreads.push_back(func);
          }
          if (!externReuse) {
            dfb.eligible = false;
            dfb.refusal = "opaque-call";
          }
        }
        if (isPipeOp(user)) {
          dfb.eligible = false;
          dfb.refusal = "pipe";
        }
        if (auto reserveOp = dyn_cast<CBReserveOp>(user)) {
          markPipeUsers(reserveOp.getResult());
        }
        if (isa<CBReserveOp, CBWaitOp>(user)) {
          dfb.acquires[func.getOperation()].push_back(user);
        }
        auto &tally = dfb.tallies[user->getBlock()];
        if (isa<CBReserveOp>(user)) {
          ++tally.reserves;
        }
        if (isa<CBPushOp>(user)) {
          ++tally.pushes;
        }
        if (isa<CBWaitOp>(user)) {
          ++tally.waits;
        }
        if (isa<CBPopOp>(user)) {
          ++tally.pops;
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
          markPipeUsers(waitOp.getResult());
          for (Operation *reader : waitOp.getResult().getUsers()) {
            extendInterval(dfb, reader, func, *order);
            if (auto attachOp = dyn_cast<AttachCBOp>(reader)) {
              markPipeUsers(attachOp.getResult());
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

    // User DFBs enter the arena when one kernel thread both produces and
    // consumes them, and -- under TTL_DFB_REUSE_CROSS_THREAD -- when a
    // producer thread and a distinct consumer thread hand pages between them
    // in a balanced way.
    //
    // The conservative reading is that static operation order cannot prove
    // that independently executing RISCs have drained a synchronization
    // channel. What it can prove is the weaker property that actually
    // matters. Two logical DFBs sharing an index have, by the class key, the
    // same producer thread and the same consumer thread, and here also the
    // same complete ring geometry. If the earlier one is balanced -- every
    // page it reserves is pushed and every page it waits on is popped in the
    // block that acquired it -- then the producer's write pointer and the
    // consumer's read pointer each advance by the same number of pages, so
    // both reach the successor at the same ring offset, and the shared
    // received/acked counters settle equal. Skew between the two RISCs is
    // then harmless rather than unproven: the successor's reserve blocks
    // until the predecessor's pages are acked, and the FIFO hands the
    // consumer the predecessor's remaining pages before any of the
    // successor's. What is genuinely unsound is an unbalanced buffer, which
    // leaves a page held, and that is exactly what `balanced()` refuses.
    bool crossThreadReuse = std::getenv("TTL_DFB_REUSE_CROSS_THREAD") != nullptr;
    bool externPaired = std::getenv("TTL_DFB_REUSE_EXTERN_PAIRED") != nullptr;
    for (auto &[idx, dfb] : dfbs) {
      // An extern that carries one side -- or both sides -- of a handoff
      // leaves that role unnamed in the IR, but not unknown: the call sits in
      // a func.func whose ttl.kernel_thread says which RISC it runs on. Fill
      // the missing role from there, so the buffer joins a class keyed by the
      // threads that really touch it.
      //
      // The soundness argument is the balance argument one step out. An
      // ttl.opaque_call is synchronous within its thread, so two calls in one
      // thread cannot overlap, and the interval machinery below already keeps
      // their lifetimes disjoint. What the compiler cannot verify is that the
      // callee pushes as many pages as it pops. It does not have to: an extern
      // that left the ring misaligned would desynchronize its own next
      // invocation, so self-consistency across iterations is a property the
      // callee already must have. Whatever TT-Lang flow control does exist is
      // still checked with `balanced()`, and the successor's reserve blocks
      // until the predecessor's pages are acked exactly as before.
      if (externPaired && dfb.externUse && !dfb.compilerAllocated &&
          !dfb.externThreads.empty() && dfb.balanced()) {
        if (!dfb.producer) {
          dfb.producer = dfb.externThreads.front();
        }
        if (!dfb.consumer) {
          dfb.consumer = dfb.externThreads.size() > 1
                             ? dfb.externThreads[1]
                             : dfb.externThreads.front();
        }
        if (dfb.producer != dfb.consumer) {
          dfb.crossThread = true;
        }
        dfb.externUse = false;
      }
      if (!dfb.producer && !dfb.consumer) {
        dfb.eligible = false;
        dfb.refusal = "no-role";
        continue;
      }
      if (dfb.externUse && !dfb.compilerAllocated) {
        // The extern's own accesses are invisible, so the only evidence that
        // the ring is left empty and aligned is the TT-Lang flow control
        // around it. Demand the full pair on both sides and a clean tally;
        // an empty tally would satisfy `balanced()` vacuously and prove
        // nothing, so require that the buffer was actually driven from IR.
        if (dfb.tallies.empty() || !dfb.producer || !dfb.consumer ||
            !dfb.balanced()) {
          dfb.eligible = false;
          if (dfb.refusal.empty()) {
            dfb.refusal = "extern-unproven";
          }
          continue;
        }
      }
      if (!dfb.compilerAllocated &&
          (!dfb.producer || !dfb.consumer || dfb.producer != dfb.consumer)) {
        if (crossThreadReuse && dfb.producer && dfb.consumer &&
            dfb.balanced()) {
          dfb.crossThread = true;
        } else {
          dfb.eligible = false;
          if (dfb.refusal.empty()) {
            dfb.refusal = !dfb.producer     ? "no-producer"
                          : !dfb.consumer   ? "no-consumer"
                          : !dfb.balanced() ? "unbalanced"
                                            : "cross-thread";
          }
        }
      }
      if (!dfb.producer) {
        dfb.producer = dfb.consumer;
      }
      if (!dfb.consumer) {
        dfb.consumer = dfb.producer;
      }
      for (auto &[func, interval] : dfb.intervals) {
        auto it = dfb.acquires.find(func);
        auto orderIt = orders.find(func);
        if (it == dfb.acquires.end() || orderIt == orders.end()) {
          continue;
        }
        // The bind is hoisted to the function entry, so the buffer is not
        // really live from there. A bind's use-list is not program ordered,
        // and acquires may sit in different nested blocks, so project every
        // acquire into the final interval block and use the earliest one.
        int64_t first = interval.end;
        for (Operation *acquire : it->second) {
          first = std::min(
              first, orderIt->second->posIn(interval.block, acquire));
        }
        interval.start =
            std::min(std::max(interval.start, first), interval.end);
      }
    }

    // Compiler DFBs are partitioned by the complete CircularBufferType. Their
    // lowering may depend on the exact physical ring geometry; widening this
    // to element type caused a silent global-attention regression. Balanced
    // thread-local user DFBs may share different capacities with the same page
    // type because the runtime sizes their arena slot to the largest member.
    // The final boolean keeps user and compiler storage strictly separate.
    using ClassKey = std::tuple<Type, Operation *, Operation *, int64_t>;
    llvm::MapVector<ClassKey, SmallVector<LogicalDFB *>> classes;
    for (auto &[idx, dfb] : dfbs) {
      if (dfb.eligible) {
        // Cross-thread shares take the page type, like thread-local ones: the
        // runtime sizes the slot to the largest member, and the placement
        // loop below refuses any group whose ring would not wrap on a block
        // boundary for every member.
        Type storageType = dfb.compilerAllocated ? dfb.cbType : dfb.elemType;
        int64_t kind = static_cast<int64_t>(dfb.compilerAllocated) +
                       2 * static_cast<int64_t>(dfb.crossThread);
        classes[{storageType, dfb.producer.getOperation(),
                 dfb.consumer.getOperation(), kind}]
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
          if (llvm::any_of(slots[s], [&](LogicalDFB *member) {
                return conflicts(*member, *dfb, orders);
              })) {
            continue;
          }
          // The physical ring is as long as the slot's largest member, and a
          // block may not straddle the wrap: every member's block size has to
          // divide that length, or a push after the wrap would land on a
          // boundary its own arithmetic does not expect.
          int64_t pages = dfb->totalPages();
          for (LogicalDFB *member : slots[s]) {
            pages = std::max(pages, member->totalPages());
          }
          auto wraps = [&](const LogicalDFB &member) {
            return member.elemsPerBlock > 0 && pages % member.elemsPerBlock == 0;
          };
          if (!wraps(*dfb) || !llvm::all_of(slots[s], [&](LogicalDFB *member) {
                return wraps(*member);
              })) {
            continue;
          }
          placed = s;
          break;
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

    if (std::getenv("TTL_DFB_REUSE_DEBUG")) {
      int64_t user = 0, userEligible = 0, cross = 0, unbalanced = 0,
              oneSided = 0;
      for (auto &[idx, dfb] : dfbs) {
        if (dfb.compilerAllocated) {
          continue;
        }
        ++user;
        if (dfb.eligible) {
          ++userEligible;
        }
        if (dfb.crossThread) {
          ++cross;
        }
        if (!dfb.balanced()) {
          ++unbalanced;
        }
        if (!dfb.producer || !dfb.consumer) {
          ++oneSided;
        }
      }
      llvm::errs() << "dfb-reuse: " << numDFBs << " indices; user " << user
                   << ", eligible " << userEligible << ", cross-thread "
                   << cross << ", unbalanced " << unbalanced << ", one-sided "
                   << oneSided << "\n";
      llvm::StringMap<int64_t> refusals;
      for (auto &[idx, dfb] : dfbs) {
        if (!dfb.compilerAllocated && !dfb.eligible) {
          ++refusals[dfb.refusal.empty() ? "other" : dfb.refusal];
          // The original index is what the logical-DFB table names, so this
          // line is what ties a refusal back to the op that declared it.
          llvm::errs() << "  refuse cb" << dfb.origIndex << " "
                       << (dfb.refusal.empty() ? "other" : dfb.refusal)
                       << " pages=" << dfb.totalPages()
                       << " producer=" << (dfb.producer ? "y" : "n")
                       << " consumer=" << (dfb.consumer ? "y" : "n")
                       << " extern=" << (dfb.externUse ? "y" : "n") << "\n";
        }
      }
      for (auto &entry : refusals) {
        llvm::errs() << "  refused " << entry.getKey() << ": "
                     << entry.getValue() << "\n";
      }
      for (auto &[key, members] : classes) {
        auto [storage, prod, cons, kind] = key;
        llvm::DenseSet<int64_t> placed;
        for (LogicalDFB *dfb : members) {
          placed.insert(dfb->finalIndex);
        }
        llvm::errs() << "  class kind=" << kind << " prod="
                     << cast<func::FuncOp>(prod).getName() << " cons="
                     << cast<func::FuncOp>(cons).getName() << " members="
                     << members.size() << " slots=" << placed.size()
                     << " type=" << storage << "\n";
        for (LogicalDFB *dfb : members) {
          llvm::errs() << "    cb" << dfb->origIndex << " -> " << dfb->finalIndex
                       << " pages=" << dfb->totalPages();
          for (auto &[func, interval] : dfb->intervals) {
            llvm::errs() << " [" << cast<func::FuncOp>(func).getName() << " blk"
                         << (const void *)interval.block << " " << interval.start
                         << ".." << interval.end << "]";
          }
          llvm::errs() << "\n";
        }
      }
    }

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
          ctx,
          {builder.getNamedAttr("old_index", builder.getI32IntegerAttr(
                                                 static_cast<int32_t>(idx))),
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
