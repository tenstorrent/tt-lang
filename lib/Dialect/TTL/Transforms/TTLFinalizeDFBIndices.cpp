// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Finalize DFB Indices
//===----------------------------------------------------------------------===//
//
// Module-level pass that runs after all DFB-creating passes. Reuses
// compiler-created DFBs within exact CircularBufferType partitions and
// balanced user DFBs local to one kernel thread. Explicit reset boundaries
// additionally compact each first-use epoch into an independent physical
// index space. The pass then computes the true DFB count, updates
// ttl.base_cta_index on every function, and publishes the final index
// assignment for the Python runtime.
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
// User and compiler DFBs never share a physical index within one epoch.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>

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
  StringAttr addressScope;
  bool addressScopeInitialized = false;
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
  int64_t firstUseEpoch = std::numeric_limits<int64_t>::max();
  llvm::SmallDenseSet<int64_t, 4> useEpochs;
  llvm::SmallDenseMap<int64_t, Operation *, 4> epochRepresentative;
  bool pinnedAcrossReset = false;

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

enum class ResetControlFlow { Linear, Cyclic };

constexpr llvm::StringLiteral kDFBResetPreservedIndicesAttrName(
    "ttl.dfb_reset_preserved_indices");

enum class HardwareDataFormat : int64_t {
  Float32 = 0,
  Tf32 = 4,
  Float16B = 5,
  Bfp8B = 6,
  Bfp4B = 7,
  Int32 = 8,
  UInt16 = 9,
  UInt32 = 24,
  UInt8 = 30,
};

HardwareDataFormat l1DataFormat(ttcore::DataType dtype) {
  switch (dtype) {
  case ttcore::DataType::Float32:
    return HardwareDataFormat::Float32;
  case ttcore::DataType::BFloat16:
    return HardwareDataFormat::Float16B;
  case ttcore::DataType::BFP_BFloat8:
    return HardwareDataFormat::Bfp8B;
  case ttcore::DataType::BFP_BFloat4:
    return HardwareDataFormat::Bfp4B;
  case ttcore::DataType::UInt32:
    return HardwareDataFormat::UInt32;
  case ttcore::DataType::UInt16:
    return HardwareDataFormat::UInt16;
  case ttcore::DataType::UInt8:
    return HardwareDataFormat::UInt8;
  case ttcore::DataType::Int32:
    return HardwareDataFormat::Int32;
  default:
    llvm_unreachable("unsupported epoch DFB data type");
  }
}

bool isEpochRuntimeDataTypeSupported(ttcore::DataType dtype) {
  switch (dtype) {
  case ttcore::DataType::BFloat16:
  case ttcore::DataType::Float32:
  case ttcore::DataType::BFP_BFloat8:
  case ttcore::DataType::BFP_BFloat4:
  case ttcore::DataType::UInt32:
  case ttcore::DataType::UInt16:
  case ttcore::DataType::UInt8:
  case ttcore::DataType::Int32:
    return true;
  default:
    return false;
  }
}

bool isBlockFloat(ttcore::DataType dtype) {
  return dtype == ttcore::DataType::BFP_BFloat8 ||
         dtype == ttcore::DataType::BFP_BFloat4;
}

bool isSupportedEpochTileShape(int64_t height, int64_t width) {
  if (width != 16 && width != 32) {
    return false;
  }
  return height == 1 || height == 2 || height == 4 || height == 8 ||
         height == 16 || height == 32;
}

HardwareDataFormat packSourceFormat(ttcore::DataType dtype,
                                    HardwareDataFormat fp32Route,
                                    bool fp32DestAcc) {
  if (dtype == ttcore::DataType::Float32) {
    return fp32DestAcc ? HardwareDataFormat::Float32 : fp32Route;
  }
  if (isBlockFloat(dtype)) {
    return HardwareDataFormat::Bfp8B;
  }
  return l1DataFormat(dtype);
}

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
  Block *shared = a.block == b.block ? a.block : commonBlock(a.block, b.block);
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
    bool invalidResetContract = false;

    llvm::SmallDenseMap<Operation *, SmallVector<OpaqueCallOp, 4>, 4>
        resetCallsByFunction;
    llvm::SmallDenseMap<Operation *, Operation *, 4> cyclicResetLoops;
    moduleOp->walk([&](OpaqueCallOp call) {
      if (call.getCallee() != kResetDataflowBuffersCallee) {
        return;
      }
      func::FuncOp func = getEnclosingKernelThread(call);
      if (!func) {
        call.emitError("must be inside a kernel thread");
        invalidResetContract = true;
        return;
      }
      resetCallsByFunction[func.getOperation()].push_back(call);
    });

    // Linear resets remain entry-block operations. A resident reset sequence
    // is unconditional and has one static order in the outermost loop body.
    std::optional<ResetControlFlow> resetControlFlow;
    moduleOp->walk([&](func::FuncOp func) {
      if (!func->hasAttr(kKernelThreadAttrName)) {
        return;
      }
      auto callsIt = resetCallsByFunction.find(func.getOperation());
      if (callsIt == resetCallsByFunction.end()) {
        return;
      }
      auto &calls = callsIt->second;
      Block *entry = &func.getBody().front();
      const bool topLevel = calls.front()->getBlock() == entry;
      Operation *residentLoop = nullptr;
      if (!topLevel) {
        auto loop = dyn_cast<scf::ForOp>(calls.front()->getParentOp());
        if (loop && loop->getBlock() == entry) {
          residentLoop = loop.getOperation();
        }
      }

      for (OpaqueCallOp call : calls) {
        bool valid = topLevel ? call->getBlock() == entry
                              : call->getParentOp() == residentLoop;
        if (!valid || (!topLevel && !residentLoop)) {
          call.emitError("must be top-level or a direct child of one "
                         "top-level resident scf.for loop");
          invalidResetContract = true;
        }
      }
      if (!topLevel && residentLoop && calls.size() < 2) {
        func.emitError("a cyclic resident loop requires at least two "
                       "reset_dataflow_buffers calls");
        invalidResetContract = true;
      }
      if (!topLevel && residentLoop) {
        cyclicResetLoops[func.getOperation()] = residentLoop;
      }
      ResetControlFlow controlFlow =
          residentLoop ? ResetControlFlow::Cyclic : ResetControlFlow::Linear;
      if (!resetControlFlow) {
        resetControlFlow = controlFlow;
      } else if (*resetControlFlow != controlFlow) {
        func.emitError("must place reset_dataflow_buffers in the same control "
                       "flow shape as every other kernel thread");
        invalidResetContract = true;
      }
    });

    std::optional<size_t> resetCount;
    moduleOp->walk([&](func::FuncOp func) {
      if (!func->hasAttr(kKernelThreadAttrName)) {
        return;
      }
      auto it = resetCallsByFunction.find(func.getOperation());
      size_t count = it == resetCallsByFunction.end() ? 0 : it->second.size();
      if (!resetCount) {
        resetCount = count;
      } else if (*resetCount != count) {
        func.emitError("must call reset_dataflow_buffers the same number of "
                       "times as every other kernel thread");
        invalidResetContract = true;
      }
    });
    if (invalidResetContract) {
      signalPassFailure();
      return;
    }
    const bool hasResetDataflowBuffers = !resetCallsByFunction.empty();
    const bool hasCyclicResetDataflowBuffers = !cyclicResetLoops.empty();

    for (auto &[funcOperation, calls] : resetCallsByFunction) {
      auto func = cast<func::FuncOp>(funcOperation);
      auto &order = orders[funcOperation];
      order = std::make_unique<ThreadOrder>(func);
      Block *scope = &func.getBody().front();
      if (auto loopIt = cyclicResetLoops.find(funcOperation);
          loopIt != cyclicResetLoops.end()) {
        scope = &loopIt->second->getRegion(0).front();
      }
      llvm::sort(calls, [&](OpaqueCallOp lhs, OpaqueCallOp rhs) {
        return order->posIn(scope, lhs) < order->posIn(scope, rhs);
      });
    }

    SmallVector<SmallVector<int64_t>> preservedByResetOrdinal(
        resetCount.value_or(0));
    SmallVector<bool> initializedPreserveOrdinal(resetCount.value_or(0),
                                                 false);
    llvm::DenseSet<int64_t> preservedLogicalIndices;
    for (func::FuncOp func : moduleOp.getOps<func::FuncOp>()) {
      auto callsIt = resetCallsByFunction.find(func.getOperation());
      if (callsIt == resetCallsByFunction.end()) {
        continue;
      }
      auto &calls = callsIt->second;
      for (auto [ordinal, call] : llvm::enumerate(calls)) {
        SmallVector<int64_t> preserved;
        for (Value operand : call.getArgOperands()) {
          if (!isa<CircularBufferType>(operand.getType())) {
            call.emitError(
                "preserve operands must be dataflow buffers");
            invalidResetContract = true;
            continue;
          }
          std::optional<int64_t> index = getCBIndex(operand);
          if (!index) {
            call.emitError(
                "cannot resolve logical index for preserved dataflow buffer");
            invalidResetContract = true;
            continue;
          }
          preserved.push_back(*index);
        }
        llvm::sort(preserved);
        if (std::adjacent_find(preserved.begin(), preserved.end()) !=
            preserved.end()) {
          call.emitError(
              "cannot preserve the same dataflow buffer more than once");
          invalidResetContract = true;
          continue;
        }
        if (!initializedPreserveOrdinal[ordinal]) {
          preservedByResetOrdinal[ordinal] = preserved;
          initializedPreserveOrdinal[ordinal] = true;
        } else if (preservedByResetOrdinal[ordinal] != preserved) {
          call.emitError()
              << "must preserve the same dataflow buffers at reset ordinal "
              << ordinal << " as every other kernel thread";
          invalidResetContract = true;
        }
      }
    }
    if (invalidResetContract) {
      signalPassFailure();
      return;
    }
    for (const auto &preserved : preservedByResetOrdinal) {
      preservedLogicalIndices.insert(preserved.begin(), preserved.end());
    }

    auto firstUseEpoch = [&](func::FuncOp func, Operation *op,
                             ThreadOrder &order) {
      auto callsIt = resetCallsByFunction.find(func.getOperation());
      if (callsIt == resetCallsByFunction.end()) {
        return int64_t{0};
      }
      Block *scope = &func.getBody().front();
      bool cyclic = false;
      if (auto loopIt = cyclicResetLoops.find(func.getOperation());
          loopIt != cyclicResetLoops.end()) {
        scope = &loopIt->second->getRegion(0).front();
        Operation *anchor =
            op->getBlock() == scope ? op : scope->findAncestorOpInBlock(*op);
        if (!anchor) {
          return int64_t{0};
        }
        cyclic = true;
      }
      int64_t opPosition = order.posIn(scope, op);
      int64_t epoch = 0;
      for (OpaqueCallOp call : callsIt->second) {
        if (order.posIn(scope, call) >= opPosition) {
          break;
        }
        ++epoch;
      }
      // The final cyclic reset restores phase zero across the loop backedge.
      return cyclic ? epoch % static_cast<int64_t>(callsIt->second.size())
                    : epoch;
    };

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
        if (hasResetDataflowBuffers && !invalidResetContract) {
          bindOp.emitError()
              << "dataflow buffer " << idx
              << " has a different type in another kernel thread";
          invalidResetContract = true;
        }
      }
      dfb.elemType = cbType.getElementType();
      dfb.blockCount = cbType.getBlockCount();
      dfb.elemsPerBlock =
          std::max(dfb.elemsPerBlock, cbType.getElementsPerBlock());
      auto addressScope = bindOp.getAddressScopeAttr();
      if (!dfb.addressScopeInitialized) {
        dfb.addressScope = addressScope;
        dfb.addressScopeInitialized = true;
      } else if (dfb.addressScope != addressScope) {
        bindOp.emitError() << "dataflow buffer " << idx
                           << " has a different address_scope in another "
                              "kernel thread";
        invalidResetContract = true;
      }
      dfb.compilerAllocated |= bindOp->hasAttr(kCompilerAllocatedAttrName);
      if (!func) {
        dfb.eligible = false;
        return;
      }

      auto &order = orders[func.getOperation()];
      if (!order) {
        order = std::make_unique<ThreadOrder>(func);
      }

      auto recordTouch = [&](Operation *op) {
        extendInterval(dfb, op, func, *order);
        if (hasResetDataflowBuffers) {
          int64_t epoch = firstUseEpoch(func, op, *order);
          dfb.firstUseEpoch = std::min(dfb.firstUseEpoch, epoch);
          dfb.useEpochs.insert(epoch);
          dfb.epochRepresentative.try_emplace(epoch, op);
        }
      };

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
        if (auto call = dyn_cast<OpaqueCallOp>(user);
            call && call.getCallee() == kResetDataflowBuffersCallee) {
          continue;
        }
        recordTouch(user);
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
            recordTouch(reader);
            if (auto attachOp = dyn_cast<AttachCBOp>(reader)) {
              markPipeUsers(attachOp.getResult());
              for (Operation *consumer : attachOp.getResult().getUsers()) {
                recordTouch(consumer);
              }
            }
          }
        }
      }
    });

    for (int64_t index : preservedLogicalIndices) {
      auto it = dfbs.find(index);
      if (it == dfbs.end()) {
        moduleOp.emitError() << "reset_dataflow_buffers preserves unknown "
                                "dataflow buffer "
                             << index;
        invalidResetContract = true;
        continue;
      }
      it->second.pinnedAcrossReset = true;
      it->second.eligible = false;
      it->second.refusal = "preserved-across-reset";
    }

    for (auto &[idx, dfb] : dfbs) {
      SmallVector<int64_t> epochs(dfb.useEpochs.begin(), dfb.useEpochs.end());
      llvm::sort(epochs);
      for (size_t i = 1; i < epochs.size(); ++i) {
        int64_t previous = epochs[i - 1];
        int64_t current = epochs[i];
        for (int64_t ordinal = previous; ordinal < current; ++ordinal) {
          bool preserved =
              ordinal < static_cast<int64_t>(preservedByResetOrdinal.size()) &&
              llvm::is_contained(preservedByResetOrdinal[ordinal], idx);
          if (preserved) {
            continue;
          }
          dfb.epochRepresentative[current]->emitError()
              << "dataflow buffer " << idx
              << " is used across a reset_dataflow_buffers boundary without "
                 "being preserved at reset ordinal "
              << ordinal;
          invalidResetContract = true;
        }
      }
    }

    if (invalidResetContract) {
      signalPassFailure();
      return;
    }

    if (!sawBind && !hasResetDataflowBuffers) {
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
    bool crossThreadReuse =
        std::getenv("TTL_DFB_REUSE_CROSS_THREAD") != nullptr;
    bool externPaired = std::getenv("TTL_DFB_REUSE_EXTERN_PAIRED") != nullptr;
    for (auto &[idx, dfb] : dfbs) {
      if (dfb.firstUseEpoch == std::numeric_limits<int64_t>::max()) {
        dfb.firstUseEpoch = 0;
      }
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
          first =
              std::min(first, orderIt->second->posIn(interval.block, acquire));
        }
        interval.start =
            std::min(std::max(interval.start, first), interval.end);
      }
    }

    llvm::DenseSet<int64_t> unpackToDestFp32DFBs;
    bool fp32DestAcc = false;
    moduleOp->walk([&](func::FuncOp func) {
      auto thread = func->getAttrOfType<ttkernel::ThreadTypeAttr>(
          kKernelThreadAttrName);
      if (!thread || thread.getValue() != ttkernel::ThreadType::Compute) {
        return;
      }
      if (auto attr = func->getAttrOfType<BoolAttr>(kFp32DestAccEnAttrName)) {
        fp32DestAcc |= attr.getValue();
      }
      if (auto attr = func->getAttrOfType<DenseI32ArrayAttr>(
              kUnpackToDestFp32AttrName)) {
        for (int32_t index : attr.asArrayRef()) {
          unpackToDestFp32DFBs.insert(index);
        }
      }
    });

    // Compiler DFBs are partitioned by the complete CircularBufferType. Their
    // lowering may depend on the exact physical ring geometry; widening this
    // to element type caused a silent global-attention regression. Balanced
    // thread-local user DFBs may share different capacities with the same page
    // type because the runtime sizes their arena slot to the largest member.
    // `kind` separates user and compiler storage; the final field separates
    // incompatible FP32 unpack routes.
    using ClassKey =
        std::tuple<Type, Operation *, Operation *, int64_t, int64_t>;
    llvm::MapVector<ClassKey, SmallVector<LogicalDFB *>> classes;
    for (auto &[idx, dfb] : dfbs) {
      if (dfb.eligible) {
        // Cross-thread shares take the page type, like thread-local ones: the
        // runtime sizes the slot to the largest member, and the placement
        // loop below refuses any group whose ring would not wrap on a block
        // boundary for every member. User and compiler slots remain separate
        // within each epoch.
        Type storageType = dfb.compilerAllocated ? dfb.cbType : dfb.elemType;
        int64_t kind = static_cast<int64_t>(dfb.compilerAllocated) +
                       2 * static_cast<int64_t>(dfb.crossThread);
        classes[{storageType, dfb.producer.getOperation(),
                 dfb.consumer.getOperation(), kind,
                 static_cast<int64_t>(
                     unpackToDestFp32DFBs.contains(dfb.origIndex))}]
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
            return member.elemsPerBlock > 0 &&
                   pages % member.elemsPerBlock == 0;
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

    if (hasResetDataflowBuffers) {
      llvm::MapVector<int64_t, SmallVector<LogicalDFB *>> dfbsByEpoch;
      for (auto &[idx, dfb] : dfbs) {
        if (!dfb.pinnedAcrossReset) {
          dfbsByEpoch[dfb.firstUseEpoch].push_back(&dfb);
        }
      }
      int64_t maxEpochLocalSlots = 0;
      for (auto &[epoch, epochDFBs] : dfbsByEpoch) {
        (void)epoch;
        SmallVector<int64_t> usedIndices;
        DenseMap<int64_t, int64_t> bytesByIndex;
        for (LogicalDFB *dfb : epochDFBs) {
          usedIndices.push_back(dfb->finalIndex);
          auto tileType = cast<ttcore::TileType>(dfb->elemType);
          int64_t bytes =
              dfb->totalPages() * static_cast<int64_t>(tileType.getSizeBytes());
          bytesByIndex[dfb->finalIndex] =
              std::max(bytesByIndex[dfb->finalIndex], bytes);
        }
        llvm::sort(usedIndices, [&](int64_t lhs, int64_t rhs) {
          if (bytesByIndex[lhs] != bytesByIndex[rhs]) {
            return bytesByIndex[lhs] > bytesByIndex[rhs];
          }
          return lhs < rhs;
        });
        usedIndices.erase(llvm::unique(usedIndices), usedIndices.end());
        maxEpochLocalSlots = std::max(
            maxEpochLocalSlots, static_cast<int64_t>(usedIndices.size()));
        DenseMap<int64_t, int64_t> rank;
        for (auto [i, index] : llvm::enumerate(usedIndices)) {
          rank[index] = static_cast<int64_t>(i);
        }
        for (LogicalDFB *dfb : epochDFBs) {
          dfb->finalIndex = rank[dfb->finalIndex];
        }
      }
      SmallVector<LogicalDFB *> pinnedDFBs;
      for (auto &[idx, dfb] : dfbs) {
        if (dfb.pinnedAcrossReset) {
          pinnedDFBs.push_back(&dfb);
        }
      }
      llvm::sort(pinnedDFBs, [](LogicalDFB *lhs, LogicalDFB *rhs) {
        return lhs->origIndex < rhs->origIndex;
      });
      for (auto [ordinal, dfb] : llvm::enumerate(pinnedDFBs)) {
        dfb->finalIndex =
            maxEpochLocalSlots + static_cast<int64_t>(ordinal);
      }
    } else {
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
    }

    // Rewrite bind_cb indices in every kernel thread.
    for (auto &[idx, dfb] : dfbs) {
      for (BindCBOp bindOp : dfb.binds) {
        bindOp->setAttr(kDFBLogicalIndexAttrName,
                        builder.getI64IntegerAttr(dfb.origIndex));
      }
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

    SmallVector<Attribute> logicalConfigs;
    logicalConfigs.reserve(dfbs.size());
    for (auto &[idx, dfb] : dfbs) {
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr(
              "logical_index",
              builder.getI32IntegerAttr(static_cast<int32_t>(dfb.origIndex))),
          builder.getNamedAttr(
              "physical_index",
              builder.getI32IntegerAttr(static_cast<int32_t>(dfb.finalIndex))),
          builder.getNamedAttr(
              "epoch", builder.getI32IntegerAttr(
                           static_cast<int32_t>(dfb.firstUseEpoch))),
          builder.getNamedAttr(
              "num_pages", builder.getI32IntegerAttr(
                               static_cast<int32_t>(dfb.totalPages()))),
          builder.getNamedAttr("element_type", TypeAttr::get(dfb.elemType)),
          builder.getNamedAttr(
              "unpack_to_dest_fp32",
              builder.getBoolAttr(
                  unpackToDestFp32DFBs.contains(dfb.origIndex))),
          builder.getNamedAttr("compiler_allocated",
                               builder.getBoolAttr(dfb.compilerAllocated)),
          builder.getNamedAttr(
              "block_count",
              builder.getI32IntegerAttr(static_cast<int32_t>(dfb.blockCount))),
          builder.getNamedAttr(
              "elems_per_block",
              builder.getI32IntegerAttr(
                  static_cast<int32_t>(dfb.elemsPerBlock)))};
      if (dfb.addressScope) {
        attrs.push_back(
            builder.getNamedAttr("address_scope", dfb.addressScope));
      }
      logicalConfigs.push_back(DictionaryAttr::get(ctx, attrs));
    }
    moduleOp->setAttr(kLogicalDFBConfigsAttrName,
                      builder.getArrayAttr(logicalConfigs));

    if (hasResetDataflowBuffers) {
      int64_t epochCount = 1;
      if (hasCyclicResetDataflowBuffers) {
        epochCount =
            static_cast<int64_t>(resetCallsByFunction.begin()->second.size());
      } else {
        for (const auto &[func, calls] : resetCallsByFunction) {
          (void)func;
          epochCount =
              std::max(epochCount, static_cast<int64_t>(calls.size()) + 1);
        }
      }

      SmallVector<llvm::MapVector<int64_t, LogicalDFB *>> configsByEpoch(
          epochCount);
      SmallVector<llvm::DenseSet<int64_t>> unpackSlotsByEpoch(epochCount);
      for (auto &[idx, dfb] : dfbs) {
        if (dfb.firstUseEpoch >= epochCount) {
          configsByEpoch.resize(dfb.firstUseEpoch + 1);
          unpackSlotsByEpoch.resize(dfb.firstUseEpoch + 1);
          epochCount = dfb.firstUseEpoch + 1;
        }
        auto tileType = dyn_cast<ttcore::TileType>(dfb.elemType);
        if (!tileType) {
          moduleOp.emitError()
              << "reset_dataflow_buffers requires tile-typed DFB elements";
          signalPassFailure();
          return;
        }
        if (!isEpochRuntimeDataTypeSupported(tileType.getDataType())) {
          moduleOp.emitError()
              << "reset_dataflow_buffers does not support DFB element type "
              << tileType;
          signalPassFailure();
          return;
        }
        int64_t tileHeight = tileType.getHeight();
        int64_t tileWidth = tileType.getWidth();
        if (!isSupportedEpochTileShape(tileHeight, tileWidth)) {
          moduleOp.emitError()
              << "reset_dataflow_buffers does not support tile shape "
              << tileHeight << "x" << tileWidth;
          signalPassFailure();
          return;
        }
        if (isBlockFloat(tileType.getDataType()) &&
            (tileHeight != 32 || tileWidth != 32)) {
          moduleOp.emitError()
              << "reset_dataflow_buffers requires 32x32 block-float tiles, "
                 "got "
              << tileType;
          signalPassFailure();
          return;
        }

        LogicalDFB *logicalDFB = &dfb;
        auto insertEpochConfig = [&, logicalDFB](int64_t epoch) {
          auto &configs = configsByEpoch[epoch];
          auto [it, inserted] =
              configs.try_emplace(logicalDFB->finalIndex, logicalDFB);
          if (!inserted) {
            auto currentTileType =
                cast<ttcore::TileType>(it->second->elemType);
            int64_t currentBytes =
                it->second->totalPages() * currentTileType.getSizeBytes();
            int64_t candidateBytes =
                logicalDFB->totalPages() * tileType.getSizeBytes();
            if (candidateBytes > currentBytes) {
              it->second = logicalDFB;
            }
          }
          if (unpackToDestFp32DFBs.contains(logicalDFB->origIndex)) {
            unpackSlotsByEpoch[epoch].insert(logicalDFB->finalIndex);
          }
        };
        if (dfb.pinnedAcrossReset) {
          for (int64_t epoch = 0; epoch < epochCount; ++epoch) {
            insertEpochConfig(epoch);
          }
        } else {
          insertEpochConfig(dfb.firstUseEpoch);
        }
      }

      const HardwareDataFormat fp32Route =
          fp32DestAcc ? HardwareDataFormat::Tf32 : HardwareDataFormat::Float16B;

      int64_t physicalSlotCount = 0;
      for (const auto &[idx, dfb] : dfbs) {
        (void)idx;
        physicalSlotCount =
            std::max(physicalSlotCount, dfb.finalIndex + 1);
      }
      SmallVector<Attribute> physicalConfigs;
      for (int64_t physicalIndex = 0; physicalIndex < physicalSlotCount;
           ++physicalIndex) {
        LogicalDFB *initialDFB = nullptr;
        int64_t maxBytes = 0;
        for (const auto &configs : configsByEpoch) {
          auto configIt = configs.find(physicalIndex);
          if (configIt == configs.end()) {
            continue;
          }
          LogicalDFB *dfb = configIt->second;
          auto tileType = cast<ttcore::TileType>(dfb->elemType);
          if (!initialDFB) {
            initialDFB = dfb;
          }
          maxBytes = std::max(
              maxBytes, dfb->totalPages() *
                            static_cast<int64_t>(tileType.getSizeBytes()));
        }
        assert(initialDFB && "epoch-local DFB indices must be dense");
        auto initialTileType = cast<ttcore::TileType>(initialDFB->elemType);
        int64_t initialPageBytes = initialTileType.getSizeBytes();
        int64_t totalSize =
            ((maxBytes + initialPageBytes - 1) / initialPageBytes) *
            initialPageBytes;
        physicalConfigs.push_back(DictionaryAttr::get(
            ctx,
            {builder.getNamedAttr("dfb_index",
                                  builder.getI32IntegerAttr(
                                      static_cast<int32_t>(physicalIndex))),
             builder.getNamedAttr("element_type",
                                  TypeAttr::get(initialDFB->elemType)),
             builder.getNamedAttr(
                 "tile_height", builder.getI32IntegerAttr(static_cast<int32_t>(
                                    initialTileType.getHeight()))),
             builder.getNamedAttr(
                 "tile_width", builder.getI32IntegerAttr(static_cast<int32_t>(
                                   initialTileType.getWidth()))),
             builder.getNamedAttr("total_size",
                                  builder.getI64IntegerAttr(totalSize))}));
      }
      moduleOp->setAttr(kDFBEpochPhysicalConfigsAttrName,
                        builder.getArrayAttr(physicalConfigs));

      auto appendConfiguration = [&](OpaqueCallOp call, int64_t epoch,
                                     ArrayRef<int64_t> preservedIndices) {
        SmallVector<Attribute> templateArgs{builder.getI64IntegerAttr(0)};

        auto &configs = configsByEpoch[epoch];
        SmallVector<int64_t> configuredIndices;
        for (const auto &[physicalIndex, dfb] : configs) {
          (void)dfb;
          if (!llvm::is_contained(preservedIndices, physicalIndex)) {
            configuredIndices.push_back(physicalIndex);
          }
        }
        llvm::sort(configuredIndices);
        templateArgs.push_back(
            builder.getI64IntegerAttr(configuredIndices.size()));
        for (int64_t physicalIndex : configuredIndices) {
          auto configIt = configs.find(physicalIndex);
          assert(configIt != configs.end());
          LogicalDFB *dfb = configIt->second;
          auto tileType = cast<ttcore::TileType>(dfb->elemType);
          int64_t pageBytes = tileType.getSizeBytes();
          int64_t numPages = dfb->totalPages();
          int64_t tileHeight = tileType.getHeight();
          int64_t tileWidth = tileType.getWidth();
          int64_t faceHeight = std::min<int64_t>(tileHeight, 16);
          int64_t faceWidth = std::min<int64_t>(tileWidth, 16);
          int64_t numFaces =
              (tileHeight / faceHeight) * (tileWidth / faceWidth);
          auto dtype = tileType.getDataType();
          HardwareDataFormat l1Format = l1DataFormat(dtype);
          HardwareDataFormat unpackDstFormat = l1Format;
          if (dtype == ttcore::DataType::Float32) {
            unpackDstFormat = unpackSlotsByEpoch[epoch].contains(physicalIndex)
                                  ? HardwareDataFormat::Float32
                                  : fp32Route;
          }
          HardwareDataFormat packSrcFormat =
              packSourceFormat(dtype, fp32Route, fp32DestAcc);
          int64_t values[] = {
              physicalIndex,
              numPages * pageBytes,
              numPages,
              pageBytes,
              static_cast<int64_t>(l1Format),
              tileHeight,
              tileWidth,
              faceHeight,
              numFaces,
              static_cast<int64_t>(unpackDstFormat),
              static_cast<int64_t>(packSrcFormat),
          };
          for (int64_t value : values) {
            templateArgs.push_back(builder.getI64IntegerAttr(value));
          }
        }
        call->setAttr("ttl.dfb_reset_epoch",
                      builder.getI32IntegerAttr(static_cast<int32_t>(epoch)));
        SmallVector<Attribute> preservedAttrs;
        for (int64_t physicalIndex : preservedIndices) {
          preservedAttrs.push_back(builder.getI64IntegerAttr(physicalIndex));
        }
        call->setAttr(kDFBResetPreservedIndicesAttrName,
                      builder.getArrayAttr(preservedAttrs));
        call->setAttr("template_args", builder.getArrayAttr(templateArgs));
      };

      SmallVector<SmallVector<int64_t>> preservedPhysicalByResetOrdinal(
          preservedByResetOrdinal.size());
      for (auto [ordinal, logicalIndices] :
           llvm::enumerate(preservedByResetOrdinal)) {
        auto &physicalIndices = preservedPhysicalByResetOrdinal[ordinal];
        for (int64_t logicalIndex : logicalIndices) {
          physicalIndices.push_back(dfbs.find(logicalIndex)->second.finalIndex);
        }
        llvm::sort(physicalIndices);
      }

      for (auto &[funcOperation, calls] : resetCallsByFunction) {
        if (hasCyclicResetDataflowBuffers) {
          for (auto [ordinal, call] : llvm::enumerate(calls)) {
            appendConfiguration(call, (static_cast<int64_t>(ordinal) + 1) %
                                          epochCount,
                                preservedPhysicalByResetOrdinal[ordinal]);
          }
        } else {
          for (auto [ordinal, call] : llvm::enumerate(calls)) {
            appendConfiguration(call, static_cast<int64_t>(ordinal) + 1,
                                preservedPhysicalByResetOrdinal[ordinal]);
          }
        }

        auto func = cast<func::FuncOp>(funcOperation);
        auto prologue = cast<OpaqueCallOp>(calls.front()->clone());
        prologue->setOperands(ValueRange{});
        appendConfiguration(prologue, 0, {});
        Block &entry = func.getBody().front();
        auto insertionPoint = entry.begin();
        while (insertionPoint != entry.end() &&
               isa<BindCBOp>(&*insertionPoint)) {
          ++insertionPoint;
        }
        entry.getOperations().insert(insertionPoint, prologue);
      }
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
        auto [storage, prod, cons, kind, unpackToDestFp32] = key;
        (void)unpackToDestFp32;
        llvm::DenseSet<int64_t> placed;
        for (LogicalDFB *dfb : members) {
          placed.insert(dfb->finalIndex);
        }
        llvm::errs() << "  class kind=" << kind
                     << " prod=" << cast<func::FuncOp>(prod).getName()
                     << " cons=" << cast<func::FuncOp>(cons).getName()
                     << " members=" << members.size()
                     << " slots=" << placed.size() << " type=" << storage
                     << "\n";
        for (LogicalDFB *dfb : members) {
          llvm::errs() << "    cb" << dfb->origIndex << " -> "
                       << dfb->finalIndex << " pages=" << dfb->totalPages();
          for (auto &[func, interval] : dfb->intervals) {
            llvm::errs() << " [" << cast<func::FuncOp>(func).getName() << " blk"
                         << (const void *)interval.block << " "
                         << interval.start << ".." << interval.end << "]";
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
