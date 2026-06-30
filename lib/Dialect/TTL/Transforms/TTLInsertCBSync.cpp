// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert CB Sync
//===----------------------------------------------------------------------===//
//
// Auto-inserts a cb_push / cb_pop after each cb_reserve / cb_wait whose
// matching release is absent in the input IR, placing each release after
// the last use of the acquired slot so the slot is not recycled before
// the consumer is done with it. "Last use" classification handles two
// different valid IR situations -- direct-CB uses and tensor-SSA uses --
// under different rules; see `docs/development/DFBManagement.md` for the
// rules and correctness argument.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "DFBAcquireReleaseAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-cb-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTCBSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static bool isCompilerAllocatedDFB(Value cb) {
  auto bind = cb.getDefiningOp<BindCBOp>();
  return bind && bind->hasAttr(kCompilerAllocatedAttrName);
}

static bool isTensorLike(Value value) {
  Type type = value.getType();
  return isa<RankedTensorType, ttcore::TileType>(type);
}

static bool valueDependsOnDFBWait(Value value, Value dfb) {
  DenseSet<Value> visited;
  SmallVector<Value, 8> worklist{value};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second) {
      continue;
    }

    Operation *definingOp = current.getDefiningOp();
    if (!definingOp) {
      continue;
    }

    if (auto wait = dyn_cast<CBWaitOp>(definingOp);
        wait && wait.getCb() == dfb) {
      return true;
    }

    for (Value operand : definingOp->getOperands()) {
      if (isTensorLike(operand)) {
        worklist.push_back(operand);
      }
    }
  }

  return false;
}

static bool hasProjectedUseAfter(Operation *barrier, Value value) {
  Block *barrierBlock = barrier->getBlock();
  DenseSet<Value> visitedValues;
  DenseSet<Operation *> visitedUsers;
  SmallVector<Value, 8> worklist{value};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visitedValues.insert(current).second) {
      continue;
    }

    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (!visitedUsers.insert(user).second) {
        continue;
      }
      if (user == barrier) {
        continue;
      }

      Operation *projected = user->getBlock() == barrierBlock
                                 ? user
                                 : barrierBlock->findAncestorOpInBlock(*user);
      if (projected && barrier->isBeforeInBlock(projected)) {
        return true;
      }

      for (Value result : user->getResults()) {
        if (isTensorLike(result)) {
          worklist.push_back(result);
        }
      }
    }
  }

  return false;
}

static bool hasSameDFBDependentValueLiveAfter(StoreOp store, Value dfb) {
  Operation *barrier = store.getOperation();
  Block *block = barrier->getBlock();

  for (Operation &operation : *block) {
    if (&operation == barrier) {
      break;
    }

    for (Value result : operation.getResults()) {
      if (!isTensorLike(result)) {
        continue;
      }
      if (valueDependsOnDFBWait(result, dfb) &&
          hasProjectedUseAfter(barrier, result)) {
        return true;
      }
    }
  }

  Value storedTensor = store.getTensor();
  return valueDependsOnDFBWait(storedTensor, dfb) &&
         hasProjectedUseAfter(barrier, storedTensor);
}

static LogicalResult
rejectUnsupportedUserDFBStateSSACrossings(func::FuncOp func) {
  WalkResult result =
      func.walk([&](StoreOp store) {
        auto reserve = findCBReserveForView(store.getView());
        if (!reserve) {
          return WalkResult::advance();
        }

        Value dfb = reserve.getCb();
        if (isCompilerAllocatedDFB(dfb)) {
          return WalkResult::advance();
        }

        if (!hasSameDFBDependentValueLiveAfter(store, dfb)) {
          return WalkResult::advance();
        }

        store.emitOpError()
            << "unsupported same-DFB tensor SSA state update: a tensor derived "
               "from the same user-declared dataflow buffer remains live after "
               "the store; materialize the value through a separate dataflow "
               "buffer before updating this state buffer";
        return WalkResult::interrupt();
      });

  return failure(result.wasInterrupted());
}

template <typename CreateReleaseFn>
static void insertMissingReleases(ArrayRef<Operation *> acquires,
                                  ArrayRef<Operation *> releases,
                                  DenseSet<Operation *> &erased,
                                  OpBuilder &builder,
                                  CreateReleaseFn createRelease) {
  for (Operation *acquire : acquires) {
    DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);
    // Cheap check first: any release inside the strict next-acquire range?
    DFBReleaseSearch releaseSearch = findOwnedDFBReleases(
        interval, /*lastOwnedUse=*/nullptr, releases, &erased);
    if (releaseSearch.hasSameLevelRelease()) {
      continue;
    }

    // Compute the last owned use; it both bounds the idempotency recheck
    // and pinpoints the insertion point.
    Operation *last = findLastDFBAcquireOwnedUse(interval);
    if (last != interval.acquire) {
      releaseSearch = findOwnedDFBReleases(interval, last, releases, &erased);
      if (releaseSearch.hasSameLevelRelease()) {
        continue;
      }
    }

    for (Operation *nestedRelease : releaseSearch.nestedReleases) {
      erased.insert(nestedRelease);
      nestedRelease->erase();
    }

    builder.setInsertionPointAfter(last);
    createRelease(builder, acquire->getLoc(), interval.dfb);
  }
}

struct TTLInsertCBSyncPass
    : public impl::TTLInsertCBSyncBase<TTLInsertCBSyncPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (failed(rejectUnsupportedUserDFBStateSSACrossings(func))) {
      signalPassFailure();
      return;
    }

    SmallVector<Operation *> reserves;
    SmallVector<Operation *> waits;
    SmallVector<Operation *> pushes;
    SmallVector<Operation *> pops;

    collectDFBAcquireReleaseOps(func, reserves, waits, pushes, pops);

    OpBuilder builder(func.getContext());

    // Track erased ops so later iterations skip them before any accessor
    // call. The set holds raw pointers to freed ops; release ownership search
    // must check the set before touching any op wrapper method.
    DenseSet<Operation *> erased;

    insertMissingReleases(reserves, pushes, erased, builder,
                          [](OpBuilder &b, Location loc, Value cb) {
                            CBPushOp::create(b, loc, cb,
                                             /*num_tiles=*/IntegerAttr{});
                          });

    insertMissingReleases(waits, pops, erased, builder,
                          [](OpBuilder &b, Location loc, Value cb) {
                            CBPopOp::create(b, loc, cb,
                                            /*num_tiles=*/IntegerAttr{});
                          });
  }
};

} // namespace

} // namespace mlir::tt::ttl
