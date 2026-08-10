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

#include "DFBAcquireReleaseAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-cb-sync"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTCBSYNC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct WaitAnyCandidate {
  WaitAnyOp waitAny;
  unsigned candidateIndex = 0;
};

struct ConditionalReceiveReleasePlan {
  DenseSet<Operation *> reserves;
  DenseMap<Operation *, SmallVector<WaitAnyCandidate>> candidatesByReserve;
  DenseMap<Operation *, SmallVector<WaitOp>> exactWaitsByReserve;
};

template <typename CreateReleaseFn>
static void insertMissingReleases(
    ArrayRef<Operation *> acquires, ArrayRef<Operation *> releases,
    const DenseSet<Operation *> &acquisitionsRequiringExplicitRelease,
    DenseSet<Operation *> &erased, OpBuilder &builder,
    CreateReleaseFn createRelease) {
  for (Operation *acquire : acquires) {
    DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);

    // Tensor SSA uses can keep this acquired slot live past the next same-DFB
    // acquire. An existing release after that final use still belongs to this
    // acquire, so pass the final use into the release search.
    Operation *last = findLastDFBAcquireOwnedUse(interval);
    DFBReleaseSearch releaseSearch =
        findOwnedDFBReleases(interval, last, releases, &erased);

    if (acquisitionsRequiringExplicitRelease.contains(acquire)) {
      continue;
    }

    if (releaseSearch.hasSameLevelRelease()) {
      continue;
    }

    for (Operation *nestedRelease : releaseSearch.nestedReleases) {
      erased.insert(nestedRelease);
      nestedRelease->erase();
    }

    builder.setInsertionPointAfter(last);
    createRelease(builder, acquire->getLoc(), interval.dfb);
  }
}

static FailureOr<ConditionalReceiveReleasePlan>
buildConditionalReceiveReleasePlan(func::FuncOp func,
                                   ValueOriginAnalysis &valueOrigins) {
  ConditionalReceiveReleasePlan plan;
  WalkResult result = func.walk([&](WaitAnyOp waitAny) {
    for (auto [candidateIndex, request] :
         llvm::enumerate(waitAny.getRequests())) {
      FailureOr<SmallVector<CopyOp>> receiveCopies =
          findPipeReceiveCopies(valueOrigins, request);
      if (failed(receiveCopies)) {
        waitAny.emitOpError()
            << "requires every request origin to be a pipe receive ttl.copy";
        return WalkResult::interrupt();
      }
      for (CopyOp receiveCopy : *receiveCopies) {
        CBReserveOp reserve = findCBReserveForPipeReceive(receiveCopy.getDst());
        if (!reserve) {
          receiveCopy.emitOpError()
              << "requires a cb_reserve destination for wait-any";
          return WalkResult::interrupt();
        }
        Operation *reserveOperation = reserve.getOperation();
        plan.reserves.insert(reserveOperation);
        plan.candidatesByReserve[reserveOperation].push_back(
            WaitAnyCandidate{waitAny, static_cast<unsigned>(candidateIndex)});
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }

  WalkResult exactWaitResult =
      func.walk([&](WaitOp wait) {
        if (!isa<ReceiveRequestType>(wait.getXf().getType())) {
          return WalkResult::advance();
        }
        FailureOr<SmallVector<CopyOp>> receiveCopies =
            findPipeReceiveCopies(valueOrigins, wait.getXf());
        if (failed(receiveCopies)) {
          wait.emitOpError()
              << "requires every request origin to be a pipe receive ttl.copy";
          return WalkResult::interrupt();
        }
        for (CopyOp receiveCopy : *receiveCopies) {
          if (CBReserveOp reserve =
                  findCBReserveForPipeReceive(receiveCopy.getDst())) {
            plan.exactWaitsByReserve[reserve.getOperation()].push_back(wait);
          }
        }
        return WalkResult::advance();
      });
  if (exactWaitResult.wasInterrupted()) {
    return failure();
  }
  return plan;
}

static bool executesBefore(Operation *before, Operation *after) {
  if (before->getBlock() == after->getBlock()) {
    return before->isBeforeInBlock(after);
  }
  Operation *afterAncestor = before->getBlock()->findAncestorOpInBlock(*after);
  return afterAncestor && before->isBeforeInBlock(afterAncestor);
}

static bool isSelectedCandidateRelease(Operation *release,
                                       WaitAnyCandidate candidate) {
  Operation *current = release;
  while (Block *block = current->getBlock()) {
    auto ifOp = dyn_cast_or_null<scf::IfOp>(block->getParentOp());
    if (ifOp) {
      std::optional<ReadyReceiveSelection> selection =
          getReadyReceiveSelection(ifOp.getCondition());
      bool inSelectedRegion =
          selection && ((selection->selectedWhenTrue &&
                         block->getParent() == &ifOp.getThenRegion()) ||
                        (!selection->selectedWhenTrue &&
                         block->getParent() == &ifOp.getElseRegion()));
      if (inSelectedRegion &&
          selection->candidateIndex ==
              static_cast<int64_t>(candidate.candidateIndex) &&
          selection->waitAny == candidate.waitAny.getOperation() &&
          executesBefore(candidate.waitAny, ifOp)) {
        return true;
      }
    }
    Operation *parent = block->getParentOp();
    if (!parent || parent == candidate.waitAny.getOperation()) {
      break;
    }
    current = parent;
  }
  return false;
}

static LogicalResult
validateConditionalReceiveReleases(ArrayRef<Operation *> reserves,
                                   ArrayRef<Operation *> pushes,
                                   const ConditionalReceiveReleasePlan &plan) {
  for (Operation *reserve : reserves) {
    if (!plan.reserves.contains(reserve)) {
      continue;
    }
    DFBAcquireInterval interval = makeDFBAcquireInterval(reserve, reserves);
    Operation *last = findLastDFBAcquireOwnedUse(interval);
    DFBReleaseSearch releaseSearch =
        findOwnedDFBReleases(interval, last, pushes);
    SmallVector<Operation *> ownedPushes = releaseSearch.sameLevelReleases;
    llvm::append_range(ownedPushes, releaseSearch.nestedReleases);
    for (Operation *push : ownedPushes) {
      auto exactWaits = plan.exactWaitsByReserve.find(reserve);
      bool hasExactWait = exactWaits != plan.exactWaitsByReserve.end() &&
                          llvm::any_of(exactWaits->second, [&](WaitOp wait) {
                            return executesBefore(wait, push);
                          });
      auto candidates = plan.candidatesByReserve.find(reserve);
      bool hasSelectedCandidate =
          candidates != plan.candidatesByReserve.end() &&
          llvm::any_of(candidates->second, [&](WaitAnyCandidate candidate) {
            return isSelectedCandidateRelease(push, candidate);
          });
      if (!hasExactWait && !hasSelectedCandidate) {
        push->emitError(
            "publishes a wait-any receive reservation without proving that "
            "candidate complete");
        return failure();
      }
    }
  }
  return success();
}

struct TTLInsertCBSyncPass
    : public impl::TTLInsertCBSyncBase<TTLInsertCBSyncPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    ValueOriginAnalysis valueOrigins(func);
    FailureOr<ConditionalReceiveReleasePlan> conditionalReleasePlan =
        buildConditionalReceiveReleasePlan(func, valueOrigins);
    if (failed(conditionalReleasePlan)) {
      signalPassFailure();
      return;
    }

    DFBAcquireReleaseOperations operations = collectDFBAcquireReleaseOps(func);
    if (failed(validateConditionalReceiveReleases(
            operations.reserves, operations.pushes, *conditionalReleasePlan))) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(func.getContext());

    // Track erased ops so later iterations skip them before any accessor
    // call. The set holds raw pointers to freed ops; release ownership search
    // must check the set before touching any op wrapper method.
    DenseSet<Operation *> erased;

    insertMissingReleases(operations.reserves, operations.pushes,
                          conditionalReleasePlan->reserves, erased, builder,
                          [](OpBuilder &b, Location loc, Value cb) {
                            CBPushOp::create(b, loc, cb,
                                             /*num_tiles=*/IntegerAttr{});
                          });

    insertMissingReleases(operations.waits, operations.pops,
                          /*acquisitionsRequiringExplicitRelease=*/{}, erased,
                          builder, [](OpBuilder &b, Location loc, Value cb) {
                            CBPopOp::create(b, loc, cb,
                                            /*num_tiles=*/IntegerAttr{});
                          });
  }
};

} // namespace

} // namespace mlir::tt::ttl
