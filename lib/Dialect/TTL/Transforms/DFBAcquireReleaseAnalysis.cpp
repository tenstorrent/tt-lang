// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBAcquireReleaseAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "llvm/ADT/DenseSet.h"

#include <optional>

//===----------------------------------------------------------------------===//
// DFB Acquire/Release Ownership Analysis
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttl {

namespace {

static bool isBefore(Operation *before, Operation *after) {
  return before->isBeforeInBlock(after);
}

/// Direct DFB copies can use the same DFB value on either source or
/// destination operands. Only the operand that corresponds to the acquire class
/// consumes the acquired slot.
static bool directDFBUseMatchesAcquire(DFBAcquireInterval interval,
                                       Operation *user) {
  auto copy = dyn_cast<CopyOp>(user);
  if (!copy) {
    return true;
  }

  switch (interval.kind) {
  case DFBAcquireReleaseKind::Producer:
    return copy.getDst() == interval.dfb;
  case DFBAcquireReleaseKind::Consumer:
    return copy.getSrc() == interval.dfb;
  }
  llvm_unreachable("unknown DFB acquire/release kind");
}

static bool isLifecycleOrAttachOp(Operation *op) {
  return isDFBAcquireOp(op) || isDFBReleaseOp(op) || isa<AttachCBOp>(op);
}

/// Project `op` into the acquire block so nested regions can be ordered
/// against the acquire interval. This keeps the interval computation block
/// local while still noticing releases nested under control-flow operations.
static bool projectToAcquireBlock(DFBAcquireInterval interval, Operation *op,
                                  Operation *&projected,
                                  bool ignoreBoundary = false) {
  Block *block = interval.acquire->getBlock();
  projected = op->getBlock() == block ? op : block->findAncestorOpInBlock(*op);
  if (!projected) {
    return false;
  }
  if (!isBefore(interval.acquire, projected)) {
    return false;
  }
  if (!ignoreBoundary && interval.kindBoundary &&
      !isBefore(projected, interval.kindBoundary)) {
    return false;
  }
  return true;
}

static void updateLatestUse(Operation *candidate, Operation *&latest) {
  if (isBefore(latest, candidate)) {
    latest = candidate;
  }
}

/// Find the first later acquire of the same class on `dfb`, projected into the
/// current block. Direct DFB uses at or after that operation belong to another
/// interval; tensor SSA uses are handled separately because they retain the
/// exact acquired slot identity.
static void updateBoundary(Value dfb, Operation *acquire,
                           ArrayRef<Operation *> acquires,
                           Operation *&boundary) {
  Block *block = acquire->getBlock();
  for (Operation *other : acquires) {
    if (other == acquire) {
      continue;
    }
    if (getDFBAcquireDFB(other) != dfb) {
      continue;
    }
    Operation *ancestor = block->findAncestorOpInBlock(*other);
    if (!ancestor) {
      continue;
    }
    if (!isBefore(acquire, ancestor)) {
      continue;
    }
    if (!boundary || isBefore(ancestor, boundary)) {
      boundary = ancestor;
    }
  }
}

static Operation *findNextSameKindAcquire(Value dfb, Operation *acquire,
                                          ArrayRef<Operation *> acquires) {
  Operation *boundary = nullptr;
  updateBoundary(dfb, acquire, acquires, boundary);
  return boundary;
}

} // namespace

bool isDFBAcquireOp(Operation *op) { return isa<CBReserveOp, CBWaitOp>(op); }

bool isDFBReleaseOp(Operation *op) { return isa<CBPushOp, CBPopOp>(op); }

Value getDFBAcquireDFB(Operation *op) {
  if (auto reserve = dyn_cast<CBReserveOp>(op)) {
    return reserve.getCb();
  }
  return cast<CBWaitOp>(op).getCb();
}

Value getDFBReleaseDFB(Operation *op) {
  if (auto push = dyn_cast<CBPushOp>(op)) {
    return push.getCb();
  }
  return cast<CBPopOp>(op).getCb();
}

static std::optional<DFBAcquireReleaseKind>
getDFBAcquireReleaseKind(Operation *op) {
  if (isa<CBReserveOp, CBPushOp>(op)) {
    return DFBAcquireReleaseKind::Producer;
  }
  if (isa<CBWaitOp, CBPopOp>(op)) {
    return DFBAcquireReleaseKind::Consumer;
  }
  return std::nullopt;
}

void collectDFBAcquireReleaseOps(func::FuncOp func,
                                 SmallVectorImpl<Operation *> &reserves,
                                 SmallVectorImpl<Operation *> &waits,
                                 SmallVectorImpl<Operation *> &pushes,
                                 SmallVectorImpl<Operation *> &pops) {
  func.walk([&](Operation *op) {
    if (isa<CBReserveOp>(op)) {
      reserves.push_back(op);
    } else if (isa<CBWaitOp>(op)) {
      waits.push_back(op);
    } else if (isa<CBPushOp>(op)) {
      pushes.push_back(op);
    } else if (isa<CBPopOp>(op)) {
      pops.push_back(op);
    }
  });
}

DFBAcquireInterval makeDFBAcquireInterval(Operation *acquire,
                                          ArrayRef<Operation *> acquires) {
  Value dfb = getDFBAcquireDFB(acquire);
  std::optional<DFBAcquireReleaseKind> kind = getDFBAcquireReleaseKind(acquire);
  assert(kind && "DFB acquire interval requires acquire operation");
  return {acquire, dfb, *kind, findNextSameKindAcquire(dfb, acquire, acquires)};
}

Operation *findLastDFBAcquireOwnedUse(DFBAcquireInterval interval) {
  Operation *last = interval.acquire;
  llvm::DenseSet<Operation *> visited;
  SmallVector<Value, 8> worklist;

  auto extend = [&](Operation *user, bool ignoreBoundary) {
    Operation *projected = nullptr;
    if (!projectToAcquireBlock(interval, user, projected, ignoreBoundary)) {
      return false;
    }
    if (!visited.insert(user).second) {
      return false;
    }
    updateLatestUse(projected, last);
    for (Value result : user->getResults()) {
      worklist.push_back(result);
    }
    return true;
  };

  // Walk result users transitively because the operation that truly ends an
  // interval can consume a value derived from an earlier direct DFB operation.
  auto drainWorklist = [&](bool ignoreBoundary) {
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (isa<CBPushOp, CBPopOp>(user)) {
          continue;
        }
        extend(user, ignoreBoundary);
      }
    }
  };

  // Direct DFB uses are tied to the current DFB pointer position. A later
  // same-kind acquire on the same DFB starts a new pointer interval, so direct
  // uses after the boundary are excluded.
  for (OpOperand &use : interval.dfb.getUses()) {
    Operation *user = use.getOwner();
    if (user == interval.acquire) {
      continue;
    }
    if (isLifecycleOrAttachOp(user)) {
      continue;
    }
    if (!directDFBUseMatchesAcquire(interval, user)) {
      continue;
    }
    extend(user, /*ignoreBoundary=*/false);
  }
  drainWorklist(/*ignoreBoundary=*/false);

  // Tensor SSA uses keep naming the slot acquired by this operation even after
  // a later DFB acquire advances the pointer. Applying the direct-DFB boundary
  // here made auto-sync insertion release a slot before its final tensor use.
  assert(interval.acquire->getNumResults() == 1 &&
         "DFB acquire ops produce exactly one tensor result");
  worklist.push_back(interval.acquire->getResult(0));
  drainWorklist(/*ignoreBoundary=*/true);

  return last;
}

DFBReleaseSearch
findOwnedDFBReleases(DFBAcquireInterval interval, Operation *lastOwnedUse,
                     ArrayRef<Operation *> releases,
                     const llvm::DenseSet<Operation *> *erased) {
  DFBReleaseSearch result;
  Block *block = interval.acquire->getBlock();

  bool useExtendsPastBoundary =
      lastOwnedUse && lastOwnedUse != interval.acquire &&
      interval.kindBoundary && !isBefore(lastOwnedUse, interval.kindBoundary);

  for (Operation *release : releases) {
    // `ttl-insert-cb-sync` may erase releases while iterating. The set contains
    // raw pointers to erased operations, so membership must be checked before
    // reading the operation through an op wrapper.
    if (erased && erased->contains(release)) {
      continue;
    }
    if (getDFBReleaseDFB(release) != interval.dfb) {
      continue;
    }

    if (release->getBlock() == block) {
      Operation *projected = nullptr;
      if (projectToAcquireBlock(interval, release, projected)) {
        result.sameLevelReleases.push_back(release);
        continue;
      }
      // Idempotency case: if the pass previously inserted a release after a
      // use that crosses the next-acquire boundary, accept that release as
      // owned by the original acquire.
      if (useExtendsPastBoundary &&
          projectToAcquireBlock(interval, release, projected,
                                /*ignoreBoundary=*/true) &&
          !isBefore(projected, lastOwnedUse)) {
        result.sameLevelReleases.push_back(release);
      }
      continue;
    }

    Operation *projected = nullptr;
    if (!projectToAcquireBlock(interval, release, projected)) {
      continue;
    }
    result.nestedReleases.push_back(release);
  }

  return result;
}

} // namespace mlir::tt::ttl
