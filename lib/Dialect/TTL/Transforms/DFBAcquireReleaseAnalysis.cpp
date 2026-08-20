// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBAcquireReleaseAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <optional>

//===----------------------------------------------------------------------===//
// DFB Acquire/Release Ownership Analysis
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttl {

namespace {

static bool isBefore(Operation *before, Operation *after) {
  return before->isBeforeInBlock(after);
}

// Matching protocol effects select one pointer side. Unknown, effect-free, and
// index-only accesses may use an acquired slot on either side.
static bool protocolUseMatchesAcquire(DFBAcquireInterval interval,
                                      DFBAccessOpInterface access) {
  if (access.hasUnknownDFBAccess()) {
    return true;
  }

  SmallVector<Value> dependencies = access.getDFBDependencyOperands();
  llvm::BitVector effectfulDependencies(dependencies.size());
  for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
    assert(effect.dependencyIndex < dependencies.size() &&
           "DFB protocol effect dependency index must be valid");
    assert(dependencies[effect.dependencyIndex] == effect.dfb &&
           "DFB protocol effect must reference its dependency occurrence");
    if (effect.dfb != interval.dfb) {
      continue;
    }
    effectfulDependencies.set(effect.dependencyIndex);
    if ((interval.kind == DFBAcquireReleaseKind::Producer &&
         isProducerDFBProtocolEffect(effect.kind)) ||
        (interval.kind == DFBAcquireReleaseKind::Consumer &&
         isConsumerDFBProtocolEffect(effect.kind))) {
      return true;
    }
  }

  bool foundDependency = false;
  for (auto [dependencyIndex, dependency] : llvm::enumerate(dependencies)) {
    if (dependency != interval.dfb) {
      continue;
    }
    foundDependency = true;
    if (!effectfulDependencies.test(dependencyIndex)) {
      return true;
    }
  }
  return !foundDependency;
}

// Operand roles and protocol effects identify the pointer side. Unclassified
// direct accesses conservatively match both sides.
static bool directDFBUseMatchesAcquire(DFBAcquireInterval interval,
                                       Operation *user) {
  if (auto access = dyn_cast<DFBAccessOpInterface>(user)) {
    return protocolUseMatchesAcquire(interval, access);
  }

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

static bool isLifecycleOrIdentityOnlyOp(Operation *operation) {
  return isDFBAcquireOp(operation) || isDFBReleaseOp(operation) ||
         !mayAccessDFBStorage(operation);
}

// Project `op` into the acquire block so nested regions can be ordered against
// the acquire interval. This keeps the interval computation block local while
// still noticing releases nested under control-flow operations.
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

// Find the first later acquire of the same class on `dfb`, projected into the
// current block. Direct DFB uses at or after that operation belong to another
// interval; tensor SSA uses are handled separately because they retain the
// exact acquired slot identity.
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

static bool hasProtocolEffect(Operation *operation, Value dfb,
                              DFBProtocolEffectKind kind) {
  auto access = dyn_cast<DFBAccessOpInterface>(operation);
  return access &&
         llvm::any_of(access.getDFBProtocolEffects(), [&](const auto &effect) {
           return effect.dfb == dfb && effect.kind == kind;
         });
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

int64_t getDFBLifecycleTileCount(Operation *operation) {
  auto access = cast<DFBAccessOpInterface>(operation);
  SmallVector<DFBProtocolEffect> effects = access.getDFBProtocolEffects();
  assert(effects.size() == 1 &&
         "concrete DFB lifecycle ops have exactly one protocol effect");
  return effects.front().numTiles;
}

std::optional<int64_t> getDFBTransactionBlockCount(Operation *operation) {
  assert((isDFBAcquireOp(operation) || isDFBReleaseOp(operation)) &&
         "DFB transaction block count requires a lifecycle operation");
  Value dfb = isDFBAcquireOp(operation) ? getDFBAcquireDFB(operation)
                                        : getDFBReleaseDFB(operation);
  auto dfbType = dyn_cast<CircularBufferType>(dfb.getType());
  if (!dfbType || dfbType.getElementsPerBlock() <= 0) {
    return std::nullopt;
  }
  int64_t numTiles = getDFBLifecycleTileCount(operation);
  if (numTiles <= 0 || numTiles % dfbType.getElementsPerBlock() != 0) {
    return std::nullopt;
  }
  return numTiles / dfbType.getElementsPerBlock();
}

struct OutstandingDFBAcquisition {
  // Acquisition contributing the oldest remaining FIFO tiles.
  Operation *operation = nullptr;

  // Tiles not yet consumed by a same-block release.
  int64_t remainingTiles = 0;
};

// Entry-block FIFO matches and control-flow-dependent releases.
struct SameBlockFIFOOwnership {
  DenseMap<Operation *, SmallVector<Operation *>> owners;
  DenseSet<Operation *> unresolvedReleases;
};

static bool operationMatchesKind(Operation *operation,
                                 DFBAcquireReleaseKind kind) {
  std::optional<DFBAcquireReleaseKind> operationKind =
      getDFBAcquireReleaseKind(operation);
  return operationKind && *operationKind == kind;
}

// Matches direct same-block transactions using the DFB's FIFO protocol.
//
// Only the kernel entry block has no incoming local transaction state. A CFG
// successor or nested block may receive an outstanding acquisition, so
// initializing a local FIFO there would be unsound. Releases outside the entry
// block remain unresolved. A nested lifecycle operation also makes the parent
// queue control-flow-dependent, so subsequent entry-block releases retain
// conservative ownership.
static FailureOr<SameBlockFIFOOwnership>
buildSameBlockFIFOOwners(func::FuncOp kernel, ArrayRef<Operation *> reserves,
                         ArrayRef<Operation *> waits,
                         ArrayRef<Operation *> releases,
                         std::optional<DFBLifecycleDiagnostic> &diagnostic) {
  SameBlockFIFOOwnership ownership;

  auto processBlock = [&](Block *block, DFBAcquireReleaseKind kind,
                          ArrayRef<Operation *> kindAcquisitions) {
    DenseMap<Value, SmallVector<OutstandingDFBAcquisition>> outstanding;
    DenseSet<Value> controlFlowDependentDFBs;

    for (Operation &operation : *block) {
      if (isDFBAcquireOp(&operation) &&
          operationMatchesKind(&operation, kind)) {
        Value dfb = getDFBAcquireDFB(&operation);
        if (!controlFlowDependentDFBs.contains(dfb)) {
          outstanding[dfb].push_back(
              {&operation, getDFBLifecycleTileCount(&operation)});
        }
      } else if (isDFBReleaseOp(&operation) &&
                 operationMatchesKind(&operation, kind)) {
        Value dfb = getDFBReleaseDFB(&operation);
        if (controlFlowDependentDFBs.contains(dfb)) {
          ownership.unresolvedReleases.insert(&operation);
          continue;
        }
        auto queueIterator = outstanding.find(dfb);
        if (queueIterator == outstanding.end()) {
          bool hasSameKindAcquisition =
              llvm::any_of(kindAcquisitions, [&](Operation *acquisition) {
                return getDFBAcquireDFB(acquisition) == dfb;
              });
          if (!hasSameKindAcquisition) {
            continue;
          }
          diagnostic.emplace(
              &operation,
              "dataflow buffer release exceeds preceding entry-block "
              "acquisitions");
          return failure();
        }
        SmallVector<OutstandingDFBAcquisition> updatedQueue =
            queueIterator->second;
        SmallVector<Operation *> candidates;
        int64_t remainingReleaseTiles = getDFBLifecycleTileCount(&operation);
        while (remainingReleaseTiles > 0 && !updatedQueue.empty()) {
          OutstandingDFBAcquisition &acquisition = updatedQueue.front();
          if (!llvm::is_contained(candidates, acquisition.operation)) {
            candidates.push_back(acquisition.operation);
          }
          int64_t releasedTiles =
              std::min(remainingReleaseTiles, acquisition.remainingTiles);
          remainingReleaseTiles -= releasedTiles;
          acquisition.remainingTiles -= releasedTiles;
          if (acquisition.remainingTiles == 0) {
            updatedQueue.erase(updatedQueue.begin());
          }
        }
        if (remainingReleaseTiles != 0) {
          diagnostic.emplace(
              &operation,
              "dataflow buffer release exceeds preceding entry-block "
              "acquisitions");
          return failure();
        }
        ownership.owners.try_emplace(&operation, std::move(candidates));
        queueIterator->second = std::move(updatedQueue);
      }

      for (Region &region : operation.getRegions()) {
        region.walk([&](Operation *nested) {
          if (!operationMatchesKind(nested, kind)) {
            return;
          }
          Value dfb = isDFBAcquireOp(nested) ? getDFBAcquireDFB(nested)
                                             : getDFBReleaseDFB(nested);
          controlFlowDependentDFBs.insert(dfb);
        });
      }
    }
    return success();
  };

  Block *entryBlock = &kernel.getBody().front();
  if (failed(processBlock(entryBlock, DFBAcquireReleaseKind::Producer,
                          reserves)) ||
      failed(
          processBlock(entryBlock, DFBAcquireReleaseKind::Consumer, waits))) {
    return failure();
  }
  for (Operation *operation : releases) {
    if (operation->getBlock() != entryBlock && isDFBReleaseOp(operation)) {
      ownership.unresolvedReleases.insert(operation);
    }
  }
  return ownership;
}

DFBAcquireReleaseOperations collectDFBAcquireReleaseOps(func::FuncOp func) {
  DFBAcquireReleaseOperations operations;
  func.walk([&](Operation *op) {
    if (isa<CBReserveOp>(op)) {
      operations.reserves.push_back(op);
      operations.acquisitions.push_back(op);
    } else if (isa<CBWaitOp>(op)) {
      operations.waits.push_back(op);
      operations.acquisitions.push_back(op);
    } else if (isa<CBPushOp>(op)) {
      operations.pushes.push_back(op);
      operations.releases.push_back(op);
    } else if (isa<CBPopOp>(op)) {
      operations.pops.push_back(op);
      operations.releases.push_back(op);
    }
    auto access = dyn_cast<DFBAccessOpInterface>(op);
    if (!access) {
      return;
    }
    for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
      if (effect.kind == DFBProtocolEffectKind::Push &&
          !llvm::is_contained(operations.producerProtocolReleases, op)) {
        operations.producerProtocolReleases.push_back(op);
      } else if (effect.kind == DFBProtocolEffectKind::Pop &&
                 !llvm::is_contained(operations.consumerProtocolReleases, op)) {
        operations.consumerProtocolReleases.push_back(op);
      }
    }
  });
  return operations;
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

  auto extend = [&](Operation *user, bool ignoreBoundary,
                    bool propagateResults) {
    Operation *projected = nullptr;
    if (!projectToAcquireBlock(interval, user, projected, ignoreBoundary)) {
      return false;
    }
    if (!visited.insert(user).second) {
      return false;
    }
    updateLatestUse(projected, last);
    if (propagateResults) {
      for (Value result : user->getResults()) {
        worklist.push_back(result);
      }
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
        extend(user, ignoreBoundary, /*propagateResults=*/true);
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
    if (isLifecycleOrIdentityOnlyOp(user)) {
      continue;
    }
    if (!directDFBUseMatchesAcquire(interval, user)) {
      continue;
    }
    extend(user, /*ignoreBoundary=*/false, /*propagateResults=*/true);
  }
  drainWorklist(/*ignoreBoundary=*/false);

  if (isUserManagedDFB(interval.dfb)) {
    // Unknown access has no SSA use of this DFB, so include it explicitly in
    // every user-managed interval that may contain the operation.
    func::FuncOp kernel = interval.acquire->getParentOfType<func::FuncOp>();
    kernel.walk([&](Operation *operation) {
      auto access = dyn_cast<DFBAccessOpInterface>(operation);
      if (access && access.hasUnknownDFBAccess()) {
        extend(operation, /*ignoreBoundary=*/false,
               /*propagateResults=*/false);
      }
    });
  }

  // Tensor SSA uses keep naming the slot acquired by this operation even after
  // a later DFB acquire advances the pointer. Applying the direct-DFB boundary
  // here made auto-sync insertion release a slot before its final tensor use.
  assert(interval.acquire->getNumResults() == 1 &&
         "DFB acquire ops produce exactly one tensor result");
  worklist.push_back(interval.acquire->getResult(0));
  drainWorklist(/*ignoreBoundary=*/true);

  return last;
}

DFBReleaseSearch findOwnedDFBReleases(DFBAcquireInterval interval,
                                      Operation *lastOwnedUse,
                                      ArrayRef<Operation *> releases) {
  DFBReleaseSearch result;
  Block *block = interval.acquire->getBlock();
  DFBProtocolEffectKind releaseEffectKind =
      interval.kind == DFBAcquireReleaseKind::Producer
          ? DFBProtocolEffectKind::Push
          : DFBProtocolEffectKind::Pop;

  bool useExtendsPastBoundary =
      lastOwnedUse && lastOwnedUse != interval.acquire &&
      interval.kindBoundary && !isBefore(lastOwnedUse, interval.kindBoundary);

  for (Operation *release : releases) {
    if (!hasProtocolEffect(release, interval.dfb, releaseEffectKind)) {
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

PlanningResult<std::unique_ptr<DFBAcquireReleaseIndex>>
DFBAcquireReleaseIndex::create(func::FuncOp kernel) {
  auto index =
      std::unique_ptr<DFBAcquireReleaseIndex>(new DFBAcquireReleaseIndex());
  std::optional<DFBLifecycleDiagnostic> diagnostic;
  if (failed(index->build(kernel, diagnostic))) {
    assert(diagnostic && "failed lifecycle indexing requires a diagnostic");
    return PlanningResult<std::unique_ptr<DFBAcquireReleaseIndex>>::invalidIR(
        diagnostic->operation, std::move(diagnostic->message));
  }
  return PlanningResult<std::unique_ptr<DFBAcquireReleaseIndex>>::planned(
      std::move(index));
}

LogicalResult DFBAcquireReleaseIndex::build(
    func::FuncOp kernel, std::optional<DFBLifecycleDiagnostic> &diagnostic) {
  DFBAcquireReleaseOperations operations = collectDFBAcquireReleaseOps(kernel);
  acquisitionOrder = operations.acquisitions;
  releaseOrder = operations.releases;

  auto recordTransactions = [&](ArrayRef<Operation *> acquires) {
    for (Operation *acquire : acquires) {
      transactions.try_emplace(
          acquire, DFBTransactionRecord{acquire, getDFBAcquireDFB(acquire),
                                        *getDFBAcquireReleaseKind(acquire),
                                        getDFBLifecycleTileCount(acquire)});
    }
  };
  recordTransactions(operations.reserves);
  recordTransactions(operations.waits);

  FailureOr<SameBlockFIFOOwnership> fifoOwnershipResult =
      buildSameBlockFIFOOwners(kernel, operations.reserves, operations.waits,
                               operations.releases, diagnostic);
  if (failed(fifoOwnershipResult)) {
    return failure();
  }
  SameBlockFIFOOwnership &fifoOwnership = *fifoOwnershipResult;
  auto recordReleaseOwnership = [&](ArrayRef<Operation *> releases,
                                    ArrayRef<Operation *> acquires) {
    for (Operation *release : releases) {
      Value dfb = getDFBReleaseDFB(release);
      SmallVector<Operation *> candidates;

      bool hasSameKindAcquisition = llvm::any_of(
          acquires, [&](Operation *op) { return getDFBAcquireDFB(op) == dfb; });
      if (!hasSameKindAcquisition) {
        diagnostic.emplace(
            release, "dataflow buffer release has no same-kind acquisition "
                     "in the enclosing kernel");
        return failure();
      }

      DFBReleaseOwnershipKind ownership = DFBReleaseOwnershipKind::Unresolved;
      if (!fifoOwnership.unresolvedReleases.contains(release)) {
        auto owners = fifoOwnership.owners.find(release);
        assert(owners != fifoOwnership.owners.end() &&
               !owners->second.empty() &&
               "positive entry-block release must have FIFO owners");
        llvm::append_range(candidates, owners->second);
        ownership = candidates.size() == 1 ? DFBReleaseOwnershipKind::Exact
                                           : DFBReleaseOwnershipKind::Multiple;
      } else {
        for (Operation *acquisition : acquires) {
          if (getDFBAcquireDFB(acquisition) == dfb) {
            candidates.push_back(acquisition);
          }
        }
      }
      releaseOwnership.try_emplace(
          release,
          DFBReleaseOwnership{release, dfb, *getDFBAcquireReleaseKind(release),
                              getDFBLifecycleTileCount(release), ownership,
                              std::move(candidates)});
    }
    return success();
  };

  if (failed(recordReleaseOwnership(operations.pushes, operations.reserves)) ||
      failed(recordReleaseOwnership(operations.pops, operations.waits))) {
    return failure();
  }
  auto recordIntervalOwners = [&](ArrayRef<Operation *> acquires,
                                  ArrayRef<Operation *> releases) {
    for (Operation *release : releases) {
      releaseIntervalOwners.try_emplace(release);
    }
    for (Operation *acquire : acquires) {
      DFBAcquireInterval interval = makeDFBAcquireInterval(acquire, acquires);
      Operation *lastOwnedUse = findLastDFBAcquireOwnedUse(interval);
      DFBReleaseSearch releaseSearch =
          findOwnedDFBReleases(interval, lastOwnedUse, releases);
      for (Operation *release : releaseSearch.sameLevelReleases) {
        releaseIntervalOwners[release].push_back(acquire);
      }
      for (Operation *release : releaseSearch.nestedReleases) {
        releaseIntervalOwners[release].push_back(acquire);
      }
    }
  };
  recordIntervalOwners(operations.reserves, operations.pushes);
  recordIntervalOwners(operations.waits, operations.pops);

  return success();
}

const DFBTransactionRecord &
DFBAcquireReleaseIndex::getTransaction(Operation *acquire) const {
  auto transaction = transactions.find(acquire);
  assert(transaction != transactions.end() &&
         "operation is not an indexed DFB acquisition");
  return transaction->second;
}

const DFBReleaseOwnership &
DFBAcquireReleaseIndex::getReleaseOwnership(Operation *release) const {
  auto ownership = releaseOwnership.find(release);
  assert(ownership != releaseOwnership.end() &&
         "operation is not an indexed DFB release");
  return ownership->second;
}

ArrayRef<Operation *>
DFBAcquireReleaseIndex::getReleaseIntervalOwners(Operation *release) const {
  auto owners = releaseIntervalOwners.find(release);
  assert(owners != releaseIntervalOwners.end() &&
         "operation is not an indexed DFB release");
  return owners->second;
}

} // namespace mlir::tt::ttl
