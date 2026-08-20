// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBACQUIRERELEASEANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBACQUIRERELEASEANALYSIS_H

//===----------------------------------------------------------------------===//
// DFB Acquire/Release Ownership Analysis
//===----------------------------------------------------------------------===//
//
// This utility computes ownership between DFB acquire operations and matching
// closing operations:
//
//   ttl.cb_reserve -> ttl.cb_push
//   ttl.cb_wait    -> ttl.cb_pop
//
// A push relinquishes producer access to reserved slots and publishes their
// tiles. A pop relinquishes consumer access to waited slots and returns their
// capacity to the producer. Neither operation deallocates the DFB.
//
// The analysis is intentionally local to a kernel and to one acquire class.
// Producer intervals and consumer intervals are independent because
// reserve/push and wait/pop advance different DFB pointers. A release is owned
// by the acquire interval whose acquired DFB slot is live until that release.
// Consumers use the result tensor as evidence of slot ownership, while direct
// DFB operations use the DFB value itself.
//
// The interval queries let `ttl-insert-cb-sync` place missing releases after
// their owned uses. They recognize release effects exposed by
// `DFBAccessOpInterface`, so an external call can close an interval without a
// separate lifecycle operation. The immutable index additionally matches
// concrete releases to acquisitions by FIFO tile count for DFB value lifetime
// analysis. Keeping both models here gives them one definition of
// acquire/release kinds, DFB identity, and transfer counts.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "ttlang/Analysis/PlanningResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

/// Classifies a reserve/push producer interval or wait/pop consumer interval.
enum class DFBAcquireReleaseKind { Producer, Consumer };

/// Half-open ownership interval for one DFB acquire operation.
///
/// `kindBoundary` is the closest later acquire of the same kind on the same
/// DFB, projected into the acquire block. Direct DFB uses after that boundary
/// belong to a later acquire interval. Tensor SSA uses may extend past the
/// boundary because they continue to name the original acquired slot.
struct DFBAcquireInterval {
  /// The `ttl.cb_reserve` or `ttl.cb_wait` that starts the interval.
  Operation *acquire = nullptr;

  /// The DFB value acquired by `acquire`.
  Value dfb;

  /// Whether this interval owns producer-side or consumer-side releases.
  DFBAcquireReleaseKind kind = DFBAcquireReleaseKind::Producer;

  /// Closest later same-kind acquire on `dfb`, or null if none exists.
  Operation *kindBoundary = nullptr;
};

/// Push or pop actions that close one acquire interval.
struct DFBReleaseSearch {
  /// Releases in the acquire block or projected into that block.
  SmallVector<Operation *> sameLevelReleases;

  /// Releases nested under operations in the acquire interval.
  SmallVector<Operation *> nestedReleases;

  bool hasSameLevelRelease() const { return !sameLevelReleases.empty(); }
};

/// Static transaction facts for one DFB acquisition.
///
/// `numTiles` is taken from the operation's `num_tiles` attribute or from the
/// DFB's elements per block when the attribute is absent. Releases may remain
/// absent before automatic synchronization inserts them.
struct DFBTransactionRecord {
  Operation *acquire = nullptr;
  Value dfb;
  DFBAcquireReleaseKind kind = DFBAcquireReleaseKind::Producer;
  int64_t numTiles = 0;
};

/// Precision of the relation between one release and its acquisition.
enum class DFBReleaseOwnershipKind {
  /// Exactly one acquisition owns the release.
  Exact,

  /// The release consumes tiles from several same-block FIFO acquisitions.
  Multiple,

  /// Control flow prevents the entry-block FIFO model from selecting one or
  /// more acquisitions that own the release.
  Unresolved,
};

/// Static ownership facts for one DFB release.
struct DFBReleaseOwnership {
  Operation *release = nullptr;
  Value dfb;
  DFBAcquireReleaseKind kind = DFBAcquireReleaseKind::Producer;
  int64_t numTiles = 0;
  DFBReleaseOwnershipKind ownership = DFBReleaseOwnershipKind::Unresolved;

  /// Exact FIFO owners, or every possible owner for an unresolved release.
  SmallVector<Operation *> candidateOwners;

  /// Returns the exact owner after asserting that ownership is exact.
  Operation *getExactOwner() const {
    assert(ownership == DFBReleaseOwnershipKind::Exact &&
           candidateOwners.size() == 1 &&
           "exact release ownership requires one candidate");
    return candidateOwners.front();
  }
};

/// Definite malformed-lifecycle result found while indexing one kernel.
using DFBLifecycleDiagnostic = PlanningDiagnostic;

/// Returns true for DFB acquire ops accepted by this analysis.
bool isDFBAcquireOp(Operation *op);

/// Returns true for DFB release ops accepted by this analysis.
bool isDFBReleaseOp(Operation *op);

/// Returns the tile count of a concrete DFB lifecycle operation.
///
/// `operation` must expose exactly one protocol effect.
int64_t getDFBLifecycleTileCount(Operation *operation);

/// Returns the DFB operand of a `ttl.cb_reserve` or `ttl.cb_wait`.
Value getDFBAcquireDFB(Operation *op);

/// Returns the DFB operand of a `ttl.cb_push` or `ttl.cb_pop`.
Value getDFBReleaseDFB(Operation *op);

/// Returns the number of whole DFB blocks acquired or released by `op`.
///
/// Returns `std::nullopt` when the transaction size is not a positive multiple
/// of the DFB block size.
std::optional<int64_t> getDFBTransactionBlockCount(Operation *op);

/// DFB lifecycle operations collected in one function traversal.
struct DFBAcquireReleaseOperations {
  SmallVector<Operation *> reserves;
  SmallVector<Operation *> waits;
  SmallVector<Operation *> pushes;
  SmallVector<Operation *> pops;
  SmallVector<Operation *> acquisitions;
  SmallVector<Operation *> releases;

  /// Producer release candidates used by synchronization insertion.
  /// Includes concrete `ttl.cb_push` ops and operations that summarize push
  /// effects. Entries identify operations; release matching selects the effect
  /// for the acquired DFB.
  SmallVector<Operation *> producerProtocolReleases;

  /// Consumer release candidates used by synchronization insertion.
  /// Includes concrete `ttl.cb_pop` ops and operations that summarize pop
  /// effects. Entries identify operations; release matching selects the effect
  /// for the acquired DFB.
  SmallVector<Operation *> consumerProtocolReleases;
};

/// Collects DFB lifecycle operations from `func` in walk order.
DFBAcquireReleaseOperations collectDFBAcquireReleaseOps(func::FuncOp func);

/// Builds the ownership interval for `acquire`.
///
/// `acquires` must contain acquire operations of the same
/// `DFBAcquireReleaseKind`, for example all reserves or all waits in the
/// enclosing kernel.
DFBAcquireInterval makeDFBAcquireInterval(Operation *acquire,
                                          ArrayRef<Operation *> acquires);

/// Finds the last operation in `interval.acquire`'s block that is owned by the
/// interval.
///
/// See `docs/development/DFBManagement.md` for the asymmetric classification of
/// direct DFB uses and tensor SSA uses.
Operation *findLastDFBAcquireOwnedUse(DFBAcquireInterval interval);

/// Finds releases owned by `interval`.
///
/// When `lastOwnedUse` is null, only the strict range before the next same-kind
/// acquire is searched. When non-null and it extends past that boundary, the
/// search also accepts releases after `lastOwnedUse`; this makes repeated
/// auto-sync insertion idempotent. `releases` may contain concrete lifecycle
/// operations or operations that summarize external protocol effects.
DFBReleaseSearch findOwnedDFBReleases(DFBAcquireInterval interval,
                                      Operation *lastOwnedUse,
                                      ArrayRef<Operation *> releases);

/// Immutable release-to-acquisition relations for one kernel.
///
/// Tile counts in the kernel entry block prove one or more FIFO owners. Other
/// blocks may receive outstanding acquisitions and therefore retain unresolved
/// release ownership. An unresolved record contains every same-kind
/// acquisition on the DFB so clients cannot apply a less conservative rule.
class DFBAcquireReleaseIndex {
public:
  /// Builds the index without modifying IR.
  ///
  /// Missing releases are accepted because automatic synchronization runs
  /// later in the pipeline. A release with no same-kind acquisition on its DFB
  /// or an entry-block release that exceeds preceding acquisitions cannot be
  /// repaired by release insertion and returns an invalid-IR result anchored
  /// at that release.
  static PlanningResult<std::unique_ptr<DFBAcquireReleaseIndex>>
  create(func::FuncOp kernel);

  /// Returns the transaction record for `acquire`.
  const DFBTransactionRecord &getTransaction(Operation *acquire) const;

  /// Returns acquisitions in deterministic kernel walk order.
  ArrayRef<Operation *> getAcquisitions() const { return acquisitionOrder; }

  /// Returns the ownership record for `release`.
  const DFBReleaseOwnership &getReleaseOwnership(Operation *release) const;

  /// Returns acquisition intervals whose operation range contains `release`.
  ///
  /// This structural relation is independent of FIFO tile ownership. Clients
  /// that require one interval must verify that the returned range has one
  /// element.
  ArrayRef<Operation *> getReleaseIntervalOwners(Operation *release) const;

  /// Returns releases in deterministic kernel walk order.
  ArrayRef<Operation *> getReleases() const { return releaseOrder; }

private:
  DFBAcquireReleaseIndex() = default;

  LogicalResult build(func::FuncOp kernel,
                      std::optional<DFBLifecycleDiagnostic> &diagnostic);

  SmallVector<Operation *> acquisitionOrder;
  SmallVector<Operation *> releaseOrder;
  llvm::DenseMap<Operation *, DFBTransactionRecord> transactions;
  llvm::DenseMap<Operation *, DFBReleaseOwnership> releaseOwnership;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> releaseIntervalOwners;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBACQUIRERELEASEANALYSIS_H
