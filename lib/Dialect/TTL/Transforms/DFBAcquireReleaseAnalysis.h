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
// release operations:
//
//   ttl.cb_reserve -> ttl.cb_push
//   ttl.cb_wait    -> ttl.cb_pop
//
// The analysis is intentionally local to a function and to one acquire class.
// Producer intervals and consumer intervals are independent because
// reserve/push and wait/pop advance different DFB pointers. A release is owned
// by the acquire interval whose acquired DFB slot is live until that release.
// Consumers use the result tensor as evidence of slot ownership, while direct
// DFB operations use the DFB value itself.
//
// The same ownership rules are used by `ttl-insert-cb-sync` and by PipeNet
// lowering proofs. Keeping the rules here avoids disagreement between the pass
// that creates implicit releases and later analyses that attach protocol
// meaning to those releases.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

/// Classifies the DFB pointer advanced by an acquire/release pair.
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

/// Releases found inside one acquire interval.
struct DFBReleaseSearch {
  /// Releases in the acquire block or projected into that block.
  SmallVector<Operation *> sameLevelReleases;

  /// Releases nested under operations in the acquire interval.
  SmallVector<Operation *> nestedReleases;

  bool hasSameLevelRelease() const { return !sameLevelReleases.empty(); }
};

/// Returns true for DFB acquire ops accepted by this analysis.
bool isDFBAcquireOp(Operation *op);

/// Returns true for DFB release ops accepted by this analysis.
bool isDFBReleaseOp(Operation *op);

/// Returns the DFB operand of a `ttl.cb_reserve` or `ttl.cb_wait`.
Value getDFBAcquireDFB(Operation *op);

/// Returns the DFB operand of a `ttl.cb_push` or `ttl.cb_pop`.
Value getDFBReleaseDFB(Operation *op);

/// Collects DFB lifecycle operations from `func` in walk order.
void collectDFBAcquireReleaseOps(func::FuncOp func,
                                 SmallVectorImpl<Operation *> &reserves,
                                 SmallVectorImpl<Operation *> &waits,
                                 SmallVectorImpl<Operation *> &pushes,
                                 SmallVectorImpl<Operation *> &pops);

/// Builds the ownership interval for `acquire`.
///
/// `acquires` must contain acquire operations of the same
/// `DFBAcquireReleaseKind`, for example all reserves or all waits in the
/// enclosing function.
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
/// auto-sync insertion idempotent.
DFBReleaseSearch
findOwnedDFBReleases(DFBAcquireInterval interval, Operation *lastOwnedUse,
                     ArrayRef<Operation *> releases,
                     const llvm::DenseSet<Operation *> *erased = nullptr);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBACQUIRERELEASEANALYSIS_H
