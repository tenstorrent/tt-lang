// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Live Interval Utilities
//===----------------------------------------------------------------------===//
//
// This file provides small interval primitives for TTL transforms that map
// logical program objects to a bounded set of physical resources. It does not
// compute full MLIR SSA liveness because these resources are often live for
// protocol-specific event ranges rather than for the full SSA value lifetime.
// For example, pipe sender-ready counters and address-table slots are reusable
// after the send consumes the posted state, even though the transfer token may
// remain live until a later wait. Callers define the interval endpoints that
// matter for the resource being allocated, then use these helpers to compare
// and color those intervals deterministically.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_LIVEINTERVALUTILS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_LIVEINTERVALUTILS_H

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <limits>

namespace mlir::tt::ttl {

/// Half-open interval for a value-like resource in a linear event order.
///
/// The interval covers events in the range [start, end). A value with end equal
/// to another value's start does not conflict with that value.
struct ValueLiveInterval {
  int64_t start = 0;
  int64_t end = 0;
  Value value;
  /// Stable tie breaker for intervals with identical event ranges.
  int64_t ordinal = 0;

  /// Deterministic order used by first-fit interval coloring.
  bool operator<(const ValueLiveInterval &rhs) const {
    if (start != rhs.start) {
      return start < rhs.start;
    }
    if (end != rhs.end) {
      return end < rhs.end;
    }
    return ordinal < rhs.ordinal;
  }
};

/// Return true when two half-open integer intervals overlap.
inline bool halfOpenIntervalsOverlap(int64_t lhsStart, int64_t lhsEnd,
                                     int64_t rhsStart, int64_t rhsEnd) {
  return lhsEnd > rhsStart && rhsEnd > lhsStart;
}

/// Return true when two value live intervals overlap in linear event order.
inline bool intervalsOverlap(const ValueLiveInterval &lhs,
                             const ValueLiveInterval &rhs) {
  assert(lhs.start <= lhs.end && rhs.start <= rhs.end &&
         "value live interval end must not precede its start");
  return halfOpenIntervalsOverlap(lhs.start, lhs.end, rhs.start, rhs.end);
}

/// Operation-bounded interval for resources whose lifetime is defined by IR
/// operations rather than by SSA use-def lifetime.
///
/// A bounded interval has a start operation that dominates its end operation
/// and an end operation that post-dominates its start operation. If those
/// conditions cannot be proven, the interval is marked unbounded. Unbounded
/// intervals conservatively conflict with every other operation interval.
struct OperationLiveInterval {
  Operation *start = nullptr;
  Operation *end = nullptr;
  int64_t startOrdinal = std::numeric_limits<int64_t>::max();
  /// Reason the interval cannot be treated as a bounded operation range.
  enum class State {
    Bounded,
    MissingEndpoint,
    IncomparableStarts,
    IncomparableEnds,
    NonPostDominatingEnd,
  };
  State state = State::Bounded;
};

/// Mark an interval unbounded, preserving the first recorded reason.
inline void setUnbounded(OperationLiveInterval &interval,
                         OperationLiveInterval::State state) {
  if (interval.state == OperationLiveInterval::State::Bounded) {
    interval.state = state;
  }
}

/// Return true when an interval must conservatively conflict with all peers.
inline bool isUnbounded(const OperationLiveInterval &interval) {
  return interval.state != OperationLiveInterval::State::Bounded;
}

/// Extend an operation interval with a candidate start operation.
///
/// The earliest dominating start is retained. If neither the existing start nor
/// the candidate start dominates the other, the interval is marked unbounded.
inline void updateIntervalStart(OperationLiveInterval &interval, Operation *op,
                                int64_t opOrdinal,
                                const DominanceInfo &dominanceInfo) {
  if (!interval.start || dominanceInfo.properlyDominates(op, interval.start)) {
    interval.start = op;
    interval.startOrdinal = opOrdinal;
    return;
  }
  if (!dominanceInfo.dominates(interval.start, op)) {
    setUnbounded(interval, OperationLiveInterval::State::IncomparableStarts);
  }
}

/// Extend an operation interval with a candidate end operation.
///
/// The latest dominated end is retained. If neither the existing end nor the
/// candidate end dominates the other, the interval is marked unbounded.
inline void updateIntervalEnd(OperationLiveInterval &interval, Operation *op,
                              const DominanceInfo &dominanceInfo) {
  if (!interval.end) {
    interval.end = op;
    return;
  }
  if (dominanceInfo.properlyDominates(interval.end, op)) {
    interval.end = op;
    return;
  }
  if (!dominanceInfo.dominates(op, interval.end)) {
    setUnbounded(interval, OperationLiveInterval::State::IncomparableEnds);
  }
}

/// Validate the final operation interval bounds.
///
/// Missing endpoints, non-dominating start/end pairs, and ends that do not
/// post-dominate starts are marked unbounded. This makes callers conservative
/// in the presence of one-sided IR fragments or control flow. Endpoint presence
/// is derived from the start and end operations recorded by the update helpers.
inline void finalizeInterval(OperationLiveInterval &interval,
                             const DominanceInfo &dominanceInfo,
                             const PostDominanceInfo &postDominanceInfo) {
  if (!interval.start || !interval.end) {
    setUnbounded(interval, OperationLiveInterval::State::MissingEndpoint);
    return;
  }

  if (!dominanceInfo.dominates(interval.start, interval.end) ||
      !postDominanceInfo.postDominates(interval.end, interval.start)) {
    setUnbounded(interval, OperationLiveInterval::State::NonPostDominatingEnd);
  }
}

/// Return true when two operation live intervals may be concurrently live.
///
/// Unbounded intervals always overlap. Otherwise, intervals are disjoint only
/// when one interval's end properly dominates the other interval's start.
inline bool intervalsOverlap(const OperationLiveInterval &lhs,
                             const OperationLiveInterval &rhs,
                             const DominanceInfo &dominanceInfo) {
  if (isUnbounded(lhs) || isUnbounded(rhs) || !lhs.start || !lhs.end ||
      !rhs.start || !rhs.end) {
    return true;
  }
  return !(dominanceInfo.properlyDominates(lhs.end, rhs.start) ||
           dominanceInfo.properlyDominates(rhs.end, lhs.start));
}

/// Assign deterministic first-fit colors to interval-like items.
///
/// Callers supply ordering and conflict predicates because TTL resources have
/// different interval construction semantics. The returned outer vector is
/// indexed by color; each inner vector contains the items assigned to that
/// color.
template <typename ItemT, typename IsBeforeFn, typename ConflictsFn>
SmallVector<SmallVector<ItemT>>
assignGreedyIntervalColors(ArrayRef<ItemT> items, IsBeforeFn isBefore,
                           ConflictsFn conflicts) {
  SmallVector<ItemT> sortedItems(items.begin(), items.end());
  llvm::sort(sortedItems, isBefore);

  SmallVector<SmallVector<ItemT>> colorUsers;
  for (const ItemT &item : sortedItems) {
    unsigned selectedColor = 0;
    for (;; ++selectedColor) {
      if (selectedColor == colorUsers.size()) {
        colorUsers.push_back({});
        break;
      }
      bool hasConflict =
          llvm::any_of(colorUsers[selectedColor], [&](const ItemT &assigned) {
            return conflicts(item, assigned);
          });
      if (!hasConflict) {
        break;
      }
    }
    colorUsers[selectedColor].push_back(item);
  }

  return colorUsers;
}

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_LIVEINTERVALUTILS_H
