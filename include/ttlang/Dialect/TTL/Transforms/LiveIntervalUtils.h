// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_LIVEINTERVALUTILS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_LIVEINTERVALUTILS_H

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>

namespace mlir::tt::ttl {

struct ValueLiveInterval {
  int64_t start = 0;
  int64_t end = 0;
  Value value;
};

inline bool halfOpenIntervalsOverlap(int64_t lhsStart, int64_t lhsEnd,
                                     int64_t rhsStart, int64_t rhsEnd) {
  return lhsEnd > rhsStart && rhsEnd > lhsStart;
}

inline bool intervalsOverlap(const ValueLiveInterval &lhs,
                             const ValueLiveInterval &rhs) {
  return halfOpenIntervalsOverlap(lhs.start, lhs.end, rhs.start, rhs.end);
}

struct OperationLiveInterval {
  Operation *start = nullptr;
  Operation *end = nullptr;
  int64_t startOrdinal = std::numeric_limits<int64_t>::max();
  bool unbounded = false;
};

inline void updateIntervalStart(OperationLiveInterval &interval, Operation *op,
                                int64_t opOrdinal,
                                const DominanceInfo &dominanceInfo) {
  if (!interval.start || dominanceInfo.properlyDominates(op, interval.start)) {
    interval.start = op;
    interval.startOrdinal = opOrdinal;
    return;
  }
  if (!dominanceInfo.dominates(interval.start, op)) {
    interval.unbounded = true;
  }
}

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
    interval.unbounded = true;
  }
}

inline void finalizeInterval(OperationLiveInterval &interval, bool hasStart,
                             bool hasEnd, const DominanceInfo &dominanceInfo,
                             const PostDominanceInfo &postDominanceInfo) {
  if (!hasStart || !hasEnd || !interval.start || !interval.end) {
    interval.unbounded = true;
    return;
  }

  if (!dominanceInfo.dominates(interval.start, interval.end) ||
      !postDominanceInfo.postDominates(interval.end, interval.start)) {
    interval.unbounded = true;
  }
}

inline bool intervalsOverlap(const OperationLiveInterval &lhs,
                             const OperationLiveInterval &rhs,
                             const DominanceInfo &dominanceInfo) {
  if (lhs.unbounded || rhs.unbounded || !lhs.start || !lhs.end || !rhs.start ||
      !rhs.end) {
    return true;
  }
  return !(dominanceInfo.properlyDominates(lhs.end, rhs.start) ||
           dominanceInfo.properlyDominates(rhs.end, lhs.start));
}

/// Deterministic first-fit coloring. Callers supply ordering and conflict
/// rules because TTL resources have different interval construction semantics.
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
