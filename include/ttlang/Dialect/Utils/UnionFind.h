// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_UNIONFIND_H
#define TTLANG_DIALECT_UTILS_UNIONFIND_H

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstddef>
#include <numeric>

namespace mlir::tt::utils {

/// Union-find over the contiguous indices `[0, elementCount)`.
class IndexUnionFind {
public:
  explicit IndexUnionFind(std::size_t elementCount) : parents(elementCount) {
    std::iota(parents.begin(), parents.end(), 0);
  }

  /// Merge the sets containing `lhsIndex` and `rhsIndex`.
  void merge(std::size_t lhsIndex, std::size_t rhsIndex) {
    std::size_t lhsRoot = findRepresentative(lhsIndex);
    std::size_t rhsRoot = findRepresentative(rhsIndex);
    if (lhsRoot != rhsRoot) {
      parents[rhsRoot] = lhsRoot;
    }
  }

  /// Return the representative index for every element.
  llvm::SmallVector<std::size_t> getRepresentatives() {
    llvm::SmallVector<std::size_t> representatives;
    representatives.reserve(parents.size());
    for (std::size_t elementIndex = 0; elementIndex < parents.size();
         ++elementIndex) {
      representatives.push_back(findRepresentative(elementIndex));
    }
    return representatives;
  }

private:
  std::size_t findRepresentative(std::size_t elementIndex) {
    assert(elementIndex < parents.size() && "union-find index out of range");
    std::size_t rootIndex = elementIndex;
    while (parents[rootIndex] != rootIndex) {
      rootIndex = parents[rootIndex];
    }
    while (parents[elementIndex] != elementIndex) {
      std::size_t parentIndex = parents[elementIndex];
      parents[elementIndex] = rootIndex;
      elementIndex = parentIndex;
    }
    return rootIndex;
  }

  llvm::SmallVector<std::size_t> parents;
};

} // namespace mlir::tt::utils

#endif // TTLANG_DIALECT_UTILS_UNIONFIND_H
