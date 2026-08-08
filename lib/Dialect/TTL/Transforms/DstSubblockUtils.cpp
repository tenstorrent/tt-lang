// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/DstSubblockUtils.h"

#include <functional>

namespace mlir::tt::ttl {

SmallVector<int64_t> computeMultiDimSubblockSizes(ArrayRef<int64_t> dimSizes,
                                                  int64_t maxTiles) {
  int64_t rank = dimSizes.size();

  // Collect divisors per dimension (sorted descending for early pruning).
  SmallVector<SmallVector<int64_t>> allDivisors(rank);
  for (int64_t d = 0; d < rank; ++d) {
    for (int64_t i = dimSizes[d]; i >= 1; --i) {
      if (dimSizes[d] % i == 0) {
        allDivisors[d].push_back(i);
      }
    }
  }

  SmallVector<int64_t> bestSizes(rank, 1);
  int64_t bestProduct = 1;
  SmallVector<int64_t> current(rank, 1);

  // Return true if `a` should be preferred over `b` when products are equal.
  // Prefers larger inner (higher-index) dimensions to minimize outer loops.
  auto prefersInner = [&](ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
    for (int64_t d = rank - 1; d >= 0; --d) {
      if (a[d] != b[d]) {
        return a[d] > b[d];
      }
    }
    return false;
  };

  // Recursive brute-force search with pruning.
  std::function<void(int64_t, int64_t)> search;
  search = [&](int64_t dim, int64_t currentProduct) {
    if (dim == rank) {
      // All dimensions have been assigned. Update best if this candidate
      // has a larger product, or the same product but larger inner dimensions.
      if (currentProduct > bestProduct ||
          (currentProduct == bestProduct && prefersInner(current, bestSizes))) {
        bestProduct = currentProduct;
        bestSizes = current;
      }
      return;
    }
    for (int64_t divisor : allDivisors[dim]) {
      int64_t newProduct = currentProduct * divisor;
      if (newProduct > maxTiles) {
        continue;
      }
      current[dim] = divisor;
      search(dim + 1, newProduct);
    }
  };

  search(0, 1);
  return bestSizes;
}

} // namespace mlir::tt::ttl
