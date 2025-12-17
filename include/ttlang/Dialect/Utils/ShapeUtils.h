// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_SHAPEUTILS_H
#define TTLANG_DIALECT_UTILS_SHAPEUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include <numeric>

namespace mlir::tt::ttl::utils {

/// Return total element count for a shaped type if it has a static shape.
/// Returns std::nullopt for dynamic shapes.
inline std::optional<int64_t> getTotalElements(mlir::ShapedType shapedTy) {
  if (!shapedTy.hasStaticShape()) {
    return std::nullopt;
  }
  return std::accumulate(shapedTy.getShape().begin(), shapedTy.getShape().end(),
                         int64_t{1}, std::multiplies<int64_t>());
}

/// Return total element count for a static shape array.
inline int64_t getTotalElements(llvm::ArrayRef<int64_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

} // namespace mlir::tt::ttl::utils

#endif // TTLANG_DIALECT_UTILS_SHAPEUTILS_H
