// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_VALUEUSEUTILS_H
#define TTLANG_DIALECT_UTILS_VALUEUSEUTILS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace mlir::tt::ttl::utils {

/// Collect all users of `value` that are of type `OpTy`.
///
/// Note: `Value::getUses()` iterates all uses, independent of block/region.
template <typename OpTy>
llvm::SmallVector<OpTy> getUsersOfType(mlir::Value value) {
  llvm::SmallVector<OpTy> users;
  for (mlir::OpOperand &use : value.getUses()) {
    if (auto op = llvm::dyn_cast<OpTy>(use.getOwner())) {
      users.push_back(op);
    }
  }
  return users;
}

template <typename OpTy>
bool hasUserOfType(mlir::Value value) {
  for (mlir::OpOperand &use : value.getUses()) {
    if (llvm::isa<OpTy>(use.getOwner())) {
      return true;
    }
  }
  return false;
}

/// Return the circular buffer attached to `tensor` via `ttl.attach_cb`, or null
/// if none/ambiguous.
mlir::Value getAttachedCB(mlir::Value tensor);

} // namespace mlir::tt::ttl::utils

#endif // TTLANG_DIALECT_UTILS_VALUEUSEUTILS_H
