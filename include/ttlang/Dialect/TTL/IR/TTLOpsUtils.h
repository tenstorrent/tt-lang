// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H
#define TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

namespace mlir::tt::ttl {

/// Return the circular buffer attached to `tensor` via `ttl.attach_cb`, or null
/// if none/ambiguous.
inline mlir::Value getAttachedCB(mlir::Value tensor) {
  if (auto attach = tensor.getDefiningOp<mlir::tt::ttl::AttachCBOp>()) {
    return attach.getCb();
  }
  return mlir::Value();
}

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H
