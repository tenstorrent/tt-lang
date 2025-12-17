// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/Utils/ValueUseUtils.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

namespace mlir::tt::ttl {
class AttachCBOp;
}

namespace mlir::tt::ttl::utils {

Value getAttachedCB(Value tensor) {
  // If the tensor is defined by an attach op, return that CB directly.
  if (auto attach = tensor.getDefiningOp<AttachCBOp>()) {
    return attach.getCb();
  }

  // Otherwise, scan uses for a unique attach op that consumes this tensor.
  Value found;
  for (OpOperand &use : tensor.getUses()) {
    if (auto attach = dyn_cast<AttachCBOp>(use.getOwner())) {
      Value cb = attach.getCb();
      if (!found) {
        found = cb;
      } else if (found != cb) {
        // Ambiguous: multiple different CBs attached to the same tensor.
        return Value();
      }
    }
  }
  return found;
}

} // namespace mlir::tt::ttl::utils
