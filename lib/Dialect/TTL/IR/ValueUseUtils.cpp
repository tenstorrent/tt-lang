// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/Utils/ValueUseUtils.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

namespace mlir::tt::ttl::utils {

Value getAttachedCB(Value tensor) {
  if (auto attach = tensor.getDefiningOp<AttachCBOp>()) {
    return attach.getCb();
  }
  return Value();
}

} // namespace mlir::tt::ttl::utils
