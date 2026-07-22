// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "TTLOpsVerifyUtils.h"

#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttl::verify {

mlir::LogicalResult verifyWaitOperandType(mlir::Operation *op,
                                          mlir::Value handle) {
  // Accept typed and untyped transfer handles. Untyped handles model async
  // pipe receive completion and are expanded before lowering.
  if (!mlir::isa<mlir::tt::ttl::TransferHandleType>(handle.getType())) {
    return op->emitOpError()
           << "expects transfer handle (!ttl.transfer_handle), got "
           << handle.getType();
  }

  return mlir::success();
}

} // namespace mlir::tt::ttl::verify
