// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_OPAQUECALLVERIFYUTILS_H
#define TTLANG_DIALECT_UTILS_OPAQUECALLVERIFYUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::utils {

/// Shared verifier for `ttl.opaque_call` / `ttkernel.opaque_call`.
///
/// Checks that callee and header are non-empty, and that each template arg
/// is defined by `arith.constant` or the dialect's `get_dfb_id` op.
template <typename GetDfbIdOpTy>
LogicalResult verifyOpaqueCall(Operation *op, StringRef callee,
                               StringRef header, ValueRange templateArgVals) {
  if (callee.empty()) {
    return op->emitOpError("callee name must not be empty");
  }
  if (header.empty()) {
    return op->emitOpError("header path must not be empty");
  }

  for (Value taVal : templateArgVals) {
    Operation *defOp = taVal.getDefiningOp();
    if (!defOp) {
      return op->emitOpError("template arg must be a compile-time evaluable "
                             "value (arith.constant or ")
             << GetDfbIdOpTy::getOperationName() << "), got a block argument";
    }
    if (!isa<arith::ConstantOp>(defOp) && !isa<GetDfbIdOpTy>(defOp)) {
      return op->emitOpError("template arg must be a compile-time evaluable "
                             "value (arith.constant or ")
             << GetDfbIdOpTy::getOperationName() << "), got '"
             << defOp->getName() << "'";
    }
  }
  return success();
}

} // namespace mlir::tt::utils

#endif // TTLANG_DIALECT_UTILS_OPAQUECALLVERIFYUTILS_H
