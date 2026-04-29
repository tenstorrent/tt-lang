// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_LIB_DIALECT_TTL_IR_TTLOPSVERIFYUTILS_H
#define TTLANG_LIB_DIALECT_TTL_IR_TTLOPSVERIFYUTILS_H

#include "mlir/IR/Value.h"

namespace mlir::tt::ttl::verify {

/// Return success if `handle` is a valid operand for `ttl.wait`.
///
/// In the current MVP, `ttl.wait` must synchronize a transfer handle
/// originating from `ttl.copy`. This helper also allows handles forwarded
/// through loop-carried state and tensor containers.
mlir::LogicalResult isValidWaitOperand(mlir::Operation *op, mlir::Value handle);

} // namespace mlir::tt::ttl::verify

#endif // TTLANG_LIB_DIALECT_TTL_IR_TTLOPSVERIFYUTILS_H
