// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_LIB_DIALECT_TTL_IR_TTLOPSVERIFYUTILS_H
#define TTLANG_LIB_DIALECT_TTL_IR_TTLOPSVERIFYUTILS_H

#include "mlir/IR/Value.h"

namespace mlir::tt::ttl::verify {

/// Verify the local operand-type requirement for `ttl.wait`.
/// Transfer provenance is validated after control-flow construction.
mlir::LogicalResult verifyWaitOperandType(mlir::Operation *op,
                                          mlir::Value handle);

} // namespace mlir::tt::ttl::verify

#endif // TTLANG_LIB_DIALECT_TTL_IR_TTLOPSVERIFYUTILS_H
