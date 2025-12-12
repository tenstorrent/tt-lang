// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_LIB_DIALECT_TTL_IR_TTLOPSVERIFYUTILS_H
#define TTLANG_LIB_DIALECT_TTL_IR_TTLOPSVERIFYUTILS_H

#include "mlir/IR/Value.h"

namespace mlir::tt::ttl::verify {

/// Return success if `handle` is eventually synchronized with a `ttl.wait` in
/// the IR use graph.
///
/// This check is used to enforce the TTL DMA MVP rule that every `ttl.copy`
/// transfer handle must be synchronized before it is dropped.
///
/// The analysis is conservative and follows common TT-Metal patterns where
/// copies and waits are separated:
/// - Direct use by `ttl.wait`.
/// - Loop-carried forwarding via LoopLikeOpInterface init operands and
/// iter_args.
/// - Loop result forwarding via `scf.yield` and LoopLikeOpInterface loop
/// results.
/// - Container forwarding via `tensor.insert` and `tensor.extract` (for
/// batching handles and waiting in a different location).
///
/// Note: Relies on `Value::getUses()`, which includes uses across blocks.
mlir::LogicalResult isEventuallyWaitedOn(mlir::Operation *op,
                                         mlir::Value handle);

/// Return success if `handle` is a valid operand for `ttl.wait`.
///
/// In the current MVP, `ttl.wait` must synchronize a transfer handle
/// originating from `ttl.copy`. This helper also allows handles forwarded
/// through the same mechanisms as isEventuallyWaitedOn (loop-carried state and
/// tensor containers).
mlir::LogicalResult isValidWaitOperand(mlir::Operation *op, mlir::Value handle);

} // namespace mlir::tt::ttl::verify

#endif // TTLANG_LIB_DIALECT_TTL_IR_TTLOPSVERIFYUTILS_H
