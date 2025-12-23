// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H
#define TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

namespace mlir::tt::ttl {

/// Trace through unrealized conversion casts to find the original value.
/// This is useful during dialect conversion when values are wrapped in
/// UnrealizedConversionCastOp to represent type conversions.
inline mlir::Value traceUnrealizedCasts(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() == 1) {
      value = cast.getInputs()[0];
    } else {
      break;
    }
  }
  return value;
}

/// Return the circular buffer attached to `tensor`, or null if none/ambiguous.
///
/// Recognized producers:
/// - `ttl.attach_cb`: explicit association between a tensor SSA value and a CB.
/// - `ttl.cb_wait`: returns a tensor view backed by the CB's pages.
/// - `unrealized_conversion_cast`: trace through to find the original producer.
///
/// Both operations establish a tensor->CB association for compute/DMA purposes.
inline mlir::Value getAttachedCB(mlir::Value tensor) {
  // Trace through unrealized conversion casts (from dialect conversion).
  tensor = traceUnrealizedCasts(tensor);

  if (auto attach = tensor.getDefiningOp<mlir::tt::ttl::AttachCBOp>()) {
    return attach.getCb();
  }
  if (auto wait = tensor.getDefiningOp<mlir::tt::ttl::CBWaitOp>()) {
    return wait.getCb();
  }
  return mlir::Value();
}

/// Check if an operation is a tile compute operation.
/// Returns true for arithmetic/math tile operations (add, mul, exp, etc.).
/// Excludes data movement ops (copy_tile) and DST lifecycle ops.
inline bool isTileComputeOp(mlir::Operation *op) {
  return op->hasTrait<TTLTileComputeOpTrait>();
}

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H
