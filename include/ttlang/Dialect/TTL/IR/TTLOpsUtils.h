// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H
#define TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "llvm/ADT/SetVector.h"

namespace mlir::tt::ttl {

/// Trace through unrealized conversion casts to find the original value.
/// This is useful during dialect conversion when values are wrapped in
/// UnrealizedConversionCastOp to represent type conversions.
///
/// Includes cycle detection because buggy conversion patterns can create cast
/// cycles (see MLIR's reconcileUnrealizedCastsImpl for similar checks).
inline mlir::Value traceUnrealizedCasts(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (!visited.insert(cast).second) {
      // Cycle detected - return current value to avoid infinite loop
      break;
    }
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

inline bool isTileUnaryOp(mlir::Operation *op) {
  return op->hasTrait<TTLTileUnaryOpTrait>();
}

inline bool isTileBinaryOp(mlir::Operation *op) {
  return op->hasTrait<TTLTileBinaryOpTrait>();
}

/// Check if an operation is a unary elementwise tensor op.
inline bool isUnaryElementwiseOp(mlir::Operation *op) {
  return op->hasTrait<TTLUnaryElementwiseOpTrait>();
}

/// Check if an operation is a binary elementwise tensor op.
inline bool isBinaryElementwiseOp(mlir::Operation *op) {
  return op->hasTrait<TTLBinaryElementwiseOpTrait>();
}

/// Check if an operation is any elementwise tensor op (unary or binary).
inline bool isElementwiseOp(mlir::Operation *op) {
  return isUnaryElementwiseOp(op) || isBinaryElementwiseOp(op);
}

/// Get the operands of an elementwise op (1 for unary, 2 for binary).
inline mlir::SmallVector<mlir::Value, 2>
getElementwiseOperands(mlir::Operation *op) {
  if (isUnaryElementwiseOp(op)) {
    return {op->getOperand(0)};
  }
  if (isBinaryElementwiseOp(op)) {
    return {op->getOperand(0), op->getOperand(1)};
  }
  return {};
}

/// Reason why elementwise tracing failed.
enum class TraceFailureReason {
  Success,
  NotCBAttached,
  NotElementwiseOp,
  MultipleUses,
};

/// Result of tracing through elementwise ops to CB-attached roots.
struct ElementwiseTraceResult {
  /// CB-attached input values that form the roots of the chain.
  llvm::SmallSetVector<mlir::Value, 2> rootInputs;
  /// Operations in the chain, topologically ordered (roots first, sink last).
  llvm::SmallSetVector<mlir::Operation *, 4> opsInOrder;
  /// Failure reason (Success if tracing succeeded).
  TraceFailureReason failureReason = TraceFailureReason::Success;
  /// The value where tracing failed (only set on failure).
  mlir::Value failedValue;
};

/// Trace a value through elementwise ops to find CB-attached roots.
/// Recursively traces through arbitrary depth elementwise chains.
///
/// On failure, sets failureReason and failedValue in the result.
/// Check failureReason == TraceFailureReason::Success to determine success.
ElementwiseTraceResult traceElementwiseToRoots(mlir::Value value);

/// Emit diagnostics explaining why elementwise fusion failed.
void emitFusionFailureDiagnostics(mlir::Operation *op,
                                  const ElementwiseTraceResult &trace);

/// Find the first operation of type OpTy in the block preceding the given
/// operation. Scans backwards from the operation, stopping at block start or
/// when stopAtOp returns true.
///
/// This is useful for finding control/sync operations that precede structured
/// ops (e.g., finding init_sfpu before ttl.compute).
template <typename OpTy, typename StopPredicate>
inline OpTy findPrecedingOp(mlir::Operation *op, StopPredicate stopAtOp) {
  mlir::Block *block = op->getBlock();
  if (!block) {
    return nullptr;
  }

  auto it = mlir::Block::iterator(op);
  if (it == block->begin()) {
    return nullptr;
  }

  for (auto revIt = mlir::Block::reverse_iterator(it); revIt != block->rend();
       ++revIt) {
    if (stopAtOp(&*revIt)) {
      break;
    }
    if (auto match = mlir::dyn_cast<OpTy>(&*revIt)) {
      return match;
    }
  }
  return nullptr;
}

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H
