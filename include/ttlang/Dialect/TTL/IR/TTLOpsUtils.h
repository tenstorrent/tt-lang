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

/// Check if an operation is a unary elementwise tensor op.
inline bool isUnaryElementwiseOp(mlir::Operation *op) {
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)               \
  if (mlir::isa<TTL_OP##Op>(op))                                               \
    return true;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"
  return false;
}

/// Check if an operation is a binary elementwise tensor op.
inline bool isBinaryElementwiseOp(mlir::Operation *op) {
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  if (mlir::isa<TTL_OP##Op>(op))                                               \
    return true;
#define TTL_BINARY_TILE_OP_SPECIAL(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"
  return false;
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
  NoDefiningOp,
  NotElementwiseOp,
  MultipleUses,
};

/// Result of tracing through elementwise ops to CB-attached roots.
struct ElementwiseTraceResult {
  /// CB-attached input values that form the roots of the chain.
  mlir::SmallVector<mlir::Value, 2> rootInputs;
  /// Operations in the chain, topologically ordered (roots first, sink last).
  mlir::SmallVector<mlir::Operation *, 4> opsInOrder;
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
inline ElementwiseTraceResult traceElementwiseToRoots(mlir::Value value) {
  ElementwiseTraceResult result;

  // Base case: CB-attached value is a root
  if (getAttachedCB(value)) {
    result.rootInputs.push_back(value);
    return result;
  }

  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = value;
    return result;
  }

  if (!isElementwiseOp(defOp)) {
    result.failureReason = TraceFailureReason::NotElementwiseOp;
    result.failedValue = value;
    return result;
  }

  // Reject if this op's result has multiple uses (would break SSA semantics)
  if (!value.hasOneUse()) {
    result.failureReason = TraceFailureReason::MultipleUses;
    result.failedValue = value;
    return result;
  }

  // Recursively trace all operands
  for (mlir::Value operand : getElementwiseOperands(defOp)) {
    auto operandTrace = traceElementwiseToRoots(operand);
    if (operandTrace.failureReason != TraceFailureReason::Success) {
      return operandTrace;
    }
    // Merge roots (avoiding duplicates)
    for (mlir::Value root : operandTrace.rootInputs) {
      if (!llvm::is_contained(result.rootInputs, root)) {
        result.rootInputs.push_back(root);
      }
    }
    // Merge ops in dependency order
    for (mlir::Operation *op : operandTrace.opsInOrder) {
      if (!llvm::is_contained(result.opsInOrder, op)) {
        result.opsInOrder.push_back(op);
      }
    }
  }

  // Add this op at the end (after all its dependencies)
  result.opsInOrder.push_back(defOp);

  return result;
}

/// Emit diagnostics explaining why elementwise fusion failed.
inline void emitFusionFailureDiagnostics(mlir::Operation *op,
                                         const ElementwiseTraceResult &trace) {
  mlir::Value v = trace.failedValue;
  switch (trace.failureReason) {
  case TraceFailureReason::Success:
    break;
  case TraceFailureReason::NotCBAttached:
    if (v) {
      op->emitError("fusion failed: value is not attached to a circular buffer")
          .attachNote(v.getLoc())
          << "this value (block argument) needs ttl.cb_wait or ttl.attach_cb";
    }
    break;
  case TraceFailureReason::NoDefiningOp:
    op->emitError("fusion failed: cannot trace through block argument");
    break;
  case TraceFailureReason::NotElementwiseOp:
    if (v && v.getDefiningOp()) {
      op->emitError("fusion failed: cannot trace through non-elementwise op")
          .attachNote(v.getDefiningOp()->getLoc())
          << "this op '" << v.getDefiningOp()->getName() << "' is not fusable";
    }
    break;
  case TraceFailureReason::MultipleUses:
    if (v && v.getDefiningOp()) {
      op->emitError("fusion failed: intermediate value has multiple uses")
          .attachNote(v.getDefiningOp()->getLoc())
          << "this op's result is used multiple times";
    }
    break;
  }
}

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
