// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

namespace mlir::tt::ttl {

//===----------------------------------------------------------------------===//
// Tile operation classification
//===----------------------------------------------------------------------===//

TileOpCategory classifyTileOp(Operation *op) {
  if (isa<CopyTileOp>(op)) {
    return TileOpCategory::CopyTile;
  }
  if (isa<CopyDstOp>(op)) {
    return TileOpCategory::CopyDst;
  }
  if (isa<TileBcastOp>(op)) {
    return TileOpCategory::Bcast;
  }
  // TODO: add TileOpCategory::Transpose case when TTL transpose op is added.

  // FPU binary: marked by kFPUBinaryAttrName attribute.
  if (op->hasAttr(kFPUBinaryAttrName)) {
    return TileOpCategory::FPUBinary;
  }
  // SFPU unary: tile unary ops that operate in-place on DST.
  if (op->hasTrait<TTLTileUnaryOpTrait>()) {
    return TileOpCategory::SFPUUnary;
  }
  // SFPU binary: tile binary ops that read both operands from DST.
  if (op->hasTrait<TTLTileBinaryOpTrait>()) {
    return TileOpCategory::SFPUBinary;
  }
  return TileOpCategory::Unknown;
}

ElementwiseTraceResult traceElementwiseToRoots(mlir::Value value) {
  ElementwiseTraceResult result;

  // Base case: CB-attached value is a root
  if (getAttachedCB(value)) {
    result.rootInputs.insert(value);
    return result;
  }

  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = value;
    return result;
  }

  // Special case: BcastOp can be fused when its input is CB-attached.
  // Bcast reads from CB and writes to DST, so it can be the first op
  // in a fused compute block.
  if (auto bcastOp = llvm::dyn_cast<BcastOp>(defOp)) {
    mlir::Value bcastInput = bcastOp.getInput();
    if (getAttachedCB(bcastInput)) {
      result.rootInputs.insert(bcastInput);
      result.opsInOrder.insert(defOp);
      return result;
    }
    // Input not CB-attached - fall through to failure
  }

  if (!isElementwiseOp(defOp)) {
    result.failureReason = TraceFailureReason::NotElementwiseOp;
    result.failedValue = value;
    return result;
  }

  // Recursively trace all operands
  for (mlir::Value operand : getElementwiseOperands(defOp)) {
    auto operandTrace = traceElementwiseToRoots(operand);
    if (operandTrace.failureReason != TraceFailureReason::Success) {
      return operandTrace;
    }
    // Merge roots and ops (SmallSetVector handles deduplication)
    for (mlir::Value root : operandTrace.rootInputs) {
      result.rootInputs.insert(root);
    }
    for (mlir::Operation *op : operandTrace.opsInOrder) {
      result.opsInOrder.insert(op);
    }
  }

  // Add this op at the end (after all its dependencies)
  result.opsInOrder.insert(defOp);

  return result;
}

void emitFusionFailureDiagnostics(mlir::Operation *op,
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

} // namespace mlir::tt::ttl
