// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H
#define TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

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

/// Return the element type for a ttcore::TileType.
inline std::optional<mlir::Type> getTileElementType(mlir::Type type) {
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(type)) {
    return tileType.getElementType();
  }
  return std::nullopt;
}

/// Return the circular buffer attached to `tensor`, or null if none/ambiguous.
///
/// Traces through ViewLikeOpInterface (cb_reserve, cb_wait),
/// tensor.extract_slice, tensor.extract, unrealized_conversion_cast,
/// and attach_cb to find the underlying CB value.
inline mlir::Value getAttachedCB(mlir::Value tensor) {
  // Trace through unrealized conversion casts (from dialect conversion).
  tensor = traceUnrealizedCasts(tensor);

  // Trace through tensor.extract_slice (from compute subblocking).
  if (auto slice = tensor.getDefiningOp<mlir::tensor::ExtractSliceOp>()) {
    return getAttachedCB(slice.getSource());
  }

  // Trace through tensor.extract (scalar element extraction, e.g. bcast
  // output).
  if (auto extract = tensor.getDefiningOp<mlir::tensor::ExtractOp>()) {
    return getAttachedCB(extract.getTensor());
  }

  if (auto attach = tensor.getDefiningOp<mlir::tt::ttl::AttachCBOp>()) {
    return attach.getCb();
  }

  // Trace through ViewLikeOpInterface: cb_reserve and cb_wait return
  // the CB directly as their view source.
  if (auto viewLike = tensor.getDefiningOp<mlir::ViewLikeOpInterface>()) {
    mlir::Value source = viewLike.getViewSource();
    if (mlir::isa<CircularBufferType>(source.getType())) {
      return source;
    }
    return getAttachedCB(source);
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
  return op->hasTrait<TTLUnaryElementwiseOpTrait>();
}

/// Check if an operation is a binary elementwise tensor op.
inline bool isBinaryElementwiseOp(mlir::Operation *op) {
  return op->hasTrait<TTLBinaryElementwiseOpTrait>();
}

/// Check if an operation is a tile-level unary op (executes in-place on DST).
inline bool isTileUnaryOp(mlir::Operation *op) {
  return op->hasTrait<TTLTileUnaryOpTrait>();
}

/// Check if an operation is a tile-level binary op (writes to fresh DST slot).
inline bool isTileBinaryOp(mlir::Operation *op) {
  return op->hasTrait<TTLTileBinaryOpTrait>();
}

/// Check if an operation reads inputs from CB at runtime, either by static
/// trait (bcast, copy_tile) or by runtime FPU binary marking.
inline bool isCBInputOp(mlir::Operation *op) {
  return op->hasTrait<TTLCBInputTileOpTrait>() ||
         op->hasAttr(kFPUBinaryAttrName);
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

//===----------------------------------------------------------------------===//
// Tile operation categories for scheduling and init consolidation
//===----------------------------------------------------------------------===//

/// Operation categories for scheduling and init consolidation.
/// Sort order matters: lower values are scheduled first within sync regions.
/// CB-input ops that configure MATH (bcast, transpose) must precede copy_tile
/// because their pipeline configuration can be disrupted by intervening ops.
enum class TileOpCategory : uint8_t {
  Bcast = 0,      // CB -> DST with PACK config (full init, must be first)
  Transpose = 1,  // CB -> DST transpose (full init, requires uninit)
  CopyTile = 2,   // CB -> DST copy (simple passthrough)
  FPUBinary = 3,  // CB -> DST FPU (UNPACK+MATH init)
  SFPUUnary = 4,  // DST -> DST in-place (MATH-only init)
  SFPUBinary = 5, // DST -> DST binary (MATH-only init)
  CopyDst = 6,    // DST -> DST copy
  Unknown = 255
};

/// Classify a TTL tile op into its category.
/// Uses TTL traits and attributes for O(1) per-call classification.
TileOpCategory classifyTileOp(mlir::Operation *op);

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

//===----------------------------------------------------------------------===//
// Iter index utilities for CB tile indexing
//===----------------------------------------------------------------------===//

/// Get or create iter_index ops at the start of a compute body. Returns
/// one Value per iteration domain dimension. Reuses existing iter_index ops
/// if present (idempotent across multiple callers).
inline SmallVector<Value> getOrCreateIterIndices(OpBuilder &builder,
                                                 ComputeOp computeOp) {
  Block &body = computeOp.getBody().front();
  unsigned iterRank = computeOp.getIteratorTypesArray().size();

  SmallVector<Value> existing(iterRank, Value());
  for (Operation &op : body) {
    if (auto iterIdx = dyn_cast<IterIndexOp>(&op)) {
      unsigned dim = static_cast<unsigned>(iterIdx.getDim());
      if (dim < iterRank && !existing[dim]) {
        existing[dim] = iterIdx.getResult();
      }
    }
  }
  if (llvm::none_of(existing, [](Value v) { return !v; })) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&body);
  Location loc = computeOp.getLoc();
  for (unsigned d = 0; d < iterRank; ++d) {
    if (!existing[d]) {
      existing[d] = IterIndexOp::create(builder, loc, d);
    }
  }
  return existing;
}

/// Apply an indexing map to iter_index values to produce operand-space
/// coordinates. For projected permutations this folds to a subset of
/// iter_index values with no extra ops.
inline SmallVector<Value>
applyIndexingMapToIterIndices(OpBuilder &builder, Location loc, AffineMap map,
                              ValueRange iterIndices) {
  SmallVector<OpFoldResult> operands(iterIndices.begin(), iterIndices.end());
  SmallVector<Value> mapped;
  mapped.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    AffineMap singleMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr);
    OpFoldResult result = affine::makeComposedFoldedAffineApply(
        builder, loc, singleMap, operands);
    mapped.push_back(
        mlir::getValueOrCreateConstantIndexOp(builder, loc, result));
  }
  return mapped;
}

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H
