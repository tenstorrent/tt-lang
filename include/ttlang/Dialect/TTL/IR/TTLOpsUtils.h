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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"

#include <cstdint>
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

/// Reason why fusion tracing failed.
enum class TraceFailureReason {
  Success,
  NotCBAttached,
  NotFusableOp,
};

/// Result of tracing through fusable ops to CB-attached roots.
struct FusionTraceResult {
  /// CB-attached input values that form the roots of the chain.
  llvm::SmallSetVector<mlir::Value, 2> rootInputs;
  /// Operations in the chain, topologically ordered (roots first, sink last).
  llvm::SmallSetVector<mlir::Operation *, 4> opsInOrder;
  /// Failure reason (Success if tracing succeeded).
  TraceFailureReason failureReason = TraceFailureReason::Success;
  /// The value where tracing failed (only set on failure).
  mlir::Value failedValue;
};

/// Trace a value through fusable ops (elementwise, matmul, bcast) to find
/// CB-attached roots. Recursively traces through arbitrary depth chains.
///
/// On failure, sets failureReason and failedValue in the result.
/// Check failureReason == TraceFailureReason::Success to determine success.
FusionTraceResult traceFusionToRoots(mlir::Value value);

/// Return a human-readable description of a trace failure reason.
llvm::StringRef describeTraceFailure(TraceFailureReason reason);

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

/// Apply an indexing map to induction variables, producing index-typed
/// Values via affine composition and folding.
inline SmallVector<Value> applyIndexingMap(OpBuilder &builder, Location loc,
                                           AffineMap map, ValueRange ivs) {
  SmallVector<OpFoldResult> operands(ivs.begin(), ivs.end());
  assert(operands.size() == map.getNumDims() &&
         "IV count must match map dimensions");

  SmallVector<Value> mapped;
  mapped.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    AffineMap singleResultMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr);
    OpFoldResult result = affine::makeComposedFoldedAffineApply(
        builder, loc, singleResultMap, operands);
    mapped.push_back(getValueOrCreateConstantIndexOp(builder, loc, result));
  }
  return mapped;
}

/// Trace a value through copy_tile (inserted by assign-dst) to its source
/// block argument. Returns the block arg index, or std::nullopt if the value
/// does not trace to a block argument.
inline std::optional<unsigned> traceToBlockArgIndex(Value val) {
  if (auto copyOp = val.getDefiningOp<CopyTileOp>()) {
    val = copyOp.getSrc();
  }
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    return blockArg.getArgNumber();
  }
  return std::nullopt;
}

/// Extract tiles from tensors by applying indexing maps to induction variables.
/// Returns one extracted tile per tensor.
inline SmallVector<Value>
extractTilesAtIndices(OpBuilder &builder, Location loc, ValueRange tensors,
                      ArrayRef<AffineMap> indexingMaps, ValueRange ivs,
                      size_t mapOffset = 0) {
  SmallVector<Value> extracted;
  extracted.reserve(tensors.size());
  for (auto [idx, tensor] : llvm::enumerate(tensors)) {
    SmallVector<Value> indices =
        applyIndexingMap(builder, loc, indexingMaps[mapOffset + idx], ivs);
    extracted.push_back(
        tensor::ExtractOp::create(builder, loc, tensor, indices));
  }
  return extracted;
}

/// Map a ComputeOp's body block arguments to extracted tile values.
/// Also maps iter_index results to the corresponding IV values.
inline void mapComputeBodyArgs(IRMapping &mapping, ComputeOp op,
                               ArrayRef<Value> extractedInputs,
                               ArrayRef<Value> extractedOutputs,
                               ValueRange ivs) {
  Block &bodyBlock = op.getBody().front();
  size_t numInputs = op.getInputs().size();
  for (auto [idx, input] : llvm::enumerate(extractedInputs)) {
    mapping.map(bodyBlock.getArgument(idx), input);
  }
  for (auto [idx, output] : llvm::enumerate(extractedOutputs)) {
    mapping.map(bodyBlock.getArgument(numInputs + idx), output);
  }
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (auto iterIdx = dyn_cast<IterIndexOp>(&bodyOp)) {
      mapping.map(iterIdx.getResult(), ivs[iterIdx.getDim()]);
    }
  }
}

//===----------------------------------------------------------------------===//
// Live interval for register/resource allocation
//===----------------------------------------------------------------------===//

/// A live interval representing the lifetime of a value or resource.
/// Used by linear scan allocators in TTLAssignDST and TTLFinalizeDFBIndices.
struct Interval {
  int64_t start; // Operation index where value becomes live
  int64_t end;   // Operation index of last use
  Value value;   // SSA value this interval represents
};

//===----------------------------------------------------------------------===//
// DST capacity computation
//===----------------------------------------------------------------------===//

/// Physical DST register size in tiles (constant across all architectures).
constexpr std::uint32_t kDstPhysicalSizeTiles = 16;

/// Compute the logical DST capacity based on element types and sync mode.
///
/// The DST register file has 16 physical tiles. Logical capacity is derived:
///   - Default (double-buffered): physical / 2 = 8 tiles
///   - Full sync (dst_full_sync_en): no halving = 16 tiles
///   - f32 accumulation: halved again (tiles are 2x wider)
///
/// Architecture-independent across Grayskull, Wormhole, and Blackhole.
inline std::uint32_t getDstCapacity(bool isFloat32, bool fullSyncEn) {
  std::uint32_t capacity = kDstPhysicalSizeTiles;
  if (!fullSyncEn) {
    capacity /= 2; // Double-buffering halves available tiles.
  }
  if (isFloat32) {
    capacity /= 2; // f32 tiles occupy 2x the space.
  }
  return capacity;
}

/// Read a per-kernel bool attribute from the enclosing func.func.
/// These are set by TTLSetComputeKernelConfig on the function (not on
/// individual compute ops) because they are per-kernel compile-time settings.
/// Returns false when the attribute is absent or the op has no enclosing func.
inline bool getKernelBoolAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
  assert(funcOp && "getKernelBoolAttr called on op outside of func.func");
  if (auto attr = funcOp->getAttrOfType<mlir::BoolAttr>(attrName)) {
    return attr.getValue();
  }
  return false;
}

/// Compute DST capacity for a compute op by inspecting block argument types
/// and the kernel-level fp32/sync-mode attributes on the enclosing function.
/// Returns failure for mixed f32/non-f32 tile arguments.
inline FailureOr<std::uint32_t> computeDSTCapacity(ComputeOp computeOp) {
  bool fullSyncEn = getKernelBoolAttr(computeOp, kDstFullSyncEnAttrName);
  bool fp32DestAccEn = getKernelBoolAttr(computeOp, kFp32DestAccEnAttrName);

  bool sawF32 = false;
  bool sawNonF32 = false;
  Block &body = computeOp.getRegion().front();
  for (BlockArgument arg : body.getArguments()) {
    std::optional<Type> currentType = getTileElementType(arg.getType());
    if (currentType) {
      if (currentType->isF32()) {
        sawF32 = true;
      } else {
        sawNonF32 = true;
      }
    }
  }

  if (sawF32) {
    fp32DestAccEn = true;
  }

  if (sawF32 && sawNonF32) {
    return computeOp.emitOpError(
        "mixed f32 and non-f32 tile arguments; "
        "DST capacity uses f32 limits (4 tiles) which may produce "
        "incorrect results");
  }

  bool isFloat32 = sawF32 || fp32DestAccEn;
  return getDstCapacity(isFloat32, fullSyncEn);
}

/// Fold a Value through constant integer arithmetic to resolve its value.
/// Handles arith.constant directly, and recursively folds arith.addi and
/// arith.muli where both operands are foldable.
inline std::optional<int64_t> foldIndexToConstant(Value val) {
  if (auto constIdx = getConstantIntValue(val)) {
    return constIdx;
  }
  auto *defOp = val.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }
  auto foldBinOp = [](auto binOp, auto combine) -> std::optional<int64_t> {
    auto lhs = foldIndexToConstant(binOp.getLhs());
    auto rhs = foldIndexToConstant(binOp.getRhs());
    if (lhs && rhs) {
      return combine(*lhs, *rhs);
    }
    return std::nullopt;
  };
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    return foldBinOp(addOp, std::plus<int64_t>{});
  }
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    return foldBinOp(mulOp, std::multiplies<int64_t>{});
  }
  return std::nullopt;
}

/// Get the dst_index Value from a tile op, or std::nullopt if the op
/// does not have TTLDstResultOpTrait.
inline std::optional<Value> getTileOpDstIndex(Operation *op) {
  if (op->hasTrait<TTLDstResultOpTrait>()) {
    return op->getOperand(op->getNumOperands() - 1);
  }
  return std::nullopt;
}

/// Set the dst_index Value on a tile op with TTLDstResultOpTrait.
inline void setTileOpDstIndex(Operation *op, Value newDstIndex) {
  assert(op->hasTrait<TTLDstResultOpTrait>() &&
         "setTileOpDstIndex called on op without TTLDstResultOpTrait");
  op->setOperand(op->getNumOperands() - 1, newDstIndex);
}

/// Temporary marker attribute for tile ops whose dst_index has not been
/// assigned yet.
constexpr llvm::StringLiteral kDstPlaceholderAttrName("ttl.dst_placeholder");

/// Sentinel value for unassigned DST indices. Obviously invalid so any
/// accidental use fails loudly at TTKernel lowering or on hardware.
constexpr int64_t kUnassignedDstIndex = -1;

/// Create a placeholder dst_index constant (-1).
inline Value createPlaceholderDstIndex(OpBuilder &builder, Location loc) {
  auto constant =
      arith::ConstantIndexOp::create(builder, loc, kUnassignedDstIndex);
  return constant.getResult();
}

/// Mark a tile op as having an unassigned dst_index.
inline void addPlaceholderDstIndexAttr(Operation *op) {
  op->setAttr(kDstPlaceholderAttrName, UnitAttr::get(op->getContext()));
}

/// Create a tile op with a placeholder dst_index and mark the op.
template <typename TileOp, typename... Args>
inline TileOp createTileOpWithPlaceholderDstIndex(OpBuilder &builder,
                                                  Location loc,
                                                  Args &&...args) {
  Value dstIndex = createPlaceholderDstIndex(builder, loc);
  TileOp tileOp =
      TileOp::create(builder, loc, std::forward<Args>(args)..., dstIndex);
  addPlaceholderDstIndexAttr(tileOp.getOperation());
  return tileOp;
}

/// Collect the CB values targeted by pack_tile ops inside a loop.
llvm::SmallDenseSet<Value, 2> getPackTileCBs(scf::ForOp loop);

/// Returns true if two loops share any pack_tile CB target.
bool sharePackCB(scf::ForOp loopA, scf::ForOp loopB);

/// A group of consecutive sibling loops that pack to the same output CB.
struct LoopGroup {
  scf::ForOp rootLoop;
  SmallVector<scf::ForOp> loops;
  Operation *scopeEnd = nullptr;
};

/// Collect groups of annotated sibling loops that share a pack CB target.
SmallVector<LoopGroup> collectLoopGroups(
    ArrayRef<scf::ForOp> l1AccLoops,
    const llvm::SmallDenseMap<Operation *, Operation *> &enablePointPerLoop);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSUTILS_H
