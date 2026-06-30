// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// LowerMatmulCompute
//===----------------------------------------------------------------------===//
//
// Lowers a ComputeOp containing tile_matmul_block into a single DstSectionOp
// with the matmul call, cloned body ops (elementwise, copy_tile, etc.), and
// per-output-view stores.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/Transforms/LowerMatmulCompute.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttl {

namespace {

enum class MatmulAccumulatorKind {
  None,
  InputTensor,
  BodySSA,
};

/// Classification decided before mutation so unsupported accumulators fail as
/// pattern mismatches instead of leaving partially lowered IR.
struct MatmulAccumulatorInfo {
  MatmulAccumulatorKind kind = MatmulAccumulatorKind::None;
  Value tensorAccumulator;
  Value bodyAccumulator;
};

/// Store emission is delayed until all math-phase ops have been emitted because
/// `ttl.dst_section` requires stores in the pack phase.
struct PendingStore {
  Value tile;
  Value view;
  SmallVector<Value> indices;
  Value dstIndex;
};

/// Non-mutating preflight for one block matmul compute. Shared by the
/// pass-level DST capacity precondition and the rewrite so both agree on the
/// per-output-tile slot accounting.
struct MatmulComputeAnalysis {
  TileMatmulBlockOp mmOp;
  int64_t numRows = 0;
  int64_t numCols = 0;
  Value lhsTensor;
  Value rhsTensor;
  MatmulAccumulatorInfo accumulatorInfo;
  SmallVector<Operation *> preMatmulOps;
  SmallVector<Operation *> postMatmulOps;
  SmallVector<TileStoreOp> stores;
  DenseMap<int64_t, int64_t> scratchIndexMap;
  int64_t dstPerIteration = 0;
};

/// Block matmul lowering expands body tiles from the original compute inputs;
/// accepting output block arguments here would describe a value TTKernel cannot
/// read through the matmul unpacker.
static FailureOr<Value> getInputTensorForBodyOperand(ComputeOp op,
                                                     Value bodyValue) {
  std::optional<unsigned> idx = traceToBlockArgIndex(bodyValue);
  if (!idx || *idx >= op.getInputs().size()) {
    return failure();
  }
  return op.getInputs()[*idx];
}

/// Read the assigned constant DST index of a tile op, or failure if it has no
/// `dst_index` or the index is not yet constant. Non-mutating so both the
/// capacity precondition and the rewrite can call it.
static FailureOr<int64_t> getAssignedDstIndex(Operation *tileOp) {
  std::optional<Value> dstIndex = getTileOpDstIndex(tileOp);
  if (!dstIndex) {
    return failure();
  }
  std::optional<int64_t> constIndex = foldIndexToConstant(*dstIndex);
  if (!constIndex) {
    return failure();
  }
  return *constIndex;
}

/// Read a constant DST index after `getAssignedDstIndex` has already accepted
/// the operation.
static int64_t getRequiredAssignedDstIndex(Operation *tileOp) {
  std::optional<Value> dstIndex = getTileOpDstIndex(tileOp);
  assert(dstIndex && "tile op was prevalidated to have dst_index");
  std::optional<int64_t> constIndex = foldIndexToConstant(*dstIndex);
  assert(constIndex && "tile op was prevalidated to have a constant dst_index");
  return *constIndex;
}

/// Restrict body SSA accumulators to values already available at the matmul
/// site. Values defined later cannot prefill the matmul output DST slots.
static bool isDefinedBeforeInBlock(Value value, Operation *anchor) {
  Operation *definingOp = value.getDefiningOp();
  return definingOp && definingOp->getBlock() == anchor->getBlock() &&
         definingOp->isBeforeInBlock(anchor);
}

/// Distinguish DFB-backed accumulators from body SSA expressions. Only the
/// former remains an operand of `ttl.tile_matmul_block`. Non-mutating; an
/// accumulator that is neither a compute input nor a prior body value yields
/// failure (the rewrite reports it as a match failure).
static LogicalResult classifyAccumulator(ComputeOp op, TileMatmulBlockOp mmOp,
                                         MatmulAccumulatorInfo &info) {
  Value accumulator = mmOp.getAccumulator();
  if (!accumulator) {
    info.kind = MatmulAccumulatorKind::None;
    return success();
  }

  std::optional<unsigned> accIdx = traceToBlockArgIndex(accumulator);
  if (accIdx && *accIdx < op.getInputs().size()) {
    info.kind = MatmulAccumulatorKind::InputTensor;
    info.tensorAccumulator = op.getInputs()[*accIdx];
    return success();
  }

  if (!isDefinedBeforeInBlock(accumulator, mmOp)) {
    return failure();
  }

  info.kind = MatmulAccumulatorKind::BodySSA;
  info.bodyAccumulator = accumulator;
  return success();
}

/// Preserve source order within the pre-matmul and post-matmul regions. The
/// split lets body SSA accumulators initialize DST before `matmul_block`.
static void collectMatmulBodyOps(Block &bodyBlock, TileMatmulBlockOp mmOp,
                                 SmallVectorImpl<Operation *> &preMatmulOps,
                                 SmallVectorImpl<Operation *> &postMatmulOps,
                                 SmallVectorImpl<TileStoreOp> &stores) {
  bool beforeMatmul = true;
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (isa<IterIndexOp>(&bodyOp)) {
      continue;
    }
    if (&bodyOp == mmOp.getOperation()) {
      beforeMatmul = false;
      continue;
    }
    if (auto store = dyn_cast<TileStoreOp>(&bodyOp)) {
      stores.push_back(store);
      continue;
    }
    (beforeMatmul ? preMatmulOps : postMatmulOps).push_back(&bodyOp);
  }
}

/// Identify the producer of a body SSA accumulator without relying on a null
/// value as a failed lookup signal.
static bool opProducesValue(Operation *op, Value value) {
  return llvm::is_contained(op->getResults(), value);
}

/// Return true when `ttl-assign-dst` proved the result can overwrite one of the
/// operation's tile operands. The expanded matmul lowering must preserve that
/// assignment because recomputing scratch slots would increase DST pressure.
static bool canReuseAssignedOperandDst(Operation *op, int64_t assignedDst) {
  for (Value operand : op->getOperands()) {
    if (!isa<ttcore::TileType>(operand.getType())) {
      continue;
    }
    FailureOr<int64_t> operandDst = getSingleConstantDstIndex(operand);
    if (succeeded(operandDst) && *operandDst == assignedDst) {
      return true;
    }
  }
  return false;
}

/// Build a compact per-tile scratch map from assigned DST slots. Values that
/// intentionally reuse an existing DST slot are excluded from scratch.
static FailureOr<DenseMap<int64_t, int64_t>>
buildScratchIndexMap(ArrayRef<Operation *> ops,
                     MatmulAccumulatorInfo accumulatorInfo) {
  // TODO: When fused broadcast add/sub/mul tile ops are added, count only the
  // DST result they produce. Their dataflow buffer source operands should not
  // allocate scratch slots unless the fused op lowers through separate
  // copy_tile inputs.
  DenseSet<int64_t> scratchIndices;
  for (Operation *bodyOp : ops) {
    if (!getTileOpDstIndex(bodyOp)) {
      continue;
    }
    if (accumulatorInfo.kind == MatmulAccumulatorKind::BodySSA &&
        opProducesValue(bodyOp, accumulatorInfo.bodyAccumulator)) {
      continue;
    }
    FailureOr<int64_t> dstIndex = getAssignedDstIndex(bodyOp);
    if (failed(dstIndex)) {
      return failure();
    }
    if (canReuseAssignedOperandDst(bodyOp, *dstIndex)) {
      continue;
    }
    scratchIndices.insert(*dstIndex);
  }

  SmallVector<int64_t> sortedIndices(scratchIndices.begin(),
                                     scratchIndices.end());
  llvm::sort(sortedIndices);
  DenseMap<int64_t, int64_t> scratchIndexMap;
  for (auto [ordinal, originalIndex] : llvm::enumerate(sortedIndices)) {
    scratchIndexMap[originalIndex] = ordinal;
  }
  return scratchIndexMap;
}

/// Translate an original in-place DST assignment through the cloned value map.
/// The original operand slot may expand to a different concrete slot per tile.
static std::optional<int64_t>
getMappedOperandReuseDst(Operation *original,
                         const DenseMap<Value, int64_t> &valueDstMap) {
  int64_t originalDst = getRequiredAssignedDstIndex(original);
  for (Value operand : original->getOperands()) {
    auto mappedIt = valueDstMap.find(operand);
    if (mappedIt == valueDstMap.end()) {
      continue;
    }
    FailureOr<int64_t> operandDst = getSingleConstantDstIndex(operand);
    if (succeeded(operandDst) && *operandDst == originalDst) {
      return mappedIt->second;
    }
  }
  return std::nullopt;
}

/// Materialize a local constant for rewritten DST operands and store indices.
static Value createConstantIndex(OpBuilder &builder, Location loc,
                                 int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

/// Assign a cloned op to either the output slot it updates in place or to the
/// tile's private scratch region. The value map records the original SSA value
/// residency so later cloned consumers can preserve in-place DST behavior.
static void
remapClonedDstIndex(OpBuilder &builder, Operation *original, Operation *cloned,
                    DenseMap<Value, int64_t> &valueDstMap,
                    const DenseMap<int64_t, int64_t> &scratchIndexMap,
                    MatmulAccumulatorInfo accumulatorInfo, int64_t outputSlot,
                    int64_t scratchBase) {
  if (!getTileOpDstIndex(cloned)) {
    return;
  }

  std::optional<int64_t> mappedDst;
  if (accumulatorInfo.kind == MatmulAccumulatorKind::BodySSA &&
      opProducesValue(original, accumulatorInfo.bodyAccumulator)) {
    mappedDst = outputSlot;
  } else {
    mappedDst = getMappedOperandReuseDst(original, valueDstMap);
  }

  if (!mappedDst) {
    int64_t originalDst = getRequiredAssignedDstIndex(original);
    auto scratchIt = scratchIndexMap.find(originalDst);
    assert(scratchIt != scratchIndexMap.end() &&
           "tile op was prevalidated to have a scratch DST assignment");
    mappedDst = scratchBase + scratchIt->second;
  }

  Value dstIndex = createConstantIndex(builder, cloned->getLoc(), *mappedDst);
  setTileOpDstIndex(cloned, dstIndex);
  for (auto [originalResult, clonedResult] :
       llvm::zip_equal(original->getResults(), cloned->getResults())) {
    if (isa<ttcore::TileType>(clonedResult.getType())) {
      valueDstMap[originalResult] = *mappedDst;
    }
  }
}

/// Run the full structural preflight for a block matmul compute. Returns
/// failure (without emitting) when `op` is not a well-formed candidate; those
/// cases are reported by the rewrite as a match failure, not by the capacity
/// precondition.
static FailureOr<MatmulComputeAnalysis> analyzeMatmulCompute(ComputeOp op) {
  MatmulComputeAnalysis analysis;
  Block &bodyBlock = op.getBody().front();

  for (Operation &bodyOp : bodyBlock) {
    if (auto matmul = dyn_cast<TileMatmulBlockOp>(&bodyOp)) {
      if (analysis.mmOp) {
        return failure();
      }
      analysis.mmOp = matmul;
    }
  }
  if (!analysis.mmOp) {
    return failure();
  }

  auto outType = cast<RankedTensorType>(op.getOutputs()[0].getType());
  if (outType.getRank() != 2 || !outType.hasStaticShape()) {
    return failure();
  }
  analysis.numRows = outType.getDimSize(0);
  analysis.numCols = outType.getDimSize(1);

  FailureOr<Value> lhsTensor =
      getInputTensorForBodyOperand(op, analysis.mmOp.getLhs());
  FailureOr<Value> rhsTensor =
      getInputTensorForBodyOperand(op, analysis.mmOp.getRhs());
  if (failed(lhsTensor) || failed(rhsTensor)) {
    return failure();
  }
  analysis.lhsTensor = *lhsTensor;
  analysis.rhsTensor = *rhsTensor;

  if (failed(
          classifyAccumulator(op, analysis.mmOp, analysis.accumulatorInfo))) {
    return failure();
  }

  collectMatmulBodyOps(bodyBlock, analysis.mmOp, analysis.preMatmulOps,
                       analysis.postMatmulOps, analysis.stores);
  if (analysis.stores.empty()) {
    return failure();
  }
  for (TileStoreOp store : analysis.stores) {
    if (failed(getSingleConstantDstIndex(store.getTile()))) {
      return failure();
    }
  }

  SmallVector<Operation *> clonedOps;
  llvm::append_range(clonedOps, analysis.preMatmulOps);
  llvm::append_range(clonedOps, analysis.postMatmulOps);
  FailureOr<DenseMap<int64_t, int64_t>> scratchIndexMap =
      buildScratchIndexMap(clonedOps, analysis.accumulatorInfo);
  if (failed(scratchIndexMap)) {
    return failure();
  }
  analysis.scratchIndexMap = std::move(*scratchIndexMap);
  analysis.dstPerIteration = 1 + analysis.scratchIndexMap.size();
  return analysis;
}

} // namespace

/// Validate that the compute's total DST usage fits within capacity.
/// The output shape determines the number of output tiles; dstSlotsPerTile
/// is the number of DST registers each output tile requires (1 for the
/// result plus any scratch slots for other body ops).
static LogicalResult validateDSTCapacity(ComputeOp computeOp,
                                         int64_t dstSlotsPerTile) {
  auto capacityOrErr = computeDSTCapacity(computeOp);
  if (failed(capacityOrErr)) {
    return failure();
  }
  auto outType = cast<RankedTensorType>(computeOp.getOutputs()[0].getType());
  int64_t outM = outType.getDimSize(0);
  int64_t outN = outType.getDimSize(1);
  int64_t totalDstSlots = outM * outN * dstSlotsPerTile;
  int64_t dstCapacity = static_cast<int64_t>(*capacityOrErr);
  if (totalDstSlots > dstCapacity) {
    computeOp.emitOpError()
        << "output " << outM << "x" << outN << " with " << dstSlotsPerTile
        << " DST slots per tile = " << totalDstSlots
        << " total slots exceeds DST capacity of " << dstCapacity
        << "; enable maximize_dst to auto-subblock";
    return failure();
  }
  return success();
}

FailureOr<int64_t> getMatmulComputeDstSlotsPerOutputTile(ComputeOp op) {
  FailureOr<MatmulComputeAnalysis> analysis = analyzeMatmulCompute(op);
  if (failed(analysis)) {
    return failure();
  }
  return analysis->dstPerIteration;
}

LogicalResult verifyMatmulComputeCapacity(ComputeOp op) {
  FailureOr<MatmulComputeAnalysis> analysis = analyzeMatmulCompute(op);
  if (failed(analysis)) {
    return op.emitOpError()
           << "invalid block matmul compute; expected one "
              "ttl.tile_matmul_block with compute-input matmul operands, "
              "constant DST indices, and a valid accumulator";
  }
  return validateDSTCapacity(op, analysis->dstPerIteration);
}

LogicalResult generateMatmulCompute(PatternRewriter &rewriter, Location loc,
                                    ComputeOp op,
                                    ArrayRef<AffineMap> indexingMaps,
                                    ArrayRef<StringAttr> iterTypes) {
  FailureOr<MatmulComputeAnalysis> analysis = analyzeMatmulCompute(op);
  if (failed(analysis)) {
    return rewriter.notifyMatchFailure(
        op, "not a well-formed block matmul compute");
  }
  // DST capacity is verified as a pass precondition before patterns run, so the
  // expansion below assumes the output and scratch slots fit.

  TileMatmulBlockOp mmOp = analysis->mmOp;
  int64_t numRows = analysis->numRows;
  int64_t numCols = analysis->numCols;
  int64_t numOutputTiles = numRows * numCols;
  Type tileType = mmOp.getResult().getType();
  MatmulAccumulatorInfo accumulatorInfo = analysis->accumulatorInfo;
  const DenseMap<int64_t, int64_t> &scratchIndexMap = analysis->scratchIndexMap;
  int64_t scratchPerTile = scratchIndexMap.size();
  ArrayRef<Operation *> preMatmulOps = analysis->preMatmulOps;
  ArrayRef<Operation *> postMatmulOps = analysis->postMatmulOps;
  ArrayRef<TileStoreOp> stores = analysis->stores;
  Value lhsTensor = analysis->lhsTensor;
  Value rhsTensor = analysis->rhsTensor;

  auto dstSection = DstSectionOp::create(rewriter, loc);
  Block &sectionBody = dstSection.getBody().front();
  OpBuilder secBuilder(&sectionBody,
                       Block::iterator(sectionBody.getTerminator()));

  // Each output tile needs its own body-argument map and DST residency map
  // because post-ops are cloned per tile but matmul_block is emitted once.
  struct TileExpansion {
    IRMapping mapping;
    DenseMap<Value, int64_t> valueDstMap;
    int64_t outputSlot = 0;
  };

  SmallVector<TileExpansion> expansions;
  expansions.reserve(numOutputTiles);

  for (int64_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
    for (int64_t colIdx = 0; colIdx < numCols; ++colIdx) {
      int64_t tileIdx = rowIdx * numCols + colIdx;
      int64_t scratchBase = numOutputTiles + tileIdx * scratchPerTile;

      SmallVector<Value> fullIVs(iterTypes.size());
      unsigned parIdx = 0;
      for (auto [dim, iterType] : llvm::enumerate(iterTypes)) {
        if (iterType.getValue() == "reduction") {
          fullIVs[dim] = createConstantIndex(secBuilder, loc, 0);
        } else {
          int64_t coord = (parIdx == 0) ? rowIdx : colIdx;
          fullIVs[dim] = createConstantIndex(secBuilder, loc, coord);
          ++parIdx;
        }
      }

      auto extractedInputs = extractTilesAtIndices(
          secBuilder, loc, op.getInputs(), indexingMaps, fullIVs);
      auto extractedOutputs =
          extractTilesAtIndices(secBuilder, loc, op.getOutputs(), indexingMaps,
                                fullIVs, op.getInputs().size());

      TileExpansion expansion;
      expansion.outputSlot = tileIdx;
      mapComputeBodyArgs(expansion.mapping, op, extractedInputs,
                         extractedOutputs, fullIVs);

      for (Operation *bodyOp : preMatmulOps) {
        Operation *cloned = secBuilder.clone(*bodyOp, expansion.mapping);
        remapClonedDstIndex(secBuilder, bodyOp, cloned, expansion.valueDstMap,
                            scratchIndexMap, accumulatorInfo,
                            expansion.outputSlot, scratchBase);
      }

      expansions.push_back(std::move(expansion));
    }
  }

  Value dstZero = createConstantIndex(secBuilder, loc, 0);
  Value accTensor = accumulatorInfo.kind == MatmulAccumulatorKind::InputTensor
                        ? accumulatorInfo.tensorAccumulator
                        : Value();
  auto newMmOp = TileMatmulBlockOp::create(secBuilder, loc, tileType, lhsTensor,
                                           rhsTensor, accTensor, dstZero);
  newMmOp.setTransposeRhsAttr(mmOp.getTransposeRhsAttr());
  Value mmResult = newMmOp.getResult();

  SmallVector<PendingStore> pendingStores;
  for (TileExpansion &expansion : expansions) {
    Value outputDstIndex =
        createConstantIndex(secBuilder, loc, expansion.outputSlot);
    Value mmTile =
        DstIndexOp::create(secBuilder, loc, tileType, mmResult, outputDstIndex)
            .getResult();
    expansion.mapping.map(mmOp.getResult(), mmTile);
    expansion.valueDstMap[mmOp.getResult()] = expansion.outputSlot;
    int64_t scratchBase =
        numOutputTiles + expansion.outputSlot * scratchPerTile;

    for (Operation *bodyOp : postMatmulOps) {
      Operation *cloned = secBuilder.clone(*bodyOp, expansion.mapping);
      remapClonedDstIndex(secBuilder, bodyOp, cloned, expansion.valueDstMap,
                          scratchIndexMap, accumulatorInfo,
                          expansion.outputSlot, scratchBase);
    }

    for (TileStoreOp store : stores) {
      Value storedTile = expansion.mapping.lookupOrDefault(store.getTile());
      FailureOr<int64_t> storedDstIndex = getSingleConstantDstIndex(storedTile);
      assert(succeeded(storedDstIndex) &&
             "stores were prevalidated to reference one concrete DST slot");
      SmallVector<Value> storeIndices;
      for (Value index : store.getIndices()) {
        storeIndices.push_back(expansion.mapping.lookupOrDefault(index));
      }
      pendingStores.push_back(
          {storedTile, expansion.mapping.lookupOrDefault(store.getView()),
           storeIndices,
           createConstantIndex(secBuilder, loc, *storedDstIndex)});
    }
  }

  for (PendingStore &store : pendingStores) {
    TileStoreOp::create(secBuilder, loc, store.tile, store.view, store.indices,
                        store.dstIndex);
  }

  SmallVector<Value> replacements;
  for (auto result : op.getResults()) {
    auto resultType = cast<RankedTensorType>(result.getType());
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType());
    replacements.push_back(emptyTensor);
  }
  rewriter.replaceOp(op, replacements);
  return success();
}

} // namespace mlir::tt::ttl
