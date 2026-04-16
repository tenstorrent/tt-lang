// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// LowerMatmulCompute
//===----------------------------------------------------------------------===//
//
// Lowers a ComputeOp containing tile_matmul_block into a single DstSectionOp
// with the matmul call, cloned body ops (elementwise, copy_tile, etc.),
// and per-output-view stores. Called from LowerComputeToLoops when the
// compute body contains a matmul.
//
// CB lifecycle (wait/pop for inputs, reserve/push for output) is NOT emitted
// here -- it comes from the user's DFB operations outside the compute.
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

namespace mlir::tt::ttl {

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
    computeOp.emitError() << "output " << outM << "x" << outN << " with "
                          << dstSlotsPerTile
                          << " DST slots per tile = " << totalDstSlots
                          << " total slots exceeds DST capacity of "
                          << dstCapacity
                          << "; enable maximize_dst to auto-subblock";
    return failure();
  }
  return success();
}

LogicalResult generateMatmulCompute(PatternRewriter &rewriter, Location loc,
                                    ComputeOp op,
                                    ArrayRef<AffineMap> indexingMaps,
                                    ArrayRef<StringAttr> iterTypes) {
  Block &bodyBlock = op.getBody().front();

  // Find the TileMatmulBlockOp in the body.
  TileMatmulBlockOp mmOp;
  for (Operation &bodyOp : bodyBlock) {
    if (auto matmul = dyn_cast<TileMatmulBlockOp>(&bodyOp)) {
      mmOp = matmul;
      break;
    }
  }
  assert(mmOp && "generateMatmulCompute requires tile_matmul_block in body");

  auto outType = cast<RankedTensorType>(op.getOutputs()[0].getType());
  int64_t numRows = outType.getDimSize(0);
  int64_t numCols = outType.getDimSize(1);
  Type tileType = mmOp.getResult().getType();

  // Map matmul body operands to compute input tensors via block arg indices.
  auto getInputForBodyOperand = [&](Value bodyVal) -> Value {
    auto idx = traceToBlockArgIndex(bodyVal);
    assert(idx && "body operand must trace to a block argument");
    return op.getInputs()[*idx];
  };

  Value lhsTensor = getInputForBodyOperand(mmOp.getLhs());
  Value rhsTensor = getInputForBodyOperand(mmOp.getRhs());

  Value accTensor;
  if (Value acc = mmOp.getAccumulator()) {
    auto accIdx = traceToBlockArgIndex(acc);
    assert(accIdx && *accIdx < op.getInputs().size() &&
           "accumulator must trace to a compute input");
    accTensor = op.getInputs()[*accIdx];
  }

  // Collect distinct output views from tile_store ops.
  SmallVector<Value> outputViews;
  Value storedTile;
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (auto store = dyn_cast<TileStoreOp>(&bodyOp)) {
      if (!storedTile) {
        storedTile = store.getTile();
      } else {
        assert(store.getTile() == storedTile &&
               "all body stores must reference the same tile value");
      }
      Value view = store.getView();
      if (!llvm::is_contained(outputViews, view)) {
        outputViews.push_back(view);
      }
    }
  }
  assert(!outputViews.empty() && "matmul compute must have tile_store(s)");

  size_t numDims = iterTypes.size();
  size_t numInputs = op.getInputs().size();

  // Collect body ops to clone: everything except the matmul, stores,
  // and iter_index (which are handled separately via the IRMapping).
  SmallVector<Operation *> bodyOpsToClone;
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (isa<TileMatmulBlockOp, TileStoreOp, IterIndexOp>(&bodyOp)) {
      continue;
    }
    bodyOpsToClone.push_back(&bodyOp);
  }

  // Determine the maximum DST index used by body ops.
  int64_t maxBodyDstIdx = 0;
  for (Operation *bodyOp : bodyOpsToClone) {
    if (auto dstVal = getTileOpDstIndex(bodyOp)) {
      auto constIdx = foldIndexToConstant(*dstVal);
      assert(constIdx && "DST index must be a constant after assignment");
      maxBodyDstIdx = std::max(maxBodyDstIdx, *constIdx);
    }
  }
  int64_t dstPerIteration = maxBodyDstIdx + 1;

  if (failed(validateDSTCapacity(op, dstPerIteration))) {
    return failure();
  }

  // Create the DstSectionOp that wraps the expanded matmul computation.
  auto dstSection = DstSectionOp::create(rewriter, loc);
  Block &sectionBody = dstSection.getBody().front();
  OpBuilder secBuilder(&sectionBody,
                       Block::iterator(sectionBody.getTerminator()));

  Value dstZero = arith::ConstantIndexOp::create(secBuilder, loc, 0);
  Value mmResult =
      TileMatmulBlockOp::create(secBuilder, loc, tileType, lhsTensor, rhsTensor,
                                accTensor, dstZero)
          .getResult();

  // Clone body ops expanded M*N times. For each output tile (m, n),
  // clone all non-matmul/non-store ops with extracted tile operands from
  // CBs and remapped DST indices. For M=N=1, this is a single iteration.
  for (int64_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
    for (int64_t colIdx = 0; colIdx < numCols; ++colIdx) {
      int64_t tileIdx = rowIdx * numCols + colIdx;
      int64_t dstBase = tileIdx * dstPerIteration;

      // Build the full IV vector with constants for all dimensions.
      SmallVector<Value> fullIVs(numDims);
      unsigned parIdx = 0;
      for (auto [dim, iterType] : llvm::enumerate(iterTypes)) {
        if (iterType.getValue() == "reduction") {
          fullIVs[dim] = arith::ConstantIndexOp::create(secBuilder, loc, 0);
        } else {
          int64_t coord = (parIdx == 0) ? rowIdx : colIdx;
          fullIVs[dim] = arith::ConstantIndexOp::create(secBuilder, loc, coord);
          ++parIdx;
        }
      }

      auto extractedInputs = extractTilesAtIndices(
          secBuilder, loc, op.getInputs(), indexingMaps, fullIVs);
      auto extractedOutputs = extractTilesAtIndices(
          secBuilder, loc, op.getOutputs(), indexingMaps, fullIVs, numInputs);

      IRMapping mapping;
      mapComputeBodyArgs(mapping, op, extractedInputs, extractedOutputs,
                         fullIVs);

      mapping.map(mmOp.getResult(), mmResult);

      // Clone body ops in original order.
      for (Operation *bodyOp : bodyOpsToClone) {
        auto *cloned = secBuilder.clone(*bodyOp, mapping);

        // Offset the DST index for ops with TTLDstResultOpTrait.
        if (dstBase != 0) {
          if (auto dstVal = getTileOpDstIndex(cloned)) {
            Value offsetVal =
                arith::ConstantIndexOp::create(secBuilder, loc, dstBase);
            Value newDstIndex =
                arith::AddIOp::create(secBuilder, loc, *dstVal, offsetVal);
            setTileOpDstIndex(cloned, newDstIndex);
          }
        }
      }
    }
  }

  // Emit M*N individual tile_store ops per output view.
  OpBuilder storeBuilder(&sectionBody,
                         Block::iterator(sectionBody.getTerminator()));
  for (Value outView : outputViews) {
    for (int64_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
      for (int64_t colIdx = 0; colIdx < numCols; ++colIdx) {
        Value mIdx = arith::ConstantIndexOp::create(storeBuilder, loc, rowIdx);
        Value nIdx = arith::ConstantIndexOp::create(storeBuilder, loc, colIdx);
        Value dstIdx = arith::ConstantIndexOp::create(
            storeBuilder, loc, rowIdx * numCols + colIdx);
        TileStoreOp::create(storeBuilder, loc, mmResult, outView,
                            ValueRange{mIdx, nIdx}, dstIdx);
      }
    }
  }

  // The compute op's SSA results may still have users (e.g., cb_push).
  // Replace each result with an empty tensor to satisfy those uses.
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
