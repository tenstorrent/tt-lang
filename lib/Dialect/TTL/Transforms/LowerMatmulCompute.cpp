// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// LowerMatmulCompute
//===----------------------------------------------------------------------===//
//
// Lowers a ComputeOp containing tile_matmul_block into a single DstSectionOp
// with the matmul call, per-tile post-ops (binary elementwise, copy_tile,
// etc.), and per-tile stores. Called from LowerComputeToLoops when the
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

/// Trace a value through copy_tile (inserted by assign-dst) to its source
/// block argument. Returns the block arg index, or std::nullopt if the value
/// does not trace to a block argument.
static std::optional<unsigned> traceToBlockArgIndex(Value val) {
  if (auto copyOp = val.getDefiningOp<CopyTileOp>()) {
    val = copyOp.getSrc();
  }
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    return blockArg.getArgNumber();
  }
  return std::nullopt;
}

/// Validate that the compute's total DST usage fits within capacity.
/// The output shape determines the number of output tiles; dstSlotsPerTile
/// is the number of DST registers each output tile requires (1 for the
/// result plus any scratch slots for post-ops).
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

/// Apply an indexing map to constant index values, producing index-typed
/// Values via affine composition and folding.
static SmallVector<Value> applyIndexingMap(OpBuilder &builder, Location loc,
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
    return idx ? op.getInputs()[*idx] : Value();
  };

  Value lhsTensor = getInputForBodyOperand(mmOp.getLhs());
  Value rhsTensor = getInputForBodyOperand(mmOp.getRhs());
  assert(lhsTensor && rhsTensor && "matmul operands must trace to inputs");

  Value accTensor;
  if (Value acc = mmOp.getAccumulator()) {
    auto accIdx = traceToBlockArgIndex(acc);
    assert(accIdx && *accIdx < op.getInputs().size() &&
           "accumulator must trace to a compute input");
    accTensor = op.getInputs()[*accIdx];
  }

  // Collect store ops from the body.
  SmallVector<TileStoreOp> bodyStores;
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (auto store = dyn_cast<TileStoreOp>(&bodyOp)) {
      bodyStores.push_back(store);
    }
  }
  assert(!bodyStores.empty() && "matmul compute must have tile_store(s)");
  Value outView = bodyStores[0].getView();

  size_t numDims = iterTypes.size();
  size_t numInputs = op.getInputs().size();

  // Collect post-matmul non-store body ops (the ops between matmul and
  // stores: copy_tile, binary ops, constants, etc.).
  SmallVector<Operation *> postMatmulOps;
  bool foundMM = false;
  for (Operation &bodyOp : bodyBlock.without_terminator()) {
    if (isa<TileMatmulBlockOp>(&bodyOp)) {
      foundMM = true;
      continue;
    }
    if (foundMM && !isa<TileStoreOp>(&bodyOp)) {
      postMatmulOps.push_back(&bodyOp);
    }
  }

  // Determine the number of DST slots used per iteration by post-ops.
  // Needed for M*N > 1 expansion to offset DST indices per tile.
  int64_t maxBodyDstIdx = 0;
  for (Operation *postOp : postMatmulOps) {
    if (auto attr = postOp->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
      maxBodyDstIdx =
          std::max(maxBodyDstIdx, static_cast<int64_t>(attr.getInt()));
    }
  }
  int64_t dstPerIteration = maxBodyDstIdx + 1;

  if (failed(validateDSTCapacity(op, dstPerIteration))) {
    return failure();
  }

  // Create the DstSectionOp that wraps matmul + post-ops + stores.
  auto dstSection = DstSectionOp::create(rewriter, loc);
  Block &sectionBody = dstSection.getBody().front();
  OpBuilder secBuilder(&sectionBody,
                       Block::iterator(sectionBody.getTerminator()));

  // Emit the matmul_block with full tensor operands.
  auto mmResultOp = TileMatmulBlockOp::create(secBuilder, loc, tileType,
                                              lhsTensor, rhsTensor, accTensor);
  mmResultOp->setAttr(kDstIdxAttrName, secBuilder.getI32IntegerAttr(0));

  // Placeholder for referencing DST-resident values. Downstream passes
  // (ConvertTTLToTTKernel) resolve tile references via dst_idx attributes.
  // The dst_idx on the placeholder allows getDstIndexFromValue to find the
  // matmul result's DST register when processing SFPU binary post-ops.
  auto placeholderOp = UnrealizedConversionCastOp::create(
      secBuilder, loc, tileType, ValueRange{});
  placeholderOp->setAttr(kDstIdxAttrName, secBuilder.getI32IntegerAttr(0));
  Value placeholder = placeholderOp.getResult(0);

  // Emit post-matmul ops expanded M*N times. For each output tile (m, n),
  // clone the post-ops with extracted tile operands from CBs and remapped
  // DST indices. For M=N=1, this is a single iteration (no loop overhead).
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

      // Extract tiles from input tensors at indexing map positions.
      SmallVector<Value> extractedInputs;
      for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
        SmallVector<Value> indices =
            applyIndexingMap(secBuilder, loc, indexingMaps[idx], fullIVs);
        Value tile = tensor::ExtractOp::create(secBuilder, loc, input, indices);
        extractedInputs.push_back(tile);
      }

      SmallVector<Value> extractedOutputs;
      for (auto [idx, output] : llvm::enumerate(op.getOutputs())) {
        SmallVector<Value> indices = applyIndexingMap(
            secBuilder, loc, indexingMaps[numInputs + idx], fullIVs);
        Value tile =
            tensor::ExtractOp::create(secBuilder, loc, output, indices);
        extractedOutputs.push_back(tile);
      }

      // Build the operand mapping for cloning body ops.
      IRMapping mapping;
      for (auto [idx, arg] : llvm::enumerate(op.getInputs())) {
        mapping.map(bodyBlock.getArgument(idx), extractedInputs[idx]);
      }
      for (auto [idx, arg] : llvm::enumerate(op.getOutputs())) {
        mapping.map(bodyBlock.getArgument(numInputs + idx),
                    extractedOutputs[idx]);
      }

      // Map iter_index ops to the constant IVs.
      for (Operation &bodyOp : bodyBlock.without_terminator()) {
        if (auto iterIdx = dyn_cast<IterIndexOp>(&bodyOp)) {
          mapping.map(iterIdx.getResult(), fullIVs[iterIdx.getDim()]);
        }
      }

      // Map the matmul result to the placeholder.
      mapping.map(mmOp.getResult(), placeholder);

      // Clone post-matmul ops with DST index remapping.
      for (Operation *postOp : postMatmulOps) {
        auto *cloned = secBuilder.clone(*postOp, mapping);

        if (auto attr = cloned->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
          cloned->setAttr(kDstIdxAttrName, secBuilder.getI32IntegerAttr(
                                               attr.getInt() + dstBase));
        }

        if (auto copyTile = dyn_cast<CopyTileOp>(cloned)) {
          if (dstBase != 0) {
            Value offsetVal =
                arith::ConstantIndexOp::create(secBuilder, loc, dstBase);
            Value newDstIndex = arith::AddIOp::create(
                secBuilder, loc, copyTile.getDstIndex(), offsetVal);
            copyTile.getDstIndexMutable().assign(newDstIndex);
          }
        }
      }
    }
  }

  // Emit M*N individual tile_store ops with explicit DST indices.
  OpBuilder storeBuilder(&sectionBody,
                         Block::iterator(sectionBody.getTerminator()));
  for (int64_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
    for (int64_t colIdx = 0; colIdx < numCols; ++colIdx) {
      Value mIdx = arith::ConstantIndexOp::create(storeBuilder, loc, rowIdx);
      Value nIdx = arith::ConstantIndexOp::create(storeBuilder, loc, colIdx);
      auto store = TileStoreOp::create(storeBuilder, loc, placeholder, outView,
                                       ValueRange{mIdx, nIdx});
      store->setAttr(kDstIdxAttrName,
                     storeBuilder.getI32IntegerAttr(rowIdx * numCols + colIdx));
    }
  }

  // Replace the compute op with a placeholder tensor.
  Value emptyTensor = tensor::EmptyOp::create(rewriter, loc, outType.getShape(),
                                              outType.getElementType());
  rewriter.replaceOp(op, emptyTensor);
  return success();
}

} // namespace mlir::tt::ttl
