// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/LowerRowNormalizationCompute.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttl {
namespace {

struct RowNormalizationComputeAnalysis {
  TileRowNormalizationBlockOp block;
  Value inputTensor;
  Value gammaTensor;
  Value outputTensor;
  TileStoreOp store;
  int64_t numTiles = 0;
  std::uint32_t dstCapacity = 0;
};

static FailureOr<Value> getInputTensor(ComputeOp compute, Value bodyValue) {
  std::optional<unsigned> argumentIndex = traceToBlockArgIndex(bodyValue);
  if (!argumentIndex || *argumentIndex >= compute.getInputs().size()) {
    return failure();
  }
  return compute.getInputs()[*argumentIndex];
}

static FailureOr<Value> getOutputTensor(ComputeOp compute, Value bodyValue) {
  std::optional<unsigned> argumentIndex = traceToBlockArgIndex(bodyValue);
  if (!argumentIndex || *argumentIndex < compute.getInputs().size()) {
    return failure();
  }
  unsigned outputIndex = *argumentIndex - compute.getInputs().size();
  if (outputIndex >= compute.getOutputs().size()) {
    return failure();
  }
  return compute.getOutputs()[outputIndex];
}

static FailureOr<RowNormalizationComputeAnalysis>
analyzeRowNormalizationCompute(ComputeOp compute, std::string &reason) {
  RowNormalizationComputeAnalysis analysis;
  Block &body = compute.getBody().front();
  for (Operation &operation : body.without_terminator()) {
    if (auto block = dyn_cast<TileRowNormalizationBlockOp>(&operation)) {
      if (analysis.block) {
        reason = "requires exactly one ttl.tile_row_normalization_block";
        return failure();
      }
      analysis.block = block;
      continue;
    }
    if (auto store = dyn_cast<TileStoreOp>(&operation)) {
      if (analysis.store) {
        reason = "requires exactly one output store";
        return failure();
      }
      analysis.store = store;
      continue;
    }
    if (!isa<IterIndexOp>(&operation)) {
      reason = "row-normalization compute contains an unsupported body op";
      return failure();
    }
  }
  if (!analysis.block || !analysis.store) {
    reason = "requires one block operation and one output store";
    return failure();
  }
  unsigned expectedInputCount = analysis.block.getHasGamma() ? 2 : 1;
  if (compute.getInputs().size() != expectedInputCount ||
      compute.getOutputs().size() != 1) {
    reason = "requires the exact input list, one output, and one output store";
    return failure();
  }
  if (!analysis.block.getHasGamma() &&
      analysis.block.getGamma() != analysis.block.getInput()) {
    reason = "gamma must equal input when gamma multiplication is disabled";
    return failure();
  }
  FailureOr<Value> inputTensor =
      getInputTensor(compute, analysis.block.getInput());
  FailureOr<Value> gammaTensor =
      getInputTensor(compute, analysis.block.getGamma());
  FailureOr<Value> outputTensor =
      getOutputTensor(compute, analysis.block.getOutput());
  if (failed(inputTensor) || failed(gammaTensor) || failed(outputTensor)) {
    reason = "block operands must map to formal compute inputs and outputs";
    return failure();
  }
  analysis.inputTensor = *inputTensor;
  analysis.gammaTensor = *gammaTensor;
  analysis.outputTensor = *outputTensor;

  auto inputType = dyn_cast<RankedTensorType>(analysis.inputTensor.getType());
  auto outputType = dyn_cast<RankedTensorType>(analysis.outputTensor.getType());
  if (!inputType || !outputType || !inputType.hasStaticShape() ||
      !outputType.hasStaticShape() || inputType.getRank() != 2 ||
      outputType.getRank() != 2 || inputType.getDimSize(0) != 1 ||
      inputType != outputType) {
    reason = "input and output must be matching static one-row tensors";
    return failure();
  }
  analysis.numTiles = inputType.getNumElements();
  if (analysis.numTiles != static_cast<int64_t>(analysis.block.getNumTiles())) {
    reason = "num_tiles must match the row tensor width";
    return failure();
  }

  FailureOr<std::uint32_t> capacity = computeDSTCapacity(compute);
  if (failed(capacity)) {
    reason = "cannot determine effective DST capacity";
    return failure();
  }
  analysis.dstCapacity = *capacity;
  if (analysis.numTiles < 1 || analysis.numTiles > analysis.dstCapacity) {
    reason =
        (Twine("row requires ") + Twine(analysis.numTiles) +
         " DST slots, but effective capacity is " + Twine(analysis.dstCapacity))
            .str();
    return failure();
  }

  if (analysis.block.getHasGamma()) {
    auto gammaType = dyn_cast<RankedTensorType>(analysis.gammaTensor.getType());
    if (!gammaType || !gammaType.hasStaticShape() || gammaType.getRank() != 2) {
      reason = "gamma must be a static rank-2 tensor";
      return failure();
    }
    if (gammaType != outputType) {
      reason = "gamma tensor shape must match the output shape";
      return failure();
    }
  }

  if (analysis.store.getTile() != analysis.block.getResult()) {
    reason = "output store must consume the block result";
    return failure();
  }
  FailureOr<unsigned> outputIndex =
      compute.getOutputIndexForView(analysis.store.getView());
  if (failed(outputIndex) || *outputIndex != 0 ||
      analysis.outputTensor != compute.getOutputs().front()) {
    reason = "block and store must map to the sole formal compute output";
    return failure();
  }
  return analysis;
}

static Value constantIndex(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

} // namespace

LogicalResult verifyRowNormalizationCompute(ComputeOp op) {
  std::string reason;
  if (failed(analyzeRowNormalizationCompute(op, reason))) {
    return op.emitOpError(reason);
  }
  return success();
}

LogicalResult generateRowNormalizationCompute(PatternRewriter &rewriter,
                                              Location loc, ComputeOp op) {
  std::string reason;
  FailureOr<RowNormalizationComputeAnalysis> analysis =
      analyzeRowNormalizationCompute(op, reason);
  if (failed(analysis)) {
    return rewriter.notifyMatchFailure(op, reason);
  }

  auto dstSection = DstSectionOp::create(rewriter, loc);
  Block &sectionBody = dstSection.getBody().front();
  OpBuilder sectionBuilder(&sectionBody,
                           Block::iterator(sectionBody.getTerminator()));
  Value scalarDstIndex = constantIndex(sectionBuilder, loc, 0);
  Type tileType = analysis->block.getResult().getType();
  auto loweredBlock = TileRowNormalizationBlockOp::create(
      sectionBuilder, loc, tileType, analysis->inputTensor,
      analysis->gammaTensor, analysis->outputTensor,
      analysis->block.getScaleAttr(), analysis->block.getEpsilonAttr(),
      analysis->block.getHasGammaAttr(), analysis->block.getNumTilesAttr(),
      scalarDstIndex);

  for (int64_t tileIndex = 0; tileIndex < analysis->numTiles; ++tileIndex) {
    Value outputDstIndex = constantIndex(sectionBuilder, loc, tileIndex);
    Value outputTile =
        DstIndexOp::create(sectionBuilder, loc, tileType,
                           loweredBlock.getResult(), outputDstIndex);
    SmallVector<Value> indices = {
        constantIndex(sectionBuilder, loc, 0),
        constantIndex(sectionBuilder, loc, tileIndex)};
    TileStoreOp::create(sectionBuilder, loc, outputTile,
                        analysis->store.getView(), indices, outputDstIndex);
  }

  rewriter.replaceOp(op, op.getOutputs());
  return success();
}

} // namespace mlir::tt::ttl
