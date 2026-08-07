// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/LowerRowNormalizationCompute.h"
#include "ttlang/Dialect/TTL/Transforms/FixedBlockComputeAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttl {
namespace {

static bool targetProvidesRowNormalizationSchedule(Operation *operation) {
  ModuleOp module = operation->getParentOfType<ModuleOp>();
  auto target =
      module ? module->getAttrOfType<ttcore::ArchAttr>(kTargetArchAttrName)
             : ttcore::ArchAttr();
  return target && target.getValue() == ttcore::Arch::Blackhole;
}

struct RowNormalizationComputeAnalysis {
  FixedBlockComputeAnalysis fixed;
  TileRowNormalizationBlockOp block;
  int64_t numTiles = 0;
  std::uint32_t dstCapacity = 0;
};

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
    }
  }
  if (!analysis.block) {
    reason = "requires exactly one ttl.tile_row_normalization_block";
    return failure();
  }

  SmallVector<Value> bodyInputs = {analysis.block.getInput()};
  if (analysis.block.getHasGamma()) {
    bodyInputs.push_back(analysis.block.getGamma());
  }
  FailureOr<FixedBlockComputeAnalysis> fixed = analyzeFixedBlockCompute(
      compute, analysis.block, bodyInputs, analysis.block.getOutput(),
      analysis.block.getResult(), reason);
  if (failed(fixed)) {
    return failure();
  }
  analysis.fixed = std::move(*fixed);

  if (!analysis.block.getHasGamma() &&
      analysis.block.getGamma() != analysis.block.getInput()) {
    reason = "gamma must equal input when gamma multiplication is disabled";
    return failure();
  }
  if (!targetProvidesRowNormalizationSchedule(compute)) {
    reason = "row-normalization block lowering requires a Blackhole target";
    return failure();
  }

  Value inputTensor = analysis.fixed.inputTensors.front();
  auto inputType = dyn_cast<RankedTensorType>(inputTensor.getType());
  auto outputType =
      dyn_cast<RankedTensorType>(analysis.fixed.outputTensor.getType());
  if (!inputType || !outputType || !inputType.hasStaticShape() ||
      !outputType.hasStaticShape() || inputType.getRank() != 2 ||
      outputType.getRank() != 2 || inputType.getDimSize(0) != 1 ||
      inputType != outputType) {
    reason = "input and output must be matching static one-row tensors";
    return failure();
  }
  analysis.numTiles = inputType.getNumElements();

  analysis.dstCapacity = std::min<std::uint32_t>(8, analysis.fixed.dstCapacity);
  if (analysis.numTiles < 1 || analysis.numTiles > analysis.dstCapacity) {
    reason =
        (Twine("row requires ") + Twine(analysis.numTiles) +
         " DST slots, but effective capacity is " + Twine(analysis.dstCapacity))
            .str();
    return failure();
  }

  if (analysis.block.getHasGamma()) {
    auto gammaType =
        dyn_cast<RankedTensorType>(analysis.fixed.inputTensors[1].getType());
    if (!gammaType || !gammaType.hasStaticShape() || gammaType.getRank() != 2) {
      reason = "gamma must be a static rank-2 tensor";
      return failure();
    }
    if (gammaType != outputType) {
      reason = "gamma tensor shape must match the output shape";
      return failure();
    }
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
      sectionBuilder, loc, tileType, analysis->fixed.inputTensors[0],
      analysis->block.getHasGamma() ? analysis->fixed.inputTensors[1]
                                    : analysis->fixed.inputTensors[0],
      analysis->fixed.outputTensor, analysis->block.getScaleAttr(),
      analysis->block.getEpsilonAttr(), analysis->block.getHasGammaAttr(),
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
                        analysis->fixed.store.getView(), indices,
                        outputDstIndex);
  }

  rewriter.replaceOp(op, op.getOutputs());
  return success();
}

} // namespace mlir::tt::ttl
