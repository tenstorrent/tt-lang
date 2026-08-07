// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/LowerFusionCompute.h"
#include "ttlang/Dialect/TTL/Transforms/FixedBlockComputeAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttl {
namespace {

struct FusionComputeAnalysis {
  FixedBlockComputeAnalysis fixed;
  TileMulReduceBlockOp block;
  Value lhsTensor;
  Value rhsTensor;
  int64_t numTiles = 0;
  std::uint32_t dstCapacity = 0;
};

static FailureOr<FusionComputeAnalysis>
analyzeFusionCompute(ComputeOp compute, std::string &reason) {
  FusionComputeAnalysis analysis;
  for (Operation &operation : compute.getBody().front().without_terminator()) {
    if (auto block = dyn_cast<TileMulReduceBlockOp>(&operation)) {
      if (analysis.block) {
        reason = "requires exactly one ttl.tile_mul_reduce_block";
        return failure();
      }
      analysis.block = block;
    }
  }
  if (!analysis.block) {
    reason = "requires exactly one ttl.tile_mul_reduce_block";
    return failure();
  }

  SmallVector<Value> bodyInputs = {analysis.block.getLhs()};
  if (analysis.block.getRhs() != analysis.block.getLhs()) {
    bodyInputs.push_back(analysis.block.getRhs());
  }
  FailureOr<FixedBlockComputeAnalysis> fixed = analyzeFixedBlockCompute(
      compute, analysis.block, bodyInputs, analysis.block.getOutput(),
      analysis.block.getResult(), reason);
  if (failed(fixed)) {
    return failure();
  }
  analysis.fixed = std::move(*fixed);

  analysis.lhsTensor = analysis.fixed.inputTensors.front();
  analysis.rhsTensor = analysis.block.getRhs() == analysis.block.getLhs()
                           ? analysis.lhsTensor
                           : analysis.fixed.inputTensors[1];
  auto lhsType = dyn_cast<RankedTensorType>(analysis.lhsTensor.getType());
  auto rhsType = dyn_cast<RankedTensorType>(analysis.rhsTensor.getType());
  auto outputType =
      dyn_cast<RankedTensorType>(analysis.fixed.outputTensor.getType());
  if (!lhsType || !rhsType || !outputType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !outputType.hasStaticShape() ||
      lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      outputType.getRank() != 2 || lhsType != rhsType ||
      outputType.getDimSize(0) != 1 || outputType.getDimSize(1) != 1) {
    reason = "requires matching static rank-2 inputs and a static 1x1 output";
    return failure();
  }
  analysis.numTiles = lhsType.getNumElements();
  if (analysis.numTiles != static_cast<int64_t>(analysis.block.getNumTiles())) {
    reason = "num_tiles must match the input tensor domain";
    return failure();
  }

  analysis.dstCapacity = std::min<std::uint32_t>(8, analysis.fixed.dstCapacity);
  if (analysis.numTiles < 1 || analysis.numTiles > analysis.dstCapacity) {
    reason =
        (Twine("multiply-reduction requires ") + Twine(analysis.numTiles) +
         " DST slots, but effective capacity is " + Twine(analysis.dstCapacity))
            .str();
    return failure();
  }
  return analysis;
}

static Value constantIndex(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

} // namespace

LogicalResult verifyFusionCompute(ComputeOp op) {
  std::string reason;
  if (failed(analyzeFusionCompute(op, reason))) {
    return op.emitOpError(reason);
  }
  return success();
}

LogicalResult generateFusionCompute(PatternRewriter &rewriter, Location loc,
                                    ComputeOp op) {
  std::string reason;
  FailureOr<FusionComputeAnalysis> analysis = analyzeFusionCompute(op, reason);
  if (failed(analysis)) {
    return rewriter.notifyMatchFailure(op, reason);
  }

  auto dstSection = DstSectionOp::create(rewriter, loc);
  Block &sectionBody = dstSection.getBody().front();
  OpBuilder sectionBuilder(&sectionBody,
                           Block::iterator(sectionBody.getTerminator()));
  Value scalarDstIndex = constantIndex(sectionBuilder, loc, 0);
  Type tileType = analysis->block.getResult().getType();
  auto loweredBlock = TileMulReduceBlockOp::create(
      sectionBuilder, loc, tileType, analysis->lhsTensor, analysis->rhsTensor,
      analysis->fixed.outputTensor, analysis->block.getScaleAttr(),
      analysis->block.getNumTilesAttr(), scalarDstIndex);
  SmallVector<Value> indices = {constantIndex(sectionBuilder, loc, 0),
                                constantIndex(sectionBuilder, loc, 0)};
  TileStoreOp::create(sectionBuilder, loc, loweredBlock.getResult(),
                      analysis->fixed.store.getView(), indices, scalarDstIndex);

  rewriter.replaceOp(op, op.getOutputs());
  return success();
}

} // namespace mlir::tt::ttl
