// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/ComputeOutputPublication.h"

#include "ComputeOpCreationPlanning.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::tt::ttl {

LogicalResult
resolveCurrentOutputPublication(Operation *source, PatternRewriter &rewriter,
                                const OutputPublicationPlan &analyzed,
                                OutputPublicationPlan &resolved) {
  PlanningResult<OutputPublicationPlan> current =
      resolveOutputPublicationOperations(analyzed);
  if (current.isInvalidIR()) {
    return rewriter.notifyMatchFailure(source, current.getInvalidIR().message);
  }
  assert(current.isPlanned() &&
         "output resolution has no recoverable rejection");
  resolved = std::move(current).takePlan();
  return success();
}

Value createOutputInitTensor(OpBuilder &builder, Location loc,
                             RankedTensorType type, Value exemplar) {
  SmallVector<Value> dynamicDimensions;
  for (auto dimension : llvm::enumerate(type.getShape())) {
    if (dimension.value() == ShapedType::kDynamic) {
      dynamicDimensions.push_back(
          tensor::DimOp::create(builder, loc, exemplar, dimension.index()));
    }
  }
  return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                 type.getElementType(), dynamicDimensions);
}

void setInsertionPointToOutputPublication(
    PatternRewriter &rewriter, const OutputPublicationPlan &outputs) {
  rewriter.setInsertionPoint(outputs.insertionAnchor);
}

void createComputeTileStore(PatternRewriter &rewriter, Location loc,
                            Value tileResult, ComputeOp computeOp,
                            StoreOp store) {
  SmallVector<Value> iterationIndices =
      getOrCreateIterIndices(rewriter, computeOp);
  auto indexingMaps = computeOp.getIndexingMapsArray();
  size_t numInputs = computeOp.getNumInputs();

  FailureOr<unsigned> outputIndex =
      computeOp.getOutputIndexForView(store.getView());
  assert(succeeded(outputIndex) &&
         "planned store must map to one formal compute output");
  AffineMap outputMap = indexingMaps[numInputs + *outputIndex];
  SmallVector<Value> indices =
      applyIndexingMap(rewriter, loc, outputMap, iterationIndices);

  createTileOpWithPlaceholderDstIndex<TileStoreOp>(rewriter, loc, tileResult,
                                                   store.getView(), indices);
}

void relocateOutputPushesAfterCompute(
    PatternRewriter &rewriter, ComputeOp computeOp,
    const OutputPublicationPlan &outputs,
    SmallVectorImpl<CBPushOp> &replacedPushes) {
  OpBuilder::InsertionGuard guard(rewriter);
  Operation *insertAfter = computeOp;
  for (CBPushOp push : outputs.pushes) {
    assert(push->getBlock() == computeOp->getBlock() &&
           "pushes absorbed into a compute must be siblings of that compute");
    if (!push->isBeforeInBlock(computeOp)) {
      continue;
    }
    rewriter.setInsertionPointAfter(insertAfter);
    auto replacement = cast<CBPushOp>(rewriter.clone(*push));
    insertAfter = replacement;
    replacedPushes.push_back(push);
  }
}

void eraseReplacedOutputPublication(PatternRewriter &rewriter,
                                    const OutputPublicationPlan &outputs,
                                    ComputeOp computeOp,
                                    ArrayRef<CBPushOp> replacedPushes) {
  for (StoreOp store : outputs.stores) {
    assert(store->getBlock() == computeOp->getBlock() &&
           "stores absorbed into a compute must be siblings of that compute");
    rewriter.eraseOp(store);
  }
  for (CBPushOp push : replacedPushes) {
    assert(push->getBlock() == computeOp->getBlock() &&
           "pushes absorbed into a compute must be siblings of that compute");
    rewriter.eraseOp(push);
  }
}

} // namespace mlir::tt::ttl
