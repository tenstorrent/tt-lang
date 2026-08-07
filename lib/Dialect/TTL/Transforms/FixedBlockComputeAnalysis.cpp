// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/FixedBlockComputeAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttl {
namespace {

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

} // namespace

FailureOr<FixedBlockComputeAnalysis>
analyzeFixedBlockCompute(ComputeOp compute, Operation *block,
                         ValueRange bodyInputs, Value bodyOutput,
                         Value blockResult, std::string &reason) {
  FixedBlockComputeAnalysis analysis;
  analysis.block = block;

  if (!block || !block->hasTrait<TTLFixedBlockComputeOpTrait>() ||
      block->getBlock() != &compute.getBody().front()) {
    reason = "requires one fixed block operation in the compute body";
    return failure();
  }

  for (Operation &operation : compute.getBody().front().without_terminator()) {
    if (&operation == block || isa<IterIndexOp>(&operation)) {
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
    reason = "fixed block compute contains an unsupported body operation";
    return failure();
  }

  if (!analysis.store) {
    reason = "requires exactly one output store";
    return failure();
  }
  if (compute.getInputs().size() != bodyInputs.size() ||
      compute.getOutputs().size() != 1) {
    reason = "requires the exact input list and one output";
    return failure();
  }

  for (Value bodyInput : bodyInputs) {
    FailureOr<Value> inputTensor = getInputTensor(compute, bodyInput);
    if (failed(inputTensor)) {
      reason = "block input must map to a formal compute input";
      return failure();
    }
    analysis.inputTensors.push_back(*inputTensor);
  }
  if (!llvm::equal(analysis.inputTensors, compute.getInputs())) {
    reason = "block operands must map to the formal compute inputs in order";
    return failure();
  }
  FailureOr<Value> outputTensor = getOutputTensor(compute, bodyOutput);
  if (failed(outputTensor)) {
    reason = "block output must map to the formal compute output";
    return failure();
  }
  analysis.outputTensor = *outputTensor;

  if (analysis.store.getTile() != blockResult) {
    reason = "output store must consume the fixed block result";
    return failure();
  }
  FailureOr<unsigned> outputIndex =
      compute.getOutputIndexForView(analysis.store.getView());
  if (failed(outputIndex) || *outputIndex != 0 ||
      analysis.outputTensor != compute.getOutputs().front()) {
    reason = "block and store must map to the sole formal compute output";
    return failure();
  }

  FailureOr<std::uint32_t> capacity = computeDSTCapacity(compute);
  if (failed(capacity)) {
    reason = "cannot determine effective DST capacity";
    return failure();
  }
  analysis.dstCapacity = *capacity;
  return analysis;
}

} // namespace mlir::tt::ttl
