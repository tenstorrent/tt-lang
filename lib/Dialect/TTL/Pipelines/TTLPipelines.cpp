// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Pipelines/TTLPipelines.h"

#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"

#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir::tt::ttl {

void createTTLToTTKernelPipeline(OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &options) {
  pm.addPass(createTTLConvertTTLToLinalg());
  pm.addPass(createTTLAssignDSTRegisters());
  pm.addPass(createTTLConvertTTLToTTKernel());
  pm.addPass(createCanonicalizerPass());
  if (options.lowerToEmitC) {
    // EmitC does not tolerate SSA aliasing introduced by CSE; keep expressions
    // unique to avoid region arg reuse errors in emitc.expression.
    pm.addPass(::mlir::tt::createConvertTTKernelToEmitC());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(mlir::emitc::createFormExpressionsPass());
  } else {
    pm.addPass(createCSEPass());
  }
}

void registerTTLPipelines() {
  PassPipelineRegistration<TTLToTTKernelPipelineOptions>(
      "ttl-to-ttkernel-pipeline",
      "Lower TTL to TTKernel, run cleanup canonicalization/CSE, and optionally "
      "lower TTKernel to EmitC.",
      createTTLToTTKernelPipeline);
}

} // namespace mlir::tt::ttl
