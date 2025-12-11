// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Pipelines/TTLPipelines.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "ttlang/Dialect/TTL/Passes.h"

using namespace mlir;

namespace mlir::tt::ttl {

void createTTLToTTKernelPipeline(OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &) {
  pm.addPass(createTTLConvertTTLToTTKernel());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void registerTTLPipelines() {
  PassPipelineRegistration<TTLToTTKernelPipelineOptions>(
      "ttl-to-ttkernel-pipeline",
      "Lower TTL to TTKernel and run cleanup canonicalization/CSE",
      createTTLToTTKernelPipeline);
}

} // namespace mlir::tt::ttl
