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
  // Note: This pipeline is primarily for NOC thread functions (DMA operations).
  // For compute thread functions that use ttl.compute with ttl.copy_tile, the
  // following passes should be added before convert-ttl-to-ttkernel:
  //   - ttl-lower-to-loops (converts ttl.compute to scf.for)
  //   - ttl-annotate-cb-associations (annotates CB indices for copy_tile)
  // See test/ttlang/Conversion/TTLToTTKernel/compute_fused_chain.mlir for an
  // example of the full compute pipeline.
  pm.addPass(createTTLConvertTTLToCompute());
  // DST register assignment and synchronization (strict ordering required):
  // 1. ttl-tile-and-assign-dst: Assigns DST indices via copy_tile insertion
  // 2. ttl-insert-tile-regs-sync: Inserts DST lifecycle ops
  // (acquire/commit/wait/release) These must run before TTKernel lowering and
  // in this specific order.
  pm.addPass(createTTLTileAndAssignDST());
  pm.addPass(createTTLInsertTileRegsSync());
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
