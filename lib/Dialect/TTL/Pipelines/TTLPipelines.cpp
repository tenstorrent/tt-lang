// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Pipelines/TTLPipelines.h"

#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir::tt::ttl {

void createTTLToTTKernelPipeline(OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &options) {
  pm.addPass(createTTLConvertTTLToCompute());

  // DST register allocation and loop transformation pipeline:
  // 1. ttl-tile-and-assign-dst: Assigns DST indices, adds ttl.unroll_factor attribute
  // 2. ttl-lower-to-loops: Converts ttl.compute to scf.for loops (no sync ops yet)
  // 3. ttl-unroll-compute-loops: Unrolls loops based on ttl.unroll_factor (optional)
  // 4. ttl-insert-tile-regs-sync: Inserts DST lifecycle ops (acquire/commit/wait/release)
  //    - Runs AFTER unrolling so sync ops wrap batches of tiles correctly
  //    - Works on both ttl.compute (if present) and scf.for loops
  pm.addPass(createTTLTileAndAssignDST());
  pm.addPass(createTTLLowerToLoops());
  if (options.enableUnroll) {
    pm.addPass(createTTLUnrollComputeLoops());
  }
  pm.addPass(createTTLInsertTileRegsSync());

  pm.addPass(createTTLAnnotateCBAssociations());
  pm.addPass(createTTLConvertTTLToTTKernel());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (options.lowerToEmitC) {
    pm.addPass(createLowerAffinePass());
    pm.addPass(::mlir::tt::createConvertTTKernelToEmitC());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(mlir::emitc::createFormExpressionsPass());
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
