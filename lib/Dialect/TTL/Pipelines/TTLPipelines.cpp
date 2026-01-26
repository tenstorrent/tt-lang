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
  // DST register assignment and synchronization (strict ordering required):
  // 1. ttl-assign-dst: DST allocation with linear scan and unary merging.
  //    Inserts copy_tile for block args, copy_dst for multi-consumer values,
  //    and assigns dst_idx attributes.
  // 2. ttl-lower-to-loops: Lowers ttl.compute to scf.for loops and marks the
  //    innermost loop with ttl.tile_loop attribute for sync insertion.
  // 3. ttl-insert-inter-loop-cb-sync: Inserts cb_wait between consecutive loops
  //    when the output CB of one loop feeds into the input CB of the next.
  // 4. ttl-insert-tile-regs-sync: Inserts DST lifecycle ops
  //    (acquire/commit/wait/release) inside the marked loops.
  pm.addPass(createTTLAssignDST());
  pm.addPass(createTTLLowerToLoops());
  pm.addPass(createTTLInsertInterLoopCBSync());
  pm.addPass(createTTLInsertTileRegsSync());
  pm.addPass(createTTLAnnotateCBAssociations());
  // Convert TTL broadcast ops directly to EmitC BEFORE TTL→TTKernel conversion.
  // Broadcast ops need special intrinsics that are emitted directly as EmitC.
  // Must run before TTLConvertTTLToTTKernel which marks TTL tile ops as
  // illegal.
  pm.addPass(createTTLConvertBcastToEmitC());
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
