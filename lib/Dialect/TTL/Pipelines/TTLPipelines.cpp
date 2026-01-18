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

  // Annotate binary operations with execution strategy (FPU vs SFPU).
  // This must run before ttl-assign-dst which consumes the annotations.
  pm.addPass(createTTLAnnotateBinaryOpStrategy());

  // DST register assignment and synchronization (strict ordering required):
  // 1. ttl-assign-dst: DST allocation with linear scan and unary merging.
  //    Inserts copy_tile for block args, copy_dst for multi-consumer values,
  //    and assigns dst_idx attributes.
  // 2. ttl-insert-tile-regs-sync: Inserts DST lifecycle ops
  //    (acquire/commit/wait/release). These must run before TTKernel lowering
  //    and in this specific order.
  pm.addPass(createTTLAssignDST());
  pm.addPass(createTTLInsertTileRegsSync());
  pm.addPass(createTTLLowerToLoops());
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
