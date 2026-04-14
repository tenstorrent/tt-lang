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
  pm.addPass(createTTLAnnotateL1AccLoops());
  pm.addPass(createTTLConvertTTLToCompute());
  {
    TTLSetComputeKernelConfigOptions configOpts;
    configOpts.reduceFullFp32 = options.reduceFullFp32;
    pm.addPass(createTTLSetComputeKernelConfig(configOpts));
  }
  {
    TTLAssignDSTOptions assignDstOpts;
    assignDstOpts.enableFPUBinaryOps = options.enableFPUBinaryOps;
    pm.addPass(createTTLAssignDST(assignDstOpts));
  }
  if (options.maximizeDST) {
    TTLSubblockComputeForDSTOptions subblockOpts;
    subblockOpts.subblockSync = options.autoSync;
    subblockOpts.strictF32Acc = options.strictF32Acc;
    pm.addPass(createTTLSubblockComputeForDST(subblockOpts));
  }
  if (options.useBlockMatmul) {
    pm.addPass(createTTLLowerMatmulBlock());
  }
  {
    TTLLowerToLoopsOptions loopOpts;
    loopOpts.dstAccumulation = options.maximizeDST;
    pm.addPass(createTTLLowerToLoops(loopOpts));
  }
  if (options.maximizeDST) {
    pm.addPass(createTTLScheduleOperations());
  }
  pm.addPass(createTTLAnnotateCBAssociations());
  {
    TTLConvertTTLToTTKernelOptions ttkOpts;
    ttkOpts.reduceFullFp32 = options.reduceFullFp32;
    pm.addPass(createTTLConvertTTLToTTKernel(ttkOpts));
  }
  pm.addPass(createTTKernelInsertInits());
  pm.addPass(createTTKernelInsertL1Accumulation());
  if (options.combinePackTiles) {
    pm.addNestedPass<func::FuncOp>(createTTKernelCombinePackTiles());
  }
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
