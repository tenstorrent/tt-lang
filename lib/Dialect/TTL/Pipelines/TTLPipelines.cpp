// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Pipelines/TTLPipelines.h"

#include "ttlang/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir::tt::ttl {

void createTTLToTTKernelPipeline(OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(createTTLMaterializeLoopState());
  pm.addNestedPass<func::FuncOp>(createTTLInsertCopyWait());
  pm.addNestedPass<func::FuncOp>(createTTLAnnotateL1AccLoops());
  pm.addNestedPass<func::FuncOp>(createTTLProducerComputeCreation());
  {
    TTLInsertIntermediateDFBsOptions dfbOpts;
    dfbOpts.enable = options.compilerDFBs;
    pm.addNestedPass<func::FuncOp>(createTTLInsertIntermediateDFBs(dfbOpts));
  }
  pm.addNestedPass<func::FuncOp>(createTTLConvertTTLToCompute());
  pm.addNestedPass<func::FuncOp>(createTTLInsertCBSync());
  // Verify the complete high-level schedule while logical DFB identities are
  // still distinct and before later transformations rewrite pipe operations.
  buildTTLVerifyPipeNetPipeline(pm);
  pm.addNestedPass<func::FuncOp>(createTTLCoalesceDFBAcquires());
  pm.addPass(createTTLFinalizeDFBIndices());
  {
    TTLSetComputeKernelConfigOptions configOpts;
    configOpts.reduceFullFp32 = options.reduceFullFp32;
    configOpts.enableFPUBinaryOps = options.enableFPUBinaryOps;
    pm.addNestedPass<func::FuncOp>(createTTLSetComputeKernelConfig(configOpts));
  }
  pm.addNestedPass<func::FuncOp>(createTTLAssignDST());
  if (options.maximizeDST) {
    TTLSubblockComputeForDSTOptions subblockOpts;
    subblockOpts.subblockSync = options.subblockSync;
    subblockOpts.strictF32Acc = options.strictF32Acc;
    pm.addNestedPass<func::FuncOp>(
        createTTLSubblockComputeForDST(subblockOpts));
  }
  {
    TTLLowerToLoopsOptions loopOpts;
    loopOpts.dstAccumulation = options.maximizeDST;
    loopOpts.useBlockMatmul = options.useBlockMatmul;
    pm.addNestedPass<func::FuncOp>(createTTLLowerToLoops(loopOpts));
  }
  if (options.maximizeDST) {
    pm.addNestedPass<func::FuncOp>(createTTLScheduleOperations());
  }
  pm.addNestedPass<func::FuncOp>(createTTLAnnotateCBAssociations());
  pm.addPass(createTTLVerifyDFBSPSC());
  pm.addPass(createTTLErasePipeNetScopes());
  pm.addPass(createTTLValidateCBBudget());
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
  if (options.specializeCores) {
    pm.addPass(createTTKernelSpecializeCores());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
  if (options.lowerToEmitC) {
    pm.addPass(createLowerAffinePass());
    pm.addNestedPass<func::FuncOp>(::mlir::tt::createConvertTTKernelToEmitC());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(mlir::emitc::createFormExpressionsPass());
  }
}

void buildTTLVerifyPipeNetPipeline(OpPassManager &pm) {
  pm.addPass(createTTLVerifyPipeNetGuards());
  pm.addPass(createTTLVerifyPipeNetSchedule());
}

void buildTTLAutoSyncPipeline(OpPassManager &pm) {
  pm.addPass(createTTLInsertCBSync());
  pm.addPass(createTTLCoalesceDFBAcquires());
}

void registerTTLPipelines() {
  PassPipelineRegistration<TTLToTTKernelPipelineOptions>(
      "ttl-to-ttkernel-pipeline",
      "Lower TTL to TTKernel, run cleanup canonicalization/CSE, and optionally "
      "lower TTKernel to EmitC.",
      createTTLToTTKernelPipeline);
  PassPipelineRegistration<>(
      "ttl-verify-pipenet",
      "Verify PipeNet launch domains and synchronization schedules.",
      buildTTLVerifyPipeNetPipeline);
  PassPipelineRegistration<>("ttl-auto-sync",
                             "Insert auto pop/push and coalesce DFB acquires.",
                             buildTTLAutoSyncPipeline);
}

} // namespace mlir::tt::ttl
