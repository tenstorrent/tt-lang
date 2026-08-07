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

static void
addProducerComputeCreation(OpPassManager &pm,
                           const TTLToTTKernelPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(createTTLProducerComputeCreation());
  TTLInsertIntermediateDFBsOptions dfbOptions;
  dfbOptions.enable = options.compilerDFBs;
  pm.addNestedPass<func::FuncOp>(createTTLInsertIntermediateDFBs(dfbOptions));
  pm.addNestedPass<func::FuncOp>(createTTLConvertTTLToCompute());
}

static void addDFBFinalization(OpPassManager &pm,
                               const TTLToTTKernelPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(createTTLInsertCBSync());
  // Verify the complete high-level schedule while logical DFB identities are
  // still distinct and before later transformations rewrite pipe operations.
  buildTTLVerifyPipeNetPipeline(pm);
  pm.addNestedPass<func::FuncOp>(createTTLCoalesceDFBAcquires());
  TTLFinalizeDFBIndicesOptions finalizeOptions;
  finalizeOptions.reuseUserDFBs = options.reuseUserDFBs;
  finalizeOptions.exactColoringSearchStateLimit =
      options.exactColoringSearchStateLimit;
  pm.addPass(createTTLFinalizeDFBIndices(finalizeOptions));
}

static void addComputePipelineScheduleSelection(
    OpPassManager &pm, const TTLToTTKernelPipelineOptions &options) {
  TTLSelectComputePipelineSchedulesOptions configOptions;
  configOptions.reduceFullFp32 = options.reduceFullFp32;
  configOptions.matmulFullFp32 = options.matmulFullFp32;
  configOptions.enableFPUBinaryOps = options.enableFPUBinaryOps;
  pm.addNestedPass<func::FuncOp>(
      createTTLSelectComputePipelineSchedules(configOptions));
}

static void addKernelConfig(OpPassManager &pm,
                            const TTLToTTKernelPipelineOptions &options) {
  TTLSetComputeKernelConfigOptions configOptions;
  configOptions.reduceFullFp32 = options.reduceFullFp32;
  configOptions.matmulFullFp32 = options.matmulFullFp32;
  configOptions.enableFPUBinaryOps = options.enableFPUBinaryOps;
  pm.addNestedPass<func::FuncOp>(
      createTTLSetComputeKernelConfig(configOptions));
}

void createTTLToTTKernelPipeline(OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(createTTLLowerComputePipelines());
  pm.addNestedPass<func::FuncOp>(createTTLMaterializeLoopState());
  pm.addNestedPass<func::FuncOp>(createTTLInsertCopyWait());
  pm.addNestedPass<func::FuncOp>(createTTLAnnotateL1AccLoops());
  addProducerComputeCreation(pm, options);
  addDFBFinalization(pm, options);
  addComputePipelineScheduleSelection(pm, options);

  // Schedule selection retains semantic pipelines until target and kernel
  // constraints are known without publishing DFB-index-derived attributes.
  pm.addNestedPass<func::FuncOp>(createTTLLowerComputePipelines());
  addProducerComputeCreation(pm, options);
  addDFBFinalization(pm, options);
  addKernelConfig(pm, options);

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
    ttkOpts.pipeComputedAddresses = options.pipeComputedAddresses;
    ttkOpts.pipeCapacitySync = options.pipeCapacitySync;
    ttkOpts.pipeGlobalSemaphoresOnly = options.pipeGlobalSemaphoresOnly;
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
