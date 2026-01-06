// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

#include "ttmlir/Conversion/ArithToD2MTileOps/ArithToD2MTileOps.h"
#include "ttmlir/Conversion/D2MToTTKernel/D2MToTTKernel.h"
#include "ttmlir/Conversion/D2MToTTMetal/D2MToTTMetal.h"
#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"
#include "ttmlir/Conversion/MathToD2MTileOps/MathToD2MTileOps.h"
#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Conversion/TTIRToD2M/TTIRToD2M.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::tt::ttmetal {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

// translates top level flags into specific disable/enable patterns for
// canonicalizer pass
std::unique_ptr<Pass> createCanonicalizerPassWithOptions(
    const TTIRToTTMetalPipelineOptions &options) {
  llvm::SmallVector<std::string, 2> disabledPatterns;
  if (options.disableToLayoutFolding) {
    disabledPatterns.push_back("ttir.ToLayoutFoldRedundantPattern");
    disabledPatterns.push_back("d2m.ToLayoutFoldRedundantPattern");
  }
  return mlir::createCanonicalizerPass({}, disabledPatterns);
}

void createTTIRBufferizationPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  bufferization::OneShotBufferizePassOptions bufferizePassOptions;
  if (options.ttnnMode) {
    bufferizePassOptions.allowUnknownOps = true;
    bufferizePassOptions.bufferizeFunctionBoundaries = false;
  } else {
    bufferizePassOptions.allowUnknownOps = false;
    bufferizePassOptions.bufferizeFunctionBoundaries = true;
  }
  bufferizePassOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  bufferizePassOptions.unknownTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizePassOptions));
}

void createOptimizationPasses(OpPassManager &pm,
                              const TTIRToTTMetalPipelineOptions &options) {
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
}

void createTTIRToTTMetalFrontendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  // Create multi-device tensor annotation for graph with mesh.
  pm.addPass(ttir::createTTIRMultiDeviceTensorAnnotation());
  ttcore::TTCoreRegisterDevicePassOptions registerDeviceOptions;
  {
    registerDeviceOptions.systemDescPath = options.systemDescPath;
    registerDeviceOptions.mockSystemDescArch = options.mockSystemDescArch;
    registerDeviceOptions.meshShape = llvm::to_vector(options.meshShape);
  }
  pm.addPass(ttcore::createTTCoreRegisterDevicePass(registerDeviceOptions));
  pm.addPass(::mlir::tt::createTTIRToTTIRDecompositionPass());
  pm.addPass(createCanonicalizerPassWithOptions(options));
  ::mlir::tt::TTIRToD2MOptions toD2MOptions;
  {
    toD2MOptions.defaultInputMemSpace = options.defaultInputMemSpace;
    toD2MOptions.defaultOutputMemSpace = options.defaultOutputMemSpace;
    toD2MOptions.ttnnMode = options.ttnnMode;
    toD2MOptions.collapseTensorsTo2D = options.collapseTensors;
  }
  pm.addPass(::mlir::tt::createTTIRToD2MPass(toD2MOptions));
  d2m::D2MGridSelectionOptions gridOptOptions;
  {
    gridOptOptions.overrideDeviceShape =
        llvm::to_vector(options.overrideDeviceShape);
  }
  pm.addPass(d2m::createD2MGridSelection(gridOptOptions));
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(d2m::createD2MLowerToLayout());
}

void createTTIRToTTMetalMiddleendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  d2m::D2MElementwiseFusionOptions elementwiseFusionOptions;
  {
    elementwiseFusionOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
  }
  pm.addPass(d2m::createD2MElementwiseFusion(elementwiseFusionOptions));
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  createTTIRBufferizationPipeline(pm, options);
  if (options.ttnnMode) {
    d2m::D2MInsertStreamsOptions insertStreamsOptions;
    {
      insertStreamsOptions.numStreamBuffers = options.numStreamBuffers;
      insertStreamsOptions.allowL1OutputSpilling =
          options.allowL1OutputSpilling;
    }
    pm.addPass(d2m::createD2MInsertStreams(insertStreamsOptions));
  } else {
    d2m::D2MAllocateOptions allocateOptions;
    {
      allocateOptions.numStreamBuffers = options.numStreamBuffers;
      allocateOptions.allowL1OutputSpilling = options.allowL1OutputSpilling;
    }
    pm.addPass(d2m::createD2MAllocate(allocateOptions));
  }

  pm.addPass(createCanonicalizerPassWithOptions(options));
  d2m::D2MGenericApplyInterchangeOptions applyInterchangeOptions;
  {
    applyInterchangeOptions.matmulInterchange =
        llvm::to_vector(options.matmulInterchange);
  }
  pm.addPass(d2m::createD2MGenericApplyInterchange(applyInterchangeOptions));
  d2m::D2MGenericTileComputeLoopsOptions tileComputeLoopsOptions;
  {
    tileComputeLoopsOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
  }
  pm.addPass(d2m::createD2MGenericTileComputeLoops(tileComputeLoopsOptions));
  d2m::D2MInsertDstRegisterAccessOptions insertDstRegisterAccessOptions;
  {
    insertDstRegisterAccessOptions.useTileMatmul = options.useTileMatmul;
    insertDstRegisterAccessOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
  }
  pm.addPass(
      d2m::createD2MInsertDstRegisterAccess(insertDstRegisterAccessOptions));

  pm.addPass(d2m::createD2MSFPUTileLoopFission());
  pm.addPass(mlir::createCanonicalizerPass());

  OpPassManager &funcPm = pm.nest<func::FuncOp>();
  funcPm.addPass(affine::createAffineLoopInvariantCodeMotionPass());

  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(d2m::createD2MGenericLinearizeMemref());
  pm.addPass(d2m::createD2MGenericGenerateDatamovement());
  pm.addPass(d2m::createD2MGenericLowerDMAs());
  pm.addPass(d2m::createD2MGenericHWThreadSelection());
  pm.addPass(d2m::createD2MGenericGenerateLoops());
  createOptimizationPasses(pm, options);
  pm.addPass(d2m::createD2MGenericRegionsToFuncs());
}

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  d2m::ConvertD2MToTTKernelOptions D2MToTTKernelOptions;
  { D2MToTTKernelOptions.ttnnMode = options.ttnnMode; }
  pm.addPass(::mlir::tt::createConvertD2MToTTKernelPass(D2MToTTKernelOptions));
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(ttkernel::createTTKernelControlDstSection());
  createOptimizationPasses(pm, options);
  if (options.ttnnMode) {
    // TODO(#5075): set MathFidelity of ttnn generic op.
    pm.addPass(::mlir::tt::createConvertD2MToTTNNPass());
  } else {
    d2m::ConvertD2MToTTMetalOptions d2mToTTMetalOptions;
    { d2mToTTMetalOptions.mathFidelity = options.mathFidelity; }
    pm.addPass(::mlir::tt::createConvertD2MToTTMetalPass(d2mToTTMetalOptions));
  }
  pm.addPass(ttkernel::createTTKernelHoistInits());
  // Insert DeviceZone scopes around selected ttkernel ops before EmitC
  // lowering.
  if (options.insertProfilerTraces) {
    pm.addPass(ttkernel::createTTKernelInsertDeviceZoneScopes());
  }
  pm.addPass(::mlir::tt::createConvertTTKernelToEmitC());
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(mlir::emitc::createFormExpressionsPass());
}

void createTTIRToTTMetalPipeline(OpPassManager &pm,
                                 const TTIRToTTMetalPipelineOptions &options) {
  // Create DeviceModule to wrap all ops.
  pm.addPass(ttcore::createTTCoreWrapDeviceModulePass());
  // Hoist manually tagged ops to CPU module (if any).
  pm.addPass(ttir::createCPUHoistManuallyTaggedOpsTransform());

  // Run regular ttir to ttmetal pipelines on IR in DeviceModule.
  OpPassManager &devicePm =
      pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();
  createTTIRToTTMetalFrontendPipeline(devicePm, options);
  createTTIRToTTMetalMiddleendPipeline(devicePm, options);
  createTTIRToTTMetalBackendPipeline(devicePm, options);

  // Run lowering to LLVM pass on hoisted funcs in CPUModule.
  ttir::TTIRToLLVMCPUPipelineOptions ttirToCPUOptions;
  ttir::createTTIRToLLVMCPUPipeline(pm, ttirToCPUOptions);
}

void createD2MToTTMetalPipeline(OpPassManager &pm,
                                const TTIRToTTMetalPipelineOptions &options) {
  pm.addPass(d2m::createD2MLowerToLayout());

  createTTIRBufferizationPipeline(pm, options);

  d2m::D2MAllocateOptions allocateOptions;
  {
    allocateOptions.numStreamBuffers = options.numStreamBuffers;
    allocateOptions.allowL1OutputSpilling = options.allowL1OutputSpilling;
  }
  pm.addPass(d2m::createD2MAllocate(allocateOptions));

  // Apply tiling and register access transformation passes.
  d2m::D2MGenericTileComputeLoopsOptions tileComputeLoopsOptions;
  {
    tileComputeLoopsOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
  }
  pm.addPass(d2m::createD2MGenericTileComputeLoops(tileComputeLoopsOptions));
  d2m::D2MInsertDstRegisterAccessOptions insertDstRegisterAccessOptions;
  {
    insertDstRegisterAccessOptions.useTileMatmul = options.useTileMatmul;
    insertDstRegisterAccessOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
  }
  pm.addPass(
      d2m::createD2MInsertDstRegisterAccess(insertDstRegisterAccessOptions));

  // Canonicalize with normal simplification mode.
  pm.addPass(createCanonicalizerPassWithOptions(options));

  // Lower affine operations and linearize memref types.
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(d2m::createD2MGenericLinearizeMemref());
  pm.addPass(mlir::createLowerAffinePass());

  // Lower DMAs and generate computation loops.
  pm.addPass(d2m::createD2MGenericLowerDMAs());
  pm.addPass(d2m::createD2MGenericHWThreadSelection());
  pm.addPass(d2m::createD2MGenericGenerateLoops());

  // Canonicalize with aggressive simplification mode.
  pm.addPass(createCanonicalizerPassWithOptions(options));

  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(d2m::createD2MGenericRegionsToFuncs());

  d2m::ConvertD2MToTTKernelOptions D2MToTTKernelOptions;
  { D2MToTTKernelOptions.ttnnMode = options.ttnnMode; }
  pm.addPass(::mlir::tt::createConvertD2MToTTKernelPass(D2MToTTKernelOptions));

  // Apply TTKernel control transformations and additional optimizations.
  pm.addPass(ttkernel::createTTKernelControlDstSection());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());

  d2m::ConvertD2MToTTMetalOptions d2mToTTMetalOptions;
  { d2mToTTMetalOptions.mathFidelity = options.mathFidelity; }
  pm.addPass(::mlir::tt::createConvertD2MToTTMetalPass(d2mToTTMetalOptions));

  pm.addPass(::mlir::tt::createConvertTTKernelToEmitC());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTMetalPipelines() {
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttir-to-ttmetal-pipeline", "Pipeline lowering ttir to ttmetal.",
      tt::ttmetal::createTTIRToTTMetalPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttir-to-ttmetal-fe-pipeline", "Frontend lowering passes.",
      tt::ttmetal::createTTIRToTTMetalFrontendPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttir-to-ttmetal-me-pipeline", "Middleend lowering passes.",
      tt::ttmetal::createTTIRToTTMetalMiddleendPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttir-to-ttmetal-be-pipeline", "Backend lowering passes.",
      tt::ttmetal::createTTIRToTTMetalBackendPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttir-bufferization-pipeline",
      "Pipeline bufferizing ttir ops on tensors to ops on buffers (memrefs).",
      tt::ttmetal::createTTIRBufferizationPipeline);
}
} // namespace mlir::tt::ttmetal
