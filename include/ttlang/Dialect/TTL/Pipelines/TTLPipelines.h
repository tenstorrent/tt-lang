// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_PIPELINES_TTLPIPLINES_H
#define TTLANG_DIALECT_TTL_PIPELINES_TTLPIPLINES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace mlir::tt::ttl {

struct TTLToTTKernelPipelineOptions
    : public mlir::PassPipelineOptions<TTLToTTKernelPipelineOptions> {
  Option<bool> lowerToEmitC{*this, "lower-to-emitc",
                            llvm::cl::desc("Lower TTKernel to EmitC."),
                            llvm::cl::init(false)};
  Option<bool> allowUnknownBufferizationOps{
      *this, "allow-unknown-bufferization-ops",
      llvm::cl::desc("Allow unknown ops during the TTL bufferization stage. "
                     "Disable only if downstream passes guarantee every TTL op "
                     "implements BufferizableOpInterface."),
      llvm::cl::init(true)};
};

void createTTLToTTKernelPipeline(mlir::OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &options);

void registerTTLPipelines();

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_PIPELINES_TTLPIPLINES_H
