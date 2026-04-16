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
  Option<bool> maximizeDST{
      *this, "maximize-dst",
      llvm::cl::desc("Enable DST maximization via subblock compute."),
      llvm::cl::init(true)};
  Option<bool> enableFPUBinaryOps{
      *this, "enable-fpu-binary-ops",
      llvm::cl::desc("Use FPU for binary add/sub/mul."), llvm::cl::init(true)};
  Option<bool> useBlockMatmul{
      *this, "use-block-matmul",
      llvm::cl::desc("Lower matmul to block-level hardware calls "
                     "(experimental::matmul_block) instead of per-tile loops."),
      llvm::cl::init(true)};
  Option<bool> subblockSync{
      *this, "subblock-sync",
      llvm::cl::desc("Refine DFB reserve/push to per-subblock granularity. "
                     "When disabled (default), user-placed reserve/push "
                     "is preserved."),
      llvm::cl::init(false)};
  Option<bool> combinePackTiles{
      *this, "combine-pack-tiles",
      llvm::cl::desc("Combine consecutive pack_tile ops into pack_tile_block."),
      llvm::cl::init(true)};
  Option<bool> reduceFullFp32{
      *this, "reduce-full-fp32",
      llvm::cl::desc("Enable FP32 accumulation for reduce operations."),
      llvm::cl::init(true)};
  Option<bool> strictF32Acc{
      *this, "strict-f32-acc",
      llvm::cl::desc("Error if accumulation output exceeds f32 DST capacity."),
      llvm::cl::init(false)};
};

void createTTLToTTKernelPipeline(mlir::OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &options);

void registerTTLPipelines();

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_PIPELINES_TTLPIPLINES_H
