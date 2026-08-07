// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_PIPELINES_TTLPIPLINES_H
#define TTLANG_DIALECT_TTL_PIPELINES_TTLPIPLINES_H

#include "mlir/Pass/PassOptions.h"

#include <cstdint>

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
      llvm::cl::desc("Allow FPU strategy selection for binary add/sub/mul."),
      llvm::cl::init(true)};
  Option<bool> useBlockMatmul{
      *this, "use-block-matmul",
      llvm::cl::desc("Lower matmul to block-level hardware calls "
                     "(matmul_block) instead of per-tile loops."),
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
      llvm::cl::desc("Prefer FP32 accumulation for reduce operations."),
      llvm::cl::init(true)};
  Option<bool> matmulFullFp32{
      *this, "matmul-full-fp32",
      llvm::cl::desc("Prefer FP32 accumulation for matmul operations."),
      llvm::cl::init(true)};
  Option<bool> strictF32Acc{
      *this, "strict-f32-acc",
      llvm::cl::desc("Error if accumulation output exceeds f32 DST capacity."),
      llvm::cl::init(false)};
  Option<bool> compilerDFBs{
      *this, "compiler-dfbs",
      llvm::cl::desc("Insert compiler-allocated intermediate DFBs when "
                     "materialization is required. When disabled, emit an "
                     "error if materialization through a compiler-allocated "
                     "DFB is required."),
      llvm::cl::init(true)};
  Option<bool> pipeComputedAddresses{
      *this, "pipe-computed-addresses",
      llvm::cl::desc("Use computed receiver DFB addresses for eligible pipe "
                     "transfers."),
      llvm::cl::init(true)};
  Option<bool> pipeCapacitySync{
      *this, "pipe-capacity-sync",
      llvm::cl::desc("Use capacity-counter synchronization for eligible pipe "
                     "transfers. When disabled, computed-address transfers "
                     "use receiver-post synchronization."),
      llvm::cl::init(true)};
  Option<bool> pipeGlobalSemaphoresOnly{
      *this, "pipe-global-semaphores-only",
      llvm::cl::desc("Allocate all compiler-managed PipeNet synchronization "
                     "counters in GlobalSemaphore storage."),
      llvm::cl::init(false)};
  Option<bool> reuseUserDFBs{
      *this, "reuse-user-dfbs",
      llvm::cl::desc("Reuse physical DFB indices when concurrent-kernel "
                     "liveness proves that logical lifetimes do not overlap."),
      llvm::cl::init(true)};
  Option<std::uint64_t> exactColoringSearchStateLimit{
      *this, "exact-coloring-search-limit",
      llvm::cl::desc("Maximum states examined by exact DFB allocation before "
                     "reporting an inconclusive result."),
      llvm::cl::init(1000000)};
  Option<bool> specializeCores{
      *this, "specialize-cores",
      llvm::cl::desc(
          "Clone TTKernel functions that branch on a core coordinate once "
          "per launch coordinate (ttkernel-specialize-cores)."),
      llvm::cl::init(false)};
};

void createTTLToTTKernelPipeline(mlir::OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &options);

/// Add DFB synchronization insertion and acquire coalescing passes.
void buildTTLAutoSyncPipeline(mlir::OpPassManager &pm);

/// Add the ordered PipeNet launch-domain and synchronization verifiers.
void buildTTLVerifyPipeNetPipeline(mlir::OpPassManager &pm);

void registerTTLPipelines();

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_PIPELINES_TTLPIPLINES_H
