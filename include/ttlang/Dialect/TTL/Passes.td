#ifndef TTLANG_DIALECT_TTL_PASSES_TD
#define TTLANG_DIALECT_TTL_PASSES_TD

include "mlir/Pass/PassBase.td"

def TTLConvertTTLToTTKernel
    : Pass<"convert-ttl-to-ttkernel", "::mlir::ModuleOp"> {
  let summary = "Lower TTL DMA ops to TTKernel using global barriers (temporary)";
  let description = [{
    Converts TTL DMA ops to TTKernel noc ops. Uses global barriers until TRID
    barriers are available. Covers bind_cb, copy, wait MVP path.

    TODO(ttl): Refine lowering to emit real CB handles and proper NOC addresses.
    Issue: #77 (umbrella issue with subtasks #78-#89).
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTLConvertTTLToCompute
    : Pass<"convert-ttl-to-compute", "::mlir::func::FuncOp"> {
  let summary = "Lower TTL elementwise tensor ops to ttl.compute with tile ops";
  let description = [{
    Lowers TTL elementwise tensor operations (e.g., add, mul, exp, relu) to
    ttl.compute with SSA-style tile operations in the body. This enables:
    - Fusion of matmul + elementwise operations in the same compute body
    - DST register tiling via TilingInterface
    - Standard SSA optimizations (CSE, DCE, canonicalization)
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def TTLAssignDST
    : Pass<"ttl-assign-dst", "::mlir::func::FuncOp"> {
  let summary = "DST allocation with copy insertion, interval analysis, and linear scan";
  let description = [{
    DST register allocator using linear scan allocation with in-place operation
    merging. Implements the algorithm from DST_Allocation.md:

    Phase 1: Copy insertion for multi-consumer values
    - When a value has multiple consumers and any is in-place (overwrites input),
      inserts ttl.copy_dst operations for all but the last consumer

    Phase 2: Build live intervals with in-place merging
    - Builds interval [start, end] for each tile value
    - Merges intervals for in-place ops (input and output share same DST)

    Phase 3: Linear scan allocation for inputs/intermediates
    - Allocates DST[0..k-1] for input block arguments and intermediate values
    - Uses interval-based linear scan (Wimmer & Franz, CGO'10)

    Phase 4: Linear scan allocation for outputs
    - Allocates DST[k..n] for values yielded from the compute body
    - Separate region ensures outputs aren't overwritten during computation

    This pass also:
    - Inserts ttl.copy_tile for block arguments
    - Assigns dst_idx attributes to tile math operations
    - Checks capacity and emits diagnostics on overflow
  }];

  let options = [
    Option<"dstCapacity", "dst-capacity", "uint32_t", "0",
           "Override DST register capacity (auto-computed by default). "
           "For single-buffering mode (dst_full_sync_en=true), use: "
           "f32=8, f16/bf16=16. For double-buffering (default): f32=4, f16/bf16=8.">,
    Option<"separateOutputRegion", "separate-output-region", "bool", "false",
           "Allocate outputs in separate DST region (needed for reductions and some loop optimizations).">,
    Option<"enableFPUBinaryOps", "enable-fpu-binary-ops", "bool", "true",
           "Use FPU for binary add/sub/mul when both operands come from CBs. "
           "When disabled, binary ops use the SFPU path instead.">
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLSetComputeKernelConfig
    : Pass<"ttl-set-compute-kernel-config", "::mlir::func::FuncOp"> {
  let summary = "Set ttl.compute kernel configuration attributes";
  let description = [{
    Sets fp32_dest_acc_en and dst_full_sync_en attributes on ttl.compute
    operations when they are not already configured. This allows downstream
    passes to use consistent compute kernel configuration decisions.
  }];

  let options = [
    Option<"fp32DestAccEn", "fp32-dest-acc-en", "bool", "false",
           "Set default fp32_dest_acc_en when not already configured.">,
    Option<"dstFullSyncEn", "dst-full-sync-en", "bool", "false",
           "Set default dst_full_sync_en when not already configured.">
  ];

  let dependentDialects = [
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLSubblockComputeForDST
    : Pass<"ttl-subblock-compute-for-dst", "::mlir::func::FuncOp"> {
  let summary = "Partition ttl.compute into DST-sized subblocks";
  let description = [{
    Partitions the iteration space of `ttl.compute` operations into subblocks
    that fit in the DST register file. Uses the `ttl.unroll_factor` attribute
    (set by `ttl-assign-dst`) to determine the subblock size.

    `ComputeOp` implements `TilingInterface`, and this pass tiles it along its
    iteration dimensions to produce smaller inner `ttl.compute` ops wrapped in
    `scf.for` loops -- analogous to tiling `linalg` structured ops.

    For each `ttl.compute` with `ttl.unroll_factor > 1`, generates an outer
    `scf.for` loop with `tensor.extract_slice` ops to create sub-tensors for
    each subblock. The inner `ttl.compute` operates on the smaller sub-tensors
    and carries `ttl.full_linearization_strides` for index adjustment.

    Example (4x4 tensor, DST capacity 8, subblock 2x4):
    ```mlir
    // Before (16 tiles, unroll_factor=8):
    %r = ttl.compute ins(%a) outs(%init)
        {ttl.unroll_factor = 8 : i64} {
      ^bb0(%a_tile, %out_tile):
        %exp = ttl.tile_exp %a_tile
        ttl.yield %exp
    } -> tensor<4x4x!ttcore.tile<32x32, f32>>

    // After (outer loop over dim 0 in steps of 2):
    scf.for %iv = %c0 to %c4 step %c2 {
      %s_in  = tensor.extract_slice %a[%iv, 0] [2, 4] [1, 1]
      %s_out = tensor.extract_slice %init[%iv, 0] [2, 4] [1, 1]
      ttl.compute ins(%s_in) outs(%s_out)
          {ttl.full_linearization_strides = dense<[4, 1]>} {
        ^bb0(%a_tile, %out_tile):
          %exp = ttl.tile_exp %a_tile
          ttl.yield %exp
      } -> tensor<2x4x!ttcore.tile<32x32, f32>>
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def TTLInsertTileRegsSync
    : Pass<"ttl-insert-tile-regs-sync", "::mlir::func::FuncOp"> {
  let summary = "Insert tile_regs_* synchronization ops around ttl.compute";
  let description = [{
    Inserts DST register synchronization ops around `ttl.compute` to enforce
    the MATH/PACK thread synchronization protocol required by the hardware.

    Two placement modes based on whether the compute was subblocked:

    Subblocked (has `ttl.full_linearization_strides`): the subblock is
    guaranteed to fit in DST by construction (`ttl-subblock-compute-for-dst`
    tiles to DST capacity), so sync ops wrap the compute from outside. One
    acquire/commit/wait/release cycle covers all tiles in the subblock:
      acquire -> [compute over N tiles] -> commit -> wait -> [N stores] -> release

    Non-subblocked: sync ops are placed inside the compute body for per-tile
    synchronization. `tile_regs_commit` and `tile_regs_wait` are inserted
    before the first `ttl.tile_store`, and `tile_regs_release` before yield:
      acquire -> [compute 1 tile] -> commit -> wait -> [1 store] -> release
  }];

  let dependentDialects = [
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLLowerMatmulBlock
    : Pass<"ttl-lower-matmul-block", "::mlir::func::FuncOp"> {
  let summary = "Lower matmul computes to block-level hardware calls";
  let description = [{
    Marks `ttl.compute` ops containing `ttl.tile_matmul_block` for block-level
    lowering by setting the `ttl.block_matmul` attribute. This attribute tells
    `ttl-lower-to-loops` to collapse the iteration domain to a single point
    (no per-tile loops), since matmul_block handles the full block atomically.

    The pass also expands the single `tile_store` in the body into M*N stores
    (one per output tile in the block), since matmul_block writes M*N DST
    registers that each need to be packed.

    The output must fit in DST (M*N <= capacity). Subblocking for larger
    outputs is not yet implemented.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLLowerToLoops
    : Pass<"ttl-lower-to-loops", "::mlir::func::FuncOp"> {
  let summary = "Lower ttl.compute structured ops to scf.for loops";
  let description = [{
    Lowers `ttl.compute` operations to nested `scf.for` loops that iterate over
    the tile block. Generates the loop body with `tensor.extract` for input tiles,
    the cloned compute body, and `tensor.insert` to write results back.

    This pass runs BEFORE bufferization. After this pass:
    - `ttl.compute` ops are replaced by `scf.for` loops
    - Loop bodies contain `tensor.extract`, `ttl.tile_*` ops, and `tensor.insert`
    - All types remain tensor types (bufferization converts to memref later)

    Example transformation:
    ```mlir
    // Before:
    %result = ttl.compute ins(%a, %b) outs(%init) {
      ^bb0(%a_tile, %b_tile, %out_tile):
        %sum = ttl.tile_add %a_tile, %b_tile
        ttl.yield
    } -> tensor<2x2x!ttcore.tile<32x32, f32>>

    // After:
    %result = scf.for %i = 0 to 2 iter_args(%out = %init) {
      %inner = scf.for %j = 0 to 2 iter_args(%out_inner = %out) {
        %a_tile = tensor.extract %a[%i, %j]
        %b_tile = tensor.extract %b[%i, %j]
        %sum = ttl.tile_add %a_tile, %b_tile
        %updated = tensor.insert %sum into %out_inner[%i, %j]
        scf.yield %updated
      }
      scf.yield %inner
    } -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ```
  }];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def TTLScheduleOperations
    : Pass<"ttl-schedule-operations", "::mlir::func::FuncOp"> {
  let summary = "Group tile operations by kind within sync regions";
  let description = [{
    Reorders tile operations within sync regions (`tile_regs_acquire` to
    `tile_regs_commit`) to group by operation kind. After `ttl-lower-to-loops`
    unrolls a subblocked compute, tile ops from different iterations are
    interleaved. This pass groups them so `ttkernel-insert-inits` can emit
    one init per consecutive group of same-type ops.

    Ops are sorted by (dependency depth, category, TypeID, init affinity,
    dst_idx). Dependency depth considers RAW (SSA def-use), WAW (same DST
    write index), and WAR (write after prior read of same DST index) hazards.
    Ops at the same depth are independent and can be reordered.

    Only tile ops between `tile_regs_acquire` and `tile_regs_commit` are
    scheduled. Store ops (after the commit/wait boundary) are not reordered.

    Category ordering:
      CopyTile < Bcast < FPUBinary < SFPUUnary < SFPUBinary < CopyDst

    Example:
    ```mlir
    // Before (interleaved by tile iteration):
    ttl.tile_regs_acquire
    %s0 = ttl.tile_add %a0, %b0 {dst_idx = 0}
    %e0 = ttl.tile_exp %s0 {dst_idx = 0}
    %s1 = ttl.tile_add %a1, %b1 {dst_idx = 1}
    %e1 = ttl.tile_exp %s1 {dst_idx = 1}
    ttl.tile_regs_commit

    // After (grouped by kind at each dependency depth):
    ttl.tile_regs_acquire
    %s0 = ttl.tile_add %a0, %b0 {dst_idx = 0}  // depth 0
    %s1 = ttl.tile_add %a1, %b1 {dst_idx = 1}  // depth 0
    %e0 = ttl.tile_exp %s0 {dst_idx = 0}       // depth 1
    %e1 = ttl.tile_exp %s1 {dst_idx = 1}       // depth 1
    ttl.tile_regs_commit
    ```

    References:
    - Hennessy & Patterson, "Computer Architecture", Ch. 3: RAW/WAR/WAW hazards
    - Cooper & Torczon, "Engineering a Compiler", Ch. 12: list scheduling with
      topological depth levels
  }];

  let dependentDialects = [];
}

def TTLAnnotateCBAssociations
    : Pass<"ttl-annotate-cb-associations", "::mlir::func::FuncOp"> {
  let summary = "Annotate ttl.compute block args with CB index associations";
  let description = [{
    Analysis pass that annotates ttl.compute block arguments with the CB index
    (0-31) of their associated circular buffer. This enables subsequent conversion
    passes to look up the correct CB without fragile state management.

    For each ttl.compute input that is produced by ttl.attach_cb or ttl.cb_wait,
    this pass:
    1. Traces the input to find the associated CB via getAttachedCB()
    2. Extracts the cb_index attribute from the CB
    3. Annotates the corresponding block argument with ttl.cb_index = N

    This annotation survives subsequent IR transformations and allows copy_tile
    lowering to find the correct CB by cb_index lookup.
  }];

  let dependentDialects = [];
}

def TTLLowerDPrintToEmitC
    : Pass<"ttl-lower-dprint-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Lower ttl.dprint ops to EmitC verbatim for debug printing";
  let description = [{
    Converts ttl.dprint operations to EmitC verbatim code that emits
    device debug print calls. Resolves CB indices, DST register slots,
    and data formats from the IR. Runs before convert-ttl-to-ttkernel
    so that DST assignments and CB annotations are still available.
  }];
}

def TTLLowerSignpostToEmitC
    : Pass<"ttl-lower-signpost-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Lower ttl.signpost ops to EmitC verbatim for profiling";
  let description = [{
    Converts ttl.signpost operations to EmitC verbatim code that emits
    DeviceZoneScopedN markers for Tracy profiling. This pass runs after
    convert-ttl-to-ttkernel to keep profiling concerns separate from
    the main dialect conversion.
  }];
}

def TTLDumpCBFlowGraph
    : Pass<"ttl-dump-cb-flow-graph", "::mlir::ModuleOp"> {
  let summary = "Analyze and dump CB producer/consumer flow graph";
  let description = [{
    Analysis pass that builds a flow graph showing how data moves through
    circular buffers between kernels. For each CB, identifies:
    - Producers: operations that write to the CB (cb_reserve + copy/store)
    - Consumers: operations that read from the CB (cb_wait)
    - DMA operations: copy ops with their direction (read/write)
    - Wait operations: ttl.wait barriers

    Outputs the graph in a human-readable format and optionally as JSON
    for consumption by the auto-profiler. The graph maps source locations
    to CB operations, enabling correlation with runtime profiler data.

    Example output:
    ```
    CB Flow Graph:
    CB[0] "inp_cb":
      producers: dm_read:74 (copy DRAM->CB)
      consumers: compute:52 (cb_wait)
    CB[1] "out_cb":
      producers: compute:55 (cb_reserve)
      consumers: dm_write:106 (cb_wait)
    ```
  }];

  let options = [
    Option<"outputPath", "output", "std::string", "\"\"",
           "Path to write JSON output (empty for stderr debug only)">
  ];
}

def TTKernelInsertInits
    : Pass<"ttkernel-insert-inits", "::mlir::ModuleOp"> {
  let summary = "Insert minimal init ops before TTKernel compute ops";
  let description = [{
    Inserts hardware initialization ops required before TTKernel compute
    operations. Two categories of inits are inserted:

    Common inits (`init_sfpu`, `binary_op_init_common`) configure the
    UNPACK and PACK data format routing for a sync region. These are placed
    once per sync region, hoisted above any enclosing loops.

    Per-op inits (`exp_tile_init`, `add_tiles_init`, etc.) configure the
    MATH pipeline for a specific operation type. An init is emitted
    whenever the op type changes from the previous compute op in
    linear block order. Init tracking resets at sync boundaries
    (`tile_regs_acquire`/`commit`/`wait`/`release`).

    Example (consecutive same-type ops share one init):
    ```mlir
    // Before:
    ttkernel.tile_regs_acquire
    ttkernel.exp_tile(%c0)
    ttkernel.exp_tile(%c1)
    ttkernel.log_tile(%c2)
    ttkernel.tile_regs_commit

    // After:
    ttkernel.init_sfpu(%cb0, %cb1)
    ttkernel.tile_regs_acquire
    ttkernel.exp_tile_init
    ttkernel.exp_tile(%c0)
    ttkernel.exp_tile(%c1)      // no re-init (same type)
    ttkernel.log_tile_init       // re-init on type switch
    ttkernel.log_tile(%c2)
    ttkernel.tile_regs_commit
    ```

    Example (interleaved ops each get their own init):
    ```mlir
    // Before:
    ttkernel.tile_regs_acquire
    ttkernel.exp_tile(%c0)
    ttkernel.log_tile(%c1)
    ttkernel.exp_tile(%c2)
    ttkernel.tile_regs_commit

    // After:
    ttkernel.init_sfpu(%cb0, %cb1)
    ttkernel.tile_regs_acquire
    ttkernel.exp_tile_init
    ttkernel.exp_tile(%c0)
    ttkernel.log_tile_init
    ttkernel.log_tile(%c1)
    ttkernel.exp_tile_init       // re-init required (type switched back)
    ttkernel.exp_tile(%c2)
    ttkernel.tile_regs_commit
    ```
  }];

  let dependentDialects = [
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

#endif // TTLANG_DIALECT_TTL_PASSES_TD
