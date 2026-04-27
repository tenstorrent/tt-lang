#ifndef TTLANG_DIALECT_TTL_PASSES_TD
#define TTLANG_DIALECT_TTL_PASSES_TD

include "mlir/Pass/PassBase.td"

def TTLInsertCBSync
    : Pass<"ttl-insert-cb-sync", "::mlir::func::FuncOp"> {
  let summary = "Insert missing cb_push/cb_pop for unmatched cb_reserve/cb_wait";
  let description = [{
    Finds `ttl.cb_reserve` ops without a matching `ttl.cb_push` and
    `ttl.cb_wait` ops without a matching `ttl.cb_pop`, then inserts the
    missing release op after the last operation that accesses the CB's data.

    The pass computes a transitive use closure starting from the acquire op:
    direct uses of the CB value and the acquire result, plus uses of values
    those ops produce (e.g. copy -> transfer_handle -> wait). The release is
    inserted after the last op in this closure.

    Explicit push/pop calls and context-manager-generated push/pop are
    preserved: the pass only fills in what is missing.
  }];

  let dependentDialects = [];
}

def TTLInsertCopyWait
    : Pass<"ttl-insert-copy-wait", "::mlir::func::FuncOp"> {
  let summary = "Insert missing ttl.wait for unmatched ttl.copy ops";
  let description = [{
    Finds `ttl.copy` ops whose transfer handle result has no `ttl.wait` user
    and inserts a `ttl.wait` immediately after the copy. Explicit `.wait()`
    calls are preserved: the pass only fills in what is missing.
  }];

  let dependentDialects = [];
}

def TTLConvertTTLToTTKernel
    : Pass<"convert-ttl-to-ttkernel", "::mlir::ModuleOp"> {
  let summary = "Lower TTL DMA ops to TTKernel using global barriers (temporary)";
  let description = [{
    Converts TTL DMA ops to TTKernel noc ops. Uses global barriers until TRID
    barriers are available. Covers bind_cb, copy, wait MVP path.

    TODO(ttl): Refine lowering to emit real CB handles and proper NOC addresses.
    Issue: #77 (umbrella issue with subtasks #78-#89).
  }];

  let options = [
    Option<"reduceFullFp32", "reduce-full-fp32", "bool", "true",
           "Enable FP32 accumulation for reduce operations.">
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTKernelInsertL1Accumulation
    : Pass<"ttkernel-insert-l1-accumulation", "::mlir::ModuleOp"> {
  let summary = "Insert L1 accumulation guards for reduction loops";
  let description = [{
    Inserts `pack_reconfig_l1_acc` guards so that pack operations inside
    annotated loops accumulate into L1 instead of overwriting. Max-reduce
    loops are excluded (max is not additive).

    Loops are grouped into accumulation scopes: consecutive sibling loops
    that pack to the same CB share a single disable pair. Nested annotated
    loops are folded into the outermost ancestor. Each group gets:
    - `pack_reconfig_l1_acc(0)` before the first loop.
    - `if (iv == lb) pack_reconfig_l1_acc(1)` after each loop's first
      iteration (the L1 acc state persists across `tile_regs` boundaries).
    - For the second+ loop in a group, an enable before the loop to re-enable
      L1 acc after init ops that may reset packer state.
    - `pack_reconfig_l1_acc(0)` after the last `cb_push_back`.

    Loops are identified by the `ttl.l1_acc_loop` (user-written) or
    `ttl.reduction_loop` (compiler-generated) attributes on `scf.for`
    ops, with `ttl.l1_acc_loop` taking precedence.

  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTLAnnotateL1AccLoops
    : Pass<"ttl-annotate-l1-acc-loops", "::mlir::func::FuncOp"> {
  let summary = "Annotate user-written scf.for loops for L1 accumulation";
  let description = [{
    Detects user-written `scf.for` loops containing `ttl.store` ops with
    the `accumulate` attribute (emitted by the `+=` operator) and annotates
    them with `ttl.l1_acc_loop`. This enables `TTKernelInsertL1Accumulation`
    to insert `pack_reconfig_l1_acc` guards so that packs accumulate across
    iterations instead of overwriting.
  }];
  let dependentDialects = [
    "::mlir::scf::SCFDialect"
  ];
}

def TTLInsertIntermediateDFBs
    : Pass<"ttl-insert-intermediate-dfbs", "::mlir::func::FuncOp"> {
  let summary = "Insert intermediate dataflow buffers at fusion split points";
  let description = [{
    Walks ops implementing `DFBInputOpInterface` and checks whether the
    operands they declare as requiring DFB-attached values are satisfied.
    When an operand comes from a fused expression chain (not DFB-attached),
    the pass inserts a compiler-allocated DFB:

      bind_cb -> cb_reserve -> store -> cb_wait -> attach_cb

    `cb_push` and `cb_pop` are omitted; `ttl-insert-cb-sync` inserts them
    based on transitive use analysis.

    Compiler-allocated DFBs are marked with `ttl.compiler_allocated` so
    that `ttl-finalize-dfb-indices` can register them for the Python
    runtime.

    When `enable` is false, the pass verifies that no operations require
    compiler-allocated DFBs. If any `DFBInputOpInterface` op has a
    non-DFB-attached operand, the pass emits an error diagnostic and
    signals failure.
  }];

  let options = [
    Option<"enable", "enable", "bool", "true",
           "Insert compiler-allocated intermediate DFBs. When false, "
           "emit an error if any operation requires a compiler-allocated DFB.">
  ];

  let dependentDialects = [
    "::mlir::tt::ttl::TTLDialect"
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
    "::mlir::tensor::TensorDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
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
    - Assigns dst_index SSA operands to tile math operations
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
    Sets fp32_dest_acc_en and dst_full_sync_en attributes on the enclosing
    func.func when they are not already configured. These are per-kernel
    compile-time settings; all compute ops in the function inherit them
    via getKernelBoolAttr().
  }];

  let options = [
    Option<"fp32DestAccEn", "fp32-dest-acc-en", "bool", "false",
           "Set default fp32_dest_acc_en when not already configured.">,
    Option<"dstFullSyncEn", "dst-full-sync-en", "bool", "false",
           "Set default dst_full_sync_en when not already configured.">,
    Option<"reduceFullFp32", "reduce-full-fp32", "bool", "true",
           "Enable FP32 accumulation for reduce operations. "
           "Sets fp32_dest_acc_en on the kernel when any compute op contains reduce.">,
    Option<"matmulFullFp32", "matmul-full-fp32", "bool", "true",
           "Enable FP32 accumulation for matmul operations. "
           "Sets fp32_dest_acc_en on the kernel when any compute op contains matmul_block.">
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

  let options = [
    Option<"subblockSync", "subblock-sync", "bool", "false",
           "Refine DFB reserve/push to per-subblock granularity, enabling "
           "pack_tile_block for contiguous subblocks. When disabled (default), "
           "user-placed reserve/push is preserved.">,
    Option<"strictF32Acc", "strict-f32-acc", "bool", "false",
           "Error if a user-written accumulation loop (+=) with non-f32 "
           "output requires subblocking. Subblocking reduces accumulation "
           "precision because bf16 L1 intermediates are narrower than f32 DST.">
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
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

  let options = [
    Option<"dstAccumulation", "dst-accumulation", "bool", "false",
           "Use DST register accumulation for reduction loops. Reorders "
           "loops (parallel-outer, reduction-inner) so DST persists across "
           "reduction iterations. When disabled, uses per-tile L1 "
           "accumulation. reduce_max always uses DST accumulation.">,
    Option<"useBlockMatmul", "use-block-matmul", "bool", "true",
           "Lower matmul to block-level hardware calls "
           "(experimental::matmul_block) instead of per-tile loops.">
  ];

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
    dst_index). Dependency depth considers RAW (SSA def-use), WAW (same DST
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
    %s0 = ttl.tile_add %a0, %b0 into dst[%c0] : ...
    %e0 = ttl.tile_exp %s0 into dst[%c0] : ...
    %s1 = ttl.tile_add %a1, %b1 into dst[%c1] : ...
    %e1 = ttl.tile_exp %s1 into dst[%c1] : ...
    ttl.tile_regs_commit

    // After (grouped by kind at each dependency depth):
    ttl.tile_regs_acquire
    %s0 = ttl.tile_add %a0, %b0 into dst[%c0] : ...  // depth 0
    %s1 = ttl.tile_add %a1, %b1 into dst[%c1] : ...  // depth 0
    %e0 = ttl.tile_exp %s0 into dst[%c0] : ...       // depth 1
    %e1 = ttl.tile_exp %s1 into dst[%c1] : ...       // depth 1
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

def TTLFinalizeDFBIndices
    : Pass<"ttl-finalize-dfb-indices", "::mlir::ModuleOp"> {
  let summary = "Reuse compiler-allocated DFB indices and emit DFB metadata";
  let description = [{
    Module-level pass that runs after all DFB-creating passes. Must
    precede ttl-annotate-cb-associations (which copies cb_index onto
    tile ops and would capture stale pre-reuse indices).

    Applies lifetime-based index reuse to compiler-allocated DFBs via
    linear scan allocation, partitioned by CircularBufferType. DFBs
    whose lifetimes (bind_cb to last cb_pop) do not overlap are assigned
    the same physical index. See docs/development/DFBManagement.md.

    After reuse, verifies the total DFB count does not exceed
    kMaxCircularBuffers (32), updates ttl.base_cta_index on every
    function, and builds the ttl.compiler_allocated_dfbs module
    attribute for the Python runtime.
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

def TTKernelCombinePackTiles
    : Pass<"ttkernel-combine-pack-tiles", "::mlir::func::FuncOp"> {
  let summary = "Combine consecutive pack_tile ops into pack_tile_block";
  let description = [{
    Scans for consecutive `ttkernel.pack_tile` ops targeting the same dataflow
    buffer with contiguous DST and CB tile indices, and replaces each run with
    a single `ttkernel.pack_tile_block`. This reduces the number of hardware
    pack operations after loop unrolling.

    Only combines runs where both DST indices and CB tile indices step by 1,
    and the first CB tile index is 0 (pack_tile_block always writes to the CB
    starting from index 0, per the op definition). Runs are broken by different
    CBs, non-contiguous indices, non-constant indices, or intervening
    non-constant ops. Runs of length 1 are skipped.

    Example:
    ```mlir
    // Before:
    ttkernel.pack_tile(%c0, %cb, %c0, true)
    ttkernel.pack_tile(%c1, %cb, %c1, true)
    ttkernel.pack_tile(%c2, %cb, %c2, true)
    ttkernel.pack_tile(%c3, %cb, %c3, true)

    // After:
    ttkernel.pack_tile_block(%c0, %cb, %c4)
    ```
  }];

  let dependentDialects = [
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

#endif // TTLANG_DIALECT_TTL_PASSES_TD
