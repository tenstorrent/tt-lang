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
    direct uses of the DFB value and the acquire result, plus uses of values
    those ops produce (e.g. copy -> transfer_handle -> wait). Direct DFB uses
    are bounded by the next acquire in the same DFB sync class (`cb_reserve`
    for producer intervals, `cb_wait` for consumer intervals) because they do
    not carry an SSA link to the acquire that owns the slot.

    Explicit push/pop calls and context-manager-generated push/pop are
    preserved: the pass only fills in what is missing.
  }];

  let dependentDialects = [];
}

def TTLCoalesceDFBAcquires
    : Pass<"ttl-coalesce-dfb-acquires", "::mlir::func::FuncOp"> {
  let summary = "Coalesce consecutive same-DFB acquires into one multi-tile acquire";
  let description = [{
    Rewrites the source pattern

    ```
    t1 = cb.wait();  t2 = cb.wait();  t3 = cb.wait();
    use(t1);         use(t2);         use(t3);
    ```

    into the canonical tt-metal "cumulative wait + indexed reads +
    coalesced pop" pattern

    ```
    cb_wait_front(cb, 3*k);
    copy_tile(cb, /*src_idx=*/0, dst);  // t1
    copy_tile(cb, /*src_idx=*/k, dst);  // t2
    copy_tile(cb, /*src_idx=*/2*k, dst);// t3
    cb_pop_front(cb, 3*k);
    ```

    used in tt-metal compute kernels (`eltwise_binary.cpp`, `bcast_h.cpp`,
    matmul kernels, etc.) when a consumer holds a fixed multi-tile window
    before processing it.

    Pre-coalesce, each `ttl.cb_wait` lowers to its own `cb_wait_front(k)`
    and each `ttl.cb_pop` to `cb_pop_front(k)`. Because metal's
    `cb_wait_front`/`cb_pop_front` are non-cumulative, this races: the
    first pop advances the front before the producer has pushed enough
    tiles to satisfy the next read.

    At the IR level the rewrite is

    ```mlir
    // Before (post `ttl-insert-cb-sync`):
    %t1 = ttl.cb_wait %cb : <[1,k],...,bc> -> tensor<1xkx...>
    %t2 = ttl.cb_wait %cb : <[1,k],...,bc> -> tensor<1xkx...>
    %t3 = ttl.cb_wait %cb : <[1,k],...,bc> -> tensor<1xkx...>
    ... use %t1 ... ttl.cb_pop %cb
    ... use %t2 ... ttl.cb_pop %cb
    ... use %t3 ... ttl.cb_pop %cb

    // After:
    %g  = ttl.cb_wait %cb {num_tiles = 3*k} : ... -> tensor<1x(3k)x...>
    %t1 = tensor.extract_slice %g [0, 0]    [1, k] [1, 1]
    %t2 = tensor.extract_slice %g [0, k]    [1, k] [1, 1]
    %t3 = tensor.extract_slice %g [0, 2*k]  [1, k] [1, 1]
    ... use %t1 ... use %t2 ... use %t3 ...
    ttl.cb_pop %cb {num_tiles = 3*k}
    ```

    `addSliceOffset` already folds each `extract_slice` offset into the
    per-tile `src_idx` at lowering time, so no lowering changes are
    required.

    The producer side (`cb_reserve` / `cb_push`) is rewritten by the same
    templated routine: a run of `N` consecutive `cb_reserve` ops on one
    DFB collapses into one `cb_reserve {num_tiles = N*k}` plus per-block
    `tensor.extract_slice` views routed into the downstream
    `ttl.tile_store` / `ttl.store` ops, and the `N` matching `cb_push`
    ops collapse into one `cb_push {num_tiles = N*k}`.

    Detection: forward walk per block; an acquire group is a maximal run
    of same-kind same-DFB acquires separated only by `ttl.attach_cb` or
    `arith.constant` ops. Acquires already carrying a `num_tiles`
    attribute (e.g. set by `ttl-subblock-compute-for-dst`) are not
    coalesced and terminate the group. The pass is idempotent.

    Result-type constraint: only acquires whose result tensor has shape
    `<1, k>` are coalesced; the coalesced result is `<1, N*k>`. Other
    ranks would require a convention for which axis to extend by `N`,
    which is not specified, and are skipped.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect"
  ];
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

def TTLMaterializeLoopState
    : Pass<"ttl-materialize-loop-state", "::mlir::func::FuncOp"> {
  let summary = "Materialize loop-carried tensor state";
  let description = [{
    Rewrites ranked tensor `scf.for` iter args into explicit TTL dataflow
    buffer state. The pass removes tensor results from the loop while
    preserving non-tensor iter args.

    Tensor recurrences use compiler-allocated dataflow buffer state:
    ```mlir
    %state = scf.for %iv = %lb to %ub step %step
        iter_args(%arg = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %next = ttl.relu %arg : ...
      scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    ```

    becomes:
    ```mlir
    %state_cb = ttl.bind_cb ... {ttl.compiler_allocated}
    %init_reserve = ttl.cb_reserve %state_cb : ...
    ttl.store %init, %init_reserve : ...
    scf.for %iv = %lb to %ub step %step {
      %current_wait = ttl.cb_wait %state_cb : ...
      %current = ttl.attach_cb %current_wait, %state_cb : ...
      %next = ttl.relu %current : ...
      %next_reserve = ttl.cb_reserve %state_cb : ...
      ttl.store %next, %next_reserve : ...
      scf.yield
    }
    %final_wait = ttl.cb_wait %state_cb : ...
    %final = ttl.attach_cb %final_wait, %state_cb : ...
    ```
  }];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::tt::ttl::TTLDialect"
  ];
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
           "Allocate outputs in separate DST region (needed for reductions and some loop optimizations).">
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
    Sets fp32_dest_acc_en, dst_full_sync_en, and ttl.enable_fpu_binary_ops
    attributes on the enclosing func.func when they are not already
    configured. These are per-kernel compile-time settings; any op in the
    function can read them via getKernelBoolAttr().
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
           "Sets fp32_dest_acc_en on the kernel when any compute op contains matmul_block.">,
    Option<"enableFPUBinaryOps", "enable-fpu-binary-ops", "bool", "true",
           "Allow lowering eligible tile add/sub/mul to FPU binary form. "
           "Written as the ttl.enable_fpu_binary_ops attribute on the func and "
           "consumed by isFPUEligibleBinaryOp(). Disable to force the SFPU path.">
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
    dst_index). Dependency depth considers SSA RAW, DST RAW, WAW (same DST
    write index), and WAR (write after prior read of same DST index) hazards.
    Ops at the same depth are independent and can be reordered.

    Only tile ops between `tile_regs_acquire` and `tile_regs_commit` are
    scheduled. Store ops (after the commit/wait boundary) are not reordered.

    Category ordering:
      Bcast < Transpose < CopyTile < FPUBinary < SFPUUnary < SFPUBinary <
      CopyDst < DstIndex

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

def TTLVerifyPipeNetGuards
    : Pass<"ttl-verify-pipenet-guards", "::mlir::ModuleOp"> {
  let summary = "Verify PipeNet-coupled DFB and pipe operations are role-guarded";
  let description = [{
    Checks that every operation coupled to a PipeNet executes only on launch
    coordinates where its role is defined. A launch coordinate is a
    `(core_x, core_y)` point in the 2D grid declared by the module's
    `ttl.launch_grid` attribute; each kernel function runs once per
    coordinate. A domain is a subset of those coordinates.

    A forward dataflow analysis tracks, at each program point, the domain on
    which it may execute under the enclosing region predicates (`scf.if`,
    `affine.if`, `ttl.if_src`, `ttl.if_dst`, `ttl.pipenet_scope`). At each
    `ttl.copy`, `ttl.cb_wait`, and `ttl.cb_push` the analysis verifies the
    point's domain is contained in the pipe's source or destination domain;
    out-of-domain uses, scopes broader than their declared role, unmatched
    `ttl.cb_wait`s, and unanalyzable coord-dependent predicates are reported.
    Each diagnostic carries an example offending coordinate, the contributing
    PipeNet(s), and a suggested guard.

    Same-thread pipe schedule diagnostics are emitted only when role and DFB
    domain checks have succeeded, so schedule-cycle reports are not mixed with
    earlier guard failures.

    The pass expects one module per `@ttl.operation`, a well-formed
    `ttl.launch_grid` (i64 array of length 2 with positive entries), static
    `I64Attr` coordinates on `ttl.create_pipe`, and finalized DFB indices
    (i.e. `ttl-annotate-cb-associations` and `ttl-finalize-dfb-indices`
    have run).
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLVerifyDFBSPSC
    : Pass<"ttl-verify-dfb-spsc", "::mlir::ModuleOp"> {
  let summary = "Reject DFBs with overlapping producer or consumer launch domains";
  let description = [{
    Walks every `ttl.cb_reserve` and `ttl.cb_wait`, groups them by `cb_index`
    (resolved via `getCBIndex`) and enclosing `ttl.kernel_thread`-tagged
    `func.func`, and tracks the launch-node domain for each producer or
    consumer participant. A DFB is valid when, for each launched node, at most
    one producer thread reserves it and at most one consumer thread waits on it.
    Disjoint domains across different kernel-thread functions are accepted.

    tt-metal circular buffers expose a counter pair (`pages_received`,
    `pages_acked`) that is not multi-writer safe on either side, so the
    per-launched-node single-producer single-consumer property is required for
    correctness; see `docs/development/DFBManagement.md` for the runtime
    contract. The pass emits an error for overlapping producer or consumer
    domains, and for unknown domains when multiple threads participate and SPSC
    cannot be proven statically.

    Prerequisite: `ttl-finalize-dfb-indices` must run first so that every
    `bind_cb` carries a concrete `cb_index`, and modules with DFB acquire ops
    must carry a valid `ttl.launch_grid` attribute. The pass asserts on
    unresolvable indices.
  }];

  let dependentDialects = [];
}

def TTLErasePipeNetScopes
    : Pass<"ttl-erase-pipenet-scopes", "::mlir::ModuleOp"> {
  let summary = "Inline and erase `ttl.pipenet_scope` regions";
  let description = [{
    `ttl.pipenet_scope` is a structural marker emitted by the frontend and
    consumed by `ttl-verify-pipenet-guards`. After verification the markers
    serve no further purpose; this pass inlines each scope's body into its
    parent block and erases the scope op so downstream lowering sees a
    scope-free IR. Runs after `ttl-verify-pipenet-guards`.
  }];

  let dependentDialects = ["::mlir::tt::ttl::TTLDialect"];
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

def TTLValidateCBBudget
    : Pass<"ttl-validate-cb-budget", "::mlir::ModuleOp"> {
  let summary = "Validate total CB allocation fits within per-core L1 budget";
  let description = [{
    Module-level pass that runs after `ttl-finalize-dfb-indices`, before
    `convert-ttl-to-ttkernel`. Sums the static backing-store size of every
    circular buffer declared in the module and compares against a per-core
    L1 budget.

    For each `BindCBOp`, the pass computes:
      cb_bytes = total_elements * tile_size_bytes

    When the same `cb_index` appears in multiple functions (e.g. compute
    and data-movement kernels sharing a CB slot), only the largest
    allocation for that index is counted.

    On overflow the pass emits an error with a per-CB breakdown and a
    hint to reduce block shapes or block counts.

    L1 budget is taken from the module's TTCore system descriptor and
    default device when present (`ChipDescAttr::getUsableL1Size()` for the
    selected chip). When the module has no system descriptor, the pass uses
    the same static per-core CB region limit as Wormhole and Blackhole
    (total L1 minus reserved kernel space).

    This is a static "CB region vs budget" check. It does not model
    other L1 consumers (kernel binaries, stack, runtime buffers).
  }];

  let options = [
    Option<"l1BudgetOverride", "l1-budget-override", "uint32_t", "0",
           "Override L1 CB budget (bytes). 0 means auto-detect from "
           "device attributes or use architecture default.">
  ];
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

def TTLLowerScalarCmpF
    : Pass<"ttl-lower-scalar-cmpf", "::mlir::ModuleOp"> {
  let summary = "Lower arith.cmpf on scalar floats to TTKernel soft-float ops";
  let description = [{
    Converts `arith.cmpf` operations on scalar floats (from `raw_element_read`
    lowering) to TTKernel soft-float comparison ops. After
    `convert-ttl-to-ttkernel`, float values from L1 loads are represented as
    integer bit patterns with `unrealized_conversion_cast` bridging to float
    types. This pass resolves those casts and emits the appropriate TTKernel
    comparison op.

    Supported predicates:
    - `ogt` -> `ttkernel.float32_greater` / `ttkernel.bfloat16_greater`
    - `olt` -> same ops with swapped operands

    All other predicates (`oeq`, `one`, `oge`, `ole`, and all unordered
    predicates `uno`, `ueq`, `ugt`, `uge`, `ult`, `ule`, `une`) emit an
    error diagnostic and signal pass failure.

    Operand requirements: each `arith.cmpf` operand must resolve to an
    integer bit pattern via one of two sources:
    1. `unrealized_conversion_cast(iN -> fN)` from a prior
       `RawElementReadLowering` (unwraps to the integer directly).
    2. `arith.constant <float>` (materializes the IEEE-754 bit pattern
       as an integer constant).

    Operands from other sources (e.g., `arith.addf` results) cannot be
    resolved and cause a diagnostic: "could not resolve float operand to
    integer bit pattern".
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
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
