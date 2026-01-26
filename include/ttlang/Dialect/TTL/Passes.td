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
    DST register allocator using linear scan allocation with unary operation
    merging. Implements the algorithm from DST_Allocation.md:

    Phase 1: Copy insertion for multi-consumer values
    - When a value has multiple consumers and any is unary (overwrites in-place),
      inserts ttl.copy_dst operations for all but the last consumer

    Phase 2: Build live intervals with unary merging
    - Builds interval [start, end] for each tile value
    - Merges intervals for unary ops (input and output share same DST)

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
           "Override DST register capacity (testing only, default auto-computed).">,
    Option<"separateOutputRegion", "separate-output-region", "bool", "false",
           "Allocate outputs in separate DST region (needed for reductions and some loop optimizations).">
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLInsertTileRegsSync
    : Pass<"ttl-insert-tile-regs-sync", "::mlir::func::FuncOp"> {
  let summary = "Insert tile_regs_* synchronization ops around ttl.compute";
  let description = [{
    Inserts DST register synchronization ops mirroring the TTKernel reg API:
    - Inside ttl.compute: tile_regs_acquire at entry, tile_regs_commit before yield.
    - Around ttl.compute: tile_regs_wait and tile_regs_release immediately after
      the compute op in the parent block.
  }];

  let dependentDialects = [
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
        ttl.yield %sum
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

#endif // TTLANG_DIALECT_TTL_PASSES_TD
