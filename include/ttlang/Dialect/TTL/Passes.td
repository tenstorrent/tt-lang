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

def TTLTileAndAssignDST
    : Pass<"ttl-tile-and-assign-dst", "::mlir::func::FuncOp"> {
  let summary = "Analyze DST usage, insert copy_tile, assign dst_idx, check capacity";
  let description = [{
    Analyzes DST register pressure in ttl.compute bodies, inserts ttl.copy_tile
    for block arguments, assigns dst_idx attributes to tile math operations, and
    emits clear diagnostics on overflow. No loop tiling is performed yet.
  }];

  let options = [
    Option<"dstCapacity", "dst-capacity", "uint32_t", "0",
           "Override DST register capacity (testing only, default auto-computed).">
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

#endif // TTLANG_DIALECT_TTL_PASSES_TD
