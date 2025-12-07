# TTL Dialect Compute Operations

**Part of**: [TTL Dialect Design Plan](01_TTL_Dialect_Plan.md)

This document specifies compute operations, circular buffers, fusion, and DST register capacity management for the TTL dialect.

## Table of Contents

- [4.1 Structural Operations](#41-structural-operations)
- [4.3 Circular Buffer Operations](#43-circular-buffer-operations)
- [4.4 Compute Operations](#44-compute-operations)
  - [4.4.1 Fusion Operations](#441-fusion-operations)
  - [4.4.2 Arithmetic Operations](#442-arithmetic-operations)
  - [4.4.3 Reduction and Broadcast Operations](#443-reduction-and-broadcast-operations)
  - [4.4.4 Materialization Operations](#444-materialization-operations)
  - [4.4.5 DST Register Management](#445-dst-register-management)
- [4.7 Utility Operations](#47-utility-operations)
- [4.8 Compute Operations Inventory](#48-compute-operations-inventory)
- [5.6 Structured Ops for DST Register Capacity-Based Tiling](#56-structured-ops-for-dst-register-capacity-based-tiling)

## Related Documents

- [Back to Overview](01_TTL_Dialect_Plan.md)
- [Type System](02_TTL_Type_System.md) - Types used by these operations
- [Data Movement Operations](04_TTL_Data_Movement_Operations.md) - Data movement and synchronization
- [Compilation Pipeline](05_TTL_Compilation_Pipeline.md) - How these operations are lowered

---

# 4. Compute Operations (Selected Sections)

### 4.1 Structural Operations

```tablegen
def TTL_ProgramOp : TTL_Op<"program", [IsolatedFromAbove]> {
  let summary = "Top-level kernel program with captured tensors";
  let arguments = (ins
    TTL_GridAttr:$grid,
    StrAttr:$memory_space,
    BoolAttr:$tiled
  );
  let regions = (region VariadicRegion<AnyRegion>:$body);
  let description = [{
    Python API `ttl.Program(compute_thread, dm_thread0, dm_thread1)(x, y)` maps
    to this operation. The `ttl.Program()` constructor is a Python API that
    constructs the `ttl.program` MLIR operation with up to three thread regions.
    The `ttl.kernel` operation is an internal representation used during compilation;
    the top-level operation exposed to the Python API is `ttl.program`.

    Container for kernel execution. Owns captured tensors and nested thread regions.
    Regions execute in parallel on grid of cores.
  }];
}

def TTL_KernelOp : TTL_Op<"kernel", [IsolatedFromAbove]> {
  let summary = "Kernel with multiple threads on grid";
  let arguments = (ins
    Variadic<AnyType>:$inputs,   // TODO: define and use a more specific type constraint
    TTL_GridAttr:$grid,
    StrAttr:$memory_space,
    BoolAttr:$tiled,
    OptionalAttr<ArrayAttr>:$block_factors,  // Block factors per tensor [[M,N], [K,N], ...]
    OptionalAttr<ArrayAttr>:$compile_time_args,  // Indices of inputs that are compile-time constants
    OptionalAttr<DictionaryAttr>:$compute_config  // Math fidelity, precision modes
  );
  let regions = (region VariadicRegion<AnyRegion>:$threads);
  let results = (outs Variadic<AnyType>:$results);
  let description = [{
    Kernel operation representing a multi-threaded program on a grid of cores.

    Optional attributes:
    - block_factors: Array of [rows, cols] tile counts per core for each tensor argument.
      Example: [[1, 1], [1, 1], [2, 2]] means first two tensors have 1 tile per core,
      third tensor has 2x2=4 tiles per core. Enables kernel specialization for different
      parallelization strategies and autotuning.

    - compile_time_args: Indices of input arguments that should be treated as compile-time
      constants (e.g., tensor layouts, loop bounds). These are baked into the kernel binary
      during code generation rather than passed as runtime parameters.

    - compute_config: Dictionary with hardware configuration:
      - "math_fidelity": "HiFi4" | "HiFi2" | "LoFi"
      - "fp32_dest_acc_en": true | false (FP32 accumulation in DST registers)
      - "math_approx_mode": true | false (approximation mode for math ops)
  }];
}

def TTL_ComputeThreadOp : TTL_Op<"compute_thread"> {
  let summary = "Compute thread executing on Tensix core";
  let arguments = (ins
    Variadic<AnyType>:$operands,  // TODO: define and use a more specific type constraint
    OptionalAttr<StrAttr>:$microcore_hint  // Future evolution path
  );
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Thread for mathematical operations on tiles. Executes on MATH microcore(s).
    Can use: block arithmetic, CB wait/pop/reserve/push, tile operations.
    Cannot use: DMA operations (use datamovement_thread).
  }];
}

def TTL_DataMovementThreadOp : TTL_Op<"datamovement_thread"> {
  let summary = "Data movement thread for DMA and synchronization";
  let arguments = (ins
    Variadic<AnyType>:$operands,
    OptionalAttr<StrAttr>:$microcore_hint  // Future: NOC0, NOC1, etc.
  );
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Thread for data transfer operations. Executes on NOC microcores.
    Can use: DMA (ttl.copy), CB reserve/push, semaphores, pipes.
    Cannot use: arithmetic operations (use compute_thread).
  }];
}
```

### 4.2 Resource Creation

```tablegen
def TTL_CreateCBOp : TTL_Op<"create_cb"> {
  let summary = "Create circular buffer";
  let arguments = (ins
    I64ArrayAttr:$shape,                // Elements per block
    TypeAttr:$element_type,             // !ttcore.tile<32x32, f32> or scalar type (f32, bf16, etc.)
    I64Attr:$buffer_factor,             // Number of blocks
    TTL_MemorySpaceAttr:$memory_space,  // L1, DRAM, DST
    OptionalAttr<I32Attr>:$buffer_index,      // Optional explicit CB number (0-31)
    OptionalAttr<I64Attr>:$page_size,         // Optional page size in bytes
    OptionalAttr<TTL_CoreMaskAttr>:$core_ranges  // Optional per-core CB mapping
  );
  let results = (outs TTL_CircularBuffer:$result);
  let description = [{
    Python API `ttl.make_circular_buffer_like(tensor, shape=..., buffer_factor=...)`
    maps to this operation. The Python frontend extracts tensor properties
    (dtype, layout, memory_space) and calls `ttl.create_cb` with the extracted
    parameters. The "likeness" refers to deriving element_type and memory_space
    from the input tensor's layout and properties.

    Shape units and L1 allocation per TT-Lang spec:
    - Shape parameter is in shape units (tiles for tiled, scalars for row-major)
    - Total L1 storage = block_size × buffer_factor
    - block_size = shape[0] × shape[1] × ... × sizeof(element_type)

    Layout examples:
    - Tiled: shape=[2,1], element_type=!ttcore.tile<32x32,f32>, buffer_factor=2
      → 2 tiles per block × 2 blocks = 4 tiles total L1 allocation

    - Row-major: shape=[64,32], element_type=f32, buffer_factor=2
      → 64×32 scalars per block × 2 blocks total L1 allocation

    The resulting CB type's isTiled() method derives layout from element_type automatically.
    Buffer factor enables producer-consumer pipelining (typically 2 for double-buffering).

    Optional attributes enable explicit control when needed:
    - buffer_index: Explicitly assign CB number (default: auto-assign in allocation pass)
    - page_size: Override computed page size for custom configurations
    - core_ranges: Specify which cores can access this CB (default: all cores in grid)

    These optional attributes support TTNN KernelDescriptor compatibility and
    enable fine-grained control for hand-optimized kernels.
  }];
}

def TTL_CreatePipeOp : TTL_Op<"create_pipe"> {
  let summary = "Create inter-core pipe for unicast or multicast";
  let arguments = (ins
    I64ArrayAttr:$src_core,                       // Source core [x, y, ...]
    OptionalAttr<I64ArrayAttr>:$dst_core,         // Unicast destination [x, y, ...]
    OptionalAttr<ArrayAttr>:$dst_core_range       // Multicast range [SliceAttr, SliceAttr, ...]
  );
  let results = (outs TTL_Pipe:$pipe);
  let description = [{
    Python API `ttl.Pipe(src_core=..., dst_core=...)` or
    `ttl.Pipe(src_core=..., dst_core_range=...)` maps to this operation.

    Unicast example:
      Python: ttl.Pipe(src_core=(1,0), dst_core=(0,0))
      TTL IR: ttl.create_pipe src_core=[1,0], dst_core=[0,0]

    Multicast example (preserves slice semantics):
      Python: ttl.Pipe(src_core=(x,0), dst_core_range=(slice(x,x+1), slice(1,grid_y)))
      TTL IR: ttl.create_pipe src_core=[x,0],
                              dst_core_range=[#ttl.slice<x,x+1,1>, #ttl.slice<1,grid_y,1>]

    The dst_core_range encodes Python slice objects per dimension with (start, stop, step).
    Half-open intervals: [start, stop). Step enables patterns like "every other core".
  }];
}

def TTL_CreateSemaphoreOp : TTL_Op<"create_semaphore"> {
  let summary = "Create semaphore with initial value";
  let arguments = (ins
    I32Attr:$initial_value,
    OptionalAttr<BoolAttr>:$is_remote
  );
  let results = (outs TTL_Semaphore:$semaphore);
}

def TTL_GetRemoteSemaphoreOp : TTL_Op<"get_remote_semaphore"> {
  let summary = "Create remote unicast semaphore reference";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I64ArrayAttr:$core                     // Target core coordinates
  );
  let results = (outs TTL_Semaphore:$remote_semaphore);
  let description = [{
    Python API `semaphore.get_remote(core)` maps to this operation.
    Creates an annotated semaphore reference for remote unicast operations.
    The returned semaphore reference is used with `ttl.semaphore_set` and
    `ttl.semaphore_inc` operations. This is a compile-time operation that
    annotates the semaphore with target core information.
  }];
}

def TTL_GetRemoteMulticastSemaphoreOp : TTL_Op<"get_remote_multicast_semaphore"> {
  let summary = "Create remote multicast semaphore reference";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    OptionalAttr<ArrayAttr>:$core_range    // Multicast core range, empty = entire grid
  );
  let results = (outs TTL_Semaphore:$remote_semaphore);
  let description = [{
    Python API `semaphore.get_remote_multicast(core_range)` maps to this operation.
    When called with no arguments, returns multicast semaphore for entire grid.
    Creates an annotated semaphore reference for remote multicast operations.
    The returned semaphore reference is used with `ttl.semaphore_set` operation
    (multicast semaphores support set but not inc). This is a compile-time
    operation that annotates the semaphore with multicast range information.
  }];
}

def TTL_TensorAccessorOp : TTL_Op<"tensor_accessor"> {
  let summary = "Create accessor for indexed tensor access";
  let arguments = (ins
    TTL_Tensor:$tensor,
  );
  let results = (outs TTL_TensorAccessor:$accessor);
  let description = [{
    Materializes a tensor accessor for the given tensor. The tensor's type
    already encodes layout metadata (e.g., sharded vs interleaved) via its
    encoding attribute, so no extra layout argument is required.
  }];
}
```

### 4.3 Circular Buffer Operations

```tablegen
def TTL_CBWaitOp : TTL_Op<"cb_wait"> {
  let summary = "Consumer acquire: wait for data in circular buffer";
  let arguments = (ins TTL_CircularBuffer:$cb);
  let results = (outs TTL_Block:$block);
  let description = [{
    Blocking operation that waits until a block of data is available in the circular buffer.
    Returns a block whose shape matches the CB's block shape. Per TT-Lang spec, users never
    specify element counts - the block shape is determined by the CB itself.

    Python API: `blk = cb.wait()` or `with cb.wait() as blk:`

    Maps to TTKernel operation:
    - ttkernel.cb_wait_front(cb, num_elements)
      where num_elements = cb.shape[0] * cb.shape[1] * ... (computed from CB type)
  }];
}

def TTL_CBPopOp : TTL_Op<"cb_pop"> {
  let summary = "Consumer release: signal block consumed";
  let arguments = (ins TTL_Block:$block);
  let description = [{
    Non-blocking operation that signals the block has been consumed and can be
    reused by the producer. Takes the block returned by cb.wait().

    Python API: `cb.pop()` (implicit: pops the block acquired by most recent wait)

    Maps to TTKernel operation:
    - ttkernel.cb_pop_front(cb, num_elements)
      where num_elements is derived from the block's shape
  }];
}

def TTL_CBReserveOp : TTL_Op<"cb_reserve"> {
  let summary = "Producer acquire: reserve space in circular buffer";
  let arguments = (ins TTL_CircularBuffer:$cb);
  let results = (outs TTL_Block:$block);
  let description = [{
    Blocking operation that waits until space is available in the circular buffer.
    Returns a block whose shape matches the CB's block shape. Per TT-Lang spec, users never
    specify element counts - the block shape is determined by the CB itself.

    Python API: `blk = cb.reserve()` or `with cb.reserve() as blk:`

    When used with Python `with` statement, the AST compiler generates `ttl.cb_reserve`
    at the start of the scope and `ttl.cb_push` at the end.

    Maps to TTKernel operation:
    - ttkernel.cb_reserve_back(cb, num_elements)
      where num_elements = cb.shape[0] * cb.shape[1] * ... (computed from CB type)
  }];
}

def TTL_CBPushOp : TTL_Op<"cb_push"> {
  let summary = "Producer release: signal block ready";
  let arguments = (ins TTL_Block:$block);
  let description = [{
    Non-blocking operation that signals the block is ready for consumers.
    Takes the block returned by cb.reserve(). Automatically generated at the
    end of a `with cb.reserve()` scope.

    Python API: `cb.push()` (implicit: pushes the block acquired by most recent reserve)

    Maps to TTKernel operation:
    - ttkernel.cb_push_back(cb, num_elements)
      where num_elements is derived from the block's shape
  }];
}

def TTL_GetTileOp : TTL_Op<"get_tile"> {
  let summary = "Extract tile from circular buffer (internal IR use)";
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    Index:$tile_idx                   // Tile index within CB
  );
  let results = (outs AnyType:$tile);  // !ttcore.tile<32x32, f32>
  let description = [{
    Internal IR operation generated when lowering block operations to tile loops.
    Not exposed in TT-Lang Python DSL.

    Copies a tile from circular buffer to DST register for computation.

    Maps to TTKernel operations:
    - ttkernel.copy_tile_init(cb)
    - ttkernel.copy_tile(cb, tile_idx, dst_idx)
  }];
}

def TTL_PackTileOp : TTL_Op<"pack_tile"> {
  let summary = "Pack computed tile into circular buffer (internal IR use)";
  let arguments = (ins
    Index:$dst_idx,                   // DST register index
    TTL_CircularBuffer:$cb,
    Index:$tile_idx                   // Tile index within CB
  );
  let description = [{
    Internal IR operation generated when lowering block store to tile loops.
    Not exposed in TT-Lang Python DSL.

    Copies a tile from DST register back to circular buffer.

    Maps to TTKernel operation:
    - ttkernel.pack_tile(dst_idx, cb, tile_idx)
  }];
}
```

### 4.4 Compute Operations

TTL compute operations work at the **block abstraction level** (blocks of
tiles), providing a higher-level interface than individual tile operations.
During lowering to TTKernel, each block operation expands into loops over tiles
with explicit DST register management.

This section is organized into:
- [4.4.1 Fusion Operations](#441-fusion-operations): Operations for fusing
  computation chains
- [4.4.2 Arithmetic Operations](#442-arithmetic-operations): Element-wise math
  (add, mul, matmul)
- [4.4.3 Reduction and Broadcast](#443-reduction-and-broadcast-operations):
  Operations for tensor dimension manipulation
- [4.4.4 Materialization](#444-materialization-operations): Operations for
  storing results
- [4.4.5 DST Register Management](#445-dst-register-management): IR-level
  constructs for register allocation

#### 4.4.1 Fusion Operations

Without fusion, each block operation requires a full circular buffer round-trip:
1. Reserve output CB slot
2. Store result to CB (DST → L1 write)
3. Push CB slot to signal data ready
4. Wait for CB slot (in next operation)
5. Load from CB (L1 → DST read)
6. Pop CB slot after consumption

Flash attention kernels, for example, have 7+ intermediate results (Q@K, scale,
exp, reduce_sum, bcast, recip, multiply), so the above pattern requires:
- 7 intermediate CB allocations (consuming precious L1 capacity)
- 14 L1 memory transfers (7 writes + 7 reads)
- Significant memory bandwidth overhead

The `ttl.compute_region` operation enables fusion: all operations execute
directly on `DST` registers with values flowing from one operation to the next
without intermediate `CB` allocations or `L1` traffic. Only the final result is
written to memory. This is critical for complex kernels that would otherwise
exceed L1 capacity or waste bandwidth.

**Dense Layer Fusion Example:**

A dense layer (fully connected layer) in neural networks computes
`output = activation(X @ W + bias)` followed by optional normalization. This
pattern fuses matmul, element-wise operations, and reductions:

```mlir
# Dense layer with ReLU activation and layer normalization
# Computation: matmul → add bias → ReLU → compute mean → center values

result = ttl.compute_region ins(x_blk, w_blk, bias_blk, ones_blk) outs(o_blk) {
  ^bb0(%x, %w, %b, %ones):
    %matmul_result = ttl.block_matmul %x, %w         // X @ W
    %with_bias = ttl.block_add %matmul_result, %b    // + bias
    %activated = ttl.block_relu %with_bias           // ReLU activation
    %mean = ttl.block_reduce_sum %activated, %ones   // Sum for mean
    %mean_bcast = ttl.block_bcast %mean              // Broadcast mean
    %centered = ttl.block_sub %activated, %mean_bcast // Center values
    ttl.yield %centered
} {keep_in_dst = true}
```

Without fusion: 5 intermediate CB allocations + 10 L1 transfers. With fusion: 0
intermediate allocations, all operations execute on DST registers.

The lowering pass generates a single affine loop with sequential TTKernel
operations, reusing DST registers as values become dead.

```tablegen
def TTL_ComputeRegionOp : TTL_Op<"compute_region", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Fused compute region with DST register reuse";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<BoolAttr>:$keep_in_dst  // Hint: keep intermediates in DST, don't spill to L1
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Wraps a chain of block compute operations for fusion into a single execution unit.
    All intermediate results remain in DST registers.

    Example: Simple attention kernel fusion
      %result = ttl.compute_region ins(%q, %k, %v, %scale : ...) outs(%out : ...) {
        ^bb0(%q_blk, %k_blk, %v_blk, %scale_blk):
          %s = ttl.block_matmul %q_blk, %k_blk
          %s_scaled = ttl.block_mul %s, %scale_blk
          %exp_s = ttl.block_exp %s_scaled
          %sum = ttl.block_reduce_sum %exp_s, %ones
          %sum_bcast = ttl.block_bcast %sum
          %recip = ttl.block_recip %sum_bcast
          %p = ttl.block_mul %exp_s, %recip
          %o = ttl.block_matmul %p, %v_blk
          ttl.yield %o
      } {keep_in_dst = true}

    The keep_in_dst attribute hints that intermediate values should not be materialized
    to circular buffers unless necessary for cross-thread communication.

    Lowering generates a single affine loop with all operations executing sequentially.
    `TTLAssignDSTRegisters` performs liveness analysis to allocate `DST` registers and
    minimize register pressure.
  }];
}

def TTL_BlockComputeOp : TTL_Op<"block_compute"> {
  let summary = "Compute region operating on blocks";
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Region containing only ttl.math.* operations and structural ops.
    Dedicated lowering pass rewrites to arith/math/tosa or TT intrinsics.

    For fused operation chains, ttl.compute_region provides explicit fusion control.
  }];
}
```

#### 4.4.2 Arithmetic Operations

Element-wise operations on blocks. Each operation lowers to an affine loop over
tiles with corresponding TTKernel tile operations.

```tablegen
def TTL_BlockAddOp : TTL_Op<"block_add"> {
  let summary = "Element-wise addition on tensor blocks";
  let arguments = (ins TTL_Tensor:$lhs, TTL_Tensor:$rhs);
  let results = (outs TTL_Tensor:$result);
  let description = [{
    Element-wise addition of two blocks.

    Maps to TTKernel operations:
    - ttkernel.add_tiles_init(in0_cb, in1_cb)
    - affine.for loop over block tiles:
      - ttkernel.add_tiles(in0_cb, in1_cb, in0_idx, in1_idx, dst_idx)
  }];
}

def TTL_BlockMulOp : TTL_Op<"block_mul"> {
  let summary = "Element-wise multiplication on tensor blocks";
  let arguments = (ins TTL_Tensor:$lhs, TTL_Tensor:$rhs);
  let results = (outs TTL_Tensor:$result);
  let description = [{
    Element-wise multiplication of two blocks.

    Maps to TTKernel operations:
    - ttkernel.mul_tiles_init(in0_cb, in1_cb)
    - affine.for loop over block tiles:
      - ttkernel.mul_tiles(in0_cb, in1_cb, in0_idx, in1_idx, dst_idx)
  }];
}

def TTL_BlockMatmulOp : TTL_Op<"block_matmul"> {
  let summary = "Matrix multiplication on tensor blocks";
  let arguments = (ins TTL_Tensor:$lhs, TTL_Tensor:$rhs);
  let results = (outs TTL_Tensor:$result);
  let description = [{
    Matrix multiplication of two blocks.

    Maps to TTKernel operations:
    - ttkernel.mm_init(in0_cb, in1_cb, out_cb, transpose) (once per kernel)
    - affine.for loop over block tiles (outer product iteration for block matmul):
      - ttkernel.mm_init_short(in0_cb, in1_cb, transpose) (if other init called between)
      - ttkernel.matmul_tiles(in0_cb, in1_cb, in0_idx, in1_idx, dst_idx, transpose)

    Note: Matmul operations can be fused with element-wise and reduction operations
    within ttl.compute_region. See section 4.4.1 for dense layer fusion example.
  }];
}

// Additional arithmetic ops: sub, div, exp, recip, relu, sigmoid, etc.
```

#### 4.4.3 Reduction and Broadcast Operations

Operations for reducing tensor dimensions and broadcasting values. Critical for
attention mechanisms and normalization patterns.

```tablegen
def TTL_BlockReduceSumOp : TTL_Op<"block_reduce_sum"> {
  let summary = "Reduce sum along specified axis";
  let arguments = (ins
    TTL_Tensor:$input,
    TTL_Tensor:$scaling,  // Scaling tensor (typically ones) for reduction
    I64Attr:$axis        // Axis to reduce along
  );
  let results = (outs TTL_Tensor:$result);
  let description = [{
    Reduces input tensor along specified axis using sum operation.
    Used in normalization patterns such as attention mechanisms.

    Example: reduce_sum(exp_S, ones, dim=1) sums along dimension 1.

    Maps to TTKernel operations:
    - ttkernel.reduce_init(in_cb, scaling_cb, out_cb, ReduceFunc::Sum, reduce_dim)
    - affine.for loop over reduction dimension tiles:
      - ttkernel.reduce_tile(in_cb, scaling_cb, in_idx, scaling_idx, dst_idx, Sum, reduce_dim)
    - ttkernel.reduce_uninit() (if next operation needs default packer state)

    Note: TTKernel reduce operations require a scaling_cb. The axis parameter maps
    to TTKernel ReduceDim enum.
  }];
}

def TTL_BlockReduceMaxOp : TTL_Op<"block_reduce_max"> {
  let summary = "Reduce max along specified axis";
  let arguments = (ins
    TTL_Tensor:$input,
    I64Attr:$axis  // Axis to reduce along
  );
  let results = (outs TTL_Tensor:$result);
  let description = [{
    Reduces input tensor along specified axis using max operation.
    Used for numerical stability in normalization patterns.

    Maps to TTKernel operations:
    - ttkernel.reduce_init(in_cb, scaling_cb, out_cb, ReduceFunc::Max, reduce_dim)
    - affine.for loop over reduction dimension tiles:
      - ttkernel.reduce_tile(in_cb, scaling_cb, in_idx, scaling_idx, dst_idx, Max, reduce_dim)
    - ttkernel.reduce_uninit() (if next operation needs default packer state)

    Note: Max reduction may also require scaling_cb depending on hardware requirements.
  }];
}

def TTL_BlockBcastOp : TTL_Op<"block_bcast"> {
  let summary = "Broadcast along specified axis";
  let arguments = (ins
    TTL_Tensor:$input,
    I64Attr:$axis  // Broadcast axis
  );
  let results = (outs TTL_Tensor:$result);
  let description = [{
    Broadcasts input tensor along the specified axis by replicating values.
    Useful for normalization patterns (such as attention mechanisms) where
    reduction fills one dimension and broadcast replicates across the other.

    Example: bcast(reduced_val, dim=1) replicates values along dimension 1.

    Maps to TTKernel operations:
    - ttkernel.unary_bcast_init(in_cb, out_cb, bcast_type)
    - affine.for loop over tiles:
      - ttkernel.unary_bcast(in_cb, in_idx, dst_idx, bcast_type)

    Note: The axis parameter maps to TTKernel BcastType enum.
  }];
}
```

#### 4.4.4 Materialization Operations

Operations for storing computed results from DST registers back to circular
buffers or memory.

**Store/Wait/Pop Pattern for Intermediate Result Reuse:**

A common pattern for reusing circular buffers efficiently within the same thread
is the store/wait/pop sequence. This enables minimal buffer_factor allocation
while reusing CB slots for multiple intermediate results:

```python
# Pattern for reusing intermediate results within same thread
o = out_cb.reserve()        # Reserve output slot
o.store(intermediate)       # Materialize result
out_cb.push()               # Signal available

intermediate = out_cb.wait()  # Re-acquire from CB
# ... use intermediate in next computation ...
out_cb.pop()                # Release slot

o = out_cb.reserve()        # Reuse same CB slot for next result
```

This pattern enables:
1. Minimal buffer_factor (often 2 is sufficient even for complex pipelines)
2. Intra-thread handoff of intermediate values
3. Efficient CB reuse without exceeding L1 capacity

**Semantics:**
- `ttl.block_store` + `ttl.cb_push` signals result is available for subsequent
  operations
- `ttl.cb_wait` + `ttl.cb_pop` acknowledges consumption and releases the slot
- `TTLValidatePass` verifies proper sequencing: each reserve has matching push,
  each wait has matching pop

```tablegen
def TTL_BlockStoreOp : TTL_Op<"block_store"> {
  let summary = "Store computation result to circular buffer block";
  let arguments = (ins TTL_Tensor:$dst, TTL_Tensor:$src);
  let description = [{
    Python API `block.store(value)` maps to this operation.
    Blocking operation that materializes computation result and stores in block.

    Block expression semantics:
    - Block expressions (e.g., `a_blk ** 2`, `a_blk + b_blk`) are evaluated lazily
    - Python operators (`**`, `+`, `-`, `*`, etc.) map to corresponding TTL block operations
    - `ttl.math.*` functions map to TTL math operations
    - `store()` materializes the result and blocks execution until complete

    Lowers to TTKernel operations:
    - affine.for loop over block tiles:
      - ttkernel.pack_tile(dst_idx, cb, tile_idx)

    Common usage: Part of store/wait/pop pattern for intermediate value reuse
    (see pattern explanation above).
  }];
}

def TTL_RequireDSTOp : TTL_Op<"require_dst", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Marker that SSA value must reside in DST register file";
  let arguments = (ins AnyType:$value);
  let description = [{
    Hint for liveness analysis to track which values need DST registers.
    Enables optimization passes to reorder operations and minimize register pressure.

    This is an IR-level marker removed during lowering after register allocation.
  }];
}
```

#### 4.4.5 `DST` Register Management

`DST` register operations are IR-level constructs for managing register
allocation at the block abstraction level.

Block-to-tile lowering context: TTL operations work on *blocks* (groups of
tiles, e.g., a 2x1 block = 2 tiles). Hardware operations in TTKernel work on
*individual tiles* (32x32 elements). During the `TTLLowerCompute` pass:

- A single `ttl.block_add` operating on a 2x1 block becomes an `affine.for` loop
  with 2 iterations
- Each iteration performs `ttkernel.add_tiles` on one tile
- `DST` register allocation happens *after* this expansion, when we know exactly
  how many tile-level operations execute

This lowering occurs because:
- Hardware constraint: TTKernel compute operations (add_tiles, mul_tiles, etc.)
  operate on single tiles, not blocks
- Register allocation: We need to see all tile operations and their lifetimes to
  allocate `DST` registers efficiently
- Fusion opportunities: Block-level IR enables analysis and fusion before
  committing to tile-level loops

`DST` register capacity is configuration-dependent (4-16 tiles) based on
`fp32_dest_acc_en` and `dst_full_sync_en` settings in `ComputeKernelConfig` (see
https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html#dst-register).
The `TTLAssignDSTRegisters` pass queries the kernel's compute_config to
determine capacity.

Spilling operates at tile granularity: if a kernel's liveness analysis reveals
more live values than available `DST` capacity, the compiler inserts spill
operations. These operate on *individual tiles*, not blocks, because spilling
occurs after block-to-tile lowering when the affine loops over tiles already
exist.

```tablegen
def TTL_AcquireDSTOp : TTL_Op<"acquire_dst"> {
  let summary = "Acquire DST register (IR marker, removed during lowering)";
  let arguments = (ins OptionalAttr<I32Attr>:$index);
  let results = (outs Index:$dst_reg);
  let description = [{
    Marks DST register acquisition for liveness analysis. Removed during lowering
    after register allocation completes.

    Example transformation through the pipeline:
      TTL IR (block level):
        %dst0 = ttl.acquire_dst
        %result = ttl.block_add %a, %b {dst_reg = %dst0}  // 2x1 block
        ttl.release_dst %dst0

      After TTLLowerCompute (tile level, DST ops still present):
        %dst0 = ttl.acquire_dst
        affine.for %i = 0 to 2 {  // 2 tiles in block
          %a_tile = ttkernel.copy_tile %a_cb, %i, 0
          %b_tile = ttkernel.copy_tile %b_cb, %i, 1
          ttkernel.add_tiles_init
          ttkernel.add_tiles 0, 1, 2
          ttkernel.pack_tile 2, %out_cb, %i
        }
        ttl.release_dst %dst0

      After TTLAssignDSTRegisters (DST ops removed, explicit indices):
        ttkernel.tile_regs_acquire
        affine.for %i = 0 to 2 {
          %a_tile = ttkernel.copy_tile %a_cb, %i, 0  // DST[0]
          %b_tile = ttkernel.copy_tile %b_cb, %i, 1  // DST[1]
          ttkernel.add_tiles_init
          ttkernel.add_tiles 0, 1, 2                  // DST[0] + DST[1] → DST[2]
          ttkernel.pack_tile 2, %out_cb, %i
        }
        ttkernel.tile_regs_commit

    Register indices (0, 1, 2) computed by TTLAssignDSTRegisters based on liveness.
  }];
}

def TTL_ReleaseDSTOp : TTL_Op<"release_dst"> {
  let summary = "Release DST register (IR marker, removed during lowering)";
  let arguments = (ins Index:$dst_reg);
  let description = [{
    Marks DST register release for liveness analysis. Removed during lowering.
  }];
}

def TTL_SpillTileToL1Op : TTL_Op<"spill_tile_to_l1"> {
  let summary = "Spill single tile from DST to L1 (operates at tile level)";
  let arguments = (ins
    Index:$tile_value,    // SSA value of tile currently in DST
    Index:$dst_reg        // DST register index
  );
  let results = (outs Index:$spilled_tile);
  let description = [{
    Spills a single tile from DST register to L1 when register pressure exceeds
    available capacity. Inserted by TTLAssignDSTRegisters after block-to-tile lowering.

    Operates at tile granularity: this operation appears inside affine loops after
    blocks have been lowered to per-tile operations. It spills individual tiles, not
    entire blocks.

    Example: Kernel with liveness exceeding DST capacity needs spilling:
      affine.for %i = 0 to %N {
        %t0 = ttkernel.copy_tile %cb0, %i, 0
        %t1 = ttkernel.copy_tile %cb1, %i, 1
        // ... operations filling available DST registers
        %t_result = ttkernel.add_tiles 0, 1, 2

        // Register pressure exceeded - spill oldest tile
        %spilled = ttl.spill_tile_to_l1 %t0, 0

        // DST[0] now available for reuse
        %t_next = ttkernel.mul_tiles 3, 4, 0  // Reuse DST[0]

        // Later restore if needed
        %restored = ttl.restore_tile_from_l1 %spilled
      }

    Lowers to TTKernel operations:
      %temp_cb = ttl.create_cb shape=[1,1], ...  // 1-tile temporary CB
      ttkernel.pack_tile %dst_reg, %temp_cb, 0   // DST → L1
  }];
}

def TTL_RestoreTileFromL1Op : TTL_Op<"restore_tile_from_l1"> {
  let summary = "Restore spilled tile from L1 to DST (operates at tile level)";
  let arguments = (ins Index:$spilled_tile);
  let results = (outs Index:$tile_value);
  let description = [{
    Restores a previously spilled tile from L1 back to DST register.
    Paired with ttl.spill_tile_to_l1.

    Lowers to TTKernel operations:
      %new_dst = ttl.acquire_dst
      ttkernel.copy_tile %temp_cb, 0, %new_dst  // L1 → DST
  }];
}
```

### 4.7 Utility Operations

```tablegen
def TTL_CoreOp : TTL_Op<"core", [Pure]> {
  let summary = "Get current core coordinates";
  let arguments = (ins OptionalAttr<I64Attr>:$dims);
  let results = (outs Variadic<Index>:$coordinates);
  let description = [{
    Returns core coordinates per TT-Lang spec v. TODO.

    Dimension rules (defaults to dims=2):
    - dims < grid_rank: Flatten highest dimensions
      grid=(8,8,8), dims=1 → x ∈ [0,64) where x = core_y*8 + core_z
    - dims > grid_rank: Pad highest dimensions with 0
      grid=(8,8), dims=3 → (x, y, 0)
    - dims == grid_rank: Return as-is

    Examples from TT-Lang spec:
      grid=(8,8), dims=1 → x ∈ [0,64)
      grid=(8,8,8), dims=2 → x ∈ [0,8), y ∈ [0,64)
      grid=(8,8), dims=3 → x ∈ [0,8), y ∈ [0,8), z=0

    Python API: `ttl.core()` or `ttl.core(dims=N)`
    Folds to constants. Validation enforced in frontend and TTLValidatePass.
  }];
}

def TTL_GridSizeOp : TTL_Op<"grid_size", [Pure]> {
  let summary = "Get grid dimensions";
  let arguments = (ins OptionalAttr<I64Attr>:$dims);
  let results = (outs Variadic<Index>:$sizes);
  let description = [{
    Returns grid dimensions per TT-Lang spec v. TODO.

    Dimension rules (defaults to dims=2):
    - dims < grid_rank: Flatten highest dimensions (multiply)
      grid=(8,8,8), dims=1 → 64 (8×8)
    - dims > grid_rank: Pad highest dimensions with 1
      grid=(8,8), dims=3 → (8, 8, 1)
    - dims == grid_rank: Return as-is

    Examples from TT-Lang spec:
      grid=(8,8), dims=1 → 64
      grid=(8,8,8), dims=2 → (8, 64)
      grid=(8,8), dims=3 → (8, 8, 1)

    Python API: `ttl.grid_size()` or `ttl.grid_size(dims=N)`
    Folds to constants. Validation enforced in frontend.
  }];
}

def TTL_CoreLinearOp : TTL_Op<"core_linear", [Pure]> {
  let summary = "Get linear core index in grid";
  let results = (outs Index:$linear_index);
  let description = [{
    Returns linearized core index for grid-based data distribution.
    For a 2D grid (X, Y), computes: core_y * grid_x + core_x.
    For higher dimensions, uses row-major linearization.

    Example: In 2x2 grid, cores map to linear indices:
      (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3

    Useful for distributing data across cores:
      linear_idx = ttl.core_dim(1)
      my_slice = tensor_accessor[linear_idx, :]

    Lowers to:
      %y = ttl.core(dims=0)
      %x = ttl.core(dims=1)
      %grid_x = ttl.grid_size(dims=1)
      %linear = affine.apply affine_map<(y, x, grid_x) -> (y * grid_x + x)>(%y, %x, %grid_x)
  }];
}

def TTL_GridSliceOp : TTL_Op<"grid_slice", [Pure]> {
  let summary = "Compute grid-strided tensor slice for current core";
  let arguments = (ins
    TTL_TensorAccessor:$accessor,
    I64ArrayAttr:$block_factors  // Tiles per core per dimension
  );
  let results = (outs TTL_TensorAccessor:$sliced_accessor);
  let description = [{
    Computes tensor slice for current core based on grid position and block factors.
    Automates the pattern of distributing tensor data across grid cores.

    Example: 2x2 grid, block_factors=[1, 1] (1 tile per core per dimension):
      Core (0,0) accesses tensor[0:1, 0:1]
      Core (0,1) accesses tensor[0:1, 1:2]
      Core (1,0) accesses tensor[1:2, 0:1]
      Core (1,1) accesses tensor[1:2, 1:2]

    Lowers to affine expressions computing slice bounds from core coordinates.
  }];
}
```

### 4.8 Compute Operations Inventory

**TTL → TTKernel Math Operation Mapping:**

| TTL Operation | TTKernel Operations | Implementation Complexity | MVP Priority |
|---------------|---------------------|---------------------------|--------------|
| **FPU Binary Operations** ||||
| `ttl.block_add` | `ttkernel.add_tiles_init`, `ttkernel.add_tiles` | Medium | **HIGH** |
| `ttl.block_sub` | `ttkernel.sub_tiles_init`, `ttkernel.sub_tiles` | Medium | **HIGH** |
| `ttl.block_mul` | `ttkernel.mul_tiles_init`, `ttkernel.mul_tiles` | Medium | **HIGH** |
| **FPU Matrix Operations** ||||
| `ttl.block_matmul` | `ttkernel.mm_init`, `ttkernel.matmul_tiles` | Medium | **HIGH** |
| **SFPU Unary Operations (Phase 2)** ||||
| `ttl.block_exp` | `ttkernel.init_sfpu`, `ttkernel.exp_tile` | Medium | MEDIUM |
| `ttl.block_log` | `ttkernel.init_sfpu`, `ttkernel.log_tile` | Medium | MEDIUM |
| `ttl.block_sqrt` | `ttkernel.init_sfpu`, `ttkernel.sqrt_tile` | Medium | MEDIUM |
| `ttl.block_rsqrt` | `ttkernel.init_sfpu`, `ttkernel.rsqrt_tile` | Medium | MEDIUM |
| `ttl.block_relu` | `ttkernel.init_sfpu`, `ttkernel.relu_tile` | Medium | MEDIUM |
| `ttl.block_gelu` | `ttkernel.init_sfpu`, `ttkernel.gelu_tile` | Medium | MEDIUM |
| `ttl.block_sigmoid` | `ttkernel.init_sfpu`, `ttkernel.sigmoid_tile` | Medium | MEDIUM |
| `ttl.block_tanh` | `ttkernel.init_sfpu`, `ttkernel.tanh_tile` | Medium | MEDIUM |
| **SFPU Trigonometric (Phase 3+)** ||||
| `ttl.block_sin` | `ttkernel.init_sfpu`, `ttkernel.sin_tile` | Medium | LOW |
| `ttl.block_cos` | `ttkernel.init_sfpu`, `ttkernel.cos_tile` | Medium | LOW |
| `ttl.block_tan` | `ttkernel.init_sfpu`, `ttkernel.tan_tile` | Medium | LOW |

**Operation Count Summary:**

- **MVP (Week 2-3)**: ~5 operations
  - 3 FPU binary: add, sub, mul
  - 1 FPU matrix: matmul
  - 1 special: block_store

- **Phase 2**: +8 operations
  - Common SFPU: exp, log, sqrt, rsqrt, relu, gelu, sigmoid, tanh

- **Full Coverage**: ~40-50 TTL math ops mapping to ~120+ TTKernel ops
  - All SFPU variants (~60 ops)
  - Reduction operations
  - Broadcast operations
  - Type conversion operations

**Implementation Strategy:**

1. **Establish pattern** with 3-5 operations (Week 2)
2. **Replicate pattern** for additional ops (incremental)
3. **Each TTL op requires:**
   - Operation definition in TTLOps.td
   - Python operator overload in operators.py
   - Lowering pattern in TTLLowerCompute.cpp
   - Lit test for validation





---

# 5.6 Structured Ops for DST Register Capacity-Based Tiling

Lower `TTL_BlockComputeOp` and block arithmetic operations to `linalg.generic`
structured operations rather than direct affine loops. This enables
capacity-constrained tiling for DST registers using the transform dialect.

#### 5.6.1 The DST Register Capacity Problem

Block compute operations process blocks of tiles (e.g., 8×8 tiles). Each tile
operation accumulates intermediate results in DST registers. Hardware provides
limited DST capacity (typically 4-16 registers depending on data type).

Consider an element-wise chain operating on an 8×8 block:

```python
# Three element-wise operations on 8×8 tile block
%result = ttl.block_add %a, %b      # 64 tiles
%result2 = ttl.block_mul %result, %c # 64 tiles
%result3 = ttl.block_exp %result2    # 64 tiles
```

Processing all 64 tiles at once would require 64 DST registers, far exceeding
capacity.

Solution: Tile the block into sub-blocks that fit within DST capacity,
processing tiles within each sub-block through the entire operation chain before
moving to the next sub-block.

#### 5.6.2 Loop Nest Structure: Block → Sub-block → Tile → Operation Chain

General loop structure for capacity-constrained fused block operations:

```
Outer sub-block loop (generated by transform tiling):
  For each sub-block fitting in DST capacity:
    For each operation in fusion chain:
      Init operation N
      Inner tile loop:
        Apply operation N to tile[i] → DST[i % capacity]
      Uninit operation N

    Pack sub-block from DST to L1
```

Parameters:
- Block size: e.g., 8×8 tiles (64 total)
- Sub-block size: Determined by DST capacity (e.g., 8×1 = 8 tiles for 8
  registers)
- Tile size: 32×32 elements (hardware granularity)
- Operations: Fused chain executing sequentially on each sub-block

Three-level hierarchy:
1. Block: User-visible abstraction (e.g., 8×8 tiles)
2. Sub-block: Compiler-inserted tiling for capacity (e.g., 8×1 tiles)
3. Tile: Hardware unit of execution (32×32 elements)

Within each sub-block, operations execute sequentially with init/process/uninit
pattern, reusing DST registers across the operation chain.

#### 5.6.3 Why linalg.generic Enables This Tiling

The transform dialect provides purpose-built primitives for capacity-constrained
tiling with linalg.generic, which are not available for direct affine loops.

Comparison:

| Capability | linalg.generic | Direct Affine Loops |
|------------|----------------|---------------------|
| Capacity-aware tiling | `transform.structured.multitile_sizes` computes sub-block sizes respecting target capacity | No equivalent operation; manual computation required |
| Dynamic tile sizes | Built-in via TileUsingInterface callbacks | Manual size computation in C++ |
| Tile-and-fuse | TilingInterface provides built-in fusion primitives | Manual fusion logic |
| Transform dialect ops | ~10 tiling operations (tile_using_for, multitile_sizes, etc.) | 2 ops (only min/max simplification) |
| Multi-level tiling | `continuous_tile_sizes` for hierarchical tiling | No support |
| Iterator type tracking | Structured op preserves parallel vs reduction semantics | Limited structural information |

Key transform operations:

1. `transform.structured.multitile_sizes` (LinalgTransformOps.td:745-824):
   Computes two sub-block sizes (sz1, sz2) both ≤ target_size, ensuring complete
   coverage: n×sz1 + m×sz2 = dimension_size. Set target_size based on DST
   capacity.

2. `transform.structured.tile_using_for` (LinalgTransformOps.td:2101-2193):
   Tiles linalg ops using scf.for loops with static or dynamic tile sizes and
   optional loop interchange.

3. `transform.structured.continuous_tile_sizes`
   (LinalgTransformOps.td:2056-2095): Generates diminishing tile sizes for
   multi-level tiling (e.g., [9, 4, 2, 1]).

#### 5.6.4 Lowering Strategy: TTL Block Ops → linalg.generic → Tiled Loops → TTKernel

Phase 1: Block operations to linalg.generic

```mlir
// Input: TTL fused compute region (3 operations on 8×8 block)
%result = ttl.compute_region ins(%a, %b, %c) outs(%out) {
^bb0(%a_blk, %b_blk, %c_blk):
  %add = ttl.block_add %a_blk, %b_blk      // Op 1
  %mul = ttl.block_mul %add, %c_blk        // Op 2
  %exp = ttl.block_exp %mul                 // Op 3
  ttl.yield %exp
}

// Output: linalg.generic with tile operations
%result = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                   affine_map<(d0, d1) -> (d0, d1)>,
                   affine_map<(d0, d1) -> (d0, d1)>,
                   affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%a, %b, %c : memref<8x8x!ttcore.tile<32x32,f32>, #L1>,
                   memref<8x8x!ttcore.tile<32x32,f32>, #L1>,
                   memref<8x8x!ttcore.tile<32x32,f32>, #L1>)
  outs(%out : memref<8x8x!ttcore.tile<32x32,f32>, #L1>) {
^bb0(%a_tile: !ttcore.tile<32x32,f32>,
     %b_tile: !ttcore.tile<32x32,f32>,
     %c_tile: !ttcore.tile<32x32,f32>,
     %out_tile: !ttcore.tile<32x32,f32>):
  %add = ttl.tile_add %a_tile, %b_tile
  %mul = ttl.tile_mul %add, %c_tile
  %exp = ttl.tile_exp %mul
  linalg.yield %exp : !ttcore.tile<32x32,f32>
}
```

The linalg.generic:
- `indexing_maps` describe how block indices (d0, d1) map to tile indices
- `iterator_types` specify parallel dimensions
- Body operates at tile granularity with full operation chain
- Implements TilingInterface for uniform tiling semantics

Phase 2: Capacity-based tiling via transform dialect

```mlir
transform.sequence failures(propagate) {
^bb0(%module: !transform.any_op):
  %generics = transform.structured.match ops{["linalg.generic"]} in %module
    : (!transform.any_op) -> !transform.any_op

  // Compute sub-block size respecting DST capacity (8 registers → 8×1 sub-blocks)
  %sz1, %sz2, %split = transform.structured.multitile_sizes %generics
    { target_size = 8, dimension = 0, divisor = 1 }
    : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>)

  // Split at computed boundary
  %low, %high = transform.structured.split %generics after %split { dimension = 0 }
    : !transform.any_op, !transform.param<i64>

  // Tile each partition (for simplicity, assume %sz1 = 1, i.e., 1×8 sub-blocks)
  %tiled_low, %loops_low = transform.structured.tile_using_for %low [1, 0]
    : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)
}
```

Result after tiling: 8×8 block split into scf.for loop with 8 iterations, each
processing 1×8 sub-block (8 tiles) through complete operation chain:

```mlir
// After transform tiling: scf.for with smaller linalg.generic inside
scf.for %row = 0 to 8 step 1 {
  // Process 1×8 sub-block (8 tiles in columns 0-7 of row %row)
  %sub_result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,    // Map to columns
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%a_sub, %b_sub, %c_sub : memref<8x!ttcore.tile<32x32,f32>, #L1>,
                                 memref<8x!ttcore.tile<32x32,f32>, #L1>,
                                 memref<8x!ttcore.tile<32x32,f32>, #L1>)
    outs(%out_sub : memref<8x!ttcore.tile<32x32,f32>, #L1>) {
  ^bb0(%a_tile: !ttcore.tile<32x32,f32>,
       %b_tile: !ttcore.tile<32x32,f32>,
       %c_tile: !ttcore.tile<32x32,f32>,
       %out_tile: !ttcore.tile<32x32,f32>):
    // Full operation chain on each tile
    %add = ttl.tile_add %a_tile, %b_tile
    %mul = ttl.tile_mul %add, %c_tile
    %exp = ttl.tile_exp %mul
    linalg.yield %exp : !ttcore.tile<32x32,f32>
  }
  // where %a_sub, %b_sub, %c_sub, %out_sub are subviews:
  // %a_sub = memref.subview %a[%row, 0] [1, 8] [1, 1]
  //   : memref<8x8x!ttcore.tile<32x32,f32>, #L1> to memref<8x!ttcore.tile<32x32,f32>, #L1>
}
```

Key properties after tiling:
- Outer loop (scf.for) iterates over sub-blocks (rows)
- Each iteration processes a 1×8 sub-block (8 tiles)
- linalg.generic body still contains full operation chain
- Subviews extract the appropriate slice of input/output memrefs
- This linalg.generic now operates on 8 tiles (fits in 8 DST registers)

Phase 3: Lower tiled linalg.generic to TTKernel

The smaller linalg.generic operations (processing 8 tiles each) are lowered to
TTKernel tile operations with init/uninit sequences and DST register reuse:

```mlir
// After lowering tiled linalg to TTKernel
scf.for %row = 0 to 8 step 1 {
  // Get subviews for this sub-block
  %a_sub = memref.subview %a[%row, 0] [1, 8] [1, 1]
  %b_sub = memref.subview %b[%row, 0] [1, 8] [1, 1]
  %c_sub = memref.subview %c[%row, 0] [1, 8] [1, 1]
  %out_sub = memref.subview %out[%row, 0] [1, 8] [1, 1]

  // Operation 1: add (load from L1 CBs to DST)
  ttkernel.add_tiles_init %cb_a, %cb_b
  affine.for %col = 0 to 8 {
    %dst_idx = affine.apply affine_map<(c) -> (c)>(%col)
    ttkernel.add_tiles %cb_a, %cb_b, %col, %col, %dst_idx  // → DST[0..7]
  }

  // Operation 2: mul (DST → DST, reusing registers)
  ttkernel.mul_tiles_init %dst, %cb_c
  affine.for %col = 0 to 8 {
    %dst_idx = affine.apply affine_map<(c) -> (c)>(%col)
    ttkernel.mul_tiles %dst, %cb_c, %dst_idx, %col, %dst_idx  // DST[i] *= c[i]
  }

  // Operation 3: exp (DST → DST, reusing registers)
  ttkernel.exp_tiles_init %dst
  affine.for %col = 0 to 8 {
    %dst_idx = affine.apply affine_map<(c) -> (c)>(%col)
    ttkernel.exp_tile %dst_idx  // DST[i] = exp(DST[i])
  }

  // Pack 1×8 sub-block from DST to L1 (single pack after all operations)
  affine.for %col = 0 to 8 {
    %dst_idx = affine.apply affine_map<(c) -> (c)>(%col)
    ttkernel.pack_tile %dst_idx, %cb_out, %col
  }
  ttkernel.cb_push_back %cb_out, 8
}
```

Each outer iteration (scf.for %row):
1. Extracts 1×8 sub-block via subviews
2. Loads 8 tiles from L1 CBs to DST via operation 1 (init → 8 add_tiles calls)
3. Applies operations 2 and 3 sequentially, reusing DST registers (no L1
   traffic)
4. Packs 8 tiles from DST back to L1 after all operations complete

Total DST usage: 8 registers (within capacity). Data flows through operation
chain in DST without intermediate L1 writes.

#### 5.6.5 Concrete Walkthrough: Element-wise Chain with Register Reuse

Scenario: Three element-wise operations on an 8×8 block (64 tiles total), 8 DST
registers available.

Strategy: Process 1×8 sub-blocks. For each sub-block, execute all three
operations in sequence with register reuse.

Full walkthrough for iteration %row = 0:

```mlir
// Iteration 0: Process sub-block at row 0, cols 0-7 (8 tiles)

// Initialize operation 1 (add)
ttkernel.add_tiles_init %cb_a, %cb_b

// Inner loop: Process 8 tiles with operation 1, store to DST[0..7]
affine.for %col = 0 to 8 {
  %tile_a = ttkernel.cb_wait_front %cb_a, 1
  %tile_b = ttkernel.cb_wait_front %cb_b, 1
  %dst_idx = affine.apply affine_map<(c) -> (c)>(%col)  // DST[0..7]
  ttkernel.add_tiles %cb_a, %cb_b, %col, %col, %dst_idx
}
// Now DST[0..7] contain results of op 1 for 8 tiles

// Uninitialize op 1, initialize operation 2 (mul)
ttkernel.mul_tiles_init %dst, %cb_c  // Note: reads from DST, not CB

// Inner loop: Process 8 tiles with operation 2 (read from DST, write to DST)
affine.for %col = 0 to 8 {
  %tile_c = ttkernel.cb_wait_front %cb_c, 1
  %dst_idx = affine.apply affine_map<(c) -> (c)>(%col)
  // Reads from DST[dst_idx], multiplies with c, writes back to DST[dst_idx]
  ttkernel.mul_tiles %dst, %cb_c, %dst_idx, %col, %dst_idx
}
// Now DST[0..7] contain results of op 2 for 8 tiles

// Uninitialize op 2, initialize operation 3 (exp)
ttkernel.exp_tiles_init %dst  // Reads from DST

// Inner loop: Process 8 tiles with operation 3 (read from DST, write to DST)
affine.for %col = 0 to 8 {
  %dst_idx = affine.apply affine_map<(c) -> (c)>(%col)
  // Reads from DST[dst_idx], applies exp, writes back to DST[dst_idx]
  ttkernel.exp_tile %dst_idx
}
// Now DST[0..7] contain final results for 8 tiles

// Uninitialize op 3

// Pack 1×8 sub-block from DST to L1 output CB
affine.for %col = 0 to 8 {
  %dst_idx = affine.apply affine_map<(c) -> (c)>(%col)
  ttkernel.pack_tile %dst_idx, %cb_out, %col
}
ttkernel.cb_push_back %cb_out, 8
```

This pattern repeats for outer loop iterations 1-7, processing successive 1×8
sub-blocks (rows 1-7).

Key insight: All three operations execute on the same 8 tiles before packing,
maximizing DST register reuse. Data flows through DST registers without
intermediate L1 writes, staying within 8 DST register capacity.

Result: 8×8 block processed in 8 sub-blocks of 8 tiles each, with complete
operation fusion within each sub-block.

#### 5.6.6 Implementation in TTLLowerCompute Pass

Pass structure:

1. Match TTL block operations and ttl.compute_region operations

2. Generate linalg.generic for block operations:
   - Compute indexing maps from block shape
   - Set iterator types (parallel for element-wise, reduction for matmul K-dim)
   - Generate body with complete tile-level operation chain

3. Implement TilingInterface for TTL operations:
   - `getLoopIteratorTypes()`: Returns parallel/reduction iterator types
   - `getIterationDomain()`: Returns loop bounds based on block shape
   - `getTiledImplementation()`: Generates tiled code with capacity constraints

4. Apply transform dialect schedule for capacity-based tiling:
   - Generates scf.for loops with smaller linalg.generic operations
   - Each linalg.generic processes a sub-block fitting in DST capacity

5. Lower tiled linalg operations to TTKernel:
   - Generate init/uninit sequences for each operation
   - Emit tile loops with DST register allocation
   - Insert pack operations after operation chains complete
   - Preserve DST register reuse across operations

#### 5.6.7 DST Register Capacity Analysis

`TTLInferDSTRequirements` pass computes:
1. DST register cost per tile operation (add: 1 DST, matmul: 4 DST, etc.)
2. Liveness ranges for each tile operation in linalg.generic body
3. Peak DST usage per iteration
4. Maximum sub-block size that fits in available registers

For fused operations, analysis accounts for register reuse. Example:

```mlir
// Fused chain: add → mul → exp
%add = ttl.tile_add %a, %b        // Requires 1 DST
%mul = ttl.tile_mul %add, %c      // Reuses same DST (add result dead)
%exp = ttl.tile_exp %mul          // Reuses same DST (mul result dead)
// Peak usage: 1 DST per tile (not 3)
```

Transform script generation:
```mlir
// If peak usage is 1 DST per tile, total capacity 16:
// Can process 16 tiles concurrently (16×1 sub-blocks)
transform.structured.multitile_sizes %op { target_size = 16, ... }

// If peak usage is 4 DST per tile (matmul), total capacity 16:
// Can process 4 tiles concurrently (4×1 sub-blocks)
transform.structured.multitile_sizes %op { target_size = 4, ... }
```

See:
- Upstream linalg transforms:
  `mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td`
- TilingInterface: `mlir/include/mlir/Interfaces/TilingInterface.td`
- SCF tiling utilities:
  `mlir/include/mlir/Dialect/SCF/Transforms/TileUsingInterface.h`


