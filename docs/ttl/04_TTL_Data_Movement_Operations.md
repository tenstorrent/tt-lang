# TTL Dialect Data Movement Operations

**Part of**: [TTL Dialect Design Plan](01_TTL_Dialect_Plan.md)

This document specifies data movement operations, resource creation,
synchronization, and MLIR interface requirements for the TTL dialect.

## Table of Contents

- [4.2 Resource Creation](#42-resource-creation)
- [4.5 Data Movement Operations](#45-data-movement-operations)
- [4.6 Synchronization Operations](#46-synchronization-operations)
- [4.9 MLIR Interface Requirements](#49-mlir-interface-requirements)
  - [4.9.1 MemoryEffectsOpInterface](#491-memoryeffectsopinterface)
  - [4.9.2 Custom TTL Interfaces](#492-custom-ttl-interfaces)
  - [4.9.3 Bufferization Interfaces](#493-bufferization-interfaces)
  - [4.9.4 Liveness Analysis Integration](#494-liveness-analysis-integration)
  - [4.9.5 Implementation Priority](#495-implementation-priority)

## Related Documents

- [Back to Overview](01_TTL_Dialect_Plan.md)
- [Type System](02_TTL_Type_System.md) - Types used by these operations
- [Compute Operations](03_TTL_Compute_Operations.md) - Compute operations
- [Compilation Pipeline](06_TTL_Compilation_Pipeline.md) - How these operations
  are lowered



# 4. Data Movement Operations (Selected Sections)

### 4.2 Resource Creation

```tablegen
def TTL_CreateCBOp : TTL_Op<"create_cb"> {
  let summary = "Create circular buffer in L1 memory";
  let arguments = (ins
    I64ArrayAttr:$shape,                // Elements per block
    TypeAttr:$element_type,             // !ttcore.tile<32x32, f32> or scalar type (f32, bf16, etc.)
    I64Attr:$buffer_factor,             // Number of blocks
    OptionalAttr<I32Attr>:$buffer_index,      // Optional explicit CB number (0-31)
    OptionalAttr<I64Attr>:$page_size,         // Optional page size in bytes
    OptionalAttr<"ttnn::CoreRangeSetAttr">:$core_ranges  // Reuse ttnn::CoreRangeSetAttr from tt-mlir
  );
  let results = (outs TTL_CircularBuffer:$result);
  let description = [{
    Python API `ttl.make_circular_buffer_like(tensor, shape=..., buffer_factor=...)`
    maps to this operation. The Python frontend extracts tensor properties
    (dtype, layout) and calls `ttl.create_cb` with the extracted parameters.
    The "likeness" refers to deriving element_type from the input tensor's layout
    and properties.

    Memory space: Circular buffers always reside in L1 memory. DRAM, DST, and System
    memory are not valid memory spaces for circular buffers. DST registers are managed
    exclusively by the TTLAssignDSTRegisters pass.

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
    I64ArrayAttr:$src,                            // Source core [x, y, ...]
    OptionalAttr<I64ArrayAttr>:$dst,              // Unicast destination [x, y, ...]
    OptionalAttr<ArrayAttr>:$dst_range            // Multicast range [SliceAttr, SliceAttr, ...]
  );
  let results = (outs TTL_Pipe:$pipe);
  let description = [{
    Python API `ttl.Pipe(src=..., dst=...)` maps to this operation.

    Per TT-Lang spec, the `dst` argument determines unicast vs multicast:
    - Unicast: dst is a single core coordinate tuple (e.g., `(0, y)`)
    - Multicast: dst contains slice objects (e.g., `(x, slice(1, grid_y))`)

    Unicast example:
      Python: ttl.Pipe(src=(1,0), dst=(0,0))
      TTL IR: ttl.create_pipe src=[1,0], dst=[0,0]

    Multicast example (preserves slice semantics):
      Python: ttl.Pipe(src=(x,0), dst=(x, slice(1, grid_y)))
      TTL IR: ttl.create_pipe src=[x,0],
                              dst_range=[#ttl.slice<x,x+1,1>, #ttl.slice<1,grid_y,1>]

    The dst_range encodes Python slice objects per dimension with (start, stop, step).
    Half-open intervals: [start, stop). Step enables patterns like "every other core".
  }];
}

def TTL_CreatePipeNetOp : TTL_Op<"create_pipenet"> {
  let summary = "Create pipe network from list of pipes";
  let arguments = (ins Variadic<TTL_Pipe>:$pipes);
  let results = (outs TTL_PipeNet:$net);
  let description = [{
    Python API `ttl.PipeNet([pipe1, pipe2, ...])` maps to this operation.

    Creates a compile-time PipeNet value grouping the provided pipes. The PipeNet stores
    the pipe list in its operands (not in the type) and provides:
    - Validation: Verifiers check all pipes have matching src/dst guards in if_pipe_src/if_pipe_dst
    - Analysis: Semaphore inference pass queries the pipe list to generate synchronization
    - Lowering: TTLLowerDataMovement extracts pipes, generates ttkernel NOC ops per pipe

    Example:
      Python (using TT-Lang spec naming src/dst):
        net = ttl.PipeNet([
          ttl.Pipe(src=(0,0), dst=(slice(1,4), 0)),  # Multicast via slice in dst
          ttl.Pipe(src=(1,0), dst=(0,0))              # Unicast
        ])

      TTL IR:
        %pipe1 = ttl.create_pipe src=[0,0], dst_range=[#ttl.slice<1,4,1>, 0]
        %pipe2 = ttl.create_pipe src=[1,0], dst=[0,0]
        %net = ttl.create_pipenet %pipe1, %pipe2

    The PipeNet is used with ttl.if_pipe_src and ttl.if_pipe_dst operations.
  }];
}
```

#### Semaphore Operations

```tablegen
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

### 4.5 Data Movement Operations

#### Copy and Accessor Operations

```tablegen
def TTL_IndexAccessorOp : TTL_Op<"index_accessor"> {
  let summary = "Index into tensor accessor with tile coordinates";
  let arguments = (ins
    TTL_TensorAccessor:$accessor,
    Variadic<Index>:$indices         // Tile coordinates
  );
  let results = (outs AnyType:$tile_ref);  // Reference to tile for DMA
}

def TTL_CopyOp : TTL_Op<"copy"> {
  let summary = "Asynchronous copy between tensor tiles, CBs, and pipes";
  let arguments = (ins
    AnyType:$src,                    // TensorAccessor, CB, or Pipe
    AnyType:$dst,                    // TensorAccessor, CB, or Pipe
    OptionalAttr<I64ArrayAttr>:$src_coords,  // Tile coordinates if accessor
    OptionalAttr<I64ArrayAttr>:$dst_coords   // Tile coordinates if accessor
  );
  let results = (outs TTL_TransferHandle:$xf);
  let description = [{
    Unified DMA operation handling all transfer combinations:
    - Tensor slice → CB (read from DRAM/L1 using TensorAccessor)
    - CB → Tensor slice (write to DRAM/L1 using TensorAccessor)
    - CB → Pipe → CB (inter-core transfer)

    Operand types and attributes determine lowering:
    - TensorAccessor operands: lower to ttkernel.tensor_accessor + ttkernel NOC operations
    - Pipe operands: lower to ttkernel.noc_async_write_multicast or unicast
    - TensorAccessor layout determines which NOC API: shard, page, or tile

    Returns transfer handle (SSA value of type !ttl.transfer_handle) for ordering.
    Python API `ttl.copy()` returns a transfer handle object with a `wait()`
    method, which maps to `ttl.wait %xf` operation.

    TensorAccessor lowering based on layout:
    - Sharded layout: ttkernel.noc_async_read_shard / ttkernel.noc_async_write_shard
    - Interleaved layout: ttkernel.noc_async_read_page / ttkernel.noc_async_write_page
    - Tiled layout: ttkernel.noc_async_read_tile / ttkernel.noc_async_write_tile
    All variants use ttkernel.tensor_accessor - see section 3.3 for complete example.

    Pipe guard requirement (TT-Lang spec compliance):
    - Every ttl.copy with a Pipe operand MUST be guarded by ttl.if_pipe_src (pipe as dst)
      or ttl.if_pipe_dst (pipe as src)
    - For each pipe, there must be a matching send/receive pair:
      - One core uses ttl.copy(block, pipe) inside ttl.if_pipe_src
      - One or more cores use ttl.copy(pipe, block) inside ttl.if_pipe_dst
    - Unguarded pipe copies are invalid per TT-Lang spec
    - Operation verifiers enforce this requirement and reject unguarded or unmatched pipes

    Note: TRID-specific barrier operations (`ttkernel.noc_async_read_barrier_with_trid`,
    `ttkernel.noc_async_write_barrier_with_trid`) and TensorAccessor-specific operations
    (`ttkernel.noc_async_read_shard`, `ttkernel.noc_async_write_shard`,
    `ttkernel.noc_async_read_page`, `ttkernel.noc_async_write_page`) need to be added to
    TTKernel dialect before TTL lowering can be implemented. These are straightforward
    additions following existing TTKernel operation patterns. The underlying TT-Metal
    runtime already provides these functions.

    See: `tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td` definitions
    `TTKernel_NocAsyncReadBarrierOp` and `TTKernel_NocAsyncWriteBarrierOp`.
  }];
}
```

#### Pipe Conditional Operations

```tablegen
def TTL_IfPipeSrcOp : TTL_Op<"if_pipe_src"> {
  let summary = "Execute region for each pipe where current core is source";
  let arguments = (ins TTL_PipeNet:$net);
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Python API `net.if_src(pipe_src_callback)` maps to this operation, where `net`
    is a PipeNet object returned by `ttl.PipeNet([...])`.

    Semantics: For each pipe in the network where current core matches src,
    logically invoke the region body with that pipe as block argument. Invocations
    for different pipes may execute in parallel (independent pipe transfers).

    The PipeNet argument enables:
    - Validation: Verifiers check all pipes have matching src/dst guards
    - Analysis: Semaphore inference queries the pipe list from create_pipenet operands
    - Lowering: TTLLowerDataMovement extracts pipes, determines core matches, generates NOC ops

    Example:
      Python (TT-Lang spec):
        net = ttl.PipeNet([pipe1, pipe2, pipe3])
        def pipe_src(pipe):
            xf = ttl.copy(blk, pipe)
            xf.wait()
        net.if_src(pipe_src)

      TTL IR:
        // Example 1: Explicit pipe list
        %pipe1 = ttl.create_pipe src=[0,0], dst=[1,0]
        %pipe2 = ttl.create_pipe src=[0,0], dst=[2,0]
        %pipe3 = ttl.create_pipe src=[0,0], dst=[3,0]
        %net = ttl.create_pipenet %pipe1, %pipe2, %pipe3

        // Example 2: Pipes from Python list comprehension (conceptual representation)
        // Note: This shows the logical structure. The Python frontend will typically
        // unroll list comprehensions at compile time and emit all pipes as separate SSA
        // values passed to create_pipenet as variadic arguments, as shown in Example 1.
        %grid_x = ttl.grid_size dims=1
        %pipe_0 = ttl.create_pipe src=[0,0], dst=[1,0]
        %pipe_1 = ttl.create_pipe src=[0,0], dst=[2,0]
        // ... (one per grid_x value, unrolled at compile time)
        %net = ttl.create_pipenet %pipe_0, %pipe_1, ...

        ttl.if_pipe_src %net {
          ^bb0(%pipe: !ttl.pipe):
            %xf = ttl.copy %blk, %pipe
            ttl.wait %xf
        }

    Lowering to TTKernel:
      The pass determines which pipes match current core and generates code for each.
      Execution model (parallel vs serial) determined by dependency analysis:
        - If region operations are independent: parallel execution (scf.forall or unrolled)
        - If region has dependencies: serial execution (scf.for)

      Example (parallel - independent pipe writes):
        // Core (0,0): matches pipe1 and pipe3 src
        scf.forall (%pipe_idx) in (0, 2) {  // Parallel over pipe1, pipe3
          %pipe = select %pipe_idx : pipe1 or pipe3
          ttkernel.noc_async_write... using %pipe coordinates
        }

    The region block argument receives each matching pipe, enabling pipe-specific operations.
  }];
}

def TTL_IfPipeDstOp : TTL_Op<"if_pipe_dst"> {
  let summary = "Execute region for each pipe where current core is destination";
  let arguments = (ins TTL_PipeNet:$net);
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Python API `net.if_dst(pipe_dst_callback)` maps to this operation, where `net`
    is a PipeNet object returned by `ttl.PipeNet([...])`.

    Semantics: For each pipe in the network where current core matches the pipe's
    dst (unicast) or is within dst_range (multicast), logically invoke
    the region body with that pipe as block argument. Invocations for different
    pipes may execute in parallel.

    The PipeNet argument provides same benefits as ttl.if_pipe_src: validation via
    verifiers, semaphore inference via pipe list query, and lowering to ttkernel NOC ops.

    Example:
      Python (TT-Lang spec):
        net = ttl.PipeNet([
          ttl.Pipe(src=(x, 0), dst_range=(x, slice(1, grid_y)))
          for x in range(grid_x)
        ])
        def pipe_dst(pipe):
            xf = ttl.copy(pipe, blk)
            xf.wait()
        net.if_dst(pipe_dst)

      TTL IR:
        // Pipes created in scf.for loop (from Python list comprehension)
        // Note: Actual implementation would use a container type (e.g., builtin list,
        // or emit all pipes as separate SSA values and pass to create_pipenet as variadic args)
        %grid_x = ttl.grid_size dims=1

        // Simplified representation - actual lowering may inline the loop and
        // create all pipes as separate SSA values:
        %pipe_0 = ttl.create_pipe src=[0,0], dst_range=[0, #ttl.slice<1,grid_y,1>]
        %pipe_1 = ttl.create_pipe src=[1,0], dst_range=[1, #ttl.slice<1,grid_y,1>]
        // ... (one per grid_x value)
        %net = ttl.create_pipenet %pipe_0, %pipe_1, ...

        ttl.if_pipe_dst %net {
          ^bb0(%pipe: !ttl.pipe):
            %xf = ttl.copy %pipe, %blk
            ttl.wait %xf
        }

    Lowering to TTKernel:
      The pass queries the PipeNet for pipes matching current core (checking
      dst or dst_range). Execution model determined by dependency analysis.

    The region block argument receives each matching pipe, enabling pipe-specific operations.
  }];
}

def TTL_WaitOp : TTL_Op<"wait"> {
  let summary = "Wait for DMA transfer to complete";
  let arguments = (ins TTL_TransferHandle:$xf);
  let description = [{
    Explicit wait on transfer handle.

    Lowering strategy: TTL transfer handles map to TTKernel transaction IDs (TRIDs) enabling
    per-transfer wait semantics. The compiler assigns each ttl.copy operation a unique TRID
    (0-15) and lowers ttl.wait to TRID-specific barriers:
    - `ttkernel.noc_async_read_barrier_with_trid(trid, noc)` for read transfers
    - `ttkernel.noc_async_write_barrier_with_trid(trid, noc)` for write transfers

    Implementation note: These TRID-specific barrier operations need to be added to the
    TTKernel dialect (straightforward addition following existing barrier operation patterns).
    The underlying TT-Metal runtime already provides these functions in `tt_metal/hw/inc/dataflow_api.h`.
    See implementation roadmap in `docs/ttl/07_TTL_Implementation_and_Runtime.md` for details.

    Per-transfer synchronization: Unlike global barriers that wait for all pending DMAs,
    TRID-based barriers wait only for the specific transfer with matching ID. This enables
    better DMA overlap by allowing other transfers to remain in flight.

    TRID allocation: The compiler manages TRID assignment (16 IDs available: 0x0-0xF) and
    barrier counter resets. When more than 16 concurrent transfers exist, the compiler may
    fall back to global barriers or insert intermediate waits to free TRIDs.

    Pipe transfers: For transfers involving pipes (inter-core communication), the wait
    resolves to semaphore synchronization that coordinates pipe producers and consumers
    instead of a DMA barrier.

    Direction determination: The barrier direction (read vs write) is determined from the
    transfer handle's source and destination:
    - Tensor/CB → CB: read barrier (DMA read from DRAM/L1)
    - CB → Tensor/CB: write barrier (DMA write to DRAM/L1)
    - CB → Pipe or Pipe → CB: semaphore wait (no DMA barrier)

    See: `tt-metal/hw/inc/dataflow_api.h` for TRID barrier functions and
    `tests/tt_metal/tt_metal/data_movement/transaction_id/README.md` for usage examples.
  }];
}

def TTL_DMABarrierOp : TTL_Op<"dma_barrier"> {
  let summary = "Global DMA barrier - wait for all pending DMAs";
  let description = [{
    Ensures all prior DMA operations complete. More efficient than
    individual waits when multiple transfers are pending.
  }];
}
```

Note: Atomic barriers for semaphore operations are inserted automatically during
lowering by `TTLLowerDataMovement` based on memory effects analysis. No explicit
`ttl.atomic_barrier` operation is provided at the TTL level but could be added
in future.

### 4.5.1 Pipe Synchronization and Semaphore Inference

Pipes in TTL are high-level abstractions that lower to hardware DMAs plus
semaphore-based synchronization. The compiler automatically infers and inserts
semaphores for correct pipe operation through the `TTLInferPipeSemaphores` pass.

For each pipe in a PipeNet, two semaphores are created:

1. Ready semaphore: Receivers signal readiness to sender
2. Validity semaphore: Sender signals data availability to receivers

Pattern for multicast pipe (1 sender → N receivers):
- Sender waits for ready_sem == N, then resets to 0
- Sender performs DMA multicast
- Sender sets validity_sem via multicast
- Each receiver increments ready_sem (unicast to sender)
- Each receiver waits for validity_sem == 1

Pattern for unicast pipe (1 sender → 1 receiver):
- Simplified protocol without multicast operations
- May omit ready semaphore for single-producer-single-consumer patterns

Users can override automatic inference by explicitly creating semaphores and
using manual synchronization operations within pipe condition bodies.

See [05_TTL_Multicast_Implementation.md](05_TTL_Multicast_Implementation.md) for
detailed multicast patterns and C++ code generation examples.

### 4.6 Synchronization Operations

```tablegen
def TTL_SemaphoreWaitEqOp : TTL_Op<"semaphore_wait_eq"> {
  let summary = "Wait for semaphore to equal specified value";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I32Attr:$value
  );
  let description = [{
    Python API `sem.wait_eq(n)` maps to this operation.

    Blocking operation that waits until semaphore equals the specified value.

    Example:
      Python: my_sem.wait_eq(num_cores)
      TTL IR: ttl.semaphore_wait_eq %sem, num_cores

    Maps to TTKernel operation:
      ttkernel.noc_semaphore_wait(sem_addr, value)
  }];
}

def TTL_SemaphoreWaitMinOp : TTL_Op<"semaphore_wait_min"> {
  let summary = "Wait for semaphore to be at least specified value";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I32Attr:$value
  );
  let description = [{
    Python API `sem.wait_min(n)` maps to this operation.

    Blocking operation that waits until semaphore is >= the specified value.

    Example:
      Python: my_sem.wait_min(threshold)
      TTL IR: ttl.semaphore_wait_min %sem, threshold

    Maps to TTKernel operation:
      ttkernel.noc_semaphore_wait_min(sem_addr, value)
  }];
}

def TTL_SemaphoreSetOp : TTL_Op<"semaphore_set"> {
  let summary = "Set semaphore value (local or remote)";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I32Attr:$value
  );
  let description = [{
    Python API `sem.set(val)` maps to this operation.

    Sets the semaphore to the specified value. The semaphore operand determines
    whether this is a local or remote operation:
    - Local semaphore (from ttl.create_semaphore): sets local semaphore
    - Remote semaphore (from ttl.get_remote_semaphore): sets remote core's semaphore
    - Multicast semaphore (from ttl.get_remote_multicast_semaphore): multicasts set

    Examples:
      Python: my_sem.set(0)  # Local set
      TTL IR: ttl.semaphore_set %sem, 0

      Python: remote_sem.set(1)  # Remote unicast set
      TTL IR: %remote = ttl.get_remote_semaphore %sem, core=[0,0]
              ttl.semaphore_set %remote, 1

      Python: mcast_sem.set(1)  # Remote multicast set
      TTL IR: %mcast = ttl.get_remote_multicast_semaphore %sem
              ttl.semaphore_set %mcast, 1

    Maps to TTKernel operations based on semaphore type:
      Local: ttkernel.noc_semaphore_set(local_core, sem_addr, value)
      Remote unicast: ttkernel.noc_semaphore_set(target_core, sem_addr, value)
      Remote multicast: ttkernel.noc_semaphore_set_multicast(noc_multicast_addr, value)

    Note: Multicast semaphores support set but not increment (hardware limitation).
  }];
}

def TTL_SemaphoreIncOp : TTL_Op<"semaphore_inc"> {
  let summary = "Increment semaphore value (remote unicast only)";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I32Attr:$value
  );
  let description = [{
    Python API `remote_sem.inc(n)` maps to this operation.

    Atomically increments a remote semaphore by the specified value.
    Only supported for unicast remote semaphores (obtained via
    ttl.get_remote_semaphore), not multicast semaphores.

    Example:
      Python: remote_sem = my_sem.get_remote((0, 0))
              remote_sem.inc(1)
      TTL IR: %remote = ttl.get_remote_semaphore %sem, core=[0,0]
              ttl.semaphore_inc %remote, 1

    Maps to TTKernel operation:
      ttkernel.noc_semaphore_inc(target_core, sem_addr, increment_value)

    Note: Multicast semaphores support set but not increment (hardware limitation).
  }];
}
```

## 4.9 MLIR Interface Requirements

To enable effective compiler analysis and optimization, TTL operations must
implement appropriate MLIR interfaces. This section specifies which operations
should implement which interfaces and why.

### 4.9.1 MemoryEffectsOpInterface

All operations that read from or write to memory must implement
`MemoryEffectsOpInterface`. This enables standard MLIR optimizations such as
common subexpression elimination (CSE), loop-invariant code motion (LICM), and
dead code elimination (DCE).

**Operations requiring MemoryEffectsOpInterface:**

```cpp
// Circular Buffer Operations
TTL_CBWaitOp        // Reads: CB state (waits for available data)
TTL_CBPopOp         // Writes: CB state (updates read pointer)
TTL_CBReserveOp     // Reads: CB state (waits for free space)
TTL_CBPushOp        // Writes: CB state (updates write pointer)
TTL_GetTileOp       // Reads: CB memory
TTL_PackTileOp      // Writes: CB memory

// Data Movement Operations
TTL_CopyOp          // Reads source, Writes destination
TTL_WaitOp          // Memory fence (all pending DMAs)
TTL_DMABarrierOp    // Memory fence (global DMA barrier)

// Synchronization Operations
TTL_SemaphoreWaitEqOp  // Reads: semaphore value (wait for equality)
TTL_SemaphoreWaitMinOp // Reads: semaphore value (wait for minimum)
TTL_SemaphoreSetOp     // Writes: semaphore value
TTL_SemaphoreIncOp     // Reads+Writes: semaphore value (atomic)

// Compute Operations (with side effects)
TTL_ComputeRegionOp // Contains operations with memory effects

// DST Register Management
TTL_RequireDSTOp    // Reads+Writes: DST register file
TTL_SpillTileToL1Op // Reads: DST, Writes: L1
TTL_RestoreTileFromL1Op // Reads: L1, Writes: DST
```

### 4.9.2 Custom TTL Interfaces

**TTLSynchronizationInterface:**

Operations that provide synchronization semantics should implement a custom
interface enabling scheduling passes to understand barrier properties.

```tablegen
// Define barrier type enum (in TTLOpsEnums.td)
def TTL_BarrierType : I32EnumAttr<"BarrierType", "TTL barrier type", [
  I32EnumAttrCase<"DMARead", 0>,
  I32EnumAttrCase<"DMAWrite", 1>,
  I32EnumAttrCase<"CB", 2>,
  I32EnumAttrCase<"Semaphore", 3>
]> {
  let cppNamespace = "::mlir::tt::ttl";
}

// Define synchronization interface (in TTLInterfaces.td)
def TTLSynchronizationInterface : OpInterface<"TTLSynchronization"> {
  let description = [{
    Interface for operations that provide synchronization barriers.
    Enables scheduling passes to understand ordering constraints.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the barrier type",
      "::mlir::tt::ttl::BarrierType", "getBarrierType"
    >,
    InterfaceMethod<
      "Returns true if this is a global barrier affecting all operations",
      "bool", "isGlobalBarrier"
    >,
  ];
}
```

**Operations implementing TTLSynchronizationInterface:**
- `TTL_WaitOp`: DMA barrier (read or write based on transfer direction)
- `TTL_DMABarrierOp`: Global DMA barrier
- `TTL_SemaphoreWaitOp`: Semaphore barrier

### 4.9.3 Bufferization Interfaces

TTL tensor types should implement `bufferization::TensorLikeType` to integrate
with MLIR's One-Shot Bufferization framework.

```cpp
// In TTLTypes.cpp
struct TTLBlockType : public bufferization::TensorLikeType {
  FailureOr<Type> getBufferType(const BufferizationOptions &options) const override {
    // Return memref with appropriate memory space attribute
    return MemRefType::get(getShape(), getElementType(),
                          MemorySpaceAttr::get(getMemorySpace()));
  }
};
```

TTL operations on tensors should implement `BufferizableOpInterface`:

```cpp
// In TTLOps.cpp
struct TTLBlockAddOp : public BufferizableOpInterface {
  FailureOr<Operation*> bufferize(RewriterBase &rewriter,
                                   const BufferizationOptions &options) override {
    // Convert tensor operands to memrefs
    // Replace with memref-based operation
  }
};
```

**Benefits:**
- Eliminates custom bufferization code
- Integrates with MLIR deallocation infrastructure (automatic memory management)
- Leverages upstream optimizations

### 4.9.4 Liveness Analysis Integration

MLIR provides built-in liveness analysis via `mlir::Liveness` utility. TTL
operations must properly implement `MemoryEffectsOpInterface` and
`RegionBranchOpInterface` (for operations with regions) to enable the built-in
liveness analysis.

**Requirements for built-in liveness:**

1. **MemoryEffectsOpInterface**: All operations that access memory (CB, DMA,
   DST) must declare their memory effects correctly. This allows liveness
   analysis to track data dependencies.

2. **RegionBranchOpInterface**: Operations with regions (ttl.compute_region,
   ttl.if_pipe_src, ttl.if_pipe_dst) must implement this interface to enable
   liveness analysis across region boundaries.

3. **SSA Form**: TTL operations produce SSA values (block references, transfer
   handles, DST registers) that can be tracked by standard MLIR liveness
   analysis.

**Usage in allocation passes:**

```cpp
// In TTLAllocateCircularBuffers.cpp
void runOnOperation() override {
  auto func = getOperation();
  mlir::Liveness liveness(func);  // Use built-in liveness analysis

  // Query live ranges for values
  for (auto &block : func) {
    auto liveIn = liveness.getLiveIn(&block);
    auto liveOut = liveness.getLiveOut(&block);
    // Use liveness info for allocation decisions
  }
}
```

**Benefits:**
- No custom liveness pass implementation needed
- Proven correct analysis from MLIR upstream
- Automatically handles complex control flow
- Works across region boundaries

### 4.9.5 Implementation Priority

| Priority | Interface | Operations | Benefit |
|----------|-----------|------------|---------|
| **HIGH** | `MemoryEffectsOpInterface` | All CB, DMA, sync ops | Enables CSE, LICM, DCE, liveness |
| **HIGH** | `RegionBranchOpInterface` | compute_region, if_pipe_* | Enables liveness across regions |
| **MEDIUM** | `TTLSynchronizationInterface` | Barrier operations | Enables scheduling analysis |
| **MEDIUM** | `bufferization::TensorLikeType` | Block tensor types | Reuses One-Shot Bufferization |
| **LOW** | Additional interfaces | As needed | Future extensibility |
