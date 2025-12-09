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

### 4.5 Data Movement Operations

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

    Returns transfer handle (SSA value of type !ttl.xf) for ordering.
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
    - TTLValidatePass enforces this requirement and rejects unguarded or unmatched pipes

    Note: ttl.wait lowers to global DMA barrier, not per-transfer wait
    (TTKernel limitation). TTKernel only provides `ttkernel.noc_async_read_barrier`
    and `ttkernel.noc_async_write_barrier` operations which wait for all pending
    DMA operations of the respective type, not individual transfers. When a copy
    involves a Pipe operand, ttl.wait instead lowers to a semaphore wait.
    See: `tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td` definitions
    `TTKernel_NocAsyncReadBarrierOp` and `TTKernel_NocAsyncWriteBarrierOp`.
  }];
}

def TTL_IfPipeSrcOp : TTL_Op<"if_pipe_src"> {
  let summary = "Execute region for each pipe where current core is source";
  let arguments = (ins Variadic<TTL_Pipe>:$pipes);
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Python API `ttl.if_pipe_src(pipes, pipe_src)` maps to this operation.

    Semantics: For each pipe where current core matches src_core, logically invoke
    the region body with that pipe as block argument. Invocations for different pipes
    may execute in parallel (independent pipe transfers).

    Example:
      Python (TT-Lang spec):
        def pipe_src(pipe):
            xf = ttl.copy(blk, pipe)
            xf.wait()
        ttl.if_pipe_src([pipe1, pipe2, pipe3], pipe_src)

      TTL IR:
        ttl.if_pipe_src %pipe1, %pipe2, %pipe3 {
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
        // Core (0,0): matches pipe1 and pipe3 src_core
        scf.forall (%pipe_idx) in (0, 2) {  // Parallel over pipe1, pipe3
          %pipe = select %pipe_idx : pipe1 or pipe3
          ttkernel.noc_async_write... using %pipe coordinates
        }

    The region block argument receives each matching pipe, enabling pipe-specific operations.
  }];
}

def TTL_IfPipeDstOp : TTL_Op<"if_pipe_dst"> {
  let summary = "Execute region for each pipe where current core is destination";
  let arguments = (ins Variadic<TTL_Pipe>:$pipes);
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Python API `ttl.if_pipe_dst(pipes, pipe_dst)` maps to this operation.

    Semantics: For each pipe where current core matches the pipe's dst_core (unicast)
    or is within dst_core_range (multicast), logically invoke the region body with
    that pipe as block argument. Invocations for different pipes may execute in parallel.

    Example:
      Python (TT-Lang spec):
        def pipe_dst(pipe):
            xf = ttl.copy(pipe, blk)
            xf.wait()
        ttl.if_pipe_dst([pipe1, pipe2], pipe_dst)

      TTL IR:
        ttl.if_pipe_dst %pipe1, %pipe2 {
          ^bb0(%pipe: !ttl.pipe):
            %xf = ttl.copy %pipe, %blk
            ttl.wait %xf
        }

    Lowering to TTKernel:
      The pass determines which pipes match current core (checking dst_core or dst_core_range).
      Execution model determined by dependency analysis (parallel vs serial).

    The region block argument receives each matching pipe, enabling pipe-specific operations.
  }];
}

def TTL_WaitOp : TTL_Op<"wait"> {
  let summary = "Wait for DMA transfer (lowers to global barrier)";
  let arguments = (ins TTL_TransferHandle:$xf);
  let description = [{
    Explicit wait on transfer handle. Lowers to `ttkernel.noc_async_read_barrier`
    or `ttkernel.noc_async_write_barrier` (global barriers, not per-transfer).
    For pipe transfers, the wait resolves to a semaphore synchronization that
    coordinates pipe producers and consumers instead of a DMA barrier.
    See: `tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td` definitions
    `TTKernel_NocAsyncReadBarrierOp` and `TTKernel_NocAsyncWriteBarrierOp`.
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
def TTL_SemaphoreWaitOp : TTL_Op<"semaphore_wait"> {
  let summary = "Wait for semaphore value condition";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I32Attr:$value,
    OptionalAttr<I32Attr>:$reset_value,  // Optional: set to this value after wait completes
    OptionalAttr<StrAttr>:$comparison    // "equal" (default) or "min" (>=)
  );
  let description = [{
    Blocking operation that waits until semaphore meets the specified condition.
    Comparison modes: "equal" (wait for exact value) or "min" (wait for value >= target).

    Optional reset_value enables atomic wait-and-reset pattern for producer/consumer
    coordination: wait for condition, then immediately reset semaphore to specified value.

    Example producer/consumer barrier pattern:
      // Consumer waits for N items, then resets to 0
      ttl.semaphore_wait %sem, 5 {reset_value = 0, comparison = "equal"}

    Maps to TTKernel operations:
      Without reset:
        ttkernel.noc_semaphore_wait(sem_addr, value)  // or wait_min

      With reset (sequence):
        ttkernel.noc_semaphore_wait(sem_addr, value)
        ttkernel.noc_semaphore_set(local_core, sem_addr, reset_value)

    Note: Reset is a software pattern (wait + local set), not atomic hardware operation.
  }];
}

def TTL_SemaphoreSetOp : TTL_Op<"semaphore_set"> {
  let summary = "Set semaphore value (local or remote)";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I32Attr:$value,
    OptionalAttr<I64ArrayAttr>:$core,      // For remote unicast
    OptionalAttr<ArrayAttr>:$mcast_range   // For remote multicast [[x0,y0],[x1,y1]]
  );
  let description = [{
    Python API `semaphore.set(value)` and `remote_semaphore.set(value)` map
    to this operation. For remote operations, use `ttl.get_remote_semaphore` or
    `ttl.get_remote_multicast_semaphore` to create annotated semaphore references
    first. The core/mcast_range attributes come from the remote semaphore reference.

    Maps to TTKernel operations:
      Local:
        ttkernel.noc_semaphore_set(local_core, sem_addr, value)

      Remote unicast:
        ttkernel.noc_semaphore_set(target_core, sem_addr, value)

      Remote multicast:
        ttkernel.noc_semaphore_set_multicast(noc_multicast_addr, value)

    Note: Multicast semaphores support set but not increment (hardware limitation).
  }];
}

def TTL_SemaphoreIncOp : TTL_Op<"semaphore_inc"> {
  let summary = "Increment semaphore value (remote unicast only)";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I32Attr:$value,
    I64ArrayAttr:$core                     // Target core
  );
  let description = [{
    Python API `unicast_remote_semaphore.inc(value)` maps to this operation.
    Only supported for unicast remote semaphores (not multicast).

    Maps to TTKernel operation:
    - ttkernel.noc_semaphore_inc(target_core, sem_addr, increment_value)
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
TTL_SemaphoreWaitOp // Reads: semaphore value
TTL_SemaphoreSetOp  // Writes: semaphore value
TTL_SemaphoreIncOp  // Reads+Writes: semaphore value (atomic)

// Compute Operations (with side effects)
TTL_ComputeRegionOp // Contains operations with memory effects
TTL_BlockStoreOp    // Writes: block memory

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



