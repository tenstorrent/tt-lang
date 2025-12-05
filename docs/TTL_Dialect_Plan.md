# TTL Dialect Design Plan

**Version**: 1.0
**Date**: 2025-01-XX
**Status**: Design Phase

This document specifies the TTL (TT-Lang) MLIR dialect, a tensor-level intermediate representation designed to directly capture the semantics of the TT-lang DSL. The TTL dialect enables multi-stage compilation with explicit transformation passes for synchronization inference, resource allocation, and hardware-specific optimizations before lowering to executable kernels.

---

## Table of Contents

1. [Motivation & Goals](#1-motivation--goals)
2. [Architecture Overview](#2-architecture-overview)
3. [Type System](#3-type-system)
4. [Operations](#4-operations)
   - 4.1 [Structural Operations](#41-structural-operations)
   - 4.2 [Resource Creation](#42-resource-creation)
   - 4.3 [Circular Buffer Operations](#43-circular-buffer-operations)
   - 4.4 [Compute Operations](#44-compute-operations)
   - 4.5 [Data Movement Operations](#45-data-movement-operations)
   - 4.6 [Synchronization Operations](#46-synchronization-operations)
   - 4.7 [Utility Operations](#47-utility-operations)
5. [Compilation Pipeline](#5-compilation-pipeline)
6. [Type Conversion & Lowering Examples](#6-type-conversion--lowering-examples)
7. [Python Integration](#7-python-integration)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Future Evolution](#9-future-evolution)
10. [Risk Mitigation](#10-risk-mitigation)
11. [Success Criteria](#11-success-criteria)
12. [Appendix: Design Rationale](#12-appendix-design-rationale)
13. [References](#13-references)

---

## Executive Summary

The TTL dialect provides a tensor-level intermediate representation for TT-lang programs (defined in `docs/TT-lang.md`), enabling multi-stage lowering with explicit compiler transformations before generating C++ kernels. TTL replaces the current direct `d2m.generic` approach, allowing the compiler to insert synchronization, allocate resources, and optimize before committing to hardware-specific details.

**Key Design Decisions:**
- **Threading Model**: Simple compute/datamovement threads (MVP), with microcore evolution path documented
- **Abstraction Level**: Tensor-level (blocks of tiles), not individual tiles
- **Type System**: Memory spaces explicit in types; SSA values for all resources (CBs, pipes, semaphores)
- **Control Flow**: Reuse `scf` dialect (standard MLIR) with TTL attributes
- **Lowering Path**: TTL → TTKernel → EmitC → C++ (bypass d2m.generic)
- **Phasing**: Multi-threaded from start (matches current examples)

---

## 1. Motivation & Goals

### Current State Issues
- **No abstraction gap**: Python DSL maps directly to `d2m.generic`, preventing compiler analysis
- **Hardcoded decisions**: Synchronization, memory allocation, and register assignment happen too early
- **Limited optimization**: Cannot reason about inter-thread communication or reorder operations
- **Single backend**: Tied to `ttir-to-ttmetal` pipeline

### TTL Dialect Goals
1. **Capture DSL semantics** in SSA form: kernels, threads, circular buffers, pipes, blocks, semaphores
2. **Enable analysis passes**: Synchronization inference, memory planning, DST register allocation
3. **Support transformations**: Liveness analysis, operation reordering, pipelining
4. **Multiple backends**: TTKernel (immediate) and standalone C++ (future)
5. **Future-proof**: Extensible to new hardware generations via attributes

### Non-Goals (MVP)
- Autotuning algorithms (IR has hooks, algorithms come later)
- Single-threaded synchronous model (start multi-threaded)
- Complete TT-lang spec (start minimal, expand incrementally)

---

## 2. Architecture Overview

### Current Flow
```
Python Kernel → Python AST → D2M Generic Ops → ttir-to-ttmetal Pipeline → TTMetal Ops → Flatbuffer Binary
                                                     ↓
                                          (Inside pipeline: fusion,
                                           bufferization, allocation,
                                           DST register assignment,
                                           DMA lowering in D2M passes)
```

### TTL Flow
```
Python Kernel → Python AST → TTL Dialect → TTL Passes → TTKernel → TTMetal Ops → Flatbuffer Binary
                                    ↓
                         Validation, Synchronization,
                         Bufferization, Allocation,
                         Register Assignment, Optimization
```

**Key difference**: TTL operations directly represent DSL concepts. Multiple transformation passes can analyze and optimize before lowering to hardware primitives.

---

## 3. Type System

### 3.1 Core Types

```tablegen
// TTL Types (TTLTypes.td)

def TTL_CircularBuffer : TTL_Type<"CircularBuffer", "cb"> {
  let summary = "Circular buffer for producer-consumer communication";
  let parameters = (ins
    "ArrayRef<int64_t>":$shape,          // [2, 1] tiles per block
    "mlir::Type":$tileType,              // !ttcore.tile<32x32, f32>
    "int64_t":$bufferFactor,             // Number of blocks/slots
    "TTL::MemorySpace":$memorySpace      // L1, DRAM, DST
  );
  let assemblyFormat = "`<` $shape `,` $tileType `,` $bufferFactor `,` $memorySpace `>`";

  let extraClassDeclaration = [{
    // Calculate total tiles for TTKernel CB conversion
    int64_t getTotalTiles() const {
      int64_t tilesPerBlock = std::accumulate(
        getShape().begin(), getShape().end(), 1, std::multiplies<int64_t>());
      return tilesPerBlock * getBufferFactor();
    }

    // Helper for lowering to !ttkernel.cb<num_tiles, tile_type>
    int64_t getTilesPerBlock() const {
      return std::accumulate(
        getShape().begin(), getShape().end(), 1, std::multiplies<int64_t>());
    }
  }];
}

def TTL_Block : TTL_Type<"Block", "block"> {
  let summary = "Logical unit of data exchanged via circular buffers";
  let parameters = (ins "TensorType":$tensorType);
  let description = [{
    Represents a block of tiles consumed/produced by compute operations.
    Carries optional DST register hint for liveness analysis.
  }];
}

def TTL_MemoryTransaction : TTL_Type<"MemoryTransaction", "mem_tx"> {
  let summary = "Handle for DMA operation ordering (lowers to global barrier)";
  let description = [{
    Note: TTKernel doesn't support per-transaction waits. All ttl.wait
    operations lower to global DMA barriers. This type exists for ordering
    and future optimization opportunities.
  }];
}

def TTL_Semaphore : TTL_Type<"Semaphore", "semaphore"> {
  let summary = "Synchronization primitive for inter-core coordination";
  let parameters = (ins OptionalParameter<"bool">:$isRemote);
}

def TTL_Pipe : TTL_Type<"Pipe", "pipe"> {
  let summary = "Inter-core communication channel (first-class SSA value)";
  let parameters = (ins
    "ArrayRef<int64_t>":$srcCore,        // Source core coordinates
    "ArrayRef<int64_t>":$dstCore,        // Destination (unicast)
    OptionalParameter<"ArrayRef<ArrayRef<int64_t>>">:$dstCoreRange  // Multicast range
  );
}

def TTL_TensorAccessor : TTL_Type<"TensorAccessor", "accessor"> {
  let summary = "Handle for indexed tensor access";
  let description = [{
    Represents a tensor with indexing capability for slicing operations.
    Carries layout metadata from python/ttlang/layouts.py (tiled, sharded, interleaved).
    Used in DMA operations to specify source/destination with coordinates.
  }];
  let parameters = (ins
    "TensorType":$tensorType,
    "TTL::LayoutAttr":$layout,           // Reuses create_metal_layout metadata
    "TTL::MemorySpace":$memorySpace
  );
}
```

### 3.2 Attributes

```tablegen
def TTL_MemorySpaceAttr : I32EnumAttr<"MemorySpace", "TTL memory space", [
  I32EnumAttrCase<"L1", 0>,
  I32EnumAttrCase<"DRAM", 1>,
  I32EnumAttrCase<"DST", 2>,
  I32EnumAttrCase<"System", 3>
]> {
  let cppNamespace = "::mlir::tt::ttl";
}

def TTL_GridAttr : AttrDef<TTL_Dialect, "Grid"> {
  let summary = "Grid topology description";
  let parameters = (ins "ArrayRef<int64_t>":$dimensions);
  let assemblyFormat = "`<` $dimensions `>`";
}

def TTL_LayoutAttr : AttrDef<TTL_Dialect, "Layout"> {
  let summary = "Tensor layout metadata (from python layouts.py)";
  let description = [{
    Captures layout information generated by create_metal_layout():
    - tiled vs row-major
    - sharded grids
    - interleaved patterns
    Allows lossless representation of all TensorAccessor configurations.
  }];
  // Parameters TBD based on MetalLayoutConfig from layouts.py
}

def TTL_CoreMaskAttr : AttrDef<TTL_Dialect, "CoreMask"> {
  let summary = "Bitmask of participating cores";
  let parameters = (ins "ArrayRef<int64_t>":$mask);
}
```

---

## 4. Operations

### 4.1 Structural Operations

```tablegen
def TTL_ProgramOp : TTL_Op<"program", [IsolatedFromAbove]> {
  let summary = "Top-level kernel program with captured tensors";
  let arguments = (ins
    TTL_GridAttr:$grid,
    StrAttr:$memory_space,
    BoolAttr:$tiled
  );
  let regions = (region VariadicRegion<SizedRegion<1>>:$body);
  let description = [{
    Container for kernel execution. Owns captured tensors and nested thread regions.
    Regions execute in parallel on grid of cores.
  }];
}

def TTL_KernelOp : TTL_Op<"kernel", [IsolatedFromAbove]> {
  let summary = "Kernel with multiple threads on grid";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    TTL_GridAttr:$grid,
    StrAttr:$memory_space,
    BoolAttr:$tiled
  );
  let regions = (region VariadicRegion<SizedRegion<1>>:$threads);
  let results = (outs Variadic<AnyType>:$results);
}

def TTL_ComputeThreadOp : TTL_Op<"compute_thread"> {
  let summary = "Compute thread executing on Tensix core";
  let arguments = (ins
    Variadic<AnyType>:$operands,
    OptionalAttr<StrAttr>:$microcore_hint  // Future evolution path
  );
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Thread for mathematical operations on tiles. Executes on MATH microcore.
    Can use: block arithmetic, CB wait/pop/reserve/push, tile operations.
    Cannot use: DMA operations (use datamovement_thread).

    Future: microcore_hint enables explicit microcore targeting for optimization.
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
    I64ArrayAttr:$shape,              // Tiles per block [2, 1]
    TypeAttr:$tile_type,              // !ttcore.tile<32x32, f32>
    I64Attr:$buffer_factor,           // Number of blocks
    TTL_MemorySpaceAttr:$memory_space // L1, DRAM, DST
  );
  let results = (outs TTL_CircularBuffer:$result);
}

def TTL_CreatePipeOp : TTL_Op<"create_pipe"> {
  let summary = "Create inter-core pipe for unicast or multicast";
  let arguments = (ins
    I64ArrayAttr:$src_core,
    OptionalAttr<I64ArrayAttr>:$dst_core,         // For unicast
    OptionalAttr<ArrayAttr>:$dst_core_range       // For multicast [[x0,y0],[x1,y1]]
  );
  let results = (outs TTL_Pipe:$pipe);
}

def TTL_CreateSemaphoreOp : TTL_Op<"create_semaphore"> {
  let summary = "Create semaphore with initial value";
  let arguments = (ins
    I32Attr:$initial_value,
    OptionalAttr<BoolAttr>:$is_remote
  );
  let results = (outs TTL_Semaphore:$semaphore);
}

def TTL_TensorAccessorOp : TTL_Op<"tensor_accessor"> {
  let summary = "Create accessor for indexed tensor access";
  let arguments = (ins
    AnyTensor:$tensor,
    TTL_LayoutAttr:$layout,           // From create_metal_layout()
    TTL_MemorySpaceAttr:$memory_space
  );
  let results = (outs TTL_TensorAccessor:$accessor);
}
```

### 4.3 Circular Buffer Operations

```tablegen
def TTL_CBWaitOp : TTL_Op<"cb_wait"> {
  let summary = "Consumer acquire: wait for data in circular buffer";
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I64Attr:$num_tiles                // Number of tiles to wait for
  );
  // Blocking operation - no result, tiles accessed via get_tile
}

def TTL_CBPopOp : TTL_Op<"cb_pop"> {
  let summary = "Consumer release: signal tiles consumed";
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I64Attr:$num_tiles                // Number of tiles to release
  );
  // Non-blocking operation
}

def TTL_CBReserveOp : TTL_Op<"cb_reserve"> {
  let summary = "Producer acquire: reserve space in circular buffer";
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I64Attr:$num_tiles                // Number of tiles to reserve
  );
  // Blocking operation - no result, tiles accessed via pack_tile
}

def TTL_CBPushOp : TTL_Op<"cb_push"> {
  let summary = "Producer release: signal data ready";
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I64Attr:$num_tiles                // Number of tiles to push
  );
  // Non-blocking operation
}

def TTL_GetTileOp : TTL_Op<"get_tile"> {
  let summary = "Extract tile from circular buffer for compute";
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    Index:$tile_idx                   // Tile index within CB
  );
  let results = (outs AnyType:$tile);  // !ttcore.tile<32x32, f32>
}

def TTL_PackTileOp : TTL_Op<"pack_tile"> {
  let summary = "Pack computed tile into circular buffer";
  let arguments = (ins
    Index:$dst_idx,                   // DST register index
    TTL_CircularBuffer:$cb,
    Index:$tile_idx                   // Tile index within CB
  );
}
```

### 4.4 Compute Operations

```tablegen
def TTL_BlockComputeOp : TTL_Op<"block_compute"> {
  let summary = "Compute region operating on blocks";
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Region containing only ttl.math.* operations and structural ops.
    Dedicated lowering pass rewrites to arith/math/tosa or TT intrinsics.
  }];
}

def TTL_BlockAddOp : TTL_Op<"block_add"> {
  let summary = "Element-wise addition on tensor blocks";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

def TTL_BlockMatmulOp : TTL_Op<"block_matmul"> {
  let summary = "Matrix multiplication on tensor blocks";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

def TTL_BlockStoreOp : TTL_Op<"block_store"> {
  let summary = "Store computation result to circular buffer block";
  let arguments = (ins AnyTensor:$dst, AnyTensor:$src);
  let description = [{
    Blocking operation that materializes result and stores in block.
    Lowers to per-tile pack operations with DST register management.
  }];
}

def TTL_RequireDSTOp : TTL_Op<"require_dst", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Marker that SSA value must reside in DST register file";
  let arguments = (ins AnyType:$value);
  let description = [{
    Hint for liveness analysis to track which values need DST registers.
    Enables optimization passes to reorder operations and minimize register pressure.
  }];
}

// Additional ops: sub, mul, div, relu, sigmoid, etc.
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
  let results = (outs TTL_MemoryTransaction:$tx);
  let description = [{
    Unified DMA operation handling all transfer combinations:
    - Tensor slice → CB (read from DRAM/L1)
    - CB → Tensor slice (write to DRAM/L1)
    - CB → Pipe → CB (inter-core transfer)

    Operand types and attributes determine lowering:
    - Unicast/multicast from pipe parameters
    - Memory space from accessor/CB types
    - NOC address calculation from coordinates

    Returns transaction handle for ordering. Note: ttl.wait lowers to
    global DMA barrier, not per-transaction wait (TTKernel limitation).
  }];
}

def TTL_WaitOp : TTL_Op<"wait"> {
  let summary = "Wait for DMA transfer (lowers to global barrier)";
  let arguments = (ins TTL_MemoryTransaction:$tx);
  let description = [{
    Explicit wait on transaction handle. Lowers to ttkernel.noc_async_*_barrier.
  }];
}

def TTL_DMABarrierOp : TTL_Op<"dma_barrier"> {
  let summary = "Global DMA barrier - wait for all pending DMAs";
  let description = [{
    Ensures all prior DMA operations complete. More efficient than
    individual waits when multiple transactions are pending.
  }];
}
```

### 4.6 Synchronization Operations

```tablegen
def TTL_SemaphoreWaitOp : TTL_Op<"semaphore_wait"> {
  let summary = "Wait for semaphore value condition";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I32Attr:$value,
    OptionalAttr<I32Attr>:$reset_value,
    OptionalAttr<StrAttr>:$comparison  // "equal" or "min"
  );
}

def TTL_SemaphoreSetOp : TTL_Op<"semaphore_set"> {
  let summary = "Set semaphore value (local or remote)";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I32Attr:$value,
    OptionalAttr<I64ArrayAttr>:$core,      // For remote unicast
    OptionalAttr<I64ArrayAttr>:$mcast      // For remote multicast
  );
}

def TTL_SemaphoreIncOp : TTL_Op<"semaphore_inc"> {
  let summary = "Increment semaphore value (remote unicast only)";
  let arguments = (ins
    TTL_Semaphore:$semaphore,
    I32Attr:$value,
    I64ArrayAttr:$core                     // Target core
  );
}
```

### 4.7 Utility Operations

```tablegen
def TTL_CoreIndexOp : TTL_Op<"core_index", [Pure]> {
  let summary = "Get current core coordinates";
  let arguments = (ins OptionalAttr<I64Attr>:$dims);
  let results = (outs Variadic<Index>:$coordinates);
  let description = [{
    Returns core coordinates in grid. Folds to constants enabling
    later canonicalization and dead code elimination.
  }];
}

def TTL_GridSizeOp : TTL_Op<"grid_size", [Pure]> {
  let summary = "Get grid dimensions";
  let arguments = (ins OptionalAttr<I64Attr>:$dims);
  let results = (outs Variadic<Index>:$sizes);
}
```

---

## 5. Compilation Pipeline

### 5.1 Pass Architecture

```
Python Kernel (@compute/@datamovement decorators)
  ↓
Python AST Parsing (TTLDialectCompiler)
  ↓
ttl.kernel (with thread regions, tensor operands)
  ↓
[Phase 1: Validation & Canonicalization]
  ├─ TTLValidatePass - Verify CB/pipe/semaphore contracts
  ├─ TTLCanonicalizePass - Fold constants, simplify patterns
  └─ TTLVerifyLayoutsPass - Check tensor accessor layouts
  ↓
[Phase 2: Analysis & Inference]
  ├─ TTLInsertSynchronization - Analyze inter-thread dataflow, insert barriers
  ├─ TTLInferDSTRequirements - Mark values needing DST registers
  └─ TTLBufferizePass - Tensor → memref conversion (One-Shot Bufferization)
  ↓
ttl.kernel (with thread regions, memref operands)
  ↓
[Phase 3: Memory Planning]
  └─ TTLAllocateCircularBuffers - Assign L1 addresses (liveness-based)
  ↓
[Phase 4: Thread Expansion]
  └─ TTLExpandThreads - Convert ttl.kernel regions → separate func.func
  ↓
func.func @compute_thread_0 (memref args)
func.func @dm_thread_0 (memref args)
  ↓
[Phase 5: Resource Assignment]
  └─ TTLAssignDSTRegisters - Allocate 16 DST registers for compute threads
  ↓
[Phase 6: Lowering to TTKernel]
  ├─ TTLLowerCompute - ttl.block_add → scf.for + ttkernel.add_tiles
  ├─ TTLLowerDataMovement - ttl.copy → ttkernel.noc_async_*
  └─ TTLLowerSynchronization - ttl.cb_* → ttkernel.cb_*
  ↓
ttkernel.* operations
  ↓
[Existing: TTKernel → TTMetal Pipeline]
  ├─ ConvertTTKernelToEmitC (generates C++ kernel source)
  └─ TTMetal dialect operations
  ↓
[Flatbuffer Serialization]
  └─ ttmetal_to_flatbuffer_bin
  ↓
Flatbuffer Binary (executable on hardware)
```

### 5.2 Key Pass Descriptions

**TTLInsertSynchronization**
- **Input**: `ttl.kernel` with thread regions (tensor operands)
- **Analysis**: Build producer-consumer DAG for blocks, semaphores, pipes
- **Transform**: Insert `ttl.dma_barrier` where needed, validate CB usage patterns
- **Output**: `ttl.kernel` with explicit synchronization

**TTLInferDSTRequirements**
- **Input**: `ttl.kernel` with compute operations
- **Analysis**: Track which SSA values participate in compute chains
- **Transform**: Insert `ttl.require_dst` markers for liveness analysis
- **Output**: `ttl.kernel` with DST hints

**TTLBufferizePass**
- **Input**: `ttl.kernel` with tensor operands
- **Analysis**: Determine bufferization strategy for each tensor
- **Transform**: Convert tensor types to memref types (One-Shot Bufferization)
- **Output**: `ttl.kernel` with memref operands
- **Note**: Critical transition - tensor semantics → buffer semantics

**TTLAllocateCircularBuffers**
- **Input**: `ttl.kernel` with memref CBs
- **Analysis**: Compute liveness ranges for each CB
- **Transform**: Assign L1 addresses using first-fit allocation
- **Output**: CBs annotated with address attributes

**TTLExpandThreads**
- **Input**: `ttl.kernel` with regions
- **Transform**: Extract each thread region into `func.func` with metadata
- **Output**: Separate functions per thread, preserving grid/memory_space attrs

**TTLLowerCompute**
- **Input**: `ttl.block_add` operations
- **Transform**: Generate `scf.for` iterating over tiles, insert `ttkernel.add_tiles_init` and `ttkernel.add_tiles`
- **Output**: TTKernel operations with explicit tile iteration

**TTLLowerDataMovement**
- **Input**: `ttl.copy` operations
- **Transform**:
  - Tensor → CB: `ttkernel.noc_async_read`
  - CB → Tensor: `ttkernel.noc_async_write`
  - CB → Pipe → CB: `ttkernel.noc_async_write_multicast*` or unicast
- **Output**: TTKernel NOC operations with computed addresses

### 5.3 Granularity and Block Shapes

**Concept**: Granularity defines how many tiles are grouped into blocks for transfer and processing.

**From TT-lang.md example:**
```python
g = 2  # granularity - process 2 tiles at a time

for rt in range(row_tiles // g):
    # Transfer g tiles as one block
    a_xf = ttl.copy(
        a[(rt * g):((rt + 1) * g), ct:(ct + 1)],  # g rows, 1 col
        a_blk)
```

**From kostas/spec simulator:**
```python
@ttl.kernel(granularity=2)  # Kernel-level parameter
def eltwise_add(a_in, b_in, out):
    # Use granularity to create CBs
    a_in_cb = ttl.make_circular_buffer_like(
        a_in, shape=(granularity, 1), buffer_factor=2
    )

    @ttl.datamovement()
    def dm():
        for rt_block in range(row_tiles // granularity):
            # Transfer granularity tiles as one block
            row_slice = slice(rt_block * granularity, (rt_block + 1) * granularity)
            a_block = a_in_cb.reserve()
            tx = ttl.copy(a_accessor[row_slice, :], a_block)
```

**Mapping to TTL:**

Granularity is encoded in CB `shape` parameter:

```python
# Python DSL (kostas/spec)
@ttl.kernel(granularity=2)
def kernel(...):
    cb = ttl.make_circular_buffer_like(tensor, shape=(granularity, 1), buffer_factor=2)
```

```mlir
// TTL IR
%cb = ttl.create_cb shape=[2, 1], tile_type=!ttcore.tile<32x32,f32>, buffer_factor=2
      : !ttl.cb<[2,1], !ttcore.tile<32x32,f32>, 2, L1>
```

**Semantic meaning:**
- `shape=[2, 1]` means each block contains 2×1 = 2 tiles
- CB operations (`wait`, `reserve`) always work on `shape[0] * shape[1]` tiles
- All CB operations from same CB use same tile count (determined by shape)
- Loop iterations reduced by granularity factor (`row_tiles // granularity`)

**Design implications:**

1. **CB shape determines operation granularity**:
   - `shape=[2, 1]` → process 2 tiles at a time
   - `shape=[1, 1]` → process 1 tile at a time (fine-grained)
   - `shape=[4, 2]` → process 8 tiles at a time (coarse-grained)

2. **No separate granularity parameter in TTL types**:
   - Granularity is implicit in CB shape
   - More explicit and type-safe
   - Aligns with TTKernel's requirement for exact tile counts

3. **Python API**:
   ```python
   # Option A: Explicit shape (current)
   cb = CircularBuffer(shape=(2, 1), buffer_factor=2)

   # Option B: Kernel-level granularity parameter (future)
   @ttl.kernel(granularity=2)  # Sets default for all CBs
   def kernel(...):
       cb = CircularBuffer(buffer_factor=2)  # Uses kernel granularity
   ```

4. **Validation**:
   - Compiler verifies all CBs in a producer-consumer chain have same shape
   - DMA transfers match CB block dimensions
   - Loop bounds are multiples of granularity

### 5.4 Source Location Tracking

**Goal**: Maintain Python source locations throughout compilation for debugging and IDE integration.

**Requirements:**
1. **Error diagnostics** point to original Python source
2. **IDE integration** enables jump-to-definition from MLIR to Python
3. **Debugging** maps compiled code back to source lines
4. **Profiling** attributes performance to Python code locations

**Implementation Strategy:**

**Phase 1: Python AST → TTL**

Capture source location during AST parsing:

```python
# TTLDialectCompiler in python/ttlang/_src/ttl_ast.py
class TTLDialectCompiler(TTCompilerBase):
    def visit_Call(self, node):
        # Get Python source location from AST node
        filename = self.source_file
        line = node.lineno
        col = node.col_offset

        # Create MLIR location
        loc = Location.file(filename, line, col, context=self.ctx)

        # Attach to generated ops
        with loc:
            result = self.emit_ttl_operation(node)
        return result
```

**Phase 2: Location Propagation Through Passes**

TTL passes must preserve and update locations:

```cpp
// Example from TTLLowerCompute.cpp
Value lowerBlockAdd(ttl::BlockAddOp op) {
    OpBuilder builder(op);
    Location loc = op.getLoc();  // Original Python location

    // Create loop with fused location showing transformation
    auto fusedLoc = builder.getFusedLoc({
        loc,  // Original Python source
        builder.getUnknownLoc()  // Or NameLoc for "TTLLowerCompute"
    });

    auto loop = builder.create<scf::ForOp>(fusedLoc, ...);
    // Child operations inherit or refine location
    builder.create<ttkernel::AddTilesOp>(loc, ...);
}
```

**Phase 3: Location Attributes on Operations**

Add optional location metadata to key TTL operations:

```tablegen
def TTL_KernelOp : TTL_Op<"kernel"> {
  let arguments = (ins
    // ... existing args
    OptionalAttr<StrAttr>:$python_source_file,
    OptionalAttr<I64Attr>:$python_source_line
  );
}
```

**Error Diagnostic Example:**

```
error: ttl.cb_wait operation timeout - consumer waiting indefinitely
  at eltwise_add.py:42:16
      a_block = a_in_cb.wait()
                       ^~~~~
  note: corresponding producer at eltwise_add.py:65:20
      a_in_cb.push()
             ^~~~~
  note: lowered from ttl.cb_wait at eltwise_add.mlir:128
```

**Implementation Details:**

1. **TTLDialectCompiler tracks source context**:
   ```python
   self.source_file = inspect.getsourcefile(f)
   self.source_lines = inspect.getsourcelines(f)
   ```

2. **MLIR FileLineColLoc for all operations**:
   ```python
   loc = Location.file(filename, line, col, context=self.ctx)
   with loc:
       op = ttl.create_cb(...)  # Op gets location
   ```

3. **Pass preservation**:
   - Use `op.getLoc()` when creating replacement ops
   - Create `FusedLoc` for multi-step transformations
   - Add `NameLoc` for compiler-generated code

4. **Verification and error emission**:
   ```cpp
   if (failed(validateCBUsage(op))) {
       return op.emitError("circular buffer protocol violation")
           << "wait() called without matching push()";
       // Automatically includes file:line:col from op.getLoc()
   }
   ```

**Benefits:**
- Python error messages show source location
- IDE can jump from error to source
- Debugger can map back to Python (future)
- Profiler shows Python-level hotspots

**Testing:**
```python
# Verify location tracking
def test_location_tracking():
    @ttl.kernel(grid=(1,1))
    def bad_kernel():
        cb = CircularBuffer(shape=(1,1))
        cb.pop()  # Error: pop without wait

    try:
        bad_kernel(a, b)
    except Exception as e:
        assert "bad_kernel.py:5" in str(e)  # Points to cb.pop() line
```

**Files to Modify:**
- `python/ttlang/_src/ttl_ast.py` - Capture locations in compiler
- `lib/Dialect/TTL/IR/*.cpp` - Preserve locations in passes
- `include/ttlang/Dialect/TTL/IR/TTLOps.td` - Optional source attrs

**See**: MLIR [Location documentation](https://mlir.llvm.org/docs/Diagnostics/#source-locations)

### 5.5 Control Flow Integration

TTL reuses SCF dialect for control flow instead of custom operations:

```mlir
// Python: for i in range(N):
scf.for %i = %c0 to %N step %c1 {
  ttl.cb_wait %cb, 1
  // ... operations
} {ttl.grid = #ttl.grid<[2, 2]>}

// Python: if core_num == 0:
%core = ttl.core_index(dims = 1)
scf.if %cond {
  // ... operations
} {ttl.core_mask = #ttl.core_mask<[0]>}
```

**Benefits:**
- Inherit MLIR loop analyses (LICM, dependence analysis)
- Standard canonicalization patterns
- No need to reinvent control flow infrastructure

---

## 6. Type Conversion & Lowering Examples

### 6.1 TTL → TTKernel Type Mapping

| TTL Type | TTKernel Type | Conversion Notes |
|----------|---------------|------------------|
| `!ttl.cb<[2,1], !ttcore.tile<32x32,f32>, 2, L1>` | `!ttkernel.cb<4, !ttcore.tile<32x32,f32>>` | Total tiles = 2×1×2 = 4; memspace drives L1 address |
| `!ttl.block<tensor<2x1x!ttcore.tile>>` | Elided | Blocks dissolved into tile operations |
| `!ttl.mem_tx` | N/A (elided) | Lowers to global barriers |
| `!ttl.semaphore` | `!ttkernel.semaphore` | Direct mapping |
| `!ttl.pipe` | Attributes on ops | Pipe dissolved into core coords + multicast flags |
| `!ttl.accessor<layout, memspace>` | N/A (resolved) | Resolves to NOC addresses + layout info |

### 6.2 Operation Lowering Examples

**Circular Buffer Operations:**
```mlir
// TTL
ttl.cb_wait %cb, 2

// TTKernel
ttkernel.cb_wait_front %cb, %c2_i32
```

**Tile Extraction:**
```mlir
// TTL
%tile0 = ttl.get_tile %cb, %c0
%tile1 = ttl.get_tile %cb, %c1

// TTKernel
%tile0 = ttkernel.get_tile %cb, 0
%tile1 = ttkernel.get_tile %cb, 1
```

**Block Compute:**
```mlir
// TTL (block-level abstraction)
ttl.cb_wait %a_cb, 2
ttl.cb_wait %b_cb, 2
%result = ttl.block_add %a_blk, %b_blk  // Abstract 2x1 block addition

// TTKernel (explicit per-tile)
ttkernel.cb_wait_front %a_cb, 2
ttkernel.cb_wait_front %b_cb, 2
ttkernel.tile_regs_acquire
scf.for %i = %c0 to %c2 step %c1 {
  %a_tile = ttkernel.get_tile %a_cb, %i
  %b_tile = ttkernel.get_tile %b_cb, %i
  ttkernel.add_tiles_init
  ttkernel.add_tiles %a_tile, %b_tile, %dst_idx
  ttkernel.pack_tile %dst_idx, %out_cb, %i
}
ttkernel.tile_regs_release
ttkernel.cb_push_back %out_cb, 2
```

**DMA Operations:**
```mlir
// TTL (high-level)
%accessor = ttl.tensor_accessor %input {layout = #ttl.layout<...>, memspace = L1}
%idx = ttl.index_accessor %accessor, %row, %col
%tx = ttl.copy %idx, %cb
ttl.wait %tx

// TTKernel (hardware-specific)
%noc_addr = ttkernel.get_noc_addr %src_core, %src_l1_addr
%cb_l1_addr = // From TTLAllocateCircularBuffers pass
ttkernel.noc_async_read %noc_addr, %cb_l1_addr, %size_bytes
ttkernel.noc_async_read_barrier
```

**Multicast Pipe:**
```mlir
// TTL
%pipe = ttl.create_pipe src_core=[0,0] dst_core_range=[[0,0],[1,3]]
%tx = ttl.copy %block, %pipe
ttl.wait %tx

// TTKernel
%noc_addrs = // Compute multicast address range
ttkernel.noc_async_write_multicast %src_addr, %noc_addrs, %size
ttkernel.noc_async_write_barrier
```

---

## 7. Python Integration

### 7.1 Frontend Compilation

**New file**: `python/ttlang/_src/ttl_ast.py`

```python
class TTLDialectCompiler(TTCompilerBase):
    """Compiler from Python AST to TTL dialect operations."""

    def __init__(self, name, kernel_type=None, captures={}, *args, **kwargs):
        super().__init__(name, kernel_type, *args, **kwargs)
        self.context = CompilerContext(
            grid=kwargs.get("grid", [1, 1]),
            memory_space=kwargs.get("memory_space", "L1"),
            tiled=kwargs.get("tiled", True),
        )
        # Register TTL ops with @syntax decorator system
        self._register_ttl_syntax()

    def visit_AsyncFunctionDef(self, node):
        """Handle @compute() and @datamovement() decorated functions."""
        if self.kernel_type == "compute":
            return self._emit_compute_thread(node)
        elif self.kernel_type == "datamovement":
            return self._emit_datamovement_thread(node)

    def _emit_compute_thread(self, node):
        """Generate ttl.compute_thread operation."""
        # ... AST traversal generating TTL ops

    def _emit_datamovement_thread(self, node):
        """Generate ttl.datamovement_thread operation."""
        # ... AST traversal generating TTL ops
```

### 7.2 Operator Mapping

Update `python/ttlang/operators.py`:

```python
@syntax("!tensor")
class TensorBlock:
    def __add__(ast_self, rhs):
        # Generate ttl.block_add instead of linalg.generic
        return ttl.block_add(ast_self, rhs)

    def __matmul__(ast_self, rhs):
        return ttl.block_matmul(ast_self, rhs)

    def store(ast_self, value):
        return ttl.block_store(ast_self, value)

class CircularBuffer:
    def wait(self):
        num_tiles = self.get_num_tiles()
        return ttl.cb_wait(self.handle, num_tiles)

    def reserve(self):
        num_tiles = self.get_num_tiles()
        return ttl.cb_reserve(self.handle, num_tiles)

    # ... pop, push methods

@syntax("dma")
def dma(src, dst, **kwargs):
    return ttl.copy(src, dst, **kwargs)
```

### 7.3 API Entry Point

Update `python/ttlang/d2m_api.py` → `python/ttlang/api.py`:

```python
def ttl_kernel(
    grid: Union[tuple, Callable],
    memory_space: str = "L1",
    tiled: bool = True,
) -> Callable:
    """
    Decorator for TTL kernel compilation (replaces pykernel_gen).

    Compiles Python functions into TTL dialect, runs transformation passes,
    lowers to TTKernel, and generates C++ kernels.
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Parse Python AST using TTLDialectCompiler
            program = compile_to_ttl(f, args, kwargs, grid, memory_space, tiled)

            # Run TTL pass pipeline
            pm = PassManager.parse("builtin.module("
                "ttcore-register-device,"
                "ttl-validate,"
                "ttl-insert-synchronization,"
                "ttl-allocate-cbs,"
                "ttl-expand-threads,"
                "ttl-assign-dst-registers,"
                "ttl-lower-to-ttkernel"
            ")")
            pm.run(program)

            # Continue with TTKernel → EmitC → C++
            # ... existing pipeline
        return wrapper
    return decorator

# Backward compatibility alias
pykernel_gen = ttl_kernel
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
**Goal**: TTL dialect compiles and registers with MLIR

1. **Dialect Definition (C++)**
   - Create `include/ttlang/Dialect/TTL/` and `lib/Dialect/TTL/`
   - Define types (TTLTypes.td): CB, Block, MemTx, Semaphore, Pipe, Accessor
   - Define ops (TTLOps.td): structural, CB, compute, DM, sync, utility
   - Define attributes (TTLOpsAttrs.td): MemorySpace, Grid, CoreMask, Layout
   - Implement builders, printers, parsers

2. **CMake Integration**
   - Update `include/ttlang/CMakeLists.txt` and `lib/CMakeLists.txt`
   - Add TTL subdirectories and dependencies

3. **Basic Testing**
   - Create `test/ttmlir/Dialect/TTL/` with lit tests
   - Test type parsing, op syntax, basic verification

**Deliverable**: TTL dialect loads, ops parse/print correctly

### Phase 2: Python Compilation (Weeks 2-3)
**Goal**: Python AST generates valid TTL operations

1. **TTL Compiler**
   - Implement `TTLDialectCompiler` in `python/ttlang/_src/ttl_ast.py`
   - Handle `@compute()` and `@datamovement()` decorators
   - Map Python operators to TTL ops (@syntax system)

2. **Operator Updates**
   - Update `python/ttlang/operators.py` for TTL ops
   - Update `python/ttlang/circular_buffer.py` for TTL CB ops
   - Update `python/ttlang/semaphore.py` for TTL semaphore ops

3. **Python Bindings**
   - Create `python/ttlang/dialects/ttl.py` with nanobind
   - Expose TTL types and op builders to Python

4. **Testing**
   - Port `test/python/test_simple_add.py` to TTL
   - Verify IR generation matches expected TTL ops

**Deliverable**: Python examples compile to TTL IR

### Phase 3: Validation & Canonicalization (Weeks 3-4)
**Goal**: IR validation and optimization patterns work

1. **Verifier Implementation**
   - CB reserve/push and wait/pop pairing
   - Pipe source/destination consistency
   - Memory space compatibility
   - Layout attribute validation

2. **Canonicalization Patterns**
   - Fold constant `ttl.core_index` / `ttl.grid_size`
   - Eliminate redundant CB ops
   - Simplify accessor indexing

3. **Testing**
   - Negative tests for invalid IR
   - Canonicalization tests

**Deliverable**: TTL IR is validated and optimized

### Phase 4: Analysis Passes (Weeks 4-6)
**Goal**: Synchronization inference and resource allocation

1. **TTLInsertSynchronization Pass**
   - Build producer-consumer DAG
   - Insert `ttl.dma_barrier` where needed
   - Validate CB/pipe/semaphore usage

2. **TTLAllocateCircularBuffers Pass**
   - Liveness analysis for CBs
   - First-fit allocation in L1
   - Annotate CBs with addresses

3. **TTLInferDSTRequirements Pass**
   - Mark values needing DST registers
   - Propagate `ttl.require_dst` hints

4. **Testing**
   - Test synchronization insertion on examples
   - Test CB allocation with various patterns

**Deliverable**: Compiler automatically inserts sync and allocates memory

### Phase 5: Thread Expansion (Week 6)
**Goal**: Convert regions to separate functions

1. **TTLExpandThreads Pass**
   - Extract `ttl.compute_thread` → `func.func @compute_X`
   - Extract `ttl.datamovement_thread` → `func.func @dm_X`
   - Preserve metadata (grid, memory_space, tiled)

2. **Testing**
   - Verify function extraction
   - Check metadata preservation

**Deliverable**: Threads become separate functions

### Phase 6: Lowering Passes (Weeks 7-9)
**Goal**: TTL → TTKernel conversion

1. **TTLAssignDSTRegisters Pass**
   - Track 16 DST register slots
   - Insert `ttkernel.tile_regs_acquire/release`

2. **TTLLowerCompute Pass**
   - `ttl.block_add` → `scf.for` + `ttkernel.add_tiles`
   - `ttl.block_matmul` → `ttkernel.matmul_tiles`
   - Insert init operations

3. **TTLLowerDataMovement Pass**
   - `ttl.copy(accessor, cb)` → `ttkernel.noc_async_read`
   - `ttl.copy(cb, accessor)` → `ttkernel.noc_async_write`
   - `ttl.copy(cb, pipe, cb)` → multicast/unicast NOC ops
   - Compute NOC addresses

4. **TTLLowerSynchronization Pass**
   - `ttl.cb_wait` → `ttkernel.cb_wait_front`
   - `ttl.cb_reserve` → `ttkernel.cb_reserve_back`
   - `ttl.semaphore_*` → `ttkernel.noc_semaphore_*`

5. **Testing**
   - Test each pass individually with lit tests
   - End-to-end test: Python → TTL → TTKernel → C++

**Deliverable**: Complete TTL → TTKernel lowering pipeline

### Phase 7: Integration & Validation (Weeks 9-10)
**Goal**: End-to-end examples working

1. **Example Porting**
   - Port `examples/eltwise_add.py` to TTL
   - Port `examples/custom_dm_matmul.py` to TTL
   - Compare generated C++ with current pipeline

2. **Performance Validation**
   - Run on hardware (if available)
   - Compare performance vs current pipeline

3. **Documentation**
   - Update `docs/HITCHHIKERS_GUIDE.md`
   - Document TTL operations
   - Document pass pipeline

**Deliverable**: Working TTL pipeline for production use

---

## 9. Future Evolution

### 9.1 Microcore Model (Post-MVP)

Once the simple threading model is validated, evolve to parametric microcore abstraction:

```tablegen
def TTL_TileProcOp : TTL_Op<"tile_proc"> {
  let summary = "Hardware execution unit with declared microcores";
  let arguments = (ins TTL_MicrocoreConfigAttr:$microcores);
  let regions = (region VariadicRegion<SizedRegion<1>>:$threads);
}

def TTL_ThreadOp : TTL_Op<"thread"> {
  let summary = "Thread bound to specific microcore";
  let arguments = (ins
    SymbolRefAttr:$microcore_role  // References parent microcores
  );
  let regions = (region SizedRegion<1>:$body);
}
```

**Migration path:**
1. Add `microcore_hint` attribute to existing thread ops
2. Introduce `ttl.tile_proc` as optional wrapper
3. Create automatic conversion pass: simple threads → microcore model
4. Update examples incrementally

### 9.2 C++ Backend (Post-TTKernel)

After TTL → TTKernel is stable, add direct C++ emission:

```
TTL → TTKernel → EmitC → C++ (current)
TTL → EmitC → C++ (future)
```

Benefits:
- Standalone kernels (no TTKernel dependency)
- Potential for custom optimizations
- Better debugging (direct C++ inspection)

### 9.3 Single-Threaded Mode

For simpler kernels, support single-threaded synchronous model:

```python
@ttl.kernel(mode="single_threaded")
def simple_add(a, b, out):
    # No explicit threads - compiler designs pipeline
    for i in range(N):
        result = a[i] + b[i]
        out[i].store(result)
```

Compiler automatically:
- Fissions into threads
- Inserts synchronization
- Allocates resources

### 9.4 Distributed Tensor Type (Major Extension)

**Goal**: Enable programming of larger distributed systems with explicit tensor distribution across cores.

**Problem**: Current TTL uses implicit SPMD model (all cores run same program with implicit sharding). For complex distributed algorithms, need explicit control over:
- How tensors are partitioned across cores
- Different programs on different core subsets
- Explicit redistribution strategies
- Per-core shard shapes

**Proposal**: `!ttl.dist_tensor` type implementing `bufferization::TensorLikeType`

```tablegen
def TTL_DistributedTensor : TTL_Type<"DistributedTensor", "dist_tensor"> {
  let summary = "Distributed tensor across grid of cores";
  let parameters = (ins
    "ArrayRef<int64_t>":$gridShape,      // Distribution grid [8, 8]
    "ArrayRef<int64_t>":$shardShape,     // Per-core shard [32, 32]
    "Type":$elementType,                  // f32, bf16, etc.
    "DistributionStrategyAttr":$strategy, // Sharded, interleaved, etc.
    "TTL::MemorySpace":$memorySpace
  );

  let extraClassDeclaration = [{
    // ShapedType interface
    ArrayRef<int64_t> getShape() const {
      // Logical shape = grid_shape × shard_shape
      // grid=[8,8], shard=[32,32] → shape=[256, 256]
    }

    // TensorLikeType interface
    FailureOr<BufferLikeType> getBufferType(...) const {
      // Bufferizes to plain per-core memrefs
      return MemRefType::get(getShardShape(), getElementType(),
                             nullptr, getMemorySpace());
    }
  }];
}
```

**Key characteristics:**
- **Tensor-level distribution**: Keeps high-level optimizations on tensors
- **Bufferizes to plain memrefs**: Each core gets standard `memref<shard_shape, memspace>`
- **Metadata tracking**: Transient `ttcore.shard_group<tensor_id, shard_idx>` attribute for DMA planning
- **Reuses MLIR bufferization**: Plugs into One-Shot Bufferization infrastructure

**Integration with TTL:**

```mlir
// TTL with distributed tensors
func.func @distributed_matmul(
    %A: !ttl.dist_tensor<grid=[8,8], shard=[32,32], f32, sharded, L1>,
    %B: !ttl.dist_tensor<grid=[8,8], shard=[32,32], f32, sharded, L1>
) -> !ttl.dist_tensor<...> {

  ttl.kernel grid=[8,8] {
    // Per-core view extraction
    %a_local = ttl.extract_shard %A
    %b_local = ttl.extract_shard %B

    // TTL operations on local shards
    ttl.compute_thread {
      // Compute on local data
    }

    // Redistribution if needed
    %B_redistributed = ttl.redistribute %B strategy=interleaved
  }
}
```

**Phasing:**
1. **Phase 1 (MVP)**: TTL with implicit sharding (current plan)
2. **Phase 2**: Add `!ttl.dist_tensor` type
3. **Phase 3**: Implement bufferization for distributed tensors
4. **Phase 4**: Add redistribution operations and strategies
5. **Phase 5**: Integration with standard MLIR deallocation

**Dependencies:**
- Working TTL → TTKernel lowering
- Proven memory allocation strategy
- Validated CB and synchronization patterns
- Understanding of real distributed algorithm needs

**Benefits:**
- Explicit control over tensor distribution
- Type-safe distributed programming
- Reuse MLIR bufferization infrastructure
- Enable complex multi-core algorithms (e.g., distributed attention, pipeline parallelism)

**See**: `docs/DistributedTensorType.md` for complete proposal

### 9.5 Transform Dialect Integration for Scheduling

**Goal**: Use MLIR's Transform dialect for composable scheduling and optimization instead of monolithic optimization passes.

**Motivation**: Transform dialect provides:
- **Composability**: Build complex schedules from simple transform operations
- **Precise targeting**: Apply transformations to specific operations via handles
- **Debugging**: Inspect intermediate IR after each transformation
- **Reusability**: Define scheduling strategies as transform sequences
- **DSL alignment**: Map TT-lang scheduling concepts directly to transform ops

**Scope**: Use Transform dialect for scheduling/optimization; keep traditional passes for lowering (TTL → TTKernel).

#### Custom Transform Operations for TTL

**DST Register Scheduling:**
```tablegen
def TTLAllocateDSTRegistersOp : TransformDialectOp<"ttl.allocate_dst_registers"> {
  let summary = "Allocate DST registers for compute operations";
  let description = [{
    Schedules TTL compute operations to use the 16 available DST register slots.
    Performs liveness analysis and assigns register indices to minimize spills.

    Returns handle to modified compute operations with register assignments.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
}
```

**Compute/DMA Overlap Scheduling:**
```tablegen
def TTLSchedulePipelineOp : TransformDialectOp<"ttl.schedule_pipeline"> {
  let summary = "Schedule compute/DMA overlap for maximum throughput";
  let description = [{
    Reorders DMA and compute operations to maximize pipeline overlap.
    Inserts barriers only where necessary for correctness.

    Strategy:
    - Issue DMAs early (prefetch next tiles)
    - Overlap compute with pending DMAs
    - Minimize bubble time
  }];
  let arguments = (ins
    TransformHandleTypeInterface:$kernel,
    OptionalAttr<StrAttr>:$strategy  // "aggressive", "conservative", "auto"
  );
  let results = (outs TransformHandleTypeInterface:$scheduled_kernel);
}
```

**CB Buffer Factor Optimization:**
```tablegen
def TTLOptimizeBufferFactorOp : TransformDialectOp<"ttl.optimize_buffer_factor"> {
  let summary = "Determine optimal circular buffer sizes";
  let description = [{
    Analyzes CB usage patterns and adjusts buffer_factor for each CB:
    - Single buffering (factor=1) when no overlap possible
    - Double buffering (factor=2) for basic pipelining
    - Triple+ buffering (factor=3+) for deep pipelines

    Considers L1 capacity constraints.
  }];
  let arguments = (ins TransformHandleTypeInterface:$cbs);
  let results = (outs TransformHandleTypeInterface:$optimized_cbs);
}
```

**DMA Coalescing:**
```tablegen
def TTLCoalesceDMAOp : TransformDialectOp<"ttl.coalesce_dma"> {
  let summary = "Merge adjacent DMA operations";
  let description = [{
    Combines multiple small DMA operations into larger transfers when:
    - Source/destination addresses are contiguous
    - No intervening operations create dependencies
    - Total size fits within NOC packet limits
  }];
  let arguments = (ins TransformHandleTypeInterface:$dma_ops);
  let results = (outs TransformHandleTypeInterface:$coalesced);
}
```

#### Example Transform Sequence

```mlir
// TTL IR with transform sequence
module {
  // Target TTL kernel
  func.func @matmul(...) {
    ttl.kernel {
      ttl.compute_thread { ... }
      ttl.datamovement_thread { ... }
    }
  }

  // Transform sequence (applied by interpreter or pass)
  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // 1. Get handle to kernel
    %kernel = transform.structured.match ops{["ttl.kernel"]} in %root

    // 2. Optimize CB buffer factors
    %cbs = transform.structured.match ops{["ttl.create_cb"]} in %kernel
    %opt_cbs = transform.ttl.optimize_buffer_factor %cbs

    // 3. Schedule pipeline overlap
    %sched_kernel = transform.ttl.schedule_pipeline %kernel {strategy = "aggressive"}

    // 4. Allocate DST registers
    %compute_ops = transform.structured.match ops{["ttl.block_add", "ttl.block_matmul"]}
                   in %sched_kernel
    %with_dst = transform.ttl.allocate_dst_registers %compute_ops

    // 5. Coalesce DMAs
    %dma_ops = transform.structured.match ops{["ttl.copy"]} in %sched_kernel
    %coalesced = transform.ttl.coalesce_dma %dma_ops

    // 6. Apply standard lowering (traditional pass)
    transform.apply_registered_pass "ttl-lower-to-ttkernel" to %sched_kernel
  }
}
```

#### Integration Architecture

**Phase 1 (MVP)**: Traditional passes only
- Implement basic lowering without Transform dialect
- Get end-to-end working quickly
- Establish baseline performance

**Phase 2**: Add Transform operations for scheduling
- Define custom transform ops (`transform.ttl.*`)
- Implement interpreters for each transform
- Keep traditional lowering passes unchanged

**Phase 3**: Composable scheduling strategies
- Define scheduling "recipes" as transform sequences
- Enable user-controlled optimization (like Halide schedules)
- Potential: Python API for scheduling (`kernel.schedule().tile(...).fuse(...)`)

**Benefits for TTL:**
1. **Separation of concerns**: Lowering logic separate from scheduling logic
2. **Experimentation**: Try different schedules without modifying passes
3. **Debugging**: Inspect IR after each transform step
4. **Autotuning**: Search space of scheduling strategies by varying transform sequences
5. **User control**: Advanced users can write custom transform sequences

#### Files to Create (Phase 2)

```
include/ttlang/Dialect/TTL/Transform/
  TTLTransformOps.td           // Transform operation definitions
  TTLTransformOps.h

lib/Dialect/TTL/Transform/
  TTLTransformOps.cpp          // Transform interpreters
  CMakeLists.txt
```

**Transform Operations to Implement:**
- `transform.ttl.allocate_dst_registers`
- `transform.ttl.schedule_pipeline`
- `transform.ttl.optimize_buffer_factor`
- `transform.ttl.coalesce_dma`
- `transform.ttl.reorder_for_locality`
- `transform.ttl.insert_prefetch`

#### Comparison: Traditional Pass vs Transform

| Aspect | Traditional Pass | Transform Dialect |
|--------|------------------|-------------------|
| **Flexibility** | Fixed algorithm | Composable sequences |
| **Targeting** | Pattern matching | Explicit handles |
| **Debugging** | Black box | Step-by-step IR |
| **Reusability** | Monolithic | Mix-and-match transforms |
| **MVP Complexity** | Lower | Higher (infrastructure needed) |
| **Long-term Composability** | Limited | Excellent |

**Recommendation**:
- **MVP**: Traditional passes (TTLAssignDSTRegisters, etc.)
- **Phase 2**: Add transform.ttl.* operations alongside traditional passes
- **Phase 3**: Migrate optimization logic to transform sequences, keep lowering as passes

This hybrid approach gets TTL working quickly while building toward the more composable Transform dialect model for scheduling.

---

## 10. Risk Mitigation

### 10.1 Complexity Management
- **Risk**: Dialect too complex, slow MVP
- **Mitigation**: Start minimal (Phase 1-2), expand incrementally
- **Fallback**: Keep simple ops, defer advanced features

### 10.2 TTKernel Compatibility
- **Risk**: TTL doesn't map cleanly to TTKernel
- **Mitigation**: Study existing pykernel lowering closely, reuse patterns
- **Validation**: Compare generated TTKernel IR with current pipeline

### 10.3 Python API Stability
- **Risk**: Breaking changes to user code
- **Mitigation**: Keep existing decorator names (`@compute`, `@datamovement`)
- **Backward compat**: Alias `pykernel_gen` to `ttl_kernel`

### 10.4 Pass Pipeline Bugs
- **Risk**: Transformation passes introduce errors
- **Mitigation**: Extensive lit tests per pass, save intermediate IR at each stage
- **Debugging**: `TTLANG_VERBOSE_PASSES=1` environment variable

### 10.5 Memory Allocation Failures
- **Risk**: L1 capacity exceeded, OOM at runtime
- **Mitigation**: Validate allocation in pass, error early with clear diagnostics
- **Future**: Spilling to DRAM, buffer factor optimization

---

## 11. Success Criteria

1. **TTL dialect compiles** and is registered in MLIR
2. **Python AST → TTL** generates valid TTL operations
3. **All passes execute** without crashing on examples
4. **TTL → TTKernel** produces correct IR (verified vs current pipeline)
5. **End-to-end test** (Python → C++) works for eltwise_add
6. **Performance parity** with current d2m.generic approach
7. **No D2M dependencies** in the TTL path

---

## 12. Appendix: Design Rationale

### Why Tensor-Level (Not Tile-Level)?
- Matches Python DSL semantics (users think in blocks)
- Enables high-level optimizations before committing to tiles
- Lowering passes make tile iteration explicit when needed

### Why Simple Threading (Not Microcore)?
- Faster MVP, matches current examples
- Clear evolution path documented
- Can add microcore abstraction later without breaking existing code

### Why Direct to TTKernel (Not via D2M)?
- Avoids `d2m.generic` complexity
- Cleaner semantic representation
- Still compatible with existing TTKernel → EmitC → C++ pipeline

### Why SCF for Control Flow?
- Reuse proven MLIR infrastructure
- Inherit canonicalization and analysis passes
- No need to reinvent loop optimizations

### Why Memory Space in Types?
- Makes IR self-descriptive
- Enables early validation of illegal transfers
- Simplifies lowering (memspace dictates NOC operations)

---

## 13. References

- **TT-lang Spec**: `docs/TT-lang.md`
- **Build System**: `docs/BUILD_SYSTEM.md`
- **Testing Guide**: `test/TESTING.md`
- **Current D2M Pipeline**: `python/ttlang/d2m_api.py`
- **TTKernel Dialect**: `../tt-mlir/include/ttmlir/Dialect/TTKernel/IR/`
- **Allocate Pass Experiment**: `_allocate_pass.diff`

---

**Document Status**: Design phase complete, ready for implementation
**Next Step**: Phase 1 - Dialect definition (C++)
