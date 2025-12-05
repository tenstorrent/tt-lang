# TTL Dialect Design Plan (Integrated)

**Version**: 1.0
**Date**: 2025-01-XX
**Status**: Design Phase

This document integrates design decisions from two planning efforts to define the TTL (TT-Lang) MLIR dialect for representing TT-lang DSL programs with explicit transformation passes.

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
Python Kernel → Python AST → D2M Generic Ops → TTIR Pipeline → TTKernel → EmitC → C++
```

### TTL Flow
```
Python Kernel → Python AST → TTL Dialect → TTL Passes → TTKernel → EmitC → C++
                                    ↓
                         Synchronization, Allocation,
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
ttl.kernel (with thread regions)
  ↓
[Phase 1: Validation & Canonicalization]
  ├─ TTLValidatePass - Verify CB/pipe/semaphore contracts
  ├─ TTLCanonicalizePass - Fold constants, simplify patterns
  └─ TTLVerifyLayoutsPass - Check tensor accessor layouts
  ↓
[Phase 2: Analysis & Inference]
  ├─ TTLInsertSynchronization - Analyze inter-thread dataflow, insert barriers
  ├─ TTLAllocateCircularBuffers - Assign L1 addresses (liveness-based)
  └─ TTLInferDSTRequirements - Mark values needing DST registers
  ↓
[Phase 3: Thread Expansion]
  └─ TTLExpandThreads - Convert ttl.kernel regions → separate func.func
  ↓
func.func @compute_thread_0
func.func @dm_thread_0
  ↓
[Phase 4: Resource Assignment]
  └─ TTLAssignDSTRegisters - Allocate 16 DST registers for compute threads
  ↓
[Phase 5: Lowering to TTKernel]
  ├─ TTLLowerCompute - ttl.block_add → scf.for + ttkernel.add_tiles
  ├─ TTLLowerDataMovement - ttl.copy → ttkernel.noc_async_*
  └─ TTLLowerSynchronization - ttl.cb_* → ttkernel.cb_*
  ↓
ttkernel.* operations
  ↓
[Existing Pipeline]
  └─ ConvertTTKernelToEmitC
  ↓
EmitC → C++ Code
```

### 5.2 Key Pass Descriptions

**TTLInsertSynchronization**
- **Input**: `ttl.kernel` with thread regions
- **Analysis**: Build producer-consumer DAG for blocks, semaphores, pipes
- **Transform**: Insert `ttl.dma_barrier` where needed, validate CB usage patterns
- **Output**: `ttl.kernel` with explicit synchronization

**TTLAllocateCircularBuffers**
- **Input**: `ttl.kernel` with CBs
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

### 5.3 Control Flow Integration

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

### 9.5 Optimization Passes

- **TTLPipelineOptimization**: Overlap compute and DMA
- **TTLCBBufferOptimization**: Minimize buffer factors
- **TTLDMACoalescing**: Merge adjacent transfers
- **TTLLoopInvariantCodeMotion**: Hoist ops out of loops

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
