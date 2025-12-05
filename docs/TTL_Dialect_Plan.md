# TTL Dialect Design Plan

**Version**: 0.6

**Modified**: 2025-12-05

**Status**: Design Phase - Under Revision

This document specifies the TTL (TT-Lang) MLIR dialect, a tensor-level
intermediate representation designed to directly capture the semantics of the
TT-lang DSL. The TTL dialect enables multi-stage compilation with explicit
transformation passes for synchronization inference, resource allocation, and
hardware-specific optimizations before lowering to executable kernels.



## Table of Contents

1. [Motivation & Goals](#1-motivation--goals)
2. [Architecture Overview](#2-architecture-overview)
3. [Type System](#3-type-system)
   - 3.1 [Core Types](#31-core-types)
   - 3.2 [Attributes](#32-attributes)
4. [Operations](#4-operations)
   - 4.1 [Structural Operations](#41-structural-operations)
   - 4.2 [Resource Creation](#42-resource-creation)
   - 4.3 [Circular Buffer Operations](#43-circular-buffer-operations)
   - 4.4 [Compute Operations (Fusion)](#44-compute-operations)
   - 4.5 [Data Movement Operations](#45-data-movement-operations)
   - 4.6 [Synchronization Operations](#46-synchronization-operations)
   - 4.7 [Utility Operations](#47-utility-operations)
   - 4.8 [Compute Operations Inventory](#48-compute-operations-inventory)
5. [Compilation Pipeline](#5-compilation-pipeline)
   - 5.1 [Pass Architecture](#51-pass-architecture)
   - 5.2 [Key Pass Descriptions](#52-key-pass-descriptions)
   - 5.3 [Granularity and Block Shapes](#53-granularity-and-block-shapes)
   - 5.4 [Source Location Tracking](#54-source-location-tracking)
   - 5.5
     [Control Flow: SCF vs Affine Dialect](#55-control-flow-scf-vs-affine-dialect)
   - 5.6 [Error Handling and Diagnostics](#56-error-handling-and-diagnostics)
6. [Type Conversion & Lowering Examples](#6-type-conversion--lowering-examples)
   - 6.1 [TTL → TTKernel Type Mapping](#61-ttl--ttkernel-type-mapping)
   - 6.2
     [TTL → TTKernel Operation Mapping](#62-ttl--ttkernel-operation-mapping)
   - 6.3 [Operation Lowering Examples](#63-operation-lowering-examples)
7. [Python Integration](#7-python-integration)
   - 7.1 [Frontend Compilation](#71-frontend-compilation)
   - 7.2 [Operator Mapping](#72-operator-mapping)
   - 7.3 [API Entry Point](#73-api-entry-point)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [TTNN Runtime Integration](#9-ttnn-runtime-integration)
10. [Future Evolution](#10-future-evolution)
    - 10.1 [Microcore Model (Post-MVP)](#101-microcore-model-post-mvp)
    - 10.3
      [Distributed Tensor Type (Major Extension)](#103-distributed-tensor-type-major-extension)
    - 10.4
      [Transform Dialect Integration for Scheduling](#104-transform-dialect-integration-for-scheduling)
11. [Success Criteria](#11-success-criteria)
12. [Appendix: Design Rationale](#12-appendix-design-rationale)
13. [References](#13-references)



## Executive Summary

The TTL dialect provides a tensor-level intermediate representation for TT-lang
programs (defined in [TT-lang.md](TT-lang.md)), enabling multi-stage lowering
with explicit compiler transformations before generating C++ kernels. Generated
kernels can execute on either the TTNN runtime (via dylib workflow) or the
TT-Metal runtime (via flatbuffer workflow). TTL is designed specifically for the
TT-lang DSL surface language, while D2M serves as a lower-level dialect for data
movement and compute operations.

**Key Design Decisions:**
- **Threading Model**: Simple compute/datamovement threads (MVP), with microcore
  evolution path documented
- **Abstraction Level**: Tensor-level (blocks of tiles), not individual tiles
- **Type System**: Memory spaces explicit in types; SSA values for all resources
  (CBs, pipes, semaphores)
- **Control Flow**: Hybrid SCF/Affine dialect with TTL attributes
- **Lowering Path**: TTL → TTKernel → EmitC → C++ kernel source (compiled
  separately)
- **Phasing**: Multi-threaded from start (matches current examples)



## 1. Motivation & Goals

### Motivation for TTL Dialect

D2M dialect serves as a general-purpose data movement and compute abstraction.
For the TT-lang DSL specifically, a dedicated dialect provides:

- DSL-level IR: Preserve TT-lang abstractions (kernels, threads, circular
  buffers, pipes) longer in compilation
- SSA semantics: CB operations with explicit SSA values rather than implicit
  state effects (D2M CB ops use `MemoryEffects<[MemRead, MemWrite]>` for state
  transitions)
- Analysis opportunities: Synchronization inference, resource allocation, and
  scheduling at DSL semantic level
- Flexibility: Experiment with TT-lang-specific optimizations and compilation
  strategies
- Multiple targets: TTKernel (immediate) and potential standalone C++ backend

### TTL Dialect Goals
1. Capture DSL semantics in SSA form: kernels, threads, circular buffers, pipes,
   blocks, semaphores
2. Enable analysis passes: Synchronization inference, memory planning, DST
   register allocation
3. Support transformations: Liveness analysis, operation reordering, pipelining
4. C++ kernel generation: Produce standalone C++ kernels that compile separately
   and link with TTNN runtime
5. Future-proof: Extensible to new hardware generations via attributes

### Non-Goals (MVP)
- Autotuning algorithms (IR has hooks, algorithms come later)
- Single-threaded synchronous model (start multi-threaded)
- Complete TT-lang spec (start minimal, expand incrementally)



## 2. Architecture Overview

### Current Flow
```
Python Kernel → Python AST → D2M Generic Ops → ttir-to-ttmetal Pipeline → TTKernel → EmitC → Flatbuffer Binary
                                                     ↓
                                          (Inside pipeline: fusion,
                                           bufferization, allocation,
                                           DST register assignment,
                                           DMA lowering in D2M passes)
```

### TTL Flow
```
Python Kernel → Python AST → TTL Dialect → TTL Passes → TTKernel → TTMetal Ops → EmitC → C++ Kernel Source
                                    ↓
                         Validation, Synchronization,
                         Bufferization, Allocation,
                         Register Assignment, Optimization
```

**Relationship to D2M Dialect:**

TTL and D2M serve different roles in the compilation pipeline:

- **TTL**: New frontend dialect specifically designed for the TT-lang DSL. TTL
  provides DSL-specific IR enabling TT-lang-aware transformations before
  lowering through TTKernel/TTMetal to C++ kernels. TTL bypasses D2M and goes
  directly to TTKernel dialect, enabling TT-lang-specific optimizations and
  transformations. The generated C++ compiles separately and can execute on
  either the TTNN runtime (dylib workflow) or TT-Metal runtime (flatbuffer
  workflow).

- **D2M**: Remains the primary dialect for framework paths (JAX/PyTorch → TTIR →
  D2M → TTKernel). D2M serves as a general-purpose data movement and compute
  abstraction.

- **Convergence**: Both TTL and D2M paths converge at the TTKernel dialect,
  sharing the same backend lowering to EmitC and C++ code generation.

This separation allows TTL to focus on TT-lang DSL semantics while D2M continues
to serve framework integration needs.



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
    operations lower to global DMA barriers (`ttkernel.noc_async_read_barrier`
    or `ttkernel.noc_async_write_barrier`). This type exists for ordering
    and future optimization opportunities.
    See: `tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td` definitions
    `TTKernel_NocAsyncReadBarrierOp` and `TTKernel_NocAsyncWriteBarrierOp`.
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

def TTL_DistributionStrategyAttr : I32EnumAttr<"DistributionStrategy", "TTL distribution strategy", [
  I32EnumAttrCase<"Sharded", 0>,
  I32EnumAttrCase<"Interleaved", 1>,
  I32EnumAttrCase<"Replicated", 2>
]> {
  let cppNamespace = "::mlir::tt::ttl";
  let description = [{
    TTL-specific distribution strategy attribute for distributed tensor types.
    Similar concepts exist in TTNN (`TTNN_TensorMemoryLayout` with Interleaved,
    HeightSharded, WidthSharded, BlockSharded) and TTCore (`TTCore_TensorMemoryLayout`),
    but TTL uses a simplified strategy enum for its distributed tensor type.
    See: `tt-mlir/include/ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.td` definition
    `TTNN_TensorMemoryLayout` and `tt-mlir/include/ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.td`
    definition `TTCore_TensorMemoryLayout`.
  }];
}
```



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
    Variadic<AnyType>:$inputs,
    TTL_GridAttr:$grid,
    StrAttr:$memory_space,
    BoolAttr:$tiled,
    OptionalAttr<ArrayAttr>:$block_factors,  // Block factors per tensor [[M,N], [K,N], ...]
    OptionalAttr<ArrayAttr>:$compile_time_args,  // Indices of inputs that are compile-time constants
    OptionalAttr<DictionaryAttr>:$compute_config  // Math fidelity, precision modes
  );
  let regions = (region VariadicRegion<SizedRegion<1>>:$threads);
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
    Variadic<AnyType>:$operands,
    OptionalAttr<StrAttr>:$microcore_hint  // Future evolution path
  );
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Thread for mathematical operations on tiles. Executes on MATH microcore(s).
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
    TTL_MemorySpaceAttr:$memory_space, // L1, DRAM, DST
    OptionalAttr<I32Attr>:$buffer_index,  // Optional explicit CB number (0-31)
    OptionalAttr<I64Attr>:$page_size,     // Optional page size in bytes
    OptionalAttr<TTL_CoreMaskAttr>:$core_ranges  // Optional per-core CB mapping
  );
  let results = (outs TTL_CircularBuffer:$result);
  let description = [{
    Python API `ttl.make_circular_buffer_like(tensor, shape=..., buffer_factor=...)`
    maps to this operation. The Python frontend extracts tensor properties
    (dtype, tile shape from layout) and calls `ttl.create_cb` with the extracted
    parameters. The "likeness" refers to deriving tile_type and memory_space
    from the input tensor's layout and properties.

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
  let description = [{
    Blocking operation that waits until the specified number of tiles are available
    in the circular buffer. Used by consumers to acquire data from producers.

    Maps to TTKernel operation:
    - ttkernel.cb_wait_front(cb, num_tiles)
  }];
}

def TTL_CBPopOp : TTL_Op<"cb_pop"> {
  let summary = "Consumer release: signal tiles consumed";
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I64Attr:$num_tiles                // Number of tiles to release
  );
  let description = [{
    Non-blocking operation that signals tiles have been consumed and can be
    reused by the producer.

    Maps to TTKernel operation:
    - ttkernel.cb_pop_front(cb, num_tiles)
  }];
}

def TTL_CBReserveOp : TTL_Op<"cb_reserve"> {
  let summary = "Producer acquire: reserve space in circular buffer";
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I64Attr:$num_tiles                // Number of tiles to reserve
  );
  let description = [{
    Python API `cb.reserve()` maps to this operation. When used with Python
    `with` statement (`with cb.reserve() as blk:`), the AST compiler generates
    `ttl.cb_reserve` at the start of the scope and `ttl.cb_push` at the end.

    Blocking operation that waits until space is available in the circular buffer.
    Used by producers to reserve space for writing data.

    Maps to TTKernel operation:
    - ttkernel.cb_reserve_back(cb, num_tiles)
  }];
}

def TTL_CBPushOp : TTL_Op<"cb_push"> {
  let summary = "Producer release: signal data ready";
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I64Attr:$num_tiles                // Number of tiles to push
  );
  let description = [{
    Python API `cb.push()` maps to this operation. Automatically generated
    at the end of a `with cb.reserve()` scope.

    Non-blocking operation that signals data is ready for consumers.

    Maps to TTKernel operation:
    - ttkernel.cb_push_back(cb, num_tiles)
  }];
}

def TTL_GetTileOp : TTL_Op<"get_tile"> {
  let summary = "Extract tile from circular buffer for compute";
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    Index:$tile_idx                   // Tile index within CB
  );
  let results = (outs AnyType:$tile);  // !ttcore.tile<32x32, f32>
  let description = [{
    Copies a tile from circular buffer to DST register for computation.
    Used in compute threads to load operands.

    Maps to TTKernel operations:
    - ttkernel.copy_tile_init(cb)
    - ttkernel.copy_tile(cb, tile_idx, dst_idx)
  }];
}

def TTL_PackTileOp : TTL_Op<"pack_tile"> {
  let summary = "Pack computed tile into circular buffer";
  let arguments = (ins
    Index:$dst_idx,                   // DST register index
    TTL_CircularBuffer:$cb,
    Index:$tile_idx                   // Tile index within CB
  );
  let description = [{
    Copies a tile from DST register back to circular buffer.
    Used to materialize computation results.

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
- **[4.4.1 Fusion Operations](#441-fusion-operations)**: Operations for fusing
  computation chains
- **[4.4.2 Arithmetic Operations](#442-arithmetic-operations)**: Element-wise
  math (add, mul, matmul)
- **[4.4.3 Reduction and Broadcast](#443-reduction-and-broadcast-operations)**:
  Operations for tensor dimension manipulation
- **[4.4.4 Materialization](#444-materialization-operations)**: Operations for
  storing results
- **[4.4.5 DST Register Management](#445-dst-register-management)**: IR-level
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
directly on DST registers with values flowing from one operation to the next
without intermediate CB allocations or L1 traffic. Only the final result is
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
    TTLAssignDSTRegisters performs liveness analysis to allocate DST registers and
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
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
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
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
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
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
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
    AnyTensor:$input,
    AnyTensor:$scaling,  // Scaling tensor (typically ones) for reduction
    I64Attr:$axis        // Axis to reduce along
  );
  let results = (outs AnyTensor:$result);
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
    AnyTensor:$input,
    I64Attr:$axis  // Axis to reduce along
  );
  let results = (outs AnyTensor:$result);
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
    AnyTensor:$input,
    I64Attr:$axis  // Broadcast axis
  );
  let results = (outs AnyTensor:$result);
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
- TTLValidatePass verifies proper sequencing: each reserve has matching push,
  each wait has matching pop

```tablegen
def TTL_BlockStoreOp : TTL_Op<"block_store"> {
  let summary = "Store computation result to circular buffer block";
  let arguments = (ins AnyTensor:$dst, AnyTensor:$src);
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

#### 4.4.5 DST Register Management

DST register operations are **IR-level constructs** for managing register
allocation at the block abstraction level.

**Block-to-tile lowering context**: TTL operations work on *blocks* (groups of
tiles, e.g., a 2x1 block = 2 tiles). Hardware operations in TTKernel work on
*individual tiles* (32x32 elements). During the TTLLowerCompute pass:

- A single `ttl.block_add` operating on a 2x1 block becomes an `affine.for` loop
  with 2 iterations
- Each iteration performs `ttkernel.add_tiles` on one tile
- DST register allocation happens *after* this expansion, when we know exactly
  how many tile-level operations execute

This lowering occurs because:
- Hardware constraint: TTKernel compute operations (add_tiles, mul_tiles, etc.)
  operate on single tiles, not blocks
- Register allocation: We need to see all tile operations and their lifetimes to
  allocate DST registers efficiently
- Fusion opportunities: Block-level IR enables analysis and fusion before
  committing to tile-level loops

**DST register capacity** is configuration-dependent (4-16 tiles) based on
`fp32_dest_acc_en` and `dst_full_sync_en` settings in `ComputeKernelConfig` (see
[docs/dst_register_capacity.md](dst_register_capacity.md)). The
TTLAssignDSTRegisters pass queries the kernel's compute_config to determine
capacity.

**Spilling** operates at tile granularity: if a kernel's liveness analysis
reveals more live values than available DST capacity, the compiler inserts spill
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

    Returns transaction handle (SSA value of type !ttl.mem_tx) for ordering.
    Python API `ttl.copy()` returns a transfer handle object with a `wait()`
    method, which maps to `ttl.wait %tx` operation.
    
    Note: ttl.wait lowers to global DMA barrier, not per-transaction wait
    (TTKernel limitation). TTKernel only provides `ttkernel.noc_async_read_barrier`
    and `ttkernel.noc_async_write_barrier` operations which wait for all pending
    DMA operations of the respective type, not individual transactions.
    See: `tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td` definitions
    `TTKernel_NocAsyncReadBarrierOp` and `TTKernel_NocAsyncWriteBarrierOp`.
  }];
}

def TTL_IfPipeSrcOp : TTL_Op<"if_pipe_src"> {
  let summary = "Conditionally execute if current core is pipe source";
  let arguments = (ins
    Variadic<TTL_Pipe>:$pipes
  );
  let regions = (region SizedRegion<1>:$thenRegion);
  let description = [{
    Python API `ttl.if_pipe_src(pipes, pipe_src)` maps to this operation.
    The region contains TTL operations that execute if the current core matches
    any pipe's source coordinates. For multiple pipes, the compiler generates
    a loop or nested conditionals to check each pipe.
    
    This operation is composable - the region can contain any TTL operations
    including nested conditionals, DMA operations, CB operations, etc.
    
    Lowers to scf.if with ttl.core comparisons against pipe source coordinates.
    The lowering pass expands multiple pipes into nested conditionals or loops.
  }];
}

def TTL_IfPipeDstOp : TTL_Op<"if_pipe_dst"> {
  let summary = "Conditionally execute if current core is pipe destination";
  let arguments = (ins
    Variadic<TTL_Pipe>:$pipes
  );
  let regions = (region SizedRegion<1>:$thenRegion);
  let description = [{
    Python API `ttl.if_pipe_dst(pipes, pipe_dst)` maps to this operation.
    The region contains TTL operations that execute if the current core matches
    any pipe's destination coordinates (for unicast) or is within the destination
    range (for multicast). For multiple pipes, the compiler generates a loop
    or nested conditionals to check each pipe.
    
    This operation is composable - the region can contain any TTL operations
    including nested conditionals, DMA operations, CB operations, etc.
    
    Lowers to scf.if with ttl.core comparisons against pipe destination coordinates.
    The lowering pass expands multiple pipes into nested conditionals or loops.
  }];
}

def TTL_WaitOp : TTL_Op<"wait"> {
  let summary = "Wait for DMA transfer (lowers to global barrier)";
  let arguments = (ins TTL_MemoryTransaction:$tx);
  let description = [{
    Explicit wait on transaction handle. Lowers to `ttkernel.noc_async_read_barrier`
    or `ttkernel.noc_async_write_barrier` (global barriers, not per-transaction).
    See: `tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td` definitions
    `TTKernel_NocAsyncReadBarrierOp` and `TTKernel_NocAsyncWriteBarrierOp`.
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

### 4.7 Utility Operations

```tablegen
def TTL_CoreOp : TTL_Op<"core", [Pure]> {
  let summary = "Get current core coordinates";
  let arguments = (ins OptionalAttr<I64Attr>:$dims);
  let results = (outs Variadic<Index>:$coordinates);
  let description = [{
    Returns core coordinates in grid. Folds to constants enabling
    later canonicalization and dead code elimination.
    
    Note: Python API `ttl.core()` maps to this MLIR operation.
  }];
}

def TTL_GridSizeOp : TTL_Op<"grid_size", [Pure]> {
  let summary = "Get grid dimensions";
  let arguments = (ins OptionalAttr<I64Attr>:$dims);
  let results = (outs Variadic<Index>:$sizes);
  let description = [{
    Returns grid dimensions. Folds to constants.

    Note: Python API `ttl.grid_size()` maps to this MLIR operation.
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
      linear_idx = ttl.core_linear()
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
def TTLSynchronizationInterface : OpInterface<"TTLSynchronization"> {
  let description = [{
    Interface for operations that provide synchronization barriers.
    Enables scheduling passes to understand ordering constraints.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the barrier type (DMA read, DMA write, CB, semaphore)",
      "TTL::BarrierType", "getBarrierType"
    >,
    InterfaceMethod<
      "Returns true if this is a global barrier affecting all operations",
      "bool", "isGlobalBarrier"
    >,
  ];
}
```

**Operations implementing TTLSynchronizationInterface:**
- `TTL_WaitOp`: DMA barrier (read or write based on transaction type)
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

3. **SSA Form**: TTL operations produce SSA values (block references,
   transaction handles, DST registers) that can be tracked by standard MLIR
   liveness analysis.

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
  │  ├─ CB protocol validation: wait/pop/reserve/push pairing
  │  ├─ Thread operation restrictions: compute vs datamovement
  │  ├─ Type compatibility checks: CB shapes, pipe connectivity
  │  └─ Resource usage validation: L1 capacity, DST register limits
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
  └─ TTLAllocateCircularBuffers - Assign L1 addresses/indices (liveness-based)
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
  ├─ TTLLowerCompute - ttl.block_add → affine.for + ttkernel.add_tiles
  ├─ TTLLowerDataMovement - ttl.copy → ttkernel.noc_async_*
  └─ TTLLowerSynchronization - ttl.cb_* → ttkernel.cb_*
  ↓
ttkernel.* operations
  ↓
[Phase 7: C++ Code Generation]
  └─ ConvertTTKernelToEmitC - Generate C++ kernel source
  ↓
C++ Kernel Source (.cpp/.h files)
  ↓
[Compilation]
  └─ Standard C++ compiler (g++/clang++)
  ↓
Compiled Kernel Object (.o files, linkable with TT-Metal runtime)
```

### 5.2 Key Pass Descriptions

**TTLInsertSynchronization**
- **Input**: `ttl.kernel` with thread regions (tensor operands)
- **Analysis**: Build producer-consumer DAG for blocks, semaphores, pipes
- **Transform**: Insert `ttl.dma_barrier` where needed, validate CB usage
  patterns
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

**TTLAssignDSTRegisters**
- **Input**: `ttl.kernel` with compute operations and DST hints.
- **Analysis**: Liveness analysis to determine register lifetime ranges.
- **Transform**: Allocate 4/8/16 (depending on dtype and config) DST register
  slots using simple graph coloring or linear scan.
- **Output**: Compute operations annotated with DST register indices.
- **Allocation Strategy**:
  - Liveness-based allocation minimizes register pressure.
  - First-fit algorithm assigns registers to values based on lifetime.
  - When register capacity exceeded: compile-time error (spill to L1 not
    supported in MVP)
  - Future: Spill strategy for complex kernels that exceed DST capacity.

**TTLLowerCompute**
- **Input**: `ttl.block_add` operations
- **Transform**: Generate `affine.for` iterating over tiles, insert
  `ttkernel.add_tiles_init` and `ttkernel.add_tiles`
- **Output**: TTKernel operations with explicit tile iteration

**TTLLowerDataMovement**
- **Input**: `ttl.copy` operations
- **Transform**:
  - Tensor → CB: `ttkernel.noc_async_read`
  - CB → Tensor: `ttkernel.noc_async_write`
  - CB → Pipe → CB: `ttkernel.noc_async_write_multicast*` or unicast
- **Output**: TTKernel NOC operations with computed addresses

### 5.3 Granularity and Block Shapes

**Concept**: Granularity defines how many tiles are grouped into blocks for
transfer and processing.

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

**Goal**: Maintain Python source locations throughout compilation for debugging
and IDE integration.

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
// Example: Adding location attributes to TTL_KernelOp
// (modify existing definition near line 258)
def TTL_KernelOp : TTL_Op<"kernel", [IsolatedFromAbove]> {
  let summary = "Kernel with multiple threads on grid";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    TTL_GridAttr:$grid,
    StrAttr:$memory_space,
    BoolAttr:$tiled,
    OptionalAttr<StrAttr>:$python_source_file,
    OptionalAttr<I64Attr>:$python_source_line
  );
  let regions = (region VariadicRegion<SizedRegion<1>>:$threads);
  let results = (outs Variadic<AnyType>:$results);
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

**See**: MLIR
[Location documentation](https://mlir.llvm.org/docs/Diagnostics/#source-locations)

### 5.6 Error Handling and Diagnostics

**Compile-time Error Messages:**

TTL passes emit errors with source location information pointing back to Python
source:

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

**Error Categories:**

1. **Validation Errors**: CB protocol violations, thread operation restrictions
   - Emitted by TTLValidatePass
   - Include source location from Python AST

2. **Resource Errors**: L1 capacity exceeded, DST register overflow
   - Emitted by TTLAllocateCircularBuffers and TTLAssignDSTRegisters
   - Provide suggestions for reducing resource usage

3. **Type Errors**: Incompatible CB shapes, pipe connectivity issues
   - Emitted during type checking passes
   - Show expected vs actual types

4. **Lowering Errors**: Operations that cannot be lowered to TTKernel
   - Emitted by lowering passes
   - Suggest alternative operations or patterns

**Runtime Error Handling:**

- Runtime errors are handled by TTNN runtime
- TTL-generated kernels include error checking where possible
- Debugging support via source location mapping (future)

**Debugging Support:**

- Source location tracking enables IDE jump-to-definition
- Profiler can attribute performance to Python source lines
- Debugger can map compiled code back to source (future)

### 5.5 Control Flow: SCF vs Affine Dialect

**Decision**: Use **Affine dialect** for loop nests, SCF for conditionals.

**Rationale**: TTL kernels have primarily **regular, statically-bounded loops**:

```python
# Typical TTL loops - all affine!
for rt_block in range(row_tiles // granularity):      # Static bound
    for ct in range(start_col, end_col):              # Affine bounds
        row_slice = slice(rt_block * granularity, ...)  # Affine indexing
```

**Affine dialect advantages for TTL:**

1. **Precise dependence analysis**: Critical for DMA/compute scheduling
   - Prove which memory operations can overlap
   - Detect when barriers are unnecessary
   - Enable DMA prefetching

2. **Loop optimization**: Standard polyhedral transformations
   - Loop fusion (merge adjacent loops)
   - Loop tiling (for better locality)
   - Loop interchange (optimize access patterns)

3. **Parallelization**: Detect parallel dimensions automatically
   - Mark which loops iterate over grid dimensions
   - Identify reduction loops vs parallel loops

4. **Memory access analysis**: Understand access patterns
   - Stride detection for DMA sizing
   - Locality analysis for CB allocation
   - Affine maps for multi-dimensional indexing

**When SCF is needed:**

Only for **conditionals** and **irregular patterns**:

```mlir
// Conditionals use SCF
%core_num = ttl.core(dims = 1)
scf.if %is_core_zero {
  // Core-specific work
}

// Affine for regular loops
affine.for %i = 0 to %N {
  affine.for %j = 0 to %M {
    %idx = affine.apply affine_map<(i,j) -> (i*M + j)>(%i, %j)
    ttl.copy %accessor[%i, %j], %cb
  }
}
```

**Comparison for TTL:**

| Aspect | SCF | Affine | TTL Need |
|--------|-----|--------|----------|
| **Loop bounds** | Dynamic OK | Must be affine | Most TTL loops are static/affine ✓ |
| **Indexing** | Arbitrary | Affine expressions | TTL uses affine indexing ✓ |
| **Dependence analysis** | Limited | Precise | Critical for DMA scheduling ✓ |
| **Loop opts** | Basic | Polyhedral | Useful for performance ✓ |
| **Conditionals** | Native | Not supported | Need SCF for if/else ✓ |
| **Implementation** | Simple | Complex | Affine worth the cost for TTL ✓ |


1. **MVP (Phase 1)**: Start with **Affine** directly
   - TTLDialectCompiler generates Affine for regular loops from Python AST
   - Falls back to SCF only for conditionals and non-affine patterns
   - Higher initial implementation cost but better long-term foundation

2. **Phase 2**: **Optimization and transforms**
   - Leverage upstream Affine transforms immediately
   - Add TTL-specific scheduling transforms that compose with Affine
   - Enable polyhedral optimizations from the start

**Code generation:**

```python
# MVP: Generate Affine directly
for i in range(N):
    ...
→ affine.for %i = 0 to %N { ... }

# Conditionals fall back to SCF
if condition:
    ...
→ scf.if %condition { ... }
```

**Pass pipeline with Affine:**

```
[Phase 2: Analysis with Affine]
  ├─ AffineLoopFusion - Merge adjacent loops
  ├─ AffineDataCopyGeneration - Optimize data movement
  ├─ AffineDependenceAnalysis - Inform scheduling
  └─ TTLSchedulePipeline - Custom scheduling with affine dependence info
```

**Upstream Transform support for Affine:**

MLIR's Transform dialect has **interface-based** operations that work with
Affine loops:
- `transform.loop.tile` - Tiles `affine.for` loops (via `LoopLikeInterface`)
- `transform.loop.unroll` - Unrolls affine loops
- `transform.loop.coalesce` - Coalesces nested affine loops
- `transform.affine.simplify_bounded_affine_ops` - Simplifies affine ops with
  known bounds
- `transform.affine.simplify_min_max_affine_ops` - Reduces min/max operations

**Key advantage**: TTL's custom `transform.ttl.*` operations can **compose with
upstream transforms**:

```mlir
transform.sequence {
  %loops = transform.structured.match ops{["affine.for"]}

  // Upstream: tile affine loops
  %tiled = transform.loop.tile %loops tile_sizes=[32, 32]

  // Custom: schedule TTL operations within tiles
  %sched = transform.ttl.schedule_pipeline %tiled

  // Upstream: simplify resulting affine expressions
  transform.affine.simplify_bounded_affine_ops %sched
}
```

**Benefits for TTL:**
- **Better scheduling**: Precise dependence info enables optimal DMA/compute
  overlap
- **Automatic optimizations**: Fusion, tiling via upstream + custom transforms
- **Composability**: Mix upstream affine transforms with TTL-specific scheduling
- **Future-proof**: Foundation for advanced loop transformations

**Trade-off accepted:**
- Higher MVP implementation cost for affine generation
- Worth it for long-term performance, analysis quality, and transform
  composability

**Decision**: Start with Affine directly (MVP), use SCF only for conditionals,
leverage upstream transforms immediately.

**See**: [Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/),
[Transform Tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)



## 6. Type Conversion & Lowering Examples

### 6.1 TTL → TTKernel Type Mapping

| TTL Type | TTKernel Type | Conversion Notes |
|----------|---------------|------------------|
| `!ttl.cb<[2,1], !ttcore.tile<32x32,f32>, 2, L1>` | `!ttkernel.cb<4, !ttcore.tile<32x32,f32>>` | Total tiles = 2×1×2 = 4; memory space drives L1 address assignment in TTLAllocateCircularBuffers pass; buffer_factor used to compute total tiles but not directly preserved in TTKernel CB type |
| `!ttl.block<tensor<2x1x!ttcore.tile>>` | Elided | Blocks decomposed into tile operations during lowering |
| `!ttl.mem_tx` | N/A (elided) | Lowers to global barriers |
| `!ttl.semaphore` | `!ttkernel.semaphore` | Direct mapping |
| `!ttl.pipe` | Attributes on ops | Pipe decomposed into core coords + multicast flags |
| `!ttl.accessor<layout, memspace>` | N/A (resolved) | Resolves to NOC addresses + layout info during lowering |

### 6.2 TTL → TTKernel Operation Mapping

| TTL Operation | TTKernel Operations | Implementation Complexity | MVP Priority |
|---------------|---------------------|---------------------------|--------------|
| **Circular Buffer** ||||
| `ttl.cb_wait` | `ttkernel.cb_wait_front` | Low | **HIGH** |
| `ttl.cb_pop` | `ttkernel.cb_pop_front` | Low | **HIGH** |
| `ttl.cb_reserve` | `ttkernel.cb_reserve_back` | Low | **HIGH** |
| `ttl.cb_push` | `ttkernel.cb_push_back` | Low | **HIGH** |
| `ttl.get_tile` | `ttkernel.copy_tile_init`, `ttkernel.copy_tile` | Medium | **HIGH** |
| `ttl.pack_tile` | `ttkernel.pack_tile` | Low | **HIGH** |
| **Compute (FPU)** ||||
| `ttl.block_add` | `ttkernel.add_tiles_init`, `ttkernel.add_tiles` | Medium | **HIGH** |
| `ttl.block_sub` | `ttkernel.sub_tiles_init`, `ttkernel.sub_tiles` | Medium | **HIGH** |
| `ttl.block_mul` | `ttkernel.mul_tiles_init`, `ttkernel.mul_tiles` | Medium | **HIGH** |
| `ttl.block_matmul` | `ttkernel.mm_init`, `ttkernel.matmul_tiles` | Medium | **HIGH** |
| **Compute (SFPU - Phase 2)** ||||
| `ttl.block_exp` | `ttkernel.init_sfpu`, `ttkernel.exp_tile` | Medium | MEDIUM |
| `ttl.block_log` | `ttkernel.init_sfpu`, `ttkernel.log_tile` | Medium | MEDIUM |
| `ttl.block_sqrt` | `ttkernel.init_sfpu`, `ttkernel.sqrt_tile` | Medium | MEDIUM |
| `ttl.block_relu` | `ttkernel.init_sfpu`, `ttkernel.relu_tile` | Medium | MEDIUM |
| `ttl.block_gelu` | `ttkernel.init_sfpu`, `ttkernel.gelu_tile` | Medium | MEDIUM |
| **Data Movement** ||||
| `ttl.copy` (DRAM→CB) | `ttkernel.get_noc_addr`, `ttkernel.noc_async_read` | High | **HIGH** |
| `ttl.copy` (CB→DRAM) | `ttkernel.get_noc_addr`, `ttkernel.noc_async_write` | High | **HIGH** |
| `ttl.copy` (CB→Pipe→CB) | `ttkernel.noc_async_write_multicast` or unicast | High | MEDIUM |
| `ttl.wait` | `ttkernel.noc_async_read_barrier` or `write_barrier` | Low | **HIGH** |
| `ttl.dma_barrier` | `ttkernel.noc_async_read_barrier`, `write_barrier` | Low | **HIGH** |
| **Synchronization** ||||
| `ttl.semaphore_wait` | `ttkernel.noc_semaphore_wait` | Medium | MEDIUM |
| `ttl.semaphore_set` | `ttkernel.noc_semaphore_set` (or multicast variant) | Medium | MEDIUM |
| `ttl.semaphore_inc` | `ttkernel.noc_semaphore_inc` | Medium | MEDIUM |
| **Tile Management** ||||
| `ttl.require_dst` | `ttkernel.tile_regs_acquire`, `ttkernel.tile_regs_commit` | Low | **HIGH** |
| **Total Operations for MVP** | **~15-20 TTL ops → ~25-30 TTKernel ops** | | **Week 2-3** |
| **Full Coverage (Future)** | **~40-50 TTL ops → ~120+ TTKernel ops** | | **Incremental** |

**Implementation Notes:**

- **MVP (5-10 ops)**: Focus on arithmetic (add/sub/mul/matmul) + CB + DMA
- **Phase 2 (10-15 ops)**: Add common SFPU (exp/log/sqrt/relu/gelu)
- **Phase 3+**: Add remaining ops as needed (60+ SFPU variants)
- **Lowering Pattern**: Once established for first few ops, adding more is
  straightforward

### 6.3 Operation Lowering Examples

**Circular Buffer Operations:**
```mlir
// TTL
ttl.cb_wait %cb, 2

// TTKernel
ttkernel.cb_wait_front %cb, 2
```

**Tile Extraction:**
```mlir
// TTL
%tile0 = ttl.get_tile %cb, %c0
%tile1 = ttl.get_tile %cb, %c1

// TTKernel (copy from CB to DST register)
ttkernel.copy_tile_init %cb
%tile0 = ttkernel.copy_tile %cb, 0, %dst_idx0
%tile1 = ttkernel.copy_tile %cb, 1, %dst_idx1
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
affine.for %i = 0 to 2 {
  ttkernel.copy_tile_init %a_cb
  %a_tile = ttkernel.copy_tile %a_cb, %i, %dst_idx_a
  ttkernel.copy_tile_init %b_cb
  %b_tile = ttkernel.copy_tile %b_cb, %i, %dst_idx_b
  ttkernel.add_tiles_init
  ttkernel.add_tiles %a_tile, %b_tile, %dst_idx_result
  ttkernel.pack_tile %dst_idx_result, %out_cb, %i
}
ttkernel.tile_regs_commit
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

**Block Operations:**

```python
@syntax("!tensor")
class TensorBlock:
    def __add__(ast_self, rhs):
        # Generate ttl.block_add instead of linalg.generic
        return ttl.block_add(ast_self, rhs)
    
    def __sub__(ast_self, rhs):
        return ttl.block_sub(ast_self, rhs)
    
    def __mul__(ast_self, rhs):
        return ttl.block_mul(ast_self, rhs)
    
    def __truediv__(ast_self, rhs):
        return ttl.block_div(ast_self, rhs)
    
    def __pow__(ast_self, rhs):
        return ttl.block_pow(ast_self, rhs)

    def __matmul__(ast_self, rhs):
        return ttl.block_matmul(ast_self, rhs)

    def store(ast_self, value):
        # Python API: block.store(value)
        # Maps to ttl.block_store operation
        return ttl.block_store(ast_self, value)
```

**Math Functions:**

```python
# ttl.math.* functions map to TTL math operations
@syntax("math")
class MathFunctions:
    @staticmethod
    def sqrt(x):
        return ttl.math.sqrt(x)
    
    @staticmethod
    def exp(x):
        return ttl.math.exp(x)
    
    # ... other math functions
```

**Circular Buffer Operations:**

```python
class CircularBuffer:
    def wait(self):
        # Python API: cb.wait() or with cb.wait() as blk:
        num_tiles = self.get_num_tiles()
        return ttl.cb_wait(self.handle, num_tiles)

    def reserve(self):
        # Python API: cb.reserve() or with cb.reserve() as blk:
        # AST compiler handles 'with' statement:
        #   - Generates ttl.cb_reserve at scope start
        #   - Generates ttl.cb_push at scope end
        num_tiles = self.get_num_tiles()
        return ttl.cb_reserve(self.handle, num_tiles)

    def pop(self):
        num_tiles = self.get_num_tiles()
        return ttl.cb_pop(self.handle, num_tiles)
    
    def push(self):
        num_tiles = self.get_num_tiles()
        return ttl.cb_push(self.handle, num_tiles)
```

**Data Movement:**

```python
@syntax("dma")
def copy(src, dst, **kwargs):
    # Python API: ttl.copy(src, dst)
    # Returns transfer handle object with wait() method
    # Transfer handle.wait() maps to ttl.wait %tx
    return ttl.copy(src, dst, **kwargs)
```

**Pipe Conditionals:**

```python
def if_pipe_src(pipes, callback):
    # Python API: ttl.if_pipe_src(pipes, pipe_src)
    # Maps to ttl.if_pipe_src operation
    return ttl.if_pipe_src(pipes, callback)

def if_pipe_dst(pipes, callback):
    # Python API: ttl.if_pipe_dst(pipes, pipe_dst)
    # Maps to ttl.if_pipe_dst operation
    return ttl.if_pipe_dst(pipes, callback)
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



## 8. Implementation Roadmap

### Phase 1: Foundation (Week 1)
**Goal**: TTL dialect compiles and registers with MLIR

1. Dialect Definition (C++)
   - Create `include/ttlang/Dialect/TTL/` and `lib/Dialect/TTL/`
   - Define types (TTLTypes.td): CB (with optional buffer_index, page_size,
     core_ranges), MemTx, Semaphore, Pipe, Accessor
   - Define ops (TTLOps.td): structural (with block_factors, compile_time_args,
     compute_config), CB, compute, DM, sync, utility
   - Define attributes (TTLOpsAttrs.td): MemorySpace, Grid, Layout (define
     concrete parameters)
   - Implement builders, printers, parsers

2. Interface Implementations (HIGH PRIORITY)
   - Implement `MemoryEffectsOpInterface` for all CB, DMA, and synchronization
     ops
   - Define `TTLSynchronizationInterface` for barrier operations
   - Implement `bufferization::TensorLikeType` for block tensor types

3. CMake Integration
   - Update `include/ttlang/CMakeLists.txt` and `lib/CMakeLists.txt`

4. Basic Testing
   - Create `test/ttmlir/Dialect/TTL/` with lit tests

**Deliverable**: TTL dialect loads, ops parse/print correctly with proper
interfaces

### Phase 2: Python Compilation (Week 1)
**Goal**: Python AST generates valid TTL operations

1. TTL Compiler
   - Implement `TTLDialectCompiler` in `python/ttlang/_src/ttl_ast.py`
   - Handle `@compute()` and `@datamovement()` decorators

2. Operator Updates
   - Update `python/ttlang/operators.py` for TTL ops

3. Python Bindings
   - Create `python/ttlang/dialects/ttl.py` with nanobind

**Deliverable**: Python examples compile to TTL IR

### Phase 3: Core Passes (Week 2)
**Goal**: Validation, thread expansion, and basic lowering

1. TTLValidatePass
   - Verify CB/pipe/semaphore contracts
   - Validate store/wait/pop pattern sequencing
   - Check thread operation restrictions (compute vs datamovement)

2. TTLExpandThreads - Extract threads to separate functions (can happen early)

3. TTLLowerCompute - Block ops → TTKernel tile ops
   - **MVP operations** (must be implemented):
     - `ttl.block_add` → `add_tiles_init` + `add_tiles`
     - `ttl.block_mul` → `mul_tiles_init` + `mul_tiles`
     - `ttl.block_matmul` → `mm_init` + `matmul_tiles`
     - `ttl.block_reduce_sum` → `reduce_init` + `reduce_tile` + `reduce_uninit`
     - `ttl.block_bcast` → `unary_bcast_init` + `unary_bcast`
   - Generate affine loops over tiles
   - Handle compute_region fusion

4. TTLLowerSynchronization - CB ops → TTKernel CB ops

**Deliverable**: Kernels with reduction/broadcast/fusion lower to TTKernel (and
C++)

### Phase 4: Memory & Scheduling (Week 3)
**Goal**: Resource allocation and optimization

1. TTLBufferizePass - Tensor → memref conversion
   - Use One-Shot Bufferization with BufferizableOpInterface implementations
   - Leverage upstream infrastructure

2. Liveness Analysis Integration
   - Ensure TTL operations implement proper interfaces for MLIR's built-in
     liveness analysis
   - Use `mlir::Liveness` utility for computing live ranges
   - No custom liveness pass needed - use upstream infrastructure

3. TTLAllocateCircularBuffers - L1 address assignment
   - Query liveness information via `mlir::Liveness`
   - Implement graph coloring (replace first-fit for better fragmentation)
   - Use liveness data to minimize L1 footprint

4. TTLAssignDSTRegisters - DST register allocation
   - Query compute_config for capacity (4-16 tiles depending on configuration)
   - Use `mlir::Liveness` for register allocation
   - Insert spill/restore operations when capacity exceeded
   - Linear scan or graph coloring algorithm

5. TTLInsertSynchronization - Barrier inference
   - Use dataflow analysis to compute minimal barrier placement
   - Validate synchronization correctness

**Deliverable**: Full resource allocation working with liveness-based
optimization using MLIR built-in infrastructure

### Phase 5: Data Movement & Integration (Week 4)
**Goal**: Complete lowering and end-to-end validation of simple kernels

1. TTLLowerDataMovement - DMA operations → TTKernel NOC ops
2. End-to-end testing: Python → TTL → TTKernel → C++
3. Port `examples/eltwise_add.py` to TTL
4. Verify generated C++

**Deliverable**: Working end-to-end TTL pipeline



## 9. TTNN Runtime Integration

### Runtime Architecture

TTL supports two runtime integration paths:

1. **TTNN dylib workflow** (primary): TTL-generated C++ compiles to shared
   libraries (.so files) that load dynamically into the TTNN runtime. This is
   the primary path for TTNN integration and prefered for iterative development.

2. **TT-Metal flatbuffer workflow** (alternative): TTL can also generate
   flatbuffer binaries for TT-Metal runtime, similar to the existing D2M
   pipeline. This path is available for compatibility with TT-Metal workflows
   and for generating production binaries.

The dylib workflow is described in detail below. The flatbuffer workflow follows
the same compilation pipeline but uses flatbuffer generation instead of emitting
C++ and relying shared library loading.

### Execution Flow (Dylib Path)

```
TTKernel → EmitC → C++ Source → C++ Compiler → Shared Library (.so) → Dynamic Loading → TTNN Runtime → Hardware
```

**Key components:**
1. EmitC code generation (existing `ConvertTTKernelToEmitC` pass)
2. TTNN standalone build system compiles C++ to .so
3. Runtime uses `dlopen()` to load shared library
4. Function dispatch via name mangling and `dlsym()`
5. Execution on TTNN runtime with ttnn::Tensor types

### Dylib Interface Requirements

**Generated C++ must provide:**

```cpp
// Main kernel function (mangled name)
std::vector<ttnn::Tensor> kernel_name(std::vector<ttnn::Tensor> inputs);

// Device context setter
void setDevice(ttnn::MeshDevice* device);

// Optional: Input creation helper
std::vector<ttnn::Tensor> create_inputs_for_kernel_name(ttnn::MeshDevice& device);
```

### Compilation Pipeline

**TTL-specific EmitC backend pipeline** (similar to TTNN); note that this will
be implemented in python, but we are showing the CLI commands here for clarity.

```bash
# 1. TTL → TTKernel (TTL passes)
ttlang-opt --ttl-lower-to-ttkernel input.mlir -o ttkernel.mlir

# 2. TTKernel → EmitC (existing pass with dylib mode)
ttlang-opt --ttkernel-to-emitc-pipeline="target-dylib=true" ttkernel.mlir -o emitc.mlir

# 3. EmitC → C++
ttlang-translate --mlir-to-cpp emitc.mlir > generated.cpp

# 4. C++ → Shared Library
cd tools/ttnn-standalone
python ci_compile_dylib.py --file generated.cpp --mode dylib
```

**Dylib-specific transformations:**
- Input tuplification (forced even for empty inputs)
- Tensor argument wrapping for ABI compatibility
- Device context management functions
- RPATH configuration for runtime library discovery

### Runtime Loading and Execution

The kernels are loaded using code similar to the following (which can also be
generated by the compiler):

```cpp
// Runtime execution (tt-mlir pattern)
void* so = dlopen("ttl_kernel.so", RTLD_LAZY);

// Get setDevice function
auto setDeviceFunc = reinterpret_cast<void(*)(ttnn::MeshDevice*)>(
    dlsym(so, "setDevice")
);
setDeviceFunc(&meshDevice);

// Get kernel function (with mangled name)
std::string mangledName = getMangledName("kernel_name");
auto kernelFunc = reinterpret_cast<
    std::vector<ttnn::Tensor>(*)(std::vector<ttnn::Tensor>)
>(dlsym(so, mangledName.c_str()));

// Execute
std::vector<ttnn::Tensor> outputs = kernelFunc(ttnnInputs);

dlclose(so);
```

### Build System Integration

**CMakeLists.txt configuration** (reuse TTNN standalone):
- Link against: `tt_metal`, `device`, `tt_stl`, `_ttnncpp.so`
- Compiler flags: `-march=x86-64-v3` for performance
- RPATH: `$ORIGIN:${METAL_LIB_DIR}` for library discovery

### TTL-Specific Considerations

1. **Circular Buffer Metadata**: Include CB configurations in generated C++
   - L1 addresses from TTLAllocateCircularBuffers pass
   - Buffer sizes and tile counts

2. **Thread Mapping**: Map TTL threads to TT-Metal program structure
   - Compute thread → Math kernel
   - Datamovement threads → NOC kernels

3. **Synchronization**: Emit barrier/semaphore operations
   - DMA barriers from TTLInsertSynchronization
   - Semaphore setup for inter-core communication

### MVP Integration (Week 2-3)

Phase 3 deliverable:
- Reuse `ConvertTTKernelToEmitC` with dylib target
- Generate C++ matching TTNN dylib interface requirements
- Build using ttnn-standalone build system
- Test loading and execution via runtime API

### Validation

```python
# Test TTL kernel via Python runtime API
import ttnn
import torch
from ttrt.runtime import Device

device = Device()
so = load_dylib("ttl_matmul.so")

# TTNN tensors as inputs
a = ttnn.from_torch(torch.randn(128, 128), device=device)
b = ttnn.from_torch(torch.randn(128, 128), device=device)

# Execute TTL kernel
result = run_dylib_function(so, "matmul", [a, b], device)

# Verify correctness
expected = torch.matmul(a.to_torch(), b.to_torch())
assert torch.allclose(result.to_torch(), expected, rtol=1e-2)
```

### Future Enhancements

- Flatbuffer generation (alternative to dylib for deployment)
- Performance comparison: TTL vs D2M pipeline
- Debugging: Source mapping, kernel profiling
- Distributed execution support



## 10. Future Evolution

### 10.1 Microcore Model (Post-MVP)

Once the simple threading model is validated, evolve to parametric microcore
abstraction:

```tablegen
// Future: TTL_MicrocoreConfigAttr definition TBD
// def TTL_MicrocoreConfigAttr : AttrDef<TTL_Dialect, "MicrocoreConfig"> { ... }

def TTL_TileProcOp : TTL_Op<"tile_proc"> {
  let summary = "Hardware execution unit with declared microcores";
  let arguments = (ins TTL_MicrocoreConfigAttr:$microcores);  // Future attribute
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

### 10.2 C++ Backend (Post-TTKernel)

After TTL → TTKernel is stable, add direct C++ emission:

```
TTL → TTKernel → EmitC → C++ (current)
TTL → EmitC → C++ (future)
```

Benefits:
- Standalone kernels (no TTKernel dependency)
- Potential for custom optimizations
- Better debugging (direct C++ inspection)

### 10.3 Distributed Tensor Type (Major Extension)

**Goal**: Enable programming of larger distributed systems with explicit tensor
distribution across cores.

**Problem**: Current TTL uses implicit SPMD model (all cores run same program
with implicit sharding). For complex distributed algorithms, need explicit
control over:
- How tensors are partitioned across cores
- Different programs on different core subsets
- Explicit redistribution strategies
- Per-core shard shapes

**Proposal**: `!ttl.dist_tensor` type implementing
`bufferization::TensorLikeType`

```tablegen
def TTL_DistributedTensor : TTL_Type<"DistributedTensor", "dist_tensor"> {
  let summary = "Distributed tensor across grid of cores";
  let parameters = (ins
    "ArrayRef<int64_t>":$gridShape,      // Distribution grid [8, 8]
    "ArrayRef<int64_t>":$shardShape,     // Per-core shard [32, 32]
    "Type":$elementType,                  // f32, bf16, etc.
    "TTL::DistributionStrategyAttr":$strategy, // Sharded, interleaved, etc.
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
- **Bufferizes to plain memrefs**: Each core gets standard
  `memref<shard_shape, memspace>`
- **Metadata tracking**: Transient `ttcore.shard_group<tensor_id, shard_idx>`
  attribute for DMA planning
- **Reuses MLIR bufferization**: Plugs into One-Shot Bufferization
  infrastructure

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
1. **Phase 1 (MVP)**: TTL w/o sharding (current plan)
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
- Enable complex multi-core algorithms (e.g., distributed attention, pipeline
  parallelism)

### 10.4 Transform Dialect Integration for Scheduling

**Goal**: Use MLIR's Transform dialect for composable scheduling and
optimization instead of monolithic optimization passes.

**Motivation**: Transform dialect provides:
- **Composability**: Build complex schedules from simple transform operations
- **Precise targeting**: Apply transformations to specific operations via
  handles
- **Debugging**: Inspect intermediate IR after each transformation
- **Reusability**: Define scheduling strategies as transform sequences
- **DSL alignment**: Map TT-lang scheduling concepts directly to transform ops

**Scope**: Use Transform dialect for scheduling/optimization; keep traditional
passes for lowering (TTL → TTKernel).

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
4. **Autotuning**: Search space of scheduling strategies by varying transform
   sequences
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
- `transform.ttl.allocate_dst_registers` - DST register allocation
- `transform.ttl.schedule_pipeline` - Compute/DMA overlap
- `transform.ttl.optimize_buffer_factor` - CB sizing
- `transform.ttl.coalesce_dma` - DMA merging
- `transform.ttl.reorder_for_locality` - Memory access optimization
- `transform.ttl.insert_prefetch` - Prefetch insertion

**Upstream transforms to leverage:**
- `transform.loop.tile` - Tile affine loops for locality
- `transform.loop.unroll` - Unroll small affine loops
- `transform.affine.simplify_bounded_affine_ops` - Simplify after
  transformations
- `transform.apply_registered_pass` - Run standard affine passes

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
- **Phase 3**: Migrate optimization logic to transform sequences, keep lowering
  as passes

This hybrid approach gets TTL working quickly while building toward the more
composable Transform dialect model for scheduling.



## 11. Success Criteria

1. **TTL dialect compiles** and is registered in MLIR
2. **Python AST → TTL** generates valid TTL operations
3. **All passes execute** without crashing on examples
4. **TTL → TTKernel** produces correct IR (verified vs current pipeline)
5. **End-to-end test** (Python → C++) works for eltwise_add
6. **Performance parity** with current d2m.generic approach
7. **No D2M dependencies** in the TTL path



## 12. Appendix: Design Rationale

### Why Tensor-Level (Not Tile-Level)?
- Matches Python DSL semantics (users think in blocks)
- Enables optimizations before committing to tiles
- Lowering passes make tile iteration explicit when needed

### Why Simple Threading (Not Microcore)?
- Faster MVP, matches current examples
- Clear evolution path documented
- Can add microcore abstraction later without breaking existing code

### Why Direct to TTKernel (Not via D2M)?
- Avoids `d2m.generic` complexity
- Cleaner semantic representation
- Still compatible with existing TTKernel → EmitC → C++ pipeline

### Why Affine for Control Flow?
- Precise dependence analysis critical for DMA/compute scheduling
- Enables polyhedral optimizations (fusion, tiling, interchange); builtin
  `transform` dialect support
- Better alignment with TTL's regular loop patterns
- SCF used only for conditionals and non-affine patterns
- Reuse proven MLIR infrastructure for both dialects

### Why Memory Space in Types?
- Makes IR self-descriptive
- Enables early validation of illegal transfers
- Simplifies lowering (memspace dictates NOC operations)



## 13. References

- **TT-lang Spec**: `docs/TT-lang.md`
- **Build System**: `docs/BUILD_SYSTEM.md`
- **Testing Guide**: `test/TESTING.md`
- **Current D2M Pipeline**: `python/ttlang/d2m_api.py`
- **TTKernel Dialect**: `../tt-mlir/include/ttmlir/Dialect/TTKernel/IR/`
- **LLVM Upstream**: https://github.com/llvm/llvm-project/

