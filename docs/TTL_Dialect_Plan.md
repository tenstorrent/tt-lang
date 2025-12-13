# TTL Dialect Design Plan

---

## ⚠️ DEPRECATED - THIS DOCUMENT HAS BEEN SPLIT ⚠️

**This monolithic document has been split into modular documents for better organization and readability.**

**Please see the new modular documentation at: [docs/ttl/01_TTL_Dialect_Plan.md](ttl/01_TTL_Dialect_Plan.md)**

The new structure includes:
- **[01_TTL_Dialect_Plan.md](ttl/01_TTL_Dialect_Plan.md)** - Top-level overview
- **[02_TTL_Type_System.md](ttl/02_TTL_Type_System.md)** - Type system specification
- **[03_TTL_Compute_Operations.md](ttl/03_TTL_Compute_Operations.md)** - Compute operations and DST tiling
- **[04_TTL_Data_Movement_Operations.md](ttl/04_TTL_Data_Movement_Operations.md)** - Data movement and synchronization
- **[05_TTL_Compilation_Pipeline.md](ttl/05_TTL_Compilation_Pipeline.md)** - Compilation passes and lowering
- **[06_TTL_Implementation_and_Runtime.md](ttl/06_TTL_Implementation_and_Runtime.md)** - Integration and roadmap

**This file is kept for historical reference only and will not be updated.**

---

**Version**: 0.7

**Modified**: 2025-12-06

**Status**: Design Phase - Under Revision - DEPRECATED

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
   - 3.3 [TensorAccessor Support Strategy](#33-tensoraccessor-support-strategy)
4. [Operations](#4-operations)
   - 4.1 [Structural Operations](#41-structural-operations)
   - 4.2 [Resource Creation](#42-resource-creation)
   - 4.3 [Circular Buffer Operations](#43-circular-buffer-operations)
   - 4.4 [Compute Operations](#44-compute-operations)
     - 4.4.1 [Fusion Operations](#441-fusion-operations)
     - 4.4.2 [Arithmetic Operations](#442-arithmetic-operations)
     - 4.4.3 [Reduction and Broadcast](#443-reduction-and-broadcast-operations)
     - 4.4.4 [Materialization](#444-materialization-operations)
     - 4.4.5 [DST Register Management](#445-dst-register-management)
   - 4.5 [Data Movement Operations](#45-data-movement-operations)
   - 4.6 [Synchronization Operations](#46-synchronization-operations)
   - 4.7 [Utility Operations](#47-utility-operations)
   - 4.8 [Compute Operations Inventory](#48-compute-operations-inventory)
   - 4.9 [MLIR Interface Requirements](#49-mlir-interface-requirements)
5. [Compilation Pipeline](#5-compilation-pipeline)
   - 5.1 [Pass Architecture](#51-pass-architecture)
   - 5.2 [Key Pass Descriptions](#52-key-pass-descriptions)
   - 5.3 [Source Location Tracking](#53-source-location-tracking)
   - 5.4
     [Control Flow: SCF vs Affine Dialect](#54-control-flow-scf-vs-affine-dialect)
   - 5.6 [Error Handling and Diagnostics](#55-error-handling-and-diagnostics)
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
kernels execute on TTNN runtime via the `ttnn.generic_op` API, which accepts C++
kernel source code and metadata through Python descriptors. TTL is designed
specifically for the TT-lang DSL surface language, while D2M serves as a
lower-level dialect for data movement and compute operations.

**Key Design Decisions:**
- **Threading Model**: Simple compute/data movement threads (MVP)
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
- SSA semantics: `CB` operations with explicit SSA values rather than implicit
  state effects (D2M `CB` ops use `MemoryEffects<[MemRead, MemWrite]>` for state
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
4. C++ kernel generation: Generate C++ kernel source strings with metadata that
   are passed to `ttnn.generic_op` for runtime compilation and execution. This
   is the MVP target.
5. Future-proof: Extensible to new hardware generations via attributes

### Non-Goals (MVP)
- Autotuning algorithms (IR has hooks, algorithms come later)
- Single-threaded synchronous model (start multi-threaded)
- Complete TT-lang spec (start minimal, expand incrementally)
- Direct C++ backend (TTL→C++ without TTKernel): Post-MVP, see section 10.2
- Custom ConvertTTLToEmitC pass: Not needed, use existing ConvertTTKernelToEmitC
  from tt-mlir



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
Python Kernel → Python AST → TTL Dialect → TTL Passes → Kernel Descriptor Generation → ttnn.generic_op → TTNN Runtime → Hardware
                                    ↓                                ↓
                         Validation, Synchronization,     C++ Kernel Source Strings
                         Bufferization, Allocation,       + Compile-time Args
                         Register Assignment,             + Runtime Args
                         Optimization                     + CB Metadata
```

Key components:
- TTL passes perform all analysis and optimization at the IR level
- Kernel descriptor generator produces C++ source strings and metadata from TTL
  IR
- Python runtime constructs `ttnn.KernelDescriptor`, `ttnn.CBDescriptor`, and
  `ttnn.ProgramDescriptor` objects
- `ttnn.generic_op` handles kernel compilation and device execution
- No separate build step required - kernels provided as source strings to
  runtime

**Relationship to D2M Dialect:**

TTL and D2M serve different roles in the compilation pipeline:

- TTL: New frontend dialect specifically designed for the TT-lang DSL. TTL
  provides DSL-specific IR enabling TT-lang-aware transformations. TTL bypasses
  D2M and generates C++ kernel descriptors directly for `ttnn.generic_op`.

- D2M: Remains the primary dialect for framework paths (JAX/PyTorch → TTIR → D2M
  → TTKernel). D2M serves as a general-purpose data movement and compute
  abstraction.

- Convergence: Both TTL and D2M can target TTNN runtime, but through different
  mechanisms. TTL uses `ttnn.generic_op` with kernel descriptors, while D2M uses
  the traditional flatbuffer workflow.

This separation allows TTL to focus on TT-lang DSL semantics while D2M continues
to serve framework integration needs.



## 3. Type System

### 3.1 Core Types

```tablegen
// TTLBase.td - Base class definitions

def TTL_Dialect : Dialect {
  let name = "ttl";
  let cppNamespace = "::mlir::tt::ttl";
  let description = [{
    The TTL (TT-Lang) dialect provides a tensor-level IR for the TT-Lang DSL.
  }];
}

class TTL_Type<string name, string mnemonic>
  : TypeDef<TTL_Dialect, name, mnemonic> {}

class TTL_Op<string mnemonic, list<Trait> traits = []>
  : Op<TTL_Dialect, mnemonic, traits> {}

// Type constraints for TTL operations
def TTL_Tile : TTL_Type<"Tile", "tile"> {
  let summary = "Tile type for tiled tensor layouts";
  let parameters = (ins
    "int64_t":$height,    // Tile height (32, 16, 4, 2, 1)
    "int64_t":$width,     // Tile width (32)
    "Type":$elementType   // Element type (f32, f16, bf16)
  );
  let description = [{
    Represents a tile for tiled tensor layouts. Hardware supports variable tile heights
    (32×32, 16×32, 4×32, 2×32, 1×32) for different operation types.

    Standard tile: !ttl.tile<32x32, f32>
    SFPU optimized: !ttl.tile<16x32, f32>
    Narrow tile: !ttl.tile<4x32, f16>

    The tile dimensions must be hardware-supported combinations. Validation enforced
    during type construction by the type verifier.
  }];
}

def TTL_TensorEncodingAttr : TTL_Attr<"TensorEncoding", "tensor_encoding"> {
  let summary = "TTL tensor encoding combining memory space and layout";
  let parameters = (ins
    TTL_MemorySpaceAttr:$memorySpace,
    TTL_LayoutAttr:$layout
  );
  let assemblyFormat = "`<` $memorySpace `,` $layout `>`";
  let description = [{
    Encoding attached to `tensor<...>` types to carry both the memory-space
    placement (L1/DRAM/DST/System) and layout metadata (tiled, sharded,
    interleaved, etc.).
  }];
}

// TODO: We may need more information here to match the TTNN tensor type
// TODO: Use a TTL_ElementType once defined
def TTL_Tensor : TensorOf<[F32, F16, BF16, TTL_Tile]> {
  let summary = "TTL tensor with layout and memory space";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    TTL_TensorEncodingAttr:$encoding
  );
  let hasCustomAssemblyFormat = 1;
  let description = [{
    Tensors in TTL always include explicit layout and memory-space metadata via
    the `TTL_TensorEncodingAttr`, e.g.,
    `tensor<64x64xf32, #ttl.tensor_encoding<DRAM,
    #ttl.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>>>`.
  }];
}


// TTL Types (TTLTypes.td)

def TTL_CircularBuffer : TTL_Type<"CircularBuffer", "cb"> {
  let summary = "Circular buffer for producer-consumer communication";
  let parameters = (ins
    "ArrayRef<int64_t>":$shape,          // Elements per block
    "mlir::Type":$elementType,           // !ttcore.tile<32x32, f32> or scalar type
    "int64_t":$bufferFactor              // Number of blocks/slots
  );
  // Note: ArrayRef<int64_t> in assemblyFormat requires custom print/parse methods
  // let assemblyFormat = "`<` $shape `,` $elementType `,` $bufferFactor `>`";
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Calculate total elements for TTKernel CB conversion
    int64_t getTotalElements() const {
      return getElementsPerBlock() * getBufferFactor();
    }

    // Helper for lowering
    int64_t getElementsPerBlock() const {
      return std::accumulate(
        getShape().begin(), getShape().end(), 1, std::multiplies<int64_t>());
    }

    // Returns true if this CB operates on tiles (derived from elementType)
    bool isTiled() const {
      return getElementType().isa<ttcore::TileType>();
    }
  }];

  let description = [{
    Circular buffer supporting both tiled and row-major tensor layouts per TT-Lang spec.

    Shape units: The shape parameter is expressed in shape units derived from the source
    tensor's layout (tiles for tiled tensors, scalars for row-major tensors).

    Layout determined by elementType:
    - Tiled layout: elementType is !ttcore.tile<32x32, dtype>, shape in tiles
      Example: shape=[2,1], elementType=!ttcore.tile<32x32,f32> → 2 tiles per block

    - Row-major layout: elementType is scalar type (f32, bf16, etc.), shape in scalars
      Example: shape=[64,32], elementType=f32 → 64×32 scalars per block

    L1 memory allocation: Total L1 storage = block_size × buffer_factor
      - block_size = shape[0] × shape[1] × ... × sizeof(elementType)
      - buffer_factor determines number of blocks for producer-consumer pipelining
      - Example: shape=[2,1], tile<32x32,f32>, buffer_factor=2 → 2 tiles × 2 blocks = 4 tiles in L1

    The isTiled() method returns true if elementType is a tile type, false otherwise.
  }];
}

def TTL_Block : TTL_Type<"Block", "block"> {
  let summary = "Logical unit of data exchanged via circular buffers";
  let parameters = (ins
    "TensorType":$tensorType,
    TTL_CircularBuffer:$circularBuffer  // Reference to originating CB
  );
  let description = [{
    Represents a block of data (tiles or scalars) consumed/produced by compute operations.
    Tied to the originating circular buffer for proper pop/push semantics.

    Memory model per TT-Lang spec: Block memory is pre-allocated when the circular buffer
    is created. The block inherits the CB's shape-unit granularity (tiles for tiled layout,
    scalars for row-major layout). No dynamic allocation occurs during block acquisition.

    The block's shape matches the CB's shape parameter. This linkage enables ttl.cb_pop
    and ttl.cb_push to operate on the block alone, determining which CB to use from the
    block's circularBuffer parameter.

    Per TT-Lang spec, blocks are acquired via cb.wait() or cb.reserve() and carry
    implicit association with their source CB for subsequent pop/push operations.
  }];
}

def TTL_TransferHandle : TTL_Type<"TransferHandle", "xf"> {
  let summary = "Handle for asynchronous transfer ordering (lowers to global barrier)";
  let description = [{
    Note: TTKernel doesn't support per-transfer waits. All ttl.wait operations
    lower to global DMA barriers (`ttkernel.noc_async_read_barrier` or
    `ttkernel.noc_async_write_barrier`). This handle exists for ordering and
    future optimization opportunities.
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
    "ArrayRef<int64_t>":$srcCore,           // Source core coordinates [x, y]
    OptionalParameter<"ArrayRef<int64_t>">:$dstCore,  // Destination (unicast) [x, y]
    OptionalParameter<"ArrayRef<TTL_SliceAttr>">:$dstCoreRange  // Multicast range (slice per dim)
  );
  let description = [{
    Represents a pipe for inter-core communication.

    Unicast: src_core=[x,y], dst_core=[x',y']
    Multicast: src_core=[x,y], dst_core_range=[slice(x0,x1,step), slice(y0,y1,step)]

    Each slice in dst_core_range is a tuple (start, stop, step) per TT-Lang spec.
    The range is half-open: [start, stop). Step defaults to 1 if not specified.

    Example from TT-Lang (multicast column):
      ttl.Pipe(src_core=(x, 0), dst_core_range=(slice(x, x+1), slice(1, grid_y)))
      Creates pipe from (x,0) to cores (x, 1), (x, 2), ..., (x, grid_y-1)
  }];
}

def TTL_TensorAccessor : TTL_Type<"TensorAccessor", "accessor"> {
  let summary = "Handle for indexed tensor access";
  let description = [{
    Represents a tensor with indexing capability for slicing operations.
    Carries layout metadata from python/ttlang/layouts.py (tiled, sharded, interleaved)
    via the encoding on the `TTL_Tensor` parameter (which also captures memory space).
    Used in DMA operations to specify source/destination with coordinates.
  }];
  let parameters = (ins
    TTL_Tensor:$tensor
  );
}
```

### 3.2 Attributes

```tablegen
// TODO: Convett to an enum attribute
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
  let parameters = (ins ArrayRefParameter<"int64_t">:$dimensions);
  let assemblyFormat = "`<` custom<DynamicIndexList>($dimensions, $static_dimensions) `>`";
}

def TTL_LayoutAttr : AttrDef<TTL_Dialect, "Layout"> {
  let summary = "Tensor layout metadata (from python layouts.py)";
  let description = [{
    Captures layout information generated by create_metal_layout():
    - tiled vs row-major
    - sharded grids
    - interleaved patterns
    Allows lossless representation of all TensorAccessor configurations. The
    attribute participates in the `TTL_TensorEncodingAttr` attached to
    `tensor<...>` types, so every `TTL_Tensor` carries explicit layout alongside
    its memory-space annotation.
  }];
  // Parameters TBD based on MetalLayoutConfig from layouts.py
}

def TTL_SliceAttr : AttrDef<TTL_Dialect, "Slice"> {
  let summary = "Slice specification for core range (start, stop, step)";
  let parameters = (ins
    "int64_t":$start,
    "int64_t":$stop,
    "int64_t":$step
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let description = [{
    Encodes a Python slice(start, stop, step) for pipe multicast ranges.
    Half-open interval: [start, stop). Step defaults to 1 in Python API.

    Examples from TT-Lang spec:
    - slice(1, grid_y) → SliceAttr<1, grid_y, 1>
    - slice(x, x+1) → SliceAttr<x, x+1, 1>
    - slice(0, grid_x, 2) → SliceAttr<0, grid_x, 2> (every other core)

    Negative indices and None are resolved by Python frontend before IR generation.

    Lowering to TTKernel:
    SliceAttr converts to affine loops and TTKernel multicast NOC operations.

    Case 1: Contiguous rectangular range (step=1 for all dimensions)
      dst_core_range = [slice(x0,x1,1), slice(y0,y1,1)]

      Lowers directly to TTKernel rectangular multicast:
        %noc_addr = ttkernel.get_noc_multicast_addr(
          noc_xy_start = (x0, y0),
          noc_xy_end = (x1-1, y1-1)  // Convert half-open to inclusive
        )
        ttkernel.noc_async_write_multicast(%src_addr, %noc_addr, %size)

    Case 2: Strided range (step != 1 in any dimension)
      dst_core_range = [slice(0,8,2), slice(0,4,1)]  // Every other core in x

      Lowers to affine loop with unicast per core:
        affine.for %x = 0 to 8 step 2 {  // Affine loop encodes step directly
          affine.for %y = 0 to 4 {
            %noc_addr = ttkernel.get_noc_addr(noc_x=%x, noc_y=%y, ...)
            ttkernel.noc_async_write(%src_addr, %noc_addr, %size)
          }
        }

    The affine.for operation natively supports start/stop/step, so SliceAttr maps
    directly to affine loop bounds. No membership test needed - the affine loop
    iterates exactly over the slice range.
  }];
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

### 3.3 TensorAccessor Support Strategy

Modern Metalium kernel development uses `TensorAccessor` objects as kernel
arguments. `TensorAccessor` APIs like `noc_async_read_shard`,
`noc_async_write_shard`, `noc_async_read_page`, and `noc_async_write_page`
abstract shard addressing and page-based access patterns (see
[Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)).

Challenge: TTL lowering must preserve tensor indexing metadata all the way to
TTKernel so generated C++ can rely on the Metalium `TensorAccessor` class.
Fortunately, TTKernel already provides the necessary abstractions.

#### Existing TTKernel Support

`ttkernel.tensor_accessor` and the associated NOC ops landed in
[`TTKernelOpsTypes.td`](https://github.com/tenstorrent/tt-mlir/blob/884462ecb00571f71faceb8df0da14d84edcdf9c/include/ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td#L89).
TTL does not need to extend TT-MLIR—our lowering simply has to materialize the
existing type and ops.

For reference, the TTKernel definitions look like:

```tablegen
// TTKernelOpsTypes.td
def TTKernel_TensorAccessor : TTKernel_Type<"TensorAccessor"> {
  let summary = "Handle for indexed tensor access with layout metadata";
  let parameters = (ins "TensorType":$tensorType, "LayoutAttr":$layout);
  let description = [{
    Represents a tensor with addressing capability for shard, page, and tile-based access.
    Carries layout metadata (sharded, interleaved, or tiled; grid dimensions; shard/page/tile size).

    Corresponds to Metalium TensorAccessor class. ConvertTTKernelToEmitC generates
    TensorAccessor construction and usage matching Metalium patterns.
  }];
}

// TTKernelOps.td
def TTKernel_NocAsyncReadShardOp : TTKernel_Op<"noc_async_read_shard"> {
  let arguments = (ins I32:$shard_id, TTKernel_TensorAccessor:$accessor,
                       I32:$dst_local_l1_addr, I8:$noc);
  let description = [{
    Async read of shard from sharded tensor using TensorAccessor.
    Corresponds to: noc_async_read_shard(shard_id, accessor, dst_addr, noc)
  }];
}

def TTKernel_NocAsyncReadPageOp : TTKernel_Op<"noc_async_read_page"> {
  let arguments = (ins I32:$page_id, TTKernel_TensorAccessor:$accessor,
                       I32:$dst_local_l1_addr, OptionalAttr<I32Attr>:$offset, I8:$noc);
  let description = [{
    Async read of page from interleaved tensor using TensorAccessor.
    Corresponds to: noc_async_read_page(page_id, accessor, dst_addr, offset, noc)
  }];
}

def TTKernel_NocAsyncReadTileOp : TTKernel_Op<"noc_async_read_tile"> {
  let arguments = (ins I32:$tile_id, TTKernel_TensorAccessor:$accessor,
                       I32:$dst_local_l1_addr, I8:$noc);
  let description = [{
    Async read of tile from tiled tensor using TensorAccessor.
    Corresponds to: noc_async_read_tile(tile_id, accessor, dst_addr, noc)
    Used for tiled DRAM/L1 tensors (common pattern in Metalium).
  }];
}

// Plus write_shard, write_page, and write_tile variants
```

ConvertTTKernelToEmitC generates Metalium TensorAccessor code:

```cpp
// Generated from ttkernel.tensor_accessor + noc_async_read_shard
constexpr auto args_a = TensorAccessorArgs<0>();
const auto a = TensorAccessor(args_a, a_addr, tile_size_bytes);

for (uint32_t i = 0; i < n_shards; i++) {
    cb_reserve_back(cb_in0, 1);
    const uint32_t cb_addr = get_write_ptr(cb_in0);
    noc_async_read_shard(i, a, cb_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in0, 1);
}
```

TTL lowering:

```mlir
// TTL IR (%tensor : tensor<..., #ttl.tensor_encoding<DRAM,
//                      #ttl.layout<sharded, grid=[2,2]>>>)
%accessor = ttl.tensor_accessor %tensor
%xf = ttl.copy %accessor[%shard_id], %cb

// After TTLLowerDataMovement → TTKernel
%ttk_accessor = // Convert tensor layout encoding → TTKernel layout attr
ttkernel.noc_async_read_shard %shard_id, %ttk_accessor, %cb_l1_addr, %noc
```

**Integration benefits:**
- Generated C++ already relies on these TTKernel constructs, so TTL inherits the
  proven Metalium addressing logic for shard/page/tile accesses.
- TensorAccessor is tensor-like, fitting neatly into MLIR bufferization
  pipelines and simplifying metadata propagation.
- Future TT-MLIR improvements to TensorAccessor automatically benefit TTL.

**TTL implementation tasks:**
- Map `TTL_TensorEncodingAttr` (layout + memory space) to the TTKernel layout
  attribute when creating `ttkernel.tensor_accessor` values.
- Ensure `ttl.copy` lowering selects the appropriate TTKernel NOC op (shard,
  page, tile) based on the tensor encoding.
- Verify existing ConvertTTKernelToEmitC support covers TTL use cases; only
  descriptor plumbing in TTL should be required.


#### Complete Lowering Example: Sharded Tensor Read

This example shows the complete pipeline from Python code through TTL, TTKernel,
to generated C++ using TensorAccessor.

**Python Kernel (TT-Lang DSL)**:

```python
import ttnn
from ttlang import ttl

@ttl.kernel(grid=(2, 2))
def sharded_elementwise_add(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    # Tensors are sharded across 2x2 grid
    # Each core processes 1 shard (1 tile in this example)

    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def dm_reader():
        shard_id = ttl.core(dim=1) # 0-3 for 2x2 grid

        a_blk = a_cb.reserve()
        xf = ttl.copy(a[shard_id], a_blk)  # Read shard using TensorAccessor
        xf.wait()
        a_cb.push()

        b_blk = b_cb.reserve()
        xf = ttl.copy(b[shard_id], b_blk)
        xf.wait()
        b_cb.push()

    @ttl.compute()
    def compute():
        a_blk = a_cb.wait()
        b_blk = b_cb.wait()
        o_blk = out_cb.reserve()

        result = a_blk + b_blk  # ttl.block_add
        o_blk.store(result)

        a_cb.pop()
        b_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_writer():
        shard_id = ttl.core_dim(1)

        o_blk = out_cb.wait()
        xf = ttl.copy(o_blk, out[shard_id])  # Write shard
        xf.wait()
        out_cb.pop()

    return ttl.Program(compute, dm_reader, dm_writer)(a, b, out)
```

**TTL IR (After Python AST Compilation)**:

```mlir
ttl.kernel @sharded_elementwise_add(
    %a: tensor<64x64xf32, #ttl.tensor_encoding<DRAM,
      #ttl.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>>>,
    %b: tensor<64x64xf32, #ttl.tensor_encoding<DRAM,
      #ttl.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>>>,
    %out: tensor<64x64xf32, #ttl.tensor_encoding<DRAM,
      #ttl.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>>>
) attributes {grid = #ttl.grid<[2, 2]>, block_factors = [[1,1], [1,1], [1,1]]} {

  // Create circular buffers (tiled layout inferred from element_type)
  %a_cb = ttl.create_cb shape=[1,1], element_type=!ttcore.tile<32x32,f32>,
                        buffer_factor=2, memory_space=L1
  %b_cb = ttl.create_cb shape=[1,1], element_type=!ttcore.tile<32x32,f32>,
                        buffer_factor=2, memory_space=L1
  %out_cb = ttl.create_cb shape=[1,1], element_type=!ttcore.tile<32x32,f32>,
                          buffer_factor=2, memory_space=L1

  // Create TensorAccessors (layout taken from tensor types)
  %a_accessor = ttl.tensor_accessor %a
  %b_accessor = ttl.tensor_accessor %b
  %out_accessor = ttl.tensor_accessor %out

  ttl.datamovement_thread {
    %shard_id = ttl.core_dim(1) : index

    %a_blk = ttl.cb_reserve %a_cb : !ttl.block<...>
    %xf_a = ttl.copy %a_accessor[%shard_id], %a_blk : !ttl.xf
    ttl.wait %xf_a
    ttl.cb_push %a_blk

    %b_blk = ttl.cb_reserve %b_cb : !ttl.block<...>
    %xf_b = ttl.copy %b_accessor[%shard_id], %b_blk : !ttl.xf
    ttl.wait %xf_b
    ttl.cb_push %b_blk
  }

  ttl.compute_thread {
    %a_blk = ttl.cb_wait %a_cb : !ttl.block<...>
    %b_blk = ttl.cb_wait %b_cb : !ttl.block<...>
    %o_blk = ttl.cb_reserve %out_cb : !ttl.block<...>

    %result = ttl.block_add %a_blk, %b_blk
    ttl.block_store %o_blk, %result

    ttl.cb_pop %a_blk
    ttl.cb_pop %b_blk
    ttl.cb_push %o_blk
  }

  ttl.datamovement_thread {
    %shard_id = ttl.core_dim(1) : index

    %o_blk = ttl.cb_wait %out_cb : !ttl.block<...>
    %xf_out = ttl.copy %o_blk, %out_accessor[%shard_id] : !ttl.xf
    ttl.wait %xf_out
    ttl.cb_pop %o_blk
  }
}
```

**After TTLExpandThreads + TTLLowerDataMovement → TTKernel IR**:

```mlir
// Thread functions extracted, TensorAccessor preserved in TTKernel

func.func @dm_reader_0(
  %a_addr: i64,  // Runtime arg: buffer address
  %b_addr: i64
) attributes {
  // Compile-time args encode TensorAccessor layout
  ttkernel.compile_time_args = [
    #ttkernel.tensor_accessor_args<layout=sharded, grid=[2,2], shard_shape=[1,1]>,
    #ttkernel.tensor_accessor_args<layout=sharded, grid=[2,2], shard_shape=[1,1]>
  ]
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // Compute shard_id
  %shard_id = affine.apply affine_map<(y, x, gx) -> (y * gx + x)>(%c0, %c0, %c2)
  %shard_id_i32 = arith.index_cast %shard_id : index to i32

  // Create TTKernel TensorAccessor values
  %a_accessor = ttkernel.tensor_accessor %a_addr {
    layout = #ttkernel.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>
  } : !ttkernel.tensor_accessor<tensor<64x64xf32>>

  %b_accessor = ttkernel.tensor_accessor %b_addr {
    layout = #ttkernel.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>
  } : !ttkernel.tensor_accessor<tensor<64x64xf32>>

  // Read shard A
  %a_cb = // CB reference
  ttkernel.cb_reserve_back %a_cb, 1
  %a_write_ptr = ttkernel.get_write_ptr %a_cb
  ttkernel.noc_async_read_shard %shard_id_i32, %a_accessor, %a_write_ptr, 0
  ttkernel.noc_async_read_barrier
  ttkernel.cb_push_back %a_cb, 1

  // Read shard B
  %b_cb = // CB reference
  ttkernel.cb_reserve_back %b_cb, 1
  %b_write_ptr = ttkernel.get_write_ptr %b_cb
  ttkernel.noc_async_read_shard %shard_id_i32, %b_accessor, %b_write_ptr, 0
  ttkernel.noc_async_read_barrier
  ttkernel.cb_push_back %b_cb, 1

  return
}

// compute_0 and dm_writer_1 similar structure
```

**After ConvertTTKernelToEmitC → Generated C++ (Metalium)**:

```cpp
// dm_reader_0.cpp
#include <dataflow_api.h>

void kernel_main() {
    // Runtime arguments
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);

    // Compile-time TensorAccessorArgs (from ttkernel.compile_time_args attribute)
    constexpr auto args_a = TensorAccessorArgs<0>();
    constexpr auto args_b = TensorAccessorArgs<args_a.next_compile_time_args_offset()>();

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Construct TensorAccessors (from ttkernel.tensor_accessor ops)
    const auto a = TensorAccessor(args_a, a_addr, tile_size_bytes);
    const auto b = TensorAccessor(args_b, b_addr, tile_size_bytes);

    // Compute shard_id
    const uint32_t core_y = my_y[0];
    const uint32_t core_x = my_x[0];
    const uint32_t shard_id = core_y * 2 + core_x;

    // Read shard A (from ttkernel.noc_async_read_shard)
    cb_reserve_back(cb_in0, 1);
    const uint32_t a_cb_addr = get_write_ptr(cb_in0);
    noc_async_read_shard(shard_id, a, a_cb_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in0, 1);

    // Read shard B
    cb_reserve_back(cb_in1, 1);
    const uint32_t b_cb_addr = get_write_ptr(cb_in1);
    noc_async_read_shard(shard_id, b, b_cb_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in1, 1);
}
```

This matches the TensorAccessor pattern in Metalium Guide (lines 193-213).

**Lowering Strategy Summary**:

| Stage | TensorAccessor Representation | Key Information |
|-------|------------------------------|-----------------|
| **Python** | `a[shard_id]` tensor indexing | Layout inferred from ttnn.Tensor properties |
| **TTL IR** | `ttl.tensor_accessor` + `ttl.copy` | Layout/memory in `#ttl.tensor_encoding<...>` |
| **TTKernel IR** | `ttkernel.tensor_accessor` + `noc_async_read_shard` | Layout in `#ttkernel.layout<...>` + function compile_time_args |
| **C++ Code** | `TensorAccessor(args, addr, size)` + `noc_async_read_shard(id, accessor, ...)` | Layout encoded in `TensorAccessorArgs<N>()` template |

The layout metadata flows through all stages, enabling ConvertTTKernelToEmitC to
generate correct TensorAccessor construction code.



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
- Operation verifiers ensure proper sequencing: each reserve has matching push,
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
    - Operation verifiers enforce this requirement and reject unguarded or unmatched pipes

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
    Folds to constants. Validation enforced in frontend and operation verifiers.
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



## 5. Compilation Pipeline

### 5.1 Pass Architecture

```
Python Kernel (@compute/@datamovement decorators)
  ↓
Python AST Parsing (TTLDialectCompiler)
  ↓
ttl.kernel (with thread regions, tensor operands)
  ↓
[Phase 1: Canonicalization & Layout Verification]
  ├─ TTLCanonicalizePass - Fold constants, simplify patterns
  └─ TTLVerifyLayoutsPass - Check tensor accessor layouts

Note: Operation and type validation (CB protocol, thread operation restrictions,
type compatibility, etc.) is performed by individual op/type verifiers during IR
construction, not by a separate validation pass.
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
  ├─ TTLLowerCompute - ttl.block_add → linalg.generic → (transform tiling) → scf.for + ttkernel.add_tiles
  ├─ TTLLowerDataMovement - ttl.copy → ttkernel.tensor_accessor + ttkernel.noc_async_shard/page/tile
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

**`TTLInsertSynchronization`**
- **Input**: `ttl.kernel` with thread regions (tensor operands)
- **Analysis**: Build producer-consumer DAG for blocks, semaphores, pipes
- **Transform**: Insert `ttl.dma_barrier` where needed, validate CB usage
  patterns
- **Output**: `ttl.kernel` with explicit synchronization

**`TTLInferDSTRequirements`**
- **Input**: `ttl.kernel` with compute operations
- **Analysis**: Track which SSA values participate in compute chains
- **Transform**: Insert `ttl.require_dst` markers for liveness analysis
- **Output**: `ttl.kernel` with DST hints

**`TTLBufferizePass`**
- **Input**: `ttl.kernel` with tensor operands
- **Analysis**: Determine bufferization strategy for each tensor
- **Transform**: Convert tensor types to memref types (One-Shot Bufferization)
- **Output**: `ttl.kernel` with memref operands
- **Note**: Critical transition - tensor semantics → buffer semantics

**`TTLAllocateCircularBuffers`**
- **Input**: `ttl.kernel` with memref CBs
- **Analysis**: Compute liveness ranges for each CB
- **Transform**: Assign L1 addresses using first-fit allocation
- **Output**: CBs annotated with address attributes

**`TTLExpandThreads`**
- **Input**: `ttl.kernel` with regions
- **Transform**: Extract each thread region into `func.func` with metadata
- **Output**: Separate functions per thread, preserving grid/memory_space attrs

**`TTLAssignDSTRegisters`**
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

**`TTLLowerCompute`**
- **Input**: `ttl.block_add`, `ttl.block_mul`, `ttl.block_matmul` and other
  block operations
- **Transform**: Lower to `linalg.generic` structured operations
  - Block operations become linalg.generic with TTL tile operations in the body
  - Implements TilingInterface for capacity-based tiling via transform dialect
  - Enables `transform.structured.multitile_sizes` for DST register capacity
    constraints
  - After transform-based tiling, linalg ops lower to TTKernel tile operations
- **Output**: `linalg.generic` operations (pre-tiling) or TTKernel tile
  operations (post-tiling)
- **Note**: See Section 5.6 for detailed linalg.generic tiling strategy

**`TTLLowerDataMovement`**
- **Input**: `ttl.copy` operations with `TensorAccessor` or `Pipe` operands
- **Transform**:
  - Tensor → CB: `ttkernel.tensor_accessor` +
    `ttkernel.noc_async_read_shard/page/tile`
  - CB → Tensor: `ttkernel.tensor_accessor` +
    `ttkernel.noc_async_write_shard/page/tile`
  - CB → Pipe → CB: `ttkernel.noc_async_write_multicast` or unicast
  - Layout attribute determines which NOC API variant (shard/page/tile)
- **Output**: TTKernel NOC operations using `TensorAccessor`

### 5.3 Source Location Tracking

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
  let regions = (region VariadicRegion<AnyRegion>:$threads);
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

### 5.4 Error Handling and Diagnostics

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
   - Emitted by individual operation verifiers during IR construction
   - Include source location from Python AST

2. **Resource Errors**: L1 capacity exceeded, DST register overflow
   - Emitted by `TTLAllocateCircularBuffers` and `TTLAssignDSTRegisters`
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

**Decision**: Use linalg.generic for block compute operations with transform
dialect tiling, Affine for explicit loops, and SCF only for conditionals.
leverage upstream transforms immediately.

**See**: [Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/),
[Transform Tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)


### 5.6 Structured Ops for DST Register Capacity-Based Tiling

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


## 6. Type Conversion & Lowering Examples

### 6.1 TTL → TTKernel Type Mapping

| TTL Type | TTKernel Type | Conversion Notes |
|----------|---------------|------------------|
| `!ttl.cb<[2,1], !ttcore.tile<32x32,f32>, 2, L1>` | `!ttkernel.cb<4, !ttcore.tile<32x32,f32>>` | Total elements = 2×1×2 = 4 tiles (layout inferred: element_type is tile → tiled layout); For row-major CBs, element_type is scalar (f32, bf16, etc.); memory space drives L1 address assignment; buffer_factor used to compute total but not preserved in TTKernel CB type |
| `!ttl.block<tensor<2x1x!ttcore.tile>>` | Elided | Blocks decomposed into tile operations during lowering |
| `!ttl.xf` | N/A (elided) | Lowers to global barriers |
| `!ttl.semaphore` | `!ttkernel.semaphore` | Direct mapping |
| `!ttl.pipe` | Attributes on ops | Pipe decomposed into core coords + multicast flags |
| `!ttl.accessor<layout, memspace>` | `!ttkernel.tensor_accessor<tensor_type, layout>` | Layout attributes converted from TTL to TTKernel format |

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
| `ttl.copy` (Tensor→CB sharded) | `ttkernel.tensor_accessor`, `ttkernel.noc_async_read_shard` | High | **HIGH** |
| `ttl.copy` (Tensor→CB interleaved) | `ttkernel.tensor_accessor`, `ttkernel.noc_async_read_page` | High | **HIGH** |
| `ttl.copy` (Tensor→CB tiled) | `ttkernel.tensor_accessor`, `ttkernel.noc_async_read_tile` | High | **HIGH** |
| `ttl.copy` (CB→Tensor sharded) | `ttkernel.tensor_accessor`, `ttkernel.noc_async_write_shard` | High | **HIGH** |
| `ttl.copy` (CB→Tensor interleaved) | `ttkernel.tensor_accessor`, `ttkernel.noc_async_write_page` | High | **HIGH** |
| `ttl.copy` (CB→Tensor tiled) | `ttkernel.tensor_accessor`, `ttkernel.noc_async_write_tile` | High | **HIGH** |
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
// TTL (Python/user-facing)
%blk = ttl.cb_wait %cb : !ttl.block<...>

// TTKernel (lowered - count derived from CB type)
ttkernel.cb_wait_front %cb, 2  // 2 computed from CB.shape
```

**Tile Extraction (Internal Lowering Artifacts):**
```mlir
// TTL (lowered form - internal IR only, not emitted by TT-Lang frontend)
// Generated during block→tile lowering pass
%tile0 = ttl.get_tile %cb, %c0
%tile1 = ttl.get_tile %cb, %c1

// TTKernel (copy from CB to DST register)
ttkernel.copy_tile_init %cb
%tile0 = ttkernel.copy_tile %cb, 0, %dst_idx0
%tile1 = ttkernel.copy_tile %cb, 1, %dst_idx1
```

Note: `ttl.get_tile` and `ttl.pack_tile` operations are generated automatically during block-to-tile lowering and are not exposed in the TT-Lang Python DSL. Users work with block-level abstractions exclusively.

**Block Compute:**
```mlir
// TTL (user-facing block-level abstraction)
%a_blk = ttl.cb_wait %a_cb : !ttl.block<...>
%b_blk = ttl.cb_wait %b_cb : !ttl.block<...>
%result = ttl.block_add %a_blk, %b_blk

// TTKernel (explicit per-tile - lowered from block ops)
ttkernel.cb_wait_front %a_cb, 2  // 2 from CB shape
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

**DMA Operations (Sharded Tensor):**
```mlir
// TTL (high-level, %input : tensor<..., #ttl.tensor_encoding<DRAM,
//                     #ttl.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>>>)
%accessor = ttl.tensor_accessor %input
%xf = ttl.copy %accessor[%shard_id], %cb
ttl.wait %xf

// TTKernel (TensorAccessor preserved)
%ttk_accessor = ttkernel.tensor_accessor %input_addr {
  layout = #ttkernel.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>
} : !ttkernel.tensor_accessor<...>
%cb_l1_addr = // From TTLAllocateCircularBuffers pass
ttkernel.noc_async_read_shard %shard_id, %ttk_accessor, %cb_l1_addr, %noc
ttkernel.noc_async_read_barrier

// Generated C++ (uses TensorAccessor class)
constexpr auto args = TensorAccessorArgs<0>();
const auto input = TensorAccessor(args, input_addr, tile_size_bytes);
noc_async_read_shard(shard_id, input, cb_l1_addr);
```

**DMA Operations (Interleaved Tensor):**
```mlir
// TTL (%input : tensor<..., #ttl.tensor_encoding<DRAM, #ttl.layout<interleaved>>> )
%accessor = ttl.tensor_accessor %input
%xf = ttl.copy %accessor[%page_id], %cb
ttl.wait %xf

// TTKernel (TensorAccessor preserved)
%ttk_accessor = ttkernel.tensor_accessor %input_addr {
  layout = #ttkernel.layout<interleaved>
} : !ttkernel.tensor_accessor<...>
%cb_l1_addr = // From TTLAllocateCircularBuffers pass
ttkernel.noc_async_read_page %page_id, %ttk_accessor, %cb_l1_addr, %offset, %noc
ttkernel.noc_async_read_barrier

// Generated C++ (uses TensorAccessor class)
constexpr auto args = TensorAccessorArgs<0>();
const auto input = TensorAccessor(args, input_addr, tile_size_bytes);
noc_async_read_page(page_id, input, cb_l1_addr, offset);
```

**Multicast Pipe:**
```mlir
// TTL
%pipe = ttl.create_pipe src_core=[0,0] dst_core_range=[[0,0],[1,3]]
%xf = ttl.copy %block, %pipe
ttl.wait %xf

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
    # Transfer handle.wait() maps to ttl.wait %xf
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

The MVP delivers TTL → TTKernel → ConvertTTKernelToEmitC → C++ kernel source.
This leverages existing tt-mlir infrastructure. Direct TTL → C++ emission
(bypassing TTKernel) is deferred to post-MVP (see section 10.2).

### Phase 1: Foundation (Week 1)
**Goal**: TTL dialect compiles and registers with MLIR

1. Dialect Definition (C++)
   - Create `include/ttlang/Dialect/TTL/` and `lib/Dialect/TTL/`
   - Define types (TTLTypes.td): CB (with optional buffer_index, page_size,
     core_ranges), TransferHandle, Semaphore, Pipe, Accessor
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
**Goal**: Thread expansion and basic lowering

Note: Validation (CB/pipe/semaphore contracts, store/wait/pop sequencing, thread
operation restrictions) is handled by individual operation and type verifiers
during IR construction, not by a separate pass.

1. `TTLExpandThreads` - Extract threads to separate functions (can happen early)

3. `TTLLowerCompute` - Block ops → TTKernel tile ops
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

TTL integrates with TTNN runtime using the `ttnn.generic_op` API, which allows
execution of custom C++ kernels by providing kernel source code and metadata
through Python descriptors. This approach enables:

- Direct kernel specification: Provide C++ kernel source files as strings with
  compile-time and runtime arguments
- Tensor metadata integration: Attach circular buffer configurations and memory
  layout information to operations
- Python-first workflow: Initially implemented in Python for rapid development
  and testing
- No separate compilation step: Kernels are provided as source strings and
  compiled by the TTNN runtime

The TTL compilation pipeline generates the necessary kernel descriptors and
metadata that are then passed to `ttnn.generic_op` for execution.

### Execution Flow

```
TTL Dialect → TTL Passes → Kernel Descriptor Generation → ttnn.generic_op → TTNN Runtime → Hardware
                ↓
     Validation, Synchronization,
     Bufferization, Allocation,
     Register Assignment, Optimization
```

Key components:
1. TTL passes perform memory planning, synchronization inference, and resource
   allocation
2. Kernel descriptor generator produces C++ kernel strings with metadata
3. Python API constructs `ttnn.KernelDescriptor`, `ttnn.CBDescriptor`, and
   `ttnn.ProgramDescriptor` objects
4. `ttnn.generic_op` compiles and executes kernels on device
5. Results returned as `ttnn.Tensor` objects

### TTNN Generic Operation API

The `ttnn.generic_op` API requires three descriptor types:

KernelDescriptor - specifies kernel source and execution parameters:
```python
reader_kernel = ttnn.KernelDescriptor(
    kernel_source="<C++ kernel source code as string>",  # Or file path
    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,  # Or INLINE
    core_ranges=core_grid,
    compile_time_args=[...],  # Shape info, addresses, etc.
    runtime_args=[[...]],     # Buffer addresses, tile counts
    defines=[("MACRO_NAME", "value"), ...],  # Preprocessor defines
    config=ttnn.ReaderConfigDescriptor(),  # Or Writer/ComputeConfigDescriptor
)
```

CBDescriptor - defines circular buffer configurations:
```python
input_cb = ttnn.CBDescriptor(
    total_size=cb_total_size,      # Total CB size in bytes
    core_ranges=core_grid,          # Cores with this CB
    format_descriptors=[
        ttnn.CBFormatDescriptor(
            format=ttnn.DataFormat.Float16_b,
            page_size=tile_size,
        )
    ],
)
```

ProgramDescriptor - combines kernels and CBs:
```python
program = ttnn.ProgramDescriptor(
    kernels=[reader_kernel, compute_kernel, writer_kernel],
    semaphores=[],  # Synchronization primitives if needed
    cbs=[input_cb, output_cb],
)

# Execute with tensors
output = ttnn.generic_op([input_tensor, output_tensor], program)
```

### Compilation Pipeline

The TTL compilation pipeline generates kernel descriptors and metadata for
`ttnn.generic_op`:

```python
# Python API workflow (ttlang runtime)
from ttlang.d2m_api import compile_ttl_kernel

# 1. TTL IR → TTL passes (validation, allocation, optimization)
# 2. Generate kernel descriptors with C++ source strings
kernel_descriptors = compile_ttl_kernel(ttl_ir, device_info)

# 3. Create TTNN descriptors
reader_desc = ttnn.KernelDescriptor(
    kernel_source=kernel_descriptors['reader_source'],
    compile_time_args=kernel_descriptors['reader_ct_args'],
    runtime_args=kernel_descriptors['reader_rt_args'],
    config=ttnn.ReaderConfigDescriptor(),
)

compute_desc = ttnn.KernelDescriptor(
    kernel_source=kernel_descriptors['compute_source'],
    compile_time_args=kernel_descriptors['compute_ct_args'],
    defines=kernel_descriptors['compute_defines'],
    config=ttnn.ComputeConfigDescriptor(),
)

writer_desc = ttnn.KernelDescriptor(
    kernel_source=kernel_descriptors['writer_source'],
    compile_time_args=kernel_descriptors['writer_ct_args'],
    runtime_args=kernel_descriptors['writer_rt_args'],
    config=ttnn.WriterConfigDescriptor(),
)

# 4. Create CB descriptors from TTL allocation pass results
cb_descs = [
    ttnn.CBDescriptor(
        total_size=cb.size,
        core_ranges=cb.cores,
        format_descriptors=[ttnn.CBFormatDescriptor(format=cb.format, page_size=cb.page_size)]
    )
    for cb in kernel_descriptors['circular_buffers']
]

# 5. Combine and execute
program = ttnn.ProgramDescriptor(
    kernels=[reader_desc, compute_desc, writer_desc],
    cbs=cb_descs,
)
output = ttnn.generic_op(io_tensors, program)
```

### C++ Kernel Generation from TTL

TTL operations lower to C++ kernel strings following the TT-Metalium
three-kernel pattern:

Reader kernel (data movement):
```cpp
// Generated from TTL datamovement thread
void MAIN {
    uint32_t src_addr = get_compile_time_arg_val(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = 0;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read(src_addr, l1_write_addr, tile_size);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
        src_addr += tile_size;
    }
}
```

Compute kernel:
```cpp
// Generated from TTL compute thread
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_out0 = 16;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_out0, 1);

        acquire_dst(tt::DstMode::Half);
        add_tiles(cb_in0, cb_in1, 0, 0, 0);
        pack_tile(0, cb_out0);
        release_dst(tt::DstMode::Half);

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
```

Writer kernel (data movement):
```cpp
// Generated from TTL datamovement thread
void MAIN {
    uint32_t dst_addr = get_compile_time_arg_val(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_out0 = 16;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_id_out0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write(l1_read_addr, dst_addr, tile_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
        dst_addr += tile_size;
    }
}
```

### TTL-Specific Metadata Generation

The TTL compilation pipeline generates metadata required by `ttnn.generic_op`:

1. Circular buffer metadata (from `TTLAllocateCircularBuffers` pass):
```python
# CB allocation results
cb_metadata = {
    'cb_id': 0,
    'address': 0x12000,  # L1 address
    'size': 32 * 1024,   # 32KB
    'num_pages': 16,
    'page_size': 2048,
    'format': 'Float16_b',
}

# Convert to TTNN CBDescriptor
cb_desc = ttnn.CBDescriptor(
    total_size=cb_metadata['size'],
    core_ranges=core_grid,
    format_descriptors=[
        ttnn.CBFormatDescriptor(
            format=getattr(ttnn.DataFormat, cb_metadata['format']),
            page_size=cb_metadata['page_size'],
        )
    ],
)
```

2. Kernel compile-time arguments (from TTL passes):
```python
# From TTL IR analysis
compile_time_args = [
    tensor_shape[0],           # Rows
    tensor_shape[1],           # Cols
    cb_metadata['address'],    # L1 address
    tile_size,                 # Tile dimensions
    num_tiles,                 # Total tiles to process
]
```

3. Kernel runtime arguments (per-core parameters):
```python
# Generated per core from TTL grid information
runtime_args = [
    [buffer_address, core_tile_count, start_tile_idx]
    for core in core_grid
]
```

4. Thread mapping (TTL threads → TT-Metal kernels):
   - `ttl.compute_thread` → Compute kernel with SFPU/matmul operations
   - `ttl.datamovement_thread` (reader) → Reader kernel with NOC operations
   - `ttl.datamovement_thread` (writer) → Writer kernel with NOC operations

5. Synchronization metadata (from `TTLInsertSynchronization` pass):
```python
# DMA barrier insertion points
barriers = [
    {'type': 'noc_async_read_barrier', 'location': 'after_read'},
    {'type': 'noc_async_write_barrier', 'location': 'after_write'},
]

# Semaphore configurations for inter-core communication
semaphores = [
    ttnn.SemaphoreDescriptor(
        initial_value=0,
        core_ranges=producer_cores,
    )
]
```

### MVP Integration Implementation

Phase 3 deliverable (Week 2-3):

1. Kernel descriptor generation pass - create `TTLGenerateKernelDescriptors`
   pass that:
   - Traverses TTL IR after allocation and synchronization passes
   - Generates C++ kernel source strings for reader, compute, writer
   - Collects compile-time and runtime arguments
   - Produces CB metadata from allocation results

2. Python runtime API - implement `ttlang.runtime` module:
   ```python
   from ttlang.runtime import execute_ttl_kernel

   # High-level API wrapping ttnn.generic_op
   result = execute_ttl_kernel(
       ttl_module,          # Compiled TTL IR
       input_tensors,       # ttnn.Tensor inputs
       device,              # ttnn.Device
   )
   ```

3. Integration with `@pykernel_gen` decorator:
   ```python
   @pykernel_gen
   def my_kernel(a: Tensor, b: Tensor) -> Tensor:
       # TT-lang DSL code
       ...

   # Decorator handles:
   # 1. Python AST → TTL IR
   # 2. TTL passes (validation, allocation, sync)
   # 3. Kernel descriptor generation
   # 4. ttnn.generic_op invocation
   result = my_kernel(ttnn_tensor_a, ttnn_tensor_b)
   ```

### Validation and Testing

```python
# Test TTL kernel via ttnn.generic_op
import ttnn
import torch
from ttlang.d2m_api import compile_ttl_kernel

# Create device
device = ttnn.open_device(device_id=0)

# Compile TTL to kernel descriptors
ttl_ir = parse_ttl_source(kernel_source)
kernel_info = compile_ttl_kernel(ttl_ir, device)

# Create TTNN tensors
a = ttnn.from_torch(torch.randn(128, 128), device=device, dtype=ttnn.float32)
b = ttnn.from_torch(torch.randn(128, 128), device=device, dtype=ttnn.float32)
output = ttnn.allocate_tensor_on_device(
    ttnn.Shape([128, 128]), ttnn.float32, ttnn.TILE_LAYOUT, device
)

# Build program descriptor
program = ttnn.ProgramDescriptor(
    kernels=kernel_info['kernel_descriptors'],
    cbs=kernel_info['cb_descriptors'],
)

# Execute
result = ttnn.generic_op([a, b, output], program)

# Verify correctness
expected = torch.matmul(a.to_torch(), b.to_torch())
actual = result.to_torch()
assert torch.allclose(actual, expected, rtol=1e-2)

ttnn.close_device(device)
```

### Future Enhancements

- C++ API support: Extend to C++ `ttnn::generic_op` for performance-critical
  paths
- Kernel caching: Cache compiled kernels to avoid recompilation
- Debugging support: Source-level debugging with kernel source mapping
- Performance profiling: Integration with TTNN profiling tools
- Distributed execution: Multi-device kernel execution with explicit data
  movement



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
  let regions = (region VariadicRegion<AnyRegion>:$threads);
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

### 10.2 Direct C++ Backend (Post-MVP Extension)

**Status:** Post-MVP, deferred until TTL → TTKernel → EmitC path is stable.

**Goal:** Add direct TTL → C++ lowering bypassing TTKernel dialect.

```
MVP (uses existing tt-mlir passes):
  TTL → TTKernel → ConvertTTKernelToEmitC → C++

Future (direct emission):
  TTL → ConvertTTLToEmitC → C++
```

**Rationale for deferring:**
- MVP leverages existing, proven ConvertTTKernelToEmitC pass from tt-mlir
- No need to duplicate C++ code generation logic
- TTKernel provides validated abstraction for hardware operations
- Direct emission is optimization, not requirement

**Potential benefits (post-MVP):**
- Standalone kernels without TTKernel dependency
- Potential for TT-lang-specific C++ optimizations
- Simplified debugging (direct C++ inspection)

**Note:** This is an optimization, not a functional requirement. The MVP path
through TTKernel is sufficient and avoids duplicating the mature
ConvertTTKernelToEmitC infrastructure.

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
    ArrayRefParameter<"int64_t">:$gridShape,      // Distribution grid [8, 8]
    ArrayRefParameter<"int64_t">:$shardShape,     // Per-core shard [32, 32]
    "Type":$elementType,                          // f32, bf16, etc.
    TTL_DistributionStrategyAttr:$strategy,     // Sharded, interleaved, etc.
    TTL_MemorySpaceAttr:$memorySpace
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

### Why linalg.generic for Block Compute Operations (Not Direct Affine)?

Block compute operations (ttl.block_add, ttl.block_mul, ttl.compute_region) are
lowered to linalg.generic structured operations rather than direct affine loops.
This decision is driven by DST register capacity constraints.

The problem: Block operations process multiple tiles (e.g., 8×8 = 64 tiles), but
hardware provides limited DST registers (typically 8-16). Processing all tiles
at once exceeds capacity. The compiler must tile blocks into sub-blocks that fit
within DST limits.

The linalg.generic approach provides purpose-built transform dialect primitives
for capacity-constrained tiling:

| Capability | linalg.generic | Direct Affine Loops |
|------------|----------------|---------------------|
| Capacity-aware tiling | `transform.structured.multitile_sizes` computes sub-block sizes ≤ target capacity | No equivalent; manual computation required |
| Dynamic tile sizes | Built-in via TilingInterface | Manual in C++ |
| Tile-and-fuse | TilingInterface provides built-in fusion | Manual fusion logic |
| Transform dialect ops | ~10 tiling operations | 2 ops (min/max simplification only) |
| Multi-level tiling | `continuous_tile_sizes` for hierarchies | No support |

Key transform operation: `transform.structured.multitile_sizes`
(LinalgTransformOps.td:745-824) computes two tile sizes (sz1, sz2) both ≤
target_size, ensuring n×sz1 + m×sz2 = dimension_size. This directly solves
capacity-constrained tiling.

Example: For an 8×8 block with 8 DST registers, transform dialect tiles into 8
sub-blocks of 1×8 tiles each. Each sub-block processes its 8 tiles through the
full operation chain (add → mul → exp) with DST register reuse before packing
back to L1.

The affine dialect's transform operations provide only min/max simplification
(AffineTransformOps.td), with no tiling primitives. Implementing equivalent
capacity-based tiling would require external C++ analysis and manual tile size
computation.

Trade-off: Higher implementation complexity for lowering to linalg.generic, but
essential for managing DST capacity constraints through composable transform
dialect primitives. See Section 5.6 for detailed walkthrough.



## 13. References

- **TT-lang Spec**: `docs/TT-lang.md`
- **Build System**: `docs/BUILD_SYSTEM.md`
- **Testing Guide**: `test/TESTING.md`
- **Current D2M Pipeline**: `python/ttlang/d2m_api.py`
- **TTKernel Dialect**: `../tt-mlir/include/ttmlir/Dialect/TTKernel/IR/`
- **LLVM Upstream**: https://github.com/llvm/llvm-project/
