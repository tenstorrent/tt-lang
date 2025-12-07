# TTL Dialect Type System

**Part of**: [TTL Dialect Design Plan](01_TTL_Dialect_Plan.md)

This document specifies the type system for the TTL (TT-Lang) MLIR dialect, including core types, attributes, and TensorAccessor support strategy.

## Table of Contents

- [3.1 Core Types](#31-core-types)
- [3.2 Attributes](#32-attributes)
- [3.3 TensorAccessor Support Strategy](#33-tensoraccessor-support-strategy)

## Related Documents

- [Back to Overview](01_TTL_Dialect_Plan.md)
- [Compute Operations](03_TTL_Compute_Operations.md) - Operations using these types
- [Data Movement Operations](04_TTL_Data_Movement_Operations.md) - Data movement with these types
- [Compilation Pipeline](05_TTL_Compilation_Pipeline.md) - How types are converted during compilation

---

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
    during type construction and in TTLValidatePass.
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



