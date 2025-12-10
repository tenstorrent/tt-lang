# TTL Dialect Type System

**Part of**: [TTL Dialect Design Plan](01_TTL_Dialect_Plan.md)

This document specifies the type system for the TTL (TT-Lang) MLIR dialect,
including core types, attributes, and TensorAccessor support strategy.

## Table of Contents

- [3.1 Core Types](#31-core-types)
- [3.2 Attributes](#32-attributes)
- [3.3 TensorAccessor Support Strategy](#33-tensoraccessor-support-strategy)

## Related Documents

- [Back to Overview](01_TTL_Dialect_Plan.md)
- [Compute Operations](03_TTL_Compute_Operations.md) - Operations using these
  types
- [Data Movement Operations](04_TTL_Data_Movement_Operations.md) - Data movement
  with these types
- [Compilation Pipeline](06_TTL_Compilation_Pipeline.md) - How types are
  converted during compilation



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

class TTL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class TTL_Op<string mnemonic, list<Trait> traits = []>
  : Op<TTL_Dialect, mnemonic, traits> {}

// Note: TTL uses standard MLIR tensor types with encoding attributes rather than
// a custom tensor type. This avoids defining a new type and reuses upstream tensor
// infrastructure. The TTL_TensorEncodingAttr provides layout and memory space metadata.
//
// Type constraint for operations (this is a constraint, not a TypeDef):
def TTL_Tensor : TensorOf<[F32, F16, BF16]> {
  let summary = "MLIR tensor with TTL encoding attribute";
  let description = [{
    TTL tensors are standard MLIR RankedTensorType with TTL_TensorEncodingAttr
    encoding attribute. The encoding carries layout and memory-space metadata, e.g.,
    `tensor<64x64xf32, #ttl.tensor_encoding<DeviceDRAM,
    #ttl.layout<tiled, tile_shape=[32,32], grid=[2,2]>>>`.

    Element types: Tensors have scalar element types (f32, f16, bf16). Tile layout
    information is encoded in the TTL_TensorEncodingAttr, not the element type.
    This design choice:
    - Reuses ttcore::TileType from tt-mlir (avoids duplicate tile type definitions)
    - Keeps tensor element types consistent with standard MLIR conventions
    - Defers tile-level representation to lowering passes (TTL operates at tensor/block level)

    During lowering to TTKernel, tiled tensors produce ttcore::TileType values where needed.
    Circular buffers store tiles using ttcore::TileType directly in their element_type parameter.

    This approach reuses upstream tensor infrastructure and enables compatibility
    with standard MLIR transformations (bufferization, tiling, etc.).
  }];
}


// TTL Types (TTLTypes.td)

def TTL_CircularBuffer : TTL_Type<"CircularBuffer", "cb"> {
  let summary = "Circular buffer for producer-consumer communication (L1 memory only)";
  let parameters = (ins
    "ArrayRef<int64_t>":$shape,          // Elements per block
    "mlir::Type":$elementType,           // !ttcore.tile<32x32, f32> or scalar type
    "int64_t":$bufferFactor              // Number of blocks/slots
  );
  // Note: ArrayRef<int64_t> in assemblyFormat requires custom print/parse methods
  // let assemblyFormat = "`<` $shape `,` $elementType `,` $bufferFactor `>`";
  let hasCustomAssemblyFormat = 1;

  // Design note: Memory space is NOT a type parameter because CBs are always L1.
  // This design choice:
  // - Simplifies type system (no invalid memory space combinations to reject)
  // - Matches hardware constraint (CBs are L1-only per TT-Metal architecture)
  // - Avoids surface for user error (cannot accidentally request DRAM CB)
  // Future SRAM tiers: If additional fast memory tiers are added, they would likely
  // require new CB types or explicit opt-in rather than memory-space parameter.

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

    Memory model: Circular buffers always reside in L1 memory. DRAM and System memory
    are not valid memory spaces for circular buffers. DST registers are managed exclusively
    by the TTLAssignDSTRegisters pass and do not participate in the CB memory space system.

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
  let summary = "Logical unit of data exchanged via circular buffers (L1 memory only)";
  let parameters = (ins
    "TensorType":$tensorType,
    TTL_CircularBuffer:$circularBuffer  // Reference to originating CB type (not value)
  );
  let description = [{
    Represents a block of data (tiles or scalars) consumed/produced by compute operations.
    Tied to the originating circular buffer for proper pop/push semantics.

    Design note: The block type references the CircularBuffer type (not a CB SSA value)
    to capture the structural relationship and enable type-level shape/layout queries.
    The specific CB instance is tracked through SSA def-use chains (blocks are produced
    by ttl.cb_wait/ttl.cb_reserve operations on specific CB values). This design enables:
    - Type-level validation of block shapes against CB shapes
    - Simplified operation signatures (ttl.cb_pop takes only the block, CB derived from type)
    - Static shape propagation through block operations

    Memory model per TT-Lang spec: Blocks always reside in L1 memory, inheriting the
    L1-only constraint from their originating circular buffer. Block memory is pre-allocated
    when the circular buffer is created. The block inherits the CB's shape-unit granularity
    (tiles for tiled layout, scalars for row-major layout). No dynamic allocation occurs
    during block acquisition.

    The block's shape matches the CB's shape parameter. This linkage enables ttl.cb_pop
    and ttl.cb_push to operate on the block alone, determining which CB to use from the
    block's circularBuffer parameter.

    Per TT-Lang spec, blocks are acquired via cb.wait() or cb.reserve() and carry
    implicit association with their source CB for subsequent pop/push operations.
  }];
}

def TTL_TransferHandle : TTL_Type<"TransferHandle", "xf"> {
  let summary = "Handle for asynchronous transfer with transaction ID tracking";
  let description = [{
    Transfer handle for DMA operations that maps to a TTKernel transaction ID (TRID).
    Each ttl.copy operation receives a unique TRID (0-15), and ttl.wait operations
    use the handle to wait for the specific transfer via TRID-based barriers.

    Lowering: ttl.wait(%xf) lowers to:
    - `ttkernel.noc_async_read_barrier_with_trid(trid, noc)` for read transfers
    - `ttkernel.noc_async_write_barrier_with_trid(trid, noc)` for write transfers

    Implementation note: TRID-specific barrier operations need to be added to TTKernel
    dialect (straightforward addition following existing barrier patterns). The underlying
    TT-Metal runtime provides these functions in `tt_metal/hw/inc/dataflow_api.h`.

    TRID allocation: Compiler manages TRID assignment and barrier counter resets. When
    more than 16 concurrent transfers exist, compiler may fall back to global barriers
    or insert intermediate waits to free TRIDs.

    See: `tt-metal/hw/inc/dataflow_api.h` for TRID barrier functions and
    `tests/tt_metal/tt_metal/data_movement/transaction_id/README.md` for usage examples.
  }];
}

```

#### Semaphore Type

```tablegen
def TTL_Semaphore : TTL_Type<"Semaphore", "semaphore"> {
  let summary = "Synchronization primitive for inter-core coordination";
  let parameters = (ins DefaultValuedParameter<"bool", "false">:$isRemote);
}
```

#### Pipe Type

```tablegen
def TTL_Pipe : TTL_Type<"Pipe", "pipe"> {
  let summary = "Inter-core communication channel (first-class SSA value)";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$srcCore,           // Source core coordinates [x, y]
    OptionalArrayRefParameter<"int64_t">:$dstCore,  // Destination (unicast) [x, y]
    OptionalArrayRefParameter<"TTL_SliceAttr">:$dstCoreRange  // Multicast range (slice per dim)
  );
  let hasCustomAssemblyFormat = 1;
  let description = [{
    Represents a pipe for inter-core communication.

    MVP restriction: 2D grids only. The initial implementation supports two-dimensional
    grids (x, y coordinates). Multi-chip grids with three or more dimensions are post-MVP.
    The TT-Lang spec supports higher-dimensional grids with flatten/pad semantics, but
    the TTL dialect MVP restricts to 2D to simplify coordinate translation and multicast
    range construction during lowering.

    Unicast: src_core=[x,y], dst_core=[x',y']
    Multicast: src_core=[x,y], dst_core_range=[slice(x0,x1,step), slice(y0,y1,step)]

    Each slice in dst_core_range is a tuple (start, stop, step) per TT-Lang spec.
    The range is half-open: [start, stop). Step defaults to 1 if not specified.

    Arity requirement: The dst_core_range tuple must have the same arity as the
    grid rank to prevent ambiguity. For a 2D grid (grid_x, grid_y), both dimensions
    must be specified explicitly. Use slice(x, x+1) for a single core in that dimension.

    Example (2D grid):
      Valid: dst_core_range=(slice(x, x+1), slice(1, grid_y))  // Explicit in both dims
      Invalid: dst_core_range=(x, slice(1, grid_y))  // Ambiguous: int x slice interpretation

    Loopback inference:
    The compiler automatically detects when src_core is within dst_core_range and
    selects the appropriate loopback multicast operation during lowering in
    TTLLowerDataMovement pass.

    Lowering:
    - If src_core NOT in dst_core_range → ttkernel.noc_async_write_multicast, ttkernel.noc_semaphore_set_multicast
    - If src_core IN dst_core_range → ttkernel.noc_async_write_multicast_loopback_src, ttkernel.noc_semaphore_set_multicast_loopback_src

    Examples from TT-Lang:
      Multicast column (sender excluded):
        ttl.Pipe(src_core=(x, 0), dst_core_range=(slice(x, x+1), slice(1, grid_y)))
        Pipe from (x,0) to cores (x, 1), (x, 2), ..., (x, grid_y-1)
        Sender at (x,0) not in range [1, grid_y) → non-loopback operation

      Symmetric barrier (sender included):
        ttl.Pipe(src_core=(0, 0), dst_core_range=(slice(0, 4), 0))
        All cores in range [0,4) receive signal, including sender at (0,0)
        Sender at (0,0) in range [0, 4) → loopback operation automatically selected
  }];
}
```
See [05_TTL_Multicast_Implementation.md](05_TTL_Multicast_Implementation.md) for
detailed multicast patterns.

#### PipeNet Type

```tablegen
def TTL_PipeNet : TTL_Type<"PipeNet", "pipenet"> {
  let summary = "Network of pipes";
  let description = [{
    Compile-time abstraction representing a collection of pipes for validation and
    conditional code generation. PipeNet values exist only during compilation and are
    eliminated during lowering to TTKernel.

    Design note: PipeNet is an opaque type with no parameters. The topology information
    is not stored in the type itself but in the defining ttl.create_pipenet operation's
    variadic operands. This design:
    - Simplifies type system (all PipeNet values share the same type)
    - Follows standard MLIR pattern for compile-time tokens
    - Enables dynamic pipe construction (variadic operation arguments)
    - Topology accessible via SSA def-use chain analysis

    Runtime representation: PipeNet carries no runtime data. During lowering to TTKernel,
    PipeNet operations are expanded and removed:
    - ttl.create_pipenet %pipe1, %pipe2, ... → stores pipe list in operation operands
    - ttl.if_pipe_src %net → compiler extracts pipes from create_pipenet, inlines region
      body for each pipe where current core matches src_core, generates ttkernel NOC operations
    - ttl.if_pipe_dst %net → compiler extracts pipes, inlines body for cores matching dst_core
      or dst_core_range, generates ttkernel NOC operations
    - After TTLLowerDataMovement pass: PipeNet SSA values and operations are removed, replaced
      with ttkernel.noc_async_write and semaphore operations

    Topology access: Compiler passes query the defining ttl.create_pipenet operation
    to extract the variadic pipe operands. Standard MLIR def-use chain traversal.

    Python API: ttl.PipeNet([pipe1, pipe2, ...])

    The PipeNet type is used with ttl.if_pipe_src and ttl.if_pipe_dst operations to
    execute code conditionally based on the current core's role in the network.
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

TTL reuses the following tt-mlir attributes instead of defining custom ones:

| TTL Usage | tt-mlir Attribute | Source File |
|-----------|-------------------|-------------|
| Memory space | `ttcore::MemorySpaceAttr` | `TTCoreOpsEnums.td` |
| Grid topology | `ttcore::GridAttr` | `TTCoreOpsAttrs.td` |
| Tensor layout | `ttnn::LayoutAttr` | `TTNNOpsAttrs.td` |
| Core ranges | `ttnn::CoreRangeSetAttr` | `TTNNOpsAttrs.td` |
| Distribution strategy | `ttnn::TensorMemoryLayoutAttr` | `TTNNOpsEnums.td` |

TTL-specific attributes are defined below:

```tablegen
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

def TTL_TensorEncodingAttr : TTL_Attr<"TensorEncoding", "tensor_encoding"> {
  let summary = "TTL tensor encoding combining memory space and layout";
  let parameters = (ins
    "ttcore::MemorySpaceAttr":$memorySpace,  // Reuse ttcore::MemorySpaceAttr from tt-mlir
    "ttnn::LayoutAttr":$layout               // Reuse ttnn::LayoutAttr from tt-mlir
  );
  let assemblyFormat = "`<` $memorySpace `,` $layout `>`";
  let description = [{
    Encoding attached to `tensor<...>` types to carry both the memory-space
    placement (DeviceL1/DeviceDRAM/System) and layout metadata (tiled, sharded, interleaved, etc.).

    Memory space: Reuses ttcore::MemorySpaceAttr from tt-mlir to maintain consistency
    with the rest of the tt-mlir ecosystem. Valid values for tensors:
    - DeviceDRAM: Device HBM (default for TTNN tensors)
    - DeviceL1: On-chip SRAM
    - System/SystemMMIO: Host memory (for host-side tensors)

    Layout attribute: Reuses ttnn::LayoutAttr from tt-mlir to maintain compatibility with
    runtime descriptors and avoid duplicate layout representation. The layout attribute
    captures tiling patterns, sharding configuration, and memory layout details.

    Tile layout information: For tiled tensors, the layout attribute specifies
    tile dimensions (e.g., 32x32, 16x32) and tiling pattern. The tensor element type
    remains scalar (f32, f16, bf16). During lowering, the compiler generates ttcore::TileType
    values as needed for tile-level operations.

    Note: RegisterDst is not a valid memory space for tensors. DST registers are managed
    exclusively by the TTLAssignDSTRegisters pass and do not participate in the
    tensor memory space system.
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
// TTL IR (%tensor : tensor<..., #ttl.tensor_encoding<DeviceDRAM,
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
        shard_id = ttl.core(dims=1)

        o_blk = out_cb.wait()
        xf = ttl.copy(o_blk, out[shard_id])  # Write shard
        xf.wait()
        out_cb.pop()

    return ttl.Program(compute, dm_reader, dm_writer)(a, b, out)
```

**TTL IR (After Python AST Compilation)**:

```mlir
ttl.kernel @sharded_elementwise_add(
    %a: tensor<64x64xf32, #ttl.tensor_encoding<DeviceDRAM,
      #ttl.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>>>,
    %b: tensor<64x64xf32, #ttl.tensor_encoding<DeviceDRAM,
      #ttl.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>>>,
    %out: tensor<64x64xf32, #ttl.tensor_encoding<DeviceDRAM,
      #ttl.layout<sharded, grid=[2,2], tiles_per_shard=[1,1]>>>
) attributes {grid = #ttcore.grid<2x2>, block_factors = [[1,1], [1,1], [1,1]]} {

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
    %shard_id = ttl.core(dims=1) : index

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
    %shard_id = ttl.core(dims=1) : index

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
