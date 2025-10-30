# Hitchhiker's Guide to TT-Lang and D2M

tt-mlir compiles Python kernel code to run on Tenstorrent AI accelerators. This guide documents the DSL path (pykernel) for writing custom data movement and compute kernels.

**Build Locally**

```bash
source env/activate
cmake -G Ninja -B build \
    -DTTMLIR_ENABLE_RUNTIME=OFF \
    -DTTMLIR_ENABLE_STABLEHLO=OFF \
    -DTT_RUNTIME_ENABLE_PERF_TRACE=OFF \
    -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DTTMLIR_ENABLE_DEBUG_STRINGS=OFF \
    -DTTMLIR_ENABLE_EXPLORER=OFF \
    -DTTMLIR_ENABLE_RUNTIME_TESTS=OFF \
    -DTTMLIR_ENABLE_PYKERNEL=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_PARALLEL_LEVEL=4 \
    -DFLATBUFFERS_COMPILER=/opt/homebrew/bin/flatc
cmake --build build
export SYSTEM_DESC_PATH=$(pwd)/system_desc.ttsys
```

**Test Locally**

```bash
# Run all compiler tests
cmake --build build --target check-ttmlir

# Run specific lit test
llvm-lit test/ttmlir/Dialect/D2M/allocate/generic_form_streams_matmul.mlir

# Run Python DSL tests
pytest test/python

# Run single pykernel test
python test/pykernel/gen/custom_dm_matmul.py
```

**Pretty Printer**

Use `ttc` to view MLIR at any stage:

```bash
# Compile and view MLIR after each pass
ttc test.mlir -ttir-to-ttmetal-frontend -print-ir-after-all

# View final MLIR
ttc test.mlir -ttir-to-ttmetal-backend

# Get flatbuffer binary
ttc test.mlir -ttir-to-ttmetal-backend -ttmetal-to-flatbuffer-bin -o out.ttb
```

**Mac Support**

Compilation works on Mac. Runtime requires Linux with Tenstorrent hardware. Use SSH workflow: develop on Mac, test on Linux server.

---

## The Complete Pipeline

Three paths exist for generating code. Focus here is the DSL path.

**Path 1: DSL Path (pykernel) - This Document**

```
Python DSL Code
  @pykernel_gen(grid=(2,2))
  def matmul(lhs, rhs, out):
      @compute() ...
      @datamovement() ...
      ↓
  d2m_api.py compiles each thread to MLIR
    - Creates MetalLayoutAttr with shapes/grid
    - Wraps Stream() args with stream_layout ops
    - Compiles thread AST to D2M dialect
    - Python operators generate linalg.generic blocks with D2M tile ops
    - Glues threads into d2m.generic op
      ↓
  D2M Dialect IR (tensor<4x4x!ttcore.tile<32x32>>)
      ↓
  Frontend Passes
    - d2m-generic-replace-globals (swap globals for function args)
    - fusion, canonicalization
      ↓
  Bufferization (tensor → memref)
      ↓
  Middleend Passes
    - d2m-allocate (assign L1 addresses)
    - d2m-generic-lower-dmas (DMA to hardware ops)
      ↓
  TTKernel Dialect
      ↓
  EmitC + Flatbuffer
      ↓
  Binary runs on hardware
```

**Path 2: TTNN Bindings**

```
Python TTNN API
  ttnn.matmul(a, b)
    ↓
TTIR Dialect
    ↓
ttir-to-d2m conversion
    ↓
(joins DSL path at D2M dialect)
```

**Path 3: TTIR Bindings (Documented Path)**

```
Framework Inputs (JAX via tt-xla, PyTorch via tt-torch, ONNX via tt-forge-FE)
    ↓
StableHLO/Torch IR
    ↓
TTIR Dialect
    ↓
ttir-to-d2m conversion
    ↓
(joins DSL path at D2M dialect)
```

The DSL path creates D2M dialect directly. TTIR paths convert high-level framework ops (JAX, PyTorch, ONNX, TensorFlow) through StableHLO/Torch IR to TTIR, then to D2M. All paths converge at D2M dialect, then flow through TTKernel, before diverging to either TTNN dialect or TTMetal dialect backends. Metalium C++ APIs (ttnn, tt-metal) are also available at the lower levels of the stack.

**Comparison: DSL vs Framework Paths**

| Aspect | DSL Path (pykernel) | TTNN Bindings | TTIR Path (JAX/PyTorch) |
|--------|---------------------|---------------|------------------------|
| Entry Point | Python DSL with @pykernel_gen | Python ttnn API | Framework models (JAX/PyTorch/ONNX) |
| Control Level | Full control over cores, DMA, memory | High-level ops only | Model-level ops |
| Custom Kernels | Yes - write compute & datamovement threads | No | No |
| Compilation | Direct to D2M dialect | TTNN → TTIR → D2M | Framework IR → StableHLO/Torch IR → TTIR → D2M |
| Use Case | Custom kernels, research, optimization | Quick deployment, standard ops | Production models, framework integration |
| Hardware Model | Explicit (grid, shards, L1/DRAM) | Abstracted | Abstracted |
| Example | See test/pykernel/gen/ | ttnn.matmul(a, b) | JAX/PyTorch model compilation |

Trade-off: DSL path requires understanding hardware model but provides maximum flexibility. Framework paths hide hardware details but offer less control.

---

## DSL Basics

**Minimal Example**

```python
from pykernel.d2m_api import *
import torch

@pykernel_gen(
    grid=(2, 2),            # 2x2 grid of cores
    block_factors=[         # Tile counts per core
        (1, 1),  # lhs: 1x1 tiles
        (1, 1),  # rhs: 1x1 tiles
        (1, 1),  # out: 1x1 tiles
    ],
)
def add(lhs, rhs, out, block_factors=None, grid=None):
    @compute()
    async def add_kernel(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        lhs_shard = lhs_cb.pop()        # Read from CB
        rhs_shard = rhs_cb.pop()
        out_shard = out_cb.reserve()    # Reserve space
        result = lhs_shard + rhs_shard  # Compute
        out_shard.store(result)         # Write
        out_cb.pop()                    # Free (compute must pop)

    @datamovement()
    async def dm_reader(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        cy = core_index(0)
        cx = core_index(1)
        idx = cy * 2 + cx  # Assuming 2x2 grid

        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_stream[idx, 0], lhs_shard)
        tx.wait()

        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_stream[idx, 0], rhs_shard)
        tx.wait()

    return Program(add_kernel, dm_reader)(lhs, rhs, out)

lhs = torch.randn(128, 128)
rhs = torch.randn(128, 128)
out = torch.zeros(128, 128)
add(lhs, rhs, out)
```

**Thread Types**

- `@compute()`: Runs on Tensix compute cores. Does math.
- `@datamovement()`: Runs on data movement processors. Does DMAs.

Each thread compiles separately, then gets glued into `d2m.generic` op.

**Grid and Block Factors**

```python
grid=(2, 2)              # 2x2 = 4 cores total
block_factors=[
    (1, 1),  # Input 0: 1x1 tiles per core (32x32 elements)
    (1, 1),  # Input 1: 1x1 tiles per core
    (1, 1),  # Output: 1x1 tiles per core
]
```

For 128x128 tensor:
- Logical shape: 128x128 elements
- Physical shape: 4x4 tiles (128/32 = 4)
- Device shape: [2, 2, 2, 2] (grid + shard)
  - First half [2, 2]: grid dimensions
  - Second half [2, 2]: tiles per core

**Circular Buffers**

Bounded FIFO queues for passing data between threads.

```python
# Producer pattern (datamovement)
shard = cb.reserve()           # Blocks if full
tx = dma(stream[x, y], shard)
tx.wait()
# No .push() - implicit after reserve

# Consumer pattern (compute)
shard = cb.pop()               # Blocks if empty
result = compute(shard)

# Output pattern (compute writes)
out_shard = out_cb.reserve()      # Get space to write
out_shard.store(result)           # Write data
out_cb.pop()                      # Signal "data ready" for consumption
```

**Thread Communication via Circular Buffers**

```
Datamovement Thread          Compute Thread
      (Producer)              (Consumer)
          │                       │
          │                       │
    cb.reserve()                  │
          │                       │
     [CB: Space allocated]        │
          │                       │
    dma(stream, shard)            │
          │                       │
      tx.wait()                   │
          │                       │
  [CB: Data available]            │
          │                   cb.pop()
          │                       │
          │              [CB: Data accessed]
          │                       │
          │                   compute()
          │                       │
          │              [CB: Space freed]
          │                       │

Blocking is implicit:
- reserve() blocks if CB is full
- pop() blocks if CB is empty

Only tx.wait() is explicit for DMA completion.
The CB ensures synchronization automatically.
```

**CB Lowering Pipeline**

Circular buffers transform through four levels: Python DSL, D2M dialect, TTKernel dialect, and C++ code generation.

```
Python DSL
  input_cb.pop()
    ↓
D2M Dialect
  d2m.wait %cb : !d2m.cb<memref<...>>
    ↓
TTKernel Dialect
  ttkernel.cb_wait_front(%cb, %numPages) : !ttkernel.cb<4, tile>
  ttkernel.cb_pop_front(%cb, %numPages)   (at block end)
    ↓
EmitC
  emitc.call_opaque "cb_wait_front"(%cb, %numPages) : (::tt::CB, i32)
  emitc.call_opaque "cb_pop_front"(%cb, %numPages)
    ↓
C++ Code
  cb_wait_front(cb, numPages);
  // ... computation ...
  cb_pop_front(cb, numPages);
    ↓
tt-metal Runtime
  Hardware semaphores + L1 buffer management
```

Single D2M operation expands into acquire/release pair at TTKernel level.

| D2M Operation | TTKernel Acquire | TTKernel Release |
|---------------|------------------|------------------|
| `d2m.wait` (consumer) | `cb_wait_front` | `cb_pop_front` |
| `d2m.reserve` (producer) | `cb_reserve_back` | `cb_push_back` |

The `D2MCBOpRewriter` template converts:
- `d2m.wait` → `ttkernel.cb_wait_front` + `ttkernel.cb_pop_front`
- `d2m.reserve` → `ttkernel.cb_reserve_back` + `ttkernel.cb_push_back`

**CB Operation Asymmetry**

The same `pop()` method generates `d2m.wait` in both cases, but semantically:
- On input CB: "wait for producer to provide data"
- On output CB: "wait for consumer to consume data" (signal production complete)

**Input CB (datamovement writes, compute reads):**

```python
# Datamovement (producer):
shard = input_cb.reserve()        # Get space to write
tx = dma(stream[idx], shard)      # Write data
tx.wait()
# Implicit: signal "data ready" at block end

# Compute (consumer):
shard = input_cb.pop()            # Wait for data + get access
# ... use shard ...
# Implicit: signal "consumed" at block end
```

**Output CB (compute writes, datamovement reads):**

```python
# Compute (producer):
out_shard = out_cb.reserve()      # Get space to write
out_shard.store(result)           # Write data
out_cb.pop()                      # ⚠️ Signal "data ready" (NOT consuming!)

# Datamovement (consumer):
shard = out_cb.pop()              # Wait for data + get access
tx = dma(shard, stream[idx])      # Read data
tx.wait()
# Implicit: signal "consumed" at block end
```

So `pop()` has two different semantic meanings depending on which thread uses it and on which CB. For output CBs, compute needs two operations (`reserve()` + `pop()`) while datamovement only needs one (`pop()`). The second `pop()` in compute ensures the release happens at the right point (immediately after writing), not at the end of the entire block.

**Operation Summary Table:**

| Role | Thread | Input CB | Output CB |
|------|--------|----------|-----------|
| Producer | Datamovement | `reserve()` → write → implicit | - |
| Producer | Compute | - | `reserve()` → write → `pop()` |
| Consumer | Compute | `pop()` → use → implicit | - |
| Consumer | Datamovement | - | `pop()` → read → implicit |

Both `pop()` and `reserve()` return memref views (zero-copy). DMA operations perform the actual data movement to/from these views.

**Streams**

Streams wrap function arguments for async data movement.

```python
lhs_stream = Stream(lhs)  # Mark lhs as streamed input

# Later, in datamovement:
tx = dma(lhs_stream[idx, 0], lhs_shard)  # DMA from DRAM to L1
```

Streams create `d2m.stream_layout` ops with backing storage for multi-buffering.

**Operators**

```python
# Element-wise
result = a + b
result = a - b
result = a * b
result = a / b

# Matrix multiply
result = a @ b

# Memory ops
out_shard.store(result)
```

**Linalg.Generic Blocks**

Python operators compile to linalg.generic blocks containing D2M tile operations.

```mlir
// Python: result = a + b
linalg.generic {
  indexing_maps = [map<(d0,d1)->(d0,d1)>, map<(d0,d1)->(d0,d1)>, map<(d0,d1)->(d0,d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%a, %b : tensor<2x2x!ttcore.tile<32x32>>) outs(%out : tensor<2x2x!ttcore.tile<32x32>>) {
  %result = d2m.tile_add(%tile_a, %tile_b) : (!ttcore.tile<32x32>, !ttcore.tile<32x32>) -> !ttcore.tile<32x32>
  linalg.yield %result
}

// Python: result = a @ b
linalg.generic {
  indexing_maps = [map<(d0,d1,d2)->(d0,d2)>, map<(d0,d1,d2)->(d2,d1)>, map<(d0,d1,d2)->(d0,d1)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%a, %b) outs(%out) {
  %result = d2m.tile_matmul(%tile_a, %tile_b, %tile_acc)
  linalg.yield %result
}
```

Linalg defines iteration structure (which tiles to process, in what order). D2M tile ops define computation (hardware intrinsics on single tiles). This separation enables linalg optimizations (fusion, tiling) before hardware lowering.

**Pass Flow:**

Pykernel path generates linalg blocks in Python (d2m_api.py). Linalg survives through bufferization (tensor → memref conversion). Canonicalizer lowers linalg to affine loops. D2MToTTKernel converts D2M tile ops to TTKernel intrinsics.

Other paths (TTIR, TTNN) use ArithToD2MTileOps pass to convert arith ops to D2M tiles after entering the D2M dialect. Pykernel skips this by generating linalg blocks upfront.

**Tile and Block Operations**

D2M operations exist at two levels:

| Level | Type | Operations | Purpose |
|-------|------|------------|---------|
| **Block** | tensor<2x2x!ttcore.tile<32x32>> | Python operators (a + b, a @ b) | Multiple tiles. Requires iteration. |
| **Tile** | !ttcore.tile<32x32, f32> | d2m.tile_add, d2m.tile_matmul | Hardware intrinsics. Single 32x32 block. |

**Block Operations**

```python
result = a + b  # Block operation
```

The compiler generates linalg.generic to define iteration over tiles, with D2M
  tile ops inside performing the actual computation:

```mlir
linalg.generic {maps=[...], iters=["parallel", "parallel"]} ins(%a, %b) outs(%out) {
  %result = d2m.tile_add(%tile_a, %tile_b)
  linalg.yield %result
}
```

**Tile Operations**

```mlir
%result = d2m.tile_add(%a, %b)
```

Single tile computation. Maps to hardware primitive. No iteration logic.

After bufferization and loop lowering, only tile-level operations remain. The `tile_add` example after lowering:
```mlir
scf.for %i ... {
  scf.for %j ... {
    d2m.tile_add(...)  // tile operation
  }
}
```

**DMA Operations**

```python
# DRAM → L1
tx = dma(stream[idx, 0], l1_buffer)
tx.wait()

# L1 → L1 with multicast (one source to many destinations)
tx = dma(src, dst, core=(cy, 1), mcast=(1, GX-1))
tx.wait()

# L1 → DRAM
tx = dma(l1_buffer, stream[idx, 0])
tx.wait()
```

**Semaphores**

For multi-core synchronization.

```python
# Producer signals consumers
sem.set(1, core=(cy, 1), mcast=(1, GX-1))

# Consumer waits for producer
sem.wait(1, reset=0)

# Increment counter
sem.inc(1, core=(cy, 0))
```

Note: Multicast is supported for `set()` (broadcast value to multiple cores) but not for `inc()` (increments single remote core only).

---

## Shape Transformations

Shapes flow through pipeline:

```
  User Code (Python)
      torch.randn(128, 128)
      │
      ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ LOGICAL SHAPE: [128, 128]                                        │
  │ - Original tensor dimensions from user                           │
  │ - Stored in MetalLayoutAttr.logicalShape                         │
  └─────────────────────────────────────────────────────────────────┘
      │
      │ applyCollapsedIntervalsAndAlignments()
      │ TTCoreOpsTypes.cpp:733-792
      │ - Apply collapse_intervals to group dimensions
      │ - Apply dim_alignments to align each dimension
      │
      ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ COLLAPSED SHAPE: [128, 128]                                      │
  │ - After applying collapse intervals and alignments               │
  │ - Still in elements, not tiles                                   │
  └─────────────────────────────────────────────────────────────────┘
      │
      │ getPhysicalShape(tileShape)
      │ TTCoreOpsTypes.cpp:796-811
      │ - Divide last 2 dimensions by tile size [32, 32]
      │   [128, 128] / [32, 32] = [4, 4]
      │
      ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ PHYSICAL SHAPE: [4, 4]                                           │
  │ - Dimensions in tiles, not elements                              │
  │ - This is the "actual tensor shape" in tile units                │
  └─────────────────────────────────────────────────────────────────┘
      │
      │ getDeviceShape(gridShape, tileShape)
      │ TTCoreOpsTypes.cpp:816-879
      │ - gridShape = [2, 2] (from @pykernel_gen(grid=(2,2)))
      │ - Distribute physical shape across grid
      │   deviceShape = [2, 2] + [4/2, 4/2] = [2, 2, 2, 2]
      │
      ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ DEVICE SHAPE: [2, 2, 2, 2]                                       │
  │ - Rank = 4 (doubled from original 2D tensor)                     │
  │ - First half [2, 2] = GRID SHAPE (number of cores)               │
  │ - Second half [2, 2] = SHARD SHAPE (tiles per core)              │
  └─────────────────────────────────────────────────────────────────┘
      │
      │ Split into components via DeviceLayoutInterface
      │ TTCoreAttrInterfaces.td:28-44
      │
      ├──────────────────────────┬──────────────────────────┐
      ▼                          ▼                          ▼
  ┌──────────────┐      ┌───────────────┐      ┌──────────────────┐
  │ GRID SHAPE   │      │ SHARD SHAPE   │      │ Total Elements   │
  │ [2, 2]       │      │ [2, 2]        │      │ per core:        │
  │              │      │               │      │ 2×2 tiles =      │
  │ - First half │      │ - Second half │      │ 64×64 elements   │
  │   of device  │      │   of device   │      │                  │
  │   shape      │      │   shape       │      │                  │
  └──────────────┘      └───────────────┘      └──────────────────┘
```

**Memory Layout**

Grid of 2×2 cores, each with 2×2 tiles (64×64 elements):

```
┌─────────────────┬─────────────────┐
│ Core (0,0)      │ Core (0,1)      │
│ Shard: [2×2]    │ Shard: [2×2]    │
│ = 64×64 elem    │ = 64×64 elem    │
├─────────────────┼─────────────────┤
│ Core (1,0)      │ Core (1,1)      │
│ Shard: [2×2]    │ Shard: [2×2]    │
│ = 64×64 elem    │ = 64×64 elem    │
└─────────────────┴─────────────────┘
```

Total tensor: 128×128 elements = 4×4 tiles

**Shape Transformation Table**

| Stage | Shape | Units | Example (128x128, grid 2x2) | Code Location |
|-------|-------|-------|----------------------------|---------------|
| Logical | [128, 128] | Elements | User input | Python DSL |
| Collapsed | [128, 128] | Elements | After alignment | TTCoreOpsTypes.cpp:733 |
| Physical | [4, 4] | Tiles | 128/32 = 4 | TTCoreOpsTypes.cpp:796 |
| Device | [2, 2, 2, 2] | Tiles | grid + shard | TTCoreOpsTypes.cpp:816 |
| Grid | [2, 2] | Cores | First half | getGridShape() |
| Shard | [2, 2] | Tiles/core | Second half | getShardShape() |

---

## Layout Attributes

Three types exist:

**MetalLayoutAttr (High-Level, Pre-Bufferization)**

Used for tensors before bufferization.

```mlir
#ttcore.metal_layout<
  logical_shape = 128x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0,1], [1,2]]>,
  undef,
  l1>
```

Contains:
- Logical shape (user dimensions)
- Grid distribution info (via collapsed_intervals)
- Memory space (L1, DRAM)
- Optional index_map for views

Created in Python by `create_metal_layout()` in d2m_api.py:41.

**ViewLayoutAttr (Low-Level, Post-Bufferization)**

Used for memrefs that are views (not storage).

```mlir
#ttcore.view<map(4)>  // Identity map for rank 4
```

Contains affine map for indexing. Created automatically by bufferization when MetalLayoutAttr has index_map.

**ShardLayoutAttr (Low-Level, Post-Bufferization)**

Used for memrefs that are storage (not views).

```mlir
#ttcore.shard<8192>  // Stride in bytes
```

Contains stride info for physical layout. Created automatically by bufferization for storage operands.

**Bufferization Converts Layouts**

```
tensor<..., #ttcore.metal_layout<...>>
    ↓ one-shot-bufferize
memref<..., #ttcore.view<...>, #ttcore.memory_space<l1>>
  OR
memref<..., #ttcore.shard<...>, #ttcore.memory_space<l1>>
```

Decision: If MetalLayoutAttr has index_map → ViewLayoutAttr, else → ShardLayoutAttr.

**Layout Attributes Comparison**

| Aspect | MetalLayoutAttr | ViewLayoutAttr | ShardLayoutAttr |
|--------|----------------|----------------|-----------------|
| Used For | Tensors | Memrefs (views) | Memrefs (storage) |
| When | Pre-bufferization | Post-bufferization | Post-bufferization |
| Contains | Logical shape, grid, memory space, index_map | Affine map | Stride |
| Purpose | High-level layout spec | Logical indexing | Physical storage |
| Created By | Python (d2m_api.py) | Bufferization | Bufferization |
| Is View? | N/A | Yes | No |
| Has Storage? | N/A | No | Yes |
| Example | `#ttcore.metal_layout<logical_shape=128x128, ...>` | `#ttcore.view<map(4)>` | `#ttcore.shard<8192>` |

---

## Pipeline Stages

```
Stage 1: DSL Compilation (d2m_api.py)
├─ Parse @pykernel_gen decorator
├─ Compile each thread separately
│  ├─ @compute() → D2MGenericCompiler
│  ├─ @datamovement() → D2MGenericCompiler
│  └─ Create globals for captured values
├─ Create MetalLayoutAttr for each arg
├─ Wrap Stream() args with stream_layout
└─ Glue threads into d2m.generic op

Stage 2: Frontend (TTMetalPipelines.cpp:73)
├─ d2m-generic-replace-globals
│  └─ Swap globals for function args
├─ Fusion passes
└─ Canonicalization

Stage 3: Bufferization (TTMetalPipelines.cpp:114)
├─ one-shot-bufferize
│  ├─ tensor → memref
│  ├─ MetalLayoutAttr → ViewLayoutAttr/ShardLayoutAttr
│  └─ Strips layouts from function boundaries
└─ d2m-bufferize-function-args
   └─ Restore layouts on function args

Stage 4: Middleend (TTMetalPipelines.cpp:119)
├─ d2m-insert-explicit-streams (now no-op)
├─ d2m-allocate
│  └─ Assign L1/DRAM addresses
└─ d2m-generic-lower-dmas
   └─ DMA ops → NOC operations

Stage 5: Backend
├─ TTKernel conversion
├─ EmitC passes
└─ Flatbuffer generation
```

---

## Important Passes

**Frontend (Before Bufferization)**

- `ttir-to-d2m`: Converts TTIR ops to D2M dialect (for TTIR path)
- `d2m-generic-replace-globals`: Replaces globals with function args
  - Each thread compiles separately with globals
  - Python glues threads together
  - Pass fixes up global references to use function args
  - Location: lib/Dialect/D2M/Transforms/GenericReplaceGlobals.cpp

**Bufferization**

- `one-shot-bufferize`: Converts tensor types to memref types
  - Configuration: IdentityLayoutMap strips layouts from function args
  - Location: TTMetalPipelines.cpp:114
  - Strips MetalLayoutAttr from function boundaries

**Middleend (After Bufferization)**

- `d2m-bufferize-function-args`: Restores layout attributes on function argument memrefs
  - Runs immediately after one-shot-bufferize (TTMetalPipelines.cpp:117)
  - Required because bufferization with IdentityLayoutMap strips MetalLayoutAttr from function boundaries
  - Re-applies layout attributes to memrefs based on original tensor layouts

- `d2m-insert-explicit-streams`: Creates stream_layout ops (now created in Python)
  - Location: lib/Dialect/D2M/Transforms/InsertExplicitStreams.cpp
  - Legacy pass that read d2m.stream attributes to create stream_layout ops
  - Stream creation moved to d2m_api.py:313 (_create_stream_layout_for_input)
  - Pass now typically no-op as streams already exist in IR

- `d2m-allocate`: Assigns L1/DRAM addresses
  - Location: lib/Dialect/D2M/Transforms/Allocate.cpp
  - Critical pass: determines where data lives

- `d2m-generic-lower-dmas`: Lowers DMA ops to hardware NOC operations
  - Location: lib/Dialect/D2M/Transforms/GenericLowerDMAs.cpp
  - Converts abstract DMAs to specific NOC commands

**Backend**

- `ttmetal-to-flatbuffer-bin`: Generates final binary
- Various EmitC passes for C code generation

---

## Important Types

**Tensor Types**

```mlir
// Simple tensor (pre-tiling)
tensor<128x128xf32>

// Tiled tensor with layout
tensor<4x4x!ttcore.tile<32x32, f32>, #ttcore.metal_layout<...>>
```

Shape is in tiles when element type is !ttcore.tile. Encoding is layout attribute.

**Memref Types**

```mlir
// View memref (streaming, indexable)
memref<2x2x2x2x!ttcore.tile<32x32, f32>,
       #ttcore.view<map(4)>,
       #ttcore.memory_space<l1>>

// Storage memref (backing buffer)
memref<2x2x2x2x!ttcore.tile<32x32, f32>,
       #ttcore.shard<8192>,
       #ttcore.memory_space<l1>>
```

Shape is device shape [grid_y, grid_x, shard_y, shard_x].

**Tile Type**

```mlir
!ttcore.tile<32x32, f32>
```

Hardware primitive. Usually 32x32 elements.

**Circular Buffer Type**

```mlir
!d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>>>
```

Wraps underlying memref for producer-consumer sync.

**DMA Transaction Type**

```mlir
!d2m.mem_tx
```

Handle for async DMA operations. Call .wait() to block.

**Semaphore Type**

```mlir
!d2m.semaphore
```

For multi-core synchronization primitives.

---

## Stream Creation

Streams moved from pass-based to Python-based creation.

**Old Flow (Pass-Based)**

```
Python: Stream(lhs)
    ↓
d2m_api.py tags arg: {d2m.stream = true}
    ↓
Creates d2m.generic (no stream_layout yet)
    ↓
... bufferization ...
    ↓
D2MInsertExplicitStreams pass reads attribute
    ↓
Creates stream_layout ops
```

**New Flow (Python-Based)**

```
Python: Stream(lhs)
    ↓
d2m_api.py creates stream_layout immediately
    ↓
Creates d2m.generic with streams
    ↓
... bufferization ...
    ↓
D2MInsertExplicitStreams is no-op
```

Location: d2m_api.py:203 (_create_stream_layout_for_input)

Benefits: Streams visible from start of IR. Simpler pipeline. Better debugging.

**stream_layout Op Structure**

```mlir
%storage = d2m.empty() : tensor<4x4x!ttcore.tile<32x32>>
%stream = d2m.stream_layout(%input, %storage)
  : (tensor<4x4x!ttcore.tile<32x32>>,    // input
     tensor<4x4x!ttcore.tile<32x32>>)    // storage
  -> tensor<4x4x!ttcore.tile<32x32>>     // result (view)
```

Input: source data. Storage: backing buffers. Result: view for indexing.

---

## Multi-Core Patterns

**Grid Indexing**

```python
cy = core_index(0)  # Y coordinate
cx = core_index(1)  # X coordinate
idx = cy * GX + cx  # Linear index
```

**Multicast DMA**

One core sends to multiple cores.

```python
# Core at (cy, 0) sends to all cores in row
tx = dma(src, dst, core=(cy, 1), mcast=(1, GX-1))
```

`core=(cy, 1)`: destination start. `mcast=(1, GX-1)`: multicast span (1 row, GX-1 columns).

**Turn-Based Execution**

Use semaphores to serialize operations.

```python
# Core 0 does DMA first
if cx == 0:
    tx = dma(stream[idx], shard)
    tx.wait()
    # Signal others
    sem.set(1, core=(cy, 1), mcast=(1, GX-1))
else:
    # Wait for core 0
    sem.wait(1, reset=0)
```

See: test/pykernel/gen/custom_dm_turn_based_matmul.py

---

**FPU and SFPU Compute Units**

**FPU (Fused Processing Unit):**
- Specialized for matrix multiplication and reduction operations
- Operations: matmul, matmul_block, reduce_sum, reduce_max
- Ternary operation: `A @ B + C` (fused multiply-add)
- Higher throughput for dense linear algebra

**SFPU (Special Function Processing Unit):**
- Specialized for element-wise operations
- Operations: exp, log, sin, cos, sqrt, add, mul, div, etc.
- Operates in-place on destination register
- Lower precision, higher throughput for transcendental functions

**Lowering Differences:**

FPU operations:
```cpp
// D2MFPUOpsRewriter generates:
MatmulInitOp(cbA, cbB, outCB, transpose);
MatmulInitShortOp(cbA, cbB, transpose);
MatmulTilesOp(cbA, cbB, tileA, tileB, tileC, transpose);
```

SFPU operations:
```cpp
// D2MSFPUOpsRewriter generates:
InitSFPUOp(inCB, outCB);      // Initialize SFPU unit
ExpTileInitOp();              // Initialize exp function
ExpTileOp(dstIdx);            // Execute on tile at dstIdx (in-place)
```

SFPU operations modify tiles in-place in the destination register. FPU operations (matmul) also use DST but have different access patterns for accumulation across the K dimension.

---

## DSL Examples

**Simple Element-Wise Add**

```python
@pykernel_gen(grid=(2,2), block_factors=[(1,1), (1,1), (1,1)])
def add(lhs, rhs, out, block_factors=None, grid=None):
    @compute()
    async def compute_add(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        l = lhs_cb.pop()
        r = rhs_cb.pop()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        cy = core_index(0)
        cx = core_index(1)
        idx = cy * 2 + cx

        l = lhs_cb.reserve()
        tx = dma(lhs_stream[idx, 0], l)
        tx.wait()

        r = rhs_cb.reserve()
        tx = dma(rhs_stream[idx, 0], r)
        tx.wait()

    return Program(compute_add, dm)(lhs, rhs, out)
```

**Matrix Multiply with Multicast**

See: test/pykernel/gen/custom_dm_matmul.py

Pattern:
- Load lhs row on leftmost core, multicast right
- Load rhs column on topmost core, multicast down
- Use semaphores for coordination
- Compute does partial sums

**FlashAttention**

```python
@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def flash_attention(Q, K, V, out, block_factors=None, grid=None):
    Q_stream = Stream(Q)
    K_stream = Stream(K)
    V_stream = Stream(V)
    NUM_KV_BLOCKS = 4

    @compute()
    async def attention_compute(Q_cb, K_cb, V_cb, out_cb):
        m_old = fill(-inf, shape)  # fill() not implemented
        l_old = fill(0, shape)      # fill() not implemented
        O_acc = fill(0, shape)      # fill() not implemented

        for kv_idx in range(NUM_KV_BLOCKS):
            Q_block = Q_cb.pop()
            K_block = K_cb.pop()
            V_block = V_cb.pop()

            S = Q_block @ transpose(K_block)  # transpose() not implemented
            S_scaled = S * (1.0 / sqrt(d_head))  # sqrt() and scalar broadcast not implemented

            m_new = rowmax(S_scaled, m_old)  # rowmax() not implemented
            P = exp(S_scaled - m_new)  # exp() not implemented
            correction = exp(m_old - m_new)  # exp() not implemented
            l_new = correction * l_old + rowsum(P)  # rowsum() not implemented

            O_acc = (l_old / l_new) * correction * O_acc + (P / l_new) @ V_block

            m_old = m_new
            l_old = l_new

        out_block = out_cb.reserve()
        out_block.store(O_acc)
        out_cb.pop()

    @datamovement()
    async def dm_reader(Q_cb, K_cb, V_cb, out_cb):
        idx = core_index(0) * 1 + core_index(1)

        for kv_idx in range(NUM_KV_BLOCKS):
            Q_shard = Q_cb.reserve()
            dma(Q_stream[idx, 0], Q_shard).wait()

            K_shard = K_cb.reserve()
            dma(K_stream[kv_idx, 0], K_shard).wait()

            V_shard = V_cb.reserve()
            dma(V_stream[kv_idx, 0], V_shard).wait()

    return Program(attention_compute, dm_reader)(Q, K, V, out)
```

---

## Tips and Tricks

**Debugging MLIR Output**

IR printing is controlled in d2m_api.py:706-740. The PassManager is configured with:
- `print_after_all=True`: Print IR after each pass
- `print_before_all=True`: Print IR before each pass
- `print_after_failure=True`: Print IR when passes fail

The `use_tile_matmul` flag in d2m_api.py controls matmul lowering. True keeps matmul at tile level (linalg → loops → tile_matmul). False converts to matmul_block optimization. Pykernel uses True.

To save kernel sources, use `kernel_source_mode="store"` parameter in `@pykernel_gen`.

---

## Glossary

**Grid**: 2D array of cores. Example: grid=(2,2) is 4 cores.

**Shard**: Data assigned to one core. Device shape [2,2,2,2] has shard [2,2].

**Tile**: 32x32 block of data. Hardware primitive.

**Physical Shape**: Shape in tiles after alignment and tiling.

**Device Shape**: Shape after grid distribution. Rank doubles: [grid_y, grid_x, shard_y, shard_x].

**Stream**: Async data source. Wraps function argument for DMA.

**Circular Buffer (CB)**: Bounded FIFO queue between threads.

**DMA**: Direct Memory Access. Hardware-accelerated data copy.

**Multicast**: One source to multiple destinations in single DMA.

**NOC**: Network-on-Chip. Hardware interconnect for DMAs.

**L1**: Per-core SRAM. Fast, small (~1.5MB).

**DRAM**: Device DRAM. Slower, larger.

**Linalg.Generic**: MLIR structured operation. Defines iteration (affine maps, iterator types) separate from computation. Wraps D2M tile ops. Enables fusion and optimization.

**Semaphore**: Synchronization primitive for multi-core coordination.

**View Layout**: Memref that indexes into storage. No actual memory.

**Shard Layout**: Memref with actual backing storage.

**Metal Layout**: High-level layout before bufferization. Contains logical shape and grid info.

**Collapsed Intervals**: Encoding of how logical dims map to device dims.

**OOB Val**: Out-of-bounds value. Usually undef.
