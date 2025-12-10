# TTL Dialect Implementation and Runtime Integration

**Part of**: [TTL Dialect Design Plan](01_TTL_Dialect_Plan.md)

This document covers Python integration, implementation roadmap, TTNN runtime
integration, future evolution, success criteria, design rationale, and
references.

## Table of Contents

- [7. Python Integration](#7-python-integration)
  - [7.1 Frontend Compilation](#71-frontend-compilation)
  - [7.2 Operator Mapping](#72-operator-mapping)
  - [7.3 API Entry Point](#73-api-entry-point)
- [8. Implementation Roadmap](#8-implementation-roadmap)
- [9. TTNN Runtime Integration](#9-ttnn-runtime-integration)
- [10. Future Evolution](#10-future-evolution)
  - [10.1 Microcore Model (Post-MVP)](#101-microcore-model-post-mvp)
  - [10.2 Direct C++ Backend (Post-MVP Extension)](#102-direct-c-backend-post-mvp-extension)
  - [10.3 Distributed Tensor Type (Major Extension)](#103-distributed-tensor-type-major-extension)
  - [10.4 Transform Dialect Integration for Scheduling](#104-transform-dialect-integration-for-scheduling)
- [11. Success Criteria](#11-success-criteria)
- [12. Appendix: Design Rationale](#12-appendix-design-rationale)
- [13. References](#13-references)

## Related Documents

- [Back to Overview](01_TTL_Dialect_Plan.md)
- [Type System](02_TTL_Type_System.md)
- [Compute Operations](03_TTL_Compute_Operations.md)
- [Data Movement Operations](04_TTL_Data_Movement_Operations.md)
- [Compilation Pipeline](06_TTL_Compilation_Pipeline.md)



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

### Prerequisites and Dependencies

Before implementing TTL, the following operations need to be added to TTKernel
dialect in tt-mlir:

1. **TRID barrier operations** (required for per-transfer wait semantics):
   - `ttkernel.noc_async_read_barrier_with_trid(trid, noc)`
   - `ttkernel.noc_async_write_barrier_with_trid(trid, noc)`
   - `ttkernel.noc_async_write_set_trid(trid, noc)`
   - Straightforward additions following existing barrier patterns
   - TT-Metal runtime already provides these in `tt_metal/hw/inc/dataflow_api.h`

2. **TensorAccessor operations** (required for shard/page access):
   - `ttkernel.noc_async_read_shard(shard_id, accessor, dst_addr, noc)`
   - `ttkernel.noc_async_write_shard(shard_id, accessor, src_addr, noc)`
   - `ttkernel.noc_async_read_page(page_id, accessor, dst_addr, offset, noc)`
   - `ttkernel.noc_async_write_page(page_id, accessor, src_addr, offset, noc)`
   - Follow existing TTKernel operation patterns
   - TT-Metal runtime provides these functions

3. **TTNN runtime interface validation** (required for kernel execution):
   - Status: Validated in bnorris/ttnn-elementwise-example branch
   - Confirmed `ttnn.generic_op` accepts C++ kernel source and descriptors
     matching TTL target format
   - Tested with elementwise and multicore examples using CB-based kernels
     similar to proposed TTL-generated code

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

4. `TTLLowerSynchronization` - CB ops → TTKernel CB ops

5. `TTLInferPipeSemaphores` - Automatic semaphore creation for pipes
   - Analyze pipe nets to determine required semaphore count
   - Create two semaphores per pipe: ready and validity
   - Insert semaphore operations in if_src and if_dst bodies
   - Handle unicast vs multicast patterns
   - See
     [05_TTL_Multicast_Implementation.md](05_TTL_Multicast_Implementation.md)
     for patterns

**Deliverable**: Kernels with reduction/broadcast/fusion lower to TTKernel (and
C++), pipes lower with automatic semaphore synchronization

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
    "ttnn::TensorMemoryLayoutAttr":$strategy,  // Reuse ttnn::TensorMemoryLayoutAttr from tt-mlir
    "ttcore::MemorySpaceAttr":$memorySpace     // Reuse ttcore::MemorySpaceAttr from tt-mlir
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
- Cleaner semantic representation that matches the python dsl
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
