# TTL Dialect Compilation Pipeline

**Part of**: [TTL Dialect Design Plan](01_TTL_Dialect_Plan.md)

This document specifies the compilation pipeline, including pass architecture,
lowering strategies, and type conversions.

## Table of Contents

- [5. Compilation Pipeline](#5-compilation-pipeline)
  - [5.1 Pass Architecture](#51-pass-architecture)
  - [5.2 Key Pass Descriptions](#52-key-pass-descriptions)
  - [5.3 Source Location Tracking](#53-source-location-tracking)
  - [5.4 Error Handling and Diagnostics](#54-error-handling-and-diagnostics)
  - [5.5 Control Flow: SCF vs Affine Dialect](#55-control-flow-scf-vs-affine-dialect)
- [6. Type Conversion & Lowering Examples](#6-type-conversion--lowering-examples)
  - [6.1 TTL → TTKernel Type Mapping](#61-ttl--ttkernel-type-mapping)
  - [6.2 TTL → TTKernel Operation Mapping](#62-ttl--ttkernel-operation-mapping)
  - [6.3 Operation Lowering Examples](#63-operation-lowering-examples)

## Related Documents

- [Back to Overview](01_TTL_Dialect_Plan.md)
- [Type System](02_TTL_Type_System.md) - Types being converted
- [Compute Operations](03_TTL_Compute_Operations.md) - Includes DST register
  tiling (section 5.6)
- [Data Movement Operations](04_TTL_Data_Movement_Operations.md) - Data movement
  lowering
- [Implementation and Runtime](07_TTL_Implementation_and_Runtime.md) - Python
  integration



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
  ├─ TTLInferPipeSemaphores - Create semaphores for pipes
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
  ├─ TTLLowerCompute - ttl.add → linalg.generic → (transform tiling) → scf.for + ttkernel.add_tiles
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

**Control Flow and Optimization Strategy**

TTL uses three complementary MLIR components for control flow and optimization:

1. **Affine dialect** for structured loops: TTL kernels iterate over grid dimensions and
   tile indices with static or affine bounds. Affine dialect provides dependence analysis
   that determines when DMA operations can overlap and when barriers can be elided. The
   polyhedral transformations (fusion, tiling, interchange) are directly applicable to
   TTL's access patterns.

2. **SCF dialect** for conditionals: Core-specific conditionals (`if core_x == 0`) and
   dynamic control flow use SCF operations. SCF also provides the outer iteration
   constructs when transform-based tiling generates sub-block loops.

3. **Linalg structured ops** for compute operations: Tensor compute operations lower to
   `linalg.generic` to access transform dialect tiling primitives. The transform dialect's
   `multitile_sizes` operation computes sub-block dimensions that respect DST register
   capacity constraints, enabling automatic register-constrained tiling without custom
   heuristics.

See [Section 5.5](#55-control-flow-scf-vs-affine-dialect) for affine vs SCF rationale and
[Section 5.6 in Compute Operations](03_TTL_Compute_Operations.md#56-structured-ops-for-dst-register-capacity-based-tiling)
for DST capacity-based tiling via linalg and transform dialect.

### 5.2 Key Pass Descriptions

**`TTLInferPipeSemaphores`**
- **Input**: `ttl.kernel` with pipe operations in datamovement threads
- **Analysis**: Analyze pipe nets to determine synchronization requirements
- **Transform**: Create required semaphores per pipe, insert semaphore
  operations in if_src/if_dst bodies
- **Output**: `ttl.kernel` with explicit semaphore operations for pipe
  synchronization
- **See**:
  [05_TTL_Multicast_Implementation.md](05_TTL_Multicast_Implementation.md) for
  detailed semaphore inference strategy

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
- **Input**: `ttl.kernel` with embedded thread regions and kernel-scope CB declarations
- **Transform**:
  1. Create `ttl.kernel_driver` that owns global state (CBs, tensor accessors)
  2. Assign globally unique CB indices in the driver
  3. Extract each thread region into separate `func.func` with CB handles as arguments
  4. Generate `ttl.launch_thread` calls to connect driver to thread functions
- **Output**: `ttl.kernel_driver` + separate `func.func` per thread
- **Key Invariant**: CB indices are assigned ONLY in the kernel driver, never in thread functions

**`TTLExpandThreads` Detailed Example:**

Input (after Python AST compilation):
```mlir
ttl.kernel @elementwise_add(%a: tensor<1024x1024xf32, #layout>,
                            %b: tensor<1024x1024xf32, #layout>,
                            %c: tensor<1024x1024xf32, #layout>) {
  // Kernel-scope resource declarations
  %a_cb = ttl.create_cb shape=[1,1], element_type=!ttcore.tile<32x32, f32>,
                        buffer_factor=2 : !ttl.cb<[1,1], !ttcore.tile<32x32, f32>, 2>
  %b_cb = ttl.create_cb shape=[1,1], element_type=!ttcore.tile<32x32, f32>,
                        buffer_factor=2 : !ttl.cb<[1,1], !ttcore.tile<32x32, f32>, 2>
  %c_cb = ttl.create_cb shape=[1,1], element_type=!ttcore.tile<32x32, f32>,
                        buffer_factor=2 : !ttl.cb<[1,1], !ttcore.tile<32x32, f32>, 2>

  %a_accessor = ttl.tensor_accessor %a : ...
  %b_accessor = ttl.tensor_accessor %b : ...
  %c_accessor = ttl.tensor_accessor %c : ...
  %shard_id = ttl.get_shard_id() : index

  ttl.datamovement_thread {
    %a_view = ttl.cb_reserve %a_cb : ... -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf_a = ttl.copy %a_accessor[%shard_id], %a_view : ... -> !ttl.transfer_handle<read>
    ttl.wait %xf_a
    ttl.cb_push %a_cb
    // ... similar for %b_cb ...
    ttl.thread_return
  }

  ttl.compute_thread {
    %a_tiles = ttl.cb_wait %a_cb : ... -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %b_tiles = ttl.cb_wait %b_cb : ... -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c_tiles = ttl.add %a_tiles, %b_tiles : ...
    ttl.cb_pop %a_cb
    ttl.cb_pop %b_cb
    %c_view = ttl.cb_reserve %c_cb : ...
    ttl.cb_push %c_cb
    ttl.thread_return
  }

  ttl.datamovement_thread {
    %c_view = ttl.cb_wait %c_cb : ...
    %xf_c = ttl.copy %c_view, %c_accessor[%shard_id] : ... -> !ttl.transfer_handle<write>
    ttl.wait %xf_c
    ttl.cb_pop %c_cb
    ttl.thread_return
  }

  ttl.kernel_return
}
```

Output (after `TTLExpandThreads`):
```mlir
// Kernel Driver: owns global state, assigns CB indices
ttl.kernel_driver @elementwise_add_driver(%a: tensor<...>, %b: tensor<...>, %c: tensor<...>) {
  // CB index assignment happens HERE - indices are globally unique
  %a_cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1,1], !ttcore.tile<32x32, f32>, 2>
  %b_cb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1,1], !ttcore.tile<32x32, f32>, 2>
  %c_cb = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1,1], !ttcore.tile<32x32, f32>, 2>

  %a_accessor = ttl.tensor_accessor %a : ...
  %b_accessor = ttl.tensor_accessor %b : ...
  %c_accessor = ttl.tensor_accessor %c : ...
  %shard_id = ttl.get_shard_id() : index

  // Launch threads with CB handles as arguments
  ttl.launch_thread @dm_reader(%a_cb, %b_cb, %a_accessor, %b_accessor, %shard_id)
  ttl.launch_thread @compute(%a_cb, %b_cb, %c_cb)
  ttl.launch_thread @dm_writer(%c_cb, %c_accessor, %shard_id)

  ttl.kernel_driver_return
}

// Data Movement Thread: Reader
func.func @dm_reader(%a_cb: !ttl.cb<...>, %b_cb: !ttl.cb<...>,
                     %a_accessor: !ttl.tensor_accessor<...>,
                     %b_accessor: !ttl.tensor_accessor<...>,
                     %shard_id: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // CB handles are arguments - NO ttl.bind_cb here!
  %a_view = ttl.cb_reserve %a_cb : ...
  %xf_a = ttl.copy %a_accessor[%shard_id], %a_view : ...
  ttl.wait %xf_a
  ttl.cb_push %a_cb
  // ... similar for %b_cb ...
  func.return
}

// Compute Thread
func.func @compute(%a_cb: !ttl.cb<...>, %b_cb: !ttl.cb<...>, %c_cb: !ttl.cb<...>)
    attributes {ttl.kernel_thread = #ttkernel.thread<tensix>} {
  // CB handles are arguments - thread doesn't allocate indices
  %a_tiles = ttl.cb_wait %a_cb : ...
  %b_tiles = ttl.cb_wait %b_cb : ...
  %c_tiles = ttl.add %a_tiles, %b_tiles : ...
  ttl.cb_pop %a_cb
  ttl.cb_pop %b_cb
  %c_view = ttl.cb_reserve %c_cb : ...
  ttl.cb_push %c_cb
  func.return
}

// Data Movement Thread: Writer
func.func @dm_writer(%c_cb: !ttl.cb<...>, %c_accessor: !ttl.tensor_accessor<...>,
                     %shard_id: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c_view = ttl.cb_wait %c_cb : ...
  %xf_c = ttl.copy %c_view, %c_accessor[%shard_id] : ...
  ttl.wait %xf_c
  ttl.cb_pop %c_cb
  func.return
}
```

**Design Rationale:**
1. **Kernel Driver as Single Source of Truth**: The `ttl.kernel_driver` is the only place where CB indices are assigned. This ensures indices are globally unique across all threads sharing L1 memory.
2. **CBs as Function Arguments**: Thread functions receive CB handles as arguments, eliminating the possibility of index collisions from per-operation CB creation.
3. **Thread Attributes**: Each extracted function has `ttl.kernel_thread` attribute indicating thread type (`#ttkernel.thread<noc>` for data movement, `#ttkernel.thread<tensix>` for compute).
4. **Explicit Launch Semantics**: `ttl.launch_thread` establishes the driver→thread relationship with explicit argument passing, making data dependencies visible in IR.

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
- **Input**: `ttl.add`, `ttl.mul`, `ttl.matmul` and other
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
  - Coordinate translation: logical → virtual/physical (see section 5.2.1)
- **Output**: TTKernel NOC operations with device-specific coordinates

### 5.2.1 Coordinate Space Translation

TTL operates on logical grid coordinates, but NOC hardware operations require
device-specific coordinate addressing. The `TTLLowerDataMovement` pass handles
coordinate translation based on device architecture.

Coordinate spaces:

Logical Grid Coordinates:
- Used by TTL operations: `ttl.core()`, `ttl.Pipe(src=...)`
- User-facing abstraction: contiguous 0-indexed coordinates
- Example: 2x2 grid has cores (0,0), (0,1), (1,0), (1,1)
- Architecture-independent representation

Device-Specific Coordinates:
- Required by TTKernel NOC operations: `ttkernel.get_noc_addr(...)`,
  `ttkernel.get_noc_multicast_addr(...)`
- May be virtual coordinates (Blackhole) or physical coordinates
  (Wormhole/Grayskull)
- Obtained via device coordinate translation APIs
- Architecture-dependent implementation

Translation approach:

The lowering pass uses device APIs to obtain appropriate coordinates for NOC
operations. On newer architectures (Blackhole), coordinate virtualization is
handled by the device layer. On older architectures (Wormhole/Grayskull),
physical coordinates are computed directly. The TTL dialect abstracts these
differences.

Lowering strategy:

For multicast operations, the pass generates coordinate computation at kernel
runtime:
1. Use `ttkernel.my_x` / `ttkernel.my_y` to get current core's virtual
   coordinates (operations take NOC index as argument)
2. Compute multicast range bounds using affine arithmetic on these coordinates
3. Pass computed coordinates to `ttkernel.get_noc_multicast_addr`
4. Device coordinate APIs handle architecture-specific translation

Example lowering:
```
TTL (logical): ttl.Pipe(src=(0,0), dst_range=(slice(1,4), 0))

TTKernel (device-specific):
  %noc = arith.constant 0 : i32
  %my_x = ttkernel.my_x %noc
  %my_y = ttkernel.my_y %noc
  %mcast_start_x = affine.apply affine_map<(x) -> (x + 1)>(%my_x)
  %mcast_end_x = affine.apply affine_map<(x) -> (x + 3)>(%my_x)
  %mcast_addr = ttkernel.get_noc_multicast_addr(
    %mcast_start_x, %my_y, %mcast_end_x, %my_y, %l1_addr)
```

The device coordinate translation layer ensures portability across hardware
generations without requiring TTL dialect changes.

See [05_TTL_Multicast_Implementation.md](05_TTL_Multicast_Implementation.md) for
complete multicast coordinate handling examples.

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
    "ttcore::GridAttr":$grid,  // Reuse ttcore::GridAttr from tt-mlir
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
   - Emitted by type verifiers and type-checking operations
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


## 6. Type Conversion & Lowering Examples

### 6.1 TTL → TTKernel Type Mapping

| TTL Type | TTKernel Type | Conversion Notes |
|----------|---------------|------------------|
| `!ttl.circular_buffer<[2,1], !ttcore.tile<32x32,f32>, 2, L1>` | `!ttkernel.cb<4, !ttcore.tile<32x32,f32>>` | Total elements = 2×1×2 = 4 tiles (layout inferred: element_type is tile → tiled layout); For row-major CBs, element_type is scalar (f32, bf16, etc.); memory space drives L1 address assignment; buffer_factor used to compute total but not preserved in TTKernel CB type |
| `tensor<2x1x!ttcore.tile>` | `memref<...>` | Tensors decomposed into tile operations during lowering; tensor layout information drives loop generation over tiles |
| `!ttl.transfer_handle` | N/A (elided) | Lowers to global barriers |
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
| `ttl.add` | `ttkernel.add_tiles_init`, `ttkernel.add_tiles` | Medium | **HIGH** |
| `ttl.sub` | `ttkernel.sub_tiles_init`, `ttkernel.sub_tiles` | Medium | **HIGH** |
| `ttl.mul` | `ttkernel.mul_tiles_init`, `ttkernel.mul_tiles` | Medium | **HIGH** |
| `ttl.matmul` | `ttkernel.mm_init`, `ttkernel.matmul_tiles` | Medium | **HIGH** |
| **Compute (SFPU - Phase 2)** ||||
| `ttl.exp` | `ttkernel.init_sfpu`, `ttkernel.exp_tile` | Medium | MEDIUM |
| `ttl.log` | `ttkernel.init_sfpu`, `ttkernel.log_tile` | Medium | MEDIUM |
| `ttl.sqrt` | `ttkernel.init_sfpu`, `ttkernel.sqrt_tile` | Medium | MEDIUM |
| `ttl.relu` | `ttkernel.init_sfpu`, `ttkernel.relu_tile` | Medium | MEDIUM |
| `ttl.gelu` | `ttkernel.init_sfpu`, `ttkernel.gelu_tile` | Medium | MEDIUM |
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
| `ttl.semaphore_wait_eq` | `ttkernel.noc_semaphore_wait` | Medium | MEDIUM |
| `ttl.semaphore_wait_min` | `ttkernel.noc_semaphore_wait_min` | Medium | MEDIUM |
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
%blk = ttl.cb_wait %cb : tensor<...>

// TTKernel (lowered - count derived from CB type)
ttkernel.cb_wait_front %cb, 2  // 2 computed from CB.shape
```

**Tile Extraction:**
```mlir
// TTL (lowered form - internal IR)
%tile0 = ttl.get_tile %cb, %c0
%tile1 = ttl.get_tile %cb, %c1

// TTKernel (copy from CB to DST register)
ttkernel.copy_tile_init %cb
%tile0 = ttkernel.copy_tile %cb, 0, %dst_idx0
%tile1 = ttkernel.copy_tile %cb, 1, %dst_idx1
```

**Block Compute:**
```mlir
// TTL (user-facing tensor-level abstraction)
%a_blk = ttl.cb_wait %a_cb : tensor<...>
%b_blk = ttl.cb_wait %b_cb : tensor<...>
%result = ttl.add %a_blk, %b_blk

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
%pipe = ttl.create_pipe src=[0,0] dst_range=[[0,0],[1,3]]
%xf = ttl.copy %block, %pipe
ttl.wait %xf

// TTKernel
%noc_addrs = // Compute multicast address range
ttkernel.noc_async_write_multicast %src_addr, %noc_addrs, %size
ttkernel.noc_async_write_barrier
```
