# TTL Dialect Multicast Implementation

**Part of**: [TTL Dialect Design Plan](01_TTL_Dialect_Plan.md)

This document analyzes multicast patterns in TT-Metal C++ kernels, verifies TTL
dialect coverage, and specifies multicast-specific implementation requirements.

## Table of Contents

- [Analysis of Multicast Patterns](#analysis-of-multicast-patterns)
- [TTL Dialect Coverage](#ttl-dialect-coverage)
- [TTKernel Operation Coverage](#ttkernel-operation-coverage)
- [Multicast Implementation Design](#multicast-implementation-design)
- [Implementation Plan](#implementation-plan)

## Related Documents

- [Back to Overview](01_TTL_Dialect_Plan.md)
- [Type System](02_TTL_Type_System.md) - Pipe and semaphore types
- [Compute Operations](03_TTL_Compute_Operations.md) - Core coordinate
  operations
- [Data Movement Operations](04_TTL_Data_Movement_Operations.md) - Pipe and
  semaphore operations
- [Compilation Pipeline](06_TTL_Compilation_Pipeline.md) - Lowering multicast to
  TTKernel
- [Implementation and Runtime](07_TTL_Implementation_and_Runtime.md) - Pass
  specifications



## Analysis of Multicast Patterns

This section analyzes multicast synchronization patterns from TT-Metal examples
to inform TTL dialect design. Our MVP implementation will focus on the first two
patterns, with additional patterns deferred to future phases.

### Simple Multicast Pattern

Source:
`tt-metal/tt_metal/programming_examples/simple_mcast/simple_multicast.cpp`

The canonical symmetric barrier pattern where all cores participate:

```cpp
// COORDINATOR (Core 0):
noc_semaphore_set(start_sem_ptr, 0);  // Initialize start semaphore
noc_semaphore_set(done_sem_ptr, 0);   // Initialize done semaphore
noc_semaphore_inc(done_sem_noc_addr, 1);  // Self-signal before multicast
noc_async_atomic_barrier();

noc_semaphore_set(start_sem_ptr, 1);
uint64_t start_sem_mcast_addr = get_noc_multicast_addr(...);
noc_semaphore_set_multicast_loopback_src(start_sem_addr, start_sem_mcast_addr, num_cores - 1);
noc_async_atomic_barrier();

noc_semaphore_wait(done_sem_ptr, num_cores);

// WORKER (Core 1-N):
noc_semaphore_wait(start_sem_ptr, 1);
noc_semaphore_inc(done_sem_noc_addr, 1);
noc_async_atomic_barrier();
```

Key characteristics:
- Two semaphores: start (coordinator → workers) and done (workers → coordinator)
- Loopback multicast for start signal (coordinator included in destinations)
- Atomic barriers after remote semaphore operations
- Device-specific coordinates obtained from `my_x[0]`, `my_y[0]`

### Gather-Multicast Pattern

Source:
`tt-metal/tt_metal/programming_examples/contributed/multicast/kernels/dataflow/coordinator_kernel.cpp`
and `inbound_kernel.cpp`

The coordinator-workers pattern with data transfer:

```cpp
// COORDINATOR:
noc_semaphore_wait(sender_addr_ptr, num_dests);  // Wait for workers ready
noc_semaphore_set(sender_addr_ptr, 0);           // Reset after wait
noc_async_read(src0_dram_noc_addr, ...);         // Gather data from DRAM
noc_async_read_barrier();

uint64_t tile_mcast_addr = get_noc_multicast_addr(...);
noc_async_write_multicast(tile_l1_addr, tile_mcast_addr, ...);

*(receiver_addr_ptr) = VALID;
uint64_t validity_mcast_addr = get_noc_multicast_addr(...);
noc_semaphore_set_multicast(receiver_addr, validity_mcast_addr, num_dests);
noc_async_write_barrier();  // Essential for ordering

// WORKERS:
cb_reserve_back(cb_id_in0, 1);
noc_semaphore_set(receiver_addr_ptr, INVALID);  // Reset to INVALID before signaling ready
uint64_t remote_sender_noc_addr = get_noc_addr(start_x, start_y, sender_addr);
noc_semaphore_inc(remote_sender_noc_addr, 1);  // Signal readiness

noc_semaphore_wait(receiver_addr_ptr, VALID);   // Wait for VALID flag
cb_push_back(cb_id_in0, 1);
```

Key characteristics:
- Two-phase protocol (typical for communication primitives): readiness
  signaling, then data + validity multicast
- INVALID/VALID protocol for validity semaphore
- Semaphore reset after wait to prevent accumulation
- Write barrier after multicast for ordering guarantees
- Separate semaphore multicast for validity (no loopback, coordinator not a
  receiver)

### Additional Multicast Patterns

Source: Various production kernels in `tt-metal/ttnn/cpp/ttnn/operations/`

Multi-Phase Multicast (GroupNorm):
- Data multicast to multiple disjoint groups with different semaphores
- Each group has independent sender/receiver semaphores
- Linked transactions chain multiple multicast operations efficiently
- Source:
  `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp`

Conditional Multicast:
- Skip multicast when `num_cores == 1` to avoid hardware hangs
- Use regular write instead of multicast for single-core case
- Source:
  `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`

Best practices observed across patterns:
- Always initialize semaphores before use
- Reset semaphores after wait to prevent value accumulation
- Include write barriers after multicast for completion guarantees
- Use physical/virtual coordinates obtained from device APIs
- Handle single-core edge cases explicitly

## TTL Dialect Coverage

The pipe abstraction captures multicast topology:

```python
# TT-Lang DSL:
net = ttl.PipeNet([ttl.Pipe(
    src=(0, 0),
    dst_range=(slice(1, 4), 0)  # Cores (1,0), (2,0), (3,0)
)])

def pipe_src(pipe):
    xf = ttl.copy(blk, pipe)
    xf.wait()

def pipe_dst(pipe):
    xf = ttl.copy(pipe, blk)
    xf.wait()

net.if_src(pipe_src)
net.if_dst(pipe_dst)
```

Captured information:
- PipeNet storing pipe list in create_pipenet operation operands
- Per pipe:
  - Source core coordinates
  - Destination core range with slice semantics
  - Multicast vs unicast distinction
  - Send/receive pairing via if_src/if_dst

Semaphore operations:

```python
my_sem = ttl.Semaphore(initial_value=0)
remote_sem = my_sem.get_remote((0, 0))
mcast_sem = my_sem.get_remote_multicast()

my_sem.wait_eq(num_cores)  # Wait for equality
my_sem.wait_min(threshold) # Wait for value >= threshold
my_sem.set(0)              # Set local semaphore value
remote_sem.inc(1)          # Increment remote semaphore (unicast only)
mcast_sem.set(1)           # Set remote semaphore (unicast or multicast)
```

Captured information:
- Local vs remote semaphore distinction (via get_remote/get_remote_multicast)
- Unicast vs multicast remote operations
- Wait conditions: `wait_eq(n)` for equality, `wait_min(n)` for >= threshold
- Increment (`inc`) and set (`set`) operations
- Note: Multicast semaphores support `set` but not `inc` (hardware limitation)


## TTKernel Operation Coverage

All required operations exist in TTKernel dialect (verified in
`TTKernelOps.td`):

| C++ API | TTKernel Operation | Line |
|---------|-------------------|------|
| `get_semaphore(idx)` | `ttkernel.get_semaphore` | 2331 |
| `noc_semaphore_set(ptr, val)` | `ttkernel.noc_semaphore_set` | 2361 |
| `noc_semaphore_wait(ptr, val)` | `ttkernel.noc_semaphore_wait` | 2377 |
| `noc_semaphore_wait_min(ptr, val)` | `ttkernel.noc_semaphore_wait_min` | 2393 |
| `noc_semaphore_inc(addr, val)` | `ttkernel.noc_semaphore_inc` | 2345 |
| `noc_semaphore_set_multicast(...)` | `ttkernel.noc_semaphore_set_multicast` | 2409 |
| `noc_semaphore_set_multicast_loopback_src(...)` | `ttkernel.noc_semaphore_set_multicast_loopback_src` | 2438 |
| `get_noc_multicast_addr(...)` | `ttkernel.get_noc_multicast_addr` | 2611 |
| `noc_async_write_multicast(...)` | `ttkernel.noc_async_write_multicast` | 2653 |
| `noc_async_write_multicast_loopback_src(...)` | `ttkernel.noc_async_write_multicast_loopback_src` | 2686 |
| `my_x[noc]` / `my_y[noc]` | `ttkernel.my_x` / `ttkernel.my_y` | Virtual coords (take NOC index arg) |
| `get_noc_addr(x, y, addr)` | `ttkernel.get_noc_addr` | Unicast |

Conclusion: TTKernel has complete coverage for generating multicast C++ kernels.


## Multicast Implementation Design

### PipeNet as First-Class Abstraction

The TTL dialect represents pipe networks as first-class SSA values through the
[`TTL_PipeNet` type](02_TTL_Type_System.md#pipenet-type) and
[`ttl.create_pipenet` operation](04_TTL_Data_Movement_Operations.md#42-resource-creation).
This design preserves the TT-Lang spec's `PipeNet` abstraction and provides the
following capabilities.

Validation: The PipeNet SSA value groups all pipes, enabling whole-network validation.
The compiler traces the PipeNet value to its defining create_pipenet operation, extracts
the pipe operands, and verifies that every pipe has matching source and destination guards,
each multicast has at least one receiver, and no pipes are only used on one side. Without
PipeNet, the IR only sees unrelated pipe values at each guard, making these checks impossible.

Synchronization inference: The `TTLInferPipeSemaphores` pass operates on PipeNet SSA values
to construct semaphores. The pass extracts the pipe list from create_pipenet operands, having
defining operation allows the pass to count destinations, determine loopback
requirements, and handle multi-pipe coordination (e.g., ready/valid cascades)
without re-deriving topology at each usage site.

Lowering optimization: The PipeNet's defining `ttl.create_pipenet` operation can
cache precomputed metadata (destination core lists, loopback flags, multicast
descriptors) that multiple `ttl.if_pipe_src` and `ttl.if_pipe_dst` regions
reuse. This avoids recomputing slice expansions and loopback detection for each
guard.

Lowering strategy:
- Python `ttl.PipeNet([...])` → `ttl.create_pipenet` operation
- `net.if_src(callback)` → `ttl.if_pipe_src %net { ... }`
- `net.if_dst(callback)` → `ttl.if_pipe_dst %net { ... }`
- During lowering to TTKernel, the pass queries the PipeNet's defining operation
  for topology information

### Automatic Semaphore Inference

The `TTLInferPipeSemaphores` pass automatically creates semaphores (see
[`TTL_Semaphore` type](02_TTL_Type_System.md#semaphore-type) and
[`ttl.create_semaphore` operation](04_TTL_Data_Movement_Operations.md#semaphore-operations))
for pipe synchronization. This design is mandatory to ensure correctness by
preventing users from forgetting critical synchronization operations.

For each pipe, two semaphores are created:
1. Ready semaphore: Receivers signal readiness to sender
2. Validity semaphore: Sender signals data availability to receivers

Multicast pipe pattern (1 sender → N receivers):
- Sender waits for ready_sem == N, then resets to 0
- Sender performs DMA multicast
- Sender sets validity_sem via multicast
- Each receiver increments ready_sem (unicast to sender)
- Each receiver waits for validity_sem == 1

This matches the current D2M approach and simplifies the user model by allowing
users to focus on data flow rather than low-level synchronization details.
Future extensions may add opt-out mechanisms via attributes if hand-optimized
patterns require manual semaphore management.


### Coordinate Space Handling

TTL operates on logical grid coordinates while NOC hardware requires
device-specific coordinates. The `TTLLowerDataMovement` pass handles this
translation.

Coordinate spaces:

Logical coordinates (TTL):
- `ttl.core(dims=2)` returns logical grid coordinates (e.g., (0,0), (1,0), ...)
- `ttl.Pipe(src=(0,0), dst_range=(slice(1,4), 0))` uses logical
  coordinates
- Architecture-independent representation

Device-specific coordinates (TTKernel):
- `ttkernel.my_x` / `ttkernel.my_y` return virtual coordinates for NOC
  operations (take NOC index as argument)
- Required by `get_noc_multicast_addr` and `get_noc_addr` functions
- Return virtual coordinates (may differ from physical on some architectures)
- Architecture-specific translation handled by device coordinate translation layer

Translation approach in `TTLLowerDataMovement`:
- Generate runtime coordinate computation using `ttkernel.my_x` /
  `ttkernel.my_y`
- Compute multicast ranges using affine arithmetic on device coordinates
- Device layer handles architecture-specific virtualization
- TTL dialect remains architecture-agnostic

Example from C++ showing device coordinate usage:
```cpp
const uint32_t my_x_coord = my_x[0];  // Device-specific coordinate
const uint32_t my_y_coord = my_y[0];
const uint32_t mcast_start_x = my_x_coord;
const uint32_t mcast_end_x = my_x_coord + (num_cores - 1);
```

### Loopback Multicast Inference

The compiler automatically detects loopback multicast patterns during lowering
and selects the appropriate TTKernel operations. Loopback occurs when the source
core is included in the destination core range.

Hardware operation selection:
- Source NOT in destination range → `ttkernel.noc_async_write_multicast`,
  `ttkernel.noc_semaphore_set_multicast`
- Source IN destination range →
  `ttkernel.noc_async_write_multicast_loopback_src`,
  `ttkernel.noc_semaphore_set_multicast_loopback_src`

The `TTLLowerDataMovement` pass performs this detection by checking if src
falls within dst_range bounds. Users specify the destination range
naturally, and the compiler infers the correct hardware operation.

Examples:

Multicast column (sender excluded):
```python
ttl.Pipe(src=(x, 0), dst_range=(slice(x, x+1), slice(1, grid_y)))
```
Sender at (x,0) not in range [1, grid_y) → non-loopback operation

Symmetric barrier (sender included):
```python
ttl.Pipe(src=(0, 0), dst_range=(slice(0, 4), 0))
```
Sender at (0,0) in range [0, 4) → loopback operation automatically selected

### Atomic Barrier Insertion

The compiler inserts atomic barriers based on builtin LLVM memory effects
analysis. Individual semaphore operations (`ttl.semaphore_inc`,
`ttl.semaphore_set`) do not implicitly insert barriers.

Semaphore operations implement `MemoryEffectsOpInterface` with custom
`SemaphoreResource` to track semaphore memory effects. The
`TTLLowerDataMovement` pass uses effect collection to determine barrier
placement, similar to GPU barrier elimination in upstream MLIR
(`mlir/lib/Dialect/GPU/Transforms/EliminateBarriers.cpp`).

Barrier insertion strategy:
1. Collect memory effects between consecutive semaphore operations
2. Check for read-after-write, write-after-write conflicts on semaphore
   resources
3. Insert barrier only when conflicts exist

Conservative insertion points:
- Before semaphore wait operations that depend on prior remote sets/increments
- At end of pipe condition bodies (if_src/if_dst) to ensure visibility
- Before DMA operations that depend on semaphore state

Example from C++ showing barrier requirement:
```cpp
noc_semaphore_inc(done_sem_noc_addr, 1);
noc_async_atomic_barrier();  // Required for correctness
```

Implementation uses MLIR's `MemoryEffectsOpInterface` for effect tracking and
dependency detection.


## Implementation Plan

The implementation is organized into focused PRs that can be reviewed and merged
independently.

### PR 1: Semaphore Inference Pass - Unicast

Tasks:
- Implement `TTLInferPipeSemaphores` pass in `lib/Dialect/TTL/Transforms/`
- Analyze pipe nets to determine required semaphore count (2 per pipe: ready +
  validity)
- Insert `ttl.create_semaphore` operations
- Handle unicast pattern (1 sender → 1 receiver)
- Add pass registration and pipeline integration
- Add lit tests for unicast pipe pattern

Deliverable: Basic semaphore inference for unicast pipes.

Files created:
- `lib/Dialect/TTL/Transforms/TTLInferPipeSemaphores.cpp`
- `test/ttmlir/Dialect/TTL/infer_semaphores_unicast.mlir`

### PR 2: Semaphore Inference Pass - Multicast

Tasks:
- Extend `TTLInferPipeSemaphores` to handle multicast patterns
- Insert semaphore operations in if_src and if_dst region bodies
- Compute num_dests from dst_range slices
- Handle loopback case (sender signals self before multicast)
- Add lit tests for multicast and loopback patterns

Deliverable: Complete semaphore inference for all pipe patterns.

Files modified:
- `lib/Dialect/TTL/Transforms/TTLInferPipeSemaphores.cpp`

Files created:
- `test/ttmlir/Dialect/TTL/infer_semaphores_multicast.mlir`
- `test/ttmlir/Dialect/TTL/infer_semaphores_loopback.mlir`

### PR 3: Coordinate Translation, Loopback Detection, and Barrier Insertion

Tasks:
- Update `TTLLowerDataMovement` pass to handle device coordinate generation
- Insert `ttkernel.my_x` / `ttkernel.my_y` for current core coordinates
- Compute multicast range bounds using affine arithmetic
- Detect loopback by checking if src falls within dst_range
- Select loopback vs non-loopback TTKernel operations based on detection
- Implement `MemoryEffectsOpInterface` for semaphore operations with
  `SemaphoreResource`
- Collect memory effects between semaphore operations using MLIR effect
  collection APIs
- Insert `ttkernel.noc_async_write_barrier` based on conflict detection (hybrid
  conservative/aggressive strategy)
- Add lit tests for coordinate computation, loopback detection, and barrier
  insertion

Deliverable: Multicast operations lower to TTKernel with correct coordinate
handling, automatic loopback detection, and barrier insertion.

Files modified:
- `lib/Dialect/TTL/Transforms/TTLLowerDataMovement.cpp`
- `lib/Dialect/TTL/IR/TTLOps.cpp` (add MemoryEffectsOpInterface implementations)

Files created:
- `test/ttmlir/Dialect/TTL/lower_multicast.mlir`
- `test/ttmlir/Dialect/TTL/lower_loopback.mlir`
- `test/ttmlir/Dialect/TTL/barrier_insertion.mlir`

### PR 4: Python API and End-to-End Integration Tests

Tasks:
- Port `simple_multicast.cpp` pattern to TT-Lang DSL
- Port `coordinator_kernel.cpp` + `inbound_kernel.cpp` pattern to TT-Lang DSL
- Create lit tests that run full pipeline: Python → TTL → TTKernel → EmitC
- Add FileCheck patterns to verify generated C++ matches expected multicast code
- Add Python tests for multicast patterns

Deliverable: Working multicast examples demonstrating full compilation pipeline.

Files created:
- `test/python/examples/simple_multicast.py`
- `test/python/examples/gather_multicast.py`
- `test/python/test_multicast.py`
- `test/ttmlir/Dialect/TTL/Integration/multicast_e2e.mlir`

### PR Dependencies

```
PR 1 (Semaphore Unicast) ─→ PR 2 (Semaphore Multicast) ─→ PR 3 (Coordinate/Loopback/Barriers) ─→ PR 4 (Integration)
```
