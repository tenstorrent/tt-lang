# TTL Dialect Design Plan

**Version**: 0.6

**Modified**: 2025-12-06

**Status**: Design Phase - Under Revision

This document provides a high-level overview of the TTL (TT-Lang) MLIR dialect, a tensor-level intermediate representation designed to directly capture the semantics of the TT-lang DSL. The TTL dialect enables multi-stage compilation with explicit transformation passes for synchronization inference, resource allocation, and hardware-specific optimizations before lowering to executable kernels.

For detailed specifications, see the linked documents below.

## Table of Contents

### This Document
1. [Motivation & Goals](#1-motivation--goals)
2. [Architecture Overview](#2-architecture-overview)
3. [Python Language Constructs to TTL Dialect Mapping](#3-python-language-constructs-to-ttl-dialect-mapping)

### Detailed Specifications
3. **[Type System](02_TTL_Type_System.md)** - Core types, attributes, and TensorAccessor support
4. **[Compute Operations](03_TTL_Compute_Operations.md)** - Compute operations, circular buffers, fusion, and DST register management
5. **[Data Movement Operations](04_TTL_Data_Movement_Operations.md)** - Data movement, pipes, synchronization, and MLIR interfaces
6. **[Multicast Implementation](05_TTL_Multicast_Implementation.md)** - Multicast patterns, semaphore inference, and coordinate translation
7. **[Compilation Pipeline](06_TTL_Compilation_Pipeline.md)** - Pass architecture, lowering strategies, and type conversions
8. **[Implementation and Runtime](07_TTL_Implementation_and_Runtime.md)** - Python integration, TTNN runtime, roadmap, and design rationale

## Executive Summary

The TTL dialect provides a tensor-level intermediate representation for TT-lang programs (defined in [TT-lang.md](../TT-lang.md)), enabling multi-stage lowering with explicit compiler transformations before generating C++ kernels. Generated kernels execute on TTNN runtime via the `ttnn.generic_op` API, which accepts C++ kernel source code and metadata through Python descriptors. TTL is designed specifically for the TT-lang DSL surface language, while D2M serves as a lower-level dialect for data movement and compute operations.

**Key Design Decisions:**
- **Threading Model**: Simple compute/data movement threads (MVP)
- **Abstraction Level**: Tensor-level (blocks of tiles), not individual tiles
- **Type System**: Memory spaces explicit in types; SSA values for all resources (CBs, pipes, semaphores)
- **Control Flow**: Hybrid SCF/Affine dialect with TTL attributes
- **Lowering Path**: TTL → TTKernel → EmitC → C++ kernel source (compiled separately)
- **Phasing**: Multi-threaded from start (matches current examples)
- **Compute Lowering**: Block operations → linalg.generic → transform dialect tiling for DST capacity → TTKernel

## 1. Motivation & Goals

### Motivation for TTL Dialect

D2M dialect serves as a general-purpose data movement and compute abstraction. For the TT-lang DSL specifically, a dedicated dialect provides:

- DSL-level IR: Preserve TT-lang abstractions (kernels, threads, circular buffers, pipes) longer in compilation
- SSA semantics: `CB` operations with explicit SSA values rather than implicit state effects (D2M `CB` ops use `MemoryEffects<[MemRead, MemWrite]>` for state transitions)
- Analysis opportunities: Synchronization inference, resource allocation, and scheduling at DSL semantic level
- Flexibility: Experiment with TT-lang-specific optimizations and compilation strategies
- Multiple targets: TTKernel (immediate) and potential standalone C++ backend

### TTL Dialect Goals

1. Capture DSL semantics in SSA form: kernels, threads, circular buffers, pipes, blocks, semaphores
2. Enable analysis passes: Synchronization inference, memory planning, DST register allocation
3. Support transformations: Liveness analysis, operation reordering, pipelining
4. C++ kernel generation: Generate C++ kernel source strings with metadata that are passed to `ttnn.generic_op` for runtime compilation and execution. This is the MVP target.
5. Future-proof: Extensible to new hardware generations via attributes

### Non-Goals (MVP)

- Autotuning algorithms (IR has hooks, algorithms come later)
- Single-threaded synchronous model (start multi-threaded)
- Complete TT-lang spec (start minimal, expand incrementally)
- Direct C++ backend (TTL→C++ without TTKernel): Post-MVP, see Implementation document section 10.2
- Custom ConvertTTLToEmitC pass: Not needed, use existing ConvertTTKernelToEmitC from tt-mlir

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
- Kernel descriptor generator produces C++ source strings and metadata from TTL IR
- Python runtime constructs `ttnn.KernelDescriptor`, `ttnn.CBDescriptor`, and `ttnn.ProgramDescriptor` objects
- `ttnn.generic_op` handles kernel compilation and device execution
- No separate build step required - kernels provided as source strings to runtime

**Relationship to D2M Dialect:**

TTL and D2M serve different roles in the compilation pipeline:

- TTL: New frontend dialect specifically designed for the TT-lang DSL. TTL provides DSL-specific IR enabling TT-lang-aware transformations. TTL bypasses D2M and generates C++ kernel descriptors directly for `ttnn.generic_op`.

- D2M: Remains the primary dialect for framework paths (JAX/PyTorch → TTIR → D2M → TTKernel). D2M serves as a general-purpose data movement and compute abstraction.

- Convergence: Both TTL and D2M can target TTNN runtime, but through different mechanisms. TTL uses `ttnn.generic_op` with kernel descriptors, while D2M uses the traditional flatbuffer workflow.

This separation allows TTL to focus on TT-lang DSL semantics while D2M continues to serve framework integration needs.

## Next Steps

Continue reading the detailed specifications:

1. **[Type System](02_TTL_Type_System.md)** - Understand the type system, attributes, and TensorAccessor support
2. **[Compute Operations](03_TTL_Compute_Operations.md)** - Learn about compute operations and DST register management
3. **[Data Movement Operations](04_TTL_Data_Movement_Operations.md)** - Understand data movement and synchronization
4. **[Multicast Implementation](05_TTL_Multicast_Implementation.md)** - Multicast patterns and semaphore inference
5. **[Compilation Pipeline](06_TTL_Compilation_Pipeline.md)** - See how the compiler transforms TTL IR
6. **[Implementation and Runtime](07_TTL_Implementation_and_Runtime.md)** - Integration details and roadmap

## 3. Python Language Constructs to TTL Dialect Mapping

The following table maps Python language constructs and `ttl` module APIs (as defined in [TT-Lang.md](../TT-Lang.md)) to corresponding TTL dialect operations:

| Python Language Construct | TTL Dialect Operation | Notes |
|---|---|---|
| **Program & Kernel Context** | | |
| `@ttl.kernel()` decorator | `ttl.kernel` (metadata) | Marks Python function as TTL kernel |
| `@ttl.compute()` decorator | `ttl.compute_thread` | Creates compute thread region |
| `@ttl.datamovement()` decorator | `ttl.datamovement_thread` | Creates data movement thread region |
| `ttl.Program(compute, dm0, dm1, ...)` | `ttl.module` | Constructs program from thread functions |
| **Grid and Core Queries** | | |
| `ttl.grid_size(dims)` | `ttl.grid_size` | Returns grid size in specified dimensions |
| `ttl.core(dims)` | `ttl.core_id` | Returns logical core coordinates |
| **Circular Buffer Operations** | | |
| `ttl.make_circular_buffer_like(tensor, shape, factor)` | `ttl.circular_buffer` | Creates circular buffer with type/shape info |
| `cb.wait()` | `ttl.cb_wait` | Consumer acquire; blocks until data available |
| `cb.reserve()` | `ttl.cb_reserve` | Producer acquire; blocks until space available |
| `cb.pop()` | `ttl.cb_pop` | Consumer release; signals block is free |
| `cb.push()` | `ttl.cb_push` | Producer release; signals block is filled |
| **Arithmetic Operators** | | |
| `a + b` (tensors) | `ttl.add` | Element-wise addition |
| `a - b` (tensors) | `ttl.sub` | Element-wise subtraction |
| `a * b` (tensors) | `ttl.mul` | Element-wise multiplication |
| `a / b` (tensors) | `ttl.div` | Element-wise division |
| `a ** b` (tensors) | `ttl.pow` | Element-wise power |
| `a @ b` (tensors) | `ttl.matmul` | Matrix multiplication |
| **Math Functions** | | |
| `ttl.math.sqrt(x)` | `ttl.sqrt` | Element-wise square root |
| `ttl.math.exp(x)` | `ttl.exp` | Element-wise exponential |
| `ttl.math.log(x)` | `ttl.log` | Element-wise natural logarithm |
| `ttl.math.rsqrt(x)` | `ttl.rsqrt` | Element-wise reciprocal square root |
| `ttl.math.relu(x)` | `ttl.relu` | Rectified linear unit activation |
| `ttl.math.gelu(x)` | `ttl.gelu` | GELU activation function |
| `ttl.math.sigmoid(x)` | `ttl.sigmoid` | Sigmoid activation function |
| `ttl.math.tanh(x)` | `ttl.tanh` | Hyperbolic tangent activation |
| `ttl.math.sin(x)` | `ttl.sin` | Element-wise sine |
| `ttl.math.cos(x)` | `ttl.cos` | Element-wise cosine |
| `ttl.math.tan(x)` | `ttl.tan` | Element-wise tangent |
| **Reduction Operations** | | |
| `ttl.reduce_sum(x, dims)` | `ttl.reduce_sum` | Sum reduction over specified dimensions |
| `ttl.reduce_max(x, dims)` | `ttl.reduce_max` | Max reduction over specified dimensions |
| **Broadcast Operations** | | |
| `ttl.bcast(x, shape)` | `ttl.bcast` | Broadcast tensor to new shape |
| **Data Movement** | | |
| `ttl.copy(src, dst, **kwargs)` | `ttl.copy` | Transfer data between memory/pipes with optional transform |
| `xf.wait()` | `ttl.wait` | Synchronize on transfer handle (implicit via scoping) |
| **Pipe Operations** | | |
| `ttl.Pipe(src, dst)` | (metadata) | Pipe definition; source and destination specification |
| `ttl.CoreRange` (tuple with slices) | (metadata) | Core range for multicast destination |
| `ttl.PipeNet([pipe1, pipe2, ...])` | (metadata) | Groups pipes into network for collective operations |
| `pnet.if_src(callback)` | `ttl.if_pipe_src` | Conditional execution for pipe source cores |
| `pnet.if_dst(callback)` | `ttl.if_pipe_dst` | Conditional execution for pipe destination cores |
| `pipe_identity.src` | (implicit) | Property access within `if_dst` condition |
| `pipe_identity.dst` | (implicit) | Property access within `if_src` condition |

**Mapping Notes:**

- All tensor operations (arithmetic, math, reduction, broadcast) implicitly operate at the tensor abstraction level and are decomposed into tile-based operations during lowering to TTKernel.
- Circular buffer operations manage acquisition and release of data blocks from communication primitives.
- Data movement and pipe operations provide inter-core communication and synchronized data transfers.
- The TTL dialect preserves the high-level structure and intent of Python programs while enabling aggressive optimization during lowering.
- See section 7.2 in [Implementation and Runtime](07_TTL_Implementation_and_Runtime.md#72-operator-mapping) for detailed Python integration code examples.

## References

See the [Implementation and Runtime](07_TTL_Implementation_and_Runtime.md#13-references) document for complete references.
