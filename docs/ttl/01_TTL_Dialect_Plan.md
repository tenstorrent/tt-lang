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

### Detailed Specifications
3. **[Type System](02_TTL_Type_System.md)** - Core types, attributes, and TensorAccessor support
4. **[Compute Operations](03_TTL_Compute_Operations.md)** - Compute operations, circular buffers, fusion, and DST register management
5. **[Data Movement Operations](04_TTL_Data_Movement_Operations.md)** - Data movement, pipes, synchronization, and MLIR interfaces
6. **[Compilation Pipeline](05_TTL_Compilation_Pipeline.md)** - Pass architecture, lowering strategies, and type conversions
7. **[Implementation and Runtime](06_TTL_Implementation_and_Runtime.md)** - Python integration, TTNN runtime, roadmap, and design rationale

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
4. **[Compilation Pipeline](05_TTL_Compilation_Pipeline.md)** - See how the compiler transforms TTL IR
5. **[Implementation and Runtime](06_TTL_Implementation_and_Runtime.md)** - Integration details and roadmap

## References

See the [Implementation and Runtime](06_TTL_Implementation_and_Runtime.md#13-references) document for complete references.
