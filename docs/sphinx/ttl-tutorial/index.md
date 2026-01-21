# TT-Lang Tutorial

TT-Lang is a Python-based domain-specific language for authoring high-performance custom kernels on Tenstorrent hardware.

## Overview

TT-Lang provides an expressive middle ground between TT-NN's high-level operations and TT-Metalium's low-level hardware control. The language centers on explicit data movement and compute threads with synchronization primitives familiar to TT-Metalium users (circular buffers, semaphores) alongside new abstractions (tensor slices, blocks, pipes).

## Key Concepts

- **Kernel function**: Python function decorated with `@ttl.kernel()` that defines thread functions.
- **Thread functions**: Decorated with `@ttl.compute()` or `@ttl.datamovement()`, these define compute and data movement logic.
- **Circular buffers**: Communication primitives for passing data between threads within a core.
- **Blocks**: Memory acquired from circular buffers, used in compute expressions or copy operations.
- **Grid**: Defines the space of Tensix cores for kernel execution.

## Tutorial Sections

```{toctree}
:maxdepth: 2

kernel-basics
circular-buffers
```

## Reference

For the complete language specification, see [TT-Lang Specification](../specs/TTLangSpecification.md).
