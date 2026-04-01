# Tour of TT-Lang

TT-Lang is a Python-based domain-specific language for authoring high-performance custom kernels on Tenstorrent hardware.

## Overview

TT-Lang provides an expressive middle ground between TT-NN's high-level operations and TT-Metalium's low-level hardware control. The language centers on explicit data movement and compute threads with synchronization primitives familiar to TT-Metalium users (dataflow buffers, semaphores) alongside new abstractions (tensor slices, blocks, pipes).

## Key Concepts

- **Kernel function**: Python function decorated with `@ttl.operation()` that defines thread functions.
- **Thread functions**: Decorated with `@ttl.compute()` or `@ttl.datamovement()`, these define compute and data movement logic.
- **Dataflow buffers**: Communication primitives for passing data between threads within a node.
- **Blocks**: Memory acquired from dataflow buffers, used in compute expressions or copy operations.
- **Grid**: Defines the space of nodes for kernel execution.

## Tour

```{toctree}
:maxdepth: 2

kernel-basics
dataflow-buffers
```

## Reference

For the complete language specification, see [TT-Lang Specification](../specs/TTLangSpecification.md).
