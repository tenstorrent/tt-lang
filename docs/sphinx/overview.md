# Introduction

TT-Lang is a Python-based domain-specific language for writing custom kernels on
Tenstorrent hardware. It provides an expressive middle ground between
[TT-NN](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html)'s
high-level operations and
[TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html)'s
low-level hardware control.

The primary use case is kernel fusion for model deployment. Engineers porting
models through TT-NN encounter operations that need to be fused for performance
or patterns that TT-NN cannot express, and today this typically requires
rewriting in TT-Metalium. TT-Lang makes this transition fast and correct: write
explicit data movement with high-level compute specification, validate
correctness through simulation, and integrate the result as a drop-in
replacement in a TT-NN graph.

The language is built around explicit data movement and compute threads with
synchronization primitives familiar to TT-Metalium users (dataflow buffers,
semaphores), alongside higher-level constructs (tensor slices, blocks, pipes)
that handle memory layout, compute APIs, and inter-core communication. Simple
kernels require minimal specification — the compiler infers compute API
operations, NOC addressing, and DST register allocation from high-level Python
syntax. Developers have control over all aspects of multi-node communication
through the pipe and pipe-net abstractions and completely control all aspects of
data movement and synchronization.

TT-Lang integrates tightly with TT-NN, enabling programs that mix high-level
TT-NN operations with low-level kernel code. The toolchain includes a
[functional simulator](simulator.md) for catching bugs before
running on hardware, line-by-line
[performance profiling](reference/performance-tools.md), and AI-assisted
development through [Claude Code](claude-skills.md) slash commands.

This project is under active development. See the
[functionality matrix](specs/TTLangSpecification.md#appendix-d-functionality-matrix)
in the language specification for current simulator and compiler support.

To get started, see [Getting Started](getting-started.md). For an
introduction, take a [tour](tour/index.md).
