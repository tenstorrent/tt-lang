# Introduction

TT-Lang is a Python-based domain-specific language for writing custom kernels on Tenstorrent hardware. It integrates tightly with [TT-NN](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html), enabling programs that mix high-level TT-NN operations with low-level kernel code.

The language is built around explicit data movement and compute threads with explicit synchronization. This provides fine-grained control over kernel execution and performance characteristics. TT-Lang includes abstractions familiar to TT-Metalium users, such as circular buffers and semaphores, alongside higher-level constructs like tensor slices, blocks, and pipes that handle the details of memory layout, compute APIs, and inter-core communication.
