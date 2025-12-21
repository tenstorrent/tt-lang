# Introduction

tt-lang is a Python DSL for authoring Tenstorrent kernels with compiler-managed memory, scheduling, and validation. It builds on tt-mlir for dialects, passes, and runtime support, giving a middle ground between TT-NN ease of use and TT-Metalium control.

Key goals:
- Express fused kernels quickly while the compiler infers compute, memory, and synchronization when possible.
- Keep low-level control available for performance tuning.
- Validate via simulation before hardware runs.
