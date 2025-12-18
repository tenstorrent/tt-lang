# Changelog

All notable changes to tt-lang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial prototype
- Python DSL with `@pykernel_gen` decorator for authoring custom kernels
- Integration with tt-mlir compiler infrastructure
- LLVM lit-based testing framework
- Pre-commit hooks for code formatting (Black for Python, clang-format for C++)
- CMake build system with flexible tt-mlir integration modes
- TTL to TTKernel pipeline now runs `ttl-lower-to-loops`, MLIR one-shot
  bufferization, and memref cleanup before TTKernel conversion, enabling
  memref→`!ttkernel.cb` lowering and circular buffer metadata preservation.
- Python bindings gain a bufferization regression (runs one-shot-bufferize with
  `ttl.attach_cb`) plus a full `ttl-to-ttkernel-pipeline` integration lit test.
- TTL copy/cb ops implement `BufferizableOpInterface`, so the entire TTL stage
  runs with `allow-unknown-bufferization-ops=false`, and new lit/python tests
  cover the tensor→memref rewrite.
- TTL CB protocol ops (`ttl.cb_reserve`, `ttl.cb_wait`) implement
  `BufferizableOpInterface`, so their tensor views become memrefs during Stage 4
  and are covered by new lit/python tests.

### Documentation
- Hitchhiker's Guide - Complete DSL documentation with examples
- Build System Guide - Detailed build configuration and integration scenarios
- Testing Guide - Instructions for writing and running tests
- Comprehensive README with quick start and examples
- Code of Conduct (Contributor Covenant 2.0)
- Apache 2.0 License with NOTICE file for third-party dependencies

### Examples
- Custom data movement with matrix multiplication
- Element-wise addition kernel
- TensorAccessor usage patterns
- Simple addition kernel demonstrations
