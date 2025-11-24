# Changelog

All notable changes to tt-lang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release
- Python DSL with `@pykernel_gen` decorator for authoring custom kernels
- D2M (Data-to-Matmul) dialect Python bindings
- TensorBlock and MemTx operations for data movement
- CircularBuffer for inter-thread communication
- Semaphore for multi-core synchronization
- MetalLayoutAttr utilities for memory layout management
- Integration with tt-mlir compiler infrastructure
- LLVM lit-based testing framework
- Pre-commit hooks for code formatting (Black for Python, clang-format for C++)
- CMake build system with flexible tt-mlir integration modes

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
