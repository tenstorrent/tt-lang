# Changelog

All notable changes to TT-Lang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 1.1.1

### Compiler

- Fix for live-interval boundary computation (issue [#536](../../issues/536))
- Fix for all-zero results in FP32 reductions (issue # [#533](../../issues/533))
- Fix for inferred `pop` and `push` (issues [#536](../../issues/536), [#554](../../issues/554))
- Fix for write pointer tracking on pipe sender accross iterations (issue [#578](../../issues/578))
- Fix to report data type mismatch error
- Fix to report DFB over allocation error (issue [#511](../../issues/511))
- Support for pipenet predicates `is_src`, `is_dst` and `is_active` (issue [#541](../../issues/541))
- Support for `ttl.math.typecast`

### Simulator

- Support for inferred `pop`, `push` and `copy`'s transfer handle `wait`
- Support for pipenet predicates `is_src`, `is_dst` and `is_active`
- Support `all_gather`
- Support `bfloat8_b`
- Improved/actionable error messages
- Improved performance by simulating math in FP32

### Infrastructure

- TT-Lang installable with `pip install tt-lang` for full installation and `pip install tt-lang-sim` for simulator only
- [Matmul benchmarks](benchmarks/matmul/README.md)

## Version 1.0.0

### Compiler

- Support `+=` syntax in conjunction with dot product (`@`) lowered to packer L1 accumulation
- Support implicit temporary compute-kernel-local DFBs
- Support `ttl.Pipenet`
- Support implicit `ttl.Block.push` and `ttl.Block.pop`
- Support implicit `ttl.Transfer.wait`
- Support for `expm1`, `exp2`, `ceil`, `sign`, `gelu`, `silu`, `hardsigmoid`, `square`, `softsign`, `signbit`, `frac`, `trunc` in `ttl.math`

### Simulator

- Support for `ttl.GroupTransfer`
- SPMD and mesh device simulation support
- Support for `ttnn.all_reduce` CCLs
- Use tracing to report statistics with `ttlang-sim-stats`
- Remote L1 reads/writes statistics

### Examples and documentation
- Matmul tutorial

## Version 0.1.8

### Compiler

- Support for dot product operator (`@`) with lowering to [`ckernel::matmul_block`](https://docs.tenstorrent.com/tt-metal/v0.55.0/tt-metalium/tt_metal/apis/kernel_apis/compute/matmul_block.html)
- Support for fusing matmul and certain elementwise operations
- Support lowering to `pack_tile_block`
- Support for `ttl.math.fill`, `ttl.math.reduce_sum`, `ttl.math.reduce_max`, and `ttl.math.transpose`
- Support for arbitrary sub-blocking including dot product K-dimension to allow maximizing L1 usage and reuse
- Support for `sin`, `cos`, `tan`, `asin`, `acos`, `atan` in `ttl.math`
- Support for L1 sharded tensors
- Support for tensors with BF8 data type
- SPMD support (`ttnn.open_mesh_device`)

### Simulator

- Track L1 space and number of DFBs usage and warn when exceeded
- Support for tensors with row-major layout
- Support for L1 sharded tensors

### Examples and documentation
- Elementwise tutorial
- Image upsample with row-major tensors

## Version 0.1.7

### Compiler

- Implemented compute expression optimizations (tiling and unrolling) to maximize DST usage
- Implemented support for elementwise operations to use FPU when possible
- Added support for debug prints
- Added support for auto-profiling, profiling with user specified scopes (`ttl.signpost`) and performance summary
- Enabled interactive visualization of profiling results with [Perfetto](https://perfetto.dev/)
- Added support for `/`, `min`, `max`, `floor`, `recip` from `ttl.math`
- Added support for 3D+ blocks

### Simulator

- Reimplemented with greenlets to enable deterministic scheduling
- Added support for greedy and fair scheduling modes
- Added CLI options for setting hardware limits such as grid size, number of DFBs etc
- Added using TT-NN golden functions for simulations of TT-NN
- Added enforcement for block state machines
- Added support for `ttl.math` functions
- Added support for 3D+ blocks
- Added support for collecting performance statistics
- Improved various error messages
- Added support for debug prints
- Added support for VSCode step-by-step debugger

### Examples

- C++ Metal examples for single-, multicore with reuse and 1D matmul
- TT-Lang  examples for single-, multicore with reuse and 1D matmul

### Infrastructure

- Simplified dependency management, build, CI and reduced Docker container size from 9.48GB to 6.47GB.

## Version 0.1.3

- Added tutorial examples under `examples/tutorial`
- Implemented compatible `ttl.math.broadcast` in simulator and compiler
- Added support for [pipes and pipenets](https://github.com/tenstorrent/tt-lang/blob/main/docs/sphinx/specs/TTLangSpecification.md#pipe) in simulator
