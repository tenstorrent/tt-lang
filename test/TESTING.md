# Python Lit Testing Guide

## Overview

Python lit tests verify DSL functionality by checking generated MLIR and C++
output at two stages:
- **Initial IR**: MLIR before the compilation pipeline (ttl dialect operations)
- **C++ output**: Generated C++ kernel code after full compilation

Most lit tests run in compile-only mode (`TTLANG_COMPILE_ONLY=1`) and do not
require hardware. Tests that exercise end-to-end execution are gated by
`REQUIRES:` directives.

## Test Categories

The project has several test suites, each with a dedicated CMake target:

| Target | Path | Runner | Description |
|--------|------|--------|-------------|
| `check-ttlang-mlir` | `test/ttlang/` | llvm-lit | MLIR dialect and pass tests (no device) |
| `check-ttlang-python-lit` | `test/python/` (non-`test_*`) | llvm-lit (`-j1`) | Python DSL lit tests |
| `check-ttlang-python-bindings` | `test/bindings/python/` | llvm-lit | Python binding tests (no device) |
| `check-ttlang-pytest` | `test/python/test_*.py` | pytest | Parametric Python tests |
| `check-ttlang-me2e` | `test/me2e/` | pytest | Middle end-to-end tests |
| `check-ttlang` | | | Combines `check-ttlang-mlir` + `check-ttlang-python-bindings` (no device) |
| `check-ttlang-all` | | | All suites, run sequentially |

Files named `test_*.py` under `test/python/` are excluded from lit collection
and collected by pytest instead. The lit test runner and pytest each have their
own `conftest.py` for fixtures and configuration.

## Test Structure

Each Python lit test generates up to two output files:
- `%t.initial.mlir` -- IR before pipeline execution (set via
  `TTLANG_INITIAL_MLIR`)
- `%t.output` -- stdout/stderr, which contains the generated C++ kernel code

**Important:** Each test file should contain only one `@ttl.kernel` decorated
function. Multiple kernels in one file will overwrite the temp files, causing
only the last kernel to be checked.

## REQUIRES and UNSUPPORTED Directives

Tests declare their prerequisites with `REQUIRES:` and platform exclusions with
`UNSUPPORTED:`:

```python
# REQUIRES: ttnn, tt-device
```

Available features:
- `ttnn` -- the `ttnn` Python package is importable and functional
- `tt-device` -- a Tenstorrent device or simulator is available (detected at
  CMake configure time, or when `TT_METAL_SIMULATOR` is set)
- `system-darwin` -- running on macOS

## RUN Commands

A typical Python lit test has two or three `RUN` lines:

**Line 1:** Execute the test in compile-only mode and save initial IR + C++
output
```python
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
```

**Line 2:** Check initial IR
```python
# RUN: FileCheck %s < %t.initial.mlir
```

**Line 3 (optional):** Check generated C++ code
```python
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
```

Some tests omit `TTLANG_COMPILE_ONLY=1` if they require device execution (these
need the `tt-device` feature). Some tests only check initial IR and omit line 3.

### Negative/Invalid Tests

Tests under `test/python/invalid/` verify that the compiler rejects malformed
input. These use the `not` command to expect failure:

```python
# RUN: not %python %s 2>&1 | FileCheck %s --check-prefix=PRETTY
# RUN: env TTLANG_VERBOSE_ERRORS=1 not %python %s 2>&1 | FileCheck %s --check-prefix=VERBOSE
```

## Writing Good Checks

### Initial IR: Thorough Verification

Check the complete structure using SSA value captures. The initial IR contains
ttl dialect operations before lowering.

**Pattern:**
```python
# CHECK: #ttnn.buffer_type<l1>
# CHECK: #ttnn_layout = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>{{.*}}>

# CHECK-LABEL: func.func @compute_fn_name
# CHECK-SAME: attributes {ttl.base_cta_index = {{[0-9]+}} : i32, {{.*}}ttl.kernel_thread = #ttkernel.thread<compute>}

# Bind circular buffers
# CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1

# Wait for input, reserve output
# CHECK: %[[L:.+]] = ttl.cb_wait %[[CB0]]
# CHECK: ttl.cb_reserve %[[CB1]]

# Compute
# CHECK: ttl.add
# CHECK: ttl.store

# Finalize
# CHECK: ttl.cb_pop %[[CB0]]
# CHECK: ttl.cb_push %[[CB1]]

# CHECK-LABEL: func.func @dm_read_fn_name
# CHECK-SAME: %arg0: tensor<{{[^>]+}}!ttcore.tile<32x32, bf16>, #ttnn_layout>
# CHECK-SAME: attributes {{{.*}}ttl.kernel_thread = #ttkernel.thread<noc>}

# Data movement: reserve, slice, copy, wait, push
# CHECK: ttl.cb_reserve %[[CB0]]
# CHECK: %[[SLICE:.+]] = ttl.tensor_slice %arg0
# CHECK: %[[TX:.+]] = ttl.copy %[[SLICE]], %[[CB0]] : {{.*}} -> !ttl.transfer_handle<read>
# CHECK: ttl.wait %[[TX]]
# CHECK: ttl.cb_push %[[CB0]]
```

**Key practices:**
- Capture SSA values: `%[[NAME:.+]]`
- Reuse captures to verify data flow
- Use `{{.*}}` for attributes that may vary
- Use `{{[0-9]+}}` for generated numeric suffixes
- Verify `ttl.kernel_thread` attribute to confirm compute vs. noc thread
- Check circular buffer indices match across compute and data movement functions

### C++ Output: Smoke Testing

Verify the generated C++ kernel code without exhaustive checks. The output
contains C++ source for each kernel function.

**Pattern:**
```python
# CHECK-CPP: // compute_fn_name
# CHECK-CPP: void kernel_main()

# CB operations
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(1),

# DST register lifecycle
# CHECK-CPP: tile_regs_acquire();
# CHECK-CPP: copy_tile_init(get_compile_time_arg_val(0));
# CHECK-CPP: copy_tile(get_compile_time_arg_val(0),

# Compute
# CHECK-CPP: add_binary_tile_init();
# CHECK-CPP: add_binary_tile(

# Sync and pack
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();
# CHECK-CPP: pack_tile<true>(
# CHECK-CPP: tile_regs_release();

# Finalize
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(1),
```

**Focus on:**
- Each kernel has `// kernel_name` comment followed by `void kernel_main()`
- CB wait/reserve/push/pop operations with correct CB indices
- DST register lifecycle (`tile_regs_acquire/commit/wait/release`)
- Core compute calls (`add_binary_tile`, `exp_tile`, `sqrt_tile`, etc.)
- Data movement calls (`noc_async_read_tile`, `noc_async_write_tile`)

**Skip:**
- Exact variable names
- Index arithmetic details
- TensorAccessor boilerplate

## Running Tests

For all tests, first activate the build environment with
`source build/env/activate`.

All device-independent lit tests:
```bash
cmake --build build --target check-ttlang
```

All tests (run sequentially):
```bash
cmake --build build --target check-ttlang-all
```

Single MLIR lit test:
```bash
llvm-lit test/ttlang/path/to/test.mlir
```

Python lit tests (run sequentially with `-j1`):
```bash
llvm-lit -v build/test/python/
```

Single Python lit test:
```bash
llvm-lit -v build/test/python/simple_add.py
```

Pytest tests (parametric DSL tests):
```bash
pytest test/python/
```

Middle end-to-end tests (requires ttnn and a TT device or simulator):
```bash
pytest -v test/me2e/
```

Simulations (software simulation of runtime behavior):
```bash
pytest test/sim/
```

### Running with Simulator

For testing without hardware, set `TT_METAL_SIMULATOR` to enable the TT device
simulator. This makes the `tt-device` lit feature available and enables
device-requiring tests:

```bash
export TT_METAL_SIMULATOR=1
```

Then run tests as normal. ME2E tests with simulator:
```bash
TT_METAL_SIMULATOR=1 pytest -v test/me2e/
```

## Debugging Failed Tests

Generated files for lit tests are saved under `build/test/python/Output/`:

```bash
# View initial IR
cat build/test/python/Output/simple_add.py.tmp.initial.mlir

# View C++ output
cat build/test/python/Output/simple_add.py.tmp.output
```

Compare CHECK patterns against actual output to identify mismatches.

Temp kernel files generated during pytest are cleaned up at exit. To preserve
them for inspection:
```bash
TTLANG_KEEP_GENERATED_KERNELS=1 pytest -v test/python/test_elementwise_ops.py
```

## Environment Variables

Several environment variables control compilation, debugging, profiling, and
testing behavior.

### Compilation Control

| Variable | Description |
|----------|-------------|
| `TTLANG_COMPILE_ONLY` | Skip device execution, only compile. Set automatically by conftest.py when no hardware is available. |
| `TTLANG_INITIAL_MLIR` | Path to save MLIR before the compilation pipeline. |
| `TTLANG_FINAL_MLIR` | Path to save MLIR after the compilation pipeline (not used in lit tests but useful for manual debugging). |

Example:
```bash
TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=/tmp/initial.mlir TTLANG_FINAL_MLIR=/tmp/final.mlir python test.py
```

### Debugging

| Variable | Description |
|----------|-------------|
| `TTLANG_VERBOSE_PASSES` | Print IR after each compiler pass (verbose output). |
| `TTLANG_VERBOSE_ERRORS` | Show raw MLIR diagnostics in addition to pretty-printed error messages. |
| `TTLANG_DEBUG_LOCATIONS` | Include source file locations in MLIR output (e.g., `loc("file.py":N:M)`). |
| `TTLANG_KEEP_GENERATED_KERNELS` | Preserve temp kernel files generated by pytest. |

Example:
```bash
# Trace compilation through all passes
TTLANG_VERBOSE_PASSES=1 python test.py 2>&1 | tee pipeline.log

# Keep generated kernel files for inspection
TTLANG_KEEP_GENERATED_KERNELS=1 pytest -v test/python/test_uneven_grids.py
```

### Profiling

| Variable | Description |
|----------|-------------|
| `TTLANG_AUTO_PROFILE` | Enable automatic profiling with signposts (requires tt-mlir configured with performance tracing). |
| `TTLANG_PERF_DUMP` | Print NOC traffic and per-thread wall time summary after kernel execution. Requires `TT_METAL_DEVICE_PROFILER=1` and related Metal env vars. |
| `TTLANG_PROFILE_CSV` | Path to save profiling data as CSV. |

### Testing

| Variable | Description |
|----------|-------------|
| `TTLANG_TEST_SEED` | Seed for random test data generation (default: 42). |
| `TTLANG_HAS_DEVICE` | Set by CMake to indicate hardware or simulator availability. |
| `TT_METAL_SIMULATOR` | Enable the TT Metal device simulator (makes `tt-device` lit feature available). |
