# Compiler Options

TT-Lang exposes compiler options at two levels: the Python API (for kernel
authors) and `ttlang-opt` (for compiler developers working directly with MLIR).

## Python API

### Kernel Decorator Parameters

The `@ttl.kernel` decorator accepts parameters that control compilation and
runtime behavior:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `grid` | `tuple` or `Callable` | (required) | Compute grid dimensions, e.g., `(2, 2)` |
| `indexing_maps` | `list[Callable]` | `None` | Lambda functions for tile indexing |
| `iterator_types` | `list[str]` | `None` | `"parallel"` or `"reduction"` per dimension |
| `num_outs` | `int` | `1` | Number of output tensor arguments |
| `memory_space` | `str` | `"L1"` | Memory space for circular buffers: `"L1"` or `"DRAM"` |
| `tiled` | `bool` | `True` | Use tiled tensor layout |
| `fp32_dest_acc_en` | `bool` or `None` | `None` | Enable f32 accumulation in the DST register file. When `None`, auto-detected from input tensor dtypes (enabled when any input is f32). Maps to TTNN `ComputeConfigDescriptor.fp32_dest_acc_en` and the `ttl-set-compute-kernel-config{fp32-dest-acc-en=1}` MLIR pass option. |
| `dst_full_sync_en` | `bool` or `None` | `None` | Enable full DST synchronization (single-buffering mode). When enabled, DST capacity doubles (f32: 8, f16/bf16: 16) at the cost of requiring a full sync between math and pack threads. Maps to TTNN `ComputeConfigDescriptor.dst_full_sync_en` and the `ttl-set-compute-kernel-config{dst-full-sync-en=1}` MLIR pass option. |
| `options` | `str` or `None` | `None` | Compiler option string, e.g., `"--no-ttl-maximize-dst"` |

### Compiler Options (`CompilerOptions`)

Three boolean flags control the MLIR pass pipeline. These are the options
printed by `--ttl-help`:

| CLI Flag | Python field | Default | Description |
|---|---|---|---|
| `--ttl-maximize-dst` / `--no-ttl-maximize-dst` | `maximize_dst` | `True` | Partition `ttl.compute` iteration spaces into subblocks that maximize DST register utilization (`ttl-subblock-compute-for-dst` pass), and reorder tile operations within sync regions to group by operation kind (`ttl-schedule-operations` pass). Disabling this falls back to per-tile synchronization. |
| `--ttl-fpu-binary-ops` / `--no-ttl-fpu-binary-ops` | `enable_fpu_binary_ops` | `True` | Emit FPU binary elementwise ops (`add_tiles`, `sub_tiles`, `mul_tiles`) when both operands come from circular buffers. When disabled, binary ops use the SFPU path instead. Passed through to `ttl-assign-dst{enable-fpu-binary-ops=...}`. |
| `--ttl-block-matmul` / `--no-ttl-block-matmul` | `use_block_matmul` | `True` | Emit `matmul_block` (which processes the full block atomically) instead of per-tile matmul loops. The `ttl-lower-matmul-block` pass collapses the iteration domain to a single point and expands the output stores to cover all M×N DST registers. |

### How to Pass Options

Options can be specified in four ways, listed from lowest to highest priority:

1. **Class defaults** — the values in the `CompilerOptions` dataclass.
2. **Decorator `options=` parameter** — parsed from a string:
   ```python
   @ttl.kernel(grid=(2, 2), options="--no-ttl-maximize-dst")
   def my_kernel(a, b): ...
   ```
3. **`TTLANG_COMPILER_OPTIONS` environment variable** — merged on top of the
   decorator string:
   ```bash
   export TTLANG_COMPILER_OPTIONS="--no-ttl-fpu-binary-ops"
   python my_kernel.py
   ```
4. **Command-line arguments** (`sys.argv`) — highest priority, overrides
   everything:
   ```bash
   python my_kernel.py --no-ttl-maximize-dst --no-ttl-block-matmul
   ```

Only explicitly-set fields override; unmentioned flags fall through from lower
priority levels. Use `--ttl-help` to print the available options:

```bash
python examples/tutorial/multicore_grid_auto.py --ttl-help
```

### Runtime Overrides

The `options` keyword argument can also be passed at call time to override the
decorator value for a single invocation:

```python
my_kernel(tensor_a, tensor_b, options="--no-ttl-block-matmul")
```

### Environment Variables

These environment variables control compilation and diagnostic output. They are
independent of the `CompilerOptions` flags above.

| Variable | Type | Default | Description |
|---|---|---|---|
| `TTLANG_COMPILE_ONLY` | `0`/`1` | `0` | Compile kernels but do not execute them on hardware. |
| `TTLANG_INITIAL_MLIR` | file path | (unset) | Write the pre-optimization MLIR module to this file. |
| `TTLANG_FINAL_MLIR` | file path | (unset) | Write the post-optimization MLIR module to this file. |
| `TTLANG_VERBOSE_PASSES` | any value | (unset) | Print the IR after every pass in the pipeline. Output is very large; redirect to a file. |
| `TTLANG_DEBUG_LOCATIONS` | `0`/`1` | `0` | Include source locations in printed MLIR (locations are always tracked internally for error messages). |
| `TTLANG_VERBOSE_ERRORS` | `0`/`1` | `0` | Include raw MLIR diagnostics in error output. |

Profiling-related environment variables (`TTLANG_AUTO_PROFILE`,
`TTLANG_PERF_DUMP`, `TTLANG_PERF_SERV`, `TTLANG_SIGNPOST_PROFILE`,
`TTLANG_PROFILE_CSV`) are documented in the
[Performance Tools](performance-tools.md) reference.

## `ttlang-opt`

`ttlang-opt` is the MLIR optimizer driver for the TTL dialect. It is used
primarily for compiler development and testing — most kernel authors interact
with the compiler through the Python API. It accepts all standard `mlir-opt`
flags (run `ttlang-opt --help` for the full list) plus the TTL-specific passes
and pipelines below.

### Pipeline: `ttl-to-ttkernel-pipeline`

The main compilation pipeline, equivalent to what the Python API runs internally.
Invoke it with:

```bash
ttlang-opt input.mlir -p 'ttl-to-ttkernel-pipeline{maximize-dst=true lower-to-emitc=true}'
```

| Option | Type | Default | Description |
|---|---|---|---|
| `maximize-dst` | bool | `true` | Enable DST maximization via subblock compute and scheduling. |
| `enable-fpu-binary-ops` | bool | `true` | Use FPU for binary add/sub/mul. |
| `use-block-matmul` | bool | `true` | Lower matmul to block-level hardware calls (`experimental::matmul_block`) instead of per-tile loops. |
| `lower-to-emitc` | bool | `false` | Run the TTKernel-to-EmitC backend after the TTL lowering (produces C++ source). |

The pipeline runs these passes in order:

1. `convert-ttl-to-compute` — lower TTL elementwise tensor ops to `ttl.compute` with tile ops
2. `ttl-set-compute-kernel-config` — set `fp32_dest_acc_en` / `dst_full_sync_en` defaults
3. `ttl-assign-dst` — DST register allocation (linear scan with copy insertion)
4. `ttl-subblock-compute-for-dst` — tile `ttl.compute` into DST-sized subblocks *(only if `maximize-dst=true`)*
5. `ttl-insert-tile-regs-sync` — insert math/pack thread synchronization
6. `ttl-lower-matmul-block` — mark block-matmul computes and expand stores *(only if `use-block-matmul=true`)*
7. `ttl-lower-to-loops` — lower `ttl.compute` to `scf.for` loops
8. `ttl-schedule-operations` — reorder tile ops by dependency depth and kind *(only if `maximize-dst=true`)*
9. `ttl-annotate-cb-associations` — annotate block args with CB indices
10. `convert-ttl-to-ttkernel` — lower TTL DMA ops to TTKernel
11. `ttkernel-insert-inits` — insert hardware init ops before compute ops
12. Canonicalization and CSE cleanup
13. *(if `lower-to-emitc=true`)* `lower-affine`, `convert-ttkernel-to-emitc`, `emitc-form-expressions`

### Individual Pass Options

Each pass can be run standalone for testing. Only passes with configurable
options are listed here; the remaining passes have no options.

#### `ttl-set-compute-kernel-config`

Set default compute kernel configuration attributes on `ttl.compute` ops.

| Option | Type | Default | Description |
|---|---|---|---|
| `fp32-dest-acc-en` | bool | `false` | Default `fp32_dest_acc_en` for compute ops that do not already have it configured. |
| `dst-full-sync-en` | bool | `false` | Default `dst_full_sync_en` for compute ops that do not already have it configured. |

```bash
ttlang-opt input.mlir -p 'func.func(ttl-set-compute-kernel-config{fp32-dest-acc-en=1})'
```

#### `ttl-assign-dst`

DST register allocator using linear scan allocation with in-place operation
merging.

| Option | Type | Default | Description |
|---|---|---|---|
| `dst-capacity` | uint32_t | `0` (auto) | Override DST register capacity. Auto-computed from `fp32_dest_acc_en` and `dst_full_sync_en` by default. For single-buffering (`dst_full_sync_en=true`): f32=8, f16/bf16=16. For double-buffering (default): f32=4, f16/bf16=8. |
| `separate-output-region` | bool | `false` | Allocate outputs in a separate DST region (needed for reductions and some loop optimizations). |
| `enable-fpu-binary-ops` | bool | `true` | Use FPU for binary add/sub/mul when both operands come from CBs. When disabled, binary ops use the SFPU path. |

```bash
ttlang-opt input.mlir -p 'func.func(ttl-assign-dst{dst-capacity=16 enable-fpu-binary-ops=0})'
```

#### `ttl-dump-cb-flow-graph`

Analyze circular buffer producer/consumer relationships and dump the flow graph.

| Option | Type | Default | Description |
|---|---|---|---|
| `output` | string | `""` | Path to write JSON output. Empty string prints to stderr only. |

```bash
ttlang-opt input.mlir -p 'ttl-dump-cb-flow-graph{output="/tmp/cb_graph.json"}'
```
