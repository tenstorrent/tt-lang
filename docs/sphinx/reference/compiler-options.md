# Compiler Options

## Code Generation Options

These flags control how TT-Lang compiles operations. Pass them on the command line,
or print the list with `--ttl-help`:

```bash
python my_kernel.py --ttl-help
python my_kernel.py --no-ttl-maximize-dst
```

| Flag | Default | Description |
|---|---|---|
| `--ttl-maximize-dst` / `--no-ttl-maximize-dst` | enabled | Partition compute iteration spaces into subblocks that maximize DST register utilization, and reorder tile operations within sync regions to group by kind. Disabling falls back to per-tile synchronization. |
| `--ttl-fpu-binary-ops` / `--no-ttl-fpu-binary-ops` | enabled | Emit FPU binary elementwise ops (`add_tiles`, `sub_tiles`, `mul_tiles`) when both operands come from circular buffers. When disabled, binary ops use the SFPU path. |
| `--ttl-block-matmul` / `--no-ttl-block-matmul` | enabled | Emit `matmul_block` (processes the full tile block atomically) instead of per-tile matmul loops. Disabling this option is not yet supported. |
| `--ttl-auto-sync` / `--no-ttl-auto-sync` | disabled | Let the compiler insert and move DFB synchronization ops. When enabled, reserve/push may be refined to per-subblock granularity. When disabled, user-placed reserve/push is preserved as written. |
| `--ttl-combine-pack-tiles` / `--no-ttl-combine-pack-tiles` | enabled | Combine consecutive `pack_tile` ops on the same CB with contiguous DST and CB indices into a single `pack_tile_block` call. |
| `--ttl-strict-f32-acc` / `--no-ttl-strict-f32-acc` | disabled | Error at compile time if a `+=` accumulation loop's output block exceeds f32 DST capacity (4 tiles with double-buffering). When enabled, guarantees each accumulation step fits in a single DST section without subblocking. |

### Other Ways to Set These

Besides the command line, the same flags can be set through three other
mechanisms. When the same flag is set in multiple places, higher-priority sources
win and unmentioned flags fall through from lower levels:

| Priority | Mechanism | Example |
|---|---|---|
| 1 (lowest) | `CompilerOptions` class defaults | — |
| 2 | `@ttl.operation` decorator `options=` parameter | `@ttl.operation(grid=(2,2), options="--no-ttl-maximize-dst")` |
| 3 | `TTLANG_COMPILER_OPTIONS` environment variable | `export TTLANG_COMPILER_OPTIONS="--no-ttl-fpu-binary-ops"` |
| 4 (highest) | Command-line arguments (`sys.argv`) | `python my_kernel.py --no-ttl-maximize-dst` |

The `options` keyword can also be passed at call time to override the decorator
for a single invocation:

```python
my_kernel(tensor_a, tensor_b, options="--no-ttl-fpu-binary-ops")
```

## Compute Configuration

These two parameters are set on the `@ttl.operation` decorator (not via command-line
flags) and control the TTNN compute kernel hardware configuration:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fp32_dest_acc_en` | `bool` or `None` | `None` | Enable f32 accumulation in the DST register file. When `None`, auto-detected from input tensor dtypes (enabled when any input is f32). |
| `dst_full_sync_en` | `bool` or `None` | `None` | Enable full DST synchronization (single-buffering mode). Doubles DST capacity (f32: 8, f16/bf16: 16) at the cost of a full sync between math and pack threads. |

```python
@ttl.operation(grid=(2, 2), fp32_dest_acc_en=True, dst_full_sync_en=False)
def my_kernel(a, b): ...
```

## Environment Variables

These environment variables control compilation behavior and diagnostic output.
They are independent of the code generation flags above.

| Variable | Type | Default | Description |
|---|---|---|---|
| `TTLANG_COMPILE_ONLY` | `0`/`1` | `0` | Compile kernels but do not execute on hardware. |
| `TTLANG_INITIAL_MLIR` | file path | (unset) | Write the pre-optimization MLIR module to this file. |
| `TTLANG_FINAL_MLIR` | file path | (unset) | Write the post-optimization MLIR module to this file. |
| `TTLANG_VERBOSE_PASSES` | any value | (unset) | Print the IR after every pass in the pipeline. Output is very large; redirect to a file. |
| `TTLANG_DEBUG_LOCATIONS` | `0`/`1` | `0` | Include source locations in printed MLIR (locations are always tracked internally for error messages). |
| `TTLANG_VERBOSE_ERRORS` | `0`/`1` | `0` | Include raw MLIR diagnostics in error output. |

Profiling-related environment variables (`TTLANG_AUTO_PROFILE`,
`TTLANG_PERF_DUMP`, `TTLANG_PERF_SERV`, `TTLANG_SIGNPOST_PROFILE`,
`TTLANG_PROFILE_CSV`) are documented in the
[Performance Tools](performance-tools.md) reference.

## Other Decorator Parameters

The `@ttl.operation` decorator also accepts these parameters for operation structure
and layout:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `grid` | `tuple` or `Callable` | (required) | Compute grid dimensions, e.g., `(2, 2)` |
| `indexing_maps` | `list[Callable]` | `None` | Lambda functions for tile indexing |
| `iterator_types` | `list[str]` | `None` | `"parallel"` or `"reduction"` per dimension |
| `num_outs` | `int` | `1` | Number of output tensor arguments |
| `memory_space` | `str` | `"L1"` | Memory space for circular buffers: `"L1"` or `"DRAM"` |
| `tiled` | `bool` | `True` | Use tiled tensor layout |

---

## `ttlang-opt` Pass Reference

`ttlang-opt` is the standalone MLIR optimizer driver for the TTL dialect, used
primarily for compiler development and testing. It accepts all standard
`mlir-opt` flags (run `ttlang-opt --help` for the full list) plus the
TTL-specific passes and pipeline documented below.

### Pipeline: `ttl-to-ttkernel-pipeline`

The main compilation pipeline, equivalent to what the Python API runs internally.

```bash
ttlang-opt input.mlir -p 'ttl-to-ttkernel-pipeline{maximize-dst=true lower-to-emitc=true}'
```

| Option | Type | Default | Description |
|---|---|---|---|
| `maximize-dst` | bool | `true` | Enable DST maximization via subblock compute and scheduling. |
| `enable-fpu-binary-ops` | bool | `true` | Use FPU for binary add/sub/mul. |
| `use-block-matmul` | bool | `true` | Lower matmul to block-level hardware calls (`experimental::matmul_block`). |
| `auto-sync` | bool | `false` | Let the compiler insert and move DFB synchronization ops. |
| `combine-pack-tiles` | bool | `true` | Combine consecutive `pack_tile` ops into `pack_tile_block`. |
| `strict-f32-acc` | bool | `false` | Error if a `+=` accumulation loop's output block exceeds f32 DST capacity. |
| `lower-to-emitc` | bool | `false` | Run the TTKernel-to-EmitC backend (produces C++ source). |

The pipeline runs these passes in order:

1. `ttl-annotate-l1-acc-loops` — detect `+=` accumulation loops and annotate for L1 packer accumulation
2. `convert-ttl-to-compute` — lower TTL elementwise tensor ops to `ttl.compute` with tile ops
3. `ttl-set-compute-kernel-config` — set `fp32_dest_acc_en` / `dst_full_sync_en` defaults
4. `ttl-assign-dst` — DST register allocation (linear scan with copy insertion)
5. `ttl-subblock-compute-for-dst` — tile `ttl.compute` into DST-sized subblocks *(only if `maximize-dst=true`)*; optionally refine reserve/push to per-subblock granularity *(only if `auto-sync=true`)*
6. `ttl-insert-tile-regs-sync` — insert math/pack thread synchronization
7. `ttl-lower-matmul-block` — mark block-matmul computes and expand stores *(only if `use-block-matmul=true`)*
8. `ttl-lower-to-loops` — lower `ttl.compute` to `scf.for` loops
9. `ttl-schedule-operations` — reorder tile ops by dependency depth and kind *(only if `maximize-dst=true`)*
10. `ttl-annotate-cb-associations` — annotate block args with CB indices
11. `convert-ttl-to-ttkernel` — lower TTL DMA ops to TTKernel
12. `ttkernel-insert-inits` — insert hardware init ops before compute ops
13. `ttkernel-insert-l1-accumulation` — insert `pack_reconfig_l1_acc` guards for `+=` and reduction loops
14. `ttkernel-combine-pack-tiles` — combine consecutive `pack_tile` into `pack_tile_block` *(only if `combine-pack-tiles=true`)*
15. Canonicalization and CSE cleanup
16. *(if `lower-to-emitc=true`)* `lower-affine`, `convert-ttkernel-to-emitc`, `emitc-form-expressions`

### Individual Pass Options

Each pass can also be run standalone for testing. Only passes with configurable
options are listed; the remaining passes have no options.

#### `ttl-set-compute-kernel-config`

Set default compute kernel configuration attributes on `ttl.compute` ops.

| Option | Type | Default | Description |
|---|---|---|---|
| `fp32-dest-acc-en` | bool | `false` | Default `fp32_dest_acc_en` when not already configured. |
| `dst-full-sync-en` | bool | `false` | Default `dst_full_sync_en` when not already configured. |

```bash
ttlang-opt input.mlir -p 'func.func(ttl-set-compute-kernel-config{fp32-dest-acc-en=1})'
```

#### `ttl-assign-dst`

DST register allocator using linear scan allocation with in-place operation
merging.

| Option | Type | Default | Description |
|---|---|---|---|
| `dst-capacity` | uint32_t | `0` (auto) | Override DST register capacity. Auto-computed from `fp32_dest_acc_en` and `dst_full_sync_en` by default. Single-buffering (`dst_full_sync_en=true`): f32=8, f16/bf16=16. Double-buffering (default): f32=4, f16/bf16=8. |
| `separate-output-region` | bool | `false` | Allocate outputs in a separate DST region (needed for reductions and some loop optimizations). |
| `enable-fpu-binary-ops` | bool | `true` | Use FPU for binary add/sub/mul when both operands come from CBs. When disabled, binary ops use the SFPU path. |

```bash
ttlang-opt input.mlir -p 'func.func(ttl-assign-dst{dst-capacity=16 enable-fpu-binary-ops=0})'
```

#### `ttl-subblock-compute-for-dst`

Partition `ttl.compute` into DST-sized subblocks.

| Option | Type | Default | Description |
|---|---|---|---|
| `subblock-sync` | bool | `false` | Refine DFB reserve/push to per-subblock granularity, enabling `pack_tile_block` for contiguous subblocks. When disabled, user-placed reserve/push is preserved. |
| `strict-f32-acc` | bool | `false` | Error if a `+=` accumulation loop with non-f32 output requires subblocking. Subblocking reduces accumulation precision because bf16 L1 intermediates truncate f32 DST values. |

```bash
ttlang-opt input.mlir -p 'func.func(ttl-subblock-compute-for-dst{subblock-sync=true})'
```

#### `ttl-dump-cb-flow-graph`

Analyze circular buffer producer/consumer relationships and dump the flow graph.

| Option | Type | Default | Description |
|---|---|---|---|
| `output` | string | `""` | Path to write JSON output. Empty string prints to stderr only. |

```bash
ttlang-opt input.mlir -p 'ttl-dump-cb-flow-graph{output="/tmp/cb_graph.json"}'
```
