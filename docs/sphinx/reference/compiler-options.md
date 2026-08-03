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
| `--ttl-accumulation-strategy {auto,dst,l1-pack}` | `auto` | Select the storage strategy for tensor recurrence accumulation scopes. DFB accumulation scopes always lower to L1 packer metadata. |
| `--ttl-fpu-binary-ops` / `--no-ttl-fpu-binary-ops` | enabled | Emit FPU binary elementwise ops (`add_tiles`, `sub_tiles`, `mul_tiles`) when both operands come from dataflow buffers. When disabled, binary ops use the SFPU path. |
| `--ttl-block-matmul` / `--no-ttl-block-matmul` | enabled | Emit `matmul_block` (processes the full tile block atomically) instead of per-tile matmul loops. Disabling this option is not yet supported. |
| `--ttl-subblock-sync` / `--no-ttl-subblock-sync` | disabled | Refine DFB reserve/push to per-subblock granularity, enabling `pack_tile_block` for contiguous subblocks. When disabled, user-placed reserve/push is preserved as written. |
| `--ttl-combine-pack-tiles` / `--no-ttl-combine-pack-tiles` | enabled | Combine consecutive `pack_tile` ops on the same DFB with contiguous DST and DFB indices into a single `pack_tile_block` call. |
| `--ttl-strict-f32-acc` / `--no-ttl-strict-f32-acc` | disabled | Error at compile time if a `+=` accumulation loop's output block exceeds f32 DST capacity (4 tiles with double-buffering). When enabled, guarantees each accumulation step fits in a single DST section without subblocking. |
| `--ttl-compiler-dfbs` / `--no-ttl-compiler-dfbs` | enabled | Insert compiler-allocated intermediate DFBs when an operation requires DFB-attached inputs or compute creation would read a source after its DFB is released. When disabled, the compiler emits an error if either materialization is required. |
| `--ttl-reuse-user-dfbs` / `--no-ttl-reuse-user-dfbs` | enabled | Reuse physical DFB indices when concurrent-kernel liveness proves that compatible logical DFB lifetimes do not overlap. Disabling retains user-declared physical indices. |
| `--ttl-pipe-computed-addresses` / `--no-ttl-pipe-computed-addresses` | enabled | Use computed receiver DFB addresses for eligible PipeNet transfers. When disabled, transfers use receiver-published destination addresses; multicast still requires proven equal runtime receiver addresses. |
| `--ttl-pipe-capacity-sync` / `--no-ttl-pipe-capacity-sync` | enabled | Use capacity-counter synchronization when the receiver wait and pop execute on the receiver NOC thread and the computed-address transfer passes the DFB ownership and count proofs. When disabled, computed-address transfers use receiver-post synchronization. |
| `--ttl-pipe-batch-tiles N` | `0` (auto) | Limit the logical transfers in one PipeTransport group. `0` selects automatically and `1` disables grouping. |
| `--ttl-l1-budget N` | target-dependent | Override the L1 allocation budget used by DFB validation and PipeTransport selection. |
| `--ttl-specialize-cores` / `--no-ttl-specialize-cores` | disabled | Clone each TTKernel function whose control flow branches on a core coordinate once per launch coordinate (`ttkernel-specialize-cores`), replacing `my_logical_x_` / `my_logical_y_` with constants and tagging clones with `ttl.core_coord` for per-core dispatch. Opt-in. |

**f32 accumulation precision:** `dst` keeps the accumulator in the DST register
but feeds it back through SRCA on each step, which truncates to tf32 (10-bit
mantissa); deep f32 recurrences therefore do not retain full f32 precision.
`auto` selects `dst` for DST-compatible recurrences and inherits the same limit.
Use `l1-pack` when full f32 accumulation precision is required; it accumulates
in f32 L1.

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
| `TTLANG_SIM_ONLY` | `0`/`1` | `0` | Force `import ttl` to skip loading the compiled MLIR extension. Used when running the simulator from a source tree without an installed `tt-lang-sim` wheel (which ships the same signal as a marker module). |

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
| `memory_space` | `str` | `"L1"` | Memory space for dataflow buffers: `"L1"` or `"DRAM"` |
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
| `accumulation-strategy` | string | `auto` | Select tensor recurrence accumulation storage strategy: `auto`, `dst`, or `l1-pack`. DFB accumulation scopes always lower to L1 packer metadata. |
| `enable-fpu-binary-ops` | bool | `true` | Use FPU for binary add/sub/mul. |
| `use-block-matmul` | bool | `true` | Lower matmul to block-level hardware calls (`matmul_block`). |
| `subblock-sync` | bool | `false` | Refine DFB reserve/push to per-subblock granularity. |
| `combine-pack-tiles` | bool | `true` | Combine consecutive `pack_tile` ops into `pack_tile_block`. |
| `strict-f32-acc` | bool | `false` | Error if a `+=` accumulation loop's output block exceeds f32 DST capacity. |
| `compiler-dfbs` | bool | `true` | Insert compiler-allocated intermediate DFBs for DFB-only operands and source-lifetime preservation. Error if disabled and any operation requires one. |
| `reuse-user-dfbs` | bool | `true` | Reuse physical DFB indices for compatible logical DFBs with proven non-overlapping concurrent lifetimes. |
| `pipe-computed-addresses` | bool | `true` | Use computed receiver DFB addresses for eligible PipeNet transfers. When disabled, transfers use receiver-published destination addresses; multicast still requires proven equal runtime receiver addresses. |
| `pipe-capacity-sync` | bool | `true` | Use capacity-counter synchronization when the receiver wait and pop execute on the receiver NOC thread and the computed-address transfer passes the DFB ownership and count proofs. When disabled, computed-address transfers use receiver-post synchronization. |
| `pipe-batch-tiles` | int64_t | `0` (auto) | Limit logical transfers per PipeTransport group. `0` selects automatically and `1` disables grouping. |
| `l1-budget-override` | uint32_t | `0` (target default) | Override the L1 allocation budget used by DFB validation and PipeTransport selection. |
| `specialize-cores` | bool | `false` | Clone TTKernel functions that branch on a core coordinate once per launch coordinate (`ttkernel-specialize-cores`), then run `canonicalize` / `cse`. Maps from `--ttl-specialize-cores`. |
| `lower-to-emitc` | bool | `false` | Run the TTKernel-to-EmitC backend (produces C++ source). |

The pipeline runs these passes in order:

- `ttl-form-accumulation-scopes` - form semantic accumulation scopes for eligible tensor recurrences
- `ttl-lower-accumulation-scopes` - lower tensor accumulation scopes with `strategy=<accumulation-strategy>`
- `ttl-materialize-loop-state` - remove ranked-tensor `scf.for` iter_args
- `ttl-insert-copy-wait` - insert missing `ttl.wait` after `ttl.copy` ops whose transfer handle has no wait user
- `ttl-auto-sync` - run `ttl-insert-cb-sync` and `ttl-coalesce-dfb-acquires`
- `ttl-insert-accumulation-scopes{kind=dfb}` - insert semantic accumulation scopes for user-written `+=` loops
- `ttl-lower-accumulation-scopes{kind=dfb}` - lower user-written `+=` scopes to L1 packer metadata
- `ttl-create-producer-compute` - create producer `ttl.compute` operations before intermediate materialization
- `ttl-insert-intermediate-dfbs` - materialize DFB-only operands and values that must be preserved before source release; verify and error when `compiler-dfbs=false`
- `convert-ttl-to-compute` - lower TTL elementwise tensor ops to `ttl.compute` with tile ops
- `ttl-insert-cb-sync` - insert missing DFB synchronization operations
- `ttl-verify-pipenet-guards`, then `ttl-verify-pipenet-schedule` - verify PipeNet launch domains and synchronization schedules while logical DFB identities remain distinct
- `ttl-form-pipe-transports` - group eligible repeated PipeNet transfers and select bounded receiver storage
- `ttl-coalesce-dfb-acquires` - coalesce compatible DFB acquisitions
- `ttl-finalize-dfb-indices` - assign logical DFBs to physical indices and emit runtime allocation metadata; controlled by `reuse-user-dfbs`
- `ttl-set-compute-kernel-config` - set `fp32_dest_acc_en` / `dst_full_sync_en` defaults
- `ttl-assign-dst` - DST register allocation (linear scan with copy insertion)
- `ttl-subblock-compute-for-dst` - tile `ttl.compute` into DST-sized subblocks *(only if `maximize-dst=true`)*; optionally refine reserve/push to per-subblock granularity *(only if `subblock-sync=true`)*
- `ttl-lower-to-loops` - lower `ttl.compute` to `scf.for` loops; matmul computes are expanded inline via `generateMatmulCompute`
- `ttl-schedule-operations` - reorder tile ops by dependency depth and kind *(only if `maximize-dst=true`)*
- `ttl-annotate-cb-associations` - annotate block args with DFB indices
- `ttl-verify-dfb-spsc` - verify one producer and one consumer per launched node
- `ttl-erase-pipenet-scopes` - remove verified PipeNet scope markers
- `ttl-validate-cb-budget` - verify static DFB storage fits the per-core L1 budget
- `convert-ttl-to-ttkernel` - lower TTL DMA and PipeNet operations to TTKernel, selecting receiver-published or computed destination addressing and receiver-post or capacity-counter synchronization
- `ttkernel-insert-inits` - insert hardware init ops before compute ops
- `ttkernel-insert-l1-accumulation` - insert `pack_reconfig_l1_acc` guards for `+=` and reduction loops
- `ttkernel-combine-pack-tiles` - combine consecutive `pack_tile` into `pack_tile_block` *(only if `combine-pack-tiles=true`)*
- Canonicalization and CSE cleanup
- `ttkernel-specialize-cores`, then `canonicalize`, `cse` -- per-core clone and const-fold of coordinate branches; tags clones with `ttl.core_coord` *(only if `specialize-cores=true`)*
- *(if `lower-to-emitc=true`)* `lower-affine`, `convert-ttkernel-to-emitc`, `emitc-form-expressions`

### Individual Pass Options

Each pass can also be run standalone for testing. Only passes with configurable
options are listed; the remaining passes have no options.

#### `ttl-insert-accumulation-scopes`

Insert semantic accumulation scopes before concrete strategy selection.

| Option | Type | Default | Description |
|---|---|---|---|
| `kind` | string | `"tensor"` | Scope insertion kind. Supported values: `tensor`, `dfb`. |

```bash
ttlang-opt input.mlir -p 'func.func(ttl-insert-accumulation-scopes{kind=tensor})'
```

#### `ttl-lower-accumulation-scopes`

Lower semantic accumulation scopes to a concrete storage strategy.

| Option | Type | Default | Description |
|---|---|---|---|
| `kind` | string | `"tensor"` | Scope lowering kind. Supported values: `tensor`, `dfb`. |
| `strategy` | string | `"auto"` | Tensor recurrence accumulation strategy. Supported values: `auto`, `dst`, `l1-pack`. Ignored for `kind=dfb`. |

```bash
ttlang-opt input.mlir -p 'func.func(ttl-lower-accumulation-scopes{strategy=dst})'
```

#### `ttl-insert-intermediate-dfbs`

Insert compiler-allocated intermediate DFBs at fusion split points.

| Option | Type | Default | Description |
|---|---|---|---|
| `enable` | bool | `true` | Insert compiler-allocated DFBs. When false, emit an error if any operation requires one. |

```bash
ttlang-opt input.mlir -p 'func.func(ttl-insert-intermediate-dfbs{enable=false})'
```

#### `ttl-finalize-dfb-indices`

Assign physical indices to logical DFBs and emit the complete runtime
allocation table.

| Option | Type | Default | Description |
|---|---|---|---|
| `reuse-user-dfbs` | bool | `true` | Reuse a physical index when concurrent-kernel liveness proves that two compatible logical DFB lifetimes cannot overlap. When false, retain user DFB indices and reuse only compiler-created DFBs within each kernel. |

```bash
ttlang-opt input.mlir -p 'builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})'
```

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
| `enable-fpu-binary-ops` | bool | `true` | Use FPU for binary add/sub/mul when both operands come from DFBs. When disabled, binary ops use the SFPU path. |

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

#### `ttl-form-pipe-transports`

Group eligible repeated PipeNet transfers and select bounded receiver storage.
Later PipeTransport planning replaces proven-private grouped DFB lifecycles
with transport-owned scratch; scalar residuals retain the original lifecycle.
Selection accounts for DFB allocation, a conservative receiver-published
address table, and transport scratch. Modules containing PipeNet foreach
callbacks remain unchanged; their record selection is lowered by
`convert-ttl-to-ttkernel`.

| Option | Type | Default | Description |
|---|---|---|---|
| `group-size` | int64_t | `0` (auto) | Limit logical transfers per group. `0` selects automatically and `1` disables grouping. |
| `l1-budget-override` | uint32_t | `0` (target default) | Override the combined DFB and pipe scratch budget used during grouping selection. |

```bash
ttlang-opt input.mlir --ttl-form-pipe-transports='group-size=8'
```

#### `convert-ttl-to-ttkernel`

Lower TTL data movement and PipeNet operations to TTKernel.

| Option | Type | Default | Description |
|---|---|---|---|
| `reduce-full-fp32` | bool | `true` | Enable FP32 accumulation for reduce operations. |
| `pipe-computed-addresses` | bool | `true` | Use computed receiver DFB addresses for eligible PipeNet transfers. When false, transfers use receiver-published destination addresses; multicast still requires proven equal runtime receiver addresses. |
| `pipe-capacity-sync` | bool | `true` | Use capacity-counter synchronization when the receiver wait and pop execute on the receiver NOC thread and the computed-address transfer passes the DFB ownership and count proofs. When false, computed-address transfers use receiver-post synchronization. |

```bash
ttlang-opt input.mlir -p 'builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false})'
```

#### `ttl-dump-cb-flow-graph`

Analyze dataflow buffer producer/consumer relationships and dump the flow graph.

| Option | Type | Default | Description |
|---|---|---|---|
| `output` | string | `""` | Path to write JSON output. Empty string prints to stderr only. |

```bash
ttlang-opt input.mlir -p 'ttl-dump-cb-flow-graph{output="/tmp/cb_graph.json"}'
```

#### `ttkernel-specialize-cores`

Clone TTKernel functions that branch on a core coordinate once per launch
coordinate. Requires a module-level `ttl.launch_grid` attribute (an i64 array
of length 2 with positive entries). Missing or malformed `ttl.launch_grid` is
a hard error. A valid single-core grid (product <= 1) skips specialization.

Only `scf.if` conditions derived from `ttkernel.my_logical_x_` /
`ttkernel.my_logical_y_` trigger cloning. Functions with symbol uses (for
example `func.call` targets) are left unspecialized with a warning so erasing
the original does not leave dangling `SymbolRefAttr`s; unrelated functions in
the module are still specialized. Each clone replaces coordinate reads with
`arith.constant`s and is tagged with `ttl.core_coord` for runtime dispatch.
Downstream `canonicalize` / `cse` fold the now-constant branches.

This pass is off by default. Enable it through the pipeline option
`specialize-cores` (Python: `--ttl-specialize-cores`):

```bash
ttlang-opt input.mlir -p 'ttl-to-ttkernel-pipeline{specialize-cores=true lower-to-emitc=true}'
# Or stand-alone:
ttlang-opt input.mlir -p 'builtin.module(ttkernel-specialize-cores,canonicalize,cse)'
```
