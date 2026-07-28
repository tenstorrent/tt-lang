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
| `--ttl-fpu-binary-ops` / `--no-ttl-fpu-binary-ops` | enabled | Emit FPU binary elementwise ops (`add_tiles`, `sub_tiles`, `mul_tiles`) when both operands come from dataflow buffers. When disabled, binary ops use the SFPU path. |
| `--ttl-block-matmul` / `--no-ttl-block-matmul` | enabled | Emit `matmul_block` (processes the full tile block atomically) instead of per-tile matmul loops. Disabling this option is not yet supported. |
| `--ttl-subblock-sync` / `--no-ttl-subblock-sync` | disabled | Refine DFB reserve/push to per-subblock granularity, enabling `pack_tile_block` for contiguous subblocks. When disabled, user-placed reserve/push is preserved as written. |
| `--ttl-combine-pack-tiles` / `--no-ttl-combine-pack-tiles` | enabled | Combine consecutive `pack_tile` ops on the same DFB with contiguous DST and DFB indices into a single `pack_tile_block` call. |
| `--ttl-strict-f32-acc` / `--no-ttl-strict-f32-acc` | disabled | Error at compile time if a `+=` accumulation loop's output block exceeds f32 DST capacity (4 tiles with double-buffering). When enabled, guarantees each accumulation step fits in a single DST section without subblocking. |
| `--ttl-compiler-dfbs` / `--no-ttl-compiler-dfbs` | enabled | Insert compiler-allocated intermediate DFBs at fusion split points where an operation requires DFB-attached inputs (reduce, broadcast, matmul, transpose). When disabled, the compiler emits an error if any fused computation requires an intermediate DFB. |
| `--ttl-specialize-cores` / `--no-ttl-specialize-cores` | disabled | Clone each TTKernel function whose control flow branches on a core coordinate once per launch coordinate (`ttkernel-specialize-cores`), replacing `my_logical_x_` / `my_logical_y_` with constants and tagging clones with `ttl.core_coord` for per-core dispatch. Opt-in. |

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
| `TTLANG_CB_TABLE` | file path or `0` | `/tmp/ttlang_cb_table.txt` | Destination for the CB table (see below). `0` disables the write. |

### The CB table

Every compile appends a table of its circular buffers to
`/tmp/ttlang_cb_table.txt`. It answers "which physical CB is `acc`?", which the
kernel source cannot: `ttl-finalize-dfb-indices` reuse-colors user DFBs and then
compacts the survivors, so ids are renumbered and several logical names can
share one slot. Each row lists every name that landed on the slot, so an id
naming two DFBs means those two were merged.

```
=== 2026-07-28T15:42:10Z pid 95555 program_hash 1234 source add.py ===
tt-lang CB table: 3 CBs, 69632 bytes of L1 backing store
  id  names          shape  tile   blk  dtype     page  pages  bytes
  0   lhs_dfb        1x1    32x32  2    BFLOAT16  2048  2      4096
  1   acc, acc_wide  2x4    32x32  3    BFLOAT16  2048  24     49152
  2   <compiler>     -      -      2    BFLOAT16  2048  8      16384
```

The first compile in a process truncates the file and later ones append, so the
last block is the program that ran last, and the stamp identifies stale files.
`<compiler>` marks a slot the compiler allocated for an intermediate, which has
no user-visible name.

## Hang detection

tt-metal's dispatch waits are progress-gated against a device-side counter that
the dispatch kernel increments as it completes each command, so a queue that is
still retiring work never trips the timeout however long the host waits. By
default there is no window at all (`0.0` means wait forever) and nothing runs
when one trips. tt-lang arms both.

Two consequences worth knowing before trusting the window. Host-side work,
compilation included, is outside the guarded waits, so it neither counts as
progress nor consumes the window: a two-minute JIT compile cannot trip a
five-second timeout. But the counter only moves when a command *completes*, so a
single dispatch command that legitimately runs longer than the window is
indistinguishable from a hang. tt-lang programs run in microseconds to
milliseconds, hence the five second default; raise `TTLANG_HANG_TIMEOUT_SECONDS`
if you deliberately launch something that occupies the device for longer than
that in one command, or if you enable DPRINT heavily enough that draining the
buffer stalls a kernel.

### Why the window is not shorter still

`TT_METAL_OPERATION_TIMEOUT_SECONDS`, the variable this sets, is overloaded. It
also bounds two things that are not hangs, as plain wall clock rather than
progress-gated:

- `wait_until_cores_done` at device init and teardown, which is *unbounded* when
  the variable is unset (`llrt.cpp:376-382`).
- the fabric topology mapping rendezvous, whose own default when the variable is
  unset is 120 seconds (`control_plane.cpp:437-440`).

Setting it at all therefore bounds device open for the first time. The fabric
rendezvous is a cross-host all-gather, so it costs nothing at `world_size 1`, and
core-done polling is milliseconds per chip. If a device open starts failing with
cores not done, or with a topology mapping timeout, raise this variable: that is
the symptom of the window being too tight, not a device fault.

| Variable | Type | Default | Description |
|---|---|---|---|
| `TTLANG_ON_HANG` | `off`/`on` | `on` | Whether a dispatch timeout is detected and collected. |
| `TTLANG_HANG_TIMEOUT_SECONDS` | seconds | `5` | Time without *any* dispatch progress that counts as a hang. |
| `TTLANG_HANG_DIR` | directory | `/tmp/ttlang_hang` | Where the incident is written. |
| `TTLANG_HANG_DEVICES` | id list | `0` | Devices to sample. Widen with `0,1,2`; a 32-chip sweep takes minutes. |
| `TTLANG_FORCE_REINIT` | `0`/`1` | `1` | Set `TT_METAL_FORCE_REINIT`, so the next device open resets the RISCs. |

`on` reports the hang and collects; `off` restores tt-metal's default of waiting
forever.

Collection **acts on nothing**. It reads every PC off the debug bus and resolves
frames from DWARF, so no core is halted, the process is not stopped, and the device
is left exactly as the hang found it. tt-metal's timeout then throws as it always
would. Inspecting the incident, killing the process and resetting the device are
yours to decide, in that order.

Two things follow from that. Halting a core to unwind frames further up the stack
is terminal on Blackhole, so only the top frame and its inlined frames are
collected. And whatever closes the device on the way out (a caller's `finally`, a
pytest fixture) will wait the full timeout for every chip's stuck dispatch cores,
because that teardown wait cannot succeed while workers are stuck; tt-metal catches
and discards the result anyway (`dispatch_kernel_initializer.cpp:249-251`). Reset
with `tt-smi` rather than waiting it out.

The incident directory holds:

```
report.txt        what happened, and where everything else is
stacks.txt        per RISC: PC, STATIONARY or ADVANCING, symbolized frames
manifest.json     identity, ELFs used, cores sampled, per-step failures
programs.jsonl    every program compiled in this process
kernels/*.cpp     copies of the generated sources, which live in /tmp
```

`STATIONARY` versus `ADVANCING` is the first thing to read: a stationary PC
inside a `cb_wait_front` is a starved consumer, while an advancing PC is a
livelock. Frames resolve into hand-written `call_extern_func` headers too,
because those are `#include`d into the generated kernel.

A timeout that surfaces outside a tt-lang launch (dispatch is asynchronous, so it
can land in the next ttnn call) is still collected, but recovery needs the
exception. Add `pytest_plugins = ["ttl.hang_pytest"]` to a repository's
`conftest.py` to handle it wherever it lands in a test, or call
`ttl.hang.handle_hang(error)` directly.

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
| `enable-fpu-binary-ops` | bool | `true` | Use FPU for binary add/sub/mul. |
| `use-block-matmul` | bool | `true` | Lower matmul to block-level hardware calls (`matmul_block`). |
| `subblock-sync` | bool | `false` | Refine DFB reserve/push to per-subblock granularity. |
| `combine-pack-tiles` | bool | `true` | Combine consecutive `pack_tile` ops into `pack_tile_block`. |
| `strict-f32-acc` | bool | `false` | Error if a `+=` accumulation loop's output block exceeds f32 DST capacity. |
| `compiler-dfbs` | bool | `true` | Insert compiler-allocated intermediate DFBs for fused computations. Error if disabled and any operation requires one. |
| `specialize-cores` | bool | `false` | Clone TTKernel functions that branch on a core coordinate once per launch coordinate (`ttkernel-specialize-cores`), then run `canonicalize` / `cse`. Maps from `--ttl-specialize-cores`. |
| `lower-to-emitc` | bool | `false` | Run the TTKernel-to-EmitC backend (produces C++ source). |

The pipeline runs these passes in order:

- `ttl-insert-intermediate-dfbs` — allocate compiler-managed DFBs for intermediate values (transposes, etc.); verify and error when `compiler-dfbs=false`
- `ttl-insert-copy-wait` — insert missing `ttl.wait` after `ttl.copy` ops whose transfer handle has no wait user
- `ttl-insert-cb-sync` — insert DFB wait/pop/reserve/push around compute regions
- `ttl-annotate-l1-acc-loops` — detect `+=` accumulation loops and annotate for L1 packer accumulation
- `convert-ttl-to-compute` — lower TTL elementwise tensor ops to `ttl.compute` with tile ops
- `ttl-set-compute-kernel-config` — set `fp32_dest_acc_en` / `dst_full_sync_en` defaults
- `ttl-assign-dst` — DST register allocation (linear scan with copy insertion)
- `ttl-subblock-compute-for-dst` — tile `ttl.compute` into DST-sized subblocks *(only if `maximize-dst=true`)*; optionally refine reserve/push to per-subblock granularity *(only if `subblock-sync=true`)*
- `ttl-insert-tile-regs-sync` — insert math/pack thread synchronization
- `ttl-lower-to-loops` — lower `ttl.compute` to `scf.for` loops; matmul computes are expanded inline via `generateMatmulCompute`
- `ttl-schedule-operations` — reorder tile ops by dependency depth and kind *(only if `maximize-dst=true`)*
- `ttl-annotate-cb-associations` — annotate block args with DFB indices
- `convert-ttl-to-ttkernel` — lower TTL DMA ops to TTKernel
- `ttkernel-insert-inits` — insert hardware init ops before compute ops
- `ttkernel-insert-l1-accumulation` — insert `pack_reconfig_l1_acc` guards for `+=` and reduction loops
- `ttkernel-combine-pack-tiles` — combine consecutive `pack_tile` into `pack_tile_block` *(only if `combine-pack-tiles=true`)*
- Canonicalization and CSE cleanup
- `ttkernel-specialize-cores`, then `canonicalize`, `cse` — per-core clone and const-fold of coordinate branches; tags clones with `ttl.core_coord` *(only if `specialize-cores=true`)*
- *(if `lower-to-emitc=true`)* `lower-affine`, `convert-ttkernel-to-emitc`, `emitc-form-expressions`

### Individual Pass Options

Each pass can also be run standalone for testing. Only passes with configurable
options are listed; the remaining passes have no options.

#### `ttl-insert-intermediate-dfbs`

Insert compiler-allocated intermediate DFBs at fusion split points.

| Option | Type | Default | Description |
|---|---|---|---|
| `enable` | bool | `true` | Insert compiler-allocated DFBs. When false, emit an error if any operation requires one. |

```bash
ttlang-opt input.mlir -p 'func.func(ttl-insert-intermediate-dfbs{enable=false})'
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
