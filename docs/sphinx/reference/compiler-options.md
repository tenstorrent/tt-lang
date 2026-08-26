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
| `--ttl-fpu-binary-ops` / `--no-ttl-fpu-binary-ops` | enabled | Allow FPU strategy selection for binary add, subtract, and multiply when their operands permit it. Disabling selects SFPU. |
| `--ttl-block-matmul` / `--no-ttl-block-matmul` | enabled | Emit `matmul_block` (processes the full tile block atomically) instead of per-tile matmul loops. Disabling this option is not yet supported. |
| `--ttl-subblock-sync` / `--no-ttl-subblock-sync` | disabled | Refine DFB reserve/push to per-subblock granularity, enabling `pack_tile_block` for contiguous subblocks. When disabled, user-placed reserve/push is preserved as written. |
| `--ttl-combine-pack-tiles` / `--no-ttl-combine-pack-tiles` | enabled | Combine consecutive `pack_tile` ops on the same DFB with contiguous DST and DFB indices into a single `pack_tile_block` call. |
| `--ttl-reduce-full-fp32` / `--no-ttl-reduce-full-fp32` | enabled | Prefer full-fp32 accumulation for reduce operations when supported by the target and the complete kernel configuration. |
| `--ttl-matmul-full-fp32` / `--no-ttl-matmul-full-fp32` | enabled | Prefer full-fp32 accumulation for matmul operations when supported by the target and the complete kernel configuration. |
| `--ttl-strict-f32-acc` / `--no-ttl-strict-f32-acc` | disabled | Error at compile time if a `+=` accumulation loop's output block exceeds f32 DST capacity (4 tiles with double-buffering). When enabled, guarantees each accumulation step fits in a single DST section without subblocking. |
| `--ttl-compiler-dfbs` / `--no-ttl-compiler-dfbs` | enabled | Insert compiler-allocated intermediate DFBs when an operation requires DFB-attached inputs, fusion would read a source after its DFB is released, or a computed value is stored by operations in multiple MLIR basic blocks. When disabled, the compiler emits an error if materialization is required. |
| `--ttl-pipe-computed-addresses` / `--no-ttl-pipe-computed-addresses` | enabled | Use computed receiver DFB addresses for eligible PipeNet transfers. When disabled, transfers use receiver-published destination addresses; multicast still requires proven equal runtime receiver addresses. |
| `--ttl-pipe-capacity-sync` / `--no-ttl-pipe-capacity-sync` | enabled | Use capacity-counter synchronization when the receiver wait and pop execute on the receiver NOC thread and the computed-address transfer passes the DFB ownership and count proofs. When disabled, computed-address transfers use receiver-post synchronization. |
| `--ttl-pipe-global-semaphores-only` / `--no-ttl-pipe-global-semaphores-only` | disabled | Allocate all compiler-managed PipeNet synchronization counters in GlobalSemaphore storage, leaving local hardware semaphore ids available to the application. |
| `--ttl-pipe-batch-tiles N` | `0` (auto) | Limit the logical transfers in one PipeTransport group. `0` selects automatically and `1` disables grouping. |
| `--ttl-l1-budget N` | target-dependent | Override the per-core L1 budget used for target-aligned DFB allocation, PipeNet resources, synchronized-reset state, reconfiguration state, and final combined validation. |
| `--ttl-reuse-user-dfbs` / `--no-ttl-reuse-user-dfbs` | enabled | Reuse physical DFB indices when concurrent-kernel liveness proves that compatible logical DFB lifetimes do not overlap. Disabling compacts provisional user indices without introducing new user-DFB sharing. |
| `--ttl-dfb-exact-coloring-search-limit N` | `1000000` | Examine at most `N` states during deterministic exact DFB allocation when order-dependent first-fit prevents acceptance or exceeds the provisional threshold after a conservative PipeNet reservation. This bounds compile time; reaching the limit reports an inconclusive result only when authoritative acceptance requires the search result. |
| `--ttl-unsafe-assume-dfb-allocation-groups` / `--no-ttl-unsafe-assume-dfb-allocation-groups` | disabled | Trust explicit `allocation_group=` handoffs that the compiler cannot prove. Accepted groups emit warnings and `ttl.assumed_dfb_allocation_groups` metadata. Descriptor, storage, static configuration, capacity, and L1 checks remain enforced. |
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

These parameters are set on the `@ttl.operation` decorator (not via command-line
flags) and control the TTNN compute kernel hardware configuration:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fp32_dest_acc_en` | `bool` or `None` | `None` | Constrain the Wormhole B0/Blackhole DST register-file element width: `true` selects 32-bit elements and `false` selects 16-bit elements. When `None`, resolve the width from target capabilities and tile-operation requirements. |
| `dst_full_sync_en` | `bool` or `None` | `None` | Enable full DST synchronization (single-buffering mode). Doubles DST capacity (32-bit elements: 8, 16-bit elements: 16) at the cost of a full sync between math and pack threads. |
| `math_fidelity` | `str` or `None` | `None` | Set the compute math fidelity to `LoFi`, `HiFi2`, `HiFi3`, or `HiFi4`. When `None`, retain the TTNN default. |

```python
@ttl.operation(
    grid=(2, 2),
    fp32_dest_acc_en=True,
    dst_full_sync_en=False,
    math_fidelity="HiFi4",
)
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
| `TTL_RELAX_DFB_SPSC` | any value | (unset) | Skip compiler verification that DFB producers, consumers, and waits execute on corresponding dynamically active launch nodes. The program must enforce those ownership and synchronization contracts. Finalized DFB preconditions, PipeNet endpoint guards, transfer correspondence, and synchronization schedules remain enabled. |

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
| `enable-fpu-binary-ops` | bool | `true` | Allow FPU strategy selection for binary add/sub/mul. |
| `use-block-matmul` | bool | `true` | Lower matmul to block-level hardware calls (`matmul_block`). |
| `subblock-sync` | bool | `false` | Refine DFB reserve/push to per-subblock granularity. |
| `combine-pack-tiles` | bool | `true` | Combine consecutive `pack_tile` ops into `pack_tile_block`. |
| `reduce-full-fp32` | bool | `true` | Prefer full-fp32 reduce accumulation when supported. |
| `matmul-full-fp32` | bool | `true` | Prefer full-fp32 matmul accumulation when supported. |
| `strict-f32-acc` | bool | `false` | Error if a `+=` accumulation loop's output block exceeds f32 DST capacity. |
| `compiler-dfbs` | bool | `true` | Insert compiler-allocated intermediate DFBs for DFB-only operands, source-lifetime preservation, and computed values stored by operations in multiple MLIR basic blocks. Error if disabled and any operation requires one. |
| `pipe-computed-addresses` | bool | `true` | Use computed receiver DFB addresses for eligible PipeNet transfers. When disabled, transfers use receiver-published destination addresses; multicast still requires proven equal runtime receiver addresses. |
| `pipe-capacity-sync` | bool | `true` | Use capacity-counter synchronization when the receiver wait and pop execute on the receiver NOC thread and the computed-address transfer passes the DFB ownership and count proofs. When disabled, computed-address transfers use receiver-post synchronization. |
| `pipe-global-semaphores-only` | bool | `false` | Allocate all compiler-managed PipeNet synchronization counters in GlobalSemaphore storage, leaving local hardware semaphore ids available to the application. |
| `pipe-batch-tiles` | int64_t | `0` (auto) | Limit logical transfers per PipeTransport group. `0` selects automatically and `1` disables grouping. |
| `l1-budget-override` | uint32_t | `0` (target default) | Override the per-core L1 budget used for target-aligned DFB allocation, PipeNet resources, synchronized-reset state, reconfiguration state, and final combined validation. |
| `reuse-user-dfbs` | bool | `true` | Reuse physical DFB indices for compatible logical DFBs with proven non-overlapping concurrent lifetimes. |
| `unsafe-assume-allocation-groups` | bool | `false` | Trust explicit DFB allocation-group handoffs that lack a complete compiler proof. Automatic reuse remains proof-based. |
| `exact-coloring-search-limit` | uint64 | `1000000` | Maximum states examined during deterministic exact DFB allocation before reporting an inconclusive result. |
| `specialize-cores` | bool | `false` | Run the `ttkernel-specialize-and-annotate-dfb-use` sub-pipeline. Maps from `--ttl-specialize-cores`. |
| `lower-to-emitc` | bool | `false` | Run the TTKernel-to-EmitC backend (produces C++ source). |

The pipeline runs these passes and subpasses in order:

- `ttl-materialize-loop-state` -- replace ranked-tensor loop-carried values with compiler-created DFBs
- `ttl-insert-copy-wait` -- insert missing `ttl.wait` after `ttl.copy` ops whose transfer handle has no wait user
- `ttl-annotate-l1-acc-loops` -- detect `+=` accumulation loops and annotate for L1 packer accumulation
- `ttl-create-producer-compute` -- create producer `ttl.compute` operations before intermediate materialization
- `ttl-insert-intermediate-dfbs` -- materialize DFB-only operands, values that must be preserved before source release, and computed values stored by operations in multiple MLIR basic blocks; verify and error when `compiler-dfbs=false`
- `convert-ttl-to-compute` -- lower TTL elementwise tensor ops to `ttl.compute` with tile ops
- `ttl-insert-cb-sync` -- insert missing DFB synchronization
- `ttl-verify-pipenet-guards`, then `ttl-verify-pipenet-schedule` -- verify PipeNet launch domains and event ordering while logical DFB identities remain distinct and before physical DFB allocation
- `ttl-form-pipe-transports` -- group eligible repeated PipeNet transfers and select bounded receiver storage while accounting for synchronized-reset and reconfiguration state
- `ttl-coalesce-dfb-acquires` -- coalesce compatible DFB acquires
- `ttl-finalize-dfb-indices` -- assign logical DFBs to physical indices, validate combined DFB and fixed-state capacity, and emit runtime metadata; `reuse-user-dfbs` controls automatic user-DFB reuse, `unsafe-assume-allocation-groups` trusts only explicit unproved group handoffs, `exact-coloring-search-limit` bounds exhaustive index and weighted-allocation queries, and `l1-budget-override` replaces the target L1 budget
- `ttl-set-compute-kernel-config` -- select tile execution strategies and resolve kernel-wide DST and per-DFB unpack configuration
- `ttl-assign-dst` -- DST register allocation (linear scan with copy insertion)
- `ttl-subblock-compute-for-dst` -- tile `ttl.compute` into DST-sized subblocks *(only if `maximize-dst=true`)*; optionally refine reserve/push to per-subblock granularity *(only if `subblock-sync=true`)*
- `ttl-lower-to-loops` -- lower `ttl.compute` to `scf.for` loops; matmul computes are expanded inline via `generateMatmulCompute`
- `ttl-schedule-operations` -- reorder tile ops by dependency depth and kind *(only if `maximize-dst=true`)*
- `ttl-annotate-cb-associations` -- annotate block args with DFB indices
- `ttl-verify-dfb-spsc` -- verify per-node DFB producer/consumer uniqueness after finalization
- `ttl-erase-pipenet-scopes` -- remove verified PipeNet structural markers
- `ttl-validate-cb-budget` -- verify target-aligned finalized DFB storage, synchronized-reset scratch, and reconfiguration tensors fit the per-core L1 budget
- `convert-ttl-to-ttkernel` -- lower TTL DMA, PipeNet, synchronized-reset, and DFB reconfiguration operations to TTKernel, select their runtime resources, and validate the exact combined per-core L1 allocation
- `ttkernel-insert-inits` -- insert hardware init ops before compute ops
- `ttkernel-insert-l1-accumulation` -- insert `pack_reconfig_l1_acc` guards for `+=` and reduction loops
- `ttkernel-combine-pack-tiles` -- combine consecutive `pack_tile` into `pack_tile_block` *(only if `combine-pack-tiles=true`)*
- Canonicalization and CSE cleanup
- `ttkernel-specialize-and-annotate-dfb-use` -- `ttkernel-specialize-cores`, `canonicalize`, `cse`, then `ttkernel-annotate-dfb-use` *(only if `specialize-cores=true`)*
- *(if `lower-to-emitc=true`)* `lower-affine`, `convert-ttkernel-to-emitc`, `emitc-form-expressions`

### Individual Pass Options

Each pass can also be run standalone for testing. Only passes with configurable
options are listed; the remaining passes have no options.

#### `ttl-insert-intermediate-dfbs`

Insert compiler-allocated intermediate DFBs where tensor SSA values require
concrete DFB storage.

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
| `reuse-user-dfbs` | bool | `true` | Reuse a physical index when concurrent-kernel liveness proves that two compatible logical DFB lifetimes cannot overlap. When false, compact provisional user indices without introducing new user-DFB sharing and apply the same lifetime proof only to compiler-created DFBs. |
| `exact-coloring-search-limit` | uint64 | `1000000` | Examine at most this many states during deterministic exact DFB allocation. Exhaustive search runs when order-dependent first-fit prevents acceptance by the index or weighted L1 limit, or exceeds the provisional threshold after a conservative PipeNet reservation. Reaching the limit fails with an inconclusive-search diagnostic only when acceptance requires the result; a reservation-only search may retain an authoritative-budget-valid assignment. |
| `l1-budget-override` | uint32_t | `0` (target default) | Override the per-core L1 budget used by target-aligned DFB allocation, synchronized-reset and reconfiguration state, and the conservative PipeNet reservation. |
| `unsafe-assume-allocation-groups` | bool | `false` | Trust explicit DFB allocation groups when launch-domain, access-completion, pointer-handoff, or lifetime-order proof is incomplete. Emit one warning per accepted group and record the assumptions in `ttl.assumed_dfb_allocation_groups`. Page-format, storage, static compute-configuration, per-member ring-envelope, target-capacity, and L1-budget errors remain fatal. |

```bash
ttlang-opt input.mlir -p 'builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=false exact-coloring-search-limit=1000000 l1-budget-override=0})'
```

#### `ttl-validate-cb-budget`

Validate the target-aligned allocation for finalized physical DFBs,
allocator-rounded synchronized-reset state, and one configuration tensor per
synchronized reconfiguration boundary. Tensor-backed DFB storage is excluded
because the tensor allocator owns it. Exact PipeNet scratch and GlobalSemaphore
allocations are added during `convert-ttl-to-ttkernel`.

| Option | Type | Default | Description |
|---|---|---|---|
| `l1-budget-override` | uint32_t | `0` (target default) | Override the per-core L1 budget used for finalized DFB, synchronized-reset, and reconfiguration-state validation. |

```bash
ttlang-opt input.mlir -p 'builtin.module(ttl-validate-cb-budget{l1-budget-override=98304})'
```

#### `ttl-set-compute-kernel-config`

Resolve tile execution strategies and shared compute-kernel configuration. See
[Compute Kernel Configuration](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/ComputeKernelConfiguration.md)
for the algorithm and invariants.

| Option | Type | Default | Description |
|---|---|---|---|
| `fp32-dest-acc-en` | string | `auto` | Select 32-bit destination elements through the Wormhole B0/Blackhole `fp32_dest_acc_en` setting: `auto`, `enabled`, or `disabled`. |
| `dst-full-sync-en` | string | `auto` | Select full DST synchronization: `auto`, `enabled`, or `disabled`. |
| `reduce-full-fp32` | bool | `true` | Prefer full-fp32 reduce accumulation when supported. |
| `matmul-full-fp32` | bool | `true` | Prefer full-fp32 matmul accumulation when supported. |
| `enable-fpu-binary-ops` | bool | `true` | Allow eligible add/sub/mul operations to select FPU. |

```bash
ttlang-opt input.mlir -p 'ttl-set-compute-kernel-config{fp32-dest-acc-en=enabled}'
```

#### `ttl-assign-dst`

DST register allocator using linear scan allocation with in-place operation
merging.

| Option | Type | Default | Description |
|---|---|---|---|
| `dst-capacity` | uint32_t | `0` (auto) | Override DST register capacity. Auto-computed from `fp32_dest_acc_en` and `dst_full_sync_en` by default. Single-buffering (`dst_full_sync_en=true`): 32-bit elements=8, 16-bit elements=16. Double-buffering (default): 32-bit elements=4, 16-bit elements=8. |
| `separate-output-region` | bool | `false` | Allocate outputs in a separate DST region (needed for reductions and some loop optimizations). |

```bash
ttlang-opt input.mlir -p 'func.func(ttl-assign-dst{dst-capacity=16})'
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
Selection uses a conservative upper bound for target-aligned DFB allocation,
receiver-published addresses, transport scratch, GlobalSemaphore counters,
record-selected callback resources, synchronized-reset state, and
reconfiguration state. A group size of one validates and records the
reservation without grouping. Exact combined validation occurs after PipeNet
planning in `convert-ttl-to-ttkernel`.

| Option | Type | Default | Description |
|---|---|---|---|
| `group-size` | int64_t | `0` (auto) | Limit logical transfers per group. `0` selects automatically and `1` disables grouping. |
| `l1-budget-override` | uint32_t | `0` (target default) | Override the combined per-core L1 budget used during conservative grouping selection. |

```bash
ttlang-opt input.mlir --ttl-form-pipe-transports='group-size=8'
```

#### `convert-ttl-to-ttkernel`

Lower TTL data movement, PipeNet, synchronized-reset, and DFB reconfiguration
operations to TTKernel.

| Option | Type | Default | Description |
|---|---|---|---|
| `reduce-full-fp32` | bool | `true` | Enable FP32 accumulation for reduce operations. |
| `pipe-computed-addresses` | bool | `true` | Use computed receiver DFB addresses for eligible PipeNet transfers. When false, transfers use receiver-published destination addresses; multicast still requires proven equal runtime receiver addresses. |
| `pipe-capacity-sync` | bool | `true` | Use capacity-counter synchronization when the receiver wait and pop execute on the receiver NOC thread and the computed-address transfer passes the DFB ownership and count proofs. When false, computed-address transfers use receiver-post synchronization. |
| `pipe-global-semaphores-only` | bool | `false` | Allocate all compiler-managed PipeNet synchronization counters in GlobalSemaphore storage. |
| `l1-budget-override` | uint32_t | `0` (target default) | Override the exact combined per-core budget for target-aligned finalized DFBs, synchronized-reset state, reconfiguration tensors, PipeNet scratch, and GlobalSemaphore allocations. |

```bash
ttlang-opt input.mlir -p 'builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false pipe-global-semaphores-only=true l1-budget-override=98304})'
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
`ttkernel-annotate-dfb-use` then records surviving DFB compile-time
argument indices on each specialized function. Debug prints of a DFB
remain only on cores that still have a non-print use of that DFB; a
print whose DFB was folded away is dropped rather than keeping the
descriptor alive for debugging.

This pass is off by default. Enable it through the pipeline option
`specialize-cores` (Python: `--ttl-specialize-cores`), which runs the
registered `ttkernel-specialize-and-annotate-dfb-use` sub-pipeline:

```bash
ttlang-opt input.mlir -p 'ttl-to-ttkernel-pipeline{specialize-cores=true lower-to-emitc=true}'
# Or stand-alone:
ttlang-opt input.mlir -p 'builtin.module(ttkernel-specialize-and-annotate-dfb-use)'
```
