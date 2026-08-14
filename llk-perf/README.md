# llk-perf

The LLK perf benchmarks tt-lang's cost estimator is built from, scoped to the
configurations tt-lang can actually generate.

`scripts/gen_cost_table.py` turns the CSVs these produce into
`lib/OpCost/CostTableBlackhole.inc`, which `TTLangOpCost` serves to the rest of
the compiler. A measurement taken in a configuration tt-lang never emits cannot
answer a lookup, so the sweep is narrowed to what the compiler can reach rather
than to what the hardware can do — that is the whole reason these files live here
instead of being used as they ship.

The CSVs themselves are not in the tree: they are large, regenerable from here,
and only meaningful against the tt-metal revision that produced them, which the
generated table records. `test/ttlang/OpCost/coverage.mlir` pins what the data
covers, so a re-sweep that changes coverage shows up as a reviewable diff.

## Nothing upstream is modified

Every file here carries a tt-lang name and is an *addition* to the tt-llk
checkout, never a replacement. `run_sweep.sh` copies them in for the duration of
a run and removes them afterwards, so the submodule is untouched whether or not a
sweep is running, and it refuses to start if a name would collide with an
upstream file.

That matters because our sweeps are narrower than upstream's. A file that
replaced `perf_eltwise_unary_sfpu.py` would hand anyone running the tt-llk perf
suite tt-lang-scoped results — 40 operations instead of 97, `iterations=8`
instead of 32, no `Float16` — with nothing to indicate the sweep had changed.
Additions cannot do that.

The staging itself is unavoidable: the harness finds test modules through its own
conftest and kernel sources through `-I<tests dir>`, neither of which reaches
outside the checkout. Build artefacts go to `$RUNNER_TEMP` (default
`/tmp/ttlang-llk-build`) rather than the shared `/tmp/tt-llk-build`, so a
concurrent gather cannot have its cache wiped.

## Contents

| file | |
|---|---|
| `sources/ttlang_datacopy_perf.cpp` | ours; nothing upstream measures a datacopy as the entire math loop |
| `sources/ttlang_reduce_perf.cpp` | upstream's, with the addressing fix and a split init zone |
| `sources/ttlang_eltwise_unary_sfpu_perf.cpp` | upstream's, with a split init zone and an elidable SFPU call |
| `sources/ttlang_eltwise_binary_fpu_perf.cpp` | upstream's, with the addressing fix and a split init zone |
| `sources/ttlang_eltwise_binary_sfpu_perf.cpp` | as above |
| `python_tests/perf_ttlang_*.py` | seven sweeps, narrowed to what tt-lang generates |

Two of the copied sources carry only the addressing fix; if that lands upstream
they go away. The rest carry measurement structure upstream has no reason to
want: a split init zone, and an elidable SFPU call.

## Running

```bash
llk-perf/run_sweep.sh                          # everything except matmul (~500 variants)
llk-perf/run_sweep.sh perf_ttlang_math_matmul  # 12288 variants; 12 builds still fail
python3 scripts/gen_cost_table.py -o lib/OpCost/CostTableBlackhole.inc
```

Needs a device and the venv at `third-party/.../tt-llk/tests/venv-llk`.

## What was narrowed, and why

Each cut is a configuration no tt-lang kernel can produce, so measuring it can
only add rows the lookup will reject.

- **`Float16`** — ttnn has no `FLOAT16` dtype, and tt-lang maps the name
  `"float16"` onto `BFLOAT16` (`ttl/dtype_utils.py:188`, "hardware implements f16
  as bf16"). No tt-lang kernel presents a Float16 CB.
- **SFPU `iterations` 32 → 8** — the SFPU kernel's inner trip count. tt-metal
  compiles 8 at 87 of its 88 call sites and the ttkernel dialect documents the
  same default inline on `exp_tile`. Measuring 32 described a kernel that never
  exists; this was the single largest block of unusable rows.
- **TopK ops, and `stable_sort` with them** — the TTKernel ops exist but nothing
  in tt-lang constructs them, and they are the only carriers of that knob.
- **SFPU ops with no TTKernel op at all** — `GeluTanh`, `Lrelu`, `ReluMin`,
  `Erfinv`, `Heaviside`, `Softshrink`, `SfpuElwrsub`. An LLK operation with no
  TTKernel counterpart cannot appear in a tt-lang kernel, so its cost has nowhere
  to go. These are exactly the mathops `gen_cost_table.py` reports as unmapped.
- **`fast_mode`, and `approx_mode` off the ops that cannot express it** — only
  `exp_tile` and `pow_binary_tiles` carry an `approx` attribute in the dialect;
  no op carries a fast-mode one. Elsewhere both are metal's default, and sweeping
  them duplicated every row under a second unmatchable key.
- **Tiny-tile matmul → full 32x32** — the matmul test ran *only* tiny tiles, the
  full-tile block having been commented out upstream for CI disk space. A tt-lang
  tile is 32x32, so every row it produced was keyed `faces != 4` and unreachable.
- **Matmul `throttle`** — the word appears nowhere in the ttkernel dialect or in
  tt-lang's lowering, so level 0 is the only reachable one.
- **Matmul `dst_index`** — kept at 0. Upstream sweeps a second value as a
  correctness edge case; for perf it is the same work at a different register
  offset, and the consumer's key has no `dst_index` field, so the two collapsed
  onto one key and were averaged.

Two changes went the other way, because scoping to tt-lang also exposes what is
missing:

- **`Float32` added to the FPU sweep**, which did not have it, although tt-lang
  runs f32 eltwise (`test/python/simple_add_f32.py`). Every f32 add/sub/mul was
  falling back to a placeholder.
- **Four fields added to the matmul key** in `gen_cost_table.py` — `r_dimm`,
  `k_dimm` and the two transpose columns. All four are constant in the tiny-tile
  data, so their absence was invisible; the full-tile sweep varies every one, at
  which point unlike measurements would silently average onto one key.

## Known remaining gaps

- **`dst_sync`** — the eltwise sources pin `DstSync::SyncHalf`, but tt-lang
  exposes `dst_full_sync_en` and examples use both. Fixing this means editing the
  kernel sources, not the sweeps.
- **`copy_tile/math` under `unpack_to_dest=true`** — `MATH_ISOLATE` cannot run
  it: the isolate returns the unpack thread early and the datacopy then spins
  forever waiting on it. The zone is elided, and `_copy_tile_math` drops the lane
  rather than emitting the empty loop's timing as a measurement.
- **`unary_bcast` and `transpose_wh_tile`** — the same obstacle. A broadcast
  leaves the tile in SrcB, and the harness's fake handshake
  (`_perf_unpack_loop_set_valid`) was written for the SrcA path a plain datacopy
  uses, so `MATH_ISOLATE` hangs. `copy_tile` runs 42/42; the shapes are retained
  in `_ALL_OP_SHAPES` in `perf_ttlang_datacopy.py`, one line from re-enabling.
- **Asserts** — sweeps run with `TT_LLK_DISABLE_ASSERTS=1`, because `LLK_ASSERT`
  is on by default here and compiled out in production. It is not free: bf16
  datacopy measures 55.12 cycles/tile on unpack with asserts against 41.62
  without. Anything measured without that flag describes a kernel nobody runs.
- **`SfpuAddTopRow`** has no TTKernel op, but it owns a whole upstream test
  function rather than one list entry, so it is left in place.
- **Data movement** — the LLK suite builds only TRISC kernels, so nothing here
  can measure NCRISC or BRISC. That gap needs its own benchmark suite.

## Divergence from upstream

The sweeps are deliberately *not* upstreamable: the LLK team needs the wide
sweep for hardware coverage, we need the reachable subset. That divergence is
permanent, and `UPSTREAM.md` covers re-applying it after an uplift.

The two copied kernel sources are a different matter. Their only change is the
addressing fix — reading operands from the stimuli buffers rather than through
`PERF_ADDRESS`, whose 4096-byte tile stride is right only for Float32. Upstream
already migrated `eltwise_unary_sfpu_perf.cpp` this way and left these two
behind, so their own benchmarks disagree with each other by 0.776x on unpack for
the same work. That fix belongs upstream, and landing it deletes both files from
here.
