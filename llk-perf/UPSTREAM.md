# Re-syncing after a tt-metal uplift

An uplift cannot touch our sweeps: every file here has a tt-lang name and is an
addition to the checkout, staged for the duration of a run and removed
afterwards. What an uplift *can* do is move the upstream file a copied source was
forked from, or the upstream test a narrowed sweep was derived from, leaving ours
silently stale.

So the work after an uplift is not restoring anything -- it is re-deriving.
Re-apply our narrowing onto the *new* upstream file rather than keeping the old
one, or upstream improvements are silently discarded. At `ea042c4ad` upstream had
rewritten
`perf_eltwise_unary_sfpu.py` to sweep the whole SFPU registry with a per-op
format matrix — worth keeping, and a straight restore would have thrown it away.

```bash
L=third-party/tt-metal/tt_metal/tt-llk/tests/python_tests
# our sweeps against the upstream tests they were derived from
diff -u llk-perf/python_tests/perf_ttlang_math_matmul.py $L/perf_math_matmul.py
# our copied sources against the upstream files they were forked from
diff -u llk-perf/sources/ttlang_eltwise_binary_fpu_perf.cpp \
        ${L%/python_tests}/sources/eltwise_binary_fpu_perf.cpp
```

## The edits to re-apply

Each is a narrowing to what tt-lang can generate; `README.md` carries the
evidence for every one.

| file | edits |
|---|---|
| `perf_ttlang_eltwise_binary_fpu.py` | drop `Float16`, add `Float32`; `tile_count=[16, 64]` |
| `perf_ttlang_eltwise_unary_sfpu.py` | `PERF_SWEEP_OPS` filtered by `_TTLANG_REACHABLE_OPS`; drop `Float16` from `_FULL_FORMATS`; `iterations=[8]`; pin `fast_mode`/`stable_sort`; gate `approx_mode` to `Exp` |
| `perf_ttlang_eltwise_binary_sfpu.py` | drop `Float16`; `iterations=[8]`; drop `SfpuElwrsub`; gate `approx_mode` to `SfpuElwpow` |
| `perf_ttlang_math_matmul.py` | drop `Float16`; full-tile combinations rather than tiny tiles; `dst_index == 0` only |

## Things upstream may fix for us

Check these before re-applying — an edit that upstream has adopted should be
dropped, not carried.

- **`compile_time_formats=True` in `perf_eltwise_binary_sfpu.py`.** Already done
  upstream at `ea042c4ad`. Before that the kernel did not compile at all:
  `eltwise_binary_sfpu_perf.cpp` passes `formats.math` as a template argument,
  which is not a constant expression under runtime formats. We had derived the
  same fix independently; it is no longer ours to carry.
- **The full-tile matmul block.** Commented out upstream for CI disk space. If it
  returns, our `ALL_TEST_PARAMS` edit becomes redundant.
- **`_TTLANG_REACHABLE_OPS`.** Derived by matching each `MathOperation` name
  against the mnemonics in `TTKernelOps.td`. It needs regenerating when the
  dialect gains SFPU ops; the assert in the test catches only the reverse, a
  registry rename.

## The environment matters as much as the sources

Measurements are only comparable within one build. Running the *same* benchmark
against two tt-llk revisions gave 212 vs 453 cycles on the FPU unpack init and
257 vs 470 on the SFPU one — a near-constant per-lane offset, deterministic to
within 4 cycles inside each build, so it is a code difference and not noise.
Tile-loop rates moved far less (42.7 vs 42.6 on unpack) but not by nothing
(+11.8 on math).

So a CSV is tied to the revision that produced it, and `perf_data/` must not mix
revisions. After an uplift, re-run every benchmark rather than refreshing some.
The generator has no field recording provenance, which is the gap that lets such
a mixture pass unnoticed.
