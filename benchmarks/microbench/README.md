# Accumulation cost-model microbenchmarks

These microbenchmarks measure composed kernel costs used by the accumulation
cost model. They run handwritten tt-metal C++ kernels through `ttnn.generic_op`
on real dataflow buffers, so the measured time includes DFB
reserve/wait/push/pop and compute-thread synchronization.

The tt-metal LLK perf suite remains the source for isolated primitive costs.
These benchmarks cover behavior that isolated primitive tests do not measure:
operation overlap across RISCs, DFB handoff cost, and strategy comparisons such
as DST-resident accumulation versus L1-pack accumulation.

The kernels mirror tt-lang lowering sequences by construction. For example, MB2
uses `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>`, matching
`tile_accumulate_add` lowering. The tt-lang compiler is not involved; tt-metal
JIT-compiles the handwritten kernels at run time.

## Benchmarks

- MB1 pack/unpack probe (`sweep.py`): zero-compute DFB->DFB round trip used to
  fit fixed plus per-tile DFB handoff cost.
- MB2 accumulation (`acc_sweep.py`): `initial + sum(contributions)`, comparing
  DST-resident accumulation against L1-pack accumulation. `--source l1|dram`
  selects contribution residency; `--expr add|mul|gelu` selects the
  per-iteration contribution expression (`mul` is L1-pack only).
- MB3 matmul K-accumulation (`matmul_sweep.py`): `C[mt,nt] = sum_k A[k] @ B[k]`,
  comparing DST-K against L1-K. The output is subblocked as the compiler would
  (`harness.dst_subblock`), covering MB3.A (output fits DST, reuse=1) and MB3.B
  (output exceeds DST, reuse>1).
- MB4 compute-op (`compute_sweep.py`): per-op SFPU math-engine tile cost
  (copy/exp/gelu/recip/sqrt/rsqrt) on the math thread (what tt-lang emits).
  A pack-thread activation arm and reduce/binary ops are planned.

## Requirements

- Hardware environment with `ttnn` available.
- A TT device visible to the process.
- Device profiler output available through `TT_METAL_HOME` or
  `TTLANG_PROFILE_CSV`.

The runners set `TT_METAL_DEVICE_PROFILER=1` and
`TT_METAL_PROFILER_MID_RUN_DUMP=1` before importing `ttnn`. When running through
Docker, pass those variables explicitly if the environment does not preserve
them.

## Running

Run from the repository root in the active hardware environment:

```bash
python -m benchmarks.microbench.sweep \
  --tiles 1,2,4,8,16 \
  --iters 128

python -m benchmarks.microbench.acc_sweep \
  --acc-tiles 1,2,4 \
  --iters 1,2,4,8,16 \
  --source l1 \
  --expr add

python -m benchmarks.microbench.matmul_sweep \
  --mt 1,2,4 \
  --nt 1,2 \
  --kt 1,2,4,8,16
```

Common runner options:

- `--device-id`: TT device id. Default is `0`.
- `--csv`: base CSV filename. The runner adds architecture, benchmark tags, and
  a UTC timestamp.
- `--no-csv`: print rows without writing CSV output.
- `--compile-only`: return before device execution.

## Output

Times are compute-thread microseconds from per-RISC `DeviceZoneScopedN` ranges,
converted with profiler `CHIP_FREQ[MHz]`. Correctness is reported as PCC against
a torch reference.

CSV files are written under `benchmarks/microbench/results/` by default and are
ignored by git. `fit.py` consumes MB1 CSVs and fits:

```text
us_per_iter(tiles) = fixed_us + per_tile_us * tiles
```

Example:

```bash
python -m benchmarks.microbench.fit "benchmarks/microbench/results/pack_unpack_*.csv"
```

## Layout

- `kernels/`: handwritten compute, reader, and writer kernels.
- `harness.py`: `ttnn.generic_op` dispatch, DFB descriptors, compute config, and
  CSV writing.
- `runner.py`: declarative benchmark runner shared by MB1-MB4.
- `profiler.py`: device-profiler CSV parsing and per-RISC zone summaries.
- `sweep.py`: MB1 pack/unpack probe.
- `acc_sweep.py`: MB2 accumulation strategy comparison.
- `matmul_sweep.py`: MB3 matmul K-accumulation strategy comparison.
- `compute_sweep.py`: MB4 compute-op (SFPU math) probe.
- `fit.py`: MB1 fixed plus per-tile regression.
- `RESULTS.md`: measurement notes and current hardware results.
