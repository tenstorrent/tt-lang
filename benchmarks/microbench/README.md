# Accumulation cost-model microbenchmarks

Handwritten Tenstorrent compute kernels that measure the cost of the operation
sequences tt-lang generates, to calibrate the accumulation cost model
(`AccumulationAnalysis`, accumulation branch). Unlike the tt-metal tt-llk perf
benchmarks — which time isolated LLK primitives on bare metal — these run through
real tt-metal dispatch (`ttnn.generic_op`) over real dataflow buffers, so the
DFB reserve/wait/push/pop and cross-thread sync are inside the measurement. The
kernels are matched by hand to what tt-lang emits: e.g. MB2's DST kernel uses
`binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>`, the op tt-lang's
`tile_accumulate_add` lowers to.

## Benchmarks

- **MB1 — pack/unpack probe** (`sweep.py`): zero-compute DFB→DFB round-trip; the
  per-tile pack/unpack + DFB-sync cost.
- **MB2 — accumulation** (`acc_sweep.py`): `acc += contribution`, DST-resident
  vs L1-pack, with `--source l1|dram` (contribution residency).
- **MB3 — matmul K-accumulation** (planned): DST-K vs L1-K.
- **MB4 — compute-op / math** (planned): per-op compute-engine tile costs.

## Running (hardware, via the bnorris-ird container)

The device profiler must be enabled at process start (the runners set the env
vars; pass them through Docker too):

    sudo docker exec -w /home/bnorris/tt/tt-lang-cursor <container> bash -c "source build-docker/env/activate && TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 python -m benchmarks.microbench.sweep --tiles 1,2,4,8,16 --iters 128"

    python -m benchmarks.microbench.acc_sweep --tiles 1,2,4 --iters 1,2,4,8,16 --source l1

Times are compute-thread (TRISC) microseconds: per-RISC `DeviceZoneScopedN`
cycles ÷ profiler `CHIP_FREQ[MHz]`. Correctness is PCC vs a torch reference.

## Layout

- `kernels/` — compute kernels (`passthrough_compute`, `acc_dst`, `acc_l1`) plus
  reader/writer data-movement kernels.
- `sweep.py`, `acc_sweep.py` — runners; `profiler.py` — per-RISC zone readback;
  `fit.py` — linear fit of the probe weights.
- `results/` — dated CSVs (tracked in git).
- `RESULTS.md` — working notes: methodology, per-benchmark pseudocode, and the
  current measurements.
