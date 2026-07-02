# passthrough subblock variants — measured summary

Kernel family: `subblock_pack_unpack_sweep.py` driving the passthrough compute
kernel. Zero-math copy; sweep the DST chunk per `tile_regs_acquire`, holding
total `tiles` fixed. bf16, single core; sync mode (DST capacity) noted per
section. `--runs 5`, `noc_active_in_zone = 0` and `pcc = 1.0` on every row below.

Variants (`--variant`):
- **self_cycle** (`passthrough_compute.cpp`): hop inside the loop, one buffer
  (`dfb_loop → dfb_loop`). Cross-iteration pack→unpack dependency.
- **hoisted** (`passthrough_hoisted.cpp`): one buffer, hop outside the loop.
- **twocb** (`passthrough_twocb.cpp`): hop outside, reads `dfb_loop` and writes a
  distinct `dfb_out` (two CBs, no self-cycle) -- the real-kernel topology.

Metric: `trisc_max` µs (bottleneck compute thread); **bold** = fastest chunk in
the row.

Reproduces RESULTS.md MB5.A to ~1% on absolute µs. Boundary best-chunk picks are
sub-1% ties (noise): re-running flips tiles=8 (half-sync chunk=1↔2, full-sync
chunk=2↔4) and the half-sync c4↔c8 crossover tile (~56–64) between runs.

---

## self_cycle — half-sync (DST cap 8), iters=1, tiles 8→176

Command (run from repo root in the active hardware env; `self_cycle` is the
default variant):

```bash
python -m benchmarks.microbench.mb5.subblock_pack_unpack_sweep \
  --tiles 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176 \
  --sub 1,2,4,8 --iters 1 --runs 5 \
  --csv benchmarks/microbench/results/pp_selfcycle_i1.csv
```

CSV: `results/pp_selfcycle_i1_blackhole_self_cycle_20260701T191606Z.csv` (88 rows)

| tiles | chunk=1 | chunk=2 | chunk=4 | chunk=8 | best | chunk=8/chunk=1 |
|------:|--------:|--------:|--------:|--------:|:----:|---:|
| 8 | 0.371 | **0.368** | 0.388 | 0.451 | chunk=2 | 1.21 |
| 16 | 0.659 | 0.659 | **0.644** | 0.698 | chunk=4 | 1.06 |
| 24 | 0.955 | 0.936 | **0.903** | 0.945 | chunk=4 | 0.99 |
| 32 | 1.249 | 1.216 | **1.154** | 1.189 | chunk=4 | 0.95 |
| 40 | 1.540 | 1.499 | **1.412** | 1.433 | chunk=4 | 0.93 |
| 48 | 1.831 | 1.771 | **1.668** | 1.682 | chunk=4 | 0.92 |
| 56 | 2.129 | 2.053 | **1.923** | 1.924 | chunk=4 | 0.90 |
| 64 | 2.421 | 2.325 | 2.183 | **2.169** | chunk=8 | 0.90 |
| 72 | 2.717 | 2.609 | 2.439 | **2.411** | chunk=8 | 0.89 |
| 80 | 3.008 | 2.885 | 2.695 | **2.686** | chunk=8 | 0.89 |
| 88 | 3.299 | 3.167 | 2.949 | **2.906** | chunk=8 | 0.88 |
| 96 | 3.596 | 3.443 | 3.206 | **3.153** | chunk=8 | 0.88 |
| 104 | 3.886 | 3.727 | 3.463 | **3.397** | chunk=8 | 0.87 |
| 112 | 4.181 | 4.001 | 3.722 | **3.641** | chunk=8 | 0.87 |
| 120 | 4.475 | 4.346 | 3.975 | **3.883** | chunk=8 | 0.87 |
| 128 | 4.764 | 4.556 | 4.233 | **4.129** | chunk=8 | 0.87 |
| 136 | 5.059 | 4.838 | 4.488 | **4.372** | chunk=8 | 0.86 |
| 144 | 5.352 | 5.115 | 4.746 | **4.624** | chunk=8 | 0.86 |
| 152 | 5.649 | 5.397 | 5.001 | **4.868** | chunk=8 | 0.86 |
| 160 | 5.941 | 5.670 | 5.256 | **5.196** | chunk=8 | 0.87 |
| 168 | 6.236 | 5.950 | 5.512 | **5.353** | chunk=8 | 0.86 |
| 176 | 6.529 | 6.330 | 5.769 | **5.601** | chunk=8 | 0.86 |

Measured observations (no interpretation):
- Best chunk shifts with size: **chunk=2 at tiles=8 → chunk=4 at tiles=16–56 →
  chunk=8 at tiles≥64**.
- Max-DST (chunk=8) is the *worst* chunk at tiles=8 (1.21× chunk=1) and becomes
  the *best* from tiles=64 up; chunk=8/chunk=1 settles at ~0.86 (chunk=8 ~14%
  faster than chunk=1) for large tiles.
- chunk=4 is fastest across tiles=16–56; the chunk=4↔chunk=8 crossover is a tie
  (~0.1%) around tiles=56–64. chunk=1 never wins.

---

## self_cycle — full-sync (DST cap 16), iters=1, tiles 8→176

Same as above but `--full-sync 1` (DST capacity 16, so the chunk sweep extends to
16 = the new max-DST).

```bash
python -m benchmarks.microbench.mb5.subblock_pack_unpack_sweep \
  --tiles 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176 \
  --sub 1,2,4,8,16 --iters 1 --runs 5 --full-sync 1 \
  --csv benchmarks/microbench/results/pp_selfcycle_i1_fs.csv
```

CSV: `results/pp_selfcycle_i1_fs_blackhole_self_cycle_20260701T191617Z.csv` (109 rows)

| tiles | chunk=1 | chunk=2 | chunk=4 | chunk=8 | chunk=16 | best |
|------:|--------:|--------:|--------:|--------:|---------:|:----:|
| 8 | 0.484 | 0.410 | **0.408** | 0.462 | — | chunk=4 |
| 16 | 0.883 | 0.741 | **0.707** | 0.752 | 0.825 | chunk=4 |
| 24 | 1.297 | 1.084 | **0.997** | 1.068 | 1.154 | chunk=4 |
| 32 | 1.716 | 1.421 | **1.289** | 1.384 | 1.518 | chunk=4 |
| 40 | 2.127 | 1.758 | **1.582** | 1.688 | 1.834 | chunk=4 |
| 48 | 2.544 | 2.097 | **1.877** | 1.992 | 2.217 | chunk=4 |
| 56 | 2.958 | 2.433 | **2.174** | 2.301 | 2.519 | chunk=4 |
| 64 | 3.375 | 2.773 | **2.468** | 2.616 | 2.908 | chunk=4 |
| 72 | 3.792 | 3.108 | **2.762** | 2.920 | 3.222 | chunk=4 |
| 80 | 4.203 | 3.450 | **3.057** | 3.243 | 3.593 | chunk=4 |
| 88 | 4.618 | 3.781 | **3.351** | 3.536 | 3.939 | chunk=4 |
| 96 | 5.031 | 4.123 | **3.644** | 3.848 | 4.285 | chunk=4 |
| 104 | 5.448 | 4.464 | **3.935** | 4.160 | 4.590 | chunk=4 |
| 112 | 5.859 | 4.801 | **4.240** | 4.469 | 4.975 | chunk=4 |
| 120 | 6.282 | 5.141 | **4.529** | 4.774 | 5.291 | chunk=4 |
| 128 | 6.698 | 5.472 | **4.827** | 5.087 | 5.667 | chunk=4 |
| 136 | 7.106 | 5.811 | **5.125** | 5.392 | 5.981 | chunk=4 |
| 144 | 7.521 | 6.156 | **5.412** | 5.707 | 6.361 | chunk=4 |
| 152 | 7.932 | 6.492 | **5.708** | 6.015 | 6.666 | chunk=4 |
| 160 | 8.349 | 6.829 | **6.008** | 6.320 | 7.047 | chunk=4 |
| 168 | 8.764 | 7.161 | **6.297** | 6.626 | 7.428 | chunk=4 |
| 176 | 9.182 | 7.500 | **6.597** | 6.937 | 7.739 | chunk=4 |

Measured observations (no interpretation):
- Best chunk is **chunk=4 at every tile size** (8→176). It does **not** climb to
  max-DST as tiles grow.
- Ordering in the mid/large range is consistently chunk=4 < chunk=8 < chunk=16:
  max-DST (chunk=16) is the *slowest* of {4,8,16} at every size, and even chunk=8
  never beats chunk=4. chunk=16/chunk=4 ≈ +17% throughout.
- Contrast with half-sync: there the optimum climbs chunk=2→4→8 and max-DST
  (chunk=8) wins at tiles≥64; under full-sync the optimum stays at chunk=4.

---

## hoisted — half-sync (DST cap 8), iters=1, tiles 8→176

Same kernel/topology as self_cycle (one buffer, `dfb_loop` front/back) but the CB
hop (`cb_wait_front`/`reserve`/`pop`/`push`) is hoisted **outside** the loop, so
the timed zone excludes it. At iters=1 the compute (one `copy_block` pass) is
identical to self_cycle; the only difference is that self_cycle times the hop and
hoisted does not.

```bash
python -m benchmarks.microbench.mb5.subblock_pack_unpack_sweep --variant hoisted \
  --tiles 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176 \
  --sub 1,2,4,8 --iters 1 --runs 5 \
  --csv benchmarks/microbench/results/pp_hoisted_i1.csv
```

CSV: `results/pp_hoisted_i1_blackhole_hoisted_20260701T192059Z.csv` (88 rows)

| tiles | chunk=1 | chunk=2 | chunk=4 | chunk=8 | best |
|------:|--------:|--------:|--------:|--------:|:----:|
| 8 | **0.367** | 0.370 | 0.381 | 0.451 | chunk=1 |
| 16 | 0.659 | 0.639 | **0.626** | 0.682 | chunk=4 |
| 24 | 0.956 | 0.898 | **0.872** | 0.921 | chunk=4 |
| 32 | 1.247 | 1.160 | **1.116** | 1.159 | chunk=4 |
| 40 | 1.537 | 1.423 | **1.358** | 1.395 | chunk=4 |
| 48 | 1.834 | 1.682 | **1.602** | 1.632 | chunk=4 |
| 56 | 2.130 | 1.943 | **1.843** | 1.868 | chunk=4 |
| 64 | 2.417 | 2.203 | **2.088** | 2.103 | chunk=4 |
| 72 | 2.711 | 2.463 | 2.344 | **2.337** | chunk=8 |
| 80 | 3.002 | 2.724 | **2.574** | 2.575 | chunk=4 |
| 88 | 3.301 | 2.987 | 2.823 | **2.814** | chunk=8 |
| 96 | 3.595 | 3.239 | 3.064 | **3.048** | chunk=8 |
| 104 | 3.887 | 3.503 | 3.313 | **3.290** | chunk=8 |
| 112 | 4.178 | 3.766 | 3.556 | **3.521** | chunk=8 |
| 120 | 4.469 | 4.026 | 3.801 | **3.760** | chunk=8 |
| 128 | 4.765 | 4.287 | 4.043 | **3.993** | chunk=8 |
| 136 | 5.060 | 4.548 | 4.290 | **4.228** | chunk=8 |
| 144 | 5.352 | 4.811 | 4.532 | **4.465** | chunk=8 |
| 152 | 5.640 | 5.073 | 4.779 | **4.707** | chunk=8 |
| 160 | 5.940 | 5.331 | 5.021 | **4.930** | chunk=8 |
| 168 | 6.233 | 5.592 | 5.266 | **5.176** | chunk=8 |
| 176 | 6.528 | 5.852 | 5.515 | **5.408** | chunk=8 |

Best chunk: chunk=1 at tiles=8 → chunk=4 at tiles=16–64 → chunk=8 at tiles≥72
(chunk=4↔chunk=8 tie around tiles=72–88). Same shape as self_cycle half-sync.

## hoisted — full-sync (DST cap 16), iters=1, tiles 8→176

```bash
python -m benchmarks.microbench.mb5.subblock_pack_unpack_sweep --variant hoisted \
  --tiles 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176 \
  --sub 1,2,4,8,16 --iters 1 --runs 5 --full-sync 1 \
  --csv benchmarks/microbench/results/pp_hoisted_i1_fs.csv
```

CSV: `results/pp_hoisted_i1_fs_blackhole_hoisted_20260701T192226Z.csv` (109 rows)

| tiles | chunk=1 | chunk=2 | chunk=4 | chunk=8 | chunk=16 | best |
|------:|--------:|--------:|--------:|--------:|---------:|:----:|
| 8 | 0.467 | **0.407** | 0.408 | 0.464 | — | chunk=2 |
| 16 | 0.883 | 0.735 | **0.704** | 0.743 | 0.832 | chunk=4 |
| 24 | 1.300 | 1.077 | **0.997** | 1.053 | 1.143 | chunk=4 |
| 32 | 1.716 | 1.412 | **1.293** | 1.362 | 1.497 | chunk=4 |
| 40 | 2.131 | 1.747 | **1.588** | 1.669 | 1.835 | chunk=4 |
| 48 | 2.545 | 2.086 | **1.879** | 1.977 | 2.187 | chunk=4 |
| 56 | 2.955 | 2.417 | **2.176** | 2.288 | 2.519 | chunk=4 |
| 64 | 3.379 | 2.755 | **2.472** | 2.599 | 2.870 | chunk=4 |
| 72 | 3.788 | 3.090 | **2.767** | 2.905 | 3.219 | chunk=4 |
| 80 | 4.206 | 3.428 | **3.057** | 3.209 | 3.540 | chunk=4 |
| 88 | 4.621 | 3.755 | **3.352** | 3.515 | 3.901 | chunk=4 |
| 96 | 5.036 | 4.097 | **3.643** | 3.820 | 4.236 | chunk=4 |
| 104 | 5.449 | 4.436 | **3.946** | 4.133 | 4.590 | chunk=4 |
| 112 | 5.867 | 4.768 | **4.241** | 4.441 | 4.966 | chunk=4 |
| 120 | 6.280 | 5.102 | **4.531** | 4.749 | 5.279 | chunk=4 |
| 128 | 6.694 | 5.434 | **4.827** | 5.054 | 5.609 | chunk=4 |
| 136 | 7.113 | 5.770 | **5.124** | 5.361 | 5.974 | chunk=4 |
| 144 | 7.525 | 6.108 | **5.413** | 5.668 | 6.287 | chunk=4 |
| 152 | 7.940 | 6.444 | **5.717** | 5.979 | 6.658 | chunk=4 |
| 160 | 8.356 | 6.781 | **6.001** | 6.282 | 6.968 | chunk=4 |
| 168 | 8.768 | 7.110 | **6.300** | 6.587 | 7.349 | chunk=4 |
| 176 | 9.187 | 7.445 | **6.599** | 6.896 | 7.654 | chunk=4 |

Best chunk: chunk=4 at every tile size (chunk=2 at tiles=8), same as self_cycle
full-sync; max-DST never wins.

## self_cycle − hoisted (measured in-zone hop cost), iters=1

Both runs are iters=1 with identical compute, so this difference is the CB hop
that self_cycle times inside the zone and hoisted times outside. Values are
differences of two separate runs, so magnitudes ≲0.05 µs are at/below run-to-run
noise. µs:

| tiles | HS c1 | HS c2 | HS c4 | HS c8 | FS c2 | FS c4 | FS c8 | FS c16 |
|------:|------:|------:|------:|------:|------:|------:|------:|-------:|
| 8   | 0.004 | -0.002 | 0.007 | -0.001 | 0.003 | 0.001 | -0.002 | — |
| 32  | 0.002 | 0.056 | 0.038 | 0.030 | 0.009 | -0.004 | 0.022 | 0.021 |
| 64  | 0.004 | 0.122 | 0.095 | 0.066 | 0.018 | -0.004 | 0.017 | 0.038 |
| 128 | -0.001 | 0.269 | 0.190 | 0.137 | 0.038 | -0.000 | 0.033 | 0.058 |
| 176 | 0.001 | 0.478 | 0.254 | 0.193 | 0.055 | -0.001 | 0.041 | 0.085 |

Measured observations (no interpretation):
- Half-sync: chunk=1 delta ≈ 0 at all tiles; chunk=2/4/8 deltas grow with tiles
  (largest for chunk=2: +0.48 µs at tiles=176; chunk=4 +0.25, chunk=8 +0.19).
- Full-sync: deltas are much smaller (≤~0.085 µs); chunk=4 delta ≈ 0 at all tiles.
- Because the deltas are small and structured (not a flat constant), the "measured
  sync cost" is not a single fixed value — it depends on chunk and tiles and sync
  mode. (Mechanism left open, per the no-interpretation rule.)

---

## twocb — hoisted, read dfb_loop → write dfb_out (two distinct CBs), iters=1

Same hoisted structure (hop outside, independent passes, untimed warmup) but the
loop writes a **separate** output CB instead of self-cycling one buffer -- the
topology a real tt-lang kernel uses (input CB → output CB). This is the only
change vs `hoisted`.

```bash
# half-sync
python -m benchmarks.microbench.mb5.subblock_pack_unpack_sweep --variant twocb \
  --tiles 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176 \
  --sub 1,2,4,8 --iters 1 --runs 5 \
  --csv benchmarks/microbench/results/pp_twocb_i1.csv
# full-sync: add --sub ...,16 --full-sync 1 --csv .../pp_twocb_i1_fs.csv
```

CSV (half): `results/pp_twocb_i1_blackhole_twocb_20260701T193821Z.csv` (88 rows)

| tiles | chunk=1 | chunk=2 | chunk=4 | chunk=8 | best |
|------:|--------:|--------:|--------:|--------:|:----:|
| 8 | **0.295** | 0.298 | 0.301 | 0.408 | chunk=1 |
| 16 | 0.588 | 0.580 | 0.534 | **0.517** | chunk=8 |
| 24 | 0.883 | 0.838 | 0.783 | **0.758** | chunk=8 |
| 32 | 1.176 | 1.098 | 1.023 | **0.993** | chunk=8 |
| 40 | 1.471 | 1.355 | 1.266 | **1.239** | chunk=8 |
| 48 | 1.766 | 1.613 | 1.511 | **1.471** | chunk=8 |
| 56 | 2.058 | 1.908 | 1.756 | **1.709** | chunk=8 |
| 64 | 2.350 | 2.138 | 2.003 | **1.944** | chunk=8 |
| 72 | 2.643 | 2.401 | 2.244 | **2.185** | chunk=8 |
| 80 | 2.940 | 2.693 | 2.488 | **2.418** | chunk=8 |
| 88 | 3.230 | 2.919 | 2.736 | **2.657** | chunk=8 |
| 96 | 3.525 | 3.226 | 2.975 | **2.887** | chunk=8 |
| 104 | 3.819 | 3.443 | 3.223 | **3.125** | chunk=8 |
| 112 | 4.111 | 3.701 | 3.470 | **3.365** | chunk=8 |
| 120 | 4.406 | 3.962 | 3.711 | **3.597** | chunk=8 |
| 128 | 4.696 | 4.222 | 3.959 | **3.832** | chunk=8 |
| 136 | 4.992 | 4.553 | 4.202 | **4.072** | chunk=8 |
| 144 | 5.286 | 4.741 | 4.451 | **4.305** | chunk=8 |
| 152 | 5.581 | 5.011 | 4.688 | **4.543** | chunk=8 |
| 160 | 5.873 | 5.262 | 4.933 | **4.779** | chunk=8 |
| 168 | 6.163 | 5.526 | 5.179 | **5.015** | chunk=8 |
| 176 | 6.457 | 5.791 | 5.423 | **5.253** | chunk=8 |

CSV (full): `results/pp_twocb_i1_fs_blackhole_twocb_20260701T193948Z.csv` (109 rows)

| tiles | chunk=1 | chunk=2 | chunk=4 | chunk=8 | chunk=16 | best |
|------:|--------:|--------:|--------:|--------:|---------:|:----:|
| 8 | 0.412 | **0.332** | 0.383 | 0.419 | — | chunk=2 |
| 16 | 0.829 | 0.663 | **0.586** | 0.610 | 0.685 | chunk=4 |
| 24 | 1.244 | 0.999 | **0.884** | 0.919 | 0.993 | chunk=4 |
| 32 | 1.659 | 1.334 | **1.179** | 1.225 | 1.362 | chunk=4 |
| 40 | 2.072 | 1.670 | **1.475** | 1.533 | 1.682 | chunk=4 |
| 48 | 2.489 | 2.008 | **1.767** | 1.842 | 2.046 | chunk=4 |
| 56 | 2.904 | 2.341 | **2.064** | 2.147 | 2.373 | chunk=4 |
| 64 | 3.320 | 2.677 | **2.359** | 2.455 | 2.726 | chunk=4 |
| 72 | 3.735 | 3.011 | **2.653** | 2.761 | 3.062 | chunk=4 |
| 80 | 4.148 | 3.349 | **2.949** | 3.073 | 3.411 | chunk=4 |
| 88 | 4.564 | 3.682 | **3.242** | 3.376 | 3.751 | chunk=4 |
| 96 | 4.975 | 4.019 | **3.536** | 3.685 | 4.093 | chunk=4 |
| 104 | 5.394 | 4.352 | **3.833** | 3.991 | 4.441 | chunk=4 |
| 112 | 5.807 | 4.687 | **4.129** | 4.301 | 4.778 | chunk=4 |
| 120 | 6.223 | 5.027 | **4.424** | 4.606 | 5.132 | chunk=4 |
| 128 | 6.637 | 5.358 | **4.717** | 4.915 | 5.461 | chunk=4 |
| 136 | 7.051 | 5.694 | **5.012** | 5.221 | 5.823 | chunk=4 |
| 144 | 7.466 | 6.030 | **5.308** | 5.529 | 6.147 | chunk=4 |
| 152 | 7.880 | 6.365 | **5.599** | 5.836 | 6.513 | chunk=4 |
| 160 | 8.295 | 6.700 | **5.899** | 6.144 | 6.829 | chunk=4 |
| 168 | 8.712 | 7.036 | **6.192** | 6.452 | 7.204 | chunk=4 |
| 176 | 9.125 | 7.371 | **6.487** | 6.759 | 7.512 | chunk=4 |

Measured observations (no interpretation):
- **half-sync: chunk=8 (max-DST) is best at every tiles≥16** (chunk=1 at tiles=8).
  No chunk=4 middle band -- max-DST wins immediately, unlike self_cycle/hoisted.
- **full-sync: chunk=4 is best at every tiles≥16** (chunk=2 at tiles=8) -- same as
  self_cycle/hoisted full-sync; max-DST (chunk=16) never wins.
- twocb absolute times are lower than self_cycle/hoisted at matched (tiles, chunk)
  (e.g. half-sync tiles=176 chunk=8: twocb 5.253 vs hoisted 5.408 vs self_cycle
  5.601).

### best chunk vs tiles across variants (measured)

Half-sync:

| tiles | self_cycle | hoisted | twocb |
|------:|:----------:|:-------:|:-----:|
| 8 | chunk=2 | chunk=1 | chunk=1 |
| 16–56 | chunk=4 | chunk=4 | **chunk=8** |
| 64 | chunk=8 | chunk=4 | chunk=8 |
| ≥72 | chunk=8 | chunk=8 | chunk=8 |

Full-sync: all three variants pin at **chunk=4** for every tiles≥16 (chunk=2/4 tie
at tiles=8); max-DST (chunk=16) never wins.

Measured takeaway: the same-buffer probes (self_cycle/hoisted) show a chunk=4
middle band under half-sync, but the two-CB topology (real-kernel shape) picks
max-DST (chunk=8) from tiles=16 up. Full-sync pins at chunk=4 regardless of
topology.

---

## compute_unary copy — two-CB baseline (alignment check for twocb)

Different kernel (`subblock_unary_op_sweep.py --op copy`, driving
`compute_unary.cpp`) but the same two-CB resident topology as `twocb`: read a
resident input CB, write a distinct output CB, hop outside, untimed warmup. This
is the independent cross-check that `twocb` reproduces the real-kernel baseline.

```bash
# half-sync
python -m benchmarks.microbench.mb5.subblock_unary_op_sweep --op copy \
  --tiles 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176 \
  --sub 1,2,4,8 --iters 1 --runs 5 \
  --csv benchmarks/microbench/results/unary_copy_i1.csv
# full-sync: add --sub ...,16 --full-sync 1 --csv .../unary_copy_i1_fs.csv
```

CSV (half): `results/unary_copy_i1_blackhole_bf16_copy_20260701T194958Z.csv` (88 rows)

| tiles | chunk=1 | chunk=2 | chunk=4 | chunk=8 | best |
|------:|--------:|--------:|--------:|--------:|:----:|
| 8 | 0.289 | 0.308 | **0.285** | 0.957 | chunk=4 |
| 16 | 0.589 | 0.595 | 0.547 | **0.521** | chunk=8 |
| 32 | 1.177 | 1.153 | 1.061 | **1.011** | chunk=8 |
| 64 | 2.351 | 2.264 | 2.084 | **2.001** | chunk=8 |
| 128 | 4.698 | 4.491 | 4.138 | **3.960** | chunk=8 |
| 176 | 6.455 | 6.163 | 5.676 | **5.433** | chunk=8 |

(chunk=8 best at every tiles≥16; full 22-row table in the CSV. tiles=8 chunk=8 is
a cold-start edge spike, 0.957.)

CSV (full): `results/unary_copy_i1_fs_blackhole_bf16_copy_20260701T195140Z.csv` (109 rows)

| tiles | chunk=1 | chunk=2 | chunk=4 | chunk=8 | chunk=16 | best |
|------:|--------:|--------:|--------:|--------:|---------:|:----:|
| 8 | 0.412 | **0.333** | 1.024 | 0.947 | — | chunk=2 |
| 16 | 0.830 | 0.668 | **0.588** | 0.613 | 0.687 | chunk=4 |
| 32 | 1.659 | 1.344 | **1.185** | 1.232 | 1.375 | chunk=4 |
| 64 | 3.318 | 2.697 | **2.359** | 2.467 | 2.757 | chunk=4 |
| 128 | 6.636 | 5.397 | **4.719** | 4.939 | 5.522 | chunk=4 |
| 176 | 9.126 | 7.425 | **6.488** | 6.790 | 7.595 | chunk=4 |

(chunk=4 best at every tiles≥16; full table in CSV.)

Alignment with `twocb` (best chunk, both sweeps):
- **Half-sync: identical at every tiles≥16 (both chunk=8).** Only tiles=8 differs
  (unary chunk=4 vs twocb chunk=1 — noise tie at the smallest size).
- **Full-sync: identical at every tile** (chunk=2 @ 8, chunk=4 ≥16).
- Absolute times within ~3% (unary slightly slower, +0.8% at tiles=16 growing to
  +3.4% at tiles=176) — a small systematic offset from minor structural
  differences (no-op `op_apply` branch, seed/warmup path, CB count), not a pattern
  difference.

Conclusion: `twocb` reproduces the compute_unary copy baseline. The real two-CB
topology picks **chunk=8 (max-DST) from tiles=16 up under half-sync** and
**chunk=4 under full-sync**, in both kernels.

---

## Full-sync optimum is chunk=6, not chunk=4 (non-power-of-2 check)

The full-sync `chunk=4` win was a power-of-2 sampling artifact. Sweeping
intermediate chunks (twocb, full-sync, iters=1) shows `trisc_max(chunk)` bottoms
at **chunk=6** at every tile (48/96/144/176), decreasing smoothly to 6 then rising
sharply at 7. E.g. tiles=176: c4 6.487, c5 6.335, **c6 6.254**, c7 6.638, c8 6.760
— chunk=6 beats chunk=4 by ~3.5% and chunk=8 by ~8% (pcc=1.0). So "full-sync pins
at chunk=4" (RESULTS.md MB5.A) is just the nearest power of two below the true
optimum (6); half-sync is unaffected (its chunk=8 = the cap).
`results/twocb_fs_np2*.csv`.

---

## compute_unary exp — math-bound op (subblock is a non-lever)

`subblock_unary_op_sweep.py --op exp`, iters=128, half-sync (DST cap 8),
`--runs 5`. The opposite end from zero-math copy: exp costs ~0.45 µs/tile (matches
MB4), which dwarfs the per-acquire overhead, so the chunk barely matters.

```bash
python -m benchmarks.microbench.mb5.subblock_unary_op_sweep --op exp \
  --tiles 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176 \
  --sub 1,2,4,8 --iters 128 --runs 5 \
  --csv benchmarks/microbench/results/unary_exp_i128.csv
```

CSV: `results/unary_exp_i128_blackhole_bf16_exp_20260701T223734Z.csv` (88 rows).
per-iter trisc_max (µs); bold = fastest chunk:

| tiles | chunk=1 | chunk=2 | chunk=4 | chunk=8 | best | spread |
|------:|--------:|--------:|--------:|--------:|:----:|-----:|
| 8 | 3.603 | 3.570 | **3.554** | 3.582 | chunk=4 | 1.4% |
| 16 | 7.206 | 7.141 | **7.108** | 7.159 | chunk=4 | 1.4% |
| 32 | 14.412 | 14.281 | **14.216** | 14.313 | chunk=4 | 1.4% |
| 64 | 28.824 | 28.563 | **28.433** | 28.644 | chunk=4 | 1.4% |
| 96 | 43.236 | 42.844 | **42.649** | 42.954 | chunk=4 | 1.4% |
| 128 | 57.647 | 57.126 | **56.865** | 57.269 | chunk=4 | 1.4% |
| 176 | 79.265 | 78.548 | **78.190** | 78.747 | chunk=4 | 1.4% |

(All 22 tiles identical in shape: best=chunk=4, spread=1.4%; full table in CSV.)

Measured observations (no interpretation):
- Spread is **1.4% at every tile** (8→176) — chunk is a non-lever for exp.
- **chunk=4 is marginally best at every tile** (beats chunk=8 by ~0.7%, chunk=1 by
  ~1.4%, perfectly consistently) — a real tiny effect, not noise. Notably **not
  max-DST (chunk=8)**, even in half-sync where copy goes to chunk=8.
- exp math ≈ 0.447 µs/tile at chunk=8, tiles=176 (matches MB4).
- iters=1 ≈ iters=128 (identical to ~3 decimals, no tiles=8 spike): for a
  math-bound op the per-iteration work (~3.6 µs at tiles=8 → ~79 µs at tiles=176)
  dwarfs the ~0.7 µs fixed startup, so iters=1 is already steady-state — unlike
  copy, where the tiny per-iter work let startup dominate iters=1.
  `results/unary_exp_i1_*.csv`.

Contrast (half-sync, large tiles): copy (data-movement bound) → chunk=8 (max-DST),
spread up to ~14%+; exp (math bound) → chunk=4, spread ~1.4%. Subblock choice
matters for data-movement-bound ops, is nearly irrelevant for math-bound ones.
