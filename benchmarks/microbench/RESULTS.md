# Microbench results — working notes

A record of what the microbenchmarks measure on hardware. Measurements only;
interpretation against any cost model lives elsewhere.

## Motivation — why not just use the LLK perf microbenchmarks?

The tt-metal tt-llk perf suite already measures per-engine tile costs (unpack,
pack, matmul/eltwise math, L1-accumulation pack surcharge) in isolation, and those
supply most of the cost-model weights directly. These benchmarks exist for what the LLK
microbenchmarks cannot give:

- **Composition.** The generated kernel runs unpack/math/pack pipelined across
  three RISCs; the LLK microbenchmarks time each engine alone, so adding them up
  does not reproduce the composed cost. Blackhole bf16, per tile:
  - LLK unpack ≈ 0.030 + LLK pack ≈ 0.026 = **0.056 µs** (serial sum)
  - slowest single engine = **0.030 µs** (perfect-overlap lower bound)
  - MB1 measured, real DFBs = **0.039 µs**

  The measured value sits between the two — the kernel pipelines unpack and pack
  across RISCs, recovering ~30% the serial sum can't see. So summing overshoots by
  ~1.4×, with no constant overlap factor across configs (it varies by op and tile
  count).
- **Dataflow-buffer handoff.** The LLK harness has no dataflow buffers, so it never
  measures the DFB reserve/wait/push/pop + cross-thread sync the real kernels pay
  (MB1's fixed term, ~0.09 µs/iter on Blackhole) — the cost model's
  `dfbHopFixedCost`.
- **The strategy decision, not just its inputs.** The model chooses DST-resident vs
  L1-pack. The LLK microbenchmarks can only feed the model's serial formula; the benchmarks
  measure the actual ranking on the generated sequence. MB2 found L1-pack marginally
  faster than DST for additive accumulation, where the serial model predicts DST
  is always lower-cost — a calibration error not observable from the LLK
  microbenchmarks alone.

The LLK microbenchmarks parameterize the model; these benchmarks check whether it is
calibrated.

## Methodology (brief)

- Handwritten C++ kernels (compute + reader/writer), modeled on the tt-metal
  tt-llk perf benchmarks, but run through **real tt-metal dispatch**
  (`ttnn.generic_op`) over real dataflow buffers — so DFB reserve/wait/push/pop +
  cross-thread sync are part of the measurement (the bare-metal LLK harness omits
  them).
- **No tt-lang compiler involved:** kernels are handwritten and JIT-compiled by
  tt-metal at run time; the sequences are matched by hand to what tt-lang emits.
- Single compute core; inputs L1-resident where possible.
- Timing: `DeviceZoneScopedN` per RISC; cycles ÷ profiler `CHIP_FREQ[MHz]` = µs.
- Correctness: PCC vs a torch reference.

Times are compute-thread (TRISC) µs. Probe runs have NCRISC/BRISC idle in the
measured zone; PCC ≈ 1.0 unless noted.

## Hardware / environment

| arch | host | freq (profiler) | notes |
|---|---|---|---|
| Blackhole | container `bnorris-ird-v1.1.3` | 1350 MHz | `source build-docker/env/activate` |
| Wormhole b0 | `aus-wh-01:49551`, `/localdev/bnorris/tt-lang` | 1000 MHz | env via `/localdev/bnorris/wh_env.sh`; toolchain built into `/opt/ttlang-toolchain`; `ttnn` works. tt-lang compiler build failed (`getTombstoneKey`) — not needed (benchmark uses `ttnn`/`generic_op`). |

Run BH: `sudo docker exec -w /home/bnorris/tt/tt-lang-cursor bnorris-ird-v1.1.3 bash -c "source build-docker/env/activate && TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 python -m benchmarks.microbench.sweep ..."`
Run WH: `ssh -p 49551 aus-wh-01 'cd /localdev/bnorris/tt-lang && source /localdev/bnorris/wh_env.sh && python -m benchmarks.microbench.sweep ...'`

## MB1 — pack/unpack probe

Summary: one compute core repeatedly packs a block of `tiles` tiles to L1 and unpacks
it back, with nothing else, to isolate the per-tile pack/unpack + DFB-sync cost.

```
reader: seed T tiles into private DFB once          # outside zone, then idle
compute:
  zone "pack_unpack_loop":
    for _ in range(ITERS):                          # ITERS >= 128 (steady state)
      cb_wait_front(cb, T)                          # unpack thread (TRISC0)
      cb_reserve_back(cb, T)                        # pack thread  (TRISC2)
      for sub in chunks(T, dst_capacity):
        acquire_dst
        for t in sub: copy_tile(cb -> DST[t])       # unpack L1->DST
        commit; wait
        for t in sub: pack_tile(DST[t] -> cb)       # pack  DST->L1
        release_dst
      cb_pop_front(cb, T); cb_push_back(cb, T)       # rotate block in place
  # per-iter time = fixed (DFB reserve/wait/push/pop + sync) + per_tile * T
```

### bf16 trisc_max µs/iter (raw)

| tiles | Blackhole | Wormhole |
|---|---|---|
| 1 | 0.107 | 0.182 |
| 2 | 0.161 | 0.272 |
| 4 | 0.252 | 0.439 |
| 8 | 0.441 | 0.768 |
| 16 | 0.693 | 1.160 |

### bf16 linear fit (`us_per_iter = fixed + per_tile·tiles`, trisc_max)

| arch | fixed µs | per_tile µs | r² |
|---|---|---|---|
| Blackhole | 0.089 | 0.039 | 0.99 |
| Wormhole | 0.160 | 0.065 | 0.98 |

- Round-trip / pipelined-throughput basis: per-RISC unpack≈math≈pack because the
  zone spans the same pipelined window, so this gives the *combined* per-tile
  cost, not the separate unpack/pack engine costs (those need the per-engine LLK microbenchmarks).

### Blackhole config matrix — trisc_max µs/iter

| dtype | full_sync | fp32_acc | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|---|---|
| bf16 | 0 | 0 | 0.107 | 0.161 | 0.252 | 0.441 |
| bf16 | 1 | 0 | 0.106 | 0.159 | 0.251 | 0.439 |
| bf16 | 0 | 1 | 0.113 | 0.164 | 0.255 | 0.388 |
| fp32 | 0 | 1 | 0.131 | 0.196 | 0.310 | 0.450 |
| fp32 | 1 | 1 | 0.129 | 0.194 | 0.307 | 0.539 |

- fp32 dest raises per-tile cost (T=4: fp32 0.310 vs bf16 0.252). full_sync has
  little effect at these tile counts.

## MB2 — accumulation (DST-resident vs L1-pack)

Summary: out = initial + sum of `iters` contributions on an accumulator of
`acc_tiles` tiles, run two ways with the same reader, timing each strategy's loop.
Seed (dfb_init) and contributions (dfb_delta) use separate DFBs.

```
# DST-resident (acc_dst.cpp): accumulator stays in DST, pack once.
# Each contribution is added in place with binary_dest_reuse_tiles<ELWADD,
# DEST_TO_SRCA> — the op tt-lang's tile_accumulate_add lowers to.
binary_op_init_common(dfb_delta, dfb_delta, dfb_out); copy_tile_init(dfb_init)
acquire_dst                                                # held across whole loop
zone "acc_loop":
  cb_wait_front(dfb_init, U); copy U tiles dfb_init->DST; cb_pop_front(dfb_init, U)  # seed
  binary_dest_reuse_tiles_init<ELWADD, DEST_TO_SRCA>(dfb_delta)
  for it in range(iters):
    cb_wait_front(dfb_delta, U)
    for u in range(U): binary_dest_reuse_tiles(dfb_delta, u, u)  # DST[u] += dfb_delta[u]
    cb_pop_front(dfb_delta, U)
commit; wait; pack U tiles DST->dfb_out once; release_dst   # single pack

# L1-pack (acc_l1.cpp): accumulator lives in L1, re-packed every step.
cb_reserve_back(dfb_out, U); pack_reconfig_l1_acc(0)
zone "acc_loop":
  cb_wait_front(dfb_init, U); copy+pack U tiles dfb_init->dfb_out; cb_pop_front(dfb_init, U)  # seed (overwrite)
  pack_reconfig_l1_acc(1)
  for it in range(iters):
    cb_wait_front(dfb_delta, U)
    copy U tiles dfb_delta->DST; pack U tiles DST->dfb_out   # packer L1-accumulate
    cb_pop_front(dfb_delta, U)
pack_reconfig_l1_acc(0); cb_push_back(dfb_out, U)
# out = initial + sum of `iters` contributions; report dst/l1-pack µs + PCC vs torch
```

Two independent axes:
- **Strategy** (the comparison): where the running accumulator lives — DST
  registers (DST-resident) vs an L1 buffer (L1-pack).
- **Contribution residency** (`--source l1|dram`, orthogonal): contributions
  re-read from one L1 block (`l1`, isolates compute-thread cost) vs one block
  streamed per iteration from DRAM (`dram`, end-to-end). This sets the absolute
  cost; it does not change the strategy ranking.

Blackhole bf16, full sweep, all PCC ≈ 1.0. trisc_max µs as **DST / L1-pack
(faster)**:

l1-resident (contributions re-read from L1):

| acc_tiles \ iters | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 | 0.77/0.77 (L1) | 1.00/0.84 (L1) | 1.09/0.97 (L1) | 1.28/1.14 (L1) | 1.60/1.41 (L1) |
| 2 | 1.15/0.93 (L1) | 1.26/0.98 (L1) | 1.40/1.19 (L1) | 1.74/1.54 (L1) | 2.41/2.12 (L1) |
| 4 | 1.36/1.26 (L1) | 1.53/1.25 (L1) | 1.88/1.52 (L1) | 2.53/2.06 (L1) | 3.85/3.17 (L1) |

dram-streamed (one contribution block per iteration from DRAM):

| acc_tiles \ iters | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 | 0.77/0.78 (DST) | 1.28/1.18 (L1) | 2.06/2.07 (DST) | 3.83/3.86 (DST) | 7.49/7.42 (L1) |
| 2 | 1.10/0.97 (L1) | 1.59/1.46 (L1) | 2.54/2.31 (L1) | 4.51/4.28 (L1) | 8.21/7.98 (L1) |
| 4 | 1.37/1.16 (L1) | 1.89/1.69 (L1) | 3.00/2.73 (L1) | 5.16/4.94 (L1) | 9.53/9.37 (L1) |

- **Strategy ranking:** L1-pack is lower than DST-resident in nearly every config,
  by a small margin (~0.1–0.3 µs, up to 0.68 at l1-resident acc_tiles=4 iters=16).
  Three dram cells (acc_tiles=1, iters 1/4/8) are within ±0.03 µs — effectively
  tied. The two strategies are close; L1-pack is marginally ahead. (With the
  earlier SFPU add the DST kernel looked far worse — that was a kernel artifact.)
- **Residency effect (orthogonal):** streaming from DRAM adds a per-iteration DRAM
  read, so absolute cost grows steeply with iters (acc_tiles=1 iters=16: ~7.4 µs
  streamed vs ~1.4 µs resident, ~5×). This is larger than the strategy difference
  and independent of it — the DST-vs-L1-pack ranking holds under both residencies.
- **Notes:**
  1. The DST kernel uses `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` with a
     `copy_tile` seed and `binary_op_init_common` — the op sequence tt-lang's
     `tile_accumulate_add` lowers to. Seed (dfb_init) and contributions (dfb_delta)
     use separate DFBs, as in tt-lang (`initial_dfb`/`delta_dfb`); optimized
     production kernels likewise use separate seed and contribution DFBs.
  2. tt-lang's DST accumulation is verified numerically correct on hardware
     (refactor branch `tensor_recurrence_dst_acc.py`, PASS).
  3. Thread overlap: DST-resident holds one acquire with the pack thread idle
     until the final pack; L1-pack pipelines unpack + pack per iteration across
     two threads — a real strategy difference.

## MB3 — matmul K-accumulation (DST-K vs L1-K)

Summary: C[mt,nt] = sum_k A[k] @ B[k] over `kt` K-tiles, output P = mt*nt tiles,
run two ways. The output is tiled into sub_mt*sub_nt subblocks chosen exactly as
the compiler would (`harness.dst_subblock`, mirroring `computeMultiDimSubblockSizes`),
so reuse = (mt/sub_mt)·(nt/sub_nt) matches what tt-lang will emit. `matmul_block`
over each subblock, `mm_block_init`, A/B prefetched.

```
# DST-K (matmul_k_dst.cpp): each subblock held in DST across K, packed once.
# Operands prefetched once; re-unpacked per subblock when reuse > 1.
mm_block_init(dfb_in0, dfb_in1, dfb_out, 0, sub_nt, sub_mt, 1)
cb_reserve_back(dfb_out, mt*nt)
zone "matmul_k_loop":
  cb_wait_front(dfb_in0, kt*mt); cb_wait_front(dfb_in1, kt*nt)   # operands resident
  for (om, on) in subblocks:
    acquire_dst
    for k in range(kt):
      matmul_block(dfb_in0, dfb_in1, k*mt+om, k*nt+on, 0, 0, sub_nt, sub_mt, 1)
    commit; wait
    pack_tile<true> sub_mt*sub_nt tiles -> row-major C positions  # one pack/subblock
    release_dst
  pop kt*mt, kt*nt

# L1-K (matmul_k_l1.cpp): mt*nt accumulator in L1; each K step matmuls every
# subblock into fresh DST and packs to L1 with packer L1-accumulate.
mm_block_init(...); cb_reserve_back(dfb_out, mt*nt); pack_reconfig_l1_acc(0)
zone "matmul_k_loop":
  for k in range(kt):
    cb_wait_front(dfb_in0, mt); cb_wait_front(dfb_in1, nt)        # A col k, B row k
    for (om, on) in subblocks:
      acquire_dst; matmul_block(dfb_in0, dfb_in1, om, on, 0,0, sub_nt, sub_mt, 1); commit; wait
      pack_tile<true> sub tiles -> L1 (packer accumulate)
      release_dst
    pop mt, nt
    if k == 0: pack_reconfig_l1_acc(1)
pack_reconfig_l1_acc(0)
```

Both strategies measure operand handoff + math + pack inside the zone (the prior
table here measured DST-K with its single pack *outside* the zone — math only —
which made DST-K look ~math-bound and always cheaper; corrected below).

### MB3.A — output fits DST (P ≤ capacity, reuse = 1)

One subblock = the whole output. DST-K accumulates all `kt` matmuls in DST and
packs P tiles once; L1-K repacks the P-tile output to L1 every K step (kt·P
packs). DST-K prefetches operands once (its best case; a streaming reuse-1 kernel
would pay slightly more handoff). Blackhole bf16 HiFi4, trisc_max µs **DST-K /
L1-K (faster)**, all PCC ≈ 1.0:

| P (mt×nt) \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 (1×1) | 0.85/0.87 (DST) | 1.62/1.59 (L1) | 3.59/3.30 (L1) | 7.08/6.87 (L1) |
| 2 (1×2) | 0.84/0.85 (DST) | 1.77/1.67 (L1) | 3.62/3.53 (L1) | 7.64/7.29 (L1) |
| 4 (2×2) | 1.03/1.18 (DST) | 2.31/2.08 (L1) | 4.67/3.95 (L1) | 9.31/7.91 (L1) |
| 8 (2×4) | 1.44/1.48 (DST) | 2.88/2.47 (L1) | 5.77/4.47 (L1) | 11.61/8.75 (L1) |

### MB3.B — output exceeds DST (P > capacity, reuse > 1)

The output no longer fits DST, so it is subblocked. DST-K processes one subblock
at a time across the whole K loop, re-unpacking that subblock's A rows and B cols
from the resident operands — operand unpack scales by reuse. L1-K still repacks
every subblock every K step. Blackhole bf16 HiFi4, DST capacity 8, trisc_max µs
**DST-K / L1-K (faster)**, all PCC ≈ 1.0:

| P (mt×nt) reuse \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 16 (4×4) reuse 2 | 1.94/1.97 (DST) | 3.81/3.02 (L1) | 7.60/5.21 (L1) | 15.81/10.22 (L1) |
| 32 (4×8) reuse 4 | 2.83/2.87 (DST) | 5.77/4.62 (L1) | 11.28/7.66 (L1) | 23.12/14.19 (L1) |

### Interpretation

- **At kt = 1** (no K-accumulation) the two are tied within run-to-run variation
  (≤~3%); DST-K is marginally ahead because L1-K pays the `pack_reconfig_l1_acc`
  setup for a single pack.
- **For any real K-accumulation (kt ≥ 2) L1-K is faster, and the margin grows with
  both kt and P (reuse)** — from ~6% (P=1, kt=2) to ~39% (P=32, kt=8). This holds
  in both regimes; reuse > 1 widens the gap but does not change the direction.
- **Why, despite L1-K moving far more tiles** (e.g. P=16, kt=8: L1-K packs 128
  tiles to DST-K's 16): DST-K cannot pack until all `kt` matmuls finish, so its
  pack engine is idle through the K loop and the pack runs as a serial tail after
  the math. L1-K interleaves matmul and pack per K step, overlapping them across
  the unpack/math/pack RISCs. The overlap recovered outweighs the extra packs.
- **Consequence for the #652 selector:** a data-movement score (tiles packed ·
  per-tile cost) ranks DST-K cheaper in every cell, the opposite of measured
  wall-clock. Ranking matmul K-accumulation needs a model term for the pack/math
  overlap L1-K gets and DST-K cannot — not just the per-engine tile weights.
- **Scope:** isolated single kernel. It omits DST occupancy / fusion, which only
  pushes further toward L1-K (a resident DST output starves fused neighbors). It
  also omits `matmul_block` granularity effects. Remaining sweeps: fp32 dest and
  full-sync.

**Fidelity (LoFi / HiFi2 / HiFi4).** Math fidelity is a parameter (1 / 2 / 4 math
passes per matmul); the tables above are HiFi4. P = 16 (4×4, reuse 2), trisc_max
µs **DST-K / L1-K (faster)**, all PCC ≈ 1.0.

Blackhole bf16:

| fidelity \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| LoFi  | 1.40/1.37 (L1) | 2.64/2.50 (L1) | 5.33/4.73 (L1) | 11.11/9.89 (L1) |
| HiFi2 | 1.58/1.61 (DST) | 3.21/2.63 (L1) | 6.07/4.78 (L1) | 12.83/10.12 (L1) |
| HiFi4 | 1.94/1.97 (DST) | 3.81/3.02 (L1) | 7.60/5.21 (L1) | 15.81/10.22 (L1) |

Wormhole b0:

| fidelity \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| LoFi  | 2.24/2.29 (DST) | 3.94/3.65 (L1) | 7.65/6.51 (L1) | 14.86/12.31 (L1) |
| HiFi2 | 2.42/2.43 (DST) | 4.21/3.69 (L1) | 8.24/6.61 (L1) | 16.27/12.50 (L1) |
| HiFi4 | 2.73/2.66 (L1) | 5.24/4.15 (L1) | 10.16/6.98 (L1) | 19.77/12.51 (L1) |

On both arches L1-K is nearly fidelity-insensitive (kt=8: BH 9.89 → 10.12 → 10.22,
WH 12.31 → 12.50 → 12.51 — flat) because it is pack-bound; its per-K-step packs
hide the math. DST-K serializes math then one pack, so it scales with fidelity
(BH 11.11 → 12.83 → 15.81, WH 14.86 → 16.27 → 19.77). The L1-K margin grows with
fidelity (~5–11% LoFi to ~21–35% HiFi4) but the ranking is unchanged: L1-K is
lower for kt ≥ 2 at every fidelity, in both MB3.A (reuse=1) and MB3.B. For plain
matmul, fidelity sets the margin, not the strategy.

### Wormhole b0 (1000 MHz)

Same sweep, bf16 HiFi4, DST capacity 8, trisc_max µs **DST-K / L1-K (faster)**,
all PCC ≈ 1.0.

MB3.A (P ≤ capacity, reuse = 1):

| P (mt×nt) \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 (1×1) | 1.15/1.13 (L1) | 2.09/1.84 (L1) | 3.93/3.81 (L1) | 8.27/7.59 (L1) |
| 2 (1×2) | 1.11/1.26 (DST) | 2.26/2.03 (L1) | 4.82/4.11 (L1) | 9.08/8.43 (L1) |
| 4 (2×2) | 1.39/1.40 (DST) | 2.84/2.64 (L1) | 5.63/4.80 (L1) | 11.32/9.35 (L1) |
| 8 (2×4) | 1.97/2.08 (DST) | 3.81/3.29 (L1) | 7.47/5.63 (L1) | 14.47/10.56 (L1) |

MB3.B (P > capacity, reuse > 1):

| P (mt×nt) reuse \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 16 (4×4) reuse 2 | 2.73/2.66 (L1) | 5.24/4.15 (L1) | 10.16/6.98 (L1) | 19.77/12.51 (L1) |
| 32 (4×8) reuse 4 | 3.93/3.93 (=) | 7.75/6.12 (L1) | 15.14/10.60 (L1) | 30.21/19.31 (L1) |

- **Blackhole vs Wormhole:** the ranking is identical — tied at kt=1, L1-K
  lower for kt ≥ 2, margin growing with kt and reuse. Wormhole absolute times are
  ~1.1–1.4× Blackhole (slower clock; costlier pack). The per-engine pack ratio
  (Wormhole pack ≈ 2.4× Blackhole, which loads L1-K's extra packs more heavily)
  does not change the lower-cost strategy: the matmul/pack overlap dominates on
  both arches, so the #652 decision for matmul K-accumulation is arch-independent
  in the isolated case.

### MB3.C — fused GELU epilogue (`--fuse gelu`)

A GELU activation is applied to the matmul output. DST-K applies it in place on
the resident DST subblock before its single pack (no reload). L1-K must reload
its L1 accumulator into DST, apply GELU, and pack again — the round trip the
resident output lets DST-K skip. Fast (tanh) GELU is used, the production default
for matmul fusion; its approximation lowers PCC to ~0.985 (accepted for fused
matmul+activation), versus ~1.0 for plain matmul. Blackhole bf16 HiFi4,
trisc_max µs **DST-K / L1-K (faster)**:

| P (mt×nt) \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 (1×1) | 1.29/1.43 (DST) | 2.09/2.23 (DST) | 3.90/3.92 (DST) | 7.95/7.66 (L1) |
| 4 (1×4) | 1.99/2.35 (DST) | 3.10/3.25 (DST) | 5.46/5.13 (L1) | 10.33/9.20 (L1) |
| 8 (2×4) | 2.80/3.31 (DST) | 4.26/4.35 (DST) | 6.99/6.35 (L1) | 12.96/10.68 (L1) |
| 16 (4×4) reuse 2 | 4.53/5.15 (DST) | 6.62/6.34 (L1) | 10.26/8.43 (L1) | 18.51/13.58 (L1) |

- **The epilogue makes DST-K the lower-cost strategy at low kt.** Without it,
  L1-K is faster for every kt ≥ 2 (MB3.A/B). With it, DST-K is faster at kt = 1
  (all P) and at kt = 2 for P ≤ 8. L1-K's matmul/pack overlap advantage over
  DST-K grows with kt (the same effect that wins MB3.A/B). The epilogue adds to
  L1-K a one-time reload + repack of the output (~2P tile movements, independent
  of kt) that DST-K avoids by applying GELU in place. At small kt that fixed
  reload cost exceeds the overlap advantage, so DST-K is cheaper; as kt grows the
  overlap advantage exceeds it and L1-K is cheaper.
- **The crossover kt shrinks as P grows** (DST-K leads through kt=4 at P=1, kt=2 at
  P≤8, only kt=1 at P=16): the reload cost includes a fixed per-phase overhead
  (the extra DFB handoff and copy/GELU init) on top of the ~2P movements; at
  larger P that overhead amortizes, so L1-K's per-kt overlap advantage overtakes
  the reload cost at a smaller kt.
- **Epilogue cost moves the crossover.** This is fast GELU (~0.4 µs/tile here). A
  costlier activation (erf-precise GELU measured ~10× heavier) widens the
  DST-favorable band to higher kt; a cheaper one (bias add) narrows it. So the
  selector's epilogue handling should weigh the L1-K reload against the activation
  cost, not assume a fixed crossover.
- **Consequence for the #652 selector:** with a fused epilogue the resident-output
  advantage is real and DST-K is the lower-cost strategy for shallow-K matmuls;
  the plain-matmul preference for L1-K (overlap) only holds at larger kt. The
  decision is epilogue-aware, not just kt/reuse-based.

Wormhole b0 (1000 MHz), fused GELU, trisc_max µs **DST-K / L1-K (faster)**:

| P (mt×nt) \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 (1×1) | 1.89/2.02 (DST) | 2.63/2.79 (DST) | 5.07/4.78 (L1) | 8.89/8.67 (L1) |
| 4 (1×4) | 2.90/3.42 (DST) | 4.51/4.46 (L1) | 7.42/6.92 (L1) | 13.13/11.78 (L1) |
| 8 (2×4) | 4.24/5.23 (DST) | 6.15/6.46 (DST) | 9.67/8.84 (L1) | 16.89/14.03 (L1) |
| 16 (4×4) reuse 2 | 7.18/8.34 (DST) | 9.50/9.60 (DST) | 15.13/12.64 (L1) | 24.24/17.94 (L1) |

- **Wormhole is slightly more DST-favorable at kt=2** (DST-K wins kt=2 up to P=16,
  vs only P≤8 on Blackhole). Wormhole's costlier pack (≈2.4× Blackhole) makes
  L1-K's reload + repack more expensive, extending DST-K's region. The kt≥4 L1-K
  preference is unchanged. So an epilogue tilts the decision toward DST-K more on
  Wormhole than on Blackhole — the one place so far where the arch matters to the
  ranking.

**Fidelity (fused).** Unlike plain matmul — where fidelity only sets the margin —
with the epilogue fidelity moves the crossover. P = 16 (4×4, reuse 2), fused GELU,
trisc_max µs **DST-K / L1-K (faster)**.

Blackhole bf16:

| fidelity \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| LoFi  | 3.95/4.65 (DST) | 5.29/5.69 (DST) | 7.82/7.87 (DST) | 13.78/12.94 (L1) |
| HiFi2 | 4.16/4.96 (DST) | 5.71/5.85 (DST) | 8.65/8.04 (L1) | 15.45/13.09 (L1) |
| HiFi4 | 4.53/5.15 (DST) | 6.62/6.34 (L1) | 10.26/8.43 (L1) | 18.51/13.58 (L1) |

Wormhole b0:

| fidelity \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| LoFi  | 6.30/7.83 (DST) | 8.30/9.11 (DST) | 11.74/12.10 (DST) | 19.13/17.89 (L1) |
| HiFi2 | 6.55/7.79 (DST) | 8.69/9.25 (DST) | 12.57/12.34 (L1) | 20.49/18.09 (L1) |
| HiFi4 | 7.18/8.34 (DST) | 9.50/9.60 (DST) | 15.13/12.64 (L1) | 24.24/17.94 (L1) |

DST-K's favorable kt range extends as fidelity drops — Blackhole: through kt = 4
(LoFi), kt = 2 (HiFi2), kt = 1 (HiFi4); Wormhole: kt = 4 (LoFi), kt = 2 (HiFi2 and
HiFi4). Lower fidelity gives L1-K less math to hide its reload + repack behind, so
the epilogue penalty dominates over more kt. Wormhole stays DST-favorable through
kt = 2 even at HiFi4 — one step further than Blackhole — because its costlier pack
makes L1-K's repack dearer. With an epilogue, fidelity is part of the decision;
without one it is not.

## MB4 — compute-op (math) microbenchmarks — planned

The data-movement benchmarks (MB1–MB3) have no compute-engine term. Per
`~/tt/perf/SDPA_Optimizations.xlsx`, SDPA/flash is compute/SFPU-bound, and that
sheet already measures each math op as INIT / KERNEL / TILE_LOOP with per-RISC
measurements. MB4 measures per-op compute-engine tile costs to feed a compute-aware
model (also what the nonadditive work needs).

Ops to cover (from the xlsx per-op sheets):

| engine | ops |
|---|---|
| FPU (matrix) | `matmul_tiles`, `matmul_block` (× kt), matmul init |
| SFPU (vector) | `exp` (fast/slow), `sub`/`sub_exp`, `add_block`, `mul_tiles_bcast`/`mul_block_bcast` (× broadcast Col/Row/Scalar), inplace variants |
| reduce | FPU fast-reduce, SFPU reduce (`reduce_max`, `reduce_sum`) + reduce init |
| copy | `copy_tile` |
| pack | fast vs slow pack, full vs half DST-bank util, pack-with-acc |

Options not yet covered:

- `math_fidelity` (LoFi / HiFi2 / HiFi4), `fp32_dest_acc`, `unpack_to_dest`,
  `broadcast_type`.
- **Init cost measured separately from the loop** (a major SDPA lever:
  "compressed inits", "re-inits", "hoist inits") — mirror the INIT vs TILE_LOOP
  marker split.
- fast-pack vs slow-pack; full vs half DST bank.
- TRISC cross-op overlap (the xlsx "overlapped" rows).

Method: same harness (handwritten compute kernel via `generic_op`,
`DeviceZoneScopedN` per op-loop, per-RISC µs), sweep `tile_cnt` + the parameters;
report per-op TILE-LOOP µs/tile and init µs, mirroring the xlsx's
INIT/KERNEL/TILE_LOOP + per-RISC columns.

## Result CSVs (local only, git-ignored; under `benchmarks/microbench/results/`)

- `pack_unpack_blackhole_bf16_*.csv` — MB1 bf16 tiles=1..16
- `pack_unpack_blackhole_matrix_*.csv` — MB1 config matrix
- `pack_unpack_wormhole_b0_bf16_*.csv` — MB1 bf16 tiles=1..16
- `accumulation_blackhole_bf16_l1_*.csv` — MB2 DST vs L1-pack, l1-resident, acc_tiles×iters
- `accumulation_blackhole_bf16_dram_*.csv` — MB2 DST vs L1-pack, dram-streamed, acc_tiles×iters
- `matmul_k_blackhole_bf16_hifi4_*.csv` — MB3 DST-K vs L1-K, P and kt sweep (MB3.A reuse=1 + MB3.B reuse>1)
- `matmul_k_wormhole_b0_bf16_hifi4_*.csv` — MB3 same sweep on Wormhole (on `aus-wh-01:/localdev/bnorris/tt-lang`)
- `matmul_k_blackhole_bf16_hifi4_gelu_*.csv` — MB3.C DST-K vs L1-K with a fused GELU epilogue
- `matmul_k_wormhole_b0_bf16_hifi4_gelu_*.csv` — MB3.C fused GELU on Wormhole
- `matmul_k_blackhole_bf16_{lofi,hifi2}_none_*.csv` — fidelity for plain matmul (MB3.A/B); HiFi4 in the hifi4 CSV
- `matmul_k_blackhole_bf16_{lofi,hifi2}_gelu_*.csv` — fidelity for fused GELU (MB3.C)
- `matmul_k_wormhole_b0_bf16_{lofi,hifi2}_{none,gelu}_*.csv` — Wormhole fidelity, P=16 (plain + fused)

## Open / next

- MB3 — done on Blackhole and Wormhole: MB3.A reuse=1, MB3.B reuse>1, MB3.C fused
  GELU, each swept over fidelity (LoFi/HiFi2/HiFi4) on both arches. Remaining:
  fp32 dest, full-sync.
- Per-engine weights: the June-16 LLK run
  (`~/tt/perf/tt_metal_llk_perf_2026-06-16_27594326478`) already supplies unpack /
  pack / l1-acc-surcharge / matmul-math (cycles, both arches); use it rather than
  re-running the suite.
- Wormhole run of MB2.
- fp32 + sync extensions for MB2; distinct-DFB superlinearity for MB1.
- MB4 — compute-op (math) microbenchmarks.
