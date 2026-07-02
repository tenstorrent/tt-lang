# Microbench Results

A record of what the microbenchmarks measure on hardware. Measurements only;
interpretation against any cost model lives elsewhere.

## Motivation

The tt-metal tt-llk perf suite already measures per-engine tile costs (unpack,
pack, matmul/eltwise math, L1-accumulation pack surcharge) in isolation, and those
supply most of the cost-model weights directly. These benchmarks cover effects
that isolated LLK measurements do not measure:

- **Composition.** The generated kernel runs unpack/math/pack pipelined across
  three RISCs; the LLK microbenchmarks time each engine alone, so adding them up
  does not reproduce the composed cost. Blackhole bf16, per tile:
  - LLK unpack ~ 0.030 + LLK pack ~ 0.026 = **0.056 µs** (serial sum)
  - slowest single engine = **0.030 µs** (perfect-overlap lower bound)
  - MB1 measured, over DFBs = **0.039 µs**

  The measured value sits between the serial sum and the perfect-overlap lower
  bound. The kernel pipelines unpack and pack across RISCs, recovering ~30% of
  the serial sum. A serial sum overestimates this case by ~1.4x, and the overlap
  factor varies by op and tile count.
- **Dataflow-buffer handoff.** The LLK harness has no dataflow buffers, so it never
  measures the DFB reserve/wait/push/pop and cross-thread synchronization paid
  by dispatched kernels. MB1 measures this fixed term at ~0.09 µs/iter on
  Blackhole for the cost model's `dfbHopFixedCost`.
- **Measured strategy order.** The model chooses between DST-resident and L1-pack
  strategies. LLK measurements provide the primitive costs for the serial model;
  these benchmarks measure the generated sequence directly. MB2 measures
  L1-pack as marginally faster than DST-resident for additive accumulation, while
  the serial model predicts DST-resident as cheaper. That mismatch is
  not observable from LLK primitive measurements alone.

The LLK microbenchmarks parameterize the model; these benchmarks check whether it is
calibrated.

## Methodology (brief)

- Handwritten C++ kernels (compute + reader/writer), modeled on the tt-metal
  tt-llk perf benchmarks, but run through **tt-metal dispatch**
  (`ttnn.generic_op`) over dataflow buffers. The measurement includes DFB
  reserve/wait/push/pop and cross-thread synchronization, which the bare-metal
  LLK harness omits.
- **No tt-lang compiler involved:** kernels are handwritten and JIT-compiled by
  tt-metal at run time; the sequences are matched by hand to what tt-lang emits.
- Single compute core; inputs L1-resident where possible.
- Timing: `DeviceZoneScopedN` per RISC; cycles / profiler `CHIP_FREQ[MHz]` = µs.
- Correctness: PCC vs a torch reference.
- DFB block count (multi-buffering depth, tt-lang's term) is held at the default
  (`block_count = 2`) in the headline comparisons. It is exposed as `--block-count`
  and swept once in the MB2 block-count section. It changes absolute streamed
  cost through reader prefetch, but it does not change the faster accumulation
  strategy. The other tables fix it to isolate the strategy comparison. A block
  count of 1 understates steady-state throughput;
  values above 2 add no benefit in these compute-bound cases.

Times are compute-thread (TRISC) µs. Inputs preloaded to L1 keep NCRISC/BRISC out
of the measured zone (MB1, and the l1-resident MB2/MB3 runs); the dram-streamed
MB2 runs intentionally overlap reader (NCRISC) activity with the compute wait, so
the zone is not DM-idle there. `noc_active_in_zone` only flags a data-movement
thread when it records the same profiler zone, so treat it as a hint, not proof of
idleness. PCC ~ 1.0 unless noted.

## Hardware / environment

Profiler clock (used for the cycles -> µs conversion):

| architecture | freq (profiler) |
|---|---|
| Blackhole | 1350 MHz |
| Wormhole b0 | 1000 MHz |

Runs use a tt-metal environment with `ttnn` available, with the device profiler
enabled: `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 python -m
benchmarks.microbench.<sweep> ...`. No tt-lang compiler build is needed because
the benchmarks dispatch handwritten kernels via `ttnn`/`generic_op`.

## MB1: Pack/Unpack Probe

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

### bf16 linear fit (`us_per_iter = fixed + per_tile*tiles`, trisc_max)

| architecture | fixed µs | per_tile µs | r^2 |
|---|---|---|---|
| Blackhole | 0.089 | 0.039 | 0.99 |
| Wormhole | 0.160 | 0.065 | 0.98 |

- Round-trip / pipelined-throughput basis: per-RISC unpack~math~pack because the
  zone spans the same pipelined window, so this gives the *combined* per-tile
  cost, not the separate unpack/pack engine costs (those need the per-engine LLK microbenchmarks).

### Blackhole Config Matrix

| dtype | full_sync | fp32_acc | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|---|---|
| bf16 | 0 | 0 | 0.107 | 0.161 | 0.252 | 0.441 |
| bf16 | 1 | 0 | 0.106 | 0.159 | 0.251 | 0.439 |
| bf16 | 0 | 1 | 0.113 | 0.164 | 0.255 | 0.388 |
| fp32 | 0 | 1 | 0.131 | 0.196 | 0.310 | 0.450 |
| fp32 | 1 | 1 | 0.129 | 0.194 | 0.307 | 0.539 |

- fp32 dest raises per-tile cost (T=4: fp32 0.310 vs bf16 0.252). full_sync has
  little effect at these tile counts.

## MB2: Accumulation (DST-Resident vs L1-Pack)

Summary: out = initial + sum of `iters` contributions on an accumulator of
`acc_tiles` tiles, run two ways with the same reader, timing each strategy's loop.
Seed (dfb_init) and contributions (dfb_delta) use separate DFBs.

```
# DST-resident (acc_dst.cpp): accumulator stays in DST, pack once.
# Each contribution is added in place with binary_dest_reuse_tiles<ELWADD,
# DEST_TO_SRCA>. This is the op tt-lang's tile_accumulate_add lowers to.
binary_op_init_common(dfb_delta, dfb_delta, dfb_out); copy_tile_init(dfb_init)
acquire_dst; cb_reserve_back(dfb_out, U)                    # acquire held across whole loop
zone "acc_loop":
  cb_wait_front(dfb_init, U); copy U tiles dfb_init->DST; cb_pop_front(dfb_init, U)  # seed
  binary_dest_reuse_tiles_init<ELWADD, DEST_TO_SRCA>(dfb_delta)
  for it in range(iters):
    cb_wait_front(dfb_delta, U)
    for u in range(U): binary_dest_reuse_tiles(dfb_delta, u, u)  # DST[u] += dfb_delta[u]
    cb_pop_front(dfb_delta, U)
  commit; wait; pack U tiles DST->dfb_out once             # single pack, timed in the zone
cb_push_back(dfb_out, U); release_dst

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
- **Strategy** (the comparison): the running accumulator lives either in DST
  registers (DST-resident) or in an L1 buffer (L1-pack).
- **Contribution residency** (`--source l1|dram`, orthogonal): contributions
  re-read from one L1 block (`l1`, isolates compute-thread cost) vs one block
  streamed per iteration from DRAM (`dram`, which adds the per-iteration DRAM read
  into the compute-thread zone). The reported value is still TRISC zone time, not
  full dispatch time. Contribution residency sets the absolute cost; it does not
  change the faster strategy.

Blackhole bf16, full sweep, all PCC ~ 1.0. trisc_max µs as **DST / L1-pack
(faster)**:

l1-resident (contributions re-read from L1):

| acc_tiles \ iters | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 | 0.76/0.78 (DST) | 1.04/0.79 (L1) | 1.15/0.94 (L1) | 1.26/1.10 (L1) | 1.64/1.40 (L1) |
| 2 | 1.15/0.95 (L1) | 1.26/1.02 (L1) | 1.41/1.26 (L1) | 1.77/1.46 (L1) | 2.48/2.08 (L1) |
| 4 | 1.42/1.11 (L1) | 1.61/1.31 (L1) | 1.88/1.51 (L1) | 2.60/2.06 (L1) | 3.86/3.24 (L1) |

dram-streamed (one contribution block per iteration from DRAM):

| acc_tiles \ iters | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 | 0.79/0.79 (DST) | 1.20/1.20 (DST) | 2.06/2.08 (DST) | 3.80/3.83 (DST) | 7.60/7.59 (L1) |
| 2 | 1.16/0.90 (L1) | 1.62/1.40 (L1) | 2.56/2.38 (L1) | 4.45/4.18 (L1) | 8.57/7.96 (L1) |
| 4 | 1.37/1.08 (L1) | 2.00/1.70 (L1) | 3.10/2.77 (L1) | 5.25/4.95 (L1) | 9.58/9.25 (L1) |

- **Measured strategy order:** L1-pack is faster than DST-resident in nearly
  every config, by a small margin (~0.1-0.6 µs, largest at l1-resident
  acc_tiles=4 iters=16: 3.86 vs 3.24). The acc_tiles=1 cells are near-ties. A
  few favor DST-resident by <=0.03 µs. The two strategies are close; L1-pack is
  marginally ahead.
- **Timed consistently with MB3:** both kernels time the pack inside the zone.
  L1-pack times its per-step packs; DST-resident times its single final pack. The
  single DST pack is small next to the accumulate loop. This differs from MB3
  matmul, where the final pack is a large serial tail. Timing the final pack does
  not change the faster MB2 strategy.
- **Residency effect (orthogonal):** streaming from DRAM adds a per-iteration DRAM
  read, so absolute cost grows steeply with iters (acc_tiles=1 iters=16: ~7.4 µs
  streamed vs ~1.4 µs resident, ~5x). This is larger than the difference between
  DST-resident and L1-pack. L1-pack remains faster under both residencies, except
  for the near-tie cells noted above.
- **Notes:**
  1. The DST kernel uses `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` with a
     `copy_tile` seed and `binary_op_init_common`. This is the op sequence
     tt-lang's `tile_accumulate_add` lowers to. Seed (dfb_init) and
     contributions (dfb_delta) use separate DFBs, as in tt-lang
     (`initial_dfb`/`delta_dfb`); optimized production kernels likewise use
     separate seed and contribution DFBs.
  2. tt-lang's DST accumulation is verified numerically correct on hardware
     (refactor branch `tensor_recurrence_dst_acc.py`, PASS).
  3. Thread overlap: DST-resident holds one acquire with the pack thread idle
     until the final pack. L1-pack pipelines unpack and pack per iteration
     across two threads. This structural difference explains the measured
     advantage for L1-pack.

### Wormhole b0 (1000 MHz)

Same sweep, bf16, all PCC ~ 1.0. trisc_max µs **DST / L1-pack (faster)**.

l1-resident:

| acc_tiles \ iters | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 | 1.28/1.03 (L1) | 1.33/1.01 (L1) | 1.56/1.30 (L1) | 1.64/1.59 (L1) | 2.03/2.17 (DST) |
| 2 | 1.51/1.18 (L1) | 1.81/1.39 (L1) | 1.91/1.77 (L1) | 2.21/2.11 (L1) | 2.99/3.11 (DST) |
| 4 | 1.86/1.49 (L1) | 2.04/1.79 (L1) | 2.51/2.28 (L1) | 3.21/3.14 (L1) | 4.79/5.10 (DST) |

dram-streamed:

| acc_tiles \ iters | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 | 1.28/0.96 (L1) | 1.74/1.58 (L1) | 2.60/2.31 (L1) | 4.87/4.29 (L1) | 8.51/8.19 (L1) |
| 2 | 1.52/1.19 (L1) | 2.05/1.65 (L1) | 3.16/2.99 (L1) | 5.28/4.98 (L1) | 9.93/9.66 (L1) |
| 4 | 1.92/1.47 (L1) | 2.55/2.19 (L1) | 3.83/3.50 (L1) | 6.57/6.24 (L1) | 12.02/11.77 (L1) |

Matches Blackhole: L1-pack is marginally ahead in nearly every config; the
dram absolute cost grows steeply with iters. The one difference is that
l1-resident at iters=16 is a near-tie that flips to DST-resident within
run-to-run variance (DST/L1 within ~5%). At the highest iter count the two
strategies converge on Wormhole. Otherwise, both architectures select the same
faster strategy.

### Block count (DFB depth)

`--block-count` (default 2) sets how many contribution/output DFB blocks the
reader can prefetch. dram-streamed, acc_tiles=2, L1-pack trisc_max µs:

| architecture, iters \ block_count | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Blackhole, iters=8  | 4.86 | 4.27 | 4.19 | 4.18 |
| Blackhole, iters=16 | 9.32 | 7.95 | 8.24 | 7.99 |
| Wormhole, iters=8   | 6.13 | 4.93 | 4.89 | 4.91 |
| Wormhole, iters=16  | 11.65 | 9.28 | 9.31 | 9.37 |

Block count 1 serializes each DRAM read against compute; block count 2 lets the
reader prefetch the next contribution, cutting ~15-20%; beyond 2 it is flat
(compute-bound). l1-resident is block-count-insensitive because it re-reads one
L1 block and does not stream (BH iters=16 ~2.1 µs, WH ~3.1 µs across block
counts 1-8). Block count changes the dram-streamed absolute cost, but it does
not change the faster strategy, since both strategies share the contribution
buffer. The cost model's DFB term should distinguish single- from multi-block
streaming; the strategy choice does not depend on it.

### fp32 dest and full-sync

**fp32 dest (dtype = fp32).** Halves DST capacity (4 vs 8), so the DST-resident
strategy is legal only for acc_tiles <= 4, and roughly doubles pack cost. The
measured strategy order matches bf16. L1-pack is faster in nearly every config;
the costlier pack does not shift the faster strategy to DST-resident
(Blackhole l1-resident acc_tiles=4 iters=16: L1 ahead ~16%; Wormhole near-tied
at high iters, as in bf16). Exact (PCC ~ 1.0).

**Full-sync (dst_full_sync_en).** Full-sync flips the measured strategy order on
Wormhole. Full-sync makes
DST a single bank (no math/pack double-buffer), so L1-pack's per-iteration copy
can no longer overlap the previous iteration's pack. L1-pack loses its
pipelining.
DST-resident accumulates in place and packs once, so full-sync does not touch it.
l1-resident, trisc_max µs **DST / L1-pack (faster)**:

| architecture, acc_tiles \ iters | 1 | 4 | 16 |
|---|---|---|---|
| Blackhole, 4 | 1.41/1.15 (L1) | 1.95/1.54 (L1) | 3.85/3.33 (L1) |
| Blackhole, 8 | 1.79/1.65 (L1) | 2.75/2.54 (L1) | 6.85/6.12 (L1) |
| Wormhole, 4  | 1.83/1.48 (L1) | 2.41/2.52 (DST) | 4.78/6.63 (DST) |
| Wormhole, 8  | 2.54/2.46 (L1) | 3.75/4.35 (DST) | 8.37/12.78 (DST) |

On Blackhole full-sync leaves L1-pack ahead (its cheap pack barely suffers from
losing the overlap: l1 acc_tiles=4 iters=16 rises only ~5% vs default). On
Wormhole the un-hidden pack is ~2.4x costlier, so L1-pack rises ~39% (4.76 -> 6.63)
and DST-resident takes the lead at iters >= 4 for acc_tiles >= 4. For the
additive recurrence, full-sync is a strategy-selection input on Wormhole because
it makes DST-resident faster. Full-sync does not change the faster strategy on
Blackhole.

### Per-Iteration Expression (`--expr add|mul|gelu`)

`--expr` sets the per-iteration contribution applied before accumulation:

- `add`: contribution = delta. DST-resident accumulates in place with
  binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>; L1-pack packs with
  pack_reconfig_l1_acc. This is the additive recurrence the rest of MB2 measures.
- `mul`: contribution = delta*delta (FPU). L1-pack only. A product cannot
  accumulate in DST in place (mul_tiles overwrites the dest tile and no FPU op
  adds two DST tiles), so there is no DST-resident candidate.
- `gelu`: contribution = gelu(delta) (SFPU, tanh approximation). DST-resident
  computes gelu in a temporary DST slot and adds it into the accumulator with
  add_binary_tile (one extra SFPU add per iteration); L1-pack computes gelu, then
  the packer accumulates.

source=l1, bf16, acc_tiles=4 (the largest output legal for gelu DST-resident).
trisc_max µs for the whole loop, DST / L1-pack (faster):

add:
| arch \ iters | 1 | 4 | 16 |
|---|---|---|---|
| Blackhole | 1.47/1.11 (L1) | 1.90/1.49 (L1) | 3.89/3.10 (L1) |
| Wormhole  | 1.89/1.53 (L1) | 2.49/2.29 (L1) | 4.78/4.78 (tie) |

gelu:
| arch \ iters | 1 | 4 | 16 |
|---|---|---|---|
| Blackhole | 2.47/1.93 (L1) | 6.14/3.95 (L1) | 20.85/12.23 (L1) |
| Wormhole  | 3.71/2.77 (L1) | 9.63/6.27 (L1) | 33.49/20.28 (L1) |

mul (L1-pack only), trisc_max µs: Blackhole 1.25 (iters=1) -> 5.00 (iters=16);
Wormhole 1.84 -> 7.07.

For the additive contribution, the two strategies are close. L1-pack is ~25-33%
faster on Blackhole across iters. On Wormhole, the L1-pack advantage shrinks as
iters grows because DST-resident packs once, while L1-pack pays Wormhole's ~2.4x
costlier per-iteration pack. The Wormhole sweep reaches a tie at acc_tiles=4
iters=16 and a small DST-resident lead at acc_tiles=1 iters=16 (2.14 vs 2.32).
For gelu, L1-pack is decisively faster on both architectures and the margin
widens with iters (Blackhole 1.28x at iters=1 to 1.71x at iters=16; Wormhole
1.34x to 1.65x). DST-resident accumulation of a non-additive contribution pays
an explicit per-iteration add_binary_tile on top of the gelu, while L1-pack folds
the accumulation into the packer. PCC ~ 1.0 for add/mul; ~0.99 for gelu (tanh
approximation).

The trend: the more compute-bearing or non-additive the per-iteration
contribution, the more L1-pack is favored. DST-resident is competitive only for the
bare additive recurrence, and even there only on Blackhole (cheap pack) and at low
iteration counts on Wormhole.

## MB3: Matmul K-Accumulation (DST-K vs L1-K)

Summary: C[mt,nt] = sum_k A[k] @ B[k] over `kt` K-tiles, output P = mt*nt tiles,
run two ways. The output is tiled into sub_mt*sub_nt subblocks chosen exactly as
the compiler would (`harness.dst_subblock`, mirroring `computeMultiDimSubblockSizes`),
so reuse = (mt/sub_mt)*(nt/sub_nt) matches what tt-lang will emit. `matmul_block`
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
table here measured DST-K with its single pack *outside* the zone, so it measured
math only. That made DST-K look approximately math-bound and always cheaper. The
measurements below include the pack.

### Per-node GEMM utilization checks

`third-party/tt-metal/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md` reports matrix-engine
throughput and this utilization formula:

```
ideal cycles = (M * K * N) / (32 * 32 * 32) * cycle_per_tile / num_cores
```

For MB3, `num_cores = 1`, `M = mt * 32`, `N = nt * 32`, and `K = kt * 32`, so:

```
matmul_ideal_cycles = mt * nt * kt * cycle_per_tile
cycle_per_tile = 16 (LoFi), 32 (HiFi2), 64 (HiFi4)
```

`matmul_sweep.py` now records this per-node check in the CSV:

- `matmul_ideal_cycles`: ideal matrix-engine cycles for the tiled matmul work.
- `trisc_max_cycles`: measured zone time converted with profiler `CHIP_FREQ[MHz]`.
- `math_cycles`: measured math-thread zone time converted with the same clock.
- `zone_utilization_pct`: `100 * matmul_ideal_cycles / trisc_max_cycles`.
- `math_utilization_pct`: `100 * matmul_ideal_cycles / math_cycles`.

`math_utilization_pct` is the raw `matmul_block` sanity check. A low value there
means the benchmark is not feeding the matrix engine efficiently.
`zone_utilization_pct` is the single-node composed-kernel utilization, including
unpack/pack and DFB handoff. Fused GELU leaves these columns blank because the
timed zone includes epilogue work and is not a plain matmul measurement.

These columns are sanity checks for MB3, not claims that MB3 is a GEMM
benchmark. The current MB3 numbers should not be read as TTNN GEMM device
utilization. The specific differences are:

- `matmul_sweep.py` is a one-node `ttnn.generic_op` strategy comparison. Its
  timed zone includes DFB handoff, `cb_wait_front`, `cb_pop_front`, tile-register
  synchronization, and pack.
- MB3 calls `matmul_block(..., kt_dim=1)` once per K tile per output subblock.
  This matches the strategy the cost model needs to compare, but it is not the
  TTNN large-block GEMM program.
- L1-K intentionally packs the accumulator to L1 every K step. That measures the
  L1 accumulation strategy, not raw matrix-engine throughput.
- The tt-metal GEMM benchmark runs `ttnn.matmul` with
  `MatmulMultiCoreReuseMultiCastProgramConfig`, the GEMM subblock preference
  order, `packer_l1_acc=True`, `math_approx_mode=True`, and
  `ThrottleLevel.NO_THROTTLE`. Its device-utilization number uses the profiler's
  TRISC1 kernel duration.
- MB3's `zone_utilization_pct` is the composed-kernel cost-model number.
  `math_utilization_pct` is the closer matrix-engine feeding check, but it is
  still from the handwritten generic-op kernel, not from TTNN matmul.

The handwritten path differs from TTNN in these ways:

- TTNN dispatches `MatmulMultiCoreReuseMultiCastProgramConfig` through
  `MatmulMultiCoreReuseMcast2DProgramFactory`. Even on a 1x1 grid, that factory
  constructs the same optimized A-reader, B-reader/output-writer, compute-kernel,
  DFB, and compile-time-argument contract used by the full-grid GEMM benchmark.
- TTNN's compute kernel is
  `bmm_large_block_zm_fused_bias_activation.cpp`. Its compute arguments include
  `in0_block_w`, `in0_num_subblocks`, `in0_block_num_tiles`,
  `in1_num_subblocks`, `in1_block_num_tiles`, output block counts, subblock
  sizes, and `cb_intermed0`. MB3 passes only `mt`, `nt`, `kt`, `sub_mt`,
  `sub_nt`, and optional epilogue state.
- TTNN consumes a K block in the large-block kernel by looping
  `inner_dim_idx < in0_block_w` and calling `matmul_block(..., kt_dim =
  in0_block_w)` while advancing A/B offsets. MB3 DST-K and L1-K call
  `matmul_block(..., kt_dim = 1)` per K tile per output subblock.
- TTNN handles `num_blocks_inner_dim > 1` with an intermediate partials DFB. It
  packs partials to `cb_intermed0` and reloads only when needed; when
  `packer_l1_acc` is enabled, longer K blocks avoid part of the spill/reload
  cost. MB3 L1-K intentionally repacks the full accumulator to L1 on every K
  step; MB3 DST-K intentionally keeps DST resident and re-unpacks operands per
  output subblock.
- TTNN readers stream operand blocks in the layout required by the large-block
  kernel. A uses `out_block_h * in0_block_w` tiles per block; B uses
  `out_block_w * in0_block_w` tiles per block. MB3's reader lays out A as
  `kt` columns of `mt` tiles and B as `kt` rows of `nt` tiles to support the
  strategy comparison.
- TTNN's B reader is also the output writer, so its output DFB and writer
  contract are part of the optimized program. MB3 uses the common
  `drain_writer.cpp`, which is intentionally generic across the microbenchmarks.
- TTNN configures `math_approx_mode=True`, `packer_l1_acc=True`, and
  `ThrottleLevel.NO_THROTTLE` in the GEMM benchmark. MB3 uses the generic-op
  harness compute config with `math_approx_mode=False` and no explicit throttle
  or TTNN matmul program-level options.

`matmul_compute_sweep.py` measures the matmul lowering cost model's compute-feed
term. It uses a 1x1 grid via `ttnn.generic_op` with single-node block layouts,
implemented as five diagnostic variants. mm2 through mm5 each change one
implementation detail relative to the previous variant, so those deltas isolate
single effects. mm1 is the naive baseline. The mm1 -> mm2 comparison also
changes subblock selection (`dst_subblock` to the TTNN preference order) and the
reader/writer scaffolding, so its delta measures the block kernel setup as a
whole. That delta is dominated by switching from `matmul_tiles` to
`matmul_block`.

- `mm1_tile_loop`: per-tile `matmul_tiles` in a K loop.
- `mm2_block`: `matmul_block` over output subblocks, whole K block resident
  (num_blocks = 1), operands waited for outside the timed zone.
- `mm3_block_stream`: K-block streaming with spill-and-reload accumulation.
- `mm4_block_stream_l1acc`: packer L1 accumulation instead of spill-reload (TTNN's
  no-bias mechanism: blocks 0..n-2 accumulate in L1, only the last block reloads).
- `mm5_block_stream_l1acc_packblock`: one `pack_tile_block` per subblock instead of
  a per-tile `pack_tile` loop.

The metric comparable to TTNN's GEMM number is **math thread (TRISC1)**
utilization. Zone utilization (trisc_max) is smaller because the **pack thread**
is the slowest thread at this size. That is a metric distinction, not evidence
of a matrix-engine feed difference.

8x8 per node, bf16 HiFi4, math-thread utilization. num_blocks=1 is kt=8 div=1;
num_blocks=3 is kt=24 div=3 (mm2_block is one-block only, so it is absent at
num_blocks=3):

num_blocks=1:

| variant | change added | Blackhole | Wormhole |
|---|---|---:|---:|
| mm1_tile_loop | baseline (`matmul_tiles`) | 80.2% | 74.6% |
| mm2_block | `matmul_block` | 87.9% | 82.9% |
| mm3_block_stream | K-block streaming | 87.8% | 82.5% |
| mm4_block_stream_l1acc | packer L1 accumulation | 88.0% | 82.9% |
| mm5_..._packblock | block pack | 88.1% | 83.0% |
| TTNN matmul (reference) | n/a | 88.0% | 82.8% |

num_blocks=3:

| variant | Blackhole | Wormhole |
|---|---:|---:|
| mm1_tile_loop | 81.8% | 76.1% |
| mm3_block_stream | 89.1% | 85.1% |
| mm4_block_stream_l1acc | 90.8% | 87.1% |
| mm5_..._packblock | 90.8% | 87.2% |
| TTNN matmul (reference) | 90.9% | 87.1% |

Per-change benefit (percentage points of math-thread utilization):

- **mm1 -> mm2: +7.7 pp Blackhole, +8.4 pp Wormhole.** This comparison bundles
  `matmul_block`, the TTNN subblock selection, and the block reader/writer. The
  gain is dominated by `matmul_block`: one MOP call per subblock reuses A/B
  across the subblock with no per-tile RISC dispatch, while `matmul_tiles`
  re-issues every tile.
- **mm2 -> mm3: ~0 pp.** K-block streaming is utilization-neutral in this sweep.
  Streaming is still needed when K exceeds what fits resident in L1.
- **mm3 -> mm4: +1.7 pp Blackhole, +2.1 pp Wormhole at num_blocks=3.** Packer L1
  accumulation removes the per-block reload copies. The benefit grows with
  num_blocks: it is neutral at 1-2 blocks, where both variants do at most one
  reload; Blackhole reaches +2.6 pp at num_blocks=4. mm4 reaches TTNN parity on
  both architectures (BH 90.8 vs 90.9; WH 87.1 vs 87.1).
- **mm4 -> mm5: ~0 pp both.** Block pack is utilization-neutral. The residual
  pack-thread overhang (zone < math) is pack-engine throughput, not RISC
  dispatch. Collapsing the pack calls does not recover it; TTNN sits at the same
  ceiling.

Wormhole reports a smaller absolute utilization percentage than Blackhole at the
same config, which indicates a lower matrix-feed ceiling. The variant sequence
behaves the same on both architectures, and mm4 matches TTNN on both. The earlier reading
that the handwritten kernel trailed TTNN compared zone/pack-thread time against
TTNN's math-thread time. On the same thread metric they match. Operand residency
also does not explain the difference, since mm2 and mm3 are equivalent at
num_blocks=1.

Blackhole also reaches mm4 91.8% (TTNN 92.1%) at num_blocks=4 (kt=32). The
streamed variants (mm3-mm5) size in0/in1 to a double-buffered K block, so they
fit Wormhole at kt=32. The resident baseline variants (mm1, mm2) hold the whole
operand, which overflows Wormhole's 1.43 MB L1 at kt=32. The complete variant
set on Wormhole is therefore shown at num_blocks=3 (kt=24, where every variant
fits). Blackhole has enough L1 for the complete variant set at num_blocks=4.

### MB3.A: Output Fits DST (P <= capacity, reuse = 1)

One subblock = the whole output. DST-K accumulates all `kt` matmuls in DST and
packs P tiles once; L1-K repacks the P-tile output to L1 every K step (kt*P
packs). DST-K prefetches operands once (its best case; a streaming reuse-1 kernel
would pay slightly more handoff). Blackhole bf16 HiFi4, trisc_max µs **DST-K /
L1-K (faster)**, all PCC ~ 1.0:

| P (mtxnt) \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 (1x1) | 0.85/0.87 (DST) | 1.62/1.59 (L1) | 3.59/3.30 (L1) | 7.08/6.87 (L1) |
| 2 (1x2) | 0.84/0.85 (DST) | 1.77/1.67 (L1) | 3.62/3.53 (L1) | 7.64/7.29 (L1) |
| 4 (2x2) | 1.03/1.18 (DST) | 2.31/2.08 (L1) | 4.67/3.95 (L1) | 9.31/7.91 (L1) |
| 8 (2x4) | 1.44/1.48 (DST) | 2.88/2.47 (L1) | 5.77/4.47 (L1) | 11.61/8.75 (L1) |

### MB3.B: Output Exceeds DST (P > capacity, reuse > 1)

The output no longer fits DST, so it is subblocked. DST-K processes one subblock
at a time across the whole K loop, re-unpacking that subblock's A rows and B cols
from the resident operands. Operand unpack scales by reuse. L1-K still repacks
every subblock every K step. Blackhole bf16 HiFi4, DST capacity 8, trisc_max µs
**DST-K / L1-K (faster)**, all PCC ~ 1.0:

| P (mtxnt) reuse \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 16 (4x4) reuse 2 | 1.94/1.97 (DST) | 3.81/3.02 (L1) | 7.60/5.21 (L1) | 15.81/10.22 (L1) |
| 32 (4x8) reuse 4 | 2.83/2.87 (DST) | 5.77/4.62 (L1) | 11.28/7.66 (L1) | 23.12/14.19 (L1) |

### Interpretation

- **At kt = 1** (no K-accumulation) the two are tied within run-to-run variation
  (<=~3%); DST-K is marginally ahead because L1-K pays the `pack_reconfig_l1_acc`
  setup for a single pack.
- **For any K-accumulation (kt >= 2), L1-K is faster.** The margin grows with
  both kt and P (reuse), from ~2% at P=1, kt=2 to ~39% at P=32, kt=8. This holds
  in both regimes; reuse > 1 widens the margin but does not change which strategy
  is faster.
- **Why, despite L1-K moving far more tiles** (e.g. P=16, kt=8: L1-K packs 128
  tiles to DST-K's 16): DST-K cannot pack until all `kt` matmuls finish, so its
  pack engine is idle through the K loop and the pack runs as a serial tail after
  the math. L1-K interleaves matmul and pack per K step, overlapping them across
  the unpack/math/pack RISCs. The overlap recovered outweighs the extra packs.
- **Consequence for the #652 selector:** a data-movement score (tiles packed *
  per-tile cost) predicts DST-K as cheaper in every cell, which is the
  opposite of measured wall-clock time. Matmul K-accumulation needs a model term
  for the pack/math overlap that L1-K gets and DST-K cannot get. Per-engine tile
  weights alone are not sufficient.
- **Scope:** isolated single kernel. It omits cross-op DST occupancy. A matmul
  that holds its output resident in DST prevents fused neighbors, such as a
  softmax sharing DST, from using DST at the same time. That pushes toward L1-K.
  This is distinct from the in-kernel epilogue fusion measured in MB3.C, which
  favors DST-K at low kt. The two fusion effects favor opposite strategies. MB3
  also omits `matmul_block` granularity.

**Fidelity (LoFi / HiFi2 / HiFi4).** Math fidelity is a parameter (1 / 2 / 4 math
passes per matmul); the tables above are HiFi4. P = 16 (4x4, reuse 2), trisc_max
µs **DST-K / L1-K (faster)**, all PCC ~ 1.0.

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

On both architectures L1-K is nearly fidelity-insensitive (kt=8: BH 9.89 -> 10.12 -> 10.22,
WH 12.31 -> 12.50 -> 12.51) because it is pack-bound; its per-K-step packs
hide the math. DST-K serializes math then one pack, so it scales with fidelity
(BH 11.11 -> 12.83 -> 15.81, WH 14.86 -> 16.27 -> 19.77). The L1-K margin grows with
fidelity (~5-11% LoFi to ~21-35% HiFi4), but the measured strategy order is
unchanged. L1-K is faster for kt >= 2 at every fidelity, in both MB3.A (reuse=1)
and MB3.B. For plain matmul, fidelity changes the size of L1-K's lead but not
the faster strategy.

**fp32 dest (dtype = fp32).** fp32 accumulation halves DST capacity (4 vs 8), so
reuse > 1 is reached at smaller outputs, and roughly doubles pack cost (2x tile
bytes). P = 16 (4x4) is reuse 4 here (vs reuse 2 at bf16). trisc_max µs **DST-K /
L1-K (faster)**, all PCC = 1.0:

| architecture \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Blackhole | 2.00/2.00 (=) | 4.05/3.27 (L1) | 8.29/6.02 (L1) | 18.27/12.94 (L1) |
| Wormhole  | 3.62/3.57 (L1) | 6.67/5.69 (L1) | 13.21/10.03 (L1) | 26.17/18.26 (L1) |

Measured strategy order is unchanged: L1-K is faster for kt >= 2 on both
architectures, exact (PCC 1.0). The
margin is slightly narrower than bf16 (BH P=16 kt=8: 29% vs 35%). fp32's costlier
pack loads L1-K's extra packs more heavily, partly offsetting its overlap
advantage. The halved capacity raises reuse at a given P but does not change the
winner.

**Full-sync (dst_full_sync_en).** Full-sync doubles DST capacity (16 vs 8 bf16),
so P = 16 (4x4) becomes reuse 1 (vs reuse 2 at the default capacity). trisc_max µs
**DST-K / L1-K (faster)**, all PCC ~ 1.0:

| architecture \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Blackhole | 2.11/2.14 (DST) | 3.92/3.23 (L1) | 7.70/5.41 (L1) | 15.79/10.41 (L1) |
| Wormhole  | 2.96/3.02 (DST) | 5.44/4.82 (L1) | 10.36/8.47 (L1) | 19.93/15.59 (L1) |

Full-sync extends the reuse-1 range to larger P but does not change which
strategy is faster or the margin (BH P=16 kt=8: 34% with full-sync at reuse 1 vs
35% at default reuse 2). L1-K is faster for kt >= 2. L1-K's advantage comes from
pipelining pack against the next K step's matmul across the unpack/math/pack
RISCs, which is independent of the DST bank mode, so full-sync does not recover
it for DST-K.

### Wormhole b0 (1000 MHz)

Same sweep, bf16 HiFi4, DST capacity 8, trisc_max µs **DST-K / L1-K (faster)**,
all PCC ~ 1.0.

MB3.A (P <= capacity, reuse = 1):

| P (mtxnt) \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 (1x1) | 1.15/1.13 (L1) | 2.09/1.84 (L1) | 3.93/3.81 (L1) | 8.27/7.59 (L1) |
| 2 (1x2) | 1.11/1.26 (DST) | 2.26/2.03 (L1) | 4.82/4.11 (L1) | 9.08/8.43 (L1) |
| 4 (2x2) | 1.39/1.40 (DST) | 2.84/2.64 (L1) | 5.63/4.80 (L1) | 11.32/9.35 (L1) |
| 8 (2x4) | 1.97/2.08 (DST) | 3.81/3.29 (L1) | 7.47/5.63 (L1) | 14.47/10.56 (L1) |

MB3.B (P > capacity, reuse > 1):

| P (mtxnt) reuse \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 16 (4x4) reuse 2 | 2.73/2.66 (L1) | 5.24/4.15 (L1) | 10.16/6.98 (L1) | 19.77/12.51 (L1) |
| 32 (4x8) reuse 4 | 3.93/3.93 (=) | 7.75/6.12 (L1) | 15.14/10.60 (L1) | 30.21/19.31 (L1) |

- **Blackhole vs Wormhole:** the measured strategy order is identical. The
  strategies are tied at kt=1, and L1-K is faster for kt >= 2. The margin grows
  with kt and reuse. Wormhole absolute times are
  ~1.1-1.4x Blackhole (slower clock; costlier pack). The per-engine pack ratio
  (Wormhole pack ~ 2.4x Blackhole, which loads L1-K's extra packs more heavily)
  does not change the faster strategy. The matmul/pack overlap dominates on both
  architectures, so the #652 decision for matmul K-accumulation is
  architecture-independent in the isolated case.

### MB3.C: Fused GELU Epilogue (`--fuse gelu`)

A GELU activation is applied to the matmul output. DST-K applies it in place on
the resident DST subblock before its single pack (no reload). L1-K must reload
its L1 accumulator into DST, apply GELU, and pack again. DST-K avoids that round
trip because its output is already resident in DST. Fast (tanh) GELU is used, the production default
for matmul fusion; its approximation lowers PCC to ~0.985 (accepted for fused
matmul+activation), versus ~1.0 for plain matmul. Blackhole bf16 HiFi4,
trisc_max µs **DST-K / L1-K (faster)**:

| P (mtxnt) \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 (1x1) | 1.29/1.43 (DST) | 2.09/2.23 (DST) | 3.90/3.92 (DST) | 7.95/7.66 (L1) |
| 4 (1x4) | 1.99/2.35 (DST) | 3.10/3.25 (DST) | 5.46/5.13 (L1) | 10.33/9.20 (L1) |
| 8 (2x4) | 2.80/3.31 (DST) | 4.26/4.35 (DST) | 6.99/6.35 (L1) | 12.96/10.68 (L1) |
| 16 (4x4) reuse 2 | 4.53/5.15 (DST) | 6.62/6.34 (L1) | 10.26/8.43 (L1) | 18.51/13.58 (L1) |

- **The epilogue makes DST-K faster at low kt.** Without it,
  L1-K is faster for every kt >= 2 (MB3.A/B). With it, DST-K is faster at kt = 1
  (all P) and at kt = 2 for P <= 8. L1-K's matmul/pack overlap advantage over
  DST-K grows with kt (the same effect that wins MB3.A/B). The epilogue adds to
  L1-K a one-time reload + repack of the output (~2P tile movements, independent
  of kt) that DST-K avoids by applying GELU in place. At small kt that fixed
  reload cost exceeds the overlap advantage, so DST-K is faster; as kt grows the
  overlap advantage exceeds it and L1-K is faster.
- **The crossover kt shrinks as P grows** (DST-K leads through kt=4 at P=1, kt=2 at
  P<=8, only kt=1 at P=16): the reload cost includes a fixed per-phase overhead
  (the extra DFB handoff and copy/GELU init) on top of the ~2P movements; at
  larger P that overhead amortizes, so L1-K's per-kt overlap advantage overtakes
  the reload cost at a smaller kt.
- **Epilogue cost moves the crossover.** This is fast GELU (~0.4 µs/tile here). A
  costlier activation (erf-precise GELU measured ~10x heavier) widens the
  DST-favorable band to higher kt; a cheaper one (bias add) narrows it. So the
  selector's epilogue handling should weigh the L1-K reload against the activation
  cost, not assume a fixed crossover.
- **Activation runs on the math thread** (`gelu_tile`), matching tt-lang's current
  codegen: its SFPU lowering emits `gelu_tile` (math thread). A pack-thread
  activation (`gelu_tile_pack`, as in tt-metal's fused matmul) keeps the math
  thread pure matmul and would make DST-K faster still. That would strengthen
  this result, not reverse it. tt-lang does not emit that today, but it is not a
  hardware limit: it needs a pack-thread TTKernel op plus lowering or direct
  EmitC.
  MB4 currently measures the math-thread SFPU ops; a pack-thread arm to quantify
  this headroom is planned.
- **Consequence for the #652 selector:** with a fused epilogue the resident-output
  advantage holds and DST-K is faster for shallow-K matmuls;
  the plain-matmul preference for L1-K (overlap) only holds at larger kt. The
  decision is epilogue-aware, not only kt/reuse-based.

Wormhole b0 (1000 MHz), fused GELU, trisc_max µs **DST-K / L1-K (faster)**:

| P (mtxnt) \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 (1x1) | 1.89/2.02 (DST) | 2.63/2.79 (DST) | 5.07/4.78 (L1) | 8.89/8.67 (L1) |
| 4 (1x4) | 2.90/3.42 (DST) | 4.51/4.46 (L1) | 7.42/6.92 (L1) | 13.13/11.78 (L1) |
| 8 (2x4) | 4.24/5.23 (DST) | 6.15/6.46 (DST) | 9.67/8.84 (L1) | 16.89/14.03 (L1) |
| 16 (4x4) reuse 2 | 7.18/8.34 (DST) | 9.50/9.60 (DST) | 15.13/12.64 (L1) | 24.24/17.94 (L1) |

- **Wormhole is slightly more DST-favorable at kt=2** (DST-K wins kt=2 up to P=16,
  vs only P<=8 on Blackhole). Wormhole's costlier pack (~2.4x Blackhole) makes
  L1-K's reload + repack more expensive, extending DST-K's region. The kt>=4 L1-K
  preference is unchanged. An epilogue tilts the decision toward DST-K more on
  Wormhole than on Blackhole. This is the one measured case where architecture
  changes the strategy decision.

**Fidelity (fused).** For plain matmul fidelity changes only the size of L1-K's
lead, not which strategy wins. With the epilogue it changes which strategy wins
by moving the crossover kt. P = 16 (4x4, reuse 2), fused GELU, trisc_max µs **DST-K
/ L1-K (faster)**.

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

DST-K's favorable kt range extends as fidelity drops. Blackhole: through kt = 4
(LoFi), kt = 2 (HiFi2), kt = 1 (HiFi4); Wormhole: kt = 4 (LoFi), kt = 2 (HiFi2 and
HiFi4). Lower fidelity gives L1-K less math to hide its reload + repack behind, so
the epilogue penalty dominates over more kt. Wormhole stays DST-favorable through
kt = 2 even at HiFi4, one step further than Blackhole, because its costlier pack
makes L1-K's repack more expensive. With an epilogue, fidelity is part of the decision;
without one it is not.

## MB4: Compute-Op Math Microbenchmarks

The data-movement benchmarks (MB1-MB3) have no compute-engine term. SDPA/flash is
compute/SFPU-bound, where each math op is best characterized as INIT / KERNEL /
TILE_LOOP with a per-RISC split. MB4 measures per-op compute-engine tile costs to
feed a compute-aware model (also what the nonadditive work needs).

**Implemented (`compute_sweep.py`, one kernel per category --
`compute_unary.cpp`, `compute_binary.cpp` (binary + broadcast),
`compute_reduce.cpp`): unary, binary, broadcast, and reduce on the math
thread.** A selected op runs on `tiles` tiles `iters`
times; the second operand is all ones (timing-neutral, so the PCC ref is trivial).
`op=copy` is the unary baseline. Blackhole bf16, tiles=4, math thread.

*Unary SFPU* -- marginal µs/tile over the copy baseline (~0.033), i.e. the SFPU
math added on top of the copy+pack pipeline:

| op | exp | gelu (fast) | recip | sqrt | rsqrt |
|---|---|---|---|---|---|
| µs/tile | 0.42 | 0.14 | 0.88 | 0.64 | 0.73 |

*Binary / broadcast / reduce* -- absolute composed µs/tile (unpack + op + pack,
pipelined; these read their operands directly, so there is no copy to subtract).
`copy` shown for reference; reduce is per *input* tile reduced (output is one tile):

| op | copy | add | sub_bcast | mul | mul_bcast | reduce_max | reduce_sum |
|---|---|---|---|---|---|---|---|
| µs/tile | 0.033 | 0.027 | 0.027 | 0.068 | 0.069 | 0.050 | 0.103 |

Observations (measured; mechanisms are tentative):

- **add/sub ≈ 0.027, below the copy baseline.** A fused FPU binary appears cheaper
  than `copy_tile`, which routes through the SFPU copy path.
- **mul ≈ 2.5x add** (0.068 vs 0.027). Likely because multiply runs through a
  fidelity-honoring FPU datapath (default HiFi4) while add/sub are single-pass --
  to confirm with a fidelity sweep, so the binary cost is op- (and probably
  fidelity-) dependent.
- **Broadcast is ~free over its binary base** (mul_bcast ≈ mul, sub_bcast ≈ add):
  the per-row column-broadcast adds negligible cost.
- **reduce_sum ≈ 2x reduce_max** (0.103 vs 0.050 per input tile): sum and max
  appear to take different reduce datapaths.
- **SFPU transcendentals dominate.** The cheapest SFPU op (gelu, 0.14 marginal)
  already exceeds mul, and recip (0.88) is ~30x a fused add -- in SDPA-like code
  the SFPU ops, not the FPU binary/bcast, set the compute cost.

PCC is the real op accuracy (SFPU sqrt/rsqrt ~0.9994, gelu ~0.9997; reduce ~1.0,
the result living in column 0). Validated across bf16/fp32 and init_hoist 0/1 (all
paths run). Pending: the clean fp32 and INIT-vs-loop tables (from the init_hoist
delta), the full tile-count sweep, a pack-thread activation arm (the MB3.C
headroom -- `gelu_tile_pack` / `exp_packthread_tile`, both confirmed in the API),
and both architectures.

Coverage status:

| engine | ops | status |
|---|---|---|
| SFPU unary | `exp`, `gelu`, `recip`, `sqrt`, `rsqrt`, `copy` | **done** (table above) |
| FPU binary | `add`, `mul` (`sub` ≈ `add`) | **done** |
| broadcast | `mul`/`sub` bcast cols | **done** (rows/scalar pending) |
| reduce | SFPU `reduce_sum`, `reduce_max` (row) | **done** (col reduce pending) |
| FPU matrix | `matmul_tiles`, `matmul_block`, `math_fidelity` | covered by MB3 / MB5.C |
| pack | fast vs slow, full vs half DST bank, pack-with-acc | pending |

Remaining:

- **Pack-thread activation arm** (the MB3.C headroom): `gelu_tile_pack` /
  `exp_packthread_tile` vs the math-thread ops above. Both confirmed in the API.
- **Clean `fp32` and INIT-vs-TILE_LOOP tables.** `fp32_dest_acc` and `init_hoist`
  are swept (paths validated), but the headline table is bf16/hoisted; the init
  µs is derivable from the `init_hoist` delta but not yet tabulated.
- Full **tile-count** sweep (the table above is tiles=4) and **both architectures**
  (Wormhole needs a WH device).
- `unpack_to_dest`, remaining `broadcast_type`s, pack micro-variants, and TRISC
  cross-op overlap (the xlsx "overlapped" rows).

Method: same harness (handwritten compute kernel via `generic_op`,
`DeviceZoneScopedN` per op-loop, per-RISC µs), sweep `tile_cnt` + the parameters;
report per-op TILE-LOOP µs/tile and init µs, mirroring the xlsx's
INIT/KERNEL/TILE_LOOP + per-RISC columns.

## MB5: Subblock-Size Selection

The compiler picks the output subblock by maximizing the tile product to fill DST
(`computeMultiDimSubblockSizes` / `harness.dst_subblock`). These sweeps test when
that is optimal. They share one method: hold the *total* tile work fixed,
force-sweep the subblock (the DST tiles held per `tile_regs_acquire`), and flag the
heuristic's choice with a `compiler_pick` column, then compare it against the
fastest row. The regimes run from zero-math (MB5.A) through light FPU (MB5.B),
plain matmul (MB5.C) and fused matmul (MB5.D).

Measured headline: **a larger subblock is faster for matmul (MB5.C/D); for the
zero-math passthrough (MB5.A) it does not always win -- the full-DST chunk can be
the worst.** Each subsection reports measured µs only; we are not attaching a
cost-model explanation yet.

### When does a larger subblock win? (measured roadmap)

A roadmap into MB5.A-D; all columns are measured:

| regime | subblock effect | larger subblock? |
|---|---|---|
| MB5.A pack/unpack (zero math) | up to ~22% | not always -- full-DST worst at small tile counts |
| MB5.B FPU ops (transpose/add/mul) | up to ~76% (chunk) | max-DST at working sizes; same-product shape spread ~19% (transpose) / ~7% (add) / ~0% (mul) |
| MB5.C matmul (clean) | ~1-11% | yes (only loser is the degenerate (1,1)) |
| MB5.D matmul + pre-seed (scaled-acc) | ~1-20% (shrinks with kt) | yes (reuse 8 best); a cap/2 heuristic would under-shoot |

Everything here is single-core Blackhole bf16. These are measured outcomes; we are
not yet proposing a cost-model mechanism for them.

### MB5.A: Zero-math pack/unpack -- chunk-size sweep

`subblock_pack_unpack_sweep.py` (reuses the MB1 `passthrough_compute.cpp`). No
arithmetic and no reuse; holds `tiles` fixed and sweeps the DST chunk per acquire
(each iteration does one DFB hop and `ceil(tiles/chunk)` acquires, so only the
acquire count changes). Multiples of 8 (so chunks 1/2/4/8 all divide evenly -- no remainder), Blackhole
bf16 half-sync (DST cap 8), trisc_max µs/iter:

| tiles | chunk=1 | chunk=2 | chunk=4 | chunk=8 | best | max-DST penalty |
|---|---|---|---|---|---|---|
| 8   | 0.367 | **0.362** | 0.384 | 0.441 | 2 | +22% |
| 16  | 0.661 | 0.649 | **0.634** | 0.693 | 4 | +9% |
| 24  | 0.953 | 0.926 | **0.898** | 0.933 | 4 | +4% |
| 32  | 1.248 | 1.206 | **1.154** | 1.176 | 4 | +2% |
| 40  | 1.540 | 1.485 | **1.410** | 1.429 | 4 | +1.3% |
| 48  | 1.834 | 1.761 | **1.666** | 1.672 | 4 | +0.3% |
| 56  | 2.128 | 2.041 | 1.923 | **1.919** | 8 | 0% (best) |
| 64  | 2.420 | 2.320 | 2.179 | **2.165** | 8 | 0% (best) |
| 128 | 4.767 | 4.548 | 4.230 | **4.126** | 8 | 0% (best) |
| 176 | 6.527 | 6.220 | 5.768 | **5.597** | 8 | 0% (best) |

Measured observations:

- **chunk=1 is slowest** at every tile count (most acquires).
- **The best chunk climbs monotonically toward the cap as `tiles` grows** -- 2 at
  tiles=8, 4 for tiles 16-48, 8 for tiles >= 56.
- **The max-DST chunk (8) is worst at small tile counts** (tiles=8: +22%,
  tiles=16: +9%) and its penalty decays smoothly to 0; the c4->c8 crossover is near
  tiles=52 (48: chunk=8 is +0.3%; 56: chunk=8 best).

**Full-sync (double-buffering off, `--full-sync 1`).** Same sweep with the DST
single-banked (cap 16, no MATH↔PACK double-buffer). per-iter µs:

| tiles | c1 | c2 | c4 | c8 | c16 | best | max-DST penalty |
|---|---|---|---|---|---|---|---|
| 8   | 0.471 | 0.410 | **0.401** | 0.439 | -- | 4 | +9% (vs c8) |
| 16  | 0.885 | 0.745 | **0.697** | 0.754 | 0.813 | 4 | +17% |
| 24  | 1.300 | 1.084 | **0.991** | 1.063 | 1.140 | 4 | +15% |
| 32  | 1.714 | 1.421 | **1.285** | 1.370 | 1.521 | 4 | +18% |
| 40  | 2.129 | 1.761 | **1.583** | 1.681 | 1.829 | 4 | +16% |
| 48  | 2.543 | 2.099 | **1.876** | 1.991 | 2.206 | 4 | +18% |
| 56  | 2.959 | 2.435 | **2.169** | 2.299 | 2.520 | 4 | +16% |
| 64  | 3.374 | 2.773 | **2.467** | 2.608 | 2.897 | 4 | +17% |
| 128 | 6.692 | 5.474 | **4.823** | 5.079 | 5.665 | 4 | +18% |
| 176 | 9.181 | 7.501 | **6.592** | 6.932 | 7.739 | 4 | +17% |

Turning the double-buffer off does **not** make the max-DST chunk win -- the
opposite. The contrast with half-sync is sharp:

| | half-sync (cap 8) | full-sync (cap 16) |
|---|---|---|
| best chunk vs size | drifts 2 → 4 → 8 (up to the cap) | **fixed at 4 at every size** |
| max-DST penalty | +22% → 0 (max-DST wins at >=56 tiles) | **~+17%, never decays** (max-DST never wins) |

Mechanism (verified in the LLK, not a fitted model): `dst_full_sync_en` removes
*only* the **MATH↔PACK** double-buffer -- which in half-sync is exactly what lets a
big chunk overlap its pack with the next acquire's math, so max-DST wins at large
totals. Without it, the big chunk's pack is an exposed serial tail. The overlap
that *survives* -- **UNPACK↔PACK** (separate RISCs, disjoint register banks: unpack
writes srcA, pack reads DST) -- rewards more, smaller acquires, so the optimum pins
at chunk=4 (small enough to keep that overlap, large enough to amortize the acquire
handshake) regardless of total size.

(tiles >= 184 exceed L1 even after trimming the single-use seed/drain DFBs;
non-power-of-two chunk sizes are deferred for now.)

### MB5.B: FPU ops (transpose / add / mul) -- subblock shape and size

`subblock_fpu_op_sweep.py` (`fpu_op_compute.cpp`) generalized from a transpose probe to three FPU ops over a
square `n x n` block, subblocked by `(sub_h, sub_w)`, repeated over a resident
input (CB hop outside the zone). Every op reads its operand(s) straight into
srcA/srcB -- no copy_tile; the binary ops feed the input as *both* operands
(`add` = 2x, `mul` = x²), so the footprint is `2·n²` tiles for all three and they
reach the same max block (n=16, 256 tiles). Blackhole bf16 half-sync, per-iter
trisc_max µs (128 iters), all PCC >= 0.9999.

**All subblock shapes** (n=8, 64 tiles -- every valid `(sub_h, sub_w)`, ranked by
subblock size (product) largest first; per-iter µs, min per op in bold):

| sub (h,w) | chunk | transpose | add | mul |
|---|---|---|---|---|
| (1,8) | 8 | **1.610** | **1.348** | **3.984** |
| (8,1) | 8 | 1.687 | 1.409 | 3.984 |
| (2,4) | 8 | 1.833 | 1.438 | 3.984 |
| (4,2) | 8 | 1.921 | 1.447 | 3.984 |
| (1,4) | 4 | 1.775 | 1.490 | 4.060 |
| (4,1) | 4 | 1.803 | 1.506 | 4.061 |
| (2,2) | 4 | 1.995 | 1.490 | 4.061 |
| (2,1) | 2 | 1.939 | 1.777 | 4.214 |
| (1,2) | 2 | 1.941 | 1.776 | 4.215 |
| (1,1) | 1 | 2.366 | 2.366 | 4.523 |

Two things vary -- the chunk (product) and the shape at a fixed product:

- **Chunk:** for all three ops per-iter falls as the product grows 1→8 (more tiles
  per acquire, fewer acquires); the chunk-8 rows (top) are fastest for every op.
- **Shape at fixed product:** the spread across same-product shapes is large for
  transpose (chunk=8 spans 1.610-1.921, ~19%), smaller for add (1.348-1.447, ~7%),
  and ~nil for mul (all 3.984). So *which* shape at a given product matters for
  transpose, barely for add, not at all for mul -- the spread shrinks as the op
  gets more math-bound (mul honors fidelity). n=16 (256 tiles) behaves the same
  (transpose chunk=8 spans 6.674-7.602). Access-pattern cause not investigated --
  measured only.

**Chunk-size sensitivity** (n=16, chunk=1 vs the best chunk=8):

| op | chunk=1 | best (chunk=8) | chunk=1 penalty |
|---|---|---|---|
| add | 9.45 | 5.38 | +76% |
| transpose | 9.45 | 6.67 | +42% |
| mul | 18.08 | 15.92 | +14% |

add (cheapest op) is most chunk-sensitive, mul (math-bound) least; add and
transpose coincide at chunk=1 (9.45 -- 256 acquires dominate, the op barely
registers) and separate at chunk=8 as the op cost surfaces. Op ordering holds:
add < transpose < mul (mul ~3-4×, fidelity).

**Best subblock: the max-DST chunk `(1,8)`** for all three ops at every size tested
(n=8 / n=16 best = transpose 1.610 / 6.674, add 1.348 / 5.382, mul 3.984 / 15.92) --
i.e. the compiler's maximize-product pick is fastest or tied.

(Square blocks cap at n=16 / 256 tiles in L1; larger or non-power-of-two sizes
are deferred.)

### MB5.C: Matmul (clean power-of-two)

`subblock_matmul_sweep.py` sweeps every valid `(sub_mt, sub_nt)` -- each dividing
`(mt, nt)` with `sub_mt*sub_nt <= DST capacity` -- through the MB3 matmul kernels,
selectable with `--accum`: DST-K (`matmul_k_dst.cpp`, default) or L1-K
(`matmul_k_l1.cpp`). `compiler_pick` flags the `computeMultiDimSubblockSizes`
shape; `reuse = (mt/sub_mt)·(nt/sub_nt)` is the measured operand re-read factor.

**Clean 8x8, Blackhole bf16 HiFi4.** trisc_max µs; the heuristic picks `(1,8)`.
All 10 valid subblocks land within ~1-2% except the degenerate `(1,1)` (reuse
64):

| kt | best | best µs | pick `(1,8)` gap | worst `(1,1)` penalty | util range |
|---|---|---|---|---|---|
| 1 | (1,8) | 4.60 | +0% | +11% | 59-66% |
| 2 | (1,8) | 9.02 | +0% | +5% | 64-67% |
| 4 | (1,8) | 18.06 | +0% | +2% | 66-67% |
| 8 | (4,2) | 37.60 | +0% (pick 37.68) | +2% | 63-65% |

On clean power-of-two shapes subblock choice is a minor lever (the pick is
within ~1% of optimal at every kt), and the only loser is `(1,1)`, which the
heuristic never picks. The penalty shrinks with kt: at low kt the matmul is
small so the per-subblock acquire + operand re-unpack (scaling with `reuse`) is
a larger fraction; at high kt the matmul amortizes it.

**L1-K accumulation (`--accum l1`, `matmul_k_l1.cpp`).** Same sweep with the
accumulator in L1 (every K step re-packs each subblock with packer accumulate).
8x8 bf16 HiFi4, trisc_max µs:

| kt | best | best µs | pick `(1,8)` gap | worst `(1,1)` penalty | util range |
|---|---|---|---|---|---|
| 1 | (8,1) | 4.59 | +0.4% | +13% | 58-66% |
| 2 | (8,1) | 7.71 | +0.6% | +15% | 68-79% |
| 4 | (8,1) | 14.14 | +0.7% | +16% | 74-86% |
| 8 | (8,1) | 27.18 | +0.3% | +17% | 76-89% |

Same subblock preference -- reuse-8 (max-DST) is best, the `(1,8)` pick within
~0.7% -- but two differences from DST-K:

- **The `(1,1)` penalty grows with kt (+13→+17%)**, opposite to DST-K's shrink
  (+11→+2%). L1-K does one acquire per subblock *per K step* (acquires = reuse·kt),
  so the high-reuse cost compounds with kt; DST-K acquires each subblock once
  (acquires = reuse) and amortizes it over kt.
- **L1-K is faster than DST-K for kt >= 2** (best µs: kt=2 7.71 vs 9.02, kt=4 14.14
  vs 18.06, kt=8 27.18 vs 37.60 -- ~15/22/28% lower; tied at kt=1), at higher util
  (kt=8: 76-89% vs DST-K's 63-65%) -- the MB3 DST-K-vs-L1-K result.

So the subblock *choice* is the same for both strategies (max-DST, avoid `(1,1)`);
L1-K just makes it matter more at high kt, and is the faster strategy there.

**Fidelity amplifies the effect (DST-K).** Lower `math_fidelity` makes the kernel less
math-bound, so the subblock (via operand re-unpack) is a bigger share. `(1,1)`
penalty vs best, 8x8:

| kt | LoFi | HiFi2 | HiFi4 |
|---|---|---|---|
| 1 | 38% | 28% | 11% |
| 2 | 40% | 11% | 5% |
| 4 | 29% | 7% | 2% |
| 8 | 26% | 3% | 2% |

util range tracks it: LoFi 20-35% (dataflow-bound), HiFi2 39-51%, HiFi4 59-67%
(math-bound). The pick `(1,8)` stays within ~0-3% of best at all three
fidelities, so fidelity changes how much the choice matters, not whether the
heuristic is good.

**fp32 reshapes the option set (HiFi4, DST-K).** fp32 dest halves DST capacity (8->4),
so the four product-8 shapes drop out: 6 valid subblocks instead of 10, minimum
`reuse` 16 instead of 8, and the pick shifts `(1,8)` -> `(1,4)`. Absolute cost
rises ~5-8% (up to +8% at kt=8) from the fp32 pack plus higher reuse. But at
HiFi4 the spread is unchanged -- the `(1,1)` penalty curve (11/5/3/2% across kt)
matches bf16 and the pick stays within ~1% of best. fp32 changes which
subblocks exist and the pick, not how much the choice matters at this fidelity.

**Takeaway.** On clean shapes the heuristic is near-optimal and subblock
selection is a ~1-2% lever (just avoid `(1,1)`). The effect is amplified by low
fidelity (up to ~40%) and by forced-high reuse (fp32). The discriminating
regime for stress-testing the heuristic is LoFi + fp32 + awkward/prime output
dims (where the heuristic is forced toward high reuse or a single-row fallback);
that combination is the planned next sweep.

### MB5.D: Matmul with a pre-seeded accumulator (scaled-acc) -- the heuristic may under-shoot

`subblock_scaled_acc_sweep.py` (`scaled_acc_compute.cpp`): `out = scale*acc + (a @ b)`.
This is a real matmul with a **pre-seeded accumulator**, not a
matmul-with-epilogue. Per output subblock, `mul_tiles` runs *first* and computes
`scale*acc` into the subblock's DST slots; then a `kt`-step `matmul_block` loop
accumulates `a @ b` onto those same slots (matmul_block adds into DST, so the seed
survives and the `+` is free). The `scale*acc` term is a prologue/bias seed that
*shares* the matmul's accumulator slots by construction, so no separate DST
scratch is needed and the full DST capacity is available for the subblock.
(Contrast MB3.C, a true *epilogue*: there an activation runs on the matmul
*output* after the K-loop.) The benchmark's `compiler_pick` column, by contrast,
*models* a heuristic reserving ~half the DST for that term (`dst_subblock(mt, nt,
cap/2)`), which would cap the subblock at reuse 16. This sweep asks whether that
modeled conservative reservation would cost anything.

mt=8, nt=8, Blackhole bf16 HiFi4, trisc_max µs, PCC ~1.0. The best is the largest
subblock `(1,8)` (reuse 8); the modeled pick is `(1,4)` (reuse 16); the worst is
`(1,1)` (reuse 64):

| kt | best `(1,8)` reuse 8 | modeled pick `(1,4)` reuse 16 | worst `(1,1)` reuse 64 | pick gap | `(1,1)` penalty |
|---|---|---|---|---|---|
| 1 | 13.58 | 14.11 | 16.23 | +3.9% | +20% |
| 4 | 24.37 | 24.80 | 27.09 | +1.7% | +11% |
| 8 | 38.87 | 39.19 | 41.61 | +0.8% | +7% |

- **Bigger subblock wins, as for plain matmul (MB5.C).** Reuse 8 is fastest at
  every kt; reuse 64 `(1,1)` is worst. The penalty shrinks with kt (+20→+11→+7%),
  the same trend as MB5.C -- a deeper matmul amortizes the per-subblock overhead.
- **The win is in the unpack time.** kt=4 unpack drops 19.18 (reuse 64) → 16.48
  (reuse 16) → 15.71 (reuse 8) as reuse falls, while math and pack move little; the
  slowest thread shifts from math (high reuse) to pack (low reuse). Measured
  per-RISC; we are not modeling why yet.
- **If the compiler does reserve cap/2 for the scale*acc seed, it would leave ~1-4% on
  the table** -- most at low kt (+3.9% at kt=1, +0.8% at kt=8). The hand-written
  kernel shows the full-cap reuse-8 subblock is both legal (the `mul` targets the
  matmul accumulator slots, so no scratch is needed) and faster. The open question
  is whether the real heuristic actually under-reserves here; if it does, targeting
  the accumulator slots directly would recover the gap (largest for shallow-K
  fused matmuls). This is a hypothesis to check against the compiler, not a claim
  the benchmark proves -- `compiler_pick` is a model of the heuristic, not its
  observed output.
- **Shape at fixed reuse is a minor effect here** (~1.7%, much smaller than the
  transpose's ~20%): at reuse 8, kt=1, thin `(1,8)`=13.58 / `(8,1)`=13.60 edge
  blocky `(2,4)`=13.69 / `(4,2)`=13.81. The matmul math seems to dominate, so the
  access-stride term that mattered for the transpose is mostly washed out.

This sits between MB3.C and MB5.C (plain-matmul subblock selection), and the
structural difference from MB3.C matters: there the *epilogue* acts on the matmul
output (favoring DST-residency at low kt); here the mul is a *prologue* that seeds
the accumulator. The pre-seed does not change that bigger-is-better, but the
modeled scratch reservation is the lever that could make a capacity heuristic
*pick* smaller than optimal.

### MB5.E: Fused SFPU chain (from tt-lang-generated code) -- math-bound, subblock is a non-lever

Unlike MB5.A-D (hand-written probes), this kernel body is **extracted verbatim
from the tt-lang codegen** for `v = abs(neg(relu(sigmoid(c) + tanh(b))))` (a
compile of a small `@ttl.operation`), then wrapped in the flat, resident,
hop-outside microbench style with a configurable subblock (`sub` = output tiles
per `tile_regs_acquire`). `subblock_fused_chain_sweep.py` /
`fused_chain_compute.cpp`. Blackhole bf16 half-sync, one pass (no iters loop),
trisc_max µs; bold = fastest:

| tiles | sub=1 | sub=2 | sub=4 | best |
|---|---|---|---|---|
| 8   | 18.32  | **18.28**  | 18.34  | 2 |
| 16  | 35.57  | **35.47**  | 35.53  | 2 |
| 32  | 70.11  | **69.81**  | 69.88  | 2 |
| 64  | 139.13 | **138.50** | 138.63 | 2 |
| 128 | **277.17** | 278.90 | 279.16 | 1 |

Structure carried from the codegen (informs the sweep): `init_sfpu` once, then per
op the `*_tile_init` immediately precedes its applies over the chunk (the SFPU is
reconfigured op-by-op, so these inits cannot be hoisted across ops or subblocks);
the fused add is `add_binary_tile` (DST->DST). Because that add keeps both operands
live, **each output tile uses two DST slots** -- so the subblock is bounded by
`2*sub <= cap` (sub in {1,2,4} at half-sync cap 8), the real cap/2 halving MB5.D
modeled.

Measured observations:
- **The chain is math-bound**: `math` is the trisc_max (bottleneck) at every
  config, and the subblock is a **non-lever -- spread <1%** at every tile count.
  The chain costs ~**2.16 µs/tile** (tiles=128: 277/128) for its 6 ops
  (2 copies + sigmoid + tanh + add + relu + neg + abs).
- **Max-DST (sub=4, the full cap-8 budget) is not the winner** -- `sub=2` is
  marginally best across sizes (`sub=1` edges it at tiles=128), same signature as
  single-op exp (MB5.B math-bound end). PCC >= 0.9999.
- This is the compiler's actual op sequence + DST layout, so it confirms the
  subblock choice barely matters for a real fused SFPU chain: max-DST (or cap/2)
  loses <=1%.

Full-sync (cap 16, so `sub` up to 8; trisc_max µs, bold = fastest):

| tiles | sub=1 | sub=2 | sub=4 | sub=8 | best |
|---|---|---|---|---|---|
| 8   | 18.50  | 18.45  | 18.41  | **18.38**  | 8 |
| 16  | 35.95  | 35.76  | 35.76  | **35.74**  | 8 |
| 32  | 70.79  | 70.44  | **70.42**  | 70.49  | 4 |
| 64  | 140.55 | 139.81 | **139.80** | 139.99 | 4 |
| 128 | 280.10 | 281.53 | 281.59 | **279.11** | 8 |

Same math-bound picture (math gates, spread <=~0.7%, PCC >= 0.9999). The only
change vs half-sync: with the doubled DST budget the winner sits in the
`sub=4`/`sub=8` (max-DST) region -- max-DST wins at 3 of 5 sizes -- vs half-sync
where `sub=2` edged it. All noise-level, so the conclusion holds: the subblock is
immaterial for this chain.

### Summary (measured)

- **Matmul (MB5.C/D):** a larger subblock is faster; the only clear loser is the
  degenerate `(1,1)`, which the heuristic never picks. The spread is ~1-2% on clean
  shapes at HiFi4, widening at low fidelity and high reuse (MB5.C), and up to ~20%
  for the scaled-acc kernel, where a modeled cap/2 reservation would under-shoot
  the fastest subblock (MB5.D).
- **Zero-math (MB5.A):** the largest subblock does not always win -- the full-DST
  chunk is worst at small tile counts. Under full-sync (double-buffering off) the
  optimum pins at chunk=4 and the max-DST chunk never wins. The chunk-size curve is
  a broad plateau with reproducible per-chunk bumps.
- **Transpose (MB5.B):** the biggest effect in the suite (~1.47x), from both chunk
  size and a thin-vs-blocky shape preference at fixed product.
- **Fused SFPU chain (MB5.E, from real codegen):** math-bound like exp -- subblock
  is a non-lever (<1% spread), and max-DST is not the winner (`sub=2` marginally
  best). Confirms the subblock choice is immaterial for math-bound compute.

We are not proposing a cost model for these yet. Open work toward one: a per-thread
(unpack / math / pack) timeline to explain the chunk-size curve, non-power-of-two
and larger tile counts beyond what fits L1, and Wormhole.

## Result CSVs (local only, git-ignored; under `benchmarks/microbench/results/`)

- `pack_unpack_blackhole_bf16_*.csv`: MB1 bf16 tiles=1..16.
- `pack_unpack_blackhole_matrix_*.csv`: MB1 config matrix.
- `pack_unpack_wormhole_b0_bf16_*.csv`: MB1 bf16 tiles=1..16.
- `accumulation_blackhole_bf16_{l1,dram}_*.csv`: MB2 add recurrence, DST vs
  L1-pack, acc_tiles x iters (x block_count).
- `accumulation_wormhole_b0_bf16_{l1,dram}_*.csv`: MB2 add recurrence on
  Wormhole (x block_count).
- `accumulation_{blackhole,wormhole_b0}_fp32_{l1,dram}_*.csv`: MB2 add
  recurrence, fp32 dest.
- `accumulation_{blackhole,wormhole_b0}_bf16_{l1,dram}_True_*.csv`: MB2 add
  recurrence, full-sync (`full_sync` in the tag).
- `matmul_k_blackhole_bf16_hifi4_*.csv`: MB3 DST-K vs L1-K, P and kt sweep
  (MB3.A reuse=1 + MB3.B reuse>1).
- `matmul_k_wormhole_b0_bf16_hifi4_*.csv`: MB3 same sweep on Wormhole.
- `matmul_k_blackhole_bf16_hifi4_gelu_*.csv`: MB3.C DST-K vs L1-K with a fused
  GELU epilogue.
- `matmul_k_wormhole_b0_bf16_hifi4_gelu_*.csv`: MB3.C fused GELU on Wormhole.
- `matmul_k_blackhole_bf16_{lofi,hifi2}_none_*.csv`: fidelity for plain matmul
  (MB3.A/B); HiFi4 in the hifi4 CSV.
- `matmul_k_blackhole_bf16_{lofi,hifi2}_gelu_*.csv`: fidelity for fused GELU
  (MB3.C).
- `compute_op_{blackhole,wormhole_b0}_*.csv`: MB4 per-op compute-engine tile costs
  (SFPU unary + FPU binary/bcast/reduce), with the `category`/`out_tiles` columns.
- `matmul_k_wormhole_b0_bf16_{lofi,hifi2}_{none,gelu}_*.csv`: Wormhole fidelity,
  P=16 (plain + fused).
- `matmul_k_{blackhole,wormhole_b0}_fp32_hifi4_none_*.csv`: fp32 dest (cap
  halved).
- `matmul_k_{blackhole,wormhole_b0}_bf16_hifi4_none_True_*.csv`: full-sync (cap
  doubled; `full_sync` now in the CSV tag).
- `subblock_matmul_blackhole_{bf16,fp32}_{lofi,hifi2,hifi4}_*.csv`: MB5.C subblock-selection
  sweep, 8x8, all valid `(sub_mt, sub_nt)` x kt with `compiler_pick` (fidelity and
  fp32 cap-halving variants in the tag).
- `subblock_pack_unpack_blackhole_*.csv`: MB5.A zero-compute chunk sweep -- DST chunk
  per acquire swept at fixed `tiles` (`dst_chunk`/`acquires` columns).
- `subblock_fpu_op_blackhole_bf16_*.csv`: MB5.B FPU-op subblock sweep (`op` ∈
  transpose/add/mul × square `n` × all valid `(sub_h, sub_w)`) with `dst_chunk`/
  `acquires`/`compiler_pick` (dtype/full_sync in the tag).
- `subblock_scaled_acc_blackhole_bf16_*.csv`: MB5.D fused `scale*acc + a@b` subblock sweep,
  all valid `(sub_mt, sub_nt)` at fixed `(mt, nt, kt)` with `reuse`/`compiler_pick`
  (kt is a scalar param, swept across runs).

## Verification

Three references check these benchmarks, each validating a different scope:

1. **Per-op LLK measurements.** The `*BH` sheets in the SDPA optimization workbook:
   UNPACK/MATH/PACK_ISOLATE, L1_TO_L1, and l1_acc for individual stock LLK ops on
   Blackhole. These validate the per-tile weights.
2. **The optimized SDPA kernel.** The deepseek_v3_b1 SDPA micro-op study (its op
   set and dates match the workbook). It is built from custom LLK ops: a custom
   matmul (`sdpa_custom_mm`), custom SFPU reduce/exp, and a block pack. tt-lang
   emits none of those ops. The optimized SDPA kernel bounds achievable
   performance, but it is not a clean cross-check of the stock-op strategy
   choice.
3. **MB1's internal fit.** Per-tile time linear in tile count, steady past
   iters ~ 128, high r^2.

Absolute per-tile CLK is not portable across the per-op measurement sheets: a plain
bf16 pack reads ~138 CLK in the FastPack sheet but ~10 CLK in the Copy sheet (different
normalization conventions). Only within-sheet ratios and structure are compared
below.

### Agrees

- **L1-accumulation pack surcharge.** FastPack BH, steady state: plain pack ~138
  CLK vs l1_acc pack ~193 CLK. That is a +40% surcharge. L1-pack pays this
  per-iteration cost in MB2, and DST-resident avoids it. This confirms the
  surcharge is real and positive on Blackhole.
- **Fidelity scaling.** Binary FPU BH: HiFi2 math per tile is ~2x LoFi (34.7 vs
  16.9 CLK), and add and mul are equal at a given fidelity; Matmul BH HiFi2 math is
  ~34 CLK/tile. Consistent with MB3's fidelity sweep and the small math share of
  the per-iteration cost.

### Disagrees

- **Serial composition predicts the wrong additive strategy.** A serial sum of
  the per-engine measurements predicts DST-resident as cheaper for the
  additive recurrence because DST packs once, while L1-pack pays N packs with the
  +40% surcharge. MB2 measures L1-pack as faster wall-clock time. The cause is
  pipelining: L1-pack's per-iteration pack overlaps the next iteration's
  unpack/math across RISCs. A serial per-engine sum cannot represent that
  overlap.
- **The optimized SDPA keeps the matmul output DST-resident. MB3 measures L1-K as
  faster for stock `matmul_block`.** The SDPA kernel acquires DST once across all
  chunks and packs the output once through its custom matmul. It does not use
  `pack_reconfig_l1_acc`. MB3 measures stock `matmul_block`; for that stock op
  sequence, DST-K is slower than L1-K at kt >= 2. These results do not
  contradict each other. SDPA's custom matmul and pack change the per-op costs,
  so DST-residency wins there. Within the stock ops tt-lang emits, MB3 measures
  L1-K as faster. The SDPA result bounds how far a custom matmul could change the
  choice; it does not validate the stock-op strategy choice.
- **Surcharge magnitude.** FastPack's +40% exceeds the +30-35% prior estimate.
  This is within run-to-run and normalization variance, but the directly measured
  value should be used.
- **No Wormhole cross-check.** The workbook is Blackhole-only. The Wormhole
  results have no per-engine reference, so they rest on the MB measurements plus
  the extrapolated Wormhole/Blackhole ratios. The Wormhole-specific conclusions
  are that full-sync makes DST-resident faster for the additive strategy and
  that the ~2.4x costlier pack sets the gelu margins.

## Open / next

MB1, MB2 (including the `--expr` pointwise expressions), and MB3 are complete on
both architectures (see their sections). Remaining:

- MB4: compute-op math microbenchmarks. Measure per-op SFPU/FPU tile costs, math-thread
  (what tt-lang emits today) and a pack-thread activation arm (achievable headroom).
- MB1: distinct-DFB superlinearity. Sweep the number of live DFBs to check whether
  the per-handoff cost stays constant or grows under L1 / semaphore pressure.
- MB2: compute-bearing recurrence beyond pointwise. The `--expr mul|gelu` sweep
  is complete on both architectures (see the per-iteration expression results
  under MB2). Remaining: reduction/broadcast sequences representative of
  softmax-like code, including max/sum reductions, broadcasted scalar or
  row-vector updates, and fused pointwise transforms. New MB2 CSVs include the
  expression tag after the source tag.
- Validation: LLK composition cross-check. Blackhole is checked against the SDPA
  optimization workbook's per-engine measurements (see Verification). Wormhole
  still needs the same per-engine measurements, which the workbook does not
  cover.
