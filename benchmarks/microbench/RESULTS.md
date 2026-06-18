# Microbench results -- working notes

A record of what the microbenchmarks measure on hardware. Measurements only;
interpretation against any cost model lives elsewhere.

## Motivation -- why not just use the LLK perf microbenchmarks?

The tt-metal tt-llk perf suite already measures per-engine tile costs (unpack,
pack, matmul/eltwise math, L1-accumulation pack surcharge) in isolation, and those
supply most of the cost-model weights directly. These benchmarks exist for what the LLK
microbenchmarks cannot give:

- **Composition.** The generated kernel runs unpack/math/pack pipelined across
  three RISCs; the LLK microbenchmarks time each engine alone, so adding them up
  does not reproduce the composed cost. Blackhole bf16, per tile:
  - LLK unpack ~ 0.030 + LLK pack ~ 0.026 = **0.056 µs** (serial sum)
  - slowest single engine = **0.030 µs** (perfect-overlap lower bound)
  - MB1 measured, over DFBs = **0.039 µs**

  The measured value sits between the two -- the kernel pipelines unpack and pack
  across RISCs, recovering ~30% the serial sum can't see. So summing overshoots by
  ~1.4x, with no constant overlap factor across configs (it varies by op and tile
  count).
- **Dataflow-buffer handoff.** The LLK harness has no dataflow buffers, so it never
  measures the DFB reserve/wait/push/pop + cross-thread sync the dispatched kernels pay
  (MB1's fixed term, ~0.09 µs/iter on Blackhole) -- the cost model's
  `dfbHopFixedCost`.
- **The strategy decision, not just its inputs.** The model chooses DST-resident vs
  L1-pack. The LLK microbenchmarks can only feed the model's serial formula; the benchmarks
  measure the actual ranking on the generated sequence. MB2 found L1-pack marginally
  faster than DST for additive accumulation, where the serial model predicts DST
  is always lower-cost -- a calibration error not observable from the LLK
  microbenchmarks alone.

The LLK microbenchmarks parameterize the model; these benchmarks check whether it is
calibrated.

## Methodology (brief)

- Handwritten C++ kernels (compute + reader/writer), modeled on the tt-metal
  tt-llk perf benchmarks, but run through **tt-metal dispatch**
  (`ttnn.generic_op`) over dataflow buffers -- so DFB reserve/wait/push/pop +
  cross-thread sync are part of the measurement (the bare-metal LLK harness omits
  them).
- **No tt-lang compiler involved:** kernels are handwritten and JIT-compiled by
  tt-metal at run time; the sequences are matched by hand to what tt-lang emits.
- Single compute core; inputs L1-resident where possible.
- Timing: `DeviceZoneScopedN` per RISC; cycles / profiler `CHIP_FREQ[MHz]` = µs.
- Correctness: PCC vs a torch reference.
- DFB block count (multi-buffering depth, tt-lang's term) is held at the default
  (`block_count = 2`) in the headline comparisons. It is exposed as `--block-count`
  and swept once (the MB2 block-count block): it changes only absolute streamed
  cost via reader prefetch, not the strategy ranking, so the other tables fix it to
  isolate the strategy comparison. A block count of 1 would understate steady-state
  throughput; above 2 adds nothing here (compute-bound).

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
benchmarks.microbench.<sweep> ...`. No tt-lang compiler build is needed -- the
benchmarks dispatch handwritten kernels via `ttnn`/`generic_op`.

## MB1 -- pack/unpack probe

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

### Blackhole config matrix -- trisc_max µs/iter

| dtype | full_sync | fp32_acc | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|---|---|
| bf16 | 0 | 0 | 0.107 | 0.161 | 0.252 | 0.441 |
| bf16 | 1 | 0 | 0.106 | 0.159 | 0.251 | 0.439 |
| bf16 | 0 | 1 | 0.113 | 0.164 | 0.255 | 0.388 |
| fp32 | 0 | 1 | 0.131 | 0.196 | 0.310 | 0.450 |
| fp32 | 1 | 1 | 0.129 | 0.194 | 0.307 | 0.539 |

- fp32 dest raises per-tile cost (T=4: fp32 0.310 vs bf16 0.252). full_sync has
  little effect at these tile counts.

## MB2 -- accumulation (DST-resident vs L1-pack)

Summary: out = initial + sum of `iters` contributions on an accumulator of
`acc_tiles` tiles, run two ways with the same reader, timing each strategy's loop.
Seed (dfb_init) and contributions (dfb_delta) use separate DFBs.

```
# DST-resident (acc_dst.cpp): accumulator stays in DST, pack once.
# Each contribution is added in place with binary_dest_reuse_tiles<ELWADD,
# DEST_TO_SRCA> -- the op tt-lang's tile_accumulate_add lowers to.
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
- **Strategy** (the comparison): where the running accumulator lives -- DST
  registers (DST-resident) vs an L1 buffer (L1-pack).
- **Contribution residency** (`--source l1|dram`, orthogonal): contributions
  re-read from one L1 block (`l1`, isolates compute-thread cost) vs one block
  streamed per iteration from DRAM (`dram`, which adds the per-iteration DRAM read
  into the compute-thread zone -- still TRISC zone time, not full dispatch time).
  This sets the absolute cost; it does not change the strategy ranking.

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

- **Strategy ranking:** L1-pack is lower than DST-resident in nearly every config,
  by a small margin (~0.1-0.6 µs, largest at l1-resident acc_tiles=4 iters=16:
  3.86 vs 3.24). The acc_tiles=1 cells are near-ties -- a few favor DST-resident by
  <=0.03 µs. The two strategies are close; L1-pack is marginally ahead.
- **Timed consistently with MB3:** both kernels time the pack inside the zone --
  L1-pack its per-step packs, DST-resident its single final pack. The single DST
  pack is small next to the accumulate loop (unlike MB3's matmul, where the
  pack-once is a heavy serial tail), so MB2's numbers and the L1-pack-ahead
  ranking are unchanged by timing it.
- **Residency effect (orthogonal):** streaming from DRAM adds a per-iteration DRAM
  read, so absolute cost grows steeply with iters (acc_tiles=1 iters=16: ~7.4 µs
  streamed vs ~1.4 µs resident, ~5x). This is larger than the strategy difference
  and independent of it -- the DST-vs-L1-pack ranking holds under both residencies.
- **Notes:**
  1. The DST kernel uses `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` with a
     `copy_tile` seed and `binary_op_init_common` -- the op sequence tt-lang's
     `tile_accumulate_add` lowers to. Seed (dfb_init) and contributions (dfb_delta)
     use separate DFBs, as in tt-lang (`initial_dfb`/`delta_dfb`); optimized
     production kernels likewise use separate seed and contribution DFBs.
  2. tt-lang's DST accumulation is verified numerically correct on hardware
     (refactor branch `tensor_recurrence_dst_acc.py`, PASS).
  3. Thread overlap: DST-resident holds one acquire with the pack thread idle
     until the final pack; L1-pack pipelines unpack + pack per iteration across
     two threads -- a structural difference between the strategies.

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
run-to-run variance (DST/L1 within ~5%) -- at the highest iter count the two
strategies converge on Wormhole. The ranking is otherwise architecture-independent.

### Block count (DFB depth)

`--block-count` (default 2) sets the contribution/output DFB block count -- how
many blocks the reader can prefetch. dram-streamed, acc_tiles=2, L1-pack
trisc_max µs:

| architecture, iters \ block_count | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Blackhole, iters=8  | 4.86 | 4.27 | 4.19 | 4.18 |
| Blackhole, iters=16 | 9.32 | 7.95 | 8.24 | 7.99 |
| Wormhole, iters=8   | 6.13 | 4.93 | 4.89 | 4.91 |
| Wormhole, iters=16  | 11.65 | 9.28 | 9.31 | 9.37 |

Block count 1 serializes each DRAM read against compute; block count 2 lets the
reader prefetch the next contribution, cutting ~15-20%; beyond 2 it is flat
(compute-bound). l1-resident is block-count-insensitive -- one re-read block, no
streaming (BH iters=16 ~2.1 µs, WH ~3.1 µs across block counts 1-8). Block count
changes the dram-streamed absolute cost, not the DST-vs-L1 ranking, since both
strategies share the contribution buffer. The cost model's DFB term should
distinguish single- from multi-block streaming; the strategy choice does not
depend on it.

### fp32 dest and full-sync

**fp32 dest (dtype = fp32).** Halves DST capacity (4 vs 8), so the DST-resident
strategy is legal only for acc_tiles <= 4, and roughly doubles pack cost. The
ranking matches bf16 -- L1-pack lower in nearly every config; the costlier pack
does not shift it to DST-resident (Blackhole l1-resident acc_tiles=4 iters=16:
L1 ahead ~16%; Wormhole near-tied at high iters, as in bf16). Exact (PCC ~ 1.0).

**Full-sync (dst_full_sync_en) -- flips the choice on Wormhole.** Full-sync makes
DST a single bank (no math/pack double-buffer), so L1-pack's per-iteration copy
can no longer overlap the previous iteration's pack -- it loses its pipelining.
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
and DST-resident takes the lead at iters >= 4 for acc_tiles >= 4. So for the
additive recurrence, full-sync is a strategy-selection input on Wormhole -- it
makes DST-resident lower-cost -- but not on Blackhole.

### Per-iteration expression (--expr add / mul / gelu)

`--expr` sets the per-iteration contribution applied before accumulation:

- `add`: contribution = delta. DST-resident accumulates in place with
  binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>; L1-pack packs with
  pack_reconfig_l1_acc. This is the additive recurrence the rest of MB2 measures.
- `mul`: contribution = delta*delta (FPU). L1-pack only -- a product cannot
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

For the additive contribution the two strategies are close: L1-pack is ~25-33%
ahead on Blackhole across iters, and on Wormhole the gap closes as iters grows --
DST-resident packs once while L1-pack pays Wormhole's ~2.4x costlier per-iteration
pack -- reaching a tie at acc_tiles=4 iters=16 and a small DST-resident lead at
acc_tiles=1 iters=16 (2.14 vs 2.32). For gelu, L1-pack is decisively lower-cost on
both architectures and the margin widens with iters (Blackhole 1.28x at iters=1 to
1.71x at iters=16; Wormhole 1.34x to 1.65x), because DST-resident accumulation of a
non-additive contribution pays an explicit per-iteration add_binary_tile on top of
the gelu, while L1-pack folds the accumulation into the packer. PCC ~ 1.0 for
add/mul; ~0.99 for gelu (tanh approximation).

The trend: the more compute-bearing or non-additive the per-iteration
contribution, the more L1-pack is favored. DST-resident is competitive only for the
bare additive recurrence, and even there only on Blackhole (cheap pack) and at low
iteration counts on Wormhole.

## MB3 -- matmul K-accumulation (DST-K vs L1-K)

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
table here measured DST-K with its single pack *outside* the zone -- math only --
which made DST-K look ~math-bound and always cheaper; corrected below).

### MB3.A -- output fits DST (P <= capacity, reuse = 1)

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

### MB3.B -- output exceeds DST (P > capacity, reuse > 1)

The output no longer fits DST, so it is subblocked. DST-K processes one subblock
at a time across the whole K loop, re-unpacking that subblock's A rows and B cols
from the resident operands -- operand unpack scales by reuse. L1-K still repacks
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
- **For any K-accumulation (kt >= 2) L1-K is faster, and the margin grows with
  both kt and P (reuse)** -- from ~2% (P=1, kt=2) to ~39% (P=32, kt=8). This holds
  in both regimes; reuse > 1 widens the gap but does not change the direction.
- **Why, despite L1-K moving far more tiles** (e.g. P=16, kt=8: L1-K packs 128
  tiles to DST-K's 16): DST-K cannot pack until all `kt` matmuls finish, so its
  pack engine is idle through the K loop and the pack runs as a serial tail after
  the math. L1-K interleaves matmul and pack per K step, overlapping them across
  the unpack/math/pack RISCs. The overlap recovered outweighs the extra packs.
- **Consequence for the #652 selector:** a data-movement score (tiles packed *
  per-tile cost) ranks DST-K cheaper in every cell, the opposite of measured
  wall-clock. Ranking matmul K-accumulation needs a model term for the pack/math
  overlap L1-K gets and DST-K cannot -- not just the per-engine tile weights.
- **Scope:** isolated single kernel. It omits cross-op DST occupancy -- a matmul
  that holds its output resident in DST starves fused neighbors (e.g. a softmax
  sharing DST), which pushes toward L1-K. That is distinct from the in-kernel
  epilogue fusion measured in MB3.C, which favors DST-K at low kt; the two
  fusion effects point opposite ways. It also omits `matmul_block` granularity.

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
WH 12.31 -> 12.50 -> 12.51 -- flat) because it is pack-bound; its per-K-step packs
hide the math. DST-K serializes math then one pack, so it scales with fidelity
(BH 11.11 -> 12.83 -> 15.81, WH 14.86 -> 16.27 -> 19.77). The L1-K margin grows with
fidelity (~5-11% LoFi to ~21-35% HiFi4) but the ranking is unchanged: L1-K is
lower for kt >= 2 at every fidelity, in both MB3.A (reuse=1) and MB3.B. For plain
matmul, fidelity changes the size of L1-K's lead but not which strategy is
lower-cost.

**fp32 dest (dtype = fp32).** fp32 accumulation halves DST capacity (4 vs 8), so
reuse > 1 is reached at smaller outputs, and roughly doubles pack cost (2x tile
bytes). P = 16 (4x4) is reuse 4 here (vs reuse 2 at bf16). trisc_max µs **DST-K /
L1-K (faster)**, all PCC = 1.0:

| architecture \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Blackhole | 2.00/2.00 (=) | 4.05/3.27 (L1) | 8.29/6.02 (L1) | 18.27/12.94 (L1) |
| Wormhole  | 3.62/3.57 (L1) | 6.67/5.69 (L1) | 13.21/10.03 (L1) | 26.17/18.26 (L1) |

Ranking unchanged: L1-K lower for kt >= 2 on both architectures, exact (PCC 1.0). The
margin is slightly narrower than bf16 (BH P=16 kt=8: 29% vs 35%) -- fp32's costlier
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

Full-sync extends the reuse-1 range to larger P but does not change the ranking or
the margin (BH P=16 kt=8: 34% with full-sync at reuse 1 vs 35% at default reuse 2)
-- L1-K is lower for kt >= 2. L1-K's advantage comes from pipelining pack against
the next K step's matmul across the unpack/math/pack RISCs, which is independent of
the DST bank mode, so full-sync does not recover it for DST-K.

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

- **Blackhole vs Wormhole:** the ranking is identical -- tied at kt=1, L1-K
  lower for kt >= 2, margin growing with kt and reuse. Wormhole absolute times are
  ~1.1-1.4x Blackhole (slower clock; costlier pack). The per-engine pack ratio
  (Wormhole pack ~ 2.4x Blackhole, which loads L1-K's extra packs more heavily)
  does not change the lower-cost strategy: the matmul/pack overlap dominates on
  both architectures, so the #652 decision for matmul K-accumulation is architecture-independent
  in the isolated case.

### MB3.C -- fused GELU epilogue (`--fuse gelu`)

A GELU activation is applied to the matmul output. DST-K applies it in place on
the resident DST subblock before its single pack (no reload). L1-K must reload
its L1 accumulator into DST, apply GELU, and pack again -- the round trip the
resident output lets DST-K skip. Fast (tanh) GELU is used, the production default
for matmul fusion; its approximation lowers PCC to ~0.985 (accepted for fused
matmul+activation), versus ~1.0 for plain matmul. Blackhole bf16 HiFi4,
trisc_max µs **DST-K / L1-K (faster)**:

| P (mtxnt) \ kt | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 (1x1) | 1.29/1.43 (DST) | 2.09/2.23 (DST) | 3.90/3.92 (DST) | 7.95/7.66 (L1) |
| 4 (1x4) | 1.99/2.35 (DST) | 3.10/3.25 (DST) | 5.46/5.13 (L1) | 10.33/9.20 (L1) |
| 8 (2x4) | 2.80/3.31 (DST) | 4.26/4.35 (DST) | 6.99/6.35 (L1) | 12.96/10.68 (L1) |
| 16 (4x4) reuse 2 | 4.53/5.15 (DST) | 6.62/6.34 (L1) | 10.26/8.43 (L1) | 18.51/13.58 (L1) |

- **The epilogue makes DST-K the lower-cost strategy at low kt.** Without it,
  L1-K is faster for every kt >= 2 (MB3.A/B). With it, DST-K is faster at kt = 1
  (all P) and at kt = 2 for P <= 8. L1-K's matmul/pack overlap advantage over
  DST-K grows with kt (the same effect that wins MB3.A/B). The epilogue adds to
  L1-K a one-time reload + repack of the output (~2P tile movements, independent
  of kt) that DST-K avoids by applying GELU in place. At small kt that fixed
  reload cost exceeds the overlap advantage, so DST-K is cheaper; as kt grows the
  overlap advantage exceeds it and L1-K is cheaper.
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
  thread pure matmul and would make DST-K cheaper still -- strengthening, not
  reversing, the result below. tt-lang does not emit that today, but it is not a
  hardware limit: it needs a pack-thread TTKernel op + lowering (or direct EmitC).
  MB4 currently measures the math-thread SFPU ops; a pack-thread arm to quantify
  this headroom is planned.
- **Consequence for the #652 selector:** with a fused epilogue the resident-output
  advantage holds and DST-K is the lower-cost strategy for shallow-K matmuls;
  the plain-matmul preference for L1-K (overlap) only holds at larger kt. The
  decision is epilogue-aware, not just kt/reuse-based.

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
  preference is unchanged. So an epilogue tilts the decision toward DST-K more on
  Wormhole than on Blackhole -- the one place so far where the architecture matters to the
  ranking.

**Fidelity (fused).** For plain matmul fidelity changes only the size of L1-K's
lead, not which strategy wins; with the epilogue it changes which strategy wins --
it moves the crossover kt. P = 16 (4x4, reuse 2), fused GELU, trisc_max µs **DST-K
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

DST-K's favorable kt range extends as fidelity drops -- Blackhole: through kt = 4
(LoFi), kt = 2 (HiFi2), kt = 1 (HiFi4); Wormhole: kt = 4 (LoFi), kt = 2 (HiFi2 and
HiFi4). Lower fidelity gives L1-K less math to hide its reload + repack behind, so
the epilogue penalty dominates over more kt. Wormhole stays DST-favorable through
kt = 2 even at HiFi4 -- one step further than Blackhole -- because its costlier pack
makes L1-K's repack dearer. With an epilogue, fidelity is part of the decision;
without one it is not.

## MB4 -- compute-op (math) microbenchmarks

The data-movement benchmarks (MB1-MB3) have no compute-engine term. SDPA/flash is
compute/SFPU-bound, where each math op is best characterized as INIT / KERNEL /
TILE_LOOP with a per-RISC split. MB4 measures per-op compute-engine tile costs to
feed a compute-aware model (also what the nonadditive work needs).

**Implemented (`compute_sweep.py`): math-thread SFPU unary.** A selected op is
applied to `tiles` tiles `iters` times; `op=copy` is the baseline (subtract it for
the SFPU op's marginal math cost). Preliminary, Blackhole bf16, tiles=4, math
thread, marginal µs/tile (over the copy baseline ~0.03):

| op | exp | gelu (fast) | recip | sqrt | rsqrt |
|---|---|---|---|---|---|
| µs/tile | 0.42 | 0.14 | 0.88 | 0.64 | 0.73 |

PCC is the real SFPU approximation accuracy (sqrt/rsqrt ~0.9994, gelu ~0.9997).
Pending: full sweep (tiles x bf16/fp32 x init-hoist, both architectures), a
pack-thread activation arm (the MB3.C headroom), and reduce/binary/bcast ops.

Planned coverage:

| engine | ops |
|---|---|
| FPU (matrix) | `matmul_tiles`, `matmul_block` (x kt), matmul init |
| SFPU (vector) | `exp` (fast/slow), `sub`/`sub_exp`, `add_block`, `mul_tiles_bcast`/`mul_block_bcast` (x broadcast Col/Row/Scalar), inplace variants |
| reduce | FPU fast-reduce, SFPU reduce (`reduce_max`, `reduce_sum`) + reduce init |
| copy | `copy_tile` |
| pack | fast vs slow pack, full vs half DST-bank util, pack-with-acc |

Options not yet covered:

- `math_fidelity` (LoFi / HiFi2 / HiFi4), `fp32_dest_acc`, `unpack_to_dest`,
  `broadcast_type`.
- **Init cost measured separately from the loop** (a major SDPA lever:
  "compressed inits", "re-inits", "hoist inits") -- mirror the INIT vs TILE_LOOP
  marker split.
- fast-pack vs slow-pack; full vs half DST bank.
- TRISC cross-op overlap (the xlsx "overlapped" rows).

Method: same harness (handwritten compute kernel via `generic_op`,
`DeviceZoneScopedN` per op-loop, per-RISC µs), sweep `tile_cnt` + the parameters;
report per-op TILE-LOOP µs/tile and init µs, mirroring the xlsx's
INIT/KERNEL/TILE_LOOP + per-RISC columns.

## Result CSVs (local only, git-ignored; under `benchmarks/microbench/results/`)

- `pack_unpack_blackhole_bf16_*.csv` -- MB1 bf16 tiles=1..16
- `pack_unpack_blackhole_matrix_*.csv` -- MB1 config matrix
- `pack_unpack_wormhole_b0_bf16_*.csv` -- MB1 bf16 tiles=1..16
- `accumulation_blackhole_bf16_{l1,dram}_*.csv` -- MB2 add recurrence, DST vs L1-pack, acc_tiles x iters (x block_count)
- `accumulation_wormhole_b0_bf16_{l1,dram}_*.csv` -- MB2 add recurrence on Wormhole (x block_count)
- `accumulation_{blackhole,wormhole_b0}_fp32_{l1,dram}_*.csv` -- MB2 add recurrence, fp32 dest
- `accumulation_{blackhole,wormhole_b0}_bf16_{l1,dram}_True_*.csv` -- MB2 add recurrence, full-sync (`full_sync` in the tag)
- `matmul_k_blackhole_bf16_hifi4_*.csv` -- MB3 DST-K vs L1-K, P and kt sweep (MB3.A reuse=1 + MB3.B reuse>1)
- `matmul_k_wormhole_b0_bf16_hifi4_*.csv` -- MB3 same sweep on Wormhole
- `matmul_k_blackhole_bf16_hifi4_gelu_*.csv` -- MB3.C DST-K vs L1-K with a fused GELU epilogue
- `matmul_k_wormhole_b0_bf16_hifi4_gelu_*.csv` -- MB3.C fused GELU on Wormhole
- `matmul_k_blackhole_bf16_{lofi,hifi2}_none_*.csv` -- fidelity for plain matmul (MB3.A/B); HiFi4 in the hifi4 CSV
- `matmul_k_blackhole_bf16_{lofi,hifi2}_gelu_*.csv` -- fidelity for fused GELU (MB3.C)
- `compute_op_{blackhole,wormhole_b0}_*.csv` -- MB4 per-op SFPU math-engine tile costs
- `matmul_k_wormhole_b0_bf16_{lofi,hifi2}_{none,gelu}_*.csv` -- Wormhole fidelity, P=16 (plain + fused)
- `matmul_k_{blackhole,wormhole_b0}_fp32_hifi4_none_*.csv` -- fp32 dest (cap halved)
- `matmul_k_{blackhole,wormhole_b0}_bf16_hifi4_none_True_*.csv` -- full-sync (cap doubled; `full_sync` now in the CSV tag)

## Verification

Three references check these benchmarks, each validating a different scope:

1. **Per-op LLK measurements** -- the `*BH` sheets in the SDPA optimization workbook:
   UNPACK/MATH/PACK_ISOLATE, L1_TO_L1, and l1_acc for individual stock LLK ops on
   Blackhole. These validate the per-tile weights.
2. **The optimized SDPA kernel** -- the deepseek_v3_b1 SDPA micro-op study (its op
   set and dates match the workbook). It is built from custom LLK ops: a custom
   matmul (`sdpa_custom_mm`), custom SFPU reduce/exp, and a block pack -- none of
   which tt-lang emits. It bounds the achievable kernel but is not a clean
   cross-check of the stock-op strategy choice.
3. **MB1's internal fit** -- per-tile time linear in tile count, steady past
   iters ~ 128, high r^2.

Absolute per-tile CLK is not portable across the per-op measurement sheets: a plain
bf16 pack reads ~138 CLK in the FastPack sheet but ~10 CLK in the Copy sheet (different
normalization conventions). Only within-sheet ratios and structure are compared
below.

### Agrees

- **L1-accumulation pack surcharge.** FastPack BH, steady state: plain pack ~138
  CLK vs l1_acc pack ~193 CLK -- a +40% surcharge, the per-iteration cost L1-pack
  pays in MB2 and DST-resident avoids. Confirms the surcharge is real and positive
  on Blackhole.
- **Fidelity scaling.** Binary FPU BH: HiFi2 math per tile is ~2x LoFi (34.7 vs
  16.9 CLK), and add and mul are equal at a given fidelity; Matmul BH HiFi2 math is
  ~34 CLK/tile. Consistent with MB3's fidelity sweep and the small math share of
  the per-iteration cost.

### Disagrees

- **Serial composition predicts the opposite additive ranking.** Composing the
  per-engine measurements serially -- and the serial cost model -- makes
  DST-resident lower-cost for the additive recurrence (DST packs once; L1-pack pays
  N packs, each +40%). MB2 measures L1-pack lower-cost. The cause is pipelining:
  L1-pack's per-iteration pack overlaps the next iteration's unpack/math across
  RISCs, which a serial per-engine sum cannot represent. The per-engine
  measurements parameterize the model; the composed benchmark gives the ranking --
  the calibration gap it exists to reveal.
- **The optimized SDPA keeps the matmul output DST-resident; MB3 ranks L1-K lower
  for stock matmul_block.** The SDPA kernel acquires DST once across all chunks and
  packs the output once -- no `pack_reconfig_l1_acc` anywhere -- through its custom
  matmul: the DST-K strategy MB3 found costlier for stock `matmul_block` at
  kt >= 2. These do not contradict. SDPA's custom matmul and pack change the per-op
  costs, so DST-residency wins there; within the stock ops tt-lang emits, MB3's
  L1-K ranking holds. The SDPA result bounds how far a custom matmul could change
  the choice -- it does not validate the stock-op ranking.
- **Surcharge magnitude.** FastPack's +40% exceeds the +30-35% prior estimate --
  within run-to-run and normalization variance, but the directly measured value
  should be used.
- **No Wormhole cross-check.** The workbook is Blackhole-only. The Wormhole results
  -- where full-sync makes DST-resident the lower-cost additive strategy and the
  ~2.4x costlier pack sets the gelu margins -- have no per-engine reference; they rest
  on the MB measurements plus the extrapolated Wormhole/Blackhole ratios.

## Open / next

MB1, MB2 (including the `--expr` pointwise expressions), and MB3 are complete on
both architectures (see their sections). Remaining:

- MB4 -- compute-op (math) microbenchmarks: per-op SFPU/FPU tile costs, math-thread
  (what tt-lang emits today) and a pack-thread activation arm (achievable headroom).
- MB1 -- distinct-DFB superlinearity: sweep the number of live DFBs to check whether
  the per-handoff cost stays constant or grows under L1 / semaphore pressure.
- MB2 -- compute-bearing recurrence beyond pointwise: the `--expr mul|gelu` sweep
  is complete on both architectures (see the per-iteration expression results
  under MB2). Remaining: reduction/broadcast sequences representative of
  softmax-like code -- max/sum reductions, broadcasted scalar or row-vector
  updates, and fused pointwise transforms. New MB2 CSVs include the expression
  tag after the source tag.
- Validation -- LLK composition cross-check: done for Blackhole against the SDPA
  optimization workbook's per-engine measurements (see Verification). Remaining: the
  same per-engine measurements for Wormhole, which the workbook does not cover.
