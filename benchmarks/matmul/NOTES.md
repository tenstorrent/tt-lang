# Matmul benchmark: findings and gotchas

Non-obvious things we learned while tuning the ksplit/SUMMA matmul against
`ttnn.matmul`. Not a user guide; intended as a cache of surprise.

## Planner / scoring

- **Exclusively tested and tuned on single Blackhole card with 130 cores.**
- **Block dims > 8 regress sharply, even with plenty of L1.** The
  profile is bimodal: when core utilization is good, pushing bm/bn/bk up
  toward 8 is monotonically profitable, but the moment any single dim
  crosses 8, wall time jumps ~30%. 4k³ `(16,2,16)/(8,13,1)` 104c was
  model-predicted ~33% faster than the chosen plan but ran ~25% slower.
  Strongly suggests sub-block decomposition inside tt-lang is producing
  lopsided sub-blocks past 8 (e.g. 10 → 8+2). Worth testing exactly
  `bm=bn=bk=16` to see whether even sub-blocks regress; if they don't,
  this is a sub-block-picker issue, not an intrinsic LLK limit. Search
  is bounded to 8 until that investigation lands.
- **Non-power-of-two block dims matter.** `(7,6,5,3)` hit exact tile
  divisors for 2.5k (80 tiles) and 3.3k (104 tiles); without them those
  shapes fall back to worse-padded plans.
- **α (per-block overhead) = 64 tile-matmul units**, refit from empirical.
  At α=64, `bv/(bv+α)` = 0.8 for bv=256 vs 0.89 for bv=512, which is what
  lets bn=4 variants win when pad/cores are equal. α=128 (initial guess)
  over-penalized small-bv and picked the wrong plan on 4k³.
- **β (per-gather cost) scales with `(Kp-1) × iter_per_core`, not
  `(Kp-1)`.** A single linear-in-Kp penalty couldn't satisfy both
  calibration points simultaneously (4k³ Kp=2 losing by 7% @ iter=6, vs
  2k×8k×2k Kp=4 winning by 12% @ iter=3). Scaling by total gather
  *events* resolved both at β≈0.035.
- **MAX_PAD = 1.25** is conservative but load-bearing. Baseline's 4k³
  pad of 1.29 at 110c gave only noise-level improvement; the pad penalty
  we were avoiding was smaller than the variance. Pushing past ~1.5
  (i.e. 50% padding) clearly regressed, so the cap is defensible even
  if 1.25 vs 1.30 is near-indistinguishable in practice.
- **Cores-first scoring** (throughput, block_vol, -pad, cores, -Kp as
  tiebreakers). Block volume beats pad as a secondary because compute
  density matters more than a few percent of wasted tiles.
- **Hand-picked `SHAPE_PLANS` still beat the heuristic.** We have not
  fully explained why. The scoring model reproduces the empirical
  ordering on most shapes but picks noticeably worse plans on enough
  of them that the override table is load-bearing. Something we are
  not modeling is either sub-block layout, mcast geometry, per-core
  NOC contention, or how iter_per_core interacts with the compute
  pipeline. Needs further investigation; the long-term goal should be
  to delete `SHAPE_PLANS`.

## ksplit / SUMMA kernel

- **Kp > 2 rarely wins once pad and iter_per_core are respected.** Higher
  Kp shrinks each core's K-slab, which helps when iter_per_core is small
  and the gather is cheap. In practice only 2k×8k×2k likes Kp=4.
- **Ksplit is not universally faster than SUMMA.** 4k³ with Kp=1 at
  `(8,4,8)/(8,12,1)` 96c (1.704ms) beats Kp=2 at `(8,8,8)/(8,6,2)` 96c
  (1.820ms). The gather cost is larger than the compute savings from
  halving the K-slab at that working-set size.
- **partial_cb ping-pong is a workaround**, not the intent. The natural
  form would be `for _ in range(Kp-1): p += recv_cb.wait()` followed by a
  single `out_cb.reserve/store`. Blocked by
  [issue #527](https://github.com/tenstorrent/tt-lang/issues/527):
  loop-reassignment drops the `+=`. The ping-pong pattern sidesteps it.
- **PCC breaks under Kp=2 for a cluster of shapes.** 2k³, 2k×4k×2k,
  2k×8k×2k all produce NaN output with the planner's Kp=2 choice. Known
  issue, deferred. The faster shapes that land on Kp=2 (e.g. 1k³) have
  perfect PCC, so it correlates with the slower ones in that set, not
  with Kp=2 itself. Likely an accumulation-timing or synchronization
  bug that shows up only past some per-core compute duration.
- **Mcast geometry matters (but DM balancing is an open question).** A
  row-mcasts and reduce-net traffic are both horizontal (same-row,
  different-column). When both ride dm_read they contend on the same
  NOC, and moving reduce to dm_write puts it on NOC1 alongside the
  vertical B mcasts and the output write. Some balanced variants
  measure slightly faster, but the delta is marginal and not yet
  consistent enough to call it a proven win.

## ttnn comparison

- **ttnn.matmul at 4k³ reports ~3.5 MB DRAM read per run, which should
  not be possible.** Inputs alone are ~64 MB. Our 4k³ runs at 88 cores,
  reads ~55 MB, hits ~72 GB/s, and is compute-bound at the Wormhole
  ceiling. ttnn's reported figure is at least 15× too low for the input
  size; either the profiler is missing traffic (mcasts, L1-staged
  transfers, something else), or ttnn is reusing data across programs
  in a way the per-run counter doesn't see. Open question, worth
  instrumenting more carefully before drawing conclusions.

## Best/worst shapes

- **Biggest wins are long-K shapes.** 2.5k×32k×3.3k at ratio ~0.85. Tall
  K amortizes per-block and gather overhead across many K-iterations.
- **1k×16k×2.5k** (tall K, narrow M×N) is also favorable.
- **Square 4k³** is the hard one. Anecdotally 4k³ is a primary shape ttnn benchmarks and tunes against, so
  their curve is likely overfit here. However, it's also one of the lowest core-utilization shapes.

## Methodology

- **Warmup 3 untimed, 5 timed, min(times)**, 10 ms sleep between.
  Run-to-run stability was good enough that config comparisons held up
  across repeats, but we never measured the variance precisely.
- **FP32 dest-accumulate on, HiFi4.** All our numbers assume this.

## TTL / compiler gotchas

- **Tile-unit slicing.** `a[mr:mr+bm]` in a kernel means tile indices, not
  element indices. Very easy to write in element units and silently
  miscompute.
- **block_count >= 2 is a pipe-graph constraint.** `recv_cb` needs
  `max(2, Kp-1)` slots for this reason, even when Kp=2 only needs 1.
- **Local/inline imports cause subtle perf/compile differences.** Keep
  imports at module top.

## Operational / environment

- **Local HW is 10×13 = 130 cores.** Most `SHAPE_PLANS` assume that
  grid. CI runner is believed to be 8×8 = 64 cores; the hand-picked
  plans will not fit and need fallback or a CI-specific plan set.
- **Profiler caps at 125 signposts/core.** Long inner loops need manual
  thinning or the tail gets dropped.
