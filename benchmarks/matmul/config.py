# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul planner for ksplit/SUMMA kernels.

Picks block shape (bm, bn, bk) in tiles and grid partitioning
(M_parts, N_parts, K_parts). K_parts == 1 routes to summa_kernel,
K_parts >= 2 to ksplit_kernel.

Scoring: maximize an estimated throughput
    throughput ~= cores * bv / (pad * (bv + ALPHA))
where bv = bm*bn*bk and ALPHA is a per-block overhead constant in
tile-matmul-equivalent units. For large blocks the factor bv/(bv+ALPHA)
saturates near 1 and the objective reduces to cores/pad (classic
effective-cores maximization). For small blocks overhead dominates and
the factor pulls the objective toward fatter blocks even at the cost
of cores.

ALPHA was fit from the 14-shape baseline sweep: the observed
throughput-vs-bv curve matches bv/(bv+128) closely across bv in
{8, 16, 64, 256, 512}. Retune if block-overhead characteristics
change (kernel refactor, different dtype, etc.).

Hard rules that reject a plan outright:
    - L1 CB footprint exceeds budget.
    - `pad` > `max_pad` (default 1.25).

`pad` is output-cell padding waste: with non-divisor Mp/Np the kernel
runs on a Mp-by-Np grid of owners each responsible for m_span * n_span
output blocks, some of which may be off the real tensor and get
dropped. `pad = (Mp*m_span*Np*n_span) / (Mb*Nb)`.

K must divide exactly: partial-reduction shape has to match across
all K-ranks. Block (bm, bn, bk) must also divide (Mt, Nt, Kt) exactly
(no sub-tile padding).

Tile misalignment and shapes with no feasible plan raise ValueError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

TILE = 32

# Empirically-best (block_cfg, part_cfg) per shape, picked from sweeps of
# our planner vs the bench_matmul_sweep.py heuristics. The
# throughput model doesn't capture all the real-HW tradeoffs (per-core
# efficiency vs core count vs pad) so for benchmarked shapes we override
# the search with the measured winner. Shapes not listed fall through to
# plan_matmul's search. Regenerate when kernels or HW change.
SHAPE_PLANS: Dict[
    Tuple[int, int, int], Tuple[Tuple[int, int, int], Tuple[int, int, int]]
] = {
    (1024, 1024, 1024): ((4, 8, 8), (8, 4, 2)),
    (1024, 2048, 1024): ((4, 4, 8), (8, 8, 1)),
    (2048, 2048, 2048): ((8, 4, 8), (8, 6, 2)),
    (2048, 4096, 2048): ((8, 4, 8), (8, 6, 2)),
    (2560, 2048, 3072): ((8, 4, 8), (10, 13, 1)),
    (2048, 8192, 2048): ((8, 4, 8), (8, 6, 2)),
    (2560, 4096, 3072): ((8, 4, 8), (10, 13, 1)),
    (2560, 8192, 3072): ((8, 4, 8), (10, 13, 1)),
    (2560, 8192, 3328): ((8, 8, 8), (10, 13, 1)),
    (1024, 16384, 2560): ((4, 8, 8), (8, 10, 1)),
    (4096, 4096, 4096): ((8, 4, 8), (8, 11, 1)),
    (4096, 8192, 4096): ((8, 4, 8), (8, 11, 1)),
    (8192, 8192, 8192): ((8, 4, 8), (8, 13, 1)),
    (10240, 8192, 13312): ((8, 8, 8), (10, 13, 1)),
    (2560, 16384, 3328): ((8, 8, 8), (10, 13, 1)),
    (2560, 32768, 3328): ((8, 8, 8), (10, 13, 1)),
    (10240, 16384, 13312): ((8, 8, 8), (10, 13, 1)),
    (5120, 32768, 6656): ((8, 8, 8), (10, 13, 1)),
}

# Wormhole worker grid. (rows, cols); N dimension lives on cols.
MAX_GRID_M = 10
MAX_GRID_N = 13

# Block dims considered, in tiles. Dims above 8 (tried 10/12/14/16) run
# slower empirically even when the bv/(bv+α) model predicts a win —
# e.g. 4k³ (16,2,16)/(8,13,1) 104c bv=512 model-predicted 33% faster
# but ran 25% slower than (8,4,8) 88c. LLK overhead grows non-linearly
# for bm/bn/bk > 8 in ways the constant-α model doesn't capture. Keep
# search bounded to 8. Non-power-of-two (7,6,5,3) hit exact divisors
# for 2.5k (80 tiles) and 3.3k (104 tiles).
BLOCK_DIMS: Tuple[int, ...] = (8, 7, 6, 5, 4, 3, 2, 1)

# Per-core L1 budget for circular buffers (bytes). Wormhole worker has
# ~1.57 MiB SRAM per core; kernel program eats ~130 KiB, so CBs have
# ~1.44 MiB to play with.
DEFAULT_L1_BUDGET_BYTES = 1_440_000

# bfloat16 tile = 32x32 half-precision = 2048 B (ignoring page padding;
# tt-metal adds a small header per page but it's a rounding error here).
BF16_BYTES = 2

# Padding budget. Plans with more than MAX_PAD fraction of padded work
# over real work are rejected. 1.25 = up to 25% waste allowed.
MAX_PAD = 1.25

# Per-block overhead in tile-matmul-equivalent units. Refit from probe:
# on 4k³ (8,4,8)/(8,12,1) Kp=1 96c (1.704ms) beats (8,8,8)/(8,6,2) Kp=2
# 96c (1.820ms) -> smaller-bv configs are relatively more competitive
# than α=128 predicted. α=64 puts the bv/(bv+α) factor for bv=256 at
# 0.8 (vs 0.89 for bv=512), letting bn=4 variants win when pad/cores
# are equal or favorable.
BLOCK_OVERHEAD_ALPHA = 64

# Per-gather cost as a fraction of block time. Scales with *total* gathers
# per core = (Kp-1) * iter_per_core, not just Kp-1. A single linear-in-Kp
# penalty couldn't satisfy both calibration points simultaneously:
#   4k³   (8,4,8)/(8,12,1) iter=6 Kp=1 @ 1.704ms  vs
#         (8,8,8)/(8, 6,2) iter=6 Kp=2 @ 1.820ms  → wants β(Kp-1)>0.21
#   2k8k2k (8,8,8)/(8,3,4) iter=3 Kp=4 @ 0.953ms  vs
#          (8,8,8)/(8,8,1) iter=1 Kp=1 @ 1.067ms  → wants β(Kp-1)<0.11
# Scaling by total_gathers=(Kp-1)*iter resolves both (flip points are
# gathers=6 and gathers=9), implying β≈0.035 per gather event.
KP_PENALTY_BETA = 0.035


@dataclass(frozen=True)
class CBShape:
    name: str
    tiles_per_block: int
    block_count: int

    @property
    def total_tiles(self) -> int:
        return self.tiles_per_block * self.block_count


@dataclass(frozen=True)
class MatmulPlan:
    M: int
    K: int
    N: int
    block_cfg: Tuple[int, int, int]  # (bm, bn, bk) in tiles
    part_cfg: Tuple[int, int, int]  # (M_parts, N_parts, K_parts)

    @property
    def cores(self) -> int:
        Mp, Np, Kp = self.part_cfg
        return Mp * Np * Kp

    @property
    def grid(self) -> Tuple[int, int]:
        Mp, Np, Kp = self.part_cfg
        return (Np * Kp, Mp)  # (cols, rows) matches @ttl.operation grid=

    @property
    def m_bpn(self) -> int:
        Mt = self.M // TILE
        bm = self.block_cfg[0]
        Mp = self.part_cfg[0]
        return -(-(Mt // bm) // Mp)  # ceil(Mb / Mp)

    @property
    def n_bpn(self) -> int:
        Nt = self.N // TILE
        bn = self.block_cfg[1]
        Np = self.part_cfg[1]
        return -(-(Nt // bn) // Np)

    @property
    def iters_per_core(self) -> int:
        return self.m_bpn * self.n_bpn

    @property
    def pad_ratio(self) -> float:
        Mt = self.M // TILE
        Nt = self.N // TILE
        bm, bn, _ = self.block_cfg
        Mp, Np, _ = self.part_cfg
        Mb, Nb = Mt // bm, Nt // bn
        return (Mp * self.m_bpn * Np * self.n_bpn) / (Mb * Nb)

    @property
    def padded_dims(self) -> Tuple[int, int]:
        """(M_padded, N_padded) tensor dims the kernel expects. K unchanged."""
        bm, bn, _ = self.block_cfg
        Mp, Np, _ = self.part_cfg
        return (Mp * self.m_bpn * bm * TILE, Np * self.n_bpn * bn * TILE)

    @property
    def l1_bytes(self) -> int:
        bm, bn, bk = self.block_cfg
        _, _, Kp = self.part_cfg
        return estimate_l1_bytes(bm, bn, bk, Kp)

    def describe(self) -> str:
        bm, bn, bk = self.block_cfg
        Mp, Np, Kp = self.part_cfg
        return (
            f"M={self.M:>5} K={self.K:>5} N={self.N:>5}  "
            f"block=({bm},{bn},{bk})  parts=({Mp:>2},{Np:>2},{Kp})  "
            f"grid={self.grid[0]:>2}x{self.grid[1]:<2} cores={self.cores:>3}  "
            f"iter/core={self.iters_per_core:>2}  "
            f"pad={self.pad_ratio:.2f}  L1={self.l1_bytes/1024:>5.0f}KiB"
        )


def cb_layout(bm: int, bn: int, bk: int, k_parts: int) -> List[CBShape]:
    """CBs allocated by the matmul kernels.

    Kp=1 uses summa_kernel: a_cb + b_cb mcast double-buffers plus a
    single out_cb double-buffer that serves as both compute accumulator
    and writer handoff.

    Kp>=2 uses ksplit_kernel: partial_cb ping-pongs the reduce chain,
    recv_cb holds Kp-1 gather slots (min 2 per PipeGraph constraint),
    and out_cb is a single-block handoff to dm_write.
    """
    cbs = [
        CBShape("a_cb", bm * bk, 2),
        CBShape("b_cb", bk * bn, 2),
    ]
    if k_parts == 1:
        cbs.append(CBShape("out_cb", bm * bn, 2))
    else:
        cbs.extend(
            [
                CBShape("partial_cb", bm * bn, 2),
                CBShape("recv_cb", bm * bn, max(2, k_parts - 1)),
                CBShape("out_cb", bm * bn, 1),
            ]
        )
    return cbs


def estimate_l1_bytes(
    bm: int, bn: int, bk: int, k_parts: int, dtype_bytes: int = BF16_BYTES
) -> int:
    tile_bytes = TILE * TILE * dtype_bytes
    return sum(cb.total_tiles for cb in cb_layout(bm, bn, bk, k_parts)) * tile_bytes


def _largest_divisor(n: int, cap: int) -> int:
    cap = max(1, min(n, cap))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def plan_matmul(
    M: int,
    K: int,
    N: int,
    *,
    grid_m: int = MAX_GRID_M,
    grid_n: int = MAX_GRID_N,
    l1_budget_bytes: int = DEFAULT_L1_BUDGET_BYTES,
    max_pad: float = MAX_PAD,
    alpha: float = BLOCK_OVERHEAD_ALPHA,
    kp_beta: float = KP_PENALTY_BETA,
    dtype_bytes: int = BF16_BYTES,
) -> MatmulPlan:
    """Plan a ksplit/SUMMA matmul for shape (M, K, N).

    See module docstring for the throughput objective. Raises
    ValueError if any dimension is not tile-aligned, or if no block
    shape admits a plan that fits in L1 and respects the pad rule.
    """
    if any(d <= 0 for d in (M, K, N)):
        raise ValueError(f"dims must be positive: M={M} K={K} N={N}")
    if any(d % TILE for d in (M, K, N)):
        raise ValueError(f"dims must be tile-aligned (TILE={TILE}): M={M} K={K} N={N}")

    if (M, K, N) in SHAPE_PLANS:
        block_cfg, part_cfg = SHAPE_PLANS[(M, K, N)]
        return MatmulPlan(M=M, K=K, N=N, block_cfg=block_cfg, part_cfg=part_cfg)

    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE

    best_score = None
    best_plan: Tuple[Tuple[int, int, int], Tuple[int, int, int]] | None = None

    for bm in BLOCK_DIMS:
        if Mt % bm:
            continue
        for bn in BLOCK_DIMS:
            if Nt % bn:
                continue
            for bk in BLOCK_DIMS:
                if Kt % bk:
                    continue

                Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk

                for Mp in range(1, min(grid_m, Mb) + 1):
                    m_span = -(-Mb // Mp)
                    pad_m = (Mp * m_span) / Mb
                    if pad_m > max_pad:
                        continue

                    for Np in range(1, min(grid_n, Nb) + 1):
                        n_span = -(-Nb // Np)
                        pad = pad_m * (Np * n_span) / Nb
                        if pad > max_pad:
                            continue

                        Kp = _largest_divisor(Kb, grid_n // Np)
                        cores = Mp * Np * Kp
                        if (
                            estimate_l1_bytes(bm, bn, bk, Kp, dtype_bytes)
                            > l1_budget_bytes
                        ):
                            continue

                        block_vol = bm * bn * bk
                        iter_per_core = m_span * n_span
                        total_gathers = (Kp - 1) * iter_per_core
                        gather_penalty = 1.0 + kp_beta * total_gathers
                        throughput = (
                            cores
                            * block_vol
                            / (pad * (block_vol + alpha) * gather_penalty)
                        )
                        # Throughput ties (common when cores/pad normalizes
                        # out): break on bv first (bigger blocks amortize
                        # per-block overhead better than bv/(bv+α) predicts
                        # for small bv — 4k³ (8,2,8) 117c ran 23% slower
                        # than (8,4,8) 96c at matched predicted throughput),
                        # then less pad, then more cores, then lower Kp.
                        score = (throughput, block_vol, -pad, cores, -Kp)

                        if best_score is None or score > best_score:
                            best_score = score
                            best_plan = ((bm, bn, bk), (Mp, Np, Kp))

    if best_plan is None:
        raise ValueError(
            f"no valid plan for M={M} K={K} N={N} "
            f"(block must divide dims; pad <= {max_pad}; "
            f"L1 <= {l1_budget_bytes} B)"
        )
    block_cfg, part_cfg = best_plan
    return MatmulPlan(M=M, K=K, N=N, block_cfg=block_cfg, part_cfg=part_cfg)


# ---------------------------------------------------------------------------
# Spot-check main
# ---------------------------------------------------------------------------

# Shapes mirror benchmarks/matmul/sweep.py so the spot check reports plans
# for exactly the inputs the bench sweeps over.
_SWEEP_SHAPES: Tuple[Tuple[int, int, int, str], ...] = (
    (1024, 1024, 1024, "1k^3"),
    (1024, 2048, 1024, "1k x 2k x 1k"),
    (2048, 2048, 2048, "2k^3"),
    (2048, 4096, 2048, "2k x 4k x 2k"),
    (2560, 2048, 3072, "2.5k x 2k x 3k"),
    (2048, 8192, 2048, "2k x 8k x 2k (long K)"),
    (2560, 4096, 3072, "2.5k x 4k x 3k"),
    (2560, 8192, 3072, "2.5k x 8k x 3k (120 cores)"),
    (2560, 8192, 3328, "2.5k x 8k x 3.3k (130 cores)"),
    (1024, 16384, 2560, "1k x 16k x 2.5k (tall K)"),
    (4096, 4096, 4096, "4k^3"),
    (4096, 8192, 4096, "4k x 8k x 4k"),
    (8192, 8192, 8192, "8k^3"),
    (10240, 8192, 13312, "10k x 8k x 13k (130 cores, 4x4)"),
)


def main() -> None:
    print("matmul planner spot check (sweep shapes)")
    print(
        f"grid={MAX_GRID_M}x{MAX_GRID_N}  L1_budget={DEFAULT_L1_BUDGET_BYTES/1024:.0f} KiB  "
        f"max_pad={MAX_PAD}  alpha={BLOCK_OVERHEAD_ALPHA}  kp_beta={KP_PENALTY_BETA}"
    )
    print("-" * 130)
    for M, K, N, label in _SWEEP_SHAPES:
        try:
            plan = plan_matmul(M, K, N)
            print(f"{label:<32}  {plan.describe()}")
        except ValueError as e:
            print(f"{label:<32}  FAIL: {e}")


if __name__ == "__main__":
    main()
