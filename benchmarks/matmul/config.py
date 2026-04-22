# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unified matmul planner.

Picks block shape (bm, bn, bk) in tiles and grid partitioning
(M_parts, N_parts, K_parts) for a single ksplit-style kernel that
subsumes pure SUMMA (K_parts = 1).

Scoring (each step only breaks ties from the previous):
    1. Maximize effective cores    (Mp * Np * Kp / pad)
    2. Maximize A+B mcast volume   (bk * (bm + bn))
    3. Maximize block compute volume (bm * bn * bk)

Hard rules that reject a plan outright:
    - L1 CB footprint exceeds budget.
    - `pad` > `max_pad` (default 1.25).
    - MIN_CORES_MULTI_ITER: if any core processes more than one
      output-block iter (m_span * n_span > 1), at least 100 cores must
      be in use. Single-iter plans may fall below 100 cores freely --
      that case cannot be improved by adding parallelism.

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
from typing import List, Tuple

TILE = 32

# Wormhole worker grid. (rows, cols); N dimension lives on cols.
MAX_GRID_M = 10
MAX_GRID_N = 13

# Block dims considered, in tiles. Power-of-two set is fine: block dims
# must divide the shape in tiles, and 3/5/6/7-tile blocks rarely beat 4
# or 8 after alignment. Iterated from fattest to thinnest.
BLOCK_DIMS: Tuple[int, ...] = (8, 4, 2, 1)

# Per-core L1 budget for circular buffers (bytes). Conservative default;
# query ttnn.device.get_max_worker_l1_unreserved_size() at runtime for
# device-accurate tuning.
DEFAULT_L1_BUDGET_BYTES = 1_000_000

# bfloat16 tile = 32x32 half-precision = 2048 B (ignoring page padding;
# tt-metal adds a small header per page but it's a rounding error here).
BF16_BYTES = 2

# Padding budget. Plans with more than MAX_PAD fraction of padded work
# over real work are rejected. 1.25 = up to 25% waste allowed.
MAX_PAD = 1.25

# If any core's output loop iterates more than once, require at least
# this many cores total. Prevents single-iter plans from leaving half
# the grid idle.
MIN_CORES_MULTI_ITER = 100


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
    part_cfg: Tuple[int, int, int]   # (M_parts, N_parts, K_parts)

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
    """CBs allocated by the unified ksplit kernel.

    A+B mcast double-buffer is block_count=2. The K-reduce path adds recv_cb
    sized to hold all concurrent gather slots (one per non-root k-rank), plus
    a threaded accumulator (sum_cb) and a single-block handoff to dm_write.
    """
    cbs = [
        CBShape("a_cb", bm * bk, 2),
        CBShape("b_cb", bk * bn, 2),
        CBShape("partial_cb", bm * bn, 2),
    ]
    if k_parts > 1:
        # recv_cb block_count = num gather senders = K_parts - 1 (min 2).
        # Matches the constraint enforced by PipeGraph::verifyGatherBlockCounts.
        cbs.extend([
            CBShape("recv_cb", bm * bn, max(2, k_parts - 1)),
            CBShape("sum_cb", bm * bn, 2),
            CBShape("out_cb", bm * bn, 1),
        ])
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
    min_cores_multi_iter: int = MIN_CORES_MULTI_ITER,
    dtype_bytes: int = BF16_BYTES,
) -> MatmulPlan:
    """Plan a unified-ksplit matmul for shape (M, K, N).

    See module docstring for priority/rules. Raises ValueError if any
    dimension is not tile-aligned, or if no block shape admits a plan
    that fits in L1 and respects the pad/multi-iter rules.
    """
    if any(d <= 0 for d in (M, K, N)):
        raise ValueError(f"dims must be positive: M={M} K={K} N={N}")
    if any(d % TILE for d in (M, K, N)):
        raise ValueError(
            f"dims must be tile-aligned (TILE={TILE}): M={M} K={K} N={N}"
        )

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
                        iters_per_core = m_span * n_span
                        cores = Mp * Np * Kp
                        if iters_per_core > 1 and cores < min_cores_multi_iter:
                            continue
                        if estimate_l1_bytes(bm, bn, bk, Kp, dtype_bytes) > l1_budget_bytes:
                            continue

                        effective_cores = cores / pad
                        mcast_volume = bk * (bm + bn)
                        compute_volume = bm * bn * bk
                        score = (effective_cores, mcast_volume, compute_volume)

                        if best_score is None or score > best_score:
                            best_score = score
                            best_plan = ((bm, bn, bk), (Mp, Np, Kp))

    if best_plan is None:
        raise ValueError(
            f"no valid plan for M={M} K={K} N={N} "
            f"(block must divide dims; pad <= {max_pad}; "
            f"if >1 iter/core then >= {min_cores_multi_iter} cores; "
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
    (1024,  1024,  1024,  "1k^3"),
    (1024,  2048,  1024,  "1k x 2k x 1k"),
    (2048,  2048,  2048,  "2k^3"),
    (2048,  4096,  2048,  "2k x 4k x 2k"),
    (2560,  2048,  3072,  "2.5k x 2k x 3k"),
    (2048,  8192,  2048,  "2k x 8k x 2k (long K)"),
    (2560,  4096,  3072,  "2.5k x 4k x 3k"),
    (2560,  8192,  3072,  "2.5k x 8k x 3k (120 cores)"),
    (2560,  8192,  3328,  "2.5k x 8k x 3.3k (130 cores)"),
    (1024,  16384, 2560,  "1k x 16k x 2.5k (tall K)"),
    (4096,  4096,  4096,  "4k^3"),
    (4096,  8192,  4096,  "4k x 8k x 4k"),
    (8192,  8192,  8192,  "8k^3"),
    (10240, 8192,  13312, "10k x 8k x 13k (130 cores, 4x4)"),
)


def main() -> None:
    print("matmul planner spot check (sweep shapes)")
    print(f"grid={MAX_GRID_M}x{MAX_GRID_N}  L1_budget={DEFAULT_L1_BUDGET_BYTES/1024:.0f} KiB  "
          f"max_pad={MAX_PAD}  min_cores_multi_iter={MIN_CORES_MULTI_ITER}")
    print("-" * 130)
    for (M, K, N, label) in _SWEEP_SHAPES:
        try:
            plan = plan_matmul(M, K, N)
            print(f"{label:<32}  {plan.describe()}")
        except ValueError as e:
            print(f"{label:<32}  FAIL: {e}")


if __name__ == "__main__":
    main()
