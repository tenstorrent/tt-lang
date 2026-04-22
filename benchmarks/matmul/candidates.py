# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Enumerate top-N plan candidates per shape under a given (alpha, beta,
max_pad) setting. Lets us scan scoring-parameter space locally without
HW runs.

Usage:
    python3 candidates.py                     # current config defaults
    python3 candidates.py --alpha 64          # override alpha
    python3 candidates.py --beta 0.2          # override Kp penalty
    python3 candidates.py --max-pad 1.4
    python3 candidates.py --top 3             # top-N per shape
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

from config import (
    BLOCK_DIMS,
    BLOCK_OVERHEAD_ALPHA,
    DEFAULT_L1_BUDGET_BYTES,
    KP_PENALTY_BETA,
    MAX_GRID_M,
    MAX_GRID_N,
    MAX_PAD,
    TILE,
    _SWEEP_SHAPES,
    _largest_divisor,
    estimate_l1_bytes,
)


@dataclass(frozen=True)
class Candidate:
    block_cfg: Tuple[int, int, int]
    part_cfg: Tuple[int, int, int]
    pad: float
    cores: int
    iter_per_core: int
    throughput: float
    l1_bytes: int


def enumerate_candidates(
    M: int, K: int, N: int,
    *,
    alpha: float,
    beta: float,
    max_pad: float,
    l1_budget: int,
    grid_m: int = MAX_GRID_M,
    grid_n: int = MAX_GRID_N,
) -> List[Candidate]:
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    out: List[Candidate] = []
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
                        l1 = estimate_l1_bytes(bm, bn, bk, Kp)
                        if l1 > l1_budget:
                            continue
                        bv = bm * bn * bk
                        iter_pc = m_span * n_span
                        gather = 1.0 + beta * (Kp - 1) * iter_pc
                        thr = cores * bv / (pad * (bv + alpha) * gather)
                        out.append(Candidate(
                            block_cfg=(bm, bn, bk),
                            part_cfg=(Mp, Np, Kp),
                            pad=pad,
                            cores=cores,
                            iter_per_core=m_span * n_span,
                            throughput=thr,
                            l1_bytes=l1,
                        ))
    out.sort(key=lambda c: (
        -c.throughput,
        -(c.block_cfg[0] * c.block_cfg[1] * c.block_cfg[2]),
        c.pad,
        -c.cores,
        c.part_cfg[2],
    ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=BLOCK_OVERHEAD_ALPHA)
    ap.add_argument("--beta", type=float, default=KP_PENALTY_BETA)
    ap.add_argument("--max-pad", type=float, default=MAX_PAD)
    ap.add_argument("--l1", type=int, default=DEFAULT_L1_BUDGET_BYTES)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--shape", default=None,
                    help="filter by label substring (e.g. '4k^3')")
    args = ap.parse_args()

    print(
        f"alpha={args.alpha}  beta={args.beta}  max_pad={args.max_pad}  "
        f"L1={args.l1/1024:.0f} KiB  top={args.top}"
    )
    print("-" * 130)
    for (M, K, N, label) in _SWEEP_SHAPES:
        if args.shape and args.shape not in label:
            continue
        cands = enumerate_candidates(
            M, K, N,
            alpha=args.alpha, beta=args.beta,
            max_pad=args.max_pad, l1_budget=args.l1,
        )
        print(f"\n{label}  (M={M} K={K} N={N})")
        if not cands:
            print("  (no valid plan)")
            continue
        for c in cands[:args.top]:
            bm, bn, bk = c.block_cfg
            Mp, Np, Kp = c.part_cfg
            print(
                f"  thr={c.throughput:>6.2f}  "
                f"({bm},{bn},{bk})/({Mp:>2},{Np:>2},{Kp})  "
                f"cores={c.cores:>3}  iter={c.iter_per_core:>2}  "
                f"pad={c.pad:.2f}  L1={c.l1_bytes/1024:.0f}KiB"
            )


if __name__ == "__main__":
    main()
