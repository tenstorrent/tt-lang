# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Indexed GEMV benchmark: routed-expert weight streams selected at runtime.

Cases mirror the Gemma per-card MoE: 32 resident experts, top-8 active,
three streams per expert (gate_up fused as wider N). The headline metric is
effective weight bandwidth across all active expert slices.

Run: ``python -m benchmarks.e2e.indexed_gemv [--plot]``.
"""

import torch
import ttnn

from ttl.ops.indexed_gemv import make_indexed_gemv

from benchmarks.common import BenchSpec, cli, pcc, time_runs

TILE = 32

# (E, K, N, topk, (Np, Kp), bn, label)
CASES = (
    (32, 2816, 704, 8, (11, 2), 2, "gemma down 32E top8 [2816x704]"),
    (32, 2816, 1408, 8, (11, 2), 4, "gemma gate_up 32E top8 [2816x1408]"),
)

FIELDS = ("label", "E", "K", "N", "topk", "ttlang_ms", "gbps", "pcc")


def _to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def run_case(device, case, *, warmup, runs):
    E, K, N, topk, grid_cfg, bn, label = case
    torch.manual_seed(0)
    x_t = torch.randn(TILE, K, dtype=torch.bfloat16) * 0.1
    w_t = torch.randn(E * K, N, dtype=torch.bfloat16) * 0.1
    ids = list(range(0, 2 * topk, 2))[:topk]
    idx_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    idx_t[0, :topk] = torch.tensor(ids, dtype=torch.bfloat16)
    expected = torch.stack(
        [torch.matmul(x_t[:1].float(), w_t[e * K:(e + 1) * K].float())[0] for e in ids]
    )

    x_d, idx_d, w_d = _to_dev(x_t, device), _to_dev(idx_t, device), _to_dev(w_t, device)
    out_d = _to_dev(torch.zeros(topk * TILE, N, dtype=torch.bfloat16), device)

    fn = make_indexed_gemv(E, K, N, topk, grid_cfg, bn)
    ttlang_s = time_runs(
        thunk=lambda: fn(x_d, idx_d, w_d, out_d),
        cleanup=lambda _r: None, device=device, warmup=warmup, runs=runs,
    )
    got = ttnn.to_torch(out_d).reshape(topk, TILE, N)[:, 0, :].float()
    pcc_v = pcc(got, expected)
    gbps = (topk * K * N * 2) / ttlang_s / 1e9

    for t in (x_d, idx_d, w_d, out_d):
        ttnn.deallocate(t)

    return {
        "label": label, "E": E, "K": K, "N": N, "topk": topk,
        "ttlang_ms": round(ttlang_s * 1000, 4),
        "gbps": round(gbps, 1), "pcc": round(pcc_v, 6),
    }


def _format_row(r):
    return (
        f"{r['label']:<36}  ttlang={r['ttlang_ms']:>8.3f}ms  "
        f"{r['gbps']:>6.1f} GB/s  pcc={r['pcc']:.4f}"
    )


SPEC = BenchSpec(
    name="indexed_gemv",
    fields=FIELDS,
    cases=CASES,
    run_case=run_case,
    label_of=lambda case: case[6],
    format_row=_format_row,
    ratio_key=None,
)


if __name__ == "__main__":
    cli(SPEC)
