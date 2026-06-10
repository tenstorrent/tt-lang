# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GEMV benchmark: ``ttl.ops.gemv`` vs ``ttnn.linear``.

Decode projections are DRAM-BW bound, so the headline metric is effective
weight-stream bandwidth (W bytes / wall-clock). Cases sweep the Gemma
26B-A4B per-card projection shapes.

Run: ``python -m benchmarks.e2e.gemv [--filter qkv] [--plot]``.
"""

import torch
import ttnn

from ttl.ops.gemv import make_gemv

from benchmarks.common import BenchSpec, cli, pcc, time_runs

TILE = 32

# (K, N, (Np, Kp), bn, label)
CASES = (
    (2816, 1536, (12, 2), 2, "qkv sliding [2816x1536]"),
    (2816, 3072, (12, 2), 2, "qkv global  [2816x3072]"),
    (1024, 2816, (11, 2), 4, "o proj      [1024x2816]"),
    (2816, 2112, (11, 2), 2, "mlp gate/up [2816x2112] (pad N)"),
    (2816, 5632, (11, 2), 4, "experts 8x gate_up-slice eq [2816x5632]"),
)

FIELDS = ("label", "K", "N", "ttlang_ms", "gbps", "ttnn_ms", "ratio", "pcc")


def _to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _pad_up(n, mult):
    return ((n + mult - 1) // mult) * mult


def run_case(device, case, *, warmup, runs):
    K, N, grid_cfg, bn, label = case
    Np, _ = grid_cfg
    Npad = _pad_up(N, Np * bn * TILE)

    torch.manual_seed(0)
    x_t = torch.randn(TILE, K, dtype=torch.bfloat16) * 0.1
    w_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.1
    expected = torch.matmul(x_t[:1].float(), w_t.float())

    x_d = _to_dev(x_t, device)
    w_d = _to_dev(torch.nn.functional.pad(w_t, (0, Npad - N)), device)
    out_d = _to_dev(torch.zeros(TILE, Npad, dtype=torch.bfloat16), device)

    fn = make_gemv(TILE, K, Npad, grid_cfg, bn)
    ttlang_s = time_runs(
        thunk=lambda: fn(x_d, w_d, out_d),
        cleanup=lambda _r: None,
        device=device,
        warmup=warmup,
        runs=runs,
    )
    got = ttnn.to_torch(out_d).reshape(TILE, Npad)[:1, :N].float()
    pcc_v = pcc(got, expected)
    gbps = (K * Npad * 2) / ttlang_s / 1e9

    def _clean(r):
        ttnn.deallocate(r)

    try:
        ttnn_s = time_runs(
            thunk=lambda: ttnn.linear(x_d, w_d),
            cleanup=_clean,
            device=device,
            warmup=warmup,
            runs=runs,
        )
        ttnn_ms = round(ttnn_s * 1000, 4)
        ratio = round(ttlang_s / ttnn_s, 4)
    except Exception as e:
        print(f"  (ttnn.linear unavailable for {label}: {e})", flush=True)
        ttnn_ms = ratio = None

    for t in (x_d, w_d, out_d):
        ttnn.deallocate(t)

    return {
        "label": label,
        "K": K,
        "N": N,
        "ttlang_ms": round(ttlang_s * 1000, 4),
        "gbps": round(gbps, 1),
        "ttnn_ms": ttnn_ms,
        "ratio": ratio,
        "pcc": round(pcc_v, 6),
    }


def _format_row(r):
    ref = "n/a" if r["ttnn_ms"] is None else f"{r['ttnn_ms']:>8.3f}ms"
    ratio = "  n/a " if r["ratio"] is None else f"{r['ratio']:.3f}"
    return (
        f"{r['label']:<38}  "
        f"ttlang={r['ttlang_ms']:>8.3f}ms  {r['gbps']:>6.1f} GB/s  "
        f"ttnn={ref}  ratio={ratio}  pcc={r['pcc']:.4f}"
    )


SPEC = BenchSpec(
    name="gemv",
    fields=FIELDS,
    cases=CASES,
    run_case=run_case,
    label_of=lambda case: case[4],
    format_row=_format_row,
    plot_title="ttlang gemv vs ttnn.linear  (bar = ratio, dotted = 1.0)",
    plot_label_of=lambda r: f"K={r['K']} N={r['N']}",
)


if __name__ == "__main__":
    cli(SPEC)
