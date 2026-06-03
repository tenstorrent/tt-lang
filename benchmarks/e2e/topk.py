# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end top-K benchmark: ``ttl.ops.topk`` vs ``ttnn.topk``.

Sweeps MoE-routing widths (top-K of N experts) over many tokens. PCC is checked
against ``torch.topk`` values; the wall-clock ratio is against ``ttnn.topk``
when it accepts the shape (otherwise the ratio is left blank and only the ttl
timing + correctness are recorded).

Run: ``python -m benchmarks.e2e.topk [--filter 256] [--plot]``.
"""

import torch
import ttnn

from ttl.ops.topk import make_topk

from benchmarks.common import BenchSpec, cli, pcc, time_runs

TILE = 32

# (n_tokens, N, K, label). N <= 256 for exact bf16 indices.
CASES = (
    (4096, 64, 4, "64 experts, top-4"),
    (4096, 128, 8, "128 experts, top-8"),
    (4096, 256, 8, "256 experts, top-8 (MoE)"),
)

FIELDS = ("label", "n_tokens", "N", "K", "ttlang_ms", "ttnn_ms", "ratio", "pcc")


def _to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def run_case(device, case, *, warmup, runs):
    n_tokens, N, K, label = case
    Wt = N // TILE
    Rt = n_tokens // TILE

    torch.manual_seed(0)
    x_t = torch.randn(n_tokens, N, dtype=torch.bfloat16)
    ramp = torch.arange(N, dtype=torch.bfloat16).unsqueeze(0).repeat(TILE, 1)
    tv, _ = torch.topk(x_t.float(), K, dim=-1)

    x_d = _to_dev(x_t, device)
    idx_d = _to_dev(ramp, device)
    ov_d = _to_dev(torch.zeros(n_tokens, K * TILE, dtype=torch.bfloat16), device)
    oi_d = _to_dev(torch.zeros(n_tokens, K * TILE, dtype=torch.bfloat16), device)

    fn = make_topk(Rt=Rt, PNt=1, Wt=Wt, K=K, N=N)
    ttlang_s = time_runs(
        thunk=lambda: fn(x_d, idx_d, ov_d, oi_d),
        cleanup=lambda _r: None,
        device=device,
        warmup=warmup,
        runs=runs,
    )
    got_v = ttnn.to_torch(ov_d).reshape(n_tokens, K * TILE)[:, 0::TILE][:, :K].float()
    pcc_v = pcc(got_v, tv)

    def _clean(r):
        for t in r:
            ttnn.deallocate(t)

    try:
        ttnn_s = time_runs(
            thunk=lambda: ttnn.topk(x_d, K, dim=-1),
            cleanup=_clean,
            device=device,
            warmup=warmup,
            runs=runs,
        )
        ttnn_ms = round(ttnn_s * 1000, 4)
        ratio = round(ttlang_s / ttnn_s, 4)
    except Exception as e:
        print(f"  (ttnn.topk reference unavailable for {label}: {e})", flush=True)
        ttnn_ms = ratio = None

    for t in (x_d, idx_d, ov_d, oi_d):
        ttnn.deallocate(t)

    return {
        "label": label,
        "n_tokens": n_tokens,
        "N": N,
        "K": K,
        "ttlang_ms": round(ttlang_s * 1000, 4),
        "ttnn_ms": ttnn_ms,
        "ratio": ratio,
        "pcc": round(pcc_v, 6),
    }


def _format_row(r):
    ref = "n/a" if r["ttnn_ms"] is None else f"{r['ttnn_ms']:>8.3f}ms"
    ratio = "  n/a " if r["ratio"] is None else f"{r['ratio']:.3f}"
    return (
        f"{r['label']:<28}  "
        f"ttlang={r['ttlang_ms']:>8.3f}ms  ttnn={ref}  "
        f"ratio={ratio}  pcc={r['pcc']:.4f}"
    )


SPEC = BenchSpec(
    name="topk",
    fields=FIELDS,
    cases=CASES,
    run_case=run_case,
    label_of=lambda case: case[3],
    format_row=_format_row,
    plot_title="ttlang topk vs ttnn.topk  (bar = ratio, dotted = 1.0)",
    plot_label_of=lambda r: f"N={r['N']} K={r['K']}\n{r['n_tokens']} tokens",
)


if __name__ == "__main__":
    cli(SPEC)
