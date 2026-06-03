# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end RMSNorm benchmark: ``ttl.ops.rmsnorm`` vs ``ttnn.rms_norm``.

Sweeps the production decoder norm widths (hidden 7168, q-proj 1536, kv-proj
512) over a fixed row count. PCC is checked against a torch reference; the
wall-clock ratio is against ``ttnn.rms_norm``.

Run: ``python -m benchmarks.e2e.rmsnorm [--filter 7168] [--plot]``.
"""

import torch
import ttnn

from ttl.ops.rmsnorm import make_rmsnorm

from benchmarks.common import BenchSpec, cli, pcc, time_runs

TILE = 32
EPS = 1e-6
WCT = 8  # width-tile chunks streamed; must divide Dt

N_ROWS = 16384

# (n_rows, D, label)
CASES = (
    (N_ROWS, 512, "512 (kv-proj)"),
    (N_ROWS, 1536, "1536 (q-proj)"),
    (N_ROWS, 7168, "7168 (hidden)"),
)

FIELDS = ("label", "n_rows", "D", "Dt", "WCt", "ttlang_ms", "ttnn_ms", "ratio", "pcc")


def _to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _golden(x, w, eps):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * w


def run_case(device, case, *, warmup, runs):
    n_rows, D, label = case
    Rt = n_rows // TILE
    Dt = D // TILE

    torch.manual_seed(0)
    x_t = torch.randn(n_rows, D, dtype=torch.bfloat16)
    w_t = torch.randn(1, D, dtype=torch.bfloat16) * 0.1 + 1.0
    expected = _golden(x_t, w_t, EPS)

    x_d = _to_dev(x_t, device)
    w_d = _to_dev(w_t, device)
    out_d = _to_dev(torch.zeros(n_rows, D, dtype=torch.bfloat16), device)

    fn = make_rmsnorm(Rt=Rt, PNt=1, Dt=Dt, WCt=WCT, D=D, eps=EPS)
    ttlang_s = time_runs(
        thunk=lambda: fn(x_d, w_d, out_d),
        cleanup=lambda _r: None,
        device=device,
        warmup=warmup,
        runs=runs,
    )
    got = ttnn.to_torch(out_d).reshape(n_rows, D).to(torch.bfloat16)
    pcc_v = pcc(got, expected)

    ttnn_s = time_runs(
        thunk=lambda: ttnn.rms_norm(x_d, epsilon=EPS, weight=w_d),
        cleanup=ttnn.deallocate,
        device=device,
        warmup=warmup,
        runs=runs,
    )

    for t in (x_d, w_d, out_d):
        ttnn.deallocate(t)

    return {
        "label": label,
        "n_rows": n_rows,
        "D": D,
        "Dt": Dt,
        "WCt": WCT,
        "ttlang_ms": round(ttlang_s * 1000, 4),
        "ttnn_ms": round(ttnn_s * 1000, 4),
        "ratio": round(ttlang_s / ttnn_s, 4),
        "pcc": round(pcc_v, 6),
    }


def _format_row(r):
    return (
        f"{r['label']:<20}  "
        f"ttlang={r['ttlang_ms']:>8.3f}ms  ttnn={r['ttnn_ms']:>8.3f}ms  "
        f"ratio={r['ratio']:.3f}  pcc={r['pcc']:.4f}  rows={r['n_rows']}"
    )


SPEC = BenchSpec(
    name="rmsnorm",
    fields=FIELDS,
    cases=CASES,
    run_case=run_case,
    label_of=lambda case: case[2],
    format_row=_format_row,
    plot_title="ttlang rmsnorm vs ttnn.rms_norm  (bar = ratio, dotted = 1.0)",
    plot_label_of=lambda r: f"D={r['D']}\n{r['n_rows']} rows",
)


if __name__ == "__main__":
    cli(SPEC)
