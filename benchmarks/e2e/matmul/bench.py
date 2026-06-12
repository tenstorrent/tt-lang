# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end matmul benchmark: ``ttl.ops.matmul`` (ksplit) vs ``ttnn.matmul``.

Measures the *op* directly -- the kernel under test is ``make_ksplit`` from the
shared op library, not a benchmark-local copy. The planner pins ``Kp == 2``
(``require_kp``) since that is the K-split the op supports today; shapes with no
feasible Kp==2 plan fail their row and the sweep moves on.

Run standalone (``python -m benchmarks.e2e.matmul [--filter 8k] [--plot]``) or
via the unified ``driver``; both consume the ``SPEC`` below.
"""

import torch
import ttnn

from ttl.ops.matmul import make_ksplit

from benchmarks.common import BenchSpec, pcc, time_runs
from benchmarks.e2e.matmul.plan import plan_matmul

# Sorted by M*K*N (rough work size). Annotations in labels flag why a shape is
# interesting (short K, long K, full grid, etc.).
SHAPES = (
    (1024, 1024, 1024, "1k^3"),
    (1024, 2048, 1024, "1k x 2k x 1k"),
    (2048, 2048, 2048, "2k^3"),
    (3072, 1024, 3072, "3k x 1k x 3k (short K)"),
    (2560, 2048, 3072, "2.5k x 2k x 3k"),
    (2048, 4096, 2048, "2k x 4k x 2k"),
    (2560, 4096, 3072, "2.5k x 4k x 3k"),
    (2048, 8192, 2048, "2k x 8k x 2k (long K)"),
    (3072, 4096, 3072, "3k x 4k x 3k"),
    (1024, 16384, 2560, "1k x 16k x 2.5k (tall K)"),
    (5120, 2048, 5120, "5k x 2k x 5k (short K)"),
    (2560, 8192, 3072, "2.5k x 8k x 3k (120 cores)"),
    (4096, 4096, 4096, "4k^3"),
    (2560, 8192, 3328, "2.5k x 8k x 3.3k (130 cores)"),
    (6144, 2048, 6144, "6k x 2k x 6k (short K)"),
    (4096, 8192, 4096, "4k x 8k x 4k"),
    (2560, 16384, 3328, "2.5k x 16k x 3.3k"),
    (2560, 32768, 3328, "2.5k x 32k x 3.3k"),
    (8192, 8192, 8192, "8k^3"),
    (10240, 8192, 13312, "10k x 8k x 13k (130 cores, 4x4)"),
    (5120, 32768, 6656, "5k x 32k x 6.5k"),
    (10240, 16384, 13312, "10k x 16k x 13k"),
)

# 128 KiB headroom for tt-metal state so large-block kernels fit the
# kernel-config buffer.
L1_BUDGET_REDUCTION_BYTES = 131072

FP32_ACC = True
TTNN_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4 if FP32_ACC else ttnn.MathFidelity.HiFi2,
    fp32_dest_acc_en=FP32_ACC,
    packer_l1_acc=True,
)

FIELDS = (
    "label",
    "M",
    "K",
    "N",
    "bm",
    "bn",
    "bk",
    "Mp",
    "Np",
    "Kp",
    "cores",
    "iter_per_core",
    "pad",
    "ttlang_ms",
    "ttnn_ms",
    "ratio",
    "pcc",
)


def _to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _pad_2d(t, rows, cols):
    r, c = t.shape
    if r == rows and c == cols:
        return t
    return torch.nn.functional.pad(t, (0, cols - c, 0, rows - r), value=0.0)


def _open_device():
    default_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    return ttnn.open_device(
        device_id=0, worker_l1_size=default_l1 - L1_BUDGET_REDUCTION_BYTES
    )


def run_case(device, case, *, warmup, runs):
    M, K, N, label = case
    plan = plan_matmul(M, K, N, require_kp=2)
    M_pad, N_pad = plan.padded_dims

    torch.manual_seed(0)
    a_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    ref = a_t.float() @ w_t.float()

    a_k = _to_dev(_pad_2d(a_t, M_pad, K), device)
    w_k = _to_dev(_pad_2d(w_t, K, N_pad), device)
    out_k = _to_dev(torch.zeros(M_pad, N_pad, dtype=torch.bfloat16), device)

    fn = make_ksplit(M_pad, K, N_pad, plan.block_cfg, plan.part_cfg)
    ttlang_s = time_runs(
        thunk=lambda: fn(a_k, w_k, out_k),
        cleanup=lambda _r: None,
        device=device,
        warmup=warmup,
        runs=runs,
    )

    result = ttnn.to_torch(out_k).float()[:M, :N]
    pcc_v = pcc(result, ref)

    a_ref = _to_dev(a_t, device)
    w_ref = _to_dev(w_t, device)
    ttnn_s = time_runs(
        thunk=lambda: ttnn.matmul(a_ref, w_ref, compute_kernel_config=TTNN_CFG),
        cleanup=ttnn.deallocate,
        device=device,
        warmup=warmup,
        runs=runs,
    )

    for t in (a_k, w_k, out_k, a_ref, w_ref):
        ttnn.deallocate(t)

    bm, bn, bk = plan.block_cfg
    Mp, Np, Kp = plan.part_cfg
    return {
        "label": label,
        "M": M,
        "K": K,
        "N": N,
        "bm": bm,
        "bn": bn,
        "bk": bk,
        "Mp": Mp,
        "Np": Np,
        "Kp": Kp,
        "cores": plan.cores,
        "iter_per_core": plan.iters_per_core,
        "pad": round(plan.pad_ratio, 4),
        "ttlang_ms": round(ttlang_s * 1000, 4),
        "ttnn_ms": round(ttnn_s * 1000, 4),
        "ratio": round(ttlang_s / ttnn_s, 4),
        "pcc": round(pcc_v, 6),
    }


def _format_row(r):
    return (
        f"{r['label']:<32}  "
        f"ttlang={r['ttlang_ms']:>8.3f}ms  ttnn={r['ttnn_ms']:>8.3f}ms  "
        f"ratio={r['ratio']:.3f}  pcc={r['pcc']:.4f}  "
        f"({r['bm']},{r['bn']},{r['bk']})/"
        f"({r['Mp']},{r['Np']},{r['Kp']}) cores={r['cores']}"
    )


def _plot_label(r):
    base = str(r["label"]).split(" (")[0].strip().replace(" x ", "×").replace("^3", "³")
    return f"{base}\n({r['bm']},{r['bn']},{r['bk']}) Kp={r['Kp']}\n{r['cores']} cores"


SPEC = BenchSpec(
    name="matmul",
    fields=FIELDS,
    cases=SHAPES,
    run_case=run_case,
    label_of=lambda case: case[3],
    open_device=_open_device,
    format_row=_format_row,
    plot_title="ttlang matmul vs ttnn.matmul  (bar = ratio, dotted = 1.0)",
    plot_label_of=_plot_label,
)
