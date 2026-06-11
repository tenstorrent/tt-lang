# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-core subblock cycle sweep -- add or matmul (fixed block, vary subblock).

One core does one FIXED block; only the forced subblock (sR, sC) changes
(``--ttl-force-subblock``), so every run is the same total work. Compute-isolated
(no-DRAM). Each shape runs in its own device open/close (Tracy writes the device
CSV on close; every run clears the log first), and the shapes are ranked by
device cycles.

    TT_METAL_DEVICE_PROFILER=1 python3 -m benchmarks.cycles.subblock --kind add
    TT_METAL_DEVICE_PROFILER=1 python3 -m benchmarks.cycles.subblock --kind matmul
"""

import argparse

import ttnn

from benchmarks.common import (
    clear_profile_log,
    parse_kernel_duration,
    read_device_profiler,
)

from .kinds import DST_FULL_SYNC_EN, KINDS

FIELDS = ("kind", "subblock", "cycles", "us", "ratio", "bottleneck", "per_risc", "error")


def _bottleneck(per_risc):
    """RISC with the largest kernel span (the effective duration driver)."""
    return max(per_risc, key=per_risc.get) if per_risc else "?"


def _error_summary(e):
    """Pull the meaningful diagnostic out of a compile exception. The ttl
    RuntimeError's first line is a version banner ('ttlang 0.0.0+unknown'); the
    real message is the 'error:' line below it."""
    lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
    for ln in lines:
        if ln.lower().startswith("error:"):
            return ln
    for ln in lines:
        if not ln.startswith("ttlang "):
            return ln
    return lines[0] if lines else type(e).__name__


def _label(subblock):
    return "auto" if subblock is None else "x".join(str(s) for s in subblock)


def run(kind, subblock, dst_full_sync_en=DST_FULL_SYNC_EN):
    """Run one subblock (or None = heuristic) for a Kind on one core."""
    device = ttnn.open_device(device_id=0)
    try:
        tensors = kind.make_tensors(device)
        op = kind.make_op(subblock, dst_full_sync_en)
        clear_profile_log()
        op(*tensors)
        ttnn.synchronize_device(device)
        read_device_profiler(device)
    finally:
        # close_device flushes the device profiler CSV to disk.
        ttnn.close_device(device)
    return parse_kernel_duration()


def run_case(kind, subblock, dst_full_sync_en):
    """Compile + run one subblock. A subblock the compiler rejects (e.g. over the
    DST budget) raises during compilation; we catch it and return an 'invalid'
    row (cycles=None) instead of pre-filtering valid subblocks ourselves."""
    label = _label(subblock)
    try:
        d = run(kind, subblock, dst_full_sync_en)
    except Exception as e:
        return {
            "kind": kind.name,
            "subblock": label,
            "cycles": None,
            "us": None,
            "per_risc": {},
            "bottleneck": "invalid",
            "ratio": None,
            "error": _error_summary(e),
        }
    return {
        "kind": kind.name,
        "subblock": label,
        "cycles": d["cycles"],
        "us": round(d["us"], 2),
        "per_risc": d["per_risc"],
        "bottleneck": _bottleneck(d["per_risc"]),
    }


def sweep(kind_name="add", filter=None, dst_full_sync_en=DST_FULL_SYNC_EN):
    """Run every subblock for a kind (compiler decides validity); return rows
    with the valid ones ranked by cycles first, then the invalid ones."""
    kind = KINDS[kind_name]
    valid, invalid = [], []
    for subblock in kind.subblocks():
        label = _label(subblock)
        if filter and filter not in label:
            continue
        row = run_case(kind, subblock, dst_full_sync_en)
        (invalid if row["cycles"] is None else valid).append(row)

    valid.sort(key=lambda r: r["cycles"])
    fastest = valid[0]["cycles"] if valid else 1
    for r in valid:
        r["ratio"] = round(r["cycles"] / fastest, 3)

    print(
        f"=== single-core {kind.name} subblock sweep  fixed block={kind.block_label}  "
        f"dst_full_sync={dst_full_sync_en}  "
        f"({len(valid)} valid / {len(invalid)} invalid) ===",
        flush=True,
    )
    for r in valid:
        print(
            f"  subblock {r['subblock']:<6} {r['cycles']:>10,} cyc  "
            f"({r['ratio']:.3f}x)  {r['us']:>8.2f} us  bottleneck={r['bottleneck']}",
            flush=True,
        )
    for r in invalid:
        print(
            f"  subblock {r['subblock']:<6} {'invalid':>10}  "
            f"(compiler rejected: {r.get('error', '')})",
            flush=True,
        )
    return valid + invalid


_TITLES = {
    "add": "single-core add subblock: device cycles (fixed block, y = a + b)",
    "matmul": "single-core matmul subblock: device cycles (fixed block, y = a @ b)",
    "bcast_add": "single-core bcast_add subblock: device cycles (fixed block, out = bcast_col(b) + a)",
    "adversarial": "single-core adversarial subblock: device cycles (fixed block, 4-in/4-out fused)",
    "comprehensive": "single-core comprehensive subblock: device cycles (fixed block, 3-in/3-out 20-op fused)",
    "silu": "single-core silu subblock: device cycles (fixed block, out = y * sigmoid(y), copy_dst)",
    "rsqrt_abs": "single-core rsqrt_abs subblock: device cycles (fixed block, out = x * rsqrt(abs(x)), copy_dst)",
    "axby": "single-core axby subblock: device cycles (fixed block, out = a*x + b*y)",
    "mc_silu": "single-core mc_silu subblock: device cycles (fixed block, out = x * sigmoid(x))",
    "mc_unary_binary": "single-core mc_unary_binary subblock: device cycles (abs(x) + (x+y) + (x*y))",
    "mc_three": "single-core mc_three subblock: device cycles (sigmoid(a), exp(a), a+b -> 3 outs)",
    "mc_square": "single-core mc_square subblock: device cycles (out = x * x)",
    "mc_branch": "single-core mc_branch subblock: device cycles (exp(abs(a)), abs(a)+b -> 2 outs)",
    "fill_add": "single-core fill_add subblock: device cycles (fixed block, out = inp + fill(1.0))",
    "fill": "single-core fill subblock: device cycles (fixed block, out = fill(-3.0))",
    "gdn": "single-core gdn subblock: device cycles (DxD block, gated delta rule step)",
    "matmul_bias": "single-core matmul_bias subblock: device cycles (fixed block, out = a @ b + c)",
    "matmul_relu": "single-core matmul_relu subblock: device cycles (fixed block, out = relu(a @ b))",
    "reduce_sum": "single-core reduce_sum subblock: device cycles (8x8 -> 8x1, dims=[1], 1-D force)",
    "reduce_max": "single-core reduce_max subblock: device cycles (8x8 -> 8x1, dims=[1], 1-D force)",
    "transpose": "single-core transpose subblock: device cycles (fixed block, out = inp^T)",
}


def panel(kind_name, rows):
    """Panel hook for the unified driver. 'ratio' is cycles vs the fastest
    subblock (not vs a reference); the ratio-bar plot is the same shape.
    Invalid (compiler-rejected) subblocks have no cycles, so drop them."""
    return {
        "rows": [r for r in rows if r.get("cycles") is not None],
        "title": _TITLES.get(kind_name, f"subblock {kind_name}"),
        "ylabel": "cycles / fastest subblock  (lower is better)",
        "ratio_key": "ratio",
        "label_fn": lambda r: r["subblock"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="single-core subblock cycle sweep (add | matmul)")
    ap.add_argument("--kind", choices=sorted(KINDS), default="add")
    ap.add_argument("--filter", default=None, help="substring to select a subblock by label")
    args = ap.parse_args(argv)
    sweep(args.kind, filter=args.filter)


if __name__ == "__main__":
    main()
