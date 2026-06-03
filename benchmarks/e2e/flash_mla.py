# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Flash-MLA decode benchmark.

Times the full ttl chain (shard -> tree_reduce -> normalize) at production
MLA-decode shapes (64 heads, kvpe_dim 576, kv_lora_rank 512, bfp8 KV cache)
across a sweep of sequence lengths. PCC is checked against the torch MLA golden
(one shared KV head broadcast over all heads; V = leading kv_lora_rank columns).

There is no drop-in ttnn equivalent for MLA decode (single shared KV head,
asymmetric qk/v head dims), so the ratio column is a best-effort baseline
against ``ttnn.transformer.scaled_dot_product_attention_decode`` run as generic
MQA flash-decode; it is left blank when that op rejects the MLA shapes. The
absolute ttl latency + correctness are always recorded.

Run: ``python -m benchmarks.e2e.flash_mla [--filter 8k] [--plot]``.
"""

import math

import torch
import ttnn

from ttl.ops.flash_mla import (
    make_flash_shard,
    make_flash_tree_reduce,
    make_flash_normalize,
)

from benchmarks.common import BenchSpec, cli, pcc, time_runs

TILE = 32

NUM_HEADS = 64
KV_LORA_RANK = 512                       # head_dim_v
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
KVPE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM        # 576
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
N_CORES = 8

PNHt = NUM_HEADS // TILE                  # 2
DHt = KVPE_DIM // TILE                     # 18
vDHt = KV_LORA_RANK // TILE                # 16
Sk_chunk_t = 2

# Production tile counts make the compute kernel large; trim worker L1 to
# enlarge the kernel-config buffer past the default TENSIX limit.
WORKER_L1 = 1448000

# (seq_len, label). N_CHUNKS = (seq / N_CORES) / (Sk_chunk_t * TILE) must be an
# integer; all of these divide cleanly.
CASES = (
    (8 * 1024, "8k"),
    (16 * 1024, "16k"),
    (32 * 1024, "32k"),
)

FIELDS = ("label", "seq", "heads", "ttlang_ms", "ttnn_ms", "ratio", "pcc")


def _to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_dev_bfp8(t, device):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _golden(q, kv_cache, seq_len, head_dim_v, scale):
    """MLA decode: one shared KV head over all query heads; V = leading
    head_dim_v columns of the cache."""
    num_heads = q.shape[2]
    kvpe_dim = q.shape[3]
    q = q.permute(1, 2, 0, 3)[0]  # (num_heads, 1, kvpe_dim)
    kv = kv_cache[0, :, :seq_len, :].expand(num_heads, seq_len, kvpe_dim)
    scores = torch.matmul(q, kv.transpose(-2, -1)) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(probs, kv[:, :, :head_dim_v])
    return out.squeeze(1).reshape(1, 1, num_heads, head_dim_v)


def _open_device():
    return ttnn.open_device(device_id=0, worker_l1_size=WORKER_L1)


def _reference_ms(device, q4, cache4, seq, scale, *, warmup, runs):
    """Best-effort ttnn flash-decode baseline; None if it rejects the shapes."""
    try:
        q_ref = _to_dev(q4, device)                 # [1, b, nh, dh]
        kv_ref = _to_dev(cache4, device)            # [b, nkv, s, dh]
        v_ref = _to_dev(cache4, device)
        cur_pos = [seq - 1]

        def ref():
            return ttnn.transformer.scaled_dot_product_attention_decode(
                q_ref, kv_ref, v_ref, is_causal=False, cur_pos=cur_pos, scale=scale
            )

        s = time_runs(ref, ttnn.deallocate, device, warmup=warmup, runs=runs)
        for t in (q_ref, kv_ref, v_ref):
            ttnn.deallocate(t)
        return s
    except Exception as e:
        print(f"  (ttnn sdpa-decode baseline unavailable: {e})", flush=True)
        return None


def run_case(device, case, *, warmup, runs):
    seq, label = case
    n_chunks = (seq // N_CORES) // (Sk_chunk_t * TILE)
    scale = QK_HEAD_DIM ** -0.5

    torch.manual_seed(42)
    q4 = torch.randn((1, 1, NUM_HEADS, KVPE_DIM), dtype=torch.bfloat16)
    cache4 = torch.randn((1, 1, seq, KVPE_DIM), dtype=torch.bfloat16)
    expected = _golden(q4, cache4, seq, KV_LORA_RANK, scale)

    q_2d = q4.reshape(NUM_HEADS, KVPE_DIM)
    k_2d = cache4.reshape(seq, KVPE_DIM)
    v_2d = k_2d[:, :KV_LORA_RANK].contiguous()
    PNr = PNHt * TILE

    q_d = _to_dev(q_2d, device)
    k_d = _to_dev_bfp8(k_2d, device)
    v_d = _to_dev_bfp8(v_2d, device)
    po_d = _to_dev(torch.zeros(N_CORES * PNr, KV_LORA_RANK, dtype=torch.bfloat16), device)
    pm_d = _to_dev(torch.zeros(N_CORES * PNr, TILE, dtype=torch.bfloat16), device)
    pl_d = _to_dev(torch.zeros(N_CORES * PNr, TILE, dtype=torch.bfloat16), device)
    o_d = _to_dev(torch.zeros(PNr, KV_LORA_RANK, dtype=torch.bfloat16), device)
    m_d = _to_dev(torch.zeros(PNr, TILE, dtype=torch.bfloat16), device)
    l_d = _to_dev(torch.zeros(PNr, TILE, dtype=torch.bfloat16), device)
    norm_d = _to_dev(torch.zeros(PNr, KV_LORA_RANK, dtype=torch.bfloat16), device)

    shard = make_flash_shard(
        n_cols=N_CORES, B=1, PNHt=PNHt, DHt=DHt, vDHt=vDHt,
        Sk_chunk_t=Sk_chunk_t, N_CHUNKS=n_chunks, scale=scale,
    )
    tree_reduce = make_flash_tree_reduce(PNHt=PNHt, vDHt=vDHt, B=1)
    normalize = make_flash_normalize(grid=(1, 1), PNHt=PNHt, vDHt=vDHt)

    def chain():
        shard(q_d, k_d, v_d, po_d, pm_d, pl_d)
        tree_reduce(po_d, pm_d, pl_d, o_d, m_d, l_d)
        normalize(o_d, l_d, norm_d)

    ttlang_s = time_runs(chain, lambda _r: None, device, warmup=warmup, runs=runs)

    got = ttnn.to_torch(norm_d).reshape(1, 1, NUM_HEADS, KV_LORA_RANK).to(torch.bfloat16)
    pcc_v = pcc(got, expected)

    ttnn_s = _reference_ms(device, q4, cache4, seq, scale, warmup=warmup, runs=runs)

    for t in (q_d, k_d, v_d, po_d, pm_d, pl_d, o_d, m_d, l_d, norm_d):
        ttnn.deallocate(t)

    ttnn_ms = None if ttnn_s is None else round(ttnn_s * 1000, 4)
    ratio = None if ttnn_s is None else round(ttlang_s / ttnn_s, 4)
    return {
        "label": label,
        "seq": seq,
        "heads": NUM_HEADS,
        "ttlang_ms": round(ttlang_s * 1000, 4),
        "ttnn_ms": ttnn_ms,
        "ratio": ratio,
        "pcc": round(pcc_v, 6),
    }


def _format_row(r):
    ref = "n/a" if r["ttnn_ms"] is None else f"{r['ttnn_ms']:>8.3f}ms"
    ratio = "  n/a " if r["ratio"] is None else f"{r['ratio']:.3f}"
    return (
        f"seq={r['label']:<6}  "
        f"ttlang={r['ttlang_ms']:>8.3f}ms  ttnn={ref}  "
        f"ratio={ratio}  pcc={r['pcc']:.4f}"
    )


SPEC = BenchSpec(
    name="flash_mla",
    fields=FIELDS,
    cases=CASES,
    run_case=run_case,
    label_of=lambda case: case[1],
    open_device=_open_device,
    format_row=_format_row,
    plot_title="ttlang flash-MLA decode vs ttnn sdpa-decode  (bar = ratio)",
    plot_label_of=lambda r: f"seq={r['label']}\n{r['heads']} heads",
)


if __name__ == "__main__":
    cli(SPEC)
