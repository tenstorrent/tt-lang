# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Flash-MLA decode benchmark.

The reference is ``ttnn.transformer.paged_flash_multi_latent_attention_decode``
run in the production layout (paged KV cache + height-sharded Q), mirroring the
deepseek_v3 MLA demo. Cases:

  - ``ctx-*``: single-user decode (1 user, 64 heads) at a fixed 32k KV cache,
    sweeping the decode position so the attended context grows 512 -> 32k. The
    ttl chain (shard -> tree_reduce -> normalize) processes exactly the attended
    length, is PCC-checked against a torch MLA golden, and the ratio is taken
    against the ttnn paged decode reading the same position via ``cur_pos``.
    Short contexts are where ttl's fixed per-decode overhead is most exposed.
  - ``deepseek-1k``: the deepseek_v3 demo's exact problem (4 users, 128 heads,
    1k paged context). Our single-user op does not run multi-user batched
    decode, so this measures the ttnn side only; ttl columns are blank.

The ttl chain is compile-time fixed on its chunk count, so it only runs when the
attended length is a multiple of 512 (its per-core chunk granularity).

Run: ``python -m benchmarks.e2e.flash_mla [--filter ctx-8k] [--plot]``.
"""

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
N_CORES = 8                              # ttl K-split (n_cols)

PNHt = NUM_HEADS // TILE                  # 2
DHt = KVPE_DIM // TILE                     # 18
vDHt = KV_LORA_RANK // TILE                # 16
Sk_chunk_t = 2

# Attended length must be a multiple of the ttl op's per-core chunk.
TTL_CHUNK = N_CORES * Sk_chunk_t * TILE   # 512

# Production tile counts make the compute kernel large; trim worker L1 to
# enlarge the kernel-config buffer past the default TENSIX limit.
WORKER_L1 = 1448000

OURS_SCALE = QK_HEAD_DIM ** -0.5                   # 192 ** -0.5
DEEPSEEK_SCALE = (192 + 64) ** -0.5                # deepseek qk_head_dim = 256

CACHE_32K = 32 * 1024

# Each case is a dict: label, users, heads, cache (KV-cache length), pos
# (per-user decode positions), scale.
CASES = (
    {"label": "ctx-512", "users": 1, "heads": NUM_HEADS, "cache": CACHE_32K, "pos": [511], "scale": OURS_SCALE},
    {"label": "ctx-1k", "users": 1, "heads": NUM_HEADS, "cache": CACHE_32K, "pos": [1023], "scale": OURS_SCALE},
    {"label": "ctx-2k", "users": 1, "heads": NUM_HEADS, "cache": CACHE_32K, "pos": [2047], "scale": OURS_SCALE},
    {"label": "ctx-8k", "users": 1, "heads": NUM_HEADS, "cache": CACHE_32K, "pos": [8191], "scale": OURS_SCALE},
    {"label": "ctx-32k", "users": 1, "heads": NUM_HEADS, "cache": CACHE_32K, "pos": [32767], "scale": OURS_SCALE},
    # The deepseek demo's exact 4-user / 128-head / 1k problem (ttnn-side only:
    # our op reads one shared K/V so it can't run 4 distinct users, and its
    # 128-head shard overflows L1 -- it puts every query-head tile on each core,
    # so it tops out at 64 heads / PNHt=2, which the ctx-* cases already cover).
    {"label": "deepseek-1k", "users": 4, "heads": 128, "cache": 1024, "pos": [0, 170, 341, 512], "scale": DEEPSEEK_SCALE},
)

FIELDS = ("label", "users", "heads", "ctx", "ttlang_ms", "ttnn_ms", "ratio", "pcc")


def _to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_dev_bfp8(t, device):
    return ttnn.from_torch(
        t.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
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


# --------------------------------------------------------------------------
# ttl side: the shard -> tree_reduce -> normalize chain (single user)
# --------------------------------------------------------------------------
def _run_ttl_chain(device, num_heads, seq, scale, *, warmup, runs):
    pnht = num_heads // TILE
    n_chunks = (seq // N_CORES) // (Sk_chunk_t * TILE)

    torch.manual_seed(42)
    q4 = torch.randn((1, 1, num_heads, KVPE_DIM), dtype=torch.bfloat16)
    cache4 = torch.randn((1, 1, seq, KVPE_DIM), dtype=torch.bfloat16)
    expected = _golden(q4, cache4, seq, KV_LORA_RANK, scale)

    q_2d = q4.reshape(num_heads, KVPE_DIM)
    k_2d = cache4.reshape(seq, KVPE_DIM)
    v_2d = k_2d[:, :KV_LORA_RANK].contiguous()
    PNr = pnht * TILE

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
        n_cols=N_CORES, B=1, PNHt=pnht, DHt=DHt, vDHt=vDHt,
        Sk_chunk_t=Sk_chunk_t, N_CHUNKS=n_chunks, scale=scale,
    )
    tree_reduce = make_flash_tree_reduce(PNHt=pnht, vDHt=vDHt, B=1)
    normalize = make_flash_normalize(grid=(1, 1), PNHt=pnht, vDHt=vDHt)

    def chain():
        shard(q_d, k_d, v_d, po_d, pm_d, pl_d)
        tree_reduce(po_d, pm_d, pl_d, o_d, m_d, l_d)
        normalize(o_d, l_d, norm_d)

    ttlang_s = time_runs(chain, lambda _r: None, device, warmup=warmup, runs=runs)
    got = ttnn.to_torch(norm_d).reshape(1, 1, num_heads, KV_LORA_RANK).to(torch.bfloat16)
    pcc_v = pcc(got, expected)

    for t in (q_d, k_d, v_d, po_d, pm_d, pl_d, o_d, m_d, l_d, norm_d):
        ttnn.deallocate(t)
    return ttlang_s, pcc_v


# --------------------------------------------------------------------------
# ttnn side: faithful paged + height-sharded MLA decode (deepseek layout)
# --------------------------------------------------------------------------
def _build_page_table(device, num_users, num_blocks):
    blocks_per_user = num_blocks // num_users
    pt = torch.randperm(num_blocks, dtype=torch.int32).reshape(num_users, blocks_per_user)
    tt = ttnn.from_torch(
        pt, dtype=ttnn.int32, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt, pt


def _build_paged_cache(device, num_users, seq, head_dim, num_blocks, block_size, mapping):
    cache = torch.randn((num_users, 1, seq, head_dim), dtype=torch.bfloat16) * 0.1
    paged = (
        cache.reshape(num_users, 1, -1, block_size, head_dim)
        .transpose(1, 2)
        .reshape(num_blocks, 1, block_size, head_dim)
    )
    paged = paged[torch.argsort(mapping.view(-1))]
    return ttnn.from_torch(
        paged, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _reference_ms(device, num_users, num_heads, cache_seq, positions, scale, *, warmup, runs):
    """Time the production paged + height-sharded ttnn MLA decode reading the
    given per-user positions. Sharding keeps the per-core CBs small so it fits
    L1. Returns mean ms, or None if the op rejects the shapes."""
    try:
        block_size = TILE
        num_blocks = (cache_seq * num_users) // block_size
        num_cores = min(num_users * num_heads, 64)  # deepseek demo core cap

        torch.manual_seed(0)
        q = torch.randn((1, num_users, num_heads, KVPE_DIM), dtype=torch.bfloat16) * 0.1
        tt_pt, pt = _build_page_table(device, num_users, num_blocks)
        tt_cache = _build_paged_cache(
            device, num_users, cache_seq, KVPE_DIM, num_blocks, block_size, pt
        )

        grid = device.compute_with_storage_grid_size()
        q_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
        q_mem = ttnn.create_sharded_memory_config(
            shape=[TILE, KVPE_DIM], core_grid=q_grid,
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        out_mem = ttnn.create_sharded_memory_config(
            shape=[TILE, KV_LORA_RANK], core_grid=q_grid,
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tt_q = ttnn.to_memory_config(
            ttnn.from_torch(
                q, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
            q_mem,
        )
        tt_pos = ttnn.from_torch(
            torch.tensor(positions, dtype=torch.int32),
            dtype=ttnn.int32, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=0,        # unused in decode
            k_chunk_size=128,
            exp_approx_mode=True,
        )
        ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        def ref():
            return ttnn.transformer.paged_flash_multi_latent_attention_decode(
                tt_q, tt_cache,
                page_table_tensor=tt_pt,
                cur_pos_tensor=tt_pos,
                head_dim_v=KV_LORA_RANK,
                scale=scale,
                program_config=pc,
                compute_kernel_config=ckc,
                memory_config=out_mem,
            )

        s = time_runs(ref, ttnn.deallocate, device, warmup=warmup, runs=runs)
        for t in (tt_q, tt_cache, tt_pt, tt_pos):
            ttnn.deallocate(t)
        return s
    except Exception as e:
        print(f"  (ttnn paged MLA-decode baseline unavailable: {e})", flush=True)
        return None


def run_case(device, case, *, warmup, runs):
    num_users = case["users"]
    num_heads = case["heads"]
    cache_seq = case["cache"]
    positions = case["pos"]
    scale = case["scale"]
    label = case["label"]

    ttlang_s = pcc_v = None
    # The ttl chain is single-user (the op reads one shared K/V; B>1 only
    # replicates it) and compile-time fixed on its chunk count, so run it only
    # for one user and only when the attended length is a multiple of its chunk.
    if num_users == 1:
        attended = positions[0] + 1
        if attended % TTL_CHUNK == 0:
            try:
                ttlang_s, pcc_v = _run_ttl_chain(
                    device, num_heads, attended, scale, warmup=warmup, runs=runs
                )
            except Exception as e:
                print(f"  ({label}: ttl chain failed: {e})", flush=True)
        else:
            print(f"  ({label}: ttl skipped, attended {attended} not a multiple of {TTL_CHUNK})", flush=True)

    ttnn_s = _reference_ms(
        device, num_users, num_heads, cache_seq, positions, scale, warmup=warmup, runs=runs
    )

    ratio = round(ttlang_s / ttnn_s, 4) if (ttlang_s is not None and ttnn_s) else None
    return {
        "label": label,
        "users": num_users,
        "heads": num_heads,
        "ctx": max(positions) + 1,
        "ttlang_ms": None if ttlang_s is None else round(ttlang_s * 1000, 4),
        "ttnn_ms": None if ttnn_s is None else round(ttnn_s * 1000, 4),
        "ratio": ratio,
        "pcc": None if pcc_v is None else round(pcc_v, 6),
    }


def _format_row(r):
    tl = "     n/a" if r["ttlang_ms"] is None else f"{r['ttlang_ms']:>8.3f}ms"
    rf = "     n/a" if r["ttnn_ms"] is None else f"{r['ttnn_ms']:>8.3f}ms"
    ratio = "  n/a " if r["ratio"] is None else f"{r['ratio']:.3f}"
    p = " n/a  " if r["pcc"] is None else f"{r['pcc']:.4f}"
    return (
        f"{r['label']:<12} u{r['users']} h{r['heads']:<3} ctx{r['ctx']:<6}  "
        f"ttlang={tl}  ttnn={rf}  ratio={ratio}  pcc={p}"
    )


SPEC = BenchSpec(
    name="flash_mla",
    fields=FIELDS,
    cases=CASES,
    run_case=run_case,
    label_of=lambda case: case["label"],
    open_device=_open_device,
    format_row=_format_row,
    plot_title="ttlang flash-MLA decode vs ttnn paged MLA-decode  (bar = ratio)",
    plot_label_of=lambda r: f"{r['label']}\nu{r['users']} h{r['heads']}",
)


if __name__ == "__main__":
    cli(SPEC)
