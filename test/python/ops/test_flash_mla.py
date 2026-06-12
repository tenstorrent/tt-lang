# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Focused tests for ttl.ops.flash_mla.

``test_flash_shard_full`` runs the shard on a single core over the whole
K/V sequence (n_cols=1) and checks the normalized output against torch
SDPA. This exercises the online-softmax chunk loop in isolation, without
the cross-core tree reduce.

``test_flash_chain`` runs the full shard -> tree-reduce -> normalize chain
at toy shapes with an 8-way K split.

``test_flash_mla_decode`` runs the same chain at production MLA-decode
shapes (64 heads, kvpe_dim=576, kv_lora_rank=512, seq_len=32768) with a
bfp8 KV cache, against the MLA golden: a single KV head broadcast over all
heads, with V the leading ``kv_lora_rank`` columns of the shared KV cache.
The ops read plain DRAM tensors, so this matches the shapes / dtype / K-V
coupling / golden but not the on-device L1 sharding of a multi-core sharded
deployment.
"""

import math

import pytest
import torch

import ttl
from ttl.ops.flash_mla import (
    make_flash_shard,
    make_flash_tree_reduce,
    make_flash_normalize,
)

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram


def to_dram_bfp8(torch_tensor, device):
    """DRAM tensor in bfloat8_b (the KV-cache dtype). The shard typecasts
    K/V back to bf16 for the matmuls."""
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


TILE = ttnn.TILE_SIZE

PNHt = 1
DHt = 2
vDHt = 1
Sk_chunk_t = 2
N_CHUNKS = 4

PN = PNHt * TILE
D = DHt * TILE
vD = vDHt * TILE
S = Sk_chunk_t * N_CHUNKS * TILE


def test_flash_shard_full(device):
    scale = 1.0 / math.sqrt(D)

    q_t = torch.randn(PN, D, dtype=torch.bfloat16) * 0.1
    k_t = torch.randn(S, D, dtype=torch.bfloat16) * 0.1
    v_t = torch.randn(S, vD, dtype=torch.bfloat16) * 0.1

    scores = (q_t.float() @ k_t.float().T) * scale
    attn = torch.softmax(scores, dim=-1)
    expected = (attn @ v_t.float()).to(torch.bfloat16)

    q_d = to_dram(q_t, device)
    k_d = to_dram_bfp8(k_t, device)
    v_d = to_dram_bfp8(v_t, device)
    o_d = to_dram(torch.zeros(PN, vD, dtype=torch.bfloat16), device)
    m_d = to_dram(torch.zeros(PN, TILE, dtype=torch.bfloat16), device)
    l_d = to_dram(torch.zeros(PN, TILE, dtype=torch.bfloat16), device)

    shard = make_flash_shard(
        n_cols=1,
        B=1,
        PNHt=PNHt,
        DHt=DHt,
        vDHt=vDHt,
        Sk_chunk_t=Sk_chunk_t,
        N_CHUNKS=N_CHUNKS,
        scale=scale,
    )
    shard(q_d, k_d, v_d, o_d, m_d, l_d)

    o_unnorm = ttnn.to_torch(o_d).reshape(PN, vD).float()
    l = ttnn.to_torch(l_d).reshape(PN, TILE).float()[:, 0:1]
    got = (o_unnorm / l).to(torch.bfloat16)

    assert_pcc(expected, got, threshold=0.99)


# Full chain: 8-way K split across cores, tree-reduce the partials, normalize.
N_COLS = 8
CHAIN_Sk_chunk_t = 2
CHAIN_N_CHUNKS = 1
CHAIN_S = N_COLS * CHAIN_Sk_chunk_t * CHAIN_N_CHUNKS * TILE


def test_flash_chain(device):
    scale = 1.0 / math.sqrt(D)

    q_t = torch.randn(PN, D, dtype=torch.bfloat16) * 0.1
    k_t = torch.randn(CHAIN_S, D, dtype=torch.bfloat16) * 0.1
    v_t = torch.randn(CHAIN_S, vD, dtype=torch.bfloat16) * 0.1

    scores = (q_t.float() @ k_t.float().T) * scale
    attn = torch.softmax(scores, dim=-1)
    expected = (attn @ v_t.float()).to(torch.bfloat16)

    q_d = to_dram(q_t, device)
    k_d = to_dram_bfp8(k_t, device)
    v_d = to_dram_bfp8(v_t, device)
    # Per-core partials: N_COLS row-blocks of (PNHt, *) tiles.
    po_d = to_dram(torch.zeros(N_COLS * PN, vD, dtype=torch.bfloat16), device)
    pm_d = to_dram(torch.zeros(N_COLS * PN, TILE, dtype=torch.bfloat16), device)
    pl_d = to_dram(torch.zeros(N_COLS * PN, TILE, dtype=torch.bfloat16), device)
    # Merged unnormalized output + normalized output.
    o_d = to_dram(torch.zeros(PN, vD, dtype=torch.bfloat16), device)
    m_d = to_dram(torch.zeros(PN, TILE, dtype=torch.bfloat16), device)
    l_d = to_dram(torch.zeros(PN, TILE, dtype=torch.bfloat16), device)
    norm_d = to_dram(torch.zeros(PN, vD, dtype=torch.bfloat16), device)

    shard = make_flash_shard(
        n_cols=N_COLS,
        B=1,
        PNHt=PNHt,
        DHt=DHt,
        vDHt=vDHt,
        Sk_chunk_t=CHAIN_Sk_chunk_t,
        N_CHUNKS=CHAIN_N_CHUNKS,
        scale=scale,
    )
    tree_reduce = make_flash_tree_reduce(PNHt=PNHt, vDHt=vDHt, B=1)
    normalize = make_flash_normalize(grid=(1, 1), PNHt=PNHt, vDHt=vDHt)

    shard(q_d, k_d, v_d, po_d, pm_d, pl_d)
    tree_reduce(po_d, pm_d, pl_d, o_d, l_d)
    normalize(o_d, l_d, norm_d)

    got = ttnn.to_torch(norm_d).reshape(PN, vD).to(torch.bfloat16)
    assert_pcc(expected, got, threshold=0.99)


def flash_mla_golden(q, kv_cache, position_ids, head_dim_v, scale):
    """PyTorch reference for MLA decode: one shared KV head broadcast over
    all query heads; V is the leading ``head_dim_v`` columns of the cache."""
    batch_size = q.shape[1]
    num_heads = q.shape[2]
    kvpe_dim = q.shape[3]

    q = q.permute(1, 2, 0, 3)

    outputs = []
    for b in range(batch_size):
        seq_len = position_ids[b].item() + 1
        kv = kv_cache[b, :, :seq_len, :]
        kv_expanded = kv.expand(num_heads, seq_len, kvpe_dim)
        q_b = q[b]
        attn_scores = torch.matmul(q_b, kv_expanded.transpose(-2, -1)) * scale
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(q.dtype)
        v = kv_expanded[:, :, :head_dim_v]
        out_b = torch.matmul(attn_probs, v)
        outputs.append(out_b)

    output = torch.stack(outputs, dim=0)
    output = output.squeeze(2).unsqueeze(0)
    return output


# Production MLA-decode shapes (seq_len=32768, single decode token).
NUM_HEADS = 64
KV_LORA_RANK = 512  # head_dim_v
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
KVPE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
N_CORES = 8
MAX_SEQ_LEN = 32 * 1024

MLA_PNHt = NUM_HEADS // TILE  # 2  (64 query rows / 32)
MLA_DHt = KVPE_DIM // TILE  # 18
MLA_vDHt = KV_LORA_RANK // TILE  # 16
# The shard typecasts each bfp8 K/V chunk to a bf16 mirror, so a 2-tile
# compute chunk is the largest that fits one core's L1 at vDHt=16. This is
# the compute-chunk granularity, independent of the cache layout.
MLA_Sk_chunk_t = 1
# Per core: (seq / n_cores) positions, split into Sk_chunk_t-tile chunks.
MLA_N_CHUNKS = (MAX_SEQ_LEN // N_CORES) // (MLA_Sk_chunk_t * TILE)  # 64


# Production tile counts make the compute kernel large; trim worker L1 to
# enlarge the kernel-config buffer past the default 70656B TENSIX limit.
@pytest.mark.parametrize("ttnn_device", [{"worker_l1_size": 1448000}], indirect=True)
def test_flash_mla_decode(device):
    torch.manual_seed(42)
    scale = QK_HEAD_DIM**-0.5
    decode_position = MAX_SEQ_LEN - 1

    torch_q = torch.randn((1, 1, NUM_HEADS, KVPE_DIM), dtype=torch.bfloat16)
    torch_cache = torch.randn((1, 1, MAX_SEQ_LEN, KVPE_DIM), dtype=torch.bfloat16)
    position_ids = torch.ones(1, dtype=torch.int32) * decode_position

    expected = flash_mla_golden(
        q=torch_q,
        kv_cache=torch_cache,
        position_ids=position_ids,
        head_dim_v=KV_LORA_RANK,
        scale=scale,
    )

    # MLA shares one KV cache: K is the full kvpe_dim, V its leading
    # kv_lora_rank columns. All query rows attend the same K/V.
    q_2d = torch_q.reshape(NUM_HEADS, KVPE_DIM)
    k_2d = torch_cache.reshape(MAX_SEQ_LEN, KVPE_DIM)
    v_2d = k_2d[:, :KV_LORA_RANK].contiguous()

    PNr = MLA_PNHt * TILE
    q_d = to_dram(q_2d, device)
    k_d = to_dram_bfp8(k_2d, device)
    v_d = to_dram_bfp8(v_2d, device)
    po_d = to_dram(
        torch.zeros(N_CORES * PNr, KV_LORA_RANK, dtype=torch.bfloat16), device
    )
    pm_d = to_dram(torch.zeros(N_CORES * PNr, TILE, dtype=torch.bfloat16), device)
    pl_d = to_dram(torch.zeros(N_CORES * PNr, TILE, dtype=torch.bfloat16), device)
    o_d = to_dram(torch.zeros(PNr, KV_LORA_RANK, dtype=torch.bfloat16), device)
    m_d = to_dram(torch.zeros(PNr, TILE, dtype=torch.bfloat16), device)
    l_d = to_dram(torch.zeros(PNr, TILE, dtype=torch.bfloat16), device)
    norm_d = to_dram(torch.zeros(PNr, KV_LORA_RANK, dtype=torch.bfloat16), device)

    shard = make_flash_shard(
        n_cols=N_CORES,
        B=1,
        PNHt=MLA_PNHt,
        DHt=MLA_DHt,
        vDHt=MLA_vDHt,
        Sk_chunk_t=MLA_Sk_chunk_t,
        N_CHUNKS=MLA_N_CHUNKS,
        scale=scale,
    )
    tree_reduce = make_flash_tree_reduce(PNHt=MLA_PNHt, vDHt=MLA_vDHt, B=1)
    normalize = make_flash_normalize(grid=(1, 1), PNHt=MLA_PNHt, vDHt=MLA_vDHt)

    shard(q_d, k_d, v_d, po_d, pm_d, pl_d)
    tree_reduce(po_d, pm_d, pl_d, o_d, l_d)
    normalize(o_d, l_d, norm_d)

    got = (
        ttnn.to_torch(norm_d).reshape(1, 1, NUM_HEADS, KV_LORA_RANK).to(torch.bfloat16)
    )
    assert_pcc(expected, got, threshold=0.99)


@pytest.mark.parametrize("ttnn_device", [{"worker_l1_size": 1430000}], indirect=True)
@pytest.mark.parametrize("sk", [1, 2])
def test_flash_mla_decode_fused(device, sk):
    """The fully fused single-kernel MLA: q multicast, partials and merged
    stats carried through DFB bridges (no inter-phase DRAM), one launch.
    Swept over the 1- and 2-tile compute chunk, the largest that fits one
    fused kernel's L1."""
    from ttl.ops.flash_mla import make_flash_mla

    n_chunks = (MAX_SEQ_LEN // N_CORES) // (sk * TILE)

    torch.manual_seed(42)
    scale = QK_HEAD_DIM**-0.5
    decode_position = MAX_SEQ_LEN - 1

    torch_q = torch.randn((1, 1, NUM_HEADS, KVPE_DIM), dtype=torch.bfloat16)
    torch_cache = torch.randn((1, 1, MAX_SEQ_LEN, KVPE_DIM), dtype=torch.bfloat16)
    position_ids = torch.ones(1, dtype=torch.int32) * decode_position

    expected = flash_mla_golden(
        q=torch_q,
        kv_cache=torch_cache,
        position_ids=position_ids,
        head_dim_v=KV_LORA_RANK,
        scale=scale,
    )

    q_2d = torch_q.reshape(NUM_HEADS, KVPE_DIM)
    k_2d = torch_cache.reshape(MAX_SEQ_LEN, KVPE_DIM)
    v_2d = k_2d[:, :KV_LORA_RANK].contiguous()

    PNr = MLA_PNHt * TILE
    q_d = to_dram(q_2d, device)
    k_d = to_dram_bfp8(k_2d, device)
    v_d = to_dram_bfp8(v_2d, device)
    norm_d = to_dram(torch.zeros(PNr, KV_LORA_RANK, dtype=torch.bfloat16), device)

    flash = make_flash_mla(
        n_cols=N_CORES,
        B=1,
        PNHt=MLA_PNHt,
        DHt=MLA_DHt,
        vDHt=MLA_vDHt,
        Sk_chunk_t=sk,
        N_CHUNKS=n_chunks,
        scale=scale,
    )
    flash(q_d, k_d, v_d, norm_d)

    got = (
        ttnn.to_torch(norm_d).reshape(1, 1, NUM_HEADS, KV_LORA_RANK).to(torch.bfloat16)
    )
    assert_pcc(expected, got, threshold=0.99)
