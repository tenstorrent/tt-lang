# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""TP=4 mesh smoke: the three sharding patterns the decode chain needs.

1. O-proj style: x col-shard + W row-shard per card, gemv partials,
   ttnn.all_reduce -> full result replicated.
2. Per-card constants staged via ShardTensorToMesh (expert base offsets).
3. all_gather of a tile row (lm_head winner merge).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: F401
from ttl.ops.gemv import make_gemv

TILE, TP = 32, 4


def to_mesh(t, mesh, mapper):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=mapper)


def shards(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def test_tp_smoke():
    if ttnn.GetNumAvailableDevices() < TP:
        pytest.skip("needs 4 devices")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    try:
        _run(mesh)
    finally:
        ttnn.close_device(mesh)


def _run(mesh):
    torch.manual_seed(0)
    K, N = 1024, 2816
    Kl = K // TP
    x = torch.randn(K) * 0.5
    w = torch.randn(K, N) * 0.02
    want = x @ w

    xs = torch.zeros(TP * TILE, Kl, dtype=torch.bfloat16)
    for c in range(TP):
        xs[c * TILE] = x[c * Kl:(c + 1) * Kl].to(torch.bfloat16)
    x_d = to_mesh(xs, mesh, ttnn.ShardTensorToMesh(mesh, dim=0))
    w_d = to_mesh(w.to(torch.bfloat16), mesh, ttnn.ShardTensorToMesh(mesh, dim=0))
    out = torch.zeros(TP * TILE, N, dtype=torch.bfloat16)
    out_d = to_mesh(out, mesh, ttnn.ShardTensorToMesh(mesh, dim=0))

    make_gemv(TILE, Kl, N, (8, 2), 11)(x_d, w_d, out_d)
    got = ttnn.to_torch(ttnn.all_reduce(out_d),
                        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    for c in range(TP):
        pcc = torch.corrcoef(torch.stack([got[c * TILE], want]))[0, 1].item()
        assert pcc > 0.999, f"card {c} pcc {pcc}"

    base = torch.arange(TP).float().unsqueeze(1).repeat(1, TILE).reshape(TP, 1, TILE)
    base = base.expand(TP, TILE, TILE).reshape(TP * TILE, TILE)
    base_d = to_mesh(base, mesh, ttnn.ShardTensorToMesh(mesh, dim=0))
    back = shards(base_d, mesh)
    for c in range(TP):
        assert back[c * TILE, 0].item() == c, f"card {c} base {back[c * TILE, 0]}"

    g = ttnn.all_gather(base_d, dim=1)
    gt = shards(g, mesh)
    assert gt.shape == (TP * TILE, TP * TILE)
    for c in range(TP):
        row = gt[c * TILE, ::TILE]
        assert row.tolist() == [0.0, 1.0, 2.0, 3.0], f"card {c} gather {row.tolist()}"
