# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host staging helpers: weight upload before generation, readback after.

Decode-step host involvement is zero; only step-invariant tensors pass
through here.
"""

import torch
import ttnn

TILE = 32

# 2112 / 4 col-shard padded to tile alignment (Nt=18 keeps bands divisible).
MLP_PAD = 576


def is_mesh(device):
    return getattr(device, "get_num_devices", lambda: 1)() > 1


def to_dev(t, device, dtype=ttnn.bfloat16, mem=None, shard=False):
    """Stage to device; on a mesh replicate, or row-shard a per-card stack."""
    mapper = None
    if is_mesh(device):
        mapper = (ttnn.ShardTensorToMesh(device, dim=0) if shard
                  else ttnn.ReplicateTensorToMesh(device))
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=mem or ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)


def shard_cards(tensors, device, dtype=ttnn.bfloat16):
    """Per-card host tensors -> one row-stacked mesh tensor."""
    return to_dev(torch.cat([t.to(torch.bfloat16) for t in tensors]),
                  device, dtype, shard=True)


def from_dev(t, card=0):
    mesh = t.device()
    if not is_mesh(mesh):
        return ttnn.to_torch(t).float()
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    rows = full.shape[0] // mesh.get_num_devices()
    return full[card * rows:(card + 1) * rows].float()


def row(t, D, device):
    """Host [D] -> [TILE, D] tile row tensor on device."""
    z = torch.zeros(TILE, D, dtype=torch.bfloat16)
    z[0] = t.to(torch.bfloat16)
    return to_dev(z, device)
