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


def to_dev(t, device, dtype=ttnn.bfloat16, mem=None):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=mem or ttnn.DRAM_MEMORY_CONFIG)


def from_dev(t):
    return ttnn.to_torch(t).float()


def row(t, D, device):
    """Host [D] -> [TILE, D] tile row tensor on device."""
    z = torch.zeros(TILE, D, dtype=torch.bfloat16)
    z[0] = t.to(torch.bfloat16)
    return to_dev(z, device)
