# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#
# Tutorial Step 0: TT-NN Baseline
# ================================
# This is the starting point: a matmul-bias-activation expressed entirely in
# TT-NN.  No custom operation is involved.  TT-NN dispatches each op separately,
# resulting in multiple DRAM round-trips.
#
# The operation: y = relu(a @ b + c)
#
# The subsequent tutorial steps replace this entire computation with a single
# fused TT-Lang operation, showing how to take control of data movement and
# compute explicitly.

import ttnn
import torch


def from_torch(tensor: torch.Tensor):

    # Upload a bfloat16 torch tensor to DRAM on the device in tiled layout.

    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


torch.manual_seed(42)

device = ttnn.open_device(device_id=0)

try:
    M, K, N = 8192, 8192, 8192

    a = torch.randn((M, K), dtype=torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.bfloat16)
    c = torch.randn((M, N), dtype=torch.bfloat16)

    expected_y = torch.relu(a @ b + c)

    a = from_torch(a)
    b = from_torch(b)
    c = from_torch(c)

    # TT-NN dispatches three separate operations: matmul, add, relu.
    # With a custom TT-Lang operation we can fuse all three into a single
    # kernel, reducing DRAM traffic and operation-launch overhead.

    y = ttnn.relu(ttnn.add(ttnn.matmul(a, b), c))

    y = ttnn.to_torch(y)

    pcc = torch.corrcoef(
        torch.stack([y.flatten().float(), expected_y.flatten().float()])
    )[0, 1].item()

    print(f"PCC {pcc:.6f}")

    assert pcc > 0.99

finally:
    ttnn.close_device(device)
