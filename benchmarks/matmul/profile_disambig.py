# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Disambiguation test: ttnn 4k³ + our 2x2=4-core ttl kernel on 512^2 shape.

If a 130-core "summa_matmul" shows up in the profile, it cannot be ours.
"""

import torch
import ttnn

from summa_kernel import make_kernel as make_summa_kernel


# ttnn on 4k³
TTNN_M, TTNN_K, TTNN_N = 4096, 4096, 4096

# Our ttl kernel: tiny 2x2 core grid
OUR_M, OUR_K, OUR_N = 256, 256, 256  # 8x8x8 tiles
OUR_BLOCK = (4, 4, 8)  # Mb = 8/4=2, Nb=2
OUR_PART = (2, 2, 1)  # 2x2 = 4 cores

FP32_ACC = True
TTNN_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4 if FP32_ACC else ttnn.MathFidelity.HiFi2,
    fp32_dest_acc_en=FP32_ACC,
    packer_l1_acc=True,
)


def to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    default_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(
        device_id=0,
        worker_l1_size=default_l1 - 131072,
    )
    try:
        torch.manual_seed(0)

        # ttnn 4k³
        a = to_dev(torch.randn(TTNN_M, TTNN_K, dtype=torch.bfloat16) * 0.02, device)
        w = to_dev(torch.randn(TTNN_K, TTNN_N, dtype=torch.bfloat16) * 0.02, device)
        out = ttnn.matmul(a, w, compute_kernel_config=TTNN_CFG)
        ttnn.synchronize_device(device)

        # Our ttl kernel: 4 cores on tiny input
        our_a = to_dev(torch.randn(OUR_M, OUR_K, dtype=torch.bfloat16) * 0.02, device)
        our_w = to_dev(torch.randn(OUR_K, OUR_N, dtype=torch.bfloat16) * 0.02, device)
        our_out = to_dev(torch.zeros(OUR_M, OUR_N, dtype=torch.bfloat16), device)

        fn = make_summa_kernel(OUR_M, OUR_K, OUR_N, OUR_BLOCK, OUR_PART)
        fn(our_a, our_w, our_out)
        ttnn.synchronize_device(device)

        for t in (a, w, out, our_a, our_w, our_out):
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
