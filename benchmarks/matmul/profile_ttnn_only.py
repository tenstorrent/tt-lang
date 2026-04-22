# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profile JUST ttnn.matmul on 4k³ with no ttl kernel to disambiguate naming.

Requires two runs per the ttnn-logs-save-after convention: first run writes
ttnn logs, second run is picked up by whatever triggers the perf dump.
Run with --perf --hw.
"""

import torch
import ttnn

from summa_kernel import make_kernel as make_summa_kernel


M, K, N = 4096, 4096, 4096

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
        a = to_dev(torch.randn(M, K, dtype=torch.bfloat16) * 0.02, device)
        w = to_dev(torch.randn(K, N, dtype=torch.bfloat16) * 0.02, device)

        # Run ttnn.matmul once — no ttl kernel to collide names.
        print("Running ttnn.matmul only (no ttl kernel)", flush=True)
        out = ttnn.matmul(a, w, compute_kernel_config=TTNN_CFG)
        ttnn.synchronize_device(device)

        # Tiny ttl kernel (1x1 identity-like) just to trigger the perf dump.
        # Use a trivial shape so its profile is obvious and can be discounted.
        trivial_a = to_dev(torch.ones(32, 32, dtype=torch.bfloat16), device)
        trivial_w = to_dev(torch.ones(32, 32, dtype=torch.bfloat16), device)
        trivial_o = to_dev(torch.zeros(32, 32, dtype=torch.bfloat16), device)
        trivial_fn = make_summa_kernel(32, 32, 32, (1, 1, 1), (1, 1, 1))
        trivial_fn(trivial_a, trivial_w, trivial_o)
        ttnn.synchronize_device(device)

        for t in (a, w, out, trivial_a, trivial_w, trivial_o):
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
