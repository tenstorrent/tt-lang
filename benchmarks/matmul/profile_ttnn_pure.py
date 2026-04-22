# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure ttnn.matmul profile — no ttl kernel at all.

If the perf summary still shows a program named "summa_matmul", that name
is coming from tt-metal, not our ttl.operation.
"""

import torch
import ttnn


M, K, N = 4096, 4096, 4096

FP32_ACC = True
TTNN_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4 if FP32_ACC else ttnn.MathFidelity.HiFi2,
    fp32_dest_acc_en=FP32_ACC,
    packer_l1_acc=True,
)


def main():
    default_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(
        device_id=0,
        worker_l1_size=default_l1 - 131072,
    )
    try:
        torch.manual_seed(0)
        a_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
        w_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
        a = ttnn.from_torch(a_t.contiguous(), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(w_t.contiguous(), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out = ttnn.matmul(a, w, compute_kernel_config=TTNN_CFG)
        ttnn.synchronize_device(device)

        ttnn.deallocate(a)
        ttnn.deallocate(w)
        ttnn.deallocate(out)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
