# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profile ttnn.matmul on 4k³ alongside our summa_matmul for comparison.

Both run once each. The named ttlang kernel (summa_matmul) triggers the
perf summary dump; the ttnn op is captured in the same dump.
"""

import torch
import ttnn

from summa_kernel import make_kernel as make_summa_kernel


M, K, N = 4096, 4096, 4096
BLOCK_CFG = (8, 4, 8)
PART_CFG = (8, 11, 1)
M_PAD = 4096  # Mp*m_span*bm*TILE = 8*2*8*32 = 4096
N_PAD = 4224  # Np*n_span*bn*TILE = 11*3*4*32 = 4224

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


def pad_2d(t, rows, cols):
    r, c = t.shape
    if r == rows and c == cols:
        return t
    return torch.nn.functional.pad(t, (0, cols - c, 0, rows - r), value=0.0)


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

        a_pad = to_dev(pad_2d(a_t, M_PAD, K), device)
        w_pad = to_dev(pad_2d(w_t, K, N_PAD), device)
        out_pad = to_dev(torch.zeros(M_PAD, N_PAD, dtype=torch.bfloat16), device)

        a_ref = to_dev(a_t, device)
        w_ref = to_dev(w_t, device)

        print("Running ttnn.matmul", flush=True)
        out_ref = ttnn.matmul(a_ref, w_ref, compute_kernel_config=TTNN_CFG)
        ttnn.synchronize_device(device)

        print("Running summa_matmul", flush=True)
        fn = make_summa_kernel(M_PAD, K, N_PAD, BLOCK_CFG, PART_CFG)
        fn(a_pad, w_pad, out_pad)
        ttnn.synchronize_device(device)

        for t in (a_pad, w_pad, out_pad, a_ref, w_ref, out_ref):
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
