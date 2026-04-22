# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the ksplit kernel.

Hardcoded small shape so the kernel path exercised is minimal:
  128 x 256 x 128, block=(1,1,4), parts=(4,4,2)
  -> 32 cores, K_parts=2 (one gather sender per root).

Run via run-test.sh after copy-file.sh has placed ksplit_kernel.py in /tmp.
"""

import time

import torch
import ttnn

from ksplit_kernel import make_kernel


def to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    M, K, N = 128, 256, 128
    block_cfg = (1, 1, 4)
    part_cfg = (4, 4, 2)

    print(f"shape: M={M} K={K} N={N}  block={block_cfg}  parts={part_cfg}", flush=True)

    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        a_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
        w_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
        ref = a_t.float() @ w_t.float()

        a = to_dev(a_t, device)
        w = to_dev(w_t, device)
        out = to_dev(torch.zeros(M, N, dtype=torch.bfloat16), device)

        fn = make_kernel(M, K, N, block_cfg, part_cfg)
        print("compiled; dispatching...", flush=True)
        t0 = time.time()
        fn(a, w, out)
        print(f"dispatch returned in {time.time() - t0:.2f}s; syncing...", flush=True)
        ttnn.synchronize_device(device)
        print(f"synced in {time.time() - t0:.2f}s total", flush=True)

        result = ttnn.to_torch(out).float()
        pcc = torch.corrcoef(torch.stack([result.flatten(), ref.flatten()]))[0, 1].item()
        max_err = (result - ref).abs().max().item()
        print(f"PCC={pcc:.6f}  max_err={max_err:.4e}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
