# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s

"""@ttl.atom tensor copy: the minimal unified body -- read one tile from
a ttnn tensor into a DFB and write it back out. The compute thread is
empty, so the splitter must emit only the two data-movement threads."""

import torch

import ttnn
import ttl


@ttl.atom(grid=(1, 1))
def atom_tensor_copy(src, dst):
    cb = ttl.make_dataflow_buffer_like(src, shape=(1, 1), block_count=2)
    blk_in = cb.reserve()
    ttl.copy(src[0:1, 0:1], blk_in)
    blk_out = cb.wait()
    ttl.copy(blk_out, dst[0:1, 0:1])


def _to_l1(device, t):
    dram = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.to_memory_config(dram, memory_config=ttnn.L1_MEMORY_CONFIG)


def main():
    from ttlang_test_utils import require_hardware

    require_hardware()
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(2026)
        tile = ttnn.TILE_SIZE
        src_t = torch.randn(tile, tile, dtype=torch.bfloat16)

        src = _to_l1(device, src_t)
        dst = _to_l1(device, torch.zeros(tile, tile, dtype=torch.bfloat16))

        atom_tensor_copy(src, dst)

        got = ttnn.to_torch(dst).reshape(tile, tile).to(torch.bfloat16)
        torch.testing.assert_close(got, src_t, rtol=1e-3, atol=1e-3)
        print("atom_tensor_copy: OK")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
