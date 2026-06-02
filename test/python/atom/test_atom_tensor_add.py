# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s

"""@ttl.atom tensor add: a single unified body (no explicit thread
functions) that reads two ttnn tensors through DFBs, adds them, and
writes the result. Exercises the thread splitter end to end."""

import torch

import ttnn
import ttl


@ttl.atom(grid=(1, 1))
def atom_tensor_add(a, b, out):
    a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_cb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    a_blk = a_cb.reserve()
    ttl.copy(a[0:1, 0:1], a_blk)
    b_blk = b_cb.reserve()
    ttl.copy(b[0:1, 0:1], b_blk)

    s = out_cb.reserve()
    a_in = a_cb.wait()
    b_in = b_cb.wait()
    s.store(a_in + b_in)

    out_done = out_cb.wait()
    ttl.copy(out_done, out[0:1, 0:1])


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
        a_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5)
        b_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5)
        expected = (a_t.float() + b_t.float()).to(torch.bfloat16)

        a = _to_l1(device, a_t)
        b = _to_l1(device, b_t)
        out = _to_l1(device, torch.zeros(tile, tile, dtype=torch.bfloat16))

        atom_tensor_add(a, b, out)

        got = ttnn.to_torch(out).reshape(tile, tile).to(torch.bfloat16)
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
        print("atom_tensor_add: OK")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
