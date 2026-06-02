# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s

"""@ttl.atom with a tensor-less DFB built via ttl.make_dfb. The output
buffer is declared from a dtype name string rather than a borrowed
tensor; compute writes exp(x) into it and data movement drains it."""

import torch

import ttnn
import ttl


@ttl.atom(grid=(1, 1))
def atom_make_dfb_exp(inp, out):
    in_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)

    a_blk = in_cb.reserve()
    ttl.copy(inp[0:1, 0:1], a_blk)

    o = out_cb.reserve()
    x = in_cb.wait()
    o.store(ttl.exp(x))

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
        inp_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)
        expected = torch.exp(inp_t.float()).to(torch.bfloat16)

        inp = _to_l1(device, inp_t)
        out = _to_l1(device, torch.zeros(tile, tile, dtype=torch.bfloat16))

        atom_make_dfb_exp(inp, out)

        got = ttnn.to_torch(out).reshape(tile, tile).to(torch.bfloat16)
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
        print("atom_make_dfb_exp: OK")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
