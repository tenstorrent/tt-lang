# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s

"""@ttl.atom-in-@ttl.atom inlining. A small DFB-parameter helper atom is
inlined into a larger atom; the helper takes its DFBs as ttl.DFB
parameters (it has no way to be supplied buffers except by inlining).
Also checks that a callee declaring its own DFB is rejected at the outer
atom's decoration time."""

import torch

import ttnn
import ttl


@ttl.atom()
def _exp_block(inp: ttl.DFB, out: ttl.DFB):
    """Per-tile exp; declares no DFBs of its own (takes them as params)."""
    x = inp.wait()
    r = out.reserve()
    r.store(ttl.exp(x))


@ttl.atom(grid=(1, 1))
def atom_outer_exp(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    a_blk = a_cb.reserve()
    ttl.copy(in_t[0:1, 0:1], a_blk)

    _exp_block(a_cb, out_cb)  # inlined at decoration time

    out_done = out_cb.wait()
    ttl.copy(out_done, out_t[0:1, 0:1])


def _to_l1(device, t):
    dram = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.to_memory_config(dram, memory_config=ttnn.L1_MEMORY_CONFIG)


def check_callee_with_dfb_decl_rejected():
    """A callee that declares its own DFB cannot be inlined."""

    @ttl.atom()
    def _declares_dfb(inp: ttl.DFB, out: ttl.DFB):
        scratch = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        x = inp.wait()
        r = out.reserve()
        r.store(x)

    try:

        @ttl.atom(grid=(1, 1))
        def _outer_bad(in_t, out_t):
            a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
            out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)
            _declares_dfb(a_cb, out_cb)

    except ValueError as e:
        assert "make_dataflow_buffer_like" in str(e), str(e)
        print("check_callee_with_dfb_decl_rejected: OK")
        return
    raise AssertionError("expected ValueError for callee declaring its own DFB")


def main():
    from ttlang_test_utils import require_hardware

    # Decoration-time check: no device needed.
    check_callee_with_dfb_decl_rejected()

    require_hardware()
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(2026)
        tile = ttnn.TILE_SIZE
        inp_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)
        expected = torch.exp(inp_t.float()).to(torch.bfloat16)

        in_t = _to_l1(device, inp_t)
        out_t = _to_l1(device, torch.zeros(tile, tile, dtype=torch.bfloat16))

        atom_outer_exp(in_t, out_t)

        got = ttnn.to_torch(out_t).reshape(tile, tile).to(torch.bfloat16)
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
        print("atom_outer_exp (inlined _exp_block): OK")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
