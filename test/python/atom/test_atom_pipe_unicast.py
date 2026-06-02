# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s

"""@ttl.atom with a single unicast PipeNet across two cores. Core (1,0)
reads a tile and sends it over the pipe; core (0,0) reads its own tile,
receives the sent tile, adds them, and writes the result.

Each pipe buffer is touched on exactly one RISC -- send_cb only in the
if_src callback (BRISC), recv_cb only in if_dst (NCRISC) -- because the
splitter routes the two callbacks onto different RISCs and rejects a
single reserve driven from both."""

import torch

import ttnn
import ttl


@ttl.atom(grid=(2, 1))
def atom_pipe_unicast(a, out):
    own_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    send_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    fwd_net = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])
    node_x, _ = ttl.node(dims=2)

    # Reserves hoisted out of the callbacks; each lands on a single RISC.
    s_blk = send_cb.reserve()
    r_dst = recv_cb.reserve()

    def send(pipe):
        ttl.copy(a[1:2, 0:1], s_blk)
        ttl.copy(s_blk, pipe)

    fwd_net.if_src(send)

    def recv(pipe):
        ttl.copy(pipe, r_dst)

    fwd_net.if_dst(recv)

    if node_x == 0:
        own_blk = own_cb.reserve()
        ttl.copy(a[0:1, 0:1], own_blk)
        s = out_cb.reserve()
        s.store(own_cb.wait() + recv_cb.wait())
        ttl.copy(out_cb.wait(), out[0:1, 0:1])


def _to_dram(device, t):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    from ttlang_test_utils import require_hardware

    require_hardware()
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(2026)
        tile = ttnn.TILE_SIZE
        # Two row-tiles: tile 0 is core (0,0)'s own, tile 1 is sent from (1,0).
        a_t = torch.randn(2 * tile, tile, dtype=torch.bfloat16) * 0.5
        expected = (a_t[:tile].float() + a_t[tile:].float()).to(torch.bfloat16)

        a = _to_dram(device, a_t)
        out = _to_dram(device, torch.zeros(tile, tile, dtype=torch.bfloat16))

        atom_pipe_unicast(a, out)

        got = ttnn.to_torch(out).reshape(tile, tile).to(torch.bfloat16)
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
        print("atom_pipe_unicast: OK")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
