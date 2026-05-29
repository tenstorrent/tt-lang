# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl


@ttl.operation(grid=(1, 1))
def _if_branch_pipe(inp, out):
    """
    Creates a loopback pipe and places send/recv wait operations in
    mutually exclusive scf.if branches to test the PipeNet verifier
    false-positive cycle bug.
    """
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(0, 0))])

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as t, out_cb.reserve() as o:
            o.store(t)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)

        # Generates an scf.if in the MLIR.
        if x == 0:

            def send(pipe):
                with inp_cb.reserve() as blk:
                    ttl.copy(inp[0, 0], blk).wait()
                    ttl.copy(blk, pipe).wait()

            net.if_src(send)

            def recv(pipe):
                with out_cb.reserve() as blk:
                    ttl.copy(pipe, blk).wait()
                    ttl.copy(blk, out[0, 0]).wait()

            net.if_dst(recv)
        else:

            def send(pipe):
                with inp_cb.reserve() as blk:
                    ttl.copy(inp[0, 0], blk).wait()
                    ttl.copy(blk, pipe).wait()

            net.if_src(send)

            def recv(pipe):
                with out_cb.reserve() as blk:
                    ttl.copy(pipe, blk).wait()
                    ttl.copy(blk, out[0, 0]).wait()

            net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        pass


def test_pipenet_scf_if_no_cycle(device):
    """
    Ensures that compiling a kernel with pipe operations inside
    sibling scf.if branches does not fail the verifier cycle check.
    """
    inp = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Just defining/compiling the operation triggers the verifier pass.
    # The fix ensures this does not throw a compile error.
    _if_branch_pipe(inp, out)
