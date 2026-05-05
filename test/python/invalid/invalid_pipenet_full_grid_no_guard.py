# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Issue #541 regression caught by `ttl-verify-pipenet-guards`.

Mirrors the failing kernel from issue #541: a small 2x2 multicast
matmul whose work extent (M_BLOCKS=2, N_BLOCKS=2) is smaller than the
launch grid. Without an `if mcast_a_net.is_active():` guard, every
launched node executes the pipe-coupled body, indexing tensors
out-of-bounds and breaking the multicast handshake.

The verifier rejects the program at the unguarded `cb_wait` on the
pipe-fed `a_cb`. The corresponding *positive* regression in
`test/python/pipe/test_pipenet_active_guard.py` carries the
`is_active()` guard the verifier expects.

Hard-coded `grid=(8, 7)` to mirror the Wormhole compute grid; using
`grid="full"` would require an active device to resolve under
`TTLANG_COMPILE_ONLY=1`.
"""

# The Python frontend's diagnostic formatter renders the primary
# error and each attached note with its own source-context block.
# CHECK: error: this region exchanges data on PipeNet
# CHECK: note: example node where the guard does not hold:
# CHECK: note: PipeNet {{[A-Za-z0-9_]+}} declared here

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch  # noqa: E402
import ttnn  # noqa: E402

import ttl  # noqa: E402

TILE = 32
BLOCK_M = 4
BLOCK_N = 4
BLOCK_K = 4
BLOCK_SIZE = BLOCK_M * TILE

M_BLOCKS = 2
N_BLOCKS = 2
K_BLOCKS = 1

M = M_BLOCKS * BLOCK_SIZE
N = N_BLOCKS * BLOCK_SIZE
K = K_BLOCKS * BLOCK_SIZE


def _host_ttnn(shape):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


@ttl.operation(grid=(8, 7))
def small_mcast_matmul_no_guard(a, w, out):
    a_pipes = [
        ttl.Pipe(src=(0, row), dst=(slice(0, N_BLOCKS), row)) for row in range(M_BLOCKS)
    ]
    mcast_a_net = ttl.PipeNet(a_pipes)
    b_pipes = [
        ttl.Pipe(src=(col, 0), dst=(col, slice(0, M_BLOCKS))) for col in range(N_BLOCKS)
    ]
    mcast_b_net = ttl.PipeNet(b_pipes)

    a_cb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_M, BLOCK_K), block_count=2)
    b_cb = ttl.make_dataflow_buffer_like(w, shape=(BLOCK_K, BLOCK_N), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(BLOCK_M, BLOCK_N), block_count=2)

    @ttl.compute()
    def compute():
        # MISSING `if mcast_a_net.is_active():` guard.
        # Compute runs on every launched node and waits on the pipe-fed
        # `a_cb` and `b_cb`, but only the (M_BLOCKS x N_BLOCKS) work
        # extent ever sees a producer push.
        with out_cb.reserve() as out_blk:
            a_blk = a_cb.wait()
            b_blk = b_cb.wait()
            out_blk.store(a_blk @ b_blk)
            a_blk.pop()
            b_blk.pop()

    @ttl.datamovement()
    def dm_read():
        node_n, node_m = ttl.node(dims=2)
        mb = node_m
        mr = mb * BLOCK_M
        nb = node_n
        nc = nb * BLOCK_N
        for kb in range(K_BLOCKS):
            kc = kb * BLOCK_K
            with a_cb.reserve() as a_blk:

                def read_a(pipe):
                    ttl.copy(a[mr : mr + BLOCK_M, kc : kc + BLOCK_K], a_blk).wait()
                    ttl.copy(a_blk, pipe).wait()

                mcast_a_net.if_src(read_a)

                def recv_a(pipe):
                    ttl.copy(pipe, a_blk).wait()

                mcast_a_net.if_dst(recv_a)

            with b_cb.reserve() as b_blk:

                def read_b(pipe):
                    ttl.copy(w[kc : kc + BLOCK_K, nc : nc + BLOCK_N], b_blk).wait()
                    ttl.copy(b_blk, pipe).wait()

                mcast_b_net.if_src(read_b)

                def recv_b(pipe):
                    ttl.copy(pipe, b_blk).wait()

                mcast_b_net.if_dst(recv_b)

    @ttl.datamovement()
    def dm_write():
        node_n, node_m = ttl.node(dims=2)
        mb = node_m
        mr = mb * BLOCK_M
        nb = node_n
        nc = nb * BLOCK_N
        with out_cb.wait() as out_blk:
            ttl.copy(out_blk, out[mr : mr + BLOCK_M, nc : nc + BLOCK_N]).wait()


def main():
    a = _host_ttnn((M, K))
    w = _host_ttnn((K, N))
    out = _host_ttnn((M, N))
    small_mcast_matmul_no_guard(a, w, out)


if __name__ == "__main__":
    main()
