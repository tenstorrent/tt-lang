# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Test Pipe DSL integration - verifies that pipe.if_src/if_dst lower to ttl ops.

This test uses TTLANG_COMPILE_ONLY to verify MLIR generation without running.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(2, 1))
def pipe_test(x, y, out):
    pipe = ttl.Pipe(src=(0, 0), dst=(1, 0))

    x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_cb.wait() as l:
            with out_cb.reserve() as o:
                o.store(l)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            with pipe.if_src():
                pass
            with pipe.if_dst():
                pass

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            pass


# =============================================================================
# Verify TTL dialect ops for Pipe
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# Check that create_pipe is emitted
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0)

# Check that if_src is emitted
# CHECK: ttl.if_src %{{.+}} : <src(0, 0) dst(1, 0) to(1, 0)>

# Check that if_dst is emitted
# CHECK: ttl.if_dst %{{.+}} : <src(0, 0) dst(1, 0) to(1, 0)>


def test_pipe():
    """Test that Pipe DSL compiles correctly."""
    x = ttnn.ones((32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    y = ttnn.ones((32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = ttnn.zeros((32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    pipe_test(x, y, out)
    print("Pipe DSL test completed")


if __name__ == "__main__":
    test_pipe()
