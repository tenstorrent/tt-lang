# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Fixture for test_unified_operation.py: a thread-unified @ttl.operation carrying a
# user decorator above @ttl.operation.
#
# Python applies that decorator at the definition site, to whatever @ttl.operation
# returns, so the rewritten body must not carry it as well -- one wrapper, one call.
# The operation itself has to keep working, on a grid larger than one node, where a
# decorator re-applied per node would show up as extra calls.

import functools

import ttl
import ttnn

wrapped_calls = 0

device = ttnn.open_device(device_id=0)


def count_calls(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global wrapped_calls
        wrapped_calls += 1
        return fn(*args, **kwargs)

    return wrapper


try:

    @count_calls
    @ttl.operation(grid=(1, 2))
    def copy_through(a: ttnn.Tensor, out: ttnn.Tensor) -> None:
        dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        blk = dfb.reserve()
        ttl.copy(a[0:1, 0:1], blk).wait()
        blk.push()
        out_blk = dfb.wait()
        ttl.copy(out_blk, out[0:1, 0:1]).wait()
        out_blk.pop()

    x = ttnn.rand(ttnn.Shape([32, 64]), layout=ttnn.TILE_LAYOUT, device=device)
    z = ttnn.zeros(ttnn.Shape([32, 64]), layout=ttnn.TILE_LAYOUT, device=device)
    copy_through(x, z)

    source = x.to_torch()[0:32, 0:32]
    destination = z.to_torch()[0:32, 0:32]
    print(f"wrapped calls: {wrapped_calls}")
    print(f"copied region matches: {bool((source == destination).all())}")

finally:
    ttnn.close_device(device)
