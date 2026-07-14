# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Example source for docs/sphinx/specs/TTLangSpecification.md.
#
# The lines between the "spec:begin" and "spec:end" markers below are included
# verbatim in the specification. Regenerate the specification after editing:
#
#     python docs/sphinx/specs/build_spec.py
#
# Everything outside the markers (imports, scaffolding) exists so the file can
# stand on its own and is not copied into the specification.

import math

import torch

import ttl
import ttnn


# spec:begin
@ttl.operation(grid=(1, 1))
def __add(
    a: ttnn.Tensor,  # input tensor
    b: ttnn.Tensor,  # input tensor
    out: ttnn.Tensor,  # output tensor
) -> None:
    # Dataflow buffers shared by the threads.
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    # One body: the compiler places the copies on the data movement threads
    # and the addition on the compute thread.
    a_dst_blk = a_dfb.reserve()
    b_dst_blk = b_dfb.reserve()
    a_tx = ttl.copy(a[0:1, 0:1], a_dst_blk)
    b_tx = ttl.copy(b[0:1, 0:1], b_dst_blk)
    a_tx.wait()
    b_tx.wait()
    a_dst_blk.push()
    b_dst_blk.push()

    out_blk = out_dfb.reserve()
    a_blk = a_dfb.wait()
    b_blk = b_dfb.wait()
    out_blk.store(a_blk + b_blk)
    a_blk.pop()
    b_blk.pop()

    out_tx = ttl.copy(out_dfb.wait(), out[0:1, 0:1])
    out_tx.wait()
    out_blk.push()


# Simple wrapper to allow returning output tensor in TT-NN style
def add(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    out = ttnn.zeros(a.shape, layout=ttnn.TILE_LAYOUT)
    __add(a, b, out)
    return out


x = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT)

# TT-Lang operations mix freely with built-in TT-NN operations.
y = ttnn.exp(add(ttnn.abs(x), x))
# spec:end
