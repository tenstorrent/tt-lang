#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Element-wise with broadcast using 1-D tensors.

Direct simulator adaptation of the "Element-wise with broadcast" example from
TTLangSpecification.md:

    Y = sqrt(A^2 + B^2),  Z = sqrt(A^2 - B^2)

Tensor shapes (all 1-D):
    A  : N         -> NT tiles     (NT = N // TILE_SIZE)
    B  : TILE_SIZE -> 1 tile       (scalar broadcast source)
    Y  : N         -> NT tiles
    Z  : N         -> NT tiles
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

from sim import ttl, ttnn


@ttl.kernel(grid=(1, 1))
def eltwise_1d_broadcast(
    A: ttnn.Tensor,
    B: ttnn.Tensor,
    Y: ttnn.Tensor,
    Z: ttnn.Tensor,
) -> None:
    # ---------------------
    # Element-wise with broadcast with two outputs: Y = sqrt(A^2 + B^2), Z = sqrt(A^2 - B^2)
    #
    # Tensor   Torch shape  Shape in tiles
    # A        N            NT
    # B        1            1  (scalar - 1 element)
    # Y        N            NT
    # Z        N            NT
    #
    TILE_SIZE = ttl.TILE_SHAPE[0]
    N = A.shape[0]
    NT = N // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1,))
    # B is a scalar (1 element) that will be broadcast to match A's shape
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1,))
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1,))
    z_dfb = ttl.make_dataflow_buffer_like(Z, shape=(1,))

    @ttl.datamovement()
    def elwise_read():
        for nt in range(NT):

            # acquire a_blk and b_blk from a_dfb and b_dfb:

            with (
                a_dfb.reserve() as a_blk,
                b_dfb.reserve() as b_blk,
            ):
                # then copy:

                a_xf = ttl.copy(A[nt], a_blk)
                b_xf = ttl.copy(B[0], b_blk)

                a_xf.wait()
                b_xf.wait()

                # release a_blk and b_blk

    @ttl.compute()
    def elwise_compute():
        for _ in range(NT):

            # acquire a_blk, b_blk, y_blk and z_blk from a_dfb, b_dfb, y_dfb and z_dfb:

            with (
                a_dfb.wait() as a_blk,
                b_dfb.wait() as b_blk,
                y_dfb.reserve() as y_blk,
                z_dfb.reserve() as z_blk,
            ):
                # then compute y = sqrt(a^2 + b^2) and z = sqrt(a^2 - b^2):

                a_squared = a_blk**2
                b_squared = b_blk**2

                # Broadcast b_squared from element_shape=(1,) to match a_squared's element_shape=(32,)
                y = ttl.math.sqrt(
                    a_squared + ttl.math.broadcast(b_squared, y_blk, dims=[0])
                )
                z = ttl.math.sqrt(
                    a_squared - ttl.math.broadcast(b_squared, z_blk, dims=[0])
                )

                y_blk.store(y)
                z_blk.store(z)

                # release a_blk, b_blk and y_blk

    @ttl.datamovement()
    def elwise_write():
        for nt in range(NT):

            # acquire y_blk and z_blk from y_dfb and z_dfb:

            with (
                y_dfb.wait() as y_blk,
                z_dfb.wait() as z_blk,
            ):

                # then copy:

                y_xf = ttl.copy(y_blk, Y[nt])
                z_xf = ttl.copy(z_blk, Z[nt])
                y_xf.wait()
                z_xf.wait()

                # release y_blk and z_blk


def main() -> int:
    N = 128  # 4 tiles

    # A = 4.0, B = 3.0 (scalar)
    # Y = sqrt(4^2 + 3^2) = sqrt(25) = 5.0
    # Z = sqrt(4^2 - 3^2) = sqrt(7) ~= 2.6458
    A = ttnn.from_torch(torch.full((N,), 4.0, dtype=torch.float32))
    B = ttnn.from_torch(torch.tensor([3.0], dtype=torch.float32))  # Scalar - 1 element
    Y = ttnn.empty((N,), dtype=torch.float32)
    Z = ttnn.empty((N,), dtype=torch.float32)

    eltwise_1d_broadcast(A, B, Y, Z)

    Y_torch = Y.to_torch()
    Z_torch = Z.to_torch()

    expected_Y = torch.full((N,), math.sqrt(4.0**2 + 3.0**2))
    expected_Z = torch.full((N,), math.sqrt(4.0**2 - 3.0**2))

    ok_Y = torch.allclose(Y_torch, expected_Y, atol=1e-5)
    ok_Z = torch.allclose(Z_torch, expected_Z, atol=1e-5)

    if ok_Y and ok_Z:
        print("PASSED!")
        return 0
    else:
        print(f"FAILED: Y ok={ok_Y}, Z ok={ok_Z}")
        print(f"  Y[0]={Y_torch[0].item():.6f}  expected={expected_Y[0].item():.6f}")
        print(f"  Z[0]={Z_torch[0].item():.6f}  expected={expected_Z[0].item():.6f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
