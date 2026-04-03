#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Simulator-backed (`from sim import ttl`). Run via pytest test/sim, not the
# hardware compiler example step.
#
# TTLANG_HARDWARE_CI: xfail-compiler

import torch
import ttl
import ttnn

BLOCK_SIZE = 128


# ---------------------
# Element-wise with broadcast with two outputs: y = sqrt(a^2 + b^2), z = sqrt(a^2 - b^2)
#
# Tensor   Torch shape   Note
# a        N
# b        1             Scalar -- broadcast to match a, y, z along dim 0
# y        N
# z        N
#
# All tensors have row-major layout


@ttl.operation(grid="auto")
def eltwise_sqrt_broadcast(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    y: ttnn.Tensor,
    z: ttnn.Tensor,
) -> None:
    N = a.shape[0]

    # Shape in blocks (N is evenly divisible by BLOCK_SIZE)
    N_BLOCKS = N // BLOCK_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_SIZE,))
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1,))
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(BLOCK_SIZE,))
    z_dfb = ttl.make_dataflow_buffer_like(z, shape=(BLOCK_SIZE,))

    @ttl.datamovement()
    def elwise_read():

        # Reserve b_blk block

        with b_dfb.reserve() as b_blk:

            # Load entire b

            b_xf = ttl.copy(b[0], b_blk)
            b_xf.wait()

            # Push b_blk to make it ready for elwise_compute

        for n_block in range(N_BLOCKS):

            # Reserve a_blk

            with a_dfb.reserve() as a_blk:

                # Load BLOCK_SIZE block of a

                a_xf = ttl.copy(
                    a[n_block * BLOCK_SIZE : (n_block + 1) * BLOCK_SIZE], a_blk
                )
                a_xf.wait()

                # Push a_blk to make it ready for elwise_compute

    @ttl.compute()
    def elwise_compute():

        # Wait for b_blk to be loaded and pushed by elwise_read

        with b_dfb.wait() as b_blk:

            for _ in range(N_BLOCKS):

                # Wait for a_blk to be loaded and pushed by elwise_read
                # Reserve y_blk and z_blk

                with (
                    a_dfb.wait() as a_blk,
                    y_dfb.reserve() as y_blk,
                    z_dfb.reserve() as z_blk,
                ):
                    a_squared = a_blk**2
                    b_squared = b_blk**2

                    # b_squared has shape (1,); broadcast expands it to (BLOCK_SIZE,) along dim 0.
                    y_val = ttl.math.sqrt(
                        a_squared + ttl.math.broadcast(b_squared, y_blk, dims=[0])
                    )
                    z_val = ttl.math.sqrt(
                        a_squared - ttl.math.broadcast(b_squared, z_blk, dims=[0])
                    )

                    y_blk.store(y_val)
                    z_blk.store(z_val)

                    # Pop a_blk to make it available for elwise_read to load and push next block
                    # Push y_blk and z_blk to make them ready for elwise_write

        # Pop b_blk

    @ttl.datamovement()
    def elwise_write():
        for n_block in range(N_BLOCKS):
            n_slice = slice(n_block * BLOCK_SIZE, (n_block + 1) * BLOCK_SIZE)

            # Wait for elwise_compute to store and push y_blk and z_blk

            with (
                y_dfb.wait() as y_blk,
                z_dfb.wait() as z_blk,
            ):
                # Store BLOCK_SIZE of y and z

                y_xf = ttl.copy(y_blk, y[n_slice])
                z_xf = ttl.copy(z_blk, z[n_slice])
                y_xf.wait()
                z_xf.wait()

                # Pop y_blk and z_blk to make them available for elwise_compute to store and push next block


def main() -> None:
    N = 1024

    # Ensure a > b element-wise so that z = sqrt(a^2 - b^2) is well-defined.
    b_torch = torch.rand(1, dtype=torch.float32)
    a_torch = b_torch.item() + 1.0 + torch.rand(N, dtype=torch.float32)
    y_torch = torch.zeros(N, dtype=torch.float32)
    z_torch = torch.zeros(N, dtype=torch.float32)

    a = ttnn.from_torch(a_torch, layout=ttnn.ROW_MAJOR_LAYOUT)
    b = ttnn.from_torch(b_torch, layout=ttnn.ROW_MAJOR_LAYOUT)
    y = ttnn.from_torch(y_torch, layout=ttnn.ROW_MAJOR_LAYOUT)
    z = ttnn.from_torch(z_torch, layout=ttnn.ROW_MAJOR_LAYOUT)

    print("Running TT-Lang operation...")
    eltwise_sqrt_broadcast(a, b, y, z)

    y_result = ttnn.to_torch(y)
    z_result = ttnn.to_torch(z)

    b_val = float(b_torch[0].item())
    y_expected = torch.sqrt(a_torch**2 + b_val**2)
    z_expected = torch.sqrt(a_torch**2 - b_val**2)

    assert torch.allclose(y_result, y_expected, atol=1e-4), "y mismatch!"
    assert torch.allclose(z_result, z_expected, atol=1e-4), "z mismatch!"
    print("PASSED!")


if __name__ == "__main__":
    main()
