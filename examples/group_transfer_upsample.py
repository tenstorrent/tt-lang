# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TTLANG_HARDWARE_CI: skip-compiler

"""
Nearest-Neighbor Upsample

Demonstrates ttl.GroupTransfer: each input pixel (N x HI x WI x C,
row-major layout) is replicated scale_factor[0] x scale_factor[1] times
into the output (N x HO x WO x C).

Instead of calling xf.wait() after each individual ttl.copy in the write
loop, all handles are collected in a ttl.GroupTransfer and waited on at
once with wait_all().

This example is adapted from the TTLang Specification (GroupTransfer section).
"""

import torch
import ttl
import ttnn

N = 1
HI = 2
WI = 2
C = 32
SCALE = (2, 2)
HO = HI * SCALE[0]
WO = WI * SCALE[1]


@ttl.operation(grid=(1, 1))
def nearest_neighbor_upsample(
    input_images: ttnn.Tensor,
    output: ttnn.Tensor,
    scale_factor: tuple,
) -> None:
    io_dfb = ttl.make_dataflow_buffer_like(input_images, shape=(C,), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def reader():
        for n in range(N):
            for hi in range(HI):
                for wi in range(WI):
                    with io_dfb.reserve() as io_blk:

                        # Copy input pixel channels

                        xf = ttl.copy(input_images[n, hi, wi, :], io_blk)

                        xf.wait()

    @ttl.datamovement()
    def writer():
        for n in range(N):
            for hi in range(HI):
                for wi in range(WI):
                    with io_dfb.wait() as io_blk:
                        gxf = ttl.GroupTransfer()

                        for h_sf in range(scale_factor[0]):
                            for w_sf in range(scale_factor[1]):

                                # Copy output pixel channels

                                xf = ttl.copy(io_blk, output[n, hi * scale_factor[0] + h_sf, wi * scale_factor[1] + w_sf, :])  # fmt: skip

                                # Add transfer handle to a group

                                gxf.add(xf)

                        # Wait for all transfers to complete

                        gxf.wait_all()


def main() -> None:
    torch.manual_seed(42)

    input_torch = torch.rand(N, HI, WI, C, dtype=torch.float32)
    output_torch = torch.zeros(N, HO, WO, C, dtype=torch.float32)

    input_images = ttnn.from_torch(input_torch, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_t = ttnn.from_torch(output_torch, layout=ttnn.ROW_MAJOR_LAYOUT)

    nearest_neighbor_upsample(input_images, output_t, SCALE)

    result = ttnn.to_torch(output_t)

    # Each input pixel is replicated SCALE[0] x SCALE[1] times.
    expected = input_torch.repeat_interleave(SCALE[0], dim=1).repeat_interleave(
        SCALE[1], dim=2
    )

    assert torch.allclose(result, expected), "Upsample mismatch!"
    print("PASSED!")


if __name__ == "__main__":
    main()
