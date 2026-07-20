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
# ---------------------
# Nearest Neighbor Upsample
#
# Tensor              Torch shape
# input_images        N, HI, WI, C
# output_images       N, HO, WO, C
#
# All tensors have row-major layout

HO = HI * H_SCALE_FACTOR
WO = WI * W_SCALE_FACTOR

io_dfb = ttl.make_dataflow_buffer_like(input_images, shape=(C,), block_count=2)


@ttl.datamovement()
def reader():
    for n in range(N):
        for hi in range(HI):
            for wi in range(WI):
                with io_dfb.reserve() as io_blk:

                    # Load input pixel channels

                    xf = ttl.copy(input_t[n, hi, wi, :], io_blk)

                    xf.wait()


@ttl.datamovement()
def writer():
    for n in range(N):
        for hi in range(HI):
            for wi in range(WI):
                with io_dfb.wait() as io_blk:
                    gxf = ttl.GroupTransfer()

                    for h_scale_index in range(H_SCALE_FACTOR):
                        for w_scale_index in range(W_SCALE_FACTOR):

                            # Copy output pixel channels

                            xf = ttl.copy(
                                io_blk,
                                output[
                                    n,
                                    hi * H_SCALE_FACTOR + h_scale_index,
                                    wi * W_SCALE_FACTOR + w_scale_index,
                                    :,
                                ],
                            )

                            # Add transfer handle to a group

                            gxf.add(xf)

                    # Wait for all transfers to complete

                    gxf.wait_all()


# spec:end
