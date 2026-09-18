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
# Everything outside the markers (imports, scaffolding, the @ttl.operation
# wrapper, device setup, and the correctness check) exists so the file can run
# standalone; it is not copied into the specification. The marked lines are
# nested inside @ttl.operation and dedented on render, so these mechanics add
# nothing to the rendered text.


import torch

import ttl
import ttnn

# Concrete compile-time sizes for a single-block run.
N = 1
HI, WI, C = 2, 2, 32
H_SCALE_FACTOR = 2
W_SCALE_FACTOR = 2


@ttl.operation(grid=(1, 1))
def group_transfer(
    input_images: ttnn.Tensor,  # input images (N, HI, WI, C)
    output: ttnn.Tensor,  # output images (N, HO, WO, C)
) -> None:
    # The rendered spec reads the input tensor via `input_t`; alias it to the
    # operation parameter so the marked lines below stay verbatim.
    input_t = input_images
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

    @ttl.compute()
    def _noop_compute() -> None:
        # The upsample is pure data movement; a no-op compute kernel satisfies
        # the simulator's 3-kernel (compute + 2 DM) operation contract.
        pass


torch.manual_seed(0)

device = ttnn.open_device(device_id=0)

try:
    HO = HI * H_SCALE_FACTOR
    WO = WI * W_SCALE_FACTOR

    input_torch = torch.rand(N, HI, WI, C, dtype=torch.float32)
    input_t = ttnn.from_torch(input_torch, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_t = ttnn.zeros(
        ttnn.Shape([N, HO, WO, C]), layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    group_transfer(input_t, output_t)

    # Nearest-neighbor upsample: each input pixel is replicated
    # H_SCALE_FACTOR x W_SCALE_FACTOR times into the output.
    #
    # What this checks is the upsample, not the group. The simulator waits for a
    # transfer whose handle is never waited on explicitly (copy-wait injection,
    # docs/sphinx/simulator.md), so the same output comes out with the gxf.add and
    # gxf.wait_all lines removed -- on hardware those transfers would still be in
    # flight. The group's own contract (transfers complete at wait_all, no add
    # after it) is pinned in test/sim/test_copy.py::TestGroupTransfer.
    expected = input_torch.repeat_interleave(H_SCALE_FACTOR, dim=1).repeat_interleave(
        W_SCALE_FACTOR, dim=2
    )
    # The upsample replicates pixels by copy alone, so the comparison is exact:
    # each output pixel is the input pixel it came from, bit for bit.
    assert torch.equal(
        expected, ttnn.to_torch(output_t)
    ), "nearest-neighbor upsample did not match torch reference"

finally:
    ttnn.close_device(device)
