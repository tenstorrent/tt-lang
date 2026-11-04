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

import torch

import ttl
import ttnn


reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)


@ttl.operation(grid=(1, 1))
def allocation_group_example(input_tensor, output_tensor):
    # spec:begin
    shared_allocation = ttl.make_dfb_allocation_group()

    first_source = ttl.make_dataflow_buffer_like(
        input_tensor,
        shape=(1, 2),
        block_count=1,
        allocation_group=shared_allocation,
    )
    second_source = ttl.make_dataflow_buffer_like(
        input_tensor,
        shape=(1, 1),
        block_count=4,
        allocation_group=shared_allocation,
    )
    # spec:end
    handoff = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
    output = ttl.make_dataflow_buffer_like(output_tensor, shape=(1, 1), block_count=2)

    @ttl.datamovement(kernel=reader_kernel)
    def read():
        with first_source.reserve() as first_destination:
            ttl.copy(input_tensor[0:1, 0:2], first_destination).wait()

        with handoff.wait():
            pass

        with second_source.reserve() as second_destination:
            ttl.copy(input_tensor[0, 0], second_destination).wait()

    @ttl.compute(kernel=compute_kernel)
    def compute():
        with first_source.wait():
            pass

        with handoff.reserve() as completion:
            completion.store(
                ttl.block.fill(0, shape=completion.shape, dtype=completion.dtype)
            )

        with second_source.wait() as second_block:
            with output.reserve() as output_block:
                output_block.store(second_block)

    @ttl.datamovement(kernel=writer_kernel)
    def write():
        with output.wait() as output_block:
            ttl.copy(output_block, output_tensor[0, 0]).wait()


torch.manual_seed(42)
device = ttnn.open_device(device_id=0)

try:
    input_torch = torch.rand((32, 64), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    allocation_group_example(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor)
    assert torch.equal(actual, input_torch[:, :32])
finally:
    ttnn.close_device(device)
