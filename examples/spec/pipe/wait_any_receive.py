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


@ttl.operation(grid=(3, 1))
def select_completed_receive(input_tensor, output_tensor):
    first_pipe = ttl.Pipe(src=(0, 0), dst=(2, 0))
    second_pipe = ttl.Pipe(src=(1, 0), dst=(2, 0))
    first_net = ttl.PipeNet([first_pipe])
    second_net = ttl.PipeNet([second_pipe])

    first_input_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=1
    )
    second_input_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=1
    )
    first_landing_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=1
    )
    second_landing_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=1
    )

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def receive_and_select():
        if first_net.is_src():
            with first_input_dfb.wait() as first_input:
                ttl.copy(first_input, first_pipe).wait()

        if second_net.is_src():
            with second_input_dfb.wait() as second_input:
                ttl.copy(second_input, second_pipe).wait()

        if first_net.is_dst():
            if second_net.is_dst():
                next_index = 1

                # spec:begin
                # Both receives are posted before either one is awaited. Waiting
                # on first_request directly would block this thread even if
                # second_request had already completed.
                first_block = first_landing_dfb.reserve()
                second_block = second_landing_dfb.reserve()
                first_request = ttl.copy(first_pipe, first_block)
                second_request = ttl.copy(second_pipe, second_block)

                completed = ttl.wait_any(
                    (first_request, second_request), start=next_index
                )
                selected = completed.index()

                if selected == 0:
                    first_block.push()
                    with first_landing_dfb.wait() as first_result:
                        ttl.copy(first_result, output_tensor[0, 0]).wait()

                    # The nonselected request remains pending until it is awaited.
                    second_request.wait()
                    second_block.push()

                if selected == 1:
                    second_block.push()
                    with second_landing_dfb.wait() as second_result:
                        ttl.copy(second_result, output_tensor[0, 0]).wait()

                    first_request.wait()
                    first_block.push()
                # spec:end

    @ttl.datamovement()
    def load_input():
        if first_net.is_src():
            with first_input_dfb.reserve() as first_input:
                ttl.copy(input_tensor[0, 0], first_input).wait()

        if second_net.is_src():
            with second_input_dfb.reserve() as second_input:
                ttl.copy(input_tensor[0, 1], second_input).wait()


torch.manual_seed(42)
device = ttnn.open_device(device_id=0)

try:
    input_torch = torch.rand((32, 64), dtype=torch.float32)
    input_tensor = ttnn.from_torch(
        input_torch,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.zeros(
        ttnn.Shape([32, 32]),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    select_completed_receive(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor)
    first_input = input_torch[:, :32]
    second_input = input_torch[:, 32:]
    assert torch.equal(actual, first_input) or torch.equal(actual, second_input)
finally:
    ttnn.close_device(device)
