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


@ttl.operation(grid=(1, 1))
def select_completed_receive(input_tensor, output_tensor):
    slow_pipe = ttl.Pipe(src=(0, 0), dst=(0, 0))
    fast_pipe = ttl.Pipe(src=(0, 0), dst=(0, 0))
    slow_net = ttl.PipeNet([slow_pipe])
    fast_net = ttl.PipeNet([fast_pipe])

    slow_input_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=1
    )
    fast_input_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=1
    )
    slow_landing_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=1
    )
    fast_landing_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=1
    )

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def receive_and_select():
        if slow_net.is_active():
            pass
        if fast_net.is_active():
            pass
        next_index = 1

        slow_input = slow_input_dfb.wait()
        fast_input = fast_input_dfb.wait()
        slow_send = ttl.copy(slow_input, slow_pipe)
        fast_send = ttl.copy(fast_input, fast_pipe)

        # spec:begin
        # Both receives are posted before either one is awaited. Waiting on
        # slow_request directly would block this thread even if fast_request had
        # already completed.
        slow_block = slow_landing_dfb.reserve()
        fast_block = fast_landing_dfb.reserve()
        slow_request = ttl.copy(slow_pipe, slow_block)
        fast_request = ttl.copy(fast_pipe, fast_block)
        # spec:end

        slow_send.wait()
        slow_input.pop()
        fast_send.wait()
        fast_input.pop()

        # spec:begin
        completed = ttl.wait_any((slow_request, fast_request), start=next_index)
        selected = completed.index()

        if selected == 0:
            slow_block.push()
            with slow_landing_dfb.wait() as slow_result:
                ttl.copy(slow_result, output_tensor[0, 0]).wait()

        if selected == 1:
            fast_block.push()
            with fast_landing_dfb.wait() as fast_result:
                ttl.copy(fast_result, output_tensor[0, 0]).wait()

        # The nonselected request and its reserved block remain pending and can be
        # included in a later ttl.wait_any call or awaited directly.
        # spec:end

    @ttl.datamovement()
    def load_input():
        with slow_input_dfb.reserve() as slow_input:
            ttl.copy(input_tensor[0, 0], slow_input).wait()
        with fast_input_dfb.reserve() as fast_input:
            ttl.copy(input_tensor[0, 1], fast_input).wait()


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
    assert torch.equal(actual, input_torch[:, 32:])
finally:
    ttnn.close_device(device)
