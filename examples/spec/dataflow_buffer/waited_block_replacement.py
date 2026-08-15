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
def waited_block_replacement(input_tensor, output_tensor):
    state_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=1)
    output_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=2
    )

    with state_dfb.reserve() as initial_state:
        ttl.copy(input_tensor[0, 0], initial_state).wait()

    # spec:begin
    with state_dfb.wait() as state:
        increment = ttl.block.fill(1, shape=state.shape)
        state.store(state + increment)

        with output_dfb.reserve() as replacement_output:
            replacement_output.store(state)
    # spec:end

    with output_dfb.wait() as written_output:
        ttl.copy(written_output, output_tensor[0, 0]).wait()


torch.manual_seed(42)
device = ttnn.open_device(device_id=0)

try:
    input_torch = torch.rand((32, 32), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.from_torch(
        torch.zeros_like(input_torch),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    waited_block_replacement(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor)
    assert torch.allclose(actual, input_torch + 1, rtol=5e-2, atol=1.0)
finally:
    ttnn.close_device(device)
