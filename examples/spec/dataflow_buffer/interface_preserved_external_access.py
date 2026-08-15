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

from pathlib import Path

import torch

import ttl
import ttnn


INTERFACE_ACCESS_HEADER = str(
    Path(__file__).with_name("interface_preserved_external_access.hpp")
)


@ttl.operation(grid=(1, 1))
def interface_preserved_external_access(input_tensor, output_tensor):
    format_descriptor = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=1
    )
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=2
    )

    @ttl.datamovement()
    def read():
        with input_dfb.reserve() as destination:
            ttl.copy(input_tensor[0, 0], destination).wait()

    @ttl.compute()
    def compute():
        # spec:begin
        ttl.call_extern_func(
            INTERFACE_ACCESS_HEADER,
            "inspect_dfb_interface",
            template_args=[ttl.dfb_descriptor(format_descriptor)],
            dfb_accesses=[
                ttl.DFBAccess.interface_preserved(format_descriptor),
            ],
        )
        # spec:end
        with input_dfb.wait() as source:
            with output_dfb.reserve() as destination:
                destination.store(source)

    @ttl.datamovement()
    def write():
        with output_dfb.wait() as source:
            ttl.copy(source, output_tensor[0, 0]).wait()


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

    interface_preserved_external_access(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor)
    assert torch.equal(actual, input_torch)
finally:
    ttnn.close_device(device)
