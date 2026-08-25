# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Use typed DFB descriptors in external functions.

TTL operations declare each DFB transaction around the opaque C++ calls. An
acknowledgment orders the first result's pop before the second result's reserve.
The calls do not summarize their storage behavior, so their result DFBs remain
distinct despite that ordering.
"""

import os
import re
import tempfile
from pathlib import Path

import torch
import ttl
import ttnn


EXTERNAL_MULTIPLY_HEADER = os.path.join(
    os.path.dirname(__file__), "external_dfb_multiply.hpp"
)


@ttl.operation(grid=(1, 1))
def external_dfb_descriptors(lhs, rhs, result):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    first_result_dfb = ttl.make_dataflow_buffer_like(
        result, shape=(1, 1), block_count=2
    )
    result_consumed_dfb = ttl.make_dataflow_buffer_like(
        result, shape=(1, 1), block_count=2
    )
    second_result_dfb = ttl.make_dataflow_buffer_like(
        result, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def compute():
        with (
            lhs_dfb.wait(),
            rhs_dfb.wait(),
            first_result_dfb.reserve(),
        ):
            ttl.call_extern_func(
                EXTERNAL_MULTIPLY_HEADER,
                "external_dfb_multiply",
                template_args=[
                    ttl.dfb_descriptor(lhs_dfb),
                    ttl.dfb_descriptor(rhs_dfb),
                    ttl.dfb_descriptor(first_result_dfb),
                ],
            )

        with result_consumed_dfb.wait():
            pass

        with (
            lhs_dfb.wait(),
            rhs_dfb.wait(),
            second_result_dfb.reserve(),
        ):
            ttl.call_extern_func(
                EXTERNAL_MULTIPLY_HEADER,
                "external_dfb_multiply",
                template_args=[
                    ttl.dfb_descriptor(lhs_dfb),
                    ttl.dfb_descriptor(rhs_dfb),
                    ttl.dfb_descriptor(second_result_dfb),
                ],
            )

    @ttl.datamovement()
    def dm_read():
        lhs_destination = lhs_dfb.reserve()
        ttl.copy(lhs[0, 0], lhs_destination).wait()
        lhs_destination.push()

        first_rhs_destination = rhs_dfb.reserve()
        ttl.copy(rhs[0, 0], first_rhs_destination).wait()
        first_rhs_destination.push()
        second_lhs_destination = lhs_dfb.reserve()
        ttl.copy(lhs[0, 0], second_lhs_destination).wait()
        second_lhs_destination.push()
        second_rhs_destination = rhs_dfb.reserve()
        ttl.copy(rhs[0, 0], second_rhs_destination).wait()
        second_rhs_destination.push()

    @ttl.datamovement()
    def dm_write():
        first_result = first_result_dfb.wait()
        ttl.copy(first_result, result[0, 0]).wait()
        first_result.pop()

        result_consumed = result_consumed_dfb.reserve()
        result_consumed.push()

        second_result = second_result_dfb.wait()
        ttl.copy(second_result, result[0, 0]).wait()
        second_result.pop()


def _descriptor_result_indices(final_mlir_path):
    """Read emitted types because execution alone cannot prove index reuse."""
    final_mlir = final_mlir_path.read_text()
    calls = re.findall(
        r'emitc\.call_opaque "external_dfb_multiply".*?template_args = \[(.*?)\]',
        final_mlir,
    )
    assert len(calls) == 2
    descriptor_indices = [
        [int(index) for index in re.findall(r"DFBDescriptor<(\d+),", call)]
        for call in calls
    ]
    assert all(len(indices) == 3 for indices in descriptor_indices)
    return [indices[2] for indices in descriptor_indices]


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        lhs_host = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        rhs_host = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        lhs = ttnn.from_torch(
            lhs_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        rhs = ttnn.from_torch(
            rhs_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        result = ttnn.from_torch(
            torch.zeros_like(lhs_host),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            final_mlir_path = Path(temporary_directory) / "final.mlir"
            previous_final_mlir = os.environ.get("TTLANG_FINAL_MLIR")
            os.environ["TTLANG_FINAL_MLIR"] = str(final_mlir_path)
            try:
                external_dfb_descriptors(
                    lhs, rhs, result, options="--ttl-reuse-user-dfbs"
                )
            finally:
                if previous_final_mlir is None:
                    del os.environ["TTLANG_FINAL_MLIR"]
                else:
                    os.environ["TTLANG_FINAL_MLIR"] = previous_final_mlir

            first_index, second_index = _descriptor_result_indices(final_mlir_path)
            assert first_index != second_index
        torch.testing.assert_close(ttnn.to_torch(result), lhs_host * rhs_host)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
