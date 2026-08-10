# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for runtime tensor addresses in external functions."""

import os
from functools import partial

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttlang_test_utils import to_dram, to_l1, to_l1_sharded


RAW_ADDRESS_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "raw_address_capture.hpp"
)
OUTPUT_WORD_COUNT = 32 * 32
INPUT_MEMORY_CONFIGS = [
    pytest.param(to_dram, id="dram-interleaved"),
    pytest.param(to_l1, id="l1-interleaved"),
    pytest.param(partial(to_l1_sharded, layout="height"), id="l1-height-sharded"),
    pytest.param(partial(to_l1_sharded, layout="width"), id="l1-width-sharded"),
    pytest.param(partial(to_l1_sharded, layout="block"), id="l1-block-sharded"),
]


@ttl.operation(grid=(1, 1))
def external_raw_address_capture(inp, out):
    address_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        ttl.call_extern_func(
            RAW_ADDRESS_HEADER,
            "raw_address_capture",
            template_args=[ttl.dfb_descriptor(address_dfb)],
            func_args=[ttl.raw_addr(inp)],
        )

    @ttl.datamovement()
    def dm_write():
        address_block = address_dfb.wait()
        ttl.copy(address_block, out[0, 0]).wait()
        address_block.pop()


@ttl.operation(grid=(1, 1))
def external_compute_raw_address_capture(inp, out):
    @ttl.compute()
    def compute():
        ttl.call_extern_func(
            RAW_ADDRESS_HEADER,
            "raw_address_capture_compute",
            template_args=[OUTPUT_WORD_COUNT],
            func_args=[ttl.raw_addr(inp), ttl.raw_addr(out)],
        )

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(1, 1))
def external_unified_raw_address_capture(inp, out):
    call_extern_func(
        RAW_ADDRESS_HEADER,
        "raw_address_capture_unified",
        template_args=[OUTPUT_WORD_COUNT],
        func_args=[ttl.raw_addr(inp), ttl.raw_addr(out)],
    )


def _assert_address_bits(output, expected_address):
    """Compare bits because arbitrary addresses need not encode finite floats."""
    output_bits = ttnn.to_torch(output).contiguous().view(torch.int32)
    expected_bits = torch.tensor(expected_address, dtype=torch.uint32).view(torch.int32)
    assert torch.all(output_bits == expected_bits)


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("to_input", INPUT_MEMORY_CONFIGS)
@pytest.mark.parametrize(
    "operation",
    [
        external_raw_address_capture,
        external_compute_raw_address_capture,
        external_unified_raw_address_capture,
    ],
    ids=["noc", "compute", "unified"],
)
def test_external_raw_address_uses_each_runtime_tensor(
    device, dtype, to_input, operation
):
    """A cached program must read each invocation's common runtime argument."""
    host_input = torch.zeros((32, 32), dtype=dtype)
    first_input = to_input(host_input, device)
    second_input = to_input(host_input, device)
    assert first_input.buffer_address() != second_input.buffer_address()

    host_output = torch.zeros((32, 32), dtype=torch.float32)
    if operation in (
        external_compute_raw_address_capture,
        external_unified_raw_address_capture,
    ):
        first_output = to_l1_sharded(host_output, device, layout="height")
        second_output = to_l1_sharded(host_output, device, layout="height")
    else:
        first_output = to_l1(host_output, device)
        second_output = to_l1(host_output, device)

    operation(first_input, first_output)
    operation(second_input, second_output)

    _assert_address_bits(first_output, first_input.buffer_address())
    _assert_address_bits(second_output, second_input.buffer_address())
