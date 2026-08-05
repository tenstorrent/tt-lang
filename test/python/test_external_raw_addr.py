# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for runtime tensor addresses in external functions."""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttlang_test_utils import to_dram, to_l1


RAW_ADDRESS_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "raw_address_capture.hpp"
)


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


def _assert_address_bits(output, expected_address):
    """Compare bits because arbitrary addresses need not encode finite floats."""
    output_bits = ttnn.to_torch(output).contiguous().view(torch.int32)
    expected_bits = torch.tensor(expected_address, dtype=torch.uint32).view(torch.int32)
    assert torch.all(output_bits == expected_bits)


@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_external_raw_address_uses_each_runtime_tensor(device, to_device):
    """A cached program must read each invocation's common runtime argument."""
    host = torch.zeros((32, 32), dtype=torch.float32)
    first_input = to_device(host, device)
    second_input = to_device(host, device)
    assert first_input.buffer_address() != second_input.buffer_address()

    first_output = to_device(host, device)
    second_output = to_device(host, device)

    external_raw_address_capture(first_input, first_output)
    external_raw_address_capture(second_input, second_output)

    _assert_address_bits(first_output, first_input.buffer_address())
    _assert_address_bits(second_output, second_input.buffer_address())
