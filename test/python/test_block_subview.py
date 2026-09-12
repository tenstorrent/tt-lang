# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Device coverage for tensor copies through DFB block subviews."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device


@ttl.operation(grid=(1, 1))
def copy_through_block_subviews(input_tensor, output_tensor):
    transfer_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 2), block_count=1
    )

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def read_input():
        transfer_block = transfer_dfb.reserve()
        first_tile = ttl.block.subview(transfer_block, offsets=(0, 0), shape=(1, 1))
        second_tile = ttl.block.subview(transfer_block, offsets=(0, 1), shape=(1, 1))
        ttl.copy(input_tensor[0:1, 0:1], first_tile).wait()
        ttl.copy(input_tensor[0:1, 1:2], second_tile).wait()
        transfer_block.push()

    @ttl.datamovement()
    def write_output():
        transfer_block = transfer_dfb.wait()
        first_tile = ttl.block.subview(transfer_block, offsets=(0, 0), shape=(1, 1))
        second_tile = ttl.block.subview(transfer_block, offsets=(0, 1), shape=(1, 1))
        ttl.copy(first_tile, output_tensor[0:1, 0:1]).wait()
        ttl.copy(second_tile, output_tensor[0:1, 1:2]).wait()
        transfer_block.pop()


@pytest.mark.parametrize(
    ("torch_dtype", "rtol", "atol"),
    [
        pytest.param(torch.bfloat16, 5e-2, 1.0, id="bf16"),
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_copy_through_block_subviews(device, torch_dtype, rtol, atol, memory):
    input_host = torch.randn((32, 64), dtype=torch_dtype)
    to_device = to_dram if memory == "dram" else to_l1
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    copy_through_block_subviews(input_tensor, output_tensor)

    result = ttnn.to_torch(output_tensor).float()
    assert_allclose(result, input_host.float(), rtol=rtol, atol=atol)
