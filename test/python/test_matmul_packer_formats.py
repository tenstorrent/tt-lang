# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul source formats remain independent of packer accumulation storage."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose, assert_pcc


def make_packer_matmul(block_tiles, k_block_tiles, k_blocks):
    @ttl.operation(grid=(1, 1), math_fidelity="HiFi4", fp32_dest_acc_en=True)
    def operation(activation, weight, output):
        activation_dfb = ttl.make_dataflow_buffer_like(
            activation, shape=(block_tiles, k_block_tiles), block_count=2
        )
        weight_dfb = ttl.make_dataflow_buffer_like(
            weight, shape=(k_block_tiles, block_tiles), block_count=2
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output, shape=(block_tiles, block_tiles), block_count=2
        )

        @ttl.compute()
        def compute():
            with output_dfb.reserve() as output_block:
                output_block.store(
                    ttl.block.fill(
                        0.0, shape=output_block.shape, dtype=output_block.dtype
                    )
                )
                for _k_block in range(k_blocks):
                    with (
                        activation_dfb.wait() as activation_block,
                        weight_dfb.wait() as weight_block,
                    ):
                        output_block += ttl.math.typecast(
                            activation_block @ weight_block, output_block.dtype
                        )

        @ttl.datamovement()
        def read():
            for k_block in range(k_blocks):
                k_begin = k_block * k_block_tiles
                k_end = k_begin + k_block_tiles
                with activation_dfb.reserve() as activation_block:
                    ttl.copy(
                        activation[0:block_tiles, k_begin:k_end], activation_block
                    ).wait()
                with weight_dfb.reserve() as weight_block:
                    ttl.copy(weight[k_begin:k_end, 0:block_tiles], weight_block).wait()

        @ttl.datamovement()
        def write():
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output[0:block_tiles, 0:block_tiles]).wait()

    return operation


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        pytest.param(torch.bfloat16, torch.bfloat16, id="bf16-bf16"),
        pytest.param(torch.bfloat16, torch.float32, id="bf16-fp32"),
        pytest.param(torch.float32, torch.float32, id="fp32-fp32"),
    ],
)
@pytest.mark.parametrize(
    "block_tiles", [1, 2, 3], ids=["one-tile", "dst-sized", "subblocked"]
)
def test_matmul_packer_formats(device, input_dtype, output_dtype, block_tiles):
    torch.manual_seed(0)
    k_block_tiles, k_blocks = 4, 8
    output_elements = block_tiles * 32
    k_elements = k_block_tiles * k_blocks * 32
    activation = torch.randn((output_elements, k_elements), dtype=input_dtype)
    weight = (
        torch.randn((k_elements, output_elements), dtype=input_dtype) / k_elements**0.5
    )
    output = to_dram(
        torch.zeros((output_elements, output_elements), dtype=output_dtype), device
    )
    make_packer_matmul(block_tiles, k_block_tiles, k_blocks)(
        to_dram(activation, device), to_dram(weight, device), output
    )
    actual = ttnn.to_torch(output).float()
    expected = activation.float() @ weight.float()
    tolerance = 0.05 if output_dtype == torch.bfloat16 else 0.005
    assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert_pcc(expected, actual, threshold=0.999)
