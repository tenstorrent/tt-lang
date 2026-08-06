# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for external NOC functions receiving TensorAccessor."""

import os
from functools import partial

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttlang_test_utils import to_dram, to_l1, to_l1_sharded
from utils.correctness import assert_allclose


TENSOR_ACCESSOR_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "tensor_accessor_read.hpp"
)
MEMORY_CONFIGS = [
    pytest.param(to_dram, id="dram-interleaved"),
    pytest.param(to_l1, id="l1-interleaved"),
    pytest.param(partial(to_l1_sharded, layout="height"), id="l1-height-sharded"),
    pytest.param(partial(to_l1_sharded, layout="width"), id="l1-width-sharded"),
    pytest.param(partial(to_l1_sharded, layout="block"), id="l1-block-sharded"),
]


def _make_tensor_accessor_copy(data_format):
    """Compile each page byte size into its DFB descriptor template argument."""

    @ttl.operation(grid=(1, 1))
    def tensor_accessor_copy(inp, out):
        transfer_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm_read():
            ttl.call_extern_func(
                TENSOR_ACCESSOR_HEADER,
                "tensor_accessor_read",
                template_args=[ttl.dfb_descriptor(transfer_dfb)],
                func_args=[inp],
            )

        @ttl.datamovement()
        def dm_write():
            source = transfer_dfb.wait()
            ttl.copy(source, out[0, 0]).wait()
            source.pop()

    return tensor_accessor_copy


BF16_TENSOR_ACCESSOR_COPY = _make_tensor_accessor_copy("bf16")
F32_TENSOR_ACCESSOR_COPY = _make_tensor_accessor_copy("float32")


def _make_multitile_tensor_accessor_copy(data_format):
    @ttl.operation(grid=(1, 1))
    def multitile_tensor_accessor_copy(inp, out):
        transfer_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm_read():
            ttl.call_extern_func(
                TENSOR_ACCESSOR_HEADER,
                "tensor_accessor_read_page",
                template_args=[ttl.dfb_descriptor(transfer_dfb), 3],
                func_args=[inp],
            )

        @ttl.datamovement()
        def dm_write():
            source = transfer_dfb.wait()
            ttl.copy(source, out[0, 0]).wait()
            source.pop()

    return multitile_tensor_accessor_copy


BF16_MULTITILE_TENSOR_ACCESSOR_COPY = _make_multitile_tensor_accessor_copy("bf16")
F32_MULTITILE_TENSOR_ACCESSOR_COPY = _make_multitile_tensor_accessor_copy("float32")


def _make_tensor_accessor_pair_copy(data_format):
    """Preserve argument order for two independent TensorAccessor values."""

    @ttl.operation(grid=(1, 1))
    def tensor_accessor_pair_copy(first_inp, second_inp, first_out, second_out):
        first_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm_read():
            ttl.call_extern_func(
                TENSOR_ACCESSOR_HEADER,
                "tensor_accessor_pair_read",
                template_args=[
                    ttl.dfb_descriptor(first_dfb),
                    ttl.dfb_descriptor(second_dfb),
                ],
                func_args=[first_inp, second_inp],
            )

        @ttl.datamovement()
        def dm_write():
            first_source = first_dfb.wait()
            ttl.copy(first_source, first_out[0, 0]).wait()
            first_source.pop()
            second_source = second_dfb.wait()
            ttl.copy(second_source, second_out[0, 0]).wait()
            second_source.pop()

    return tensor_accessor_pair_copy


BF16_TENSOR_ACCESSOR_PAIR_COPY = _make_tensor_accessor_pair_copy("bf16")
F32_TENSOR_ACCESSOR_PAIR_COPY = _make_tensor_accessor_pair_copy("float32")


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (BF16_TENSOR_ACCESSOR_COPY, torch.bfloat16),
        (F32_TENSOR_ACCESSOR_COPY, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("to_device", MEMORY_CONFIGS)
def test_external_tensor_accessor(device, operation, dtype, to_device):
    """TensorAccessor preserves one tiled page across dtype and memory types."""

    # TTNN interop supports tiled tensors for this external kernel interface.
    host = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32).to(dtype)
    inp = to_device(host, device)
    out = to_device(torch.zeros_like(host), device)

    operation(inp, out)

    actual = ttnn.to_torch(out).float()
    expected = host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (BF16_MULTITILE_TENSOR_ACCESSOR_COPY, torch.bfloat16),
        (F32_MULTITILE_TENSOR_ACCESSOR_COPY, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("to_device", MEMORY_CONFIGS)
def test_external_tensor_accessor_multitile(device, operation, dtype, to_device):
    """TensorAccessor page IDs address nonzero tiles in a larger tensor."""

    host = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64).to(dtype)
    inp = to_device(host, device)
    out = to_device(torch.zeros((32, 32), dtype=dtype), device)

    operation(inp, out)

    actual = ttnn.to_torch(out).float()
    expected = host[32:64, 32:64].float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (BF16_TENSOR_ACCESSOR_PAIR_COPY, torch.bfloat16),
        (F32_TENSOR_ACCESSOR_PAIR_COPY, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("to_device", MEMORY_CONFIGS)
def test_external_tensor_accessor_operand_order(device, operation, dtype, to_device):
    """Two TensorAccessor operands retain source order and runtime addresses."""

    element_indices = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    first_host = element_indices.to(dtype)
    second_host = (1000 - element_indices).to(dtype)
    first_inp = to_device(first_host, device)
    second_inp = to_device(second_host, device)
    first_out = to_device(torch.zeros_like(first_host), device)
    second_out = to_device(torch.zeros_like(second_host), device)
    swapped_first_out = to_device(torch.zeros_like(second_host), device)
    swapped_second_out = to_device(torch.zeros_like(first_host), device)

    operation(first_inp, second_inp, first_out, second_out)
    operation(second_inp, first_inp, swapped_first_out, swapped_second_out)

    first_actual = ttnn.to_torch(first_out).float()
    second_actual = ttnn.to_torch(second_out).float()
    swapped_first_actual = ttnn.to_torch(swapped_first_out).float()
    swapped_second_actual = ttnn.to_torch(swapped_second_out).float()
    first_expected = first_host.float()
    second_expected = second_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(first_actual, first_expected, rtol=0.05, atol=1.0)
        assert_allclose(second_actual, second_expected, rtol=0.05, atol=1.0)
        assert_allclose(swapped_first_actual, second_expected, rtol=0.05, atol=1.0)
        assert_allclose(swapped_second_actual, first_expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(first_actual, first_expected, rtol=1e-5, atol=1e-6)
        assert_allclose(second_actual, second_expected, rtol=1e-5, atol=1e-6)
        assert_allclose(swapped_first_actual, second_expected, rtol=1e-5, atol=1e-6)
        assert_allclose(swapped_second_actual, first_expected, rtol=1e-5, atol=1e-6)
