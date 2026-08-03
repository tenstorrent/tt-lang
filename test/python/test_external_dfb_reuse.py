# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for DFB allocation with an external C++ operation.

The external multiply follows a direct-DFB interface: every accessed DFB is a
function argument, while the C++ body owns its compute-thread DFB protocol.
The allocation pass must remain correct with reuse enabled even though that
hidden protocol prevents proving these DFBs bounded.
"""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

# TTNN interop rejects non-tilized tensors before DFB lowering, so TILE is the
# only supported tensor layout for these runtime cases.
TILE = 32
OVER_CAPACITY_COMPOSITION_LEVELS = 5
EXTERNAL_COMPOSITION_LOGICAL_DFBS = (1 << OVER_CAPACITY_COMPOSITION_LEVELS) + 3
EXTERNAL_MULTIPLY_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "external_eltwise_mul.hpp"
)


@ttl.operation()
def _external_eltwise_mul(lhs: ttl.DFB, rhs: ttl.DFB, result: ttl.DFB):
    call_extern_func(
        EXTERNAL_MULTIPLY_HEADER,
        "ttl_external_eltwise_mul",
        func_args=[lhs, rhs, result],
    )


def _make_external_multiply_kernel(data_format):
    @ttl.operation(grid=(1, 1))
    def external_multiply_kernel(lhs, rhs, result):
        lhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        rhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        lhs_destination = lhs_dfb.reserve()
        ttl.copy(lhs[0, 0], lhs_destination).wait()
        lhs_destination.push()
        rhs_destination = rhs_dfb.reserve()
        ttl.copy(rhs[0, 0], rhs_destination).wait()
        rhs_destination.push()

        _external_eltwise_mul(lhs_dfb, rhs_dfb, result_dfb)

        result_source = result_dfb.wait()
        ttl.copy(result_source, result[0, 0]).wait()
        result_source.pop()

    return external_multiply_kernel


def _make_nested_copy_atom(data_format, level_count):
    @ttl.operation()
    def copy_stage(source: ttl.DFB, destination: ttl.DFB):
        destination_block = destination.reserve()
        destination_block.store(source.wait())

    nested_copy = copy_stage
    for _composition_level in range(level_count):
        inner_copy = nested_copy

        @ttl.operation()
        def doubled_copy(source: ttl.DFB, destination: ttl.DFB):
            intermediate_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
            inner_copy(source, intermediate_dfb)
            inner_copy(intermediate_dfb, destination)

        nested_copy = doubled_copy

    return nested_copy


def _make_external_composition_kernel(data_format):
    nested_copy = _make_nested_copy_atom(data_format, OVER_CAPACITY_COMPOSITION_LEVELS)

    @ttl.operation(grid=(1, 1))
    def external_composition_kernel(lhs, rhs, result):
        lhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        rhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        product_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        lhs_destination = lhs_dfb.reserve()
        ttl.copy(lhs[0, 0], lhs_destination).wait()
        lhs_destination.push()
        rhs_destination = rhs_dfb.reserve()
        ttl.copy(rhs[0, 0], rhs_destination).wait()
        rhs_destination.push()

        _external_eltwise_mul(lhs_dfb, rhs_dfb, product_dfb)
        nested_copy(product_dfb, result_dfb)

        result_source = result_dfb.wait()
        ttl.copy(result_source, result[0, 0]).wait()
        result_source.pop()

    return external_composition_kernel


_external_bf16_multiply = _make_external_multiply_kernel("bf16")
_external_f32_multiply = _make_external_multiply_kernel("float32")
_external_bf16_composition = _make_external_composition_kernel("bf16")
_external_f32_composition = _make_external_composition_kernel("float32")

assert EXTERNAL_COMPOSITION_LOGICAL_DFBS > 32


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_external_bf16_multiply, torch.bfloat16),
        (_external_f32_multiply, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize("reuse_user_dfbs", [True, False], ids=["reuse", "distinct"])
def test_external_multiply_with_dfb_allocation(
    device, operation, dtype, memory_config, to_device, reuse_user_dfbs
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    lhs_host = ((element_indices.remainder(41) - 20) / 16).to(dtype)
    rhs_host = (((3 * element_indices).remainder(37) - 18) / 16).to(dtype)

    lhs = to_device(lhs_host, device)
    rhs = to_device(rhs_host, device)
    result = to_device(torch.zeros_like(lhs_host), device)

    reuse_option = (
        "--ttl-reuse-user-dfbs" if reuse_user_dfbs else "--no-ttl-reuse-user-dfbs"
    )
    operation(lhs, rhs, result, options=reuse_option)

    actual = ttnn.to_torch(result).float()
    expected = lhs_host.float() * rhs_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_external_bf16_composition, torch.bfloat16),
        (_external_f32_composition, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_external_composition_requires_dfb_reuse(
    device, operation, dtype, memory_config, to_device
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    lhs_host = ((element_indices.remainder(41) - 20) / 16).to(dtype)
    rhs_host = (((3 * element_indices).remainder(37) - 18) / 16).to(dtype)

    lhs = to_device(lhs_host, device)
    rhs = to_device(rhs_host, device)
    result = to_device(torch.zeros_like(lhs_host), device)

    # The disabled mode requires one physical index for each of the 35 logical
    # DFBs, proving that enabled execution cannot fit without index reuse.
    with pytest.raises(
        RuntimeError,
        match=("need 35 DFB indices " "but hardware supports at most 32"),
    ):
        operation(lhs, rhs, result, options="--no-ttl-reuse-user-dfbs")

    operation(lhs, rhs, result, options="--ttl-reuse-user-dfbs")

    actual = ttnn.to_torch(result).float()
    expected = lhs_host.float() * rhs_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)
