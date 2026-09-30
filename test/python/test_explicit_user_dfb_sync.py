# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Explicit external protocols retain automatic compiler-intermediate synchronization."""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device
EXTERNAL_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "external_eltwise_mul.hpp"
)


@ttl.operation(grid=(1, 1), options="--no-ttl-auto-sync-user-dfbs")
def explicit_external_protocol(lhs, rhs, result):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    result_dfb = ttl.make_dataflow_buffer_like(result, shape=(1, 1), block_count=2)

    for iteration in range(64):
        if iteration % 2 == 0:
            lhs_destination = lhs_dfb.reserve()
            ttl.copy(lhs[0, 0], lhs_destination).wait()
            lhs_destination.push()
            rhs_destination = rhs_dfb.reserve()
            ttl.copy(rhs[0, 0], rhs_destination).wait()
            rhs_destination.push()

            # The header executes its queue protocol only on the compute kernel.
            # Its data-movement calls remain opaque to synchronization inference.
            ttl.call_extern_func(
                EXTERNAL_HEADER,
                "ttl_external_eltwise_mul",
                template_args=[
                    ttl.dfb_descriptor(lhs_dfb),
                    ttl.dfb_descriptor(rhs_dfb),
                    ttl.dfb_descriptor(result_dfb),
                ],
                kernel=(
                    ttl.KernelKind.COMPUTE,
                    ttl.KernelKind.DATA_MOVEMENT,
                    ttl.PIPE_SOURCE_KERNEL,
                ),
            )

            result_source = result_dfb.wait()
            ttl.copy(result_source, result[0, 0]).wait()
            result_source.pop()


def make_compiler_intermediate_operation(auto_sync_user_dfbs):
    option = (
        "--ttl-auto-sync-user-dfbs"
        if auto_sync_user_dfbs
        else "--no-ttl-auto-sync-user-dfbs"
    )

    @ttl.operation(grid=(1, 1), options=option)
    def compiler_intermediate_operation(lhs, rhs, result):
        lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
        rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dataflow_buffer_like(result, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with lhs_dfb.wait() as lhs_value, rhs_dfb.wait() as rhs_value:
                summed = ttl.add(lhs_value, rhs_value)
                with result_dfb.reserve() as output_value:
                    output_value.store(ttl.math.reduce_sum(summed, dims=[0, 1]))

        @ttl.datamovement()
        def read():
            with lhs_dfb.reserve() as destination:
                ttl.copy(lhs[0, 0], destination).wait()
            with rhs_dfb.reserve() as destination:
                ttl.copy(rhs[0, 0], destination).wait()

        @ttl.datamovement()
        def write():
            with result_dfb.wait() as source:
                ttl.copy(source, result[0, 0]).wait()

    return compiler_intermediate_operation


@pytest.fixture(params=[torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def tensor_dtype(request):
    return request.param


@pytest.fixture(params=[to_dram, to_l1], ids=["dram", "l1"])
def tensor_transfer(request):
    return request.param


def test_explicit_external_protocol(device, tensor_dtype, tensor_transfer):
    indices = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    lhs_host = (indices.remainder(8) / 8).to(tensor_dtype)
    rhs_host = ((indices.remainder(4) + 1) / 4).to(tensor_dtype)
    lhs = tensor_transfer(lhs_host, device)
    rhs = tensor_transfer(rhs_host, device)
    result = tensor_transfer(torch.zeros_like(lhs_host), device)

    explicit_external_protocol(lhs, rhs, result)

    tolerance = 0.05 if tensor_dtype == torch.bfloat16 else 1e-5
    assert_allclose(
        ttnn.to_torch(result).float(),
        lhs_host.float() * rhs_host.float(),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize(
    "auto_sync_user_dfbs", [True, False], ids=["automatic", "explicit"]
)
def test_compiler_intermediate_sync(
    device, tensor_dtype, tensor_transfer, auto_sync_user_dfbs
):
    lhs_host = torch.full((32, 32), 0.25, dtype=tensor_dtype)
    rhs_host = torch.full((32, 32), 0.5, dtype=tensor_dtype)
    lhs = tensor_transfer(lhs_host, device)
    rhs = tensor_transfer(rhs_host, device)
    result = tensor_transfer(torch.zeros_like(lhs_host), device)

    operation = make_compiler_intermediate_operation(auto_sync_user_dfbs)
    operation(lhs, rhs, result)

    tolerance = 0.05 if tensor_dtype == torch.bfloat16 else 1e-5
    assert_allclose(
        ttnn.to_torch(result).float()[0, 0],
        (lhs_host.float() + rhs_host.float()).sum(),
        rtol=tolerance,
        atol=tolerance,
    )
