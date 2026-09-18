# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for protocol actions performed inside external C++."""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

TILE = 32
EXTERNAL_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "external_eltwise_mul.hpp"
)


@ttl.operation(grid=(1, 1))
def external_effect_operation(lhs, rhs, result):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    result_dfb = ttl.make_dataflow_buffer_like(result, shape=(1, 1), block_count=2)

    lhs_destination = lhs_dfb.reserve()
    ttl.copy(lhs[0, 0], lhs_destination).wait()
    lhs_destination.push()
    rhs_destination = rhs_dfb.reserve()
    ttl.copy(rhs[0, 0], rhs_destination).wait()
    rhs_destination.push()

    ttl.call_extern_func(
        EXTERNAL_HEADER,
        "ttl_external_eltwise_mul",
        template_args=[
            ttl.dfb_descriptor(lhs_dfb),
            ttl.dfb_descriptor(rhs_dfb),
            ttl.dfb_descriptor(result_dfb),
        ],
        dfb_effects=[
            ttl.DFBEffect.repeat(
                1,
                [
                    ttl.DFBEffect.reserve(result_dfb, tiles=TILE // TILE),
                    ttl.DFBEffect.wait(lhs_dfb, tiles=1),
                    ttl.DFBEffect.wait(rhs_dfb, tiles=1),
                    ttl.DFBEffect.pop(lhs_dfb, tiles=1),
                    ttl.DFBEffect.pop(rhs_dfb, tiles=1),
                    ttl.DFBEffect.push(result_dfb, tiles=1),
                ],
            )
        ],
        kernel=ttl.KernelKind.COMPUTE,
    )

    result_source = result_dfb.wait()
    ttl.copy(result_source, result[0, 0]).wait()
    result_source.pop()


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_external_dfb_protocol_effects(device, dtype, memory_config, to_device):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    lhs_host = ((element_indices.remainder(41) - 20) / 16).to(dtype)
    rhs_host = (((3 * element_indices).remainder(37) - 18) / 16).to(dtype)

    lhs = to_device(lhs_host, device)
    rhs = to_device(rhs_host, device)
    result = to_device(torch.zeros_like(lhs_host), device)

    external_effect_operation(lhs, rhs, result)

    actual = ttnn.to_torch(result).float()
    expected = lhs_host.float() * rhs_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=5e-3, atol=1e-4)
