# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for static DFB descriptor packing."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import kernel_runner  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

TILE = 32


def _make_static_dfb_packing_kernel(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(2, 1))
    def static_dfb_packing_kernel(input_tensor, output_tensor):
        first_node_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        shared_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=4)
        second_node_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=4)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            pass

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_node_dfb.reserve() as first_node_block:
                    ttl.copy(input_tensor[0, 0], first_node_block).wait()
            with shared_dfb.reserve() as shared_block:
                ttl.copy(input_tensor[0, node_x], shared_block).wait()
            if node_x == 1:
                with second_node_dfb.reserve() as second_node_block:
                    ttl.copy(input_tensor[0, 1], second_node_block).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_node_dfb.wait():
                    pass
            with shared_dfb.wait() as shared_block:
                ttl.copy(shared_block, output_tensor[0, node_x]).wait()
            if node_x == 1:
                with second_node_dfb.wait():
                    pass

    return static_dfb_packing_kernel


# Device-reported tensor addresses determine the remaining static DFB interval.
def test_remaining_l1_budget_matches_live_tensor_addresses(device):
    input_host = torch.zeros((TILE, TILE), dtype=torch.bfloat16)
    input_tensor = to_l1(input_host, device)
    assert input_tensor is not None

    static_dfb_base_address = ttnn.get_allocator_base_address(
        device, ttnn.BufferType.L1
    )
    l1_pages = [
        page
        for page in ttnn._ttnn.reports.get_buffer_pages(device)
        if page.buffer_type == ttnn.BufferType.L1
    ]
    assert l1_pages
    cores = {(page.core_x, page.core_y) for page in l1_pages}
    expected_remaining_by_core = {
        core: min(
            page.page_address for page in l1_pages if (page.core_x, page.core_y) == core
        )
        - static_dfb_base_address
        for core in cores
    }

    assert (
        kernel_runner._get_remaining_l1_by_core_for_device(device, cores)
        == expected_remaining_by_core
    )
    assert kernel_runner.get_min_remaining_l1_for_device(device) == min(
        expected_remaining_by_core.values()
    )


# Reordered static descriptors fit per-core L1 and preserve device results.
@pytest.mark.parametrize(
    ("data_format", "dtype", "dfb_page_size"),
    [
        ("bf16", torch.bfloat16, 2048),
        ("float32", torch.float32, 4096),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_static_dfb_descriptor_packing_fits_budget(
    device,
    data_format,
    dtype,
    dfb_page_size,
    to_device,
    monkeypatch,
):
    operation = _make_static_dfb_packing_kernel(data_format)
    input_host = (
        torch.arange(2 * TILE * TILE, dtype=torch.float32)
        .reshape(TILE, 2 * TILE)
        .to(dtype)
    )
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)
    descriptor_orders = []
    original_ordering = kernel_runner._order_static_dfb_descriptor_plans

    def record_descriptor_order(descriptor_plans, remaining_bytes_by_core):
        plans_by_physical_index = {
            plan.physical_index: plan for plan in descriptor_plans
        }
        over_budget_plans = [
            plans_by_physical_index[physical_index] for physical_index in (0, 2, 1)
        ]
        ordered_plans = original_ordering(over_budget_plans, remaining_bytes_by_core)
        descriptor_orders.append(
            tuple(
                plan.physical_index for plan in ordered_plans if plan.has_static_storage
            )
        )
        return ordered_plans

    monkeypatch.setattr(
        kernel_runner,
        "_get_remaining_l1_by_core_for_device",
        lambda _device, cores: {core: (17 * dfb_page_size) // 2 for core in cores},
    )
    monkeypatch.setattr(
        kernel_runner,
        "_order_static_dfb_descriptor_plans",
        record_descriptor_order,
    )

    operation(input_tensor, output_tensor, options="--no-ttl-specialize-cores")

    assert descriptor_orders == [(0, 1, 2)]
    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
