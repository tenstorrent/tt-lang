# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for synchronized multi-epoch DFB configuration."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import ttl_api  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device


def _make_reconfiguration_operation(data_format, grid_cols):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    first_boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )
    second_boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(grid_cols, 1))
    def reconfiguration_operation(
        first_input,
        first_output,
        second_input,
        second_output,
        third_input,
        third_output,
    ):
        first_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        first_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_source = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        second_result = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        third_source = ttl.make_dfb(data_format, shape=(2, 1), block_count=4)
        third_result = ttl.make_dfb(data_format, shape=(2, 1), block_count=4)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            with first_source.wait() as source:
                with first_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(first_boundary)
            with second_source.wait() as source:
                with second_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(second_boundary)
            with third_source.wait() as source:
                with third_result.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, _ = ttl.node(dims=2)
            with first_source.reserve() as destination:
                ttl.copy(first_input[0, node_x], destination).wait()
            ttl.reconfigure_dfbs(first_boundary)
            with second_source.reserve() as destination:
                ttl.copy(
                    second_input[0:1, node_x * 2 : node_x * 2 + 2],
                    destination,
                ).wait()
            ttl.reconfigure_dfbs(second_boundary)
            with third_source.reserve() as destination:
                ttl.copy(third_input[0:2, node_x : node_x + 1], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, _ = ttl.node(dims=2)
            with first_result.wait() as source:
                ttl.copy(source, first_output[0, node_x]).wait()
            ttl.reconfigure_dfbs(first_boundary)
            with second_result.wait() as source:
                ttl.copy(
                    source,
                    second_output[0:1, node_x * 2 : node_x * 2 + 2],
                ).wait()
            ttl.reconfigure_dfbs(second_boundary)
            with third_result.wait() as source:
                ttl.copy(source, third_output[0:2, node_x : node_x + 1]).wait()

    return reconfiguration_operation


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("grid_cols", [1, 2], ids=["one-core", "two-core"])
@pytest.mark.parametrize(
    "to_device",
    [to_dram, to_l1],
    ids=["dram", "l1"],
)
def test_reconfiguration_reuses_ids_with_different_capacity_and_cached_execution(
    device, dtype, grid_cols, to_device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_reconfiguration_operation(data_format, grid_cols)
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(tmp_path / "reconfiguration.mlir"))

    first_host = (
        torch.arange(32 * 32 * grid_cols, dtype=torch.float32)
        .reshape(32, 32 * grid_cols)
        .to(dtype)
    )
    second_host = (
        torch.arange(32 * 64 * grid_cols, dtype=torch.float32)
        .reshape(32, 64 * grid_cols)
        .remainder(257)
    ).to(dtype)
    third_host = (
        torch.arange(64 * 32 * grid_cols, dtype=torch.float32)
        .reshape(64, 32 * grid_cols)
        .remainder(193)
    ).to(dtype)
    first_output = to_device(torch.zeros_like(first_host), device)
    second_output = to_device(torch.zeros_like(second_host), device)
    third_output = to_device(torch.zeros_like(third_host), device)
    operation(
        to_device(first_host, device),
        first_output,
        to_device(second_host, device),
        second_output,
        to_device(third_host, device),
        third_output,
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = (tmp_path / "reconfiguration.mlir").read_text()
    allocation_metadata = final_mlir.partition("ttl.dfb_reconfiguration_plan")[0]
    assert allocation_metadata.count("dfb_index = ") == 2
    assert final_mlir.count("entry_reconfiguration = 0 : i64") == 2
    assert final_mlir.count("entry_reconfiguration = 1 : i64") == 2
    assert final_mlir.count("block_count = 4 : i32") == 2
    assert final_mlir.count("num_tiles = 2 : i32") == 4
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 6

    cached_first_host = (first_host.float() + 3).to(dtype)
    cached_second_host = (second_host.float() - 5).to(dtype)
    cached_third_host = (third_host.float() + 7).to(dtype)
    cached_first_output = to_device(torch.zeros_like(first_host), device)
    cached_second_output = to_device(torch.zeros_like(second_host), device)
    cached_third_output = to_device(torch.zeros_like(third_host), device)
    operation(
        to_device(cached_first_host, device),
        cached_first_output,
        to_device(cached_second_host, device),
        cached_second_output,
        to_device(cached_third_host, device),
        cached_third_output,
        options="--ttl-reuse-user-dfbs",
    )

    tolerance = (0.05, 1.0) if dtype == torch.bfloat16 else (1e-5, 1e-6)
    for actual, expected in (
        (first_output, first_host),
        (second_output, second_host),
        (third_output, third_host),
        (cached_first_output, cached_first_host),
        (cached_second_output, cached_second_host),
        (cached_third_output, cached_third_host),
    ):
        assert_allclose(
            ttnn.to_torch(actual).float(),
            expected.float(),
            rtol=tolerance[0],
            atol=tolerance[1],
        )
