# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for discarding DFB state during reconfiguration."""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import ttl_api  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose, assert_pcc  # noqa: E402

pytestmark = pytest.mark.requires_device

DFB_RECONFIGURATION_TEST_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "dfb_reconfiguration_test_helpers.hpp"
)


def _make_repeated_discarded_opaque_state_operation(data_format, iterations, grid_cols):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    first_boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel),
        discard_dfb_state=True,
    )
    second_boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel),
        discard_dfb_state=True,
    )

    @ttl.operation(grid=(grid_cols, 1))
    def repeated_discarded_opaque_state_operation(input_tensor, output_tensor):
        unread_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        copied_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=3)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            for _iteration in range(iterations):
                ttl.reconfigure_dfbs(first_boundary)
                ttl.reconfigure_dfbs(second_boundary)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, _node_y = ttl.node(dims=2)
            for iteration in range(iterations):
                if iteration == 1:
                    ttl.call_extern_func(
                        DFB_RECONFIGURATION_TEST_HEADER,
                        "publish_unread_tile",
                        template_args=[ttl.dfb_descriptor(unread_dfb)],
                        dfb_effects=[
                            ttl.DFBEffect.reserve(unread_dfb, tiles=1),
                            ttl.DFBEffect.push(unread_dfb, tiles=1),
                        ],
                    )
                ttl.reconfigure_dfbs(first_boundary)
                with copied_dfb.reserve() as destination:
                    ttl.copy(
                        input_tensor[
                            iteration : iteration + 1,
                            node_x : node_x + 1,
                        ],
                        destination,
                    ).wait()
                ttl.reconfigure_dfbs(second_boundary)

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, _node_y = ttl.node(dims=2)
            for iteration in range(iterations):
                ttl.reconfigure_dfbs(first_boundary)
                with copied_dfb.wait() as source:
                    ttl.copy(
                        source,
                        output_tensor[
                            iteration : iteration + 1,
                            node_x : node_x + 1,
                        ],
                    ).wait()
                ttl.reconfigure_dfbs(second_boundary)

    return repeated_discarded_opaque_state_operation


def _make_discarded_wait_state_operation(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reconfigure = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel),
        discard_dfb_state=True,
    )

    @ttl.operation(grid=(1, 1))
    def discarded_wait_state_operation(input_tensor, output_tensor):
        stale_dfb = ttl.make_dfb(
            data_format,
            shape=(1, 1),
            block_count=2,
        )
        current_dfb = ttl.make_dfb(
            data_format,
            shape=(1, 1),
            block_count=2,
        )

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.call_extern_func(
                DFB_RECONFIGURATION_TEST_HEADER,
                "wait_without_pop",
                template_args=[ttl.dfb_descriptor(stale_dfb)],
                dfb_effects=[ttl.DFBEffect.wait(stale_dfb, tiles=1)],
            )
            ttl.reconfigure_dfbs(reconfigure)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with stale_dfb.reserve() as stale_destination:
                ttl.copy(input_tensor[0, 0], stale_destination).wait()
            ttl.reconfigure_dfbs(reconfigure)
            with current_dfb.reserve() as current_destination:
                ttl.copy(input_tensor[0, 1], current_destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            ttl.reconfigure_dfbs(reconfigure)
            with current_dfb.wait() as current_source:
                ttl.copy(current_source, output_tensor[0, 0]).wait()

    return discarded_wait_state_operation


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_repeated_reconfiguration_discards_opaque_state_before_reuse(
    device, dtype, to_device, monkeypatch, tmp_path
):
    """A synchronized reconfiguration discards conditional external state."""
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    iterations = 3
    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_repeated_discarded_opaque_state_operation(
        data_format, iterations, grid_cols=2
    )
    initial_mlir_file = tmp_path / "discarded_opaque_state.initial.mlir"
    final_mlir_file = tmp_path / "discarded_opaque_state.final.mlir"
    monkeypatch.setenv("TTLANG_INITIAL_MLIR", str(initial_mlir_file))
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_file))
    host_input = (
        torch.arange(iterations * 32 * 64, dtype=torch.float32)
        .reshape(iterations * 32, 64)
        .remainder(257)
        .to(dtype)
    )
    output = to_device(torch.zeros_like(host_input), device)

    operation(
        to_device(host_input, device),
        output,
        options="--ttl-reuse-user-dfbs",
    )

    initial_mlir = initial_mlir_file.read_text()
    final_mlir = final_mlir_file.read_text()
    allocation_metadata = final_mlir.partition("ttl.dfb_reconfiguration_plan")[0]
    assert allocation_metadata.count("dfb_index = ") == 1
    assert initial_mlir.count("discard_dfb_state = true") == 6
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 6
    actual = ttnn.to_torch(output).float()
    expected = host_input.float()
    assert_pcc(expected, actual, 0.9999)
    tolerance = (0.05, 1.0) if dtype == torch.bfloat16 else (1e-5, 1e-6)
    assert_allclose(actual, expected, rtol=tolerance[0], atol=tolerance[1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_reconfiguration_discards_wait_state_before_reuse(
    device, dtype, to_device, monkeypatch, tmp_path
):
    """Reconfiguration discards a waited but unconsumed DFB payload."""
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_discarded_wait_state_operation(data_format)
    final_mlir_file = tmp_path / "discarded_wait_state.final.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_file))
    host_input = (
        torch.arange(2 * 32 * 32, dtype=torch.float32)
        .reshape(32, 64)
        .remainder(257)
        .to(dtype)
    )
    output = to_device(torch.zeros((32, 32), dtype=dtype), device)

    operation(
        to_device(host_input, device),
        output,
        options="--ttl-reuse-user-dfbs",
    )

    allocation_metadata = final_mlir_file.read_text().partition(
        "ttl.dfb_reconfiguration_plan"
    )[0]
    assert allocation_metadata.count("dfb_index = ") == 1
    actual = ttnn.to_torch(output).float()
    expected = host_input[:, 32:].float()
    assert_pcc(expected, actual, 0.9999)
    tolerance = (0.05, 1.0) if dtype == torch.bfloat16 else (1e-5, 1e-6)
    assert_allclose(actual, expected, rtol=tolerance[0], atol=tolerance[1])
