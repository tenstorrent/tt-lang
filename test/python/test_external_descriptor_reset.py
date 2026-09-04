# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device regression for core-specialized physical DFB descriptor dependencies."""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import ttl_api  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

TILE = 32
REPEATED_DFB_TRANSACTIONS_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "repeated_dfb_transactions.hpp"
)


def _make_external_descriptor_reset_kernel(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel),
    )

    @ttl.operation(grid=(2, 1))
    def external_descriptor_reset_kernel(input_tensor, output_tensor):
        external_stream = ttl.make_dfb(
            data_format,
            shape=(1, 4),
            block_count=3,
        )

        @ttl.compute(kernel=compute_kernel)
        def compute():
            core_x, _core_y = ttl.node(dims=2)
            if core_x == 1:
                ttl.reset_dfbs(reset, dfbs=[external_stream])

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            core_x, _core_y = ttl.node(dims=2)
            if core_x == 1:
                ttl.call_extern_func(
                    REPEATED_DFB_TRANSACTIONS_HEADER,
                    "read_high_water_dfb_logical_dm",
                    template_args=[ttl.dfb_descriptor(external_stream)],
                    func_args=[input_tensor],
                    dfb_effects=[
                        ttl.DFBEffect.reserve(external_stream, tiles=8),
                        ttl.DFBEffect.push(external_stream, tiles=4),
                        ttl.DFBEffect.reserve(external_stream, tiles=8),
                        ttl.DFBEffect.push(external_stream, tiles=4),
                        ttl.DFBEffect.reserve(external_stream, tiles=8),
                        ttl.DFBEffect.push(external_stream, tiles=4),
                        ttl.DFBEffect.reserve(external_stream, tiles=8),
                        ttl.DFBEffect.push(external_stream, tiles=4),
                    ],
                )
                ttl.reset_dfbs(reset, dfbs=[external_stream])

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            core_x, _core_y = ttl.node(dims=2)
            if core_x == 1:
                ttl.call_extern_func(
                    REPEATED_DFB_TRANSACTIONS_HEADER,
                    "write_high_water_dfb_logical_dm",
                    template_args=[ttl.dfb_descriptor(external_stream)],
                    func_args=[output_tensor],
                    dfb_effects=[
                        ttl.DFBEffect.wait(external_stream, tiles=4),
                        ttl.DFBEffect.pop(external_stream, tiles=4),
                        ttl.DFBEffect.wait(external_stream, tiles=4),
                        ttl.DFBEffect.pop(external_stream, tiles=4),
                        ttl.DFBEffect.wait(external_stream, tiles=4),
                        ttl.DFBEffect.pop(external_stream, tiles=4),
                        ttl.DFBEffect.wait(external_stream, tiles=4),
                        ttl.DFBEffect.pop(external_stream, tiles=4),
                    ],
                )
                ttl.reset_dfbs(reset, dfbs=[external_stream])

    return external_descriptor_reset_kernel


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "to_device",
    [to_dram, to_l1],
    ids=["dram", "l1"],
)
def test_external_descriptor_survives_specialization_and_synchronized_reset(
    device,
    dtype,
    to_device,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reset support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_external_descriptor_reset_kernel(data_format)
    final_mlir_path = tmp_path / "external_descriptor_reset.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))

    element_indices = torch.arange(TILE * 16 * TILE, dtype=torch.float32).reshape(
        TILE, 16 * TILE
    )
    input_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    operation(
        input_tensor,
        output_tensor,
        options="--ttl-reuse-user-dfbs --ttl-specialize-cores",
    )

    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    final_mlir = final_mlir_path.read_text()
    assert final_mlir.count("dfb_index =") == 1
    assert final_mlir.count("ttl.core_coord") == 6
    assert final_mlir.count("ttl.used_dfb_indices = array<i32: 0>") == 3
    assert final_mlir.count("ttl.used_dfb_indices = array<i32>") == 3
