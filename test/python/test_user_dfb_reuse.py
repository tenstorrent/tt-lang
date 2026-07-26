# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for physical reuse of user-declared DFBs."""

import re

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

TILE = 32


@ttl.operation(grid=(1, 1))
def _user_dfb_reuse_kernel(first, second, out):
    first_dfb = ttl.make_dataflow_buffer_like(first, shape=(1, 1), block_count=2)
    acknowledgment_dfb = ttl.make_dataflow_buffer_like(
        first, shape=(1, 1), block_count=2
    )
    second_dfb = ttl.make_dataflow_buffer_like(
        second, shape=(1, 1), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with first_dfb.wait():
            pass

        with acknowledgment_dfb.reserve() as acknowledgment:
            acknowledgment.store(
                ttl.block.fill(
                    0,
                    shape=acknowledgment.shape,
                    dtype=acknowledgment.dtype,
                )
            )

        with second_dfb.wait() as second_block, out_dfb.reserve() as out_block:
            out_block.store(second_block)

    @ttl.datamovement()
    def dm_read():
        with first_dfb.reserve() as first_block:
            ttl.copy(first[0, 0], first_block).wait()

        with acknowledgment_dfb.wait():
            pass

        with second_dfb.reserve() as second_block:
            ttl.copy(second[0, 0], second_block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0, 0]).wait()


def _physical_allocation_indices(final_mlir):
    allocations = re.search(
        r"ttl\.dfb_allocations = \[(.*?)\]",
        final_mlir,
        re.DOTALL,
    )
    assert allocations is not None
    return [
        int(index)
        for index in re.findall(r"dfb_index = (\d+) : i32", allocations.group(1))
    ]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_user_dfb_reuse_runtime(
    device, dtype, memory_config, to_device, tmp_path, monkeypatch
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(
        TILE, TILE
    )
    first_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    second_host = (((17 * element_indices).remainder(509) - 254) / 128).to(
        dtype
    )
    out_host = torch.zeros((TILE, TILE), dtype=dtype)

    first = to_device(first_host, device)
    second = to_device(second_host, device)
    out = to_device(out_host, device)

    final_mlir_path = tmp_path / f"user_dfb_reuse_{memory_config}.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    _user_dfb_reuse_kernel(first, second, out)

    actual = ttnn.to_torch(out).float()
    expected = second_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    final_mlir = final_mlir_path.read_text()
    assert _physical_allocation_indices(final_mlir) == [0, 1, 2]
