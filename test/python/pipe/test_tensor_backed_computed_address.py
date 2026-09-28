# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor-backed PipeNet addresses preserve shard offsets and untouched tiles."""

import pytest
import torch
import ttl

from ttlang_test_utils import to_dram, to_l1, to_l1_sharded
from utils.correctness import assert_allclose

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device

TILE = 32


def _make_tensor_backed_receiver(byte_offset):
    @ttl.operation(grid=(1, 1))
    def tensor_backed_receiver(inp, out):
        pipe = ttl.Pipe(src=(0, 0), dst=(0, 0))
        pipe_net = ttl.PipeNet([pipe])
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
        receive_dfb = ttl.make_tensor_backed_dfb(
            out, shape=(1, 1), byte_offset=byte_offset
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def transfer():
            if pipe_net.is_active():
                pass

            with receive_dfb.reserve() as receive_block:
                receive_request = ttl.copy(pipe, receive_block)
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()
                receive_request.wait()
            with receive_dfb.wait():
                pass

        @ttl.datamovement()
        def unused_transfer():
            pass

    return tensor_backed_receiver


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.float32, ttnn.bfloat4_b, ttnn.bfloat8_b],
    ids=["bf16", "fp32", "bfp4", "bfp8"],
)
@pytest.mark.parametrize("shard_layout", ["height", "width", "block"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("tile_offset", [0, 1], ids=["base", "offset"])
@pytest.mark.parametrize("computed", [False, True], ids=["published", "computed"])
def test_tensor_backed_receiver_address(
    device, dtype, shard_layout, to_device, tile_offset, computed, monkeypatch, tmp_path
):
    # Tensor-backed DFBs require tiled sharded L1 storage and these four dtypes.
    host_input = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    host_input = host_input.remainder(17) - 8
    input_tensor = ttnn.typecast(to_device(host_input, device), dtype)
    output_tensor = ttnn.typecast(
        to_l1_sharded(torch.zeros((TILE, 3 * TILE)), device, layout=shard_layout),
        dtype,
    )
    page_size = int(output_tensor.get_tile().get_tile_size(dtype))
    operation = _make_tensor_backed_receiver(tile_offset * page_size)
    final_mlir_file = tmp_path / "tensor_backed_receiver.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_file))

    for _invocation_index in range(2):
        operation(
            input_tensor,
            output_tensor,
            options=(
                "--ttl-pipe-computed-addresses"
                if computed
                else "--no-ttl-pipe-computed-addresses"
            ),
        )

    expected = torch.zeros((TILE, 3 * TILE))
    expected[:, tile_offset * TILE : (tile_offset + 1) * TILE] = ttnn.to_torch(
        input_tensor
    ).float()
    assert_allclose(ttnn.to_torch(output_tensor).float(), expected, rtol=0, atol=0)
    assert (
        "ttl.pipe_computed_address_dfb_indices" in final_mlir_file.read_text()
    ) == computed
