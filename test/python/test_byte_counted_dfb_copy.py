# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Device coverage for byte-counted local and PipeNet DFB copies.

The transfer stages fourteen 1x32 pages into a full 32x32 page, sends only the
valid byte prefix to another core, and restores the compact representation.
This is the representation change required by TreeReduce: compute can retain a
full-tile accumulator while each communication stage transfers only valid rows.
"""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

VALID_ROWS = 14
TILE_WIDTH = 32
COMPACT_TILE = (1, TILE_WIDTH)
FULL_TILE = (32, TILE_WIDTH)


def _make_byte_counted_round_trip(data_format, byte_count):
    @ttl.operation(grid=(2, 1))
    def byte_counted_round_trip(inp, out):
        pipe_net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
        compact_input_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(VALID_ROWS, 1), block_count=1
        )
        send_dfb = ttl.make_dfb(
            data_format, shape=(1, 1), block_count=1, tile=FULL_TILE
        )
        receive_dfb = ttl.make_dfb(
            data_format, shape=(1, 1), block_count=1, tile=FULL_TILE
        )
        compact_output_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(VALID_ROWS, 1), block_count=1
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm():
            def send(pipe):
                with compact_input_dfb.reserve() as compact_input_block:
                    ttl.copy(inp[0:VALID_ROWS, 0:1], compact_input_block).wait()

                with compact_input_dfb.wait() as compact_input_block:
                    with send_dfb.reserve() as send_block:
                        ttl.copy(
                            compact_input_block,
                            send_block,
                            byte_count=byte_count,
                        ).wait()

                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe, byte_count=byte_count).wait()

            pipe_net.if_src(send)

            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block, byte_count=byte_count).wait()

                with receive_dfb.wait() as receive_block:
                    with compact_output_dfb.reserve() as compact_output_block:
                        ttl.copy(
                            receive_block,
                            compact_output_block,
                            byte_count=byte_count,
                        ).wait()

                with compact_output_dfb.wait() as compact_output_block:
                    ttl.copy(compact_output_block, out[0:VALID_ROWS, 0:1]).wait()

            pipe_net.if_dst(receive)

        @ttl.datamovement()
        def dm_brisc():
            pass

    return byte_counted_round_trip


BYTE_COUNTED_ROUND_TRIP_CASES = [
    pytest.param(
        _make_byte_counted_round_trip("bf16", VALID_ROWS * TILE_WIDTH * 2),
        torch.bfloat16,
        2,
        VALID_ROWS * TILE_WIDTH * 2,
        5e-2,
        1.0,
        id="bf16-full-rows",
    ),
    pytest.param(
        _make_byte_counted_round_trip("bf16", 3),
        torch.bfloat16,
        2,
        3,
        5e-2,
        1.0,
        id="bf16-partial-element",
    ),
    pytest.param(
        _make_byte_counted_round_trip("float32", VALID_ROWS * TILE_WIDTH * 4),
        torch.float32,
        4,
        VALID_ROWS * TILE_WIDTH * 4,
        1e-5,
        1e-5,
        id="fp32-full-rows",
    ),
    pytest.param(
        _make_byte_counted_round_trip("float32", 5),
        torch.float32,
        4,
        5,
        1e-5,
        1e-5,
        id="fp32-partial-element",
    ),
]


@pytest.mark.parametrize(
    ("operation", "dtype", "bytes_per_element", "byte_count", "rtol", "atol"),
    BYTE_COUNTED_ROUND_TRIP_CASES,
)
def test_byte_counted_dfb_round_trip(
    device, operation, dtype, bytes_per_element, byte_count, rtol, atol
):
    element_indices = torch.arange(VALID_ROWS * TILE_WIDTH, dtype=torch.int64).reshape(
        VALID_ROWS, TILE_WIDTH
    )
    input_torch = (element_indices.remainder(97) - 48).to(dtype)
    output_torch = torch.zeros_like(input_torch)

    inp = to_dram(input_torch, device, tile=COMPACT_TILE)
    out = to_dram(output_torch, device, tile=COMPACT_TILE)

    operation(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out).reshape(VALID_ROWS, TILE_WIDTH).to(dtype)
    full_byte_count = input_torch.numel() * bytes_per_element
    if byte_count == full_byte_count:
        assert_allclose(result.float(), input_torch.float(), rtol=rtol, atol=atol)
        assert torch.equal(result, input_torch)
    else:
        complete_element_count = byte_count // bytes_per_element
        assert_allclose(
            result.reshape(-1)[:complete_element_count].float(),
            input_torch.reshape(-1)[:complete_element_count].float(),
            rtol=rtol,
            atol=atol,
        )
        result_bytes = result.view(dtype=torch.uint8).reshape(-1)
        input_bytes = input_torch.view(dtype=torch.uint8).reshape(-1)
        assert torch.equal(result_bytes[:byte_count], input_bytes[:byte_count])

    ttnn.deallocate(inp)
    ttnn.deallocate(out)
