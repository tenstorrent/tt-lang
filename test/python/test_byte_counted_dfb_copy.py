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
    def byte_counted_round_trip(
        inp,
        send_seed,
        receive_seed,
        compact_seed,
        local_out,
        pipe_out,
        compact_out,
    ):
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
            compact_out, shape=(VALID_ROWS, 1), block_count=1
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(send_seed[0:1, 0:1], send_block).wait()
                with send_dfb.wait():
                    pass

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
                    ttl.copy(send_block, local_out[0:1, 0:1]).wait()
                    ttl.copy(send_block, pipe, byte_count=byte_count).wait()

            pipe_net.if_src(send)

            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(receive_seed[0:1, 0:1], receive_block).wait()
                with receive_dfb.wait():
                    pass

                with compact_output_dfb.reserve() as compact_output_block:
                    ttl.copy(
                        compact_seed[0:VALID_ROWS, 0:1], compact_output_block
                    ).wait()
                with compact_output_dfb.wait():
                    pass

                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block, byte_count=byte_count).wait()

                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, pipe_out[0:1, 0:1]).wait()
                    with compact_output_dfb.reserve() as compact_output_block:
                        ttl.copy(
                            receive_block,
                            compact_output_block,
                            byte_count=byte_count,
                        ).wait()

                with compact_output_dfb.wait() as compact_output_block:
                    ttl.copy(
                        compact_output_block, compact_out[0:VALID_ROWS, 0:1]
                    ).wait()

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
    seed_magnitude = 1024.0001220703125
    send_seed_torch = torch.full(FULL_TILE, -seed_magnitude, dtype=dtype)
    receive_seed_torch = torch.full(FULL_TILE, seed_magnitude, dtype=dtype)
    compact_seed_torch = torch.full_like(input_torch, -2 * seed_magnitude)

    inp = to_dram(input_torch, device, tile=COMPACT_TILE)
    send_seed = to_dram(send_seed_torch, device, tile=FULL_TILE)
    receive_seed = to_dram(receive_seed_torch, device, tile=FULL_TILE)
    compact_seed = to_dram(compact_seed_torch, device, tile=COMPACT_TILE)
    local_out = to_dram(torch.zeros_like(send_seed_torch), device, tile=FULL_TILE)
    pipe_out = to_dram(torch.zeros_like(receive_seed_torch), device, tile=FULL_TILE)
    compact_out = to_dram(
        torch.zeros_like(compact_seed_torch), device, tile=COMPACT_TILE
    )

    operation(
        inp,
        send_seed,
        receive_seed,
        compact_seed,
        local_out,
        pipe_out,
        compact_out,
    )
    ttnn.synchronize_device(device)

    def replace_byte_prefix(seed):
        expected = seed.clone()
        expected_bytes = expected.view(dtype=torch.uint8).reshape(-1)
        input_bytes = input_torch.view(dtype=torch.uint8).reshape(-1)
        expected_bytes[:byte_count].copy_(input_bytes[:byte_count])
        return expected

    def assert_same_element_bits(actual, expected):
        bit_dtype = torch.int16 if bytes_per_element == 2 else torch.int32
        actual_bits = actual.contiguous().view(dtype=bit_dtype).reshape(-1)
        expected_bits = expected.contiguous().view(dtype=bit_dtype).reshape(-1)
        assert torch.equal(
            torch.sort(actual_bits).values, torch.sort(expected_bits).values
        )

    local_result = ttnn.to_torch(local_out).reshape(FULL_TILE).to(dtype)
    pipe_result = ttnn.to_torch(pipe_out).reshape(FULL_TILE).to(dtype)
    local_expected = replace_byte_prefix(send_seed_torch)
    pipe_expected = replace_byte_prefix(receive_seed_torch)
    assert_same_element_bits(local_result, local_expected)
    assert_same_element_bits(pipe_result, pipe_expected)

    compact_result = (
        ttnn.to_torch(compact_out).reshape(VALID_ROWS, TILE_WIDTH).to(dtype)
    )
    compact_expected = replace_byte_prefix(compact_seed_torch)
    assert_allclose(
        compact_result.float(),
        compact_expected.float(),
        rtol=rtol,
        atol=atol,
    )
    result_bytes = compact_result.view(dtype=torch.uint8).reshape(-1)
    expected_bytes = compact_expected.view(dtype=torch.uint8).reshape(-1)
    assert torch.equal(result_bytes, expected_bytes)

    for full_result, full_expected in (
        (local_result, local_expected),
        (pipe_result, pipe_expected),
    ):
        assert_allclose(
            torch.sort(full_result.float().reshape(-1)).values,
            torch.sort(full_expected.float().reshape(-1)).values,
            rtol=rtol,
            atol=atol,
        )

    for tensor in (
        inp,
        send_seed,
        receive_seed,
        compact_seed,
        local_out,
        pipe_out,
        compact_out,
    ):
        ttnn.deallocate(tensor)
