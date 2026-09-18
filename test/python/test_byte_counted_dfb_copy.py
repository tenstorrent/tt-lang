# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Device coverage for byte-counted local and PipeNet DFB copies.

BF16 and FP32 transfer fourteen 1x32 pages through full 32x32 pages. Integer
formats transfer the same initial byte range between full pages because tile
shape changes may alter their physical padding and face order. Packed formats
transfer a complete encoded page because their shared exponents have no host
byte representation. Every case traverses both a local DFB copy and a two-core
PipeNet transfer.
"""

import pytest
import torch

import ttl
from ttl.dtype_utils import tile_bytes_from_dtype

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

VALID_ROWS = 14
TILE_WIDTH = 32
COMPACT_TILE = (1, TILE_WIDTH)
FULL_TILE = (32, TILE_WIDTH)
BFP4_FULL_TILE_BYTES = tile_bytes_from_dtype(ttnn.bfloat4_b, FULL_TILE)
BFP8_FULL_TILE_BYTES = tile_bytes_from_dtype(ttnn.bfloat8_b, FULL_TILE)


def _make_byte_counted_round_trip(data_format, byte_count, compact_tile_rows):
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
            inp, shape=(compact_tile_rows, 1), block_count=1
        )
        send_dfb = ttl.make_dfb(
            data_format, shape=(1, 1), block_count=1, tile=FULL_TILE
        )
        receive_dfb = ttl.make_dfb(
            data_format, shape=(1, 1), block_count=1, tile=FULL_TILE
        )
        compact_output_dfb = ttl.make_dataflow_buffer_like(
            compact_out, shape=(compact_tile_rows, 1), block_count=1
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
                    ttl.copy(inp[0:compact_tile_rows, 0:1], compact_input_block).wait()

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
                        compact_seed[0:compact_tile_rows, 0:1],
                        compact_output_block,
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
                        compact_output_block,
                        compact_out[0:compact_tile_rows, 0:1],
                    ).wait()

            pipe_net.if_dst(receive)

        @ttl.datamovement()
        def dm_brisc():
            pass

    return byte_counted_round_trip


BYTE_COUNTED_ROUND_TRIP_CASES = [
    pytest.param(
        _make_byte_counted_round_trip("bf16", VALID_ROWS * TILE_WIDTH * 2, VALID_ROWS),
        ttnn.bfloat16,
        torch.bfloat16,
        2,
        VALID_ROWS * TILE_WIDTH * 2,
        COMPACT_TILE,
        VALID_ROWS,
        5e-2,
        1.0,
        id="bf16-full-rows",
    ),
    pytest.param(
        _make_byte_counted_round_trip("bf16", 3, VALID_ROWS),
        ttnn.bfloat16,
        torch.bfloat16,
        2,
        3,
        COMPACT_TILE,
        VALID_ROWS,
        5e-2,
        1.0,
        id="bf16-partial-element",
    ),
    pytest.param(
        _make_byte_counted_round_trip(
            "float32", VALID_ROWS * TILE_WIDTH * 4, VALID_ROWS
        ),
        ttnn.float32,
        torch.float32,
        4,
        VALID_ROWS * TILE_WIDTH * 4,
        COMPACT_TILE,
        VALID_ROWS,
        1e-5,
        1e-5,
        id="fp32-full-rows",
    ),
    pytest.param(
        _make_byte_counted_round_trip("float32", 5, VALID_ROWS),
        ttnn.float32,
        torch.float32,
        4,
        5,
        COMPACT_TILE,
        VALID_ROWS,
        1e-5,
        1e-5,
        id="fp32-partial-element",
    ),
    pytest.param(
        _make_byte_counted_round_trip("bfloat4_b", BFP4_FULL_TILE_BYTES, 1),
        ttnn.bfloat4_b,
        torch.bfloat16,
        None,
        BFP4_FULL_TILE_BYTES,
        FULL_TILE,
        1,
        0.0,
        0.0,
        id="bfp4-full-page",
    ),
    pytest.param(
        _make_byte_counted_round_trip("bfloat8_b", BFP8_FULL_TILE_BYTES, 1),
        ttnn.bfloat8_b,
        torch.bfloat16,
        None,
        BFP8_FULL_TILE_BYTES,
        FULL_TILE,
        1,
        0.0,
        0.0,
        id="bfp8-full-page",
    ),
    pytest.param(
        _make_byte_counted_round_trip("int32", VALID_ROWS * TILE_WIDTH * 4, VALID_ROWS),
        ttnn.int32,
        torch.int32,
        4,
        VALID_ROWS * TILE_WIDTH * 4,
        FULL_TILE,
        1,
        0.0,
        0.0,
        id="i32-full-rows",
    ),
    pytest.param(
        _make_byte_counted_round_trip(
            "uint32", VALID_ROWS * TILE_WIDTH * 4, VALID_ROWS
        ),
        ttnn.uint32,
        torch.uint32,
        4,
        VALID_ROWS * TILE_WIDTH * 4,
        FULL_TILE,
        1,
        0.0,
        0.0,
        id="u32-full-rows",
    ),
    pytest.param(
        _make_byte_counted_round_trip(
            "uint16", VALID_ROWS * TILE_WIDTH * 2, VALID_ROWS
        ),
        ttnn.uint16,
        torch.uint16,
        2,
        VALID_ROWS * TILE_WIDTH * 2,
        FULL_TILE,
        1,
        0.0,
        0.0,
        id="u16-full-rows",
    ),
    pytest.param(
        _make_byte_counted_round_trip("uint8", VALID_ROWS * TILE_WIDTH, VALID_ROWS),
        ttnn.uint8,
        torch.uint8,
        1,
        VALID_ROWS * TILE_WIDTH,
        FULL_TILE,
        1,
        0.0,
        0.0,
        id="u8-full-rows",
    ),
]


@pytest.mark.parametrize(
    (
        "operation",
        "ttnn_dtype",
        "torch_dtype",
        "bytes_per_element",
        "byte_count",
        "compact_tile",
        "compact_tile_rows",
        "rtol",
        "atol",
    ),
    BYTE_COUNTED_ROUND_TRIP_CASES,
)
@pytest.mark.parametrize(
    "memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram", "l1"],
)
def test_byte_counted_dfb_round_trip(
    device,
    operation,
    ttnn_dtype,
    torch_dtype,
    bytes_per_element,
    byte_count,
    compact_tile,
    compact_tile_rows,
    rtol,
    atol,
    memory_config,
):
    compact_shape = (
        compact_tile_rows * compact_tile[0],
        compact_tile[1],
    )
    if torch_dtype.is_floating_point:
        element_indices = torch.arange(
            compact_shape[0] * compact_shape[1], dtype=torch.int64
        ).reshape(compact_shape)
        input_torch = ((element_indices.remainder(97) - 48).float() / 7).to(torch_dtype)
        send_seed_torch = torch.full(FULL_TILE, -16.0, dtype=torch_dtype)
        receive_seed_torch = torch.full(FULL_TILE, 16.0, dtype=torch_dtype)
        compact_seed_torch = torch.full(compact_shape, -24.0, dtype=torch_dtype)
    else:
        # A constant payload verifies byte counts without assuming that compact
        # and full tiles use the same format-specific face order.
        input_torch = torch.full(compact_shape, 17, dtype=torch_dtype)
        send_seed_torch = torch.full(FULL_TILE, 101, dtype=torch_dtype)
        receive_seed_torch = torch.full(FULL_TILE, 103, dtype=torch_dtype)
        compact_seed_torch = torch.full(compact_shape, 107, dtype=torch_dtype)

    def to_device(host_tensor, tile):
        return ttnn.from_torch(
            host_tensor,
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile(tile),
            device=device,
            memory_config=memory_config,
        )

    inp = to_device(input_torch, compact_tile)
    send_seed = to_device(send_seed_torch, FULL_TILE)
    receive_seed = to_device(receive_seed_torch, FULL_TILE)
    compact_seed = to_device(compact_seed_torch, compact_tile)
    local_out = to_device(torch.zeros_like(send_seed_torch), FULL_TILE)
    pipe_out = to_device(torch.zeros_like(receive_seed_torch), FULL_TILE)
    compact_out = to_device(torch.zeros_like(compact_seed_torch), compact_tile)

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

    input_expected = ttnn.to_torch(inp).reshape(compact_shape).to(torch_dtype)
    send_seed_expected = ttnn.to_torch(send_seed).reshape(FULL_TILE).to(torch_dtype)
    receive_seed_expected = (
        ttnn.to_torch(receive_seed).reshape(FULL_TILE).to(torch_dtype)
    )
    compact_seed_expected = (
        ttnn.to_torch(compact_seed).reshape(compact_shape).to(torch_dtype)
    )

    def replace_initial_bytes(seed):
        expected = seed.clone()
        expected_bytes = expected.view(dtype=torch.uint8).reshape(-1)
        input_bytes = input_expected.view(dtype=torch.uint8).reshape(-1)
        expected_bytes[:byte_count].copy_(input_bytes[:byte_count])
        return expected

    def assert_same_element_bit_multiset(actual, expected):
        """Ignore format-specific tile face order while checking exact values."""
        bit_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[bytes_per_element]
        actual_bits = actual.contiguous().view(dtype=bit_dtype).reshape(-1)
        expected_bits = expected.contiguous().view(dtype=bit_dtype).reshape(-1)
        assert torch.equal(
            torch.sort(actual_bits).values, torch.sort(expected_bits).values
        )

    local_result = ttnn.to_torch(local_out).reshape(FULL_TILE).to(torch_dtype)
    pipe_result = ttnn.to_torch(pipe_out).reshape(FULL_TILE).to(torch_dtype)
    if bytes_per_element is None:
        local_expected = input_expected
        pipe_expected = input_expected
    else:
        local_expected = replace_initial_bytes(send_seed_expected)
        pipe_expected = replace_initial_bytes(receive_seed_expected)
        assert_same_element_bit_multiset(local_result, local_expected)
        assert_same_element_bit_multiset(pipe_result, pipe_expected)

    compact_result = ttnn.to_torch(compact_out).reshape(compact_shape).to(torch_dtype)
    compact_expected = (
        input_expected
        if bytes_per_element is None
        else replace_initial_bytes(compact_seed_expected)
    )
    if bytes_per_element is not None and compact_tile == FULL_TILE:
        assert_same_element_bit_multiset(compact_result, compact_expected)
    else:
        assert_allclose(
            compact_result.float(),
            compact_expected.float(),
            rtol=rtol,
            atol=atol,
        )
        assert torch.equal(compact_result, compact_expected)

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
