# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for physical reuse of user-declared DFBs."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

TILE = 32
OVER_CAPACITY_COMPOSITION_LEVELS = 5
OVER_CAPACITY_LOGICAL_DFBS = (1 << OVER_CAPACITY_COMPOSITION_LEVELS) + 1


def _make_exp_via_scratch_atom(data_format, shape=(1, 1)):
    @ttl.operation()
    def exp_via_scratch(source: ttl.DFB, destination: ttl.DFB):
        scratch_dfb = ttl.make_dfb(data_format, shape=shape, block_count=2)
        scratch_output = scratch_dfb.reserve()
        scratch_output.store(ttl.exp(source.wait()))
        destination_output = destination.reserve()
        destination_output.store(scratch_dfb.wait())

    return exp_via_scratch


@ttl.operation()
def _increment_in_place(source: ttl.DFB, result: ttl.DFB):
    source_block = source.wait()
    result_block = result.reserve()
    result_block.store(
        ttl.add(
            source_block,
            ttl.block.fill(
                1,
                shape=source_block.shape,
                dtype=source_block.dtype,
            ),
        )
    )


def _make_in_place_atom_kernel(data_format):
    @ttl.operation(grid=(1, 1))
    def in_place_atom_kernel(input_tensor, output_tensor):
        input_dfb, state_dfb, output_dfb = [
            ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
            for buffer_index in range(3)
        ]

        ttl.copy(input_tensor[0, 0], input_dfb.reserve()).wait()
        input_block = input_dfb.wait()
        state_block = state_dfb.reserve()
        state_block.store(input_block)
        _increment_in_place(state_dfb, state_dfb)
        state_block = state_dfb.wait()
        output_block = output_dfb.reserve()
        output_block.store(state_block)
        ttl.copy(output_dfb.wait(), output_tensor[0, 0]).wait()

    return in_place_atom_kernel


def _make_repeated_dfb_atom_kernel(data_format):
    exp_via_scratch = _make_exp_via_scratch_atom(data_format)

    @ttl.operation(grid=(1, 1))
    def repeated_dfb_atom_kernel(input_tensor, output_tensor):
        stage_dfbs = [
            ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
            for stage_index in range(3)
        ]

        ttl.copy(input_tensor[0, 0], stage_dfbs[0].reserve()).wait()
        exp_via_scratch(stage_dfbs[0], stage_dfbs[1])
        exp_via_scratch(stage_dfbs[1], stage_dfbs[2])
        ttl.copy(stage_dfbs[2].wait(), output_tensor[0, 0]).wait()

    return repeated_dfb_atom_kernel


def _make_nested_copy_atom(data_format, level_count):
    @ttl.operation()
    def copy_stage(source: ttl.DFB, destination: ttl.DFB):
        destination_block = destination.reserve()
        destination_block.store(source.wait())

    nested_copy = copy_stage
    for composition_level in range(level_count):
        inner_copy = nested_copy

        @ttl.operation()
        def doubled_copy(source: ttl.DFB, destination: ttl.DFB):
            intermediate_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
            inner_copy(source, intermediate_dfb)
            inner_copy(intermediate_dfb, destination)

        nested_copy = doubled_copy

    return nested_copy


def _make_over_capacity_atom_kernel(data_format):
    nested_copy = _make_nested_copy_atom(data_format, OVER_CAPACITY_COMPOSITION_LEVELS)

    @ttl.operation(grid=(1, 1))
    def over_capacity_atom_kernel(input_tensor, output_tensor):
        first_stage_dfb, last_stage_dfb = [
            ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
            for endpoint_index in range(2)
        ]

        ttl.copy(input_tensor[0, 0], first_stage_dfb.reserve()).wait()
        nested_copy(first_stage_dfb, last_stage_dfb)
        ttl.copy(last_stage_dfb.wait(), output_tensor[0, 0]).wait()

    return over_capacity_atom_kernel


_repeated_bf16_atom_kernel = _make_repeated_dfb_atom_kernel("bf16")
_repeated_f32_atom_kernel = _make_repeated_dfb_atom_kernel("float32")
_in_place_bf16_atom_kernel = _make_in_place_atom_kernel("bf16")
_in_place_f32_atom_kernel = _make_in_place_atom_kernel("float32")
_over_capacity_bf16_atom_kernel = _make_over_capacity_atom_kernel("bf16")
_over_capacity_f32_atom_kernel = _make_over_capacity_atom_kernel("float32")

assert OVER_CAPACITY_LOGICAL_DFBS > 32


_exp_scalar_via_scratch = _make_exp_via_scratch_atom("bf16", shape=(1, 1))
_exp_two_tiles_via_scratch = _make_exp_via_scratch_atom("bf16", shape=(1, 2))


@ttl.operation(grid=(1, 1))
def _mixed_capacity_atom_kernel(
    scalar_input, scalar_output, two_tile_input, two_tile_output
):
    scalar_stages = [
        ttl.make_dfb("bf16", shape=(1, 1), block_count=2) for stage_index in range(3)
    ]
    two_tile_stages = [
        ttl.make_dfb("bf16", shape=(1, 2), block_count=2) for stage_index in range(3)
    ]

    ttl.copy(scalar_input[0, 0], scalar_stages[0].reserve()).wait()
    ttl.copy(two_tile_input[0:1, 0:2], two_tile_stages[0].reserve()).wait()
    _exp_scalar_via_scratch(scalar_stages[0], scalar_stages[1])
    _exp_two_tiles_via_scratch(two_tile_stages[0], two_tile_stages[1])
    _exp_scalar_via_scratch(scalar_stages[1], scalar_stages[2])
    _exp_two_tiles_via_scratch(two_tile_stages[1], two_tile_stages[2])
    ttl.copy(scalar_stages[2].wait(), scalar_output[0, 0]).wait()
    ttl.copy(two_tile_stages[2].wait(), two_tile_output[0:1, 0:2]).wait()


@ttl.operation(grid=(1, 1))
def _user_dfb_reuse_kernel(first, second, out):
    first_dfb = ttl.make_dataflow_buffer_like(first, shape=(1, 1), block_count=2)
    acknowledgment_dfb = ttl.make_dataflow_buffer_like(
        first, shape=(1, 1), block_count=2
    )
    second_dfb = ttl.make_dataflow_buffer_like(second, shape=(1, 1), block_count=2)
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_user_dfb_reuse_runtime(device, dtype, memory_config, to_device):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    first_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    second_host = (((17 * element_indices).remainder(509) - 254) / 128).to(dtype)
    out_host = torch.zeros((TILE, TILE), dtype=dtype)

    first = to_device(first_host, device)
    second = to_device(second_host, device)
    out = to_device(out_host, device)

    _user_dfb_reuse_kernel(first, second, out)

    actual = ttnn.to_torch(out).float()
    expected = second_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_repeated_bf16_atom_kernel, torch.bfloat16),
        (_repeated_f32_atom_kernel, torch.float32),
    ],
    ids=["bf16", "f32"],
)
def test_repeated_dfb_declaring_atom_runtime(
    device, memory_config, to_device, operation, dtype
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    input_host = ((element_indices.remainder(97) - 48) / 128).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros((TILE, TILE), dtype=dtype), device)

    operation(input_tensor, output_tensor)

    intermediate = torch.exp(input_host.float()).to(dtype)
    expected = torch.exp(intermediate.float()).to(dtype).float()
    actual = ttnn.to_torch(output_tensor).float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_in_place_bf16_atom_kernel, torch.bfloat16),
        (_in_place_f32_atom_kernel, torch.float32),
    ],
    ids=["bf16", "f32"],
)
def test_atom_accepts_same_dfb_as_source_and_result(
    device, memory_config, to_device, operation, dtype
):
    input_host = (
        torch.arange(TILE * TILE, dtype=torch.float32).remainder(7).reshape(TILE, TILE)
        - 3
    ).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    operation(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host.float() + 1
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_over_capacity_bf16_atom_kernel, torch.bfloat16),
        (_over_capacity_f32_atom_kernel, torch.float32),
    ],
    ids=["bf16", "f32"],
)
def test_over_capacity_atom_composition_requires_dfb_reuse(
    device, memory_config, to_device, operation, dtype
):
    input_host = torch.linspace(-1.0, 1.0, TILE * TILE, dtype=torch.float32).reshape(
        TILE, TILE
    )
    input_host = input_host.to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    operation(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_mixed_capacity_dfb_atoms_runtime(device, memory_config, to_device):
    scalar_input_host = (
        torch.linspace(-0.375, 0.375, TILE * TILE, dtype=torch.float32)
        .reshape(TILE, TILE)
        .to(torch.bfloat16)
    )
    two_tile_input_host = (
        torch.linspace(-0.375, 0.375, TILE * TILE * 2, dtype=torch.float32)
        .reshape(TILE, 2 * TILE)
        .to(torch.bfloat16)
    )

    scalar_input = to_device(scalar_input_host, device)
    scalar_output = to_device(torch.zeros_like(scalar_input_host), device)
    two_tile_input = to_device(two_tile_input_host, device)
    two_tile_output = to_device(torch.zeros_like(two_tile_input_host), device)

    _mixed_capacity_atom_kernel(
        scalar_input, scalar_output, two_tile_input, two_tile_output
    )

    for input_host, output_tensor in (
        (scalar_input_host, scalar_output),
        (two_tile_input_host, two_tile_output),
    ):
        intermediate = torch.exp(input_host.float()).to(torch.bfloat16)
        expected = torch.exp(intermediate.float()).to(torch.bfloat16).float()
        actual = ttnn.to_torch(output_tensor).float()
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
