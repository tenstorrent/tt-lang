# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for physical reuse of user-declared DFBs."""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import ttl_api  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

# These cases use native TTL data movement and compute operations, which require
# tiled tensor arguments.
TILE = 32
CAPACITY_TEST_COMPOSITION_LEVELS = 5
CAPACITY_TEST_LOGICAL_DFBS = (1 << CAPACITY_TEST_COMPOSITION_LEVELS) + 1
# Two approximate SFPU exponential evaluations need operation-level error
# bounds; f32 data movement and addition tests retain 1e-5 relative tolerance.
F32_REPEATED_EXP_RTOL = 2e-3
F32_REPEATED_EXP_ATOL = 5e-4
SCALAR_RESULT_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "scalar_result_op.hpp"
)


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
        input_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        state_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        output_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

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
        first_stage_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_stage_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        third_stage_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        ttl.copy(input_tensor[0, 0], first_stage_dfb.reserve()).wait()
        exp_via_scratch(first_stage_dfb, second_stage_dfb)
        exp_via_scratch(second_stage_dfb, third_stage_dfb)
        ttl.copy(third_stage_dfb.wait(), output_tensor[0, 0]).wait()

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


def _make_capacity_test_atom_kernel(data_format):
    nested_copy = _make_nested_copy_atom(data_format, CAPACITY_TEST_COMPOSITION_LEVELS)

    @ttl.operation(grid=(1, 1))
    def capacity_test_atom_kernel(input_tensor, output_tensor):
        first_stage_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        last_stage_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        ttl.copy(input_tensor[0, 0], first_stage_dfb.reserve()).wait()
        nested_copy(first_stage_dfb, last_stage_dfb)
        ttl.copy(last_stage_dfb.wait(), output_tensor[0, 0]).wait()

    return capacity_test_atom_kernel


def _make_conditional_lifecycle_kernel(data_format):
    @ttl.operation(grid=(1, 1))
    def conditional_lifecycle_kernel(input_tensor, output_tensor):
        input_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        first_scratch_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_scratch_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        output_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            runtime_predicate = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_result_i32",
                result_type="i32",
            )
            with input_dfb.wait() as input_block:
                if runtime_predicate:
                    with first_scratch_dfb.reserve() as first_scratch_block:
                        first_scratch_block.store(input_block)
                    with first_scratch_dfb.wait():
                        pass

                if runtime_predicate:
                    with second_scratch_dfb.reserve() as second_scratch_block:
                        second_scratch_block.store(input_block)
                    with second_scratch_dfb.wait() as second_scratch_block:
                        with output_dfb.reserve() as output_block:
                            output_block.store(second_scratch_block)

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[0, 0], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output_tensor[0, 0]).wait()

    return conditional_lifecycle_kernel


def _make_exact_execution_domain_kernel(data_format):
    @ttl.operation(grid=(1, 1))
    def exact_execution_domain_kernel(input_tensor, output_tensor):
        input_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        first_scratch_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_scratch_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        output_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            node_x, _ = ttl.node(dims=2)
            first_producer_predicate = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_result_i32",
                result_type="i32",
            )
            first_consumer_predicate = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_result_i32",
                result_type="i32",
            )
            second_producer_predicate = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_result_i32",
                result_type="i32",
            )
            second_consumer_predicate = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_result_i32",
                result_type="i32",
            )

            with input_dfb.wait() as input_block:
                if first_producer_predicate or node_x == 0:
                    with first_scratch_dfb.reserve() as first_scratch_block:
                        first_scratch_block.store(input_block)
                if first_consumer_predicate or node_x == 0:
                    with first_scratch_dfb.wait():
                        pass

                if second_producer_predicate or node_x == 0:
                    with second_scratch_dfb.reserve() as second_scratch_block:
                        second_scratch_block.store(input_block)
                if second_consumer_predicate or node_x == 0:
                    with second_scratch_dfb.wait() as second_scratch_block:
                        with output_dfb.reserve() as output_block:
                            output_block.store(second_scratch_block)

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[0, 0], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output_tensor[0, 0]).wait()

    return exact_execution_domain_kernel


_repeated_bf16_atom_kernel = _make_repeated_dfb_atom_kernel("bf16")
_repeated_f32_atom_kernel = _make_repeated_dfb_atom_kernel("float32")
_in_place_bf16_atom_kernel = _make_in_place_atom_kernel("bf16")
_in_place_f32_atom_kernel = _make_in_place_atom_kernel("float32")
_capacity_test_bf16_atom_kernel = _make_capacity_test_atom_kernel("bf16")
_capacity_test_f32_atom_kernel = _make_capacity_test_atom_kernel("float32")
_conditional_bf16_lifecycle_kernel = _make_conditional_lifecycle_kernel("bf16")
_conditional_f32_lifecycle_kernel = _make_conditional_lifecycle_kernel("float32")
_exact_bf16_execution_domain_kernel = _make_exact_execution_domain_kernel("bf16")
_exact_f32_execution_domain_kernel = _make_exact_execution_domain_kernel("float32")

assert CAPACITY_TEST_LOGICAL_DFBS == 33


def _count_final_dfb_allocations(final_mlir_path):
    return final_mlir_path.read_text().count("dfb_index =")


_exp_scalar_via_scratch = _make_exp_via_scratch_atom("bf16", shape=(1, 1))
_exp_two_tiles_via_scratch = _make_exp_via_scratch_atom("bf16", shape=(1, 2))


@ttl.operation(grid=(1, 1))
def _mixed_capacity_atom_kernel(
    scalar_input, scalar_output, two_tile_input, two_tile_output
):
    scalar_input_dfb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
    scalar_intermediate_dfb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
    scalar_output_dfb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
    two_tile_input_dfb = ttl.make_dfb("bf16", shape=(1, 2), block_count=2)
    two_tile_intermediate_dfb = ttl.make_dfb("bf16", shape=(1, 2), block_count=2)
    two_tile_output_dfb = ttl.make_dfb("bf16", shape=(1, 2), block_count=2)

    ttl.copy(scalar_input[0, 0], scalar_input_dfb.reserve()).wait()
    ttl.copy(two_tile_input[0:1, 0:2], two_tile_input_dfb.reserve()).wait()
    _exp_scalar_via_scratch(scalar_input_dfb, scalar_intermediate_dfb)
    _exp_two_tiles_via_scratch(two_tile_input_dfb, two_tile_intermediate_dfb)
    _exp_scalar_via_scratch(scalar_intermediate_dfb, scalar_output_dfb)
    _exp_two_tiles_via_scratch(two_tile_intermediate_dfb, two_tile_output_dfb)
    ttl.copy(scalar_output_dfb.wait(), scalar_output[0, 0]).wait()
    ttl.copy(two_tile_output_dfb.wait(), two_tile_output[0:1, 0:2]).wait()


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


@ttl.operation(grid=(1, 1))
def _exact_nested_user_dfb_reuse_kernel(first, second, out):
    first_forward_dfb = ttl.make_dataflow_buffer_like(
        first, shape=(1, 1), block_count=2
    )
    first_return_dfb = ttl.make_dataflow_buffer_like(first, shape=(1, 1), block_count=2)
    second_forward_dfb = ttl.make_dataflow_buffer_like(
        second, shape=(1, 1), block_count=2
    )
    second_return_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _ = ttl.node(dims=2)
        if node_x == 0:
            with first_forward_dfb.wait():
                pass

            with first_return_dfb.reserve() as acknowledgment:
                acknowledgment.store(
                    ttl.block.fill(
                        0,
                        shape=acknowledgment.shape,
                        dtype=acknowledgment.dtype,
                    )
                )

            with (
                second_forward_dfb.wait() as second_block,
                second_return_dfb.reserve() as result_block,
            ):
                result_block.store(second_block)

    @ttl.datamovement()
    def data_movement():
        node_x, _ = ttl.node(dims=2)
        if node_x == 0:
            with first_forward_dfb.reserve() as first_block:
                ttl.copy(first[0, 0], first_block).wait()

            with first_return_dfb.wait():
                pass

            with second_forward_dfb.reserve() as second_block:
                ttl.copy(second[0, 0], second_block).wait()

            with second_return_dfb.wait() as result_block:
                ttl.copy(result_block, out[0, 0]).wait()

    # The target runtime requires both NOC kernels. Keeping the complete
    # protocol on one NOC gives both return DFBs the same read-pointer owner.
    @ttl.datamovement()
    def secondary_data_movement():
        pass


@ttl.operation(grid=(1, 1))
def _distinct_noc_owner_kernel(first, second, out):
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

        # The acknowledgment proves that the first DFB is quiescent before
        # NOC1 produces the second DFB.
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

    @ttl.datamovement()
    def dm_write():
        with acknowledgment_dfb.wait():
            pass

        # NOC1 owns this producer pointer, while NOC0 owns first_dfb's
        # producer pointer. They must remain distinct despite ordered lifetimes.
        with second_dfb.reserve() as second_block:
            ttl.copy(second[0, 0], second_block).wait()

        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def _exact_zero_access_domain_kernel(input_tensor, output_tensor):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    inactive_dfb = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=2
    )
    output_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def compute():
        runtime_predicate = ttl.call_extern_func(
            SCALAR_RESULT_HEADER,
            "scalar_result_i32",
            result_type="i32",
        )
        node_x, _ = ttl.node(dims=2)
        if runtime_predicate:
            if node_x == 1:
                inactive_dfb.reserve()

        with input_dfb.wait() as input_block, output_dfb.reserve() as output_block:
            output_block.store(input_block)

    @ttl.datamovement()
    def dm_read():
        with input_dfb.reserve() as input_block:
            ttl.copy(input_tensor[0, 0], input_block).wait()

    @ttl.datamovement()
    def dm_write():
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, output_tensor[0, 0]).wait()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize("reuse_user_dfbs", [True, False], ids=["reuse", "distinct"])
def test_user_dfb_allocation_runtime(
    device, dtype, memory_config, to_device, reuse_user_dfbs
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    first_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    second_host = (((17 * element_indices).remainder(509) - 254) / 128).to(dtype)
    out_host = torch.zeros((TILE, TILE), dtype=dtype)

    first = to_device(first_host, device)
    second = to_device(second_host, device)
    out = to_device(out_host, device)

    reuse_option = (
        "--ttl-reuse-user-dfbs" if reuse_user_dfbs else "--no-ttl-reuse-user-dfbs"
    )
    _user_dfb_reuse_kernel(first, second, out, options=reuse_option)

    actual = ttnn.to_torch(out).float()
    expected = second_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_exact_nested_cross_kernel_protocol_order_reuses_dfbs(
    device, dtype, memory_config, to_device, monkeypatch, tmp_path
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    first_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    second_host = (((17 * element_indices).remainder(509) - 254) / 128).to(dtype)
    out_host = torch.zeros((TILE, TILE), dtype=dtype)

    first = to_device(first_host, device)
    second = to_device(second_host, device)
    out = to_device(out_host, device)

    final_mlir_path = tmp_path / "exact_nested_cross_kernel_order.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    _exact_nested_user_dfb_reuse_kernel(
        first,
        second,
        out,
        options="--ttl-reuse-user-dfbs",
    )

    # The first return transaction orders the two forward DFBs. Matching NOC
    # and compute pointer owners then permit two physical indices for four
    # logical DFBs: one forward index and one return index.
    assert _count_final_dfb_allocations(final_mlir_path) == 2

    actual = ttnn.to_torch(out).float()
    expected = second_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_conditional_bf16_lifecycle_kernel, torch.bfloat16),
        (_conditional_f32_lifecycle_kernel, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_same_runtime_condition_reuses_sequential_dfbs(
    device, operation, dtype, memory_config, to_device, monkeypatch, tmp_path
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    input_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    final_mlir_path = tmp_path / "conditional_lifecycle.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    operation(
        input_tensor,
        output_tensor,
        options="--ttl-reuse-user-dfbs",
    )

    # Input and output ownership prevent them from sharing with the compute
    # scratch DFBs. Three allocations therefore prove that the two guarded
    # scratch lifecycles share one physical index.
    assert _count_final_dfb_allocations(final_mlir_path) == 3

    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_exact_bf16_execution_domain_kernel, torch.bfloat16),
        (_exact_f32_execution_domain_kernel, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_exact_execution_domain_reuses_sequential_dfbs(
    device, operation, dtype, memory_config, to_device, monkeypatch, tmp_path
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    input_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    final_mlir_path = tmp_path / "exact_execution_domain.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    operation(input_tensor, output_tensor, options="--ttl-reuse-user-dfbs")

    # Input and output ownership keep them distinct from the two compute
    # scratch DFBs, whose exact domains permit one shared physical index.
    assert _count_final_dfb_allocations(final_mlir_path) == 3

    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize("reuse_user_dfbs", [True, False], ids=["reuse", "distinct"])
def test_ordered_dfbs_with_distinct_noc_owners(
    device, dtype, memory_config, to_device, reuse_user_dfbs
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    first_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    second_host = (((17 * element_indices).remainder(509) - 254) / 128).to(dtype)
    out_host = torch.zeros((TILE, TILE), dtype=dtype)

    first = to_device(first_host, device)
    second = to_device(second_host, device)
    out = to_device(out_host, device)

    reuse_option = (
        "--ttl-reuse-user-dfbs" if reuse_user_dfbs else "--no-ttl-reuse-user-dfbs"
    )
    _distinct_noc_owner_kernel(first, second, out, options=reuse_option)

    actual = ttnn.to_torch(out).float()
    expected = second_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_exact_zero_access_domain_runtime(device, dtype, memory_config, to_device):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    input_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros((TILE, TILE), dtype=dtype), device)

    _exact_zero_access_domain_kernel(
        input_tensor, output_tensor, options="--ttl-reuse-user-dfbs"
    )

    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host.float()
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
@pytest.mark.parametrize(
    "specialize_cores",
    [False, True],
    ids=["generic-cores", "specialized-cores"],
)
def test_repeated_dfb_declaring_atom_runtime(
    device, memory_config, to_device, operation, dtype, specialize_cores
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    input_host = ((element_indices.remainder(97) - 48) / 128).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros((TILE, TILE), dtype=dtype), device)

    specialization_option = (
        "--ttl-specialize-cores" if specialize_cores else "--no-ttl-specialize-cores"
    )
    operation(input_tensor, output_tensor, options=specialization_option)

    intermediate = torch.exp(input_host.float()).to(dtype)
    expected = torch.exp(intermediate.float()).to(dtype).float()
    actual = ttnn.to_torch(output_tensor).float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(
            actual,
            expected,
            rtol=F32_REPEATED_EXP_RTOL,
            atol=F32_REPEATED_EXP_ATOL,
        )


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
        (_capacity_test_bf16_atom_kernel, torch.bfloat16),
        (_capacity_test_f32_atom_kernel, torch.float32),
    ],
    ids=["bf16", "f32"],
)
def test_33_dfb_atom_composition_with_reuse(
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


def test_blackhole_accepts_33_distinct_dfb_indices(device):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires the Blackhole 64-index DFB capacity")

    input_host = torch.linspace(-1.0, 1.0, TILE * TILE, dtype=torch.bfloat16).reshape(
        TILE, TILE
    )
    input_tensor = to_dram(input_host, device)
    output_tensor = to_dram(torch.zeros_like(input_host), device)

    _capacity_test_bf16_atom_kernel(
        input_tensor,
        output_tensor,
        options="--no-ttl-reuse-user-dfbs",
    )

    actual = ttnn.to_torch(output_tensor).float()
    assert_allclose(actual, input_host.float(), rtol=0.05, atol=1.0)


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
