# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for physical reuse of user-declared DFBs."""

import os
import re

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import ttl_api  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

# TTNN interop rejects non-tilized tensors before DFB lowering, so TILE is the
# only supported tensor layout for these runtime cases.
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
REPEATED_TRANSACTION_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "repeated_dfb_transactions.hpp"
)
DFB_RESET_TEST_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "dfb_reset_test_helpers.hpp"
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


def _make_composed_control_resource_kernel():
    @ttl.operation()
    def copy_through_dfb(input_tensor, output_tensor):
        transfer_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=(1, 1), block_count=2
        )

        with transfer_dfb.reserve() as destination:
            ttl.copy(input_tensor[0, 0], destination).wait()
        with transfer_dfb.wait() as source:
            ttl.copy(source, output_tensor[0, 0]).wait()

    @ttl.operation(grid=(1, 1))
    def composed_control_resource_kernel(input_tensor, output_tensor):
        for iteration in range(2):
            copy_through_dfb(input_tensor, output_tensor)

    return composed_control_resource_kernel


@ttl.operation()
def _store_waited_block(state_dfb: ttl.DFB, output_dfb: ttl.DFB):
    with state_dfb.wait() as state_block:
        increment = ttl.block.fill(
            1,
            shape=state_block.shape,
            dtype=state_block.dtype,
        )
        updated_state = ttl.add(state_block, increment)
        state_block.store(updated_state)

        with output_dfb.reserve() as output_block:
            output_block.store(state_block)


@ttl.operation()
def _iadd_waited_block(state_dfb: ttl.DFB, output_dfb: ttl.DFB):
    with state_dfb.wait() as state_block:
        increment = ttl.block.fill(
            1,
            shape=state_block.shape,
            dtype=state_block.dtype,
        )
        state_block += increment

        with output_dfb.reserve() as output_block:
            output_block.store(
                ttl.add(
                    state_block,
                    ttl.block.fill(
                        0,
                        shape=state_block.shape,
                        dtype=state_block.dtype,
                    ),
                )
            )


def _make_waited_block_store_kernel(data_format, tile_columns=1):
    @ttl.operation(grid=(1, 1))
    def waited_block_mutation_kernel(input_tensor, output_tensor):
        state_dfb = ttl.make_dfb(data_format, shape=(1, tile_columns), block_count=1)
        output_dfb = ttl.make_dfb(data_format, shape=(1, tile_columns), block_count=2)

        with state_dfb.reserve() as state_destination:
            ttl.copy(input_tensor[0:1, 0:tile_columns], state_destination).wait()
        _store_waited_block(state_dfb, output_dfb)
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, output_tensor[0:1, 0:tile_columns]).wait()

    return waited_block_mutation_kernel


def _make_waited_block_iadd_kernel(data_format):
    @ttl.operation(grid=(1, 1))
    def waited_block_mutation_kernel(input_tensor, output_tensor):
        state_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        output_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        with state_dfb.reserve() as state_destination:
            ttl.copy(input_tensor[0, 0], state_destination).wait()
        _iadd_waited_block(state_dfb, output_dfb)
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, output_tensor[0, 0]).wait()

    return waited_block_mutation_kernel


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


def _make_exact_execution_domain_kernel(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(2, 1))
    def exact_execution_domain_kernel(input_tensor, output_tensor):
        scratch_allocation = ttl.make_dfb_allocation_group()
        input_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        first_scratch_dfb = ttl.make_dfb(
            data_format,
            shape=(1, 1),
            block_count=2,
            allocation_group=scratch_allocation,
        )
        second_scratch_dfb = ttl.make_dfb(
            data_format,
            shape=(1, 1),
            block_count=2,
            allocation_group=scratch_allocation,
        )
        output_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            node_x, _ = ttl.node(dims=2)
            first_runtime_predicate = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_result_from_coordinate",
                template_args=[32],
                func_args=[node_x],
                result_type=ttl.ScalarType.I32,
            )
            second_runtime_predicate = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_result_from_coordinate",
                template_args=[32],
                func_args=[node_x],
                result_type=ttl.ScalarType.I32,
            )
            first_node = node_x == 0
            second_node = node_x == 1
            first_active = (first_runtime_predicate and first_node) or first_node
            second_active = (second_runtime_predicate and second_node) or second_node

            with input_dfb.wait() as input_block:
                if first_active:
                    with first_scratch_dfb.reserve() as first_scratch_block:
                        first_scratch_block.store(input_block)
                    with first_scratch_dfb.wait() as first_scratch_block:
                        with output_dfb.reserve() as output_block:
                            output_block.store(first_scratch_block)
                if second_active:
                    with second_scratch_dfb.reserve() as second_scratch_block:
                        second_scratch_block.store(input_block)
                    with second_scratch_dfb.wait() as second_scratch_block:
                        with output_dfb.reserve() as output_block:
                            output_block.store(second_scratch_block)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, _ = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[0, node_x], input_block).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, _ = ttl.node(dims=2)
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output_tensor[0, node_x]).wait()

    return exact_execution_domain_kernel


@ttl.operation(grid=(2, 1))
def _incompatible_static_configuration_kernel(input_tensor, output_tensor):
    first_input_dfb = ttl.make_dfb("float32", shape=(1, 1), block_count=2)
    second_input_dfb = ttl.make_dfb("float32", shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dfb("float32", shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _ = ttl.node(dims=2)
        if node_x == 0:
            with first_input_dfb.wait() as input_block:
                with output_dfb.reserve() as output_block:
                    output_block.store(ttl.exp(input_block))
        if node_x == 1:
            with second_input_dfb.wait() as input_block:
                with output_dfb.reserve() as output_block:
                    output_block.store(
                        ttl.block.broadcast(input_block, dims=[0], shape=(1, 1))
                    )

    @ttl.datamovement()
    def read():
        node_x, _ = ttl.node(dims=2)
        if node_x == 0:
            with first_input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[0, 0], input_block).wait()
        if node_x == 1:
            with second_input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[0, 1], input_block).wait()

    @ttl.datamovement()
    def write():
        node_x, _ = ttl.node(dims=2)
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, output_tensor[0, node_x]).wait()


def _make_repeated_transaction_kernel(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(1, 1))
    def repeated_transaction_kernel(input_tensor, output_tensor):
        first_source = ttl.make_dfb(data_format, shape=(1, 4), block_count=2)
        completion = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_source = ttl.make_dfb(data_format, shape=(1, 4), block_count=2)
        output = ttl.make_dfb(data_format, shape=(1, 4), block_count=2)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            for transaction in range(4):
                with first_source.reserve() as destination:
                    ttl.copy(
                        input_tensor[
                            0:1,
                            transaction * 4 : transaction * 4 + 4,
                        ],
                        destination,
                    ).wait()

            with completion.wait():
                pass

            for transaction in range(4):
                with second_source.reserve() as destination:
                    ttl.copy(
                        input_tensor[
                            0:1,
                            transaction * 4 : transaction * 4 + 4,
                        ],
                        destination,
                    ).wait()

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.call_extern_func(
                REPEATED_TRANSACTION_HEADER,
                "consume_repeated_dfb_and_signal",
                template_args=[
                    ttl.dfb_descriptor(first_source),
                    ttl.dfb_descriptor(completion),
                ],
                dfb_effects=[
                    ttl.DFBEffect.wait(first_source, tiles=4),
                    ttl.DFBEffect.pop(first_source, tiles=4),
                    ttl.DFBEffect.wait(first_source, tiles=4),
                    ttl.DFBEffect.pop(first_source, tiles=4),
                    ttl.DFBEffect.wait(first_source, tiles=4),
                    ttl.DFBEffect.pop(first_source, tiles=4),
                    ttl.DFBEffect.wait(first_source, tiles=4),
                    ttl.DFBEffect.pop(first_source, tiles=4),
                    ttl.DFBEffect.reserve(completion, tiles=1),
                    ttl.DFBEffect.push(completion, tiles=1),
                ],
            )
            ttl.call_extern_func(
                REPEATED_TRANSACTION_HEADER,
                "copy_repeated_dfb",
                template_args=[
                    ttl.dfb_descriptor(second_source),
                    ttl.dfb_descriptor(output),
                ],
                dfb_effects=[
                    ttl.DFBEffect.wait(second_source, tiles=4),
                    ttl.DFBEffect.reserve(output, tiles=4),
                    ttl.DFBEffect.pop(second_source, tiles=4),
                    ttl.DFBEffect.push(output, tiles=4),
                    ttl.DFBEffect.wait(second_source, tiles=4),
                    ttl.DFBEffect.reserve(output, tiles=4),
                    ttl.DFBEffect.pop(second_source, tiles=4),
                    ttl.DFBEffect.push(output, tiles=4),
                    ttl.DFBEffect.wait(second_source, tiles=4),
                    ttl.DFBEffect.reserve(output, tiles=4),
                    ttl.DFBEffect.pop(second_source, tiles=4),
                    ttl.DFBEffect.push(output, tiles=4),
                    ttl.DFBEffect.wait(second_source, tiles=4),
                    ttl.DFBEffect.reserve(output, tiles=4),
                    ttl.DFBEffect.pop(second_source, tiles=4),
                    ttl.DFBEffect.push(output, tiles=4),
                ],
            )

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            for transaction in range(4):
                with output.wait() as source:
                    ttl.copy(
                        source,
                        output_tensor[
                            0:1,
                            transaction * 4 : transaction * 4 + 4,
                        ],
                    ).wait()

    return repeated_transaction_kernel


def _make_conditional_lifecycle_kernel(data_format, predicate_value):
    @ttl.operation(grid=(1, 1))
    def conditional_lifecycle_kernel(input_tensor, output_tensor):
        input_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        first_scratch_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_scratch_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        final_scratch_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        output_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            runtime_predicate = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_predicate",
                template_args=[predicate_value],
                result_type=ttl.ScalarType.I32,
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
                    with second_scratch_dfb.wait():
                        pass

                with final_scratch_dfb.reserve() as final_scratch_block:
                    final_scratch_block.store(input_block)
                with final_scratch_dfb.wait() as final_scratch_block:
                    with output_dfb.reserve() as output_block:
                        output_block.store(final_scratch_block)

        @ttl.datamovement()
        def read():
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[0, 0], input_block).wait()

        @ttl.datamovement()
        def write():
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output_tensor[0, 0]).wait()

    return conditional_lifecycle_kernel


def _make_dispatch_condition_lifecycle_kernel(data_format, predicate_value):
    active = ttl.DispatchCondition(ttl.ScalarType.I32)
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(1, 1))
    def dispatch_condition_lifecycle_kernel(input_tensor, output_tensor):
        first_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        acknowledgment = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        output = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            first_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_predicate",
                template_args=[predicate_value],
                condition_result=active,
            )
            if first_active:
                with first_source.reserve() as destination:
                    ttl.copy(input_tensor[0, 0], destination).wait()

            with acknowledgment.wait():
                pass

            second_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_predicate",
                template_args=[predicate_value],
                condition_result=active,
            )
            if second_active:
                with second_source.reserve() as destination:
                    ttl.copy(input_tensor[0, 0], destination).wait()

        @ttl.compute(kernel=compute_kernel)
        def compute():
            first_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_predicate",
                template_args=[predicate_value],
                condition_result=active,
            )
            if first_active:
                with first_source.wait():
                    pass

            with acknowledgment.reserve() as acknowledgment_block:
                acknowledgment_block.store(
                    ttl.block.fill(
                        0,
                        shape=acknowledgment_block.shape,
                        dtype=acknowledgment_block.dtype,
                    )
                )

            second_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_predicate",
                template_args=[predicate_value],
                condition_result=active,
            )
            if second_active:
                with second_source.wait() as source:
                    with output.reserve() as destination:
                        destination.store(source)

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            output_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_predicate",
                template_args=[predicate_value],
                condition_result=active,
            )
            if output_active:
                with output.wait() as source:
                    ttl.copy(source, output_tensor[0, 0]).wait()

    return dispatch_condition_lifecycle_kernel


def _make_synchronized_reset_kernel(
    data_format,
    reset_all,
    grid_cols=1,
    tile=(32, 32),
    use_compute_passthrough=True,
):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel),
    )

    if reset_all:

        @ttl.operation(grid=(grid_cols, 1))
        def synchronized_reset_kernel(input_tensor, output_tensor):
            reset_allocation = ttl.make_dfb_allocation_group()
            stale_dfb = ttl.make_dfb(
                data_format,
                shape=(1, 1),
                block_count=1,
                tile=tile,
                allocation_group=reset_allocation,
            )
            current_dfb = ttl.make_dfb(
                data_format,
                shape=(1, 1),
                block_count=1,
                tile=tile,
                allocation_group=reset_allocation,
            )
            if use_compute_passthrough:
                output_dfb = ttl.make_dfb(
                    data_format, shape=(1, 1), block_count=1, tile=tile
                )

                @ttl.compute(kernel=compute_kernel)
                def compute():
                    ttl.reset_all_dfbs(reset)
                    with current_dfb.wait() as current_source:
                        with output_dfb.reserve() as output_destination:
                            output_destination.store(current_source)

                source_dfb = output_dfb

            else:

                @ttl.compute(kernel=compute_kernel)
                def compute():
                    ttl.reset_all_dfbs(reset)

                source_dfb = current_dfb

            @ttl.datamovement(kernel=reader_kernel)
            def read():
                node_x, _ = ttl.node(dims=2)
                with stale_dfb.reserve() as stale_destination:
                    ttl.copy(input_tensor[0, node_x], stale_destination).wait()
                ttl.reset_all_dfbs(reset)
                with current_dfb.reserve() as current_destination:
                    ttl.copy(
                        input_tensor[0, grid_cols + node_x], current_destination
                    ).wait()

            @ttl.datamovement(kernel=writer_kernel)
            def write():
                node_x, _ = ttl.node(dims=2)
                ttl.reset_all_dfbs(reset)
                with source_dfb.wait() as output_source:
                    ttl.copy(output_source, output_tensor[0, node_x]).wait()

    else:

        @ttl.operation(grid=(grid_cols, 1))
        def synchronized_reset_kernel(input_tensor, output_tensor):
            reset_allocation = ttl.make_dfb_allocation_group()
            stale_dfb = ttl.make_dfb(
                data_format,
                shape=(1, 1),
                block_count=1,
                tile=tile,
                allocation_group=reset_allocation,
            )
            current_dfb = ttl.make_dfb(
                data_format,
                shape=(1, 1),
                block_count=1,
                tile=tile,
                allocation_group=reset_allocation,
            )
            if use_compute_passthrough:
                output_dfb = ttl.make_dfb(
                    data_format, shape=(1, 1), block_count=1, tile=tile
                )

                @ttl.compute(kernel=compute_kernel)
                def compute():
                    ttl.reset_dfbs(reset, dfbs=[stale_dfb])
                    with current_dfb.wait() as current_source:
                        with output_dfb.reserve() as output_destination:
                            output_destination.store(current_source)

                source_dfb = output_dfb

            else:

                @ttl.compute(kernel=compute_kernel)
                def compute():
                    ttl.reset_dfbs(reset, dfbs=[stale_dfb])

                source_dfb = current_dfb

            @ttl.datamovement(kernel=reader_kernel)
            def read():
                node_x, _ = ttl.node(dims=2)
                with stale_dfb.reserve() as stale_destination:
                    ttl.copy(input_tensor[0, node_x], stale_destination).wait()
                ttl.reset_dfbs(reset, dfbs=[stale_dfb])
                with current_dfb.reserve() as current_destination:
                    ttl.copy(
                        input_tensor[0, grid_cols + node_x], current_destination
                    ).wait()

            @ttl.datamovement(kernel=writer_kernel)
            def write():
                node_x, _ = ttl.node(dims=2)
                ttl.reset_dfbs(reset, dfbs=[stale_dfb])
                with source_dfb.wait() as output_source:
                    ttl.copy(output_source, output_tensor[0, node_x]).wait()

    return synchronized_reset_kernel


def _make_compute_interface_reset_kernel(data_format, tile):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel),
    )

    @ttl.operation(grid=(1, 1))
    def compute_interface_reset_kernel(input_tensor, output_tensor):
        initial_source = ttl.make_dfb(
            data_format, shape=(1, 1), block_count=1, tile=tile
        )
        before_reset = ttl.make_dfb(data_format, shape=(1, 1), block_count=1, tile=tile)
        current_source = ttl.make_dfb(
            data_format, shape=(1, 1), block_count=1, tile=tile
        )
        current_output = ttl.make_dfb(
            data_format, shape=(1, 1), block_count=1, tile=tile
        )

        @ttl.compute(kernel=compute_kernel)
        def compute():
            with initial_source.wait() as initial_block:
                with before_reset.reserve() as before_reset_block:
                    before_reset_block.store(initial_block)
            ttl.reset_dfbs(reset, dfbs=[before_reset])
            with current_source.wait() as current_source_block:
                with before_reset.reserve() as after_reset_block:
                    after_reset_block.store(current_source_block)
            with before_reset.wait() as after_reset_block:
                with current_output.reserve() as current_output_block:
                    current_output_block.store(after_reset_block)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with initial_source.reserve() as initial_block:
                ttl.copy(input_tensor[0, 0], initial_block).wait()
            ttl.reset_dfbs(reset, dfbs=[before_reset])
            with current_source.reserve() as current_source_block:
                ttl.copy(input_tensor[0, 1], current_source_block).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            ttl.reset_dfbs(reset, dfbs=[before_reset])
            with current_output.wait() as output_block:
                ttl.copy(output_block, output_tensor[0, 0]).wait()

    return compute_interface_reset_kernel


def _make_high_index_synchronized_reset_kernel(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel),
    )

    @ttl.operation(grid=(1, 1))
    def synchronized_reset_kernel(input_tensor, output_tensor):
        padding_dfb_00 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_01 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_02 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_03 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_04 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_05 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_06 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_07 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_08 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_09 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_10 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_11 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_12 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_13 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_14 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_15 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_16 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_17 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_18 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_19 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_20 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_21 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_22 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_23 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_24 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_25 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_26 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_27 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_28 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_29 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_30 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_31 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        initial_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        before_reset = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        current_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        current_output = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.call_extern_func(
                DFB_RESET_TEST_HEADER,
                "retain_dfb_liveness",
                template_args=[
                    ttl.dfb_descriptor(padding_dfb_00),
                    ttl.dfb_descriptor(padding_dfb_01),
                    ttl.dfb_descriptor(padding_dfb_02),
                    ttl.dfb_descriptor(padding_dfb_03),
                    ttl.dfb_descriptor(padding_dfb_04),
                    ttl.dfb_descriptor(padding_dfb_05),
                    ttl.dfb_descriptor(padding_dfb_06),
                    ttl.dfb_descriptor(padding_dfb_07),
                    ttl.dfb_descriptor(padding_dfb_08),
                    ttl.dfb_descriptor(padding_dfb_09),
                    ttl.dfb_descriptor(padding_dfb_10),
                    ttl.dfb_descriptor(padding_dfb_11),
                    ttl.dfb_descriptor(padding_dfb_12),
                    ttl.dfb_descriptor(padding_dfb_13),
                    ttl.dfb_descriptor(padding_dfb_14),
                    ttl.dfb_descriptor(padding_dfb_15),
                    ttl.dfb_descriptor(padding_dfb_16),
                    ttl.dfb_descriptor(padding_dfb_17),
                    ttl.dfb_descriptor(padding_dfb_18),
                    ttl.dfb_descriptor(padding_dfb_19),
                    ttl.dfb_descriptor(padding_dfb_20),
                    ttl.dfb_descriptor(padding_dfb_21),
                    ttl.dfb_descriptor(padding_dfb_22),
                    ttl.dfb_descriptor(padding_dfb_23),
                    ttl.dfb_descriptor(padding_dfb_24),
                    ttl.dfb_descriptor(padding_dfb_25),
                    ttl.dfb_descriptor(padding_dfb_26),
                    ttl.dfb_descriptor(padding_dfb_27),
                    ttl.dfb_descriptor(padding_dfb_28),
                    ttl.dfb_descriptor(padding_dfb_29),
                    ttl.dfb_descriptor(padding_dfb_30),
                    ttl.dfb_descriptor(padding_dfb_31),
                ],
                unknown_dfb_access=True,
            )
            with initial_source.wait() as initial_block:
                with before_reset.reserve() as before_reset_block:
                    before_reset_block.store(initial_block)
            ttl.reset_dfbs(reset, dfbs=[before_reset])
            with current_source.wait() as current_source_block:
                with before_reset.reserve() as after_reset_block:
                    after_reset_block.store(current_source_block)
            with before_reset.wait() as after_reset_block:
                with current_output.reserve() as current_output_block:
                    current_output_block.store(after_reset_block)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with initial_source.reserve() as initial_block:
                ttl.copy(input_tensor[0, 0], initial_block).wait()
            ttl.reset_dfbs(reset, dfbs=[before_reset])
            with current_source.reserve() as current_source_block:
                ttl.copy(input_tensor[0, 1], current_source_block).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            ttl.reset_dfbs(reset, dfbs=[before_reset])
            with current_output.wait() as output_block:
                ttl.copy(output_block, output_tensor[0, 0]).wait()

    return synchronized_reset_kernel


def _make_selected_reset_alias_domain_kernel(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel),
    )

    @ttl.operation(grid=(2, 1))
    def selected_reset_alias_domain_kernel(input_tensor, output_tensor):
        selected_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        crossing_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, _ = ttl.node(dims=2)
            if node_x == 0:
                with selected_dfb.reserve() as selected_block:
                    ttl.copy(input_tensor[0, 0], selected_block).wait()
            if node_x == 1:
                with crossing_dfb.reserve() as crossing_block:
                    ttl.copy(input_tensor[0, 1], crossing_block).wait()
            ttl.reset_dfbs(reset, dfbs=[selected_dfb])

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reset_dfbs(reset, dfbs=[selected_dfb])

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, _ = ttl.node(dims=2)
            if node_x == 0:
                with selected_dfb.wait() as selected_block:
                    ttl.copy(selected_block, output_tensor[0, 0]).wait()
            ttl.reset_dfbs(reset, dfbs=[selected_dfb])
            if node_x == 1:
                with crossing_dfb.wait() as crossing_block:
                    ttl.copy(crossing_block, output_tensor[0, 1]).wait()

    return selected_reset_alias_domain_kernel


def _make_allocation_group_kernel(data_format):
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(1, 1))
    def allocation_group_kernel(input_tensor, output_tensor):
        shared_allocation = ttl.make_dfb_allocation_group()
        first_source = ttl.make_dfb(
            data_format,
            shape=(1, 2),
            block_count=1,
            allocation_group=shared_allocation,
        )
        handoff = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_source = ttl.make_dfb(
            data_format,
            shape=(1, 1),
            block_count=4,
            allocation_group=shared_allocation,
        )
        output = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with first_source.reserve() as destination:
                ttl.copy(input_tensor[0:1, 0:2], destination).wait()

            with handoff.wait():
                pass

            with second_source.reserve() as destination:
                ttl.copy(input_tensor[0, 0], destination).wait()
            with second_source.reserve() as destination:
                ttl.copy(input_tensor[0, 0], destination).wait()

        @ttl.compute(kernel=compute_kernel)
        def compute():
            with first_source.wait():
                pass

            with handoff.reserve() as signal:
                signal.store(ttl.block.fill(0, shape=signal.shape, dtype=signal.dtype))

            with second_source.wait():
                pass

            with second_source.wait() as source:
                with output.reserve() as destination:
                    destination.store(source)

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            with output.wait() as source:
                ttl.copy(source, output_tensor[0, 0]).wait()

    return allocation_group_kernel


_repeated_bf16_atom_kernel = _make_repeated_dfb_atom_kernel("bf16")
_repeated_f32_atom_kernel = _make_repeated_dfb_atom_kernel("float32")
_composed_control_resource_kernel = _make_composed_control_resource_kernel()
_in_place_bf16_atom_kernel = _make_in_place_atom_kernel("bf16")
_in_place_f32_atom_kernel = _make_in_place_atom_kernel("float32")
_capacity_test_bf16_atom_kernel = _make_capacity_test_atom_kernel("bf16")
_capacity_test_f32_atom_kernel = _make_capacity_test_atom_kernel("float32")
_waited_mutation_bf16_store_kernel = _make_waited_block_store_kernel("bf16")
_waited_mutation_f32_store_kernel = _make_waited_block_store_kernel("float32")
_waited_mutation_bf16_two_tile_store_kernel = _make_waited_block_store_kernel(
    "bf16", tile_columns=2
)
_waited_mutation_f32_two_tile_store_kernel = _make_waited_block_store_kernel(
    "float32", tile_columns=2
)
_waited_mutation_bf16_iadd_kernel = _make_waited_block_iadd_kernel("bf16")
_waited_mutation_f32_iadd_kernel = _make_waited_block_iadd_kernel("float32")
_exact_bf16_execution_domain_kernel = _make_exact_execution_domain_kernel("bf16")
_exact_f32_execution_domain_kernel = _make_exact_execution_domain_kernel("float32")
_conditional_bf16_true_lifecycle_kernel = _make_conditional_lifecycle_kernel(
    "bf16", True
)
_conditional_bf16_false_lifecycle_kernel = _make_conditional_lifecycle_kernel(
    "bf16", False
)
_conditional_f32_true_lifecycle_kernel = _make_conditional_lifecycle_kernel(
    "float32", True
)
_conditional_f32_false_lifecycle_kernel = _make_conditional_lifecycle_kernel(
    "float32", False
)
_dispatch_condition_bf16_true_lifecycle_kernel = (
    _make_dispatch_condition_lifecycle_kernel("bf16", True)
)
_dispatch_condition_bf16_false_lifecycle_kernel = (
    _make_dispatch_condition_lifecycle_kernel("bf16", False)
)
_dispatch_condition_f32_true_lifecycle_kernel = (
    _make_dispatch_condition_lifecycle_kernel("float32", True)
)
_dispatch_condition_f32_false_lifecycle_kernel = (
    _make_dispatch_condition_lifecycle_kernel("float32", False)
)
_allocation_group_bf16_kernel = _make_allocation_group_kernel("bf16")
_allocation_group_f32_kernel = _make_allocation_group_kernel("float32")

assert CAPACITY_TEST_LOGICAL_DFBS == 33


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
def test_exact_disjoint_execution_domains_reuse_dfb(
    device, operation, dtype, memory_config, to_device, monkeypatch, tmp_path
):
    element_indices = torch.arange(2 * TILE * TILE, dtype=torch.float32).reshape(
        TILE, 2 * TILE
    )
    input_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    final_mlir_path = tmp_path / "exact_execution_domain.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    operation(input_tensor, output_tensor, options="--ttl-reuse-user-dfbs")

    # Input and output hardware owners keep them distinct from the compute
    # scratch DFBs. The scratch DFBs share because their exact node domains are
    # disjoint, without requiring a local lifetime-order proof.
    physical_dfb_count = final_mlir_path.read_text().count("dfb_index =")
    assert physical_dfb_count == 3

    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("data_format", "dtype"),
    [
        ("bf16", torch.bfloat16),
        ("float32", torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_repeated_transaction_lifecycles_reuse_dfb(
    device, data_format, dtype, memory_config, to_device, monkeypatch, tmp_path
):
    operation = _make_repeated_transaction_kernel(data_format)
    element_indices = torch.arange(TILE * 16 * TILE, dtype=torch.float32).reshape(
        TILE, 16 * TILE
    )
    input_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    final_mlir_path = tmp_path / "repeated_transactions.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    operation(input_tensor, output_tensor, options="--ttl-reuse-user-dfbs")

    # The two source DFBs have identical pointer owners and non-overlapping
    # lifetimes. Completion and output retain distinct DFB types or owners.
    physical_dfb_count = final_mlir_path.read_text().count("dfb_index =")
    assert physical_dfb_count == 3

    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_conditional_bf16_true_lifecycle_kernel, torch.bfloat16),
        (_conditional_bf16_false_lifecycle_kernel, torch.bfloat16),
        (_conditional_f32_true_lifecycle_kernel, torch.float32),
        (_conditional_f32_false_lifecycle_kernel, torch.float32),
    ],
    ids=["bf16-true", "bf16-false", "f32-true", "f32-false"],
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
    operation(input_tensor, output_tensor, options="--ttl-reuse-user-dfbs")

    # Input and output ownership prevent them from sharing with compute DFBs.
    # Three allocations prove that all three scratch lifecycles share one index.
    physical_dfb_count = final_mlir_path.read_text().count("dfb_index =")
    assert physical_dfb_count == 3

    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "dtype", "predicate_value"),
    [
        (_dispatch_condition_bf16_true_lifecycle_kernel, torch.bfloat16, True),
        (_dispatch_condition_bf16_false_lifecycle_kernel, torch.bfloat16, False),
        (_dispatch_condition_f32_true_lifecycle_kernel, torch.float32, True),
        (_dispatch_condition_f32_false_lifecycle_kernel, torch.float32, False),
    ],
    ids=["bf16-true", "bf16-false", "f32-true", "f32-false"],
)
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_dispatch_condition_reuses_dfbs_across_logical_kernels(
    device,
    operation,
    dtype,
    predicate_value,
    memory_config,
    to_device,
    monkeypatch,
    tmp_path,
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    input_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    final_mlir_path = tmp_path / "dispatch_condition_lifecycle.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    operation(input_tensor, output_tensor, options="--ttl-reuse-user-dfbs")

    # The first and second sources have equal types and pointer owners. Their
    # separately evaluated producer and consumer conditions share one identity.
    physical_dfb_count = final_mlir_path.read_text().count("dfb_index =")
    assert physical_dfb_count == 3

    actual = ttnn.to_torch(output_tensor).float()
    expected = (
        input_host.float() if predicate_value else torch.zeros_like(input_host).float()
    )
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("reset_all", [False, True], ids=["selected", "all"])
@pytest.mark.parametrize("grid_cols", [1, 2], ids=["one-core", "two-core"])
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_synchronized_reset_terminates_producer_epoch(
    device,
    dtype,
    reset_all,
    grid_cols,
    memory_config,
    to_device,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reset support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_synchronized_reset_kernel(data_format, reset_all, grid_cols)

    element_indices = torch.arange(
        2 * TILE * TILE * grid_cols, dtype=torch.float32
    ).reshape(TILE, 2 * TILE * grid_cols)
    input_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    input_tensor = to_device(input_host, device)
    output_host = torch.zeros(TILE, TILE * grid_cols, dtype=dtype)
    output_tensor = to_device(output_host, device)

    final_mlir_path = tmp_path / "synchronized_reset.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    for _invocation_index in range(2):
        operation(input_tensor, output_tensor, options="--ttl-reuse-user-dfbs")

    # The producer-only DFB becomes canonical at the reset and shares with the
    # following source. The compute-produced output retains a distinct index.
    physical_dfb_count = final_mlir_path.read_text().count("dfb_index =")
    assert physical_dfb_count == 2

    actual = ttnn.to_torch(output_tensor).float()
    expected = input_host[:, TILE * grid_cols :].float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_synchronized_reset_executes_above_physical_index_31(
    device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires the Blackhole 64-index DFB capacity")

    operation = _make_high_index_synchronized_reset_kernel("bf16")
    final_mlir_path = tmp_path / "synchronized_reset_high_index.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))

    for invocation_index in range(2):
        input_host = (
            torch.arange(2 * TILE * TILE, dtype=torch.float32).reshape(TILE, 2 * TILE)
            + invocation_index * 13
        ).to(torch.bfloat16)
        input_tensor = to_dram(input_host, device)
        output_tensor = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        operation(input_tensor, output_tensor, options="--no-ttl-reuse-user-dfbs")
        assert_allclose(
            ttnn.to_torch(output_tensor).float(),
            input_host[:, TILE:].float(),
            rtol=0.05,
            atol=1.0,
        )

    final_mlir = final_mlir_path.read_text()
    assert "dfb_index = 33 : i32" in final_mlir
    assert "get_compile_time_arg_val(33)" in final_mlir
    assert "value = 2 : i32" in final_mlir
    assert "experimental::reset_dfb_interfaces" in final_mlir


@pytest.mark.parametrize(
    ("data_format", "ttnn_dtype", "torch_dtype", "tile", "compute_interface"),
    [
        ("bfp_bf4", ttnn.bfloat4_b, torch.bfloat16, (8, 32), False),
        ("uint8", ttnn.uint8, torch.uint8, (1, 16), False),
        ("bfp_bf4", ttnn.bfloat4_b, torch.bfloat16, (32, 32), True),
        ("uint16", ttnn.uint16, torch.uint16, (16, 32), True),
    ],
    ids=[
        "bfp4-8x32-dm",
        "uint8-1x16-dm",
        "bfp4-32x32-compute",
        "uint16-16x32-compute",
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram", "l1"],
)
def test_synchronized_reset_supports_packed_and_integer_subtiles(
    device,
    data_format,
    ttnn_dtype,
    torch_dtype,
    tile,
    compute_interface,
    memory_config,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reset support")

    operation = (
        _make_compute_interface_reset_kernel(data_format, tile)
        if compute_interface
        else _make_synchronized_reset_kernel(
            data_format,
            reset_all=False,
            tile=tile,
            use_compute_passthrough=False,
        )
    )
    tile_height, tile_width = tile
    input_host = (
        torch.arange(2 * tile_height * tile_width, dtype=torch.int64)
        .remainder(127)
        .reshape(tile_height, 2 * tile_width)
        .to(torch_dtype)
    )
    input_tensor = ttnn.from_torch(
        input_host,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=ttnn.Tile(tile),
        memory_config=memory_config,
    )
    output_tensor = ttnn.from_torch(
        torch.zeros((tile_height, tile_width), dtype=torch_dtype),
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=ttnn.Tile(tile),
        memory_config=memory_config,
    )
    expected = ttnn.to_torch(input_tensor).reshape(tile_height, 2 * tile_width)[
        :, tile_width:
    ]

    for _invocation_index in range(2):
        operation(input_tensor, output_tensor, options="--ttl-reuse-user-dfbs")

    actual = ttnn.to_torch(output_tensor).reshape(tile)
    if torch_dtype == torch.uint8:
        assert torch.equal(actual.to(torch.int64), expected.to(torch.int64))
    else:
        assert_allclose(actual.float(), expected.float(), rtol=0.0, atol=0.0)


def test_selected_reset_preserves_non_target_live_aliases(
    device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reset support")

    operation = _make_selected_reset_alias_domain_kernel("bf16")
    input_host = (
        torch.arange(2 * TILE * TILE, dtype=torch.float32)
        .reshape(TILE, 2 * TILE)
        .to(torch.bfloat16)
    )
    input_tensor = to_dram(input_host, device)
    output_tensor = to_dram(torch.zeros_like(input_host), device)
    final_mlir_path = tmp_path / "selected_reset_alias_domain.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))

    for _invocation_index in range(2):
        operation(input_tensor, output_tensor, options="--ttl-reuse-user-dfbs")

    physical_dfb_count = final_mlir_path.read_text().count("dfb_index =")
    assert physical_dfb_count == 2
    assert_allclose(
        ttnn.to_torch(output_tensor).float(),
        input_host.float(),
        rtol=0.05,
        atol=1.0,
    )


@pytest.mark.parametrize(
    ("operation", "dtype", "tile_columns"),
    [
        (_waited_mutation_bf16_store_kernel, torch.bfloat16, 1),
        (_waited_mutation_f32_store_kernel, torch.float32, 1),
        (_waited_mutation_bf16_iadd_kernel, torch.bfloat16, 1),
        (_waited_mutation_f32_iadd_kernel, torch.float32, 1),
        (_waited_mutation_bf16_two_tile_store_kernel, torch.bfloat16, 2),
        (_waited_mutation_f32_two_tile_store_kernel, torch.float32, 2),
    ],
    ids=[
        "store-bf16",
        "store-f32",
        "iadd-bf16",
        "iadd-f32",
        "store-two-tiles-bf16",
        "store-two-tiles-f32",
    ],
)
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
def test_waited_block_replacement_preserves_updated_value(
    device, operation, dtype, tile_columns, memory_config, to_device
):
    element_indices = torch.arange(
        TILE * TILE * tile_columns, dtype=torch.float32
    ).reshape(TILE, TILE * tile_columns)
    input_host = ((element_indices.remainder(257) - 128) / 64).to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    operation(input_tensor, output_tensor)
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
def test_disjoint_incompatible_static_configurations_do_not_reuse_dfb(
    device, memory_config, to_device, monkeypatch, tmp_path
):
    # Only FP32 requires distinct unpack modes for exponential and broadcast.
    first_input = (
        torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE) / 2048
    )
    second_input = torch.zeros((TILE, TILE), dtype=torch.float32)
    second_input[0, :] = torch.arange(TILE, dtype=torch.float32) / 32
    input_host = torch.cat((first_input, second_input), dim=1)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    final_mlir_path = tmp_path / "incompatible_static_configurations.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    _incompatible_static_configuration_kernel(
        input_tensor, output_tensor, options="--ttl-reuse-user-dfbs"
    )

    # The two input DFBs are active on disjoint nodes, but their unpack modes
    # require distinct physical indices. The output DFB adds a third index.
    physical_dfb_count = final_mlir_path.read_text().count("dfb_index =")
    assert physical_dfb_count == 3

    first_expected = torch.exp(first_input)
    second_expected = second_input[0:1, :].expand(TILE, TILE)
    expected = torch.cat((first_expected, second_expected), dim=1)
    actual = ttnn.to_torch(output_tensor).float()
    assert_allclose(actual, expected.float(), rtol=2e-3, atol=5e-4)


@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_allocation_group_bf16_kernel, torch.bfloat16),
        (_allocation_group_f32_kernel, torch.float32),
    ],
    ids=["bf16", "f32"],
)
def test_allocation_group_reuses_one_capacity_envelope(
    device,
    operation,
    dtype,
    memory_config,
    to_device,
    tmp_path,
    monkeypatch,
):
    input_host = torch.linspace(-1, 1, TILE * TILE * 2, dtype=torch.float32).reshape(
        TILE, TILE * 2
    )
    input_host = input_host.to(dtype)
    input_tensor = to_device(input_host, device)
    expected = input_host[:, :TILE]
    output_tensor = to_device(torch.zeros_like(expected), device)
    final_mlir_path = tmp_path / "allocation_group.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))

    operation(input_tensor, output_tensor)
    operation(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected.float(), rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected.float(), rtol=1e-5, atol=1e-6)

    final_mlir = final_mlir_path.read_text()
    reader_mlir = final_mlir.split("func.func @read", 1)[1].split(
        "func.func @compute", 1
    )[0]
    reader_dfb_indices = [
        int(index)
        for index in re.findall(r"ttkernel\.cb_ctarg_idx = (\d+)", reader_mlir)
    ]
    assert reader_dfb_indices == [0, 1, 0]
    assert "block_count = 4 : i32, dfb_index = 0 : i32" in final_mlir


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
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
def test_composed_control_resource_runtime(device, memory_config, to_device, dtype):
    input_host = torch.linspace(-1.0, 1.0, TILE * TILE, dtype=torch.float32).reshape(
        TILE, TILE
    )
    input_host = input_host.to(dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    _composed_control_resource_kernel(input_tensor, output_tensor)
    _composed_control_resource_kernel(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor).float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, input_host.float(), rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, input_host.float(), rtol=1e-5, atol=1e-6)


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
