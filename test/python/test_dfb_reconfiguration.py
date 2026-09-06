# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for synchronized multi-epoch DFB configuration."""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import ttl_api  # noqa: E402
from ttl.dtype_utils import format_name_to_ttnn_dtype  # noqa: E402
from ttlang_test_utils import to_dram, to_l1, to_l1_sharded  # noqa: E402
from utils.correctness import assert_allclose, assert_pcc  # noqa: E402

pytestmark = pytest.mark.requires_device

SCALAR_RESULT_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "scalar_result_op.hpp"
)
DFB_RECONFIGURATION_TEST_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "dfb_reconfiguration_test_helpers.hpp"
)
DFB_LIVENESS_TEST_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "dfb_liveness_test_helpers.hpp"
)


def _make_reconfiguration_operation(data_format, grid_cols):
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    first_boundary = ttl.DFBReconfiguration(
        participants=(ttl.KernelKind.COMPUTE, reader_kernel, writer_kernel)
    )
    second_boundary = ttl.DFBReconfiguration(
        participants=(ttl.KernelKind.COMPUTE, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(grid_cols, 1))
    def reconfiguration_operation(
        first_input,
        first_output,
        second_input,
        second_output,
        third_input,
        third_output,
    ):
        first_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        first_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_source = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        second_result = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        third_source = ttl.make_dfb(data_format, shape=(2, 1), block_count=4)
        third_result = ttl.make_dfb(data_format, shape=(2, 1), block_count=4)

        @ttl.compute()
        def compute():
            with first_source.wait() as source:
                with first_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(first_boundary)
            with second_source.wait() as source:
                with second_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(second_boundary)
            with third_source.wait() as source:
                with third_result.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, _ = ttl.node(dims=2)
            with first_source.reserve() as destination:
                ttl.copy(first_input[0, node_x], destination).wait()
            ttl.reconfigure_dfbs(first_boundary)
            with second_source.reserve() as destination:
                ttl.copy(
                    second_input[0:1, node_x * 2 : node_x * 2 + 2],
                    destination,
                ).wait()
            ttl.reconfigure_dfbs(second_boundary)
            with third_source.reserve() as destination:
                ttl.copy(third_input[0:2, node_x : node_x + 1], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, _ = ttl.node(dims=2)
            with first_result.wait() as source:
                ttl.copy(source, first_output[0, node_x]).wait()
            ttl.reconfigure_dfbs(first_boundary)
            with second_result.wait() as source:
                ttl.copy(
                    source,
                    second_output[0:1, node_x * 2 : node_x * 2 + 2],
                ).wait()
            ttl.reconfigure_dfbs(second_boundary)
            with third_result.wait() as source:
                ttl.copy(source, third_output[0:2, node_x : node_x + 1]).wait()

    return reconfiguration_operation


# Builds alternating DFB lifecycles while one payload remains live across the
# first boundary.
def _make_repeated_reconfiguration_operation(data_format, iterations, grid_cols):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    first_boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )
    second_boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(grid_cols, 1))
    def repeated_reconfiguration_operation(
        first_input,
        first_output,
        preserved_input,
        preserved_output,
        second_input,
        second_output,
    ):
        first_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        first_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        preserved_source = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        preserved_result = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        second_source = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        second_result = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            for _iteration in range(iterations):
                with first_source.wait() as source:
                    with first_result.reserve() as result:
                        result.store(source)
                ttl.reconfigure_dfbs(first_boundary)
                with preserved_source.wait() as source:
                    with preserved_result.reserve() as result:
                        result.store(source)
                with second_source.wait() as source:
                    with second_result.reserve() as result:
                        result.store(source)
                ttl.reconfigure_dfbs(second_boundary)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, _ = ttl.node(dims=2)
            for iteration in range(iterations):
                with first_source.reserve() as destination:
                    ttl.copy(
                        first_input[iteration : iteration + 1, node_x : node_x + 1],
                        destination,
                    ).wait()
                with preserved_source.reserve() as destination:
                    ttl.copy(
                        preserved_input[
                            iteration : iteration + 1,
                            node_x * 2 : node_x * 2 + 2,
                        ],
                        destination,
                    ).wait()
                ttl.reconfigure_dfbs(first_boundary)
                with second_source.reserve() as destination:
                    ttl.copy(
                        second_input[
                            iteration : iteration + 1,
                            node_x * 2 : node_x * 2 + 2,
                        ],
                        destination,
                    ).wait()
                ttl.reconfigure_dfbs(second_boundary)

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, _ = ttl.node(dims=2)
            for iteration in range(iterations):
                with first_result.wait() as source:
                    ttl.copy(
                        source,
                        first_output[iteration : iteration + 1, node_x : node_x + 1],
                    ).wait()
                ttl.reconfigure_dfbs(first_boundary)
                with preserved_result.wait() as source:
                    ttl.copy(
                        source,
                        preserved_output[
                            iteration : iteration + 1,
                            node_x * 2 : node_x * 2 + 2,
                        ],
                    ).wait()
                with second_result.wait() as source:
                    ttl.copy(
                        source,
                        second_output[
                            iteration : iteration + 1,
                            node_x * 2 : node_x * 2 + 2,
                        ],
                    ).wait()
                ttl.reconfigure_dfbs(second_boundary)

    return repeated_reconfiguration_operation


def _make_live_crossing_operation(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def live_crossing_operation(
        before_input,
        before_output,
        crossing_input,
        crossing_output,
        after_input,
        after_output,
    ):
        before_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        before_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        crossing_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        crossing_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        after_source = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        after_result = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)

        @ttl.compute(kernel=compute_kernel)
        def live_payload_compute():
            with before_source.wait() as source:
                with before_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(boundary)
            with crossing_source.wait() as source:
                with crossing_result.reserve() as result:
                    result.store(source)
            with after_source.wait() as source:
                with after_result.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def live_payload_reader():
            with before_source.reserve() as destination:
                ttl.copy(before_input[0, 0], destination).wait()
            with crossing_source.reserve() as destination:
                ttl.copy(crossing_input[0, 0], destination).wait()
            ttl.reconfigure_dfbs(boundary)
            with after_source.reserve() as destination:
                ttl.copy(after_input[0:1, 0:2], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def live_payload_writer():
            with before_result.wait() as source:
                ttl.copy(source, before_output[0, 0]).wait()
            ttl.reconfigure_dfbs(boundary)
            with crossing_result.wait() as source:
                ttl.copy(source, crossing_output[0, 0]).wait()
            with after_result.wait() as source:
                ttl.copy(source, after_output[0:1, 0:2]).wait()

    return live_crossing_operation


def _make_conditional_reconfiguration_operation(data_format, enabled_column):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(2, 1))
    def conditional_reconfiguration_operation(input_tensor, output_tensor):
        source_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            node_x, node_y = ttl.node(dims=2)
            if node_x == enabled_column:
                ttl.reconfigure_dfbs(boundary)
                with source_dfb.wait() as source:
                    with result_dfb.reserve() as result:
                        result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, node_y = ttl.node(dims=2)
            if node_x == enabled_column:
                ttl.reconfigure_dfbs(boundary)
                with source_dfb.reserve() as destination:
                    ttl.copy(input_tensor[0, node_x], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, node_y = ttl.node(dims=2)
            if node_x == enabled_column:
                ttl.reconfigure_dfbs(boundary)
                with result_dfb.wait() as source:
                    ttl.copy(source, output_tensor[0, node_x]).wait()

    return conditional_reconfiguration_operation


def _make_dispatch_condition_reconfiguration_operation(data_format, active_value):
    active = ttl.DispatchCondition(ttl.ScalarType.I32)
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def dispatch_condition_reconfiguration_operation(input_tensor, output_tensor):
        source_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            is_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_predicate",
                template_args=[active_value],
                condition_result=active,
            )
            if is_active:
                ttl.reconfigure_dfbs(boundary)
                with source_dfb.wait() as source:
                    with result_dfb.reserve() as result:
                        result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            is_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_predicate",
                template_args=[active_value],
                condition_result=active,
            )
            if is_active:
                ttl.reconfigure_dfbs(boundary)
                with source_dfb.reserve() as destination:
                    ttl.copy(input_tensor[0, 0], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            is_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_predicate",
                template_args=[active_value],
                condition_result=active,
            )
            if is_active:
                ttl.reconfigure_dfbs(boundary)
                with result_dfb.wait() as source:
                    ttl.copy(source, output_tensor[0, 0]).wait()

    return dispatch_condition_reconfiguration_operation


def _make_high_index_reconfiguration_operation(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def high_index_reconfiguration_operation(input_tensor, output_tensor):
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
        source_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.call_extern_func(
                DFB_LIVENESS_TEST_HEADER,
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
                dfb_accesses=[
                    ttl.DFBAccess.inspect(padding_dfb_00),
                    ttl.DFBAccess.inspect(padding_dfb_01),
                    ttl.DFBAccess.inspect(padding_dfb_02),
                    ttl.DFBAccess.inspect(padding_dfb_03),
                    ttl.DFBAccess.inspect(padding_dfb_04),
                    ttl.DFBAccess.inspect(padding_dfb_05),
                    ttl.DFBAccess.inspect(padding_dfb_06),
                    ttl.DFBAccess.inspect(padding_dfb_07),
                    ttl.DFBAccess.inspect(padding_dfb_08),
                    ttl.DFBAccess.inspect(padding_dfb_09),
                    ttl.DFBAccess.inspect(padding_dfb_10),
                    ttl.DFBAccess.inspect(padding_dfb_11),
                    ttl.DFBAccess.inspect(padding_dfb_12),
                    ttl.DFBAccess.inspect(padding_dfb_13),
                    ttl.DFBAccess.inspect(padding_dfb_14),
                    ttl.DFBAccess.inspect(padding_dfb_15),
                    ttl.DFBAccess.inspect(padding_dfb_16),
                    ttl.DFBAccess.inspect(padding_dfb_17),
                    ttl.DFBAccess.inspect(padding_dfb_18),
                    ttl.DFBAccess.inspect(padding_dfb_19),
                    ttl.DFBAccess.inspect(padding_dfb_20),
                    ttl.DFBAccess.inspect(padding_dfb_21),
                    ttl.DFBAccess.inspect(padding_dfb_22),
                    ttl.DFBAccess.inspect(padding_dfb_23),
                    ttl.DFBAccess.inspect(padding_dfb_24),
                    ttl.DFBAccess.inspect(padding_dfb_25),
                    ttl.DFBAccess.inspect(padding_dfb_26),
                    ttl.DFBAccess.inspect(padding_dfb_27),
                    ttl.DFBAccess.inspect(padding_dfb_28),
                    ttl.DFBAccess.inspect(padding_dfb_29),
                    ttl.DFBAccess.inspect(padding_dfb_30),
                    ttl.DFBAccess.inspect(padding_dfb_31),
                ],
            )
            ttl.reconfigure_dfbs(boundary)
            with source_dfb.wait() as source:
                with result_dfb.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            ttl.reconfigure_dfbs(boundary)
            with source_dfb.reserve() as destination:
                ttl.copy(input_tensor[0, 0], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            ttl.reconfigure_dfbs(boundary)
            with result_dfb.wait() as source:
                ttl.copy(source, output_tensor[0, 0]).wait()

    return high_index_reconfiguration_operation


def _make_tensor_backed_reconfiguration_operation(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def tensor_backed_reconfiguration_operation(
        tensor_backed_input,
        tensor_backed_output,
        scratch_input,
        scratch_output,
    ):
        tensor_backed_source = ttl.make_tensor_backed_dfb(
            tensor_backed_input, shape=(1, 1)
        )
        tensor_backed_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        scratch_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=3)
        scratch_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=3)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            with tensor_backed_source.wait() as source:
                with tensor_backed_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(boundary)
            with scratch_source.wait() as source:
                with scratch_result.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            tensor_backed_source.publish()
            ttl.reconfigure_dfbs(boundary)
            with scratch_source.reserve() as destination:
                ttl.copy(scratch_input[0, 0], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            with tensor_backed_result.wait() as source:
                ttl.copy(source, tensor_backed_output[0, 0]).wait()
            ttl.reconfigure_dfbs(boundary)
            with scratch_result.wait() as source:
                ttl.copy(source, scratch_output[0, 0]).wait()

    return tensor_backed_reconfiguration_operation


def _make_cross_core_tensor_backed_reconfiguration_operation():
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(2, 1))
    def cross_core_tensor_backed_reconfiguration_operation(input_tensor, output):
        first_source = ttl.make_tensor_backed_dfb(input_tensor, shape=(1, 1))
        second_source = ttl.make_tensor_backed_dfb(input_tensor, shape=(1, 1))

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reconfigure_dfbs(boundary)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, _ = ttl.node(dims=2)
            if node_x == 0:
                first_source.publish()
            ttl.reconfigure_dfbs(boundary)
            if node_x == 1:
                second_source.publish()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, _ = ttl.node(dims=2)
            if node_x == 0:
                with first_source.wait() as source:
                    ttl.copy(source, output[node_x, 0]).wait()
            ttl.reconfigure_dfbs(boundary)
            if node_x == 1:
                with second_source.wait() as source:
                    ttl.copy(source, output[node_x, 0]).wait()

    return cross_core_tensor_backed_reconfiguration_operation


def _make_native_format_reconfiguration_operation(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def native_format_reconfiguration_operation(
        first_input, first_output, second_input, second_output
    ):
        first_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        first_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_source = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        second_result = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            with first_source.wait() as source:
                with first_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(boundary)
            with second_source.wait() as source:
                with second_result.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with first_source.reserve() as destination:
                ttl.copy(first_input[0, 0], destination).wait()
            ttl.reconfigure_dfbs(boundary)
            with second_source.reserve() as destination:
                ttl.copy(second_input[0:1, 0:2], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            with first_result.wait() as source:
                ttl.copy(source, first_output[0, 0]).wait()
            ttl.reconfigure_dfbs(boundary)
            with second_result.wait() as source:
                ttl.copy(source, second_output[0:1, 0:2]).wait()

    return native_format_reconfiguration_operation


def _make_data_movement_format_reconfiguration_operation(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def data_movement_format_reconfiguration_operation(
        first_input, first_output, second_input, second_output
    ):
        first_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_dfb = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reconfigure_dfbs(boundary)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with first_dfb.reserve() as destination:
                ttl.copy(first_input[0, 0], destination).wait()
            ttl.reconfigure_dfbs(boundary)
            with second_dfb.reserve() as destination:
                ttl.copy(second_input[0:1, 0:2], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            with first_dfb.wait() as source:
                ttl.copy(source, first_output[0, 0]).wait()
            ttl.reconfigure_dfbs(boundary)
            with second_dfb.wait() as source:
                ttl.copy(source, second_output[0:1, 0:2]).wait()

    return data_movement_format_reconfiguration_operation


def _make_caller_runtime_arg_reconfiguration_operation():
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    def runtime_resource_factory(**_kwargs):
        return ttl.ProgramRuntimeResources(
            kernel_resources=(
                ttl.KernelRuntimeResources(
                    kernel=reader_kernel,
                    runtime_args=(
                        ttl.CoreRuntimeArgs(ttnn.CoreCoord(0, 0), (0x3F80,)),
                        ttl.CoreRuntimeArgs(ttnn.CoreCoord(1, 0), (0x4000, 0x4040)),
                    ),
                ),
            )
        )

    @ttl.operation(grid=(2, 1), runtime_resource_factory=runtime_resource_factory)
    def caller_runtime_arg_reconfiguration_operation(
        runtime_output, second_input, second_output
    ):
        runtime_dfb = ttl.make_dfb("bfloat16", shape=(1, 1), block_count=2)
        second_dfb = ttl.make_dfb("bfloat16", shape=(1, 2), block_count=3)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reconfigure_dfbs(boundary)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            ttl.call_extern_func(
                DFB_RECONFIGURATION_TEST_HEADER,
                "write_reconfiguration_runtime_value",
                template_args=[ttl.dfb_descriptor(runtime_dfb)],
                dfb_effects=[
                    ttl.DFBEffect.reserve(runtime_dfb, tiles=1),
                    ttl.DFBEffect.push(runtime_dfb, tiles=1),
                ],
            )
            ttl.reconfigure_dfbs(boundary)
            node_x, _ = ttl.node(dims=2)
            with second_dfb.reserve() as destination:
                ttl.copy(
                    second_input[0:1, node_x * 2 : node_x * 2 + 2], destination
                ).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, _ = ttl.node(dims=2)
            with runtime_dfb.wait() as source:
                ttl.copy(source, runtime_output[0, node_x]).wait()
            ttl.reconfigure_dfbs(boundary)
            with second_dfb.wait() as source:
                ttl.copy(source, second_output[0:1, node_x * 2 : node_x * 2 + 2]).wait()

    return caller_runtime_arg_reconfiguration_operation


def _to_device_with_dtype(torch_tensor, device, ttnn_dtype, memory_config):
    device_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if memory_config == ttnn.DRAM_MEMORY_CONFIG:
        return device_tensor
    return ttnn.to_memory_config(device_tensor, memory_config=memory_config)


_COMPUTE_FORMAT_CASES = (
    pytest.param("bfloat16", torch.bfloat16, id="bf16"),
    pytest.param("float32", torch.float32, id="f32"),
    pytest.param("bfloat8_b", torch.float32, id="bfp8-b"),
    pytest.param("bfloat4_b", torch.float32, id="bfp4-b"),
    pytest.param("uint32", torch.uint32, id="u32"),
    pytest.param("uint16", torch.uint16, id="u16"),
    pytest.param("int32", torch.int32, id="i32"),
)


def _native_format_host_tensor(shape, torch_dtype, offset):
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int64)
    values = values.reshape(shape)
    if torch_dtype.is_floating_point:
        return ((values.remainder(97) - 48).float() / 7 + offset).to(torch_dtype)
    if torch_dtype == torch.int32:
        return (values.remainder(97) - 48 + offset).to(torch_dtype)
    return (values.remainder(97) + offset).to(torch_dtype)


def _assert_output(actual, expected, dtype):
    tolerance = (0.05, 1.0) if dtype == torch.bfloat16 else (1e-5, 1e-6)
    assert_allclose(
        ttnn.to_torch(actual).float(),
        expected.float(),
        rtol=tolerance[0],
        atol=tolerance[1],
    )


# Compute-capable DFB formats preserve unpack and pack state after reconfiguration.
@pytest.mark.parametrize("data_format,torch_dtype", _COMPUTE_FORMAT_CASES)
@pytest.mark.parametrize(
    "memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram", "l1"],
)
def test_reconfiguration_supports_compute_dfb_formats(
    device, data_format, torch_dtype, memory_config, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    operation = _make_native_format_reconfiguration_operation(data_format)
    ttnn_dtype = format_name_to_ttnn_dtype(data_format, ttnn)
    mlir_file = tmp_path / f"compute_format_{data_format}.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(mlir_file))
    first_host = _native_format_host_tensor((32, 32), torch_dtype, 3)
    second_host = _native_format_host_tensor((32, 64), torch_dtype, 11)
    first_input = _to_device_with_dtype(first_host, device, ttnn_dtype, memory_config)
    second_input = _to_device_with_dtype(second_host, device, ttnn_dtype, memory_config)
    first_output = _to_device_with_dtype(
        torch.zeros_like(first_host), device, ttnn_dtype, memory_config
    )
    second_output = _to_device_with_dtype(
        torch.zeros_like(second_host), device, ttnn_dtype, memory_config
    )
    expected_first = ttnn.to_torch(first_input).float()
    expected_second = ttnn.to_torch(second_input).float()

    operation(
        first_input,
        first_output,
        second_input,
        second_output,
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = mlir_file.read_text()
    allocation_metadata = final_mlir.partition("ttl.dfb_reconfiguration_plan")[0]
    assert allocation_metadata.count("dfb_index = ") == 2
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3
    actual_first = ttnn.to_torch(first_output).float()
    actual_second = ttnn.to_torch(second_output).float()
    assert_pcc(expected_first.float(), actual_first.float(), 0.9999)
    assert_pcc(expected_second.float(), actual_second.float(), 0.9999)
    assert_allclose(actual_first, expected_first, rtol=0, atol=0)
    assert_allclose(actual_second, expected_second, rtol=0, atol=0)


# U8 reconfiguration is qualified only for data-movement DFB interfaces.
@pytest.mark.parametrize(
    "memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram", "l1"],
)
def test_reconfiguration_supports_data_movement_u8(device, memory_config):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    operation = _make_data_movement_format_reconfiguration_operation("uint8")
    first_host = _native_format_host_tensor((32, 32), torch.uint8, 3)
    second_host = _native_format_host_tensor((32, 64), torch.uint8, 11)
    first_input = _to_device_with_dtype(
        first_host, device, ttnn.DataType.UINT8, memory_config
    )
    second_input = _to_device_with_dtype(
        second_host, device, ttnn.DataType.UINT8, memory_config
    )
    first_output = _to_device_with_dtype(
        torch.zeros_like(first_host), device, ttnn.DataType.UINT8, memory_config
    )
    second_output = _to_device_with_dtype(
        torch.zeros_like(second_host), device, ttnn.DataType.UINT8, memory_config
    )

    operation(
        first_input,
        first_output,
        second_input,
        second_output,
        options="--ttl-reuse-user-dfbs",
    )

    assert_allclose(
        ttnn.to_torch(first_output).float(), first_host.float(), rtol=0, atol=0
    )
    assert_allclose(
        ttnn.to_torch(second_output).float(), second_host.float(), rtol=0, atol=0
    )


# Per-core caller arguments retain their indices before configuration addresses.
def test_reconfiguration_composes_varying_caller_runtime_args(device):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    operation = _make_caller_runtime_arg_reconfiguration_operation()
    runtime_expected = torch.cat(
        (
            torch.full((32, 32), 1.0, dtype=torch.bfloat16),
            torch.full((32, 32), 4.0, dtype=torch.bfloat16),
        ),
        dim=1,
    )
    second_host = _native_format_host_tensor((32, 128), torch.bfloat16, 11)
    runtime_output = to_dram(torch.zeros_like(runtime_expected), device)
    second_output = to_dram(torch.zeros_like(second_host), device)
    second_input = to_dram(second_host, device)

    operation(
        runtime_output,
        second_input,
        second_output,
        options="--ttl-reuse-user-dfbs",
    )

    runtime_actual = ttnn.to_torch(runtime_output).float()
    assert_pcc(runtime_expected.float(), runtime_actual.float(), 0.9999)
    assert_allclose(runtime_actual, runtime_expected.float(), rtol=0, atol=0)
    assert_allclose(
        ttnn.to_torch(second_output).float(), second_host.float(), rtol=0, atol=0
    )


# Repeated boundaries preserve a live payload, reuse physical indices, and
# restore the initial descriptors for the next iteration.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("grid_cols", [1, 2], ids=["one-core", "two-core"])
@pytest.mark.parametrize(
    "to_device",
    [to_dram, to_l1],
    ids=["dram", "l1"],
)
def test_reconfiguration_reuses_ids_with_different_capacity_and_cached_execution(
    device, dtype, grid_cols, to_device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_reconfiguration_operation(data_format, grid_cols)
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(tmp_path / "reconfiguration.mlir"))

    first_host = (
        torch.arange(32 * 32 * grid_cols, dtype=torch.float32)
        .reshape(32, 32 * grid_cols)
        .to(dtype)
    )
    second_host = (
        torch.arange(32 * 64 * grid_cols, dtype=torch.float32)
        .reshape(32, 64 * grid_cols)
        .remainder(257)
    ).to(dtype)
    third_host = (
        torch.arange(64 * 32 * grid_cols, dtype=torch.float32)
        .reshape(64, 32 * grid_cols)
        .remainder(193)
    ).to(dtype)
    first_output = to_device(torch.zeros_like(first_host), device)
    second_output = to_device(torch.zeros_like(second_host), device)
    third_output = to_device(torch.zeros_like(third_host), device)
    operation(
        to_device(first_host, device),
        first_output,
        to_device(second_host, device),
        second_output,
        to_device(third_host, device),
        third_output,
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = (tmp_path / "reconfiguration.mlir").read_text()
    allocation_metadata = final_mlir.partition("ttl.dfb_reconfiguration_plan")[0]
    assert allocation_metadata.count("dfb_index = ") == 2
    assert final_mlir.count("entry_reconfiguration = 0 : i64") == 2
    assert final_mlir.count("entry_reconfiguration = 1 : i64") == 2
    assert final_mlir.count("block_count = 4 : i32") == 2
    assert final_mlir.count("num_tiles = 2 : i32") == 4
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 6

    cached_first_host = (first_host.float() + 3).to(dtype)
    cached_second_host = (second_host.float() - 5).to(dtype)
    cached_third_host = (third_host.float() + 7).to(dtype)
    cached_first_output = to_device(torch.zeros_like(first_host), device)
    cached_second_output = to_device(torch.zeros_like(second_host), device)
    cached_third_output = to_device(torch.zeros_like(third_host), device)
    operation(
        to_device(cached_first_host, device),
        cached_first_output,
        to_device(cached_second_host, device),
        cached_second_output,
        to_device(cached_third_host, device),
        cached_third_output,
        options="--ttl-reuse-user-dfbs",
    )

    for actual, expected in (
        (first_output, first_host),
        (second_output, second_host),
        (third_output, third_host),
        (cached_first_output, cached_first_host),
        (cached_second_output, cached_second_host),
        (cached_third_output, cached_third_host),
    ):
        _assert_output(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("grid_cols", [1, 2], ids=["one-core", "two-core"])
def test_repeated_reconfiguration_reuses_ids_preserves_live_payload_and_restores_initial_epoch(
    device, dtype, to_device, grid_cols, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    iterations = 3
    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_repeated_reconfiguration_operation(
        data_format, iterations, grid_cols
    )
    mlir_file = tmp_path / "repeated_reconfiguration.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(mlir_file))
    first_host = (
        torch.arange(iterations * 32 * 32 * grid_cols, dtype=torch.float32)
        .reshape(iterations * 32, 32 * grid_cols)
        .to(dtype)
    )
    second_host = (
        torch.arange(iterations * 32 * 64 * grid_cols, dtype=torch.float32)
        .reshape(iterations * 32, 64 * grid_cols)
        .remainder(257)
        .to(dtype)
    )
    preserved_host = (second_host.float() + 17).to(dtype)
    first_output = to_device(torch.zeros_like(first_host), device)
    preserved_output = to_device(torch.zeros_like(preserved_host), device)
    second_output = to_device(torch.zeros_like(second_host), device)

    operation(
        to_device(first_host, device),
        first_output,
        to_device(preserved_host, device),
        preserved_output,
        to_device(second_host, device),
        second_output,
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = mlir_file.read_text()
    allocation_metadata = final_mlir.partition("ttl.dfb_reconfiguration_plan")[0]
    assert allocation_metadata.count("dfb_index = ") == 4
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 6
    for actual, expected in (
        (first_output, first_host),
        (preserved_output, preserved_host),
        (second_output, second_host),
    ):
        _assert_output(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_live_payload_crosses_reconfiguration_and_cached_execution(
    device, dtype, to_device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_live_crossing_operation(data_format)
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(tmp_path / "live_crossing.mlir"))

    initial_inputs = (
        torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32).to(dtype),
        torch.arange(32 * 32, dtype=torch.float32)
        .reshape(32, 32)
        .remainder(251)
        .to(dtype),
        torch.arange(32 * 64, dtype=torch.float32)
        .reshape(32, 64)
        .remainder(197)
        .to(dtype),
    )
    initial_outputs = tuple(
        to_device(torch.zeros_like(input_tensor), device)
        for input_tensor in initial_inputs
    )
    operation(
        to_device(initial_inputs[0], device),
        initial_outputs[0],
        to_device(initial_inputs[1], device),
        initial_outputs[1],
        to_device(initial_inputs[2], device),
        initial_outputs[2],
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = (tmp_path / "live_crossing.mlir").read_text()
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3
    assert "ttl.dfb_reconfiguration_plan" in final_mlir
    for actual, expected in zip(initial_outputs, initial_inputs):
        _assert_output(actual, expected, dtype)

    cached_inputs = tuple(
        (input_tensor.float() + offset).to(dtype)
        for input_tensor, offset in zip(initial_inputs, (3, -5, 7))
    )
    cached_outputs = tuple(
        to_device(torch.zeros_like(input_tensor), device)
        for input_tensor in cached_inputs
    )
    operation(
        to_device(cached_inputs[0], device),
        cached_outputs[0],
        to_device(cached_inputs[1], device),
        cached_outputs[1],
        to_device(cached_inputs[2], device),
        cached_outputs[2],
        options="--ttl-reuse-user-dfbs",
    )

    for actual, expected in zip(cached_outputs, cached_inputs):
        _assert_output(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("enabled_column", [0, 1], ids=["left", "right"])
def test_conditional_reconfiguration_executes_with_post_boundary_dfbs(
    device,
    dtype,
    to_device,
    enabled_column,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_conditional_reconfiguration_operation(data_format, enabled_column)
    monkeypatch.setenv(
        "TTLANG_FINAL_MLIR",
        str(tmp_path / f"conditional_{enabled_column}.mlir"),
    )
    input_host = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64).to(dtype)
    output = to_device(torch.zeros_like(input_host), device)
    operation(
        to_device(input_host, device),
        output,
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = (tmp_path / f"conditional_{enabled_column}.mlir").read_text()
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3
    expected = torch.zeros_like(input_host)
    column_start = enabled_column * 32
    expected[:, column_start : column_start + 32] = input_host[
        :, column_start : column_start + 32
    ]
    _assert_output(output, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_dispatch_condition_reconfiguration_executes_active_and_inactive(
    device,
    dtype,
    to_device,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    for active_value in (1, 0):
        operation = _make_dispatch_condition_reconfiguration_operation(
            data_format, active_value
        )
        mlir_file = tmp_path / f"dispatch_condition_{active_value}.mlir"
        monkeypatch.setenv("TTLANG_FINAL_MLIR", str(mlir_file))
        for invocation in range(2):
            input_host = (
                torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
                + invocation * 7
            ).to(dtype)
            output = to_device(torch.zeros_like(input_host), device)
            operation(
                to_device(input_host, device),
                output,
                options="--ttl-reuse-user-dfbs",
            )
            expected = input_host if active_value else torch.zeros_like(input_host)
            _assert_output(output, expected, dtype)

        final_mlir = mlir_file.read_text()
        assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3
        assert final_mlir.count("scalar_predicate") == 3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_reconfiguration_executes_with_physical_indices_above_31(
    device,
    dtype,
    to_device,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires the Blackhole 64-index DFB capacity")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_high_index_reconfiguration_operation(data_format)
    mlir_file = tmp_path / "high_index_reconfiguration.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(mlir_file))
    input_host = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32).to(dtype)
    output = to_device(torch.zeros_like(input_host), device)
    operation(
        to_device(input_host, device),
        output,
        options="--no-ttl-reuse-user-dfbs",
    )

    final_mlir = mlir_file.read_text()
    assert "dfb_index = 32 : i32" in final_mlir
    assert "dfb_index = 33 : i32" in final_mlir
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3
    _assert_output(output, input_host, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_reconfiguration_switches_tensor_backed_storage_and_cached_execution(
    device,
    dtype,
    to_device,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_tensor_backed_reconfiguration_operation(data_format)
    mlir_file = tmp_path / "tensor_backed_reconfiguration.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(mlir_file))

    for invocation in range(2):
        tensor_backed_host = (
            torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32) + invocation * 3
        ).to(dtype)
        scratch_host = (
            torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
            + invocation * 5
            + 11
        ).to(dtype)
        tensor_backed_output = to_device(torch.zeros_like(tensor_backed_host), device)
        scratch_output = to_device(torch.zeros_like(scratch_host), device)
        operation(
            to_l1_sharded(tensor_backed_host, device, layout="height"),
            tensor_backed_output,
            to_device(scratch_host, device),
            scratch_output,
            options="--ttl-reuse-user-dfbs",
        )
        _assert_output(tensor_backed_output, tensor_backed_host, dtype)
        _assert_output(scratch_output, scratch_host, dtype)

    final_mlir = mlir_file.read_text()
    allocation_metadata = final_mlir.partition("ttl.dfb_reconfiguration_plan")[0]
    assert allocation_metadata.count("dfb_index = ") == 2
    assert "tensor_index = 0" in final_mlir
    assert final_mlir.count("entry_reconfiguration = 0 : i64") == 2
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3


# A later-active core receives static placeholder storage before reconfiguration.
def test_reconfiguration_enables_later_tensor_backed_core(
    device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    operation = _make_cross_core_tensor_backed_reconfiguration_operation()
    mlir_file = tmp_path / "cross_core_tensor_backed_reconfiguration.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(mlir_file))
    input_host = (
        torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32).to(torch.bfloat16)
    )
    shard_ranges = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}
    )
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_ranges,
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    input_tensor = ttnn.to_memory_config(
        to_dram(input_host, device), memory_config=input_memory_config
    )
    output = to_dram(torch.zeros_like(input_host), device)
    operation(
        input_tensor,
        output,
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = mlir_file.read_text()
    allocation_metadata = final_mlir.partition("ttl.dfb_reconfiguration_plan")[0]
    assert allocation_metadata.count("dfb_index = ") == 1
    actual = ttnn.to_torch(output).float()
    assert_pcc(input_host.float(), actual, 0.9999)
    assert_allclose(actual, input_host.float(), rtol=0, atol=0)
