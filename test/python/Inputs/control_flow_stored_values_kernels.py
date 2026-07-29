# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared control-flow store kernels for lit and pytest coverage."""

import os

import torch
import ttl


def host_tensor(shape):
    return torch.zeros(shape, dtype=torch.bfloat16)


def _is_compile_only():
    return os.environ.get("TTLANG_COMPILE_ONLY") == "1"


@ttl.operation(grid=(2, 1))
def then_only_store_kernel(input_tensor, output_tensor):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
            if node_x == 0:
                with output_dfb.reserve() as output_block:
                    output_block.store(value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with output_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x == 0:
                with output_dfb.wait() as output_block:
                    ttl.copy(output_block, output_tensor[node_y, 0]).wait()


@ttl.operation(grid=(2, 1))
def else_only_store_kernel(input_tensor, output_tensor):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
            if node_x != 0:
                with output_dfb.reserve() as output_block:
                    output_block.store(value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x != 0:
                with output_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x != 0:
                with output_dfb.wait() as output_block:
                    ttl.copy(output_block, output_tensor[node_y, 0]).wait()


@ttl.operation(grid=(2, 1))
def if_else_stored_value_kernel(input_tensor, then_output, else_output):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    then_dfb = ttl.make_dataflow_buffer_like(then_output, shape=(1, 1), block_count=2)
    else_dfb = ttl.make_dataflow_buffer_like(else_output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
            if node_x == 0:
                with then_dfb.reserve() as output_block:
                    output_block.store(value)
            else:
                with else_dfb.reserve() as output_block:
                    output_block.store(value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with then_dfb.wait() as _output_block:
                    pass
            else:
                with else_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x == 0:
                with then_dfb.wait() as output_block:
                    ttl.copy(output_block, then_output[node_y, 0]).wait()
            else:
                with else_dfb.wait() as output_block:
                    ttl.copy(output_block, else_output[node_y, 0]).wait()


@ttl.operation(grid=(2, 1))
def released_input_stored_value_kernel(input_tensor, then_output, else_output):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    then_dfb = ttl.make_dataflow_buffer_like(then_output, shape=(1, 1), block_count=2)
    else_dfb = ttl.make_dataflow_buffer_like(else_output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
        if node_x == 0:
            with then_dfb.reserve() as output_block:
                output_block.store(value)
        else:
            with else_dfb.reserve() as output_block:
                output_block.store(value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with then_dfb.wait() as _output_block:
                    pass
            else:
                with else_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x == 0:
                with then_dfb.wait() as output_block:
                    ttl.copy(output_block, then_output[node_y, 0]).wait()
            else:
                with else_dfb.wait() as output_block:
                    ttl.copy(output_block, else_output[node_y, 0]).wait()


@ttl.operation(grid=(3, 1))
def elif_chain_stored_value_kernel(
    input_tensor, first_output, second_output, third_output
):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    first_dfb = ttl.make_dataflow_buffer_like(first_output, shape=(1, 1), block_count=2)
    second_dfb = ttl.make_dataflow_buffer_like(
        second_output, shape=(1, 1), block_count=2
    )
    third_dfb = ttl.make_dataflow_buffer_like(third_output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
            if node_x == 0:
                with first_dfb.reserve() as output_block:
                    output_block.store(value)
            elif node_x == 1:
                with second_dfb.reserve() as output_block:
                    output_block.store(value)
            else:
                with third_dfb.reserve() as output_block:
                    output_block.store(value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_dfb.wait() as _output_block:
                    pass
            elif node_x == 1:
                with second_dfb.wait() as _output_block:
                    pass
            else:
                with third_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_dfb.wait() as output_block:
                    ttl.copy(output_block, first_output[node_y, 0]).wait()
            elif node_x == 1:
                with second_dfb.wait() as output_block:
                    ttl.copy(output_block, second_output[node_y, 0]).wait()
            else:
                with third_dfb.wait() as output_block:
                    ttl.copy(output_block, third_output[node_y, 0]).wait()


@ttl.operation(grid=(3, 1))
def elif_gap_stored_value_kernel(input_tensor, first_output, third_output):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    first_dfb = ttl.make_dataflow_buffer_like(first_output, shape=(1, 1), block_count=2)
    third_dfb = ttl.make_dataflow_buffer_like(third_output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
            if node_x == 0:
                with first_dfb.reserve() as output_block:
                    output_block.store(value)
            elif node_x == 2:
                with third_dfb.reserve() as output_block:
                    output_block.store(value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_dfb.wait() as _output_block:
                    pass
            elif node_x == 2:
                with third_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_dfb.wait() as output_block:
                    ttl.copy(output_block, first_output[node_y, 0]).wait()
            elif node_x == 2:
                with third_dfb.wait() as output_block:
                    ttl.copy(output_block, third_output[node_y, 0]).wait()


@ttl.operation(grid=(3, 1))
def nested_if_stored_value_kernel(
    input_tensor, first_output, second_output, third_output
):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    first_dfb = ttl.make_dataflow_buffer_like(first_output, shape=(1, 1), block_count=2)
    second_dfb = ttl.make_dataflow_buffer_like(
        second_output, shape=(1, 1), block_count=2
    )
    third_dfb = ttl.make_dataflow_buffer_like(third_output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
            if node_x < 2:
                if node_x == 0:
                    with first_dfb.reserve() as output_block:
                        output_block.store(value)
                else:
                    with second_dfb.reserve() as output_block:
                        output_block.store(value)
            else:
                with third_dfb.reserve() as output_block:
                    output_block.store(value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x < 2:
                if node_x == 0:
                    with first_dfb.wait() as _output_block:
                        pass
                else:
                    with second_dfb.wait() as _output_block:
                        pass
            else:
                with third_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x < 2:
                if node_x == 0:
                    with first_dfb.wait() as output_block:
                        ttl.copy(output_block, first_output[node_y, 0]).wait()
                else:
                    with second_dfb.wait() as output_block:
                        ttl.copy(output_block, second_output[node_y, 0]).wait()
            else:
                with third_dfb.wait() as output_block:
                    ttl.copy(output_block, third_output[node_y, 0]).wait()


@ttl.operation(grid=(3, 1))
def sibling_if_stored_value_kernel(
    input_tensor, first_output, second_output, third_output
):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    first_dfb = ttl.make_dataflow_buffer_like(first_output, shape=(1, 1), block_count=2)
    second_dfb = ttl.make_dataflow_buffer_like(
        second_output, shape=(1, 1), block_count=2
    )
    third_dfb = ttl.make_dataflow_buffer_like(third_output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
            if node_x == 0:
                with first_dfb.reserve() as output_block:
                    output_block.store(value)
            if node_x == 1:
                with second_dfb.reserve() as output_block:
                    output_block.store(value)
            if node_x == 2:
                with third_dfb.reserve() as output_block:
                    output_block.store(value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_dfb.wait() as _output_block:
                    pass
            if node_x == 1:
                with second_dfb.wait() as _output_block:
                    pass
            if node_x == 2:
                with third_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_dfb.wait() as output_block:
                    ttl.copy(output_block, first_output[node_y, 0]).wait()
            if node_x == 1:
                with second_dfb.wait() as output_block:
                    ttl.copy(output_block, second_output[node_y, 0]).wait()
            if node_x == 2:
                with third_dfb.wait() as output_block:
                    ttl.copy(output_block, third_output[node_y, 0]).wait()


@ttl.operation(grid=(3, 1))
def nested_def_stored_value_kernel(
    input_tensor, first_output, second_output, third_output
):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    first_dfb = ttl.make_dataflow_buffer_like(first_output, shape=(1, 1), block_count=2)
    second_dfb = ttl.make_dataflow_buffer_like(
        second_output, shape=(1, 1), block_count=2
    )
    third_dfb = ttl.make_dataflow_buffer_like(third_output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            if node_x < 2:
                value = ttl.exp(input_block)
                if node_x == 0:
                    with first_dfb.reserve() as output_block:
                        output_block.store(value)
                else:
                    with second_dfb.reserve() as output_block:
                        output_block.store(value)
            else:
                fallback = ttl.neg(input_block)
                with third_dfb.reserve() as output_block:
                    output_block.store(fallback)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x < 2:
                if node_x == 0:
                    with first_dfb.wait() as _output_block:
                        pass
                else:
                    with second_dfb.wait() as _output_block:
                        pass
            else:
                with third_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x < 2:
                if node_x == 0:
                    with first_dfb.wait() as output_block:
                        ttl.copy(output_block, first_output[node_y, 0]).wait()
                else:
                    with second_dfb.wait() as output_block:
                        ttl.copy(output_block, second_output[node_y, 0]).wait()
            else:
                with third_dfb.wait() as output_block:
                    ttl.copy(output_block, third_output[node_y, 0]).wait()


@ttl.operation(grid=(2, 1))
def loop_wrapped_stored_value_kernel(input_tensor, first_output, second_output):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    first_dfb = ttl.make_dataflow_buffer_like(first_output, shape=(1, 1), block_count=2)
    second_dfb = ttl.make_dataflow_buffer_like(
        second_output, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
            for _iteration in range(2):
                if node_x == 0:
                    with first_dfb.reserve() as output_block:
                        output_block.store(value)
                else:
                    with second_dfb.reserve() as output_block:
                        output_block.store(value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            for _iteration in range(2):
                if node_x == 0:
                    with first_dfb.wait() as _output_block:
                        pass
                else:
                    with second_dfb.wait() as _output_block:
                        pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            for _iteration in range(2):
                if node_x == 0:
                    with first_dfb.wait() as output_block:
                        ttl.copy(output_block, first_output[node_y, 0]).wait()
                else:
                    with second_dfb.wait() as output_block:
                        ttl.copy(output_block, second_output[node_y, 0]).wait()


@ttl.operation(grid=(2, 1))
def external_use_stored_value_kernel(
    input_tensor, first_output, second_output, side_output
):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    first_dfb = ttl.make_dataflow_buffer_like(first_output, shape=(1, 1), block_count=2)
    second_dfb = ttl.make_dataflow_buffer_like(
        second_output, shape=(1, 1), block_count=2
    )
    side_dfb = ttl.make_dataflow_buffer_like(side_output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
            if node_x == 0:
                with first_dfb.reserve() as output_block:
                    output_block.store(value)
            else:
                with second_dfb.reserve() as output_block:
                    output_block.store(value)

            side_value = ttl.neg(value)
            with side_dfb.reserve() as output_block:
                output_block.store(side_value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_dfb.wait() as _output_block:
                    pass
            else:
                with second_dfb.wait() as _output_block:
                    pass
            with side_dfb.wait() as _output_block:
                pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_dfb.wait() as output_block:
                    ttl.copy(output_block, first_output[node_y, 0]).wait()
            else:
                with second_dfb.wait() as output_block:
                    ttl.copy(output_block, second_output[node_y, 0]).wait()

            with side_dfb.wait() as output_block:
                ttl.copy(output_block, side_output[node_y, node_x]).wait()


@ttl.operation(grid=(2, 1))
def parent_and_branch_stored_value_kernel(input_tensor, always_output, branch_output):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    always_dfb = ttl.make_dataflow_buffer_like(
        always_output, shape=(1, 1), block_count=2
    )
    branch_dfb = ttl.make_dataflow_buffer_like(
        branch_output, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            value = ttl.exp(input_block)
            with always_dfb.reserve() as output_block:
                output_block.store(value)
            if node_x == 0:
                with branch_dfb.reserve() as output_block:
                    output_block.store(value)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            with always_dfb.wait() as _output_block:
                pass
            if node_x == 0:
                with branch_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            with always_dfb.wait() as output_block:
                ttl.copy(output_block, always_output[node_y, node_x]).wait()
            if node_x == 0:
                with branch_dfb.wait() as output_block:
                    ttl.copy(output_block, branch_output[node_y, 0]).wait()


@ttl.operation(grid=(2, 1))
def attached_input_stored_value_kernel(input_tensor, first_output, second_output):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    first_dfb = ttl.make_dataflow_buffer_like(first_output, shape=(1, 1), block_count=2)
    second_dfb = ttl.make_dataflow_buffer_like(
        second_output, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with input_dfb.wait() as input_block:
            if node_x == 0:
                with first_dfb.reserve() as output_block:
                    output_block.store(input_block)
            else:
                with second_dfb.reserve() as output_block:
                    output_block.store(input_block)

    if _is_compile_only():

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as _input_block:
                pass

        @ttl.datamovement()
        def dm_write():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_dfb.wait() as _output_block:
                    pass
            else:
                with second_dfb.wait() as _output_block:
                    pass

    else:

        @ttl.datamovement()
        def dm_read():
            node_x, node_y = ttl.node(dims=2)
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[node_y, node_x], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            node_x, node_y = ttl.node(dims=2)
            if node_x == 0:
                with first_dfb.wait() as output_block:
                    ttl.copy(output_block, first_output[node_y, 0]).wait()
            else:
                with second_dfb.wait() as output_block:
                    ttl.copy(output_block, second_output[node_y, 0]).wait()


CONTROL_FLOW_CASES = [
    ("then_only", then_only_store_kernel, 2, 1),
    ("else_only", else_only_store_kernel, 2, 1),
    ("if_else", if_else_stored_value_kernel, 2, 2),
    ("released_input", released_input_stored_value_kernel, 2, 2),
    ("elif_chain", elif_chain_stored_value_kernel, 3, 3),
    ("elif_gap", elif_gap_stored_value_kernel, 3, 2),
    ("nested_if", nested_if_stored_value_kernel, 3, 3),
    ("sibling_ifs", sibling_if_stored_value_kernel, 3, 3),
    ("nested_def", nested_def_stored_value_kernel, 3, 3),
    ("loop_wrapped", loop_wrapped_stored_value_kernel, 2, 2),
    ("external_use", external_use_stored_value_kernel, 2, 3),
    ("parent_and_branch", parent_and_branch_stored_value_kernel, 2, 2),
    ("attached_input", attached_input_stored_value_kernel, 2, 2),
]

RUNTIME_CASES = [
    ("if_else", if_else_stored_value_kernel, 2, 2),
    ("released_input", released_input_stored_value_kernel, 2, 2),
    ("elif_chain", elif_chain_stored_value_kernel, 3, 3),
    ("nested_if", nested_if_stored_value_kernel, 3, 3),
    ("sibling_ifs", sibling_if_stored_value_kernel, 3, 3),
    ("loop_wrapped", loop_wrapped_stored_value_kernel, 2, 2),
]

DFB_FALLBACK_RUNTIME_CASES = [
    ("released_input", released_input_stored_value_kernel, 2, 2),
    ("sibling_ifs", sibling_if_stored_value_kernel, 3, 3),
    ("loop_wrapped", loop_wrapped_stored_value_kernel, 2, 2),
]

SINGLE_BRANCH_RUNTIME_CASES = [
    ("then_only", then_only_store_kernel, 2, 0),
    ("else_only", else_only_store_kernel, 2, 1),
]
