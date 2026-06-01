# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
#
# RUN: env TTLANG_FINAL_MLIR=%t.final.mlir timeout 180 %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=FINAL < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=RUNTIME < %t.output

"""Runtime coverage for liveness-based PipeNet resource allocation.

This test is derived from the reproducer in
https://github.com/tenstorrent/tt-lang/issues/625. The report stated that
either PipeNet delivery route alone completed, while enabling both routes
deadlocked.

The runtime RUN uses GRID_DIM=2, the original small issue-625 reproducer. It
keeps the same row/column/helper PipeNet structure, both-route semantics,
float32 tensors, and compute-side DFB waits. It fixes the schedule by posting
loopback receives before sending and by popping send DFB blocks before reusing
them. Each node writes one result tile per K-pair, which verifies both
successful kernel execution and the received row/column payload values.

The input is a GRID_DIM x TRANSFER_K_TILES tile grid. Each 32x32 tile is
constant:

  input_tile[source_row, k_tile] = source_row * TRANSFER_K_TILES + k_tile + 1

The output is an OUTPUT_K_PAIRS*GRID_DIM x (GRID_DIM+1) tile grid. Each output
tile is the sum of the row-route tile and column-route tile received by that
node. For the runtime GRID_DIM=2 run, the expected output tile values are:

  4 6 2
  4 8 6
"""

import contextlib  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402
import ttnn  # noqa: E402

import ttl  # noqa: E402
from ttlang_test_utils import assert_allclose, to_dram  # noqa: E402

TILE = 32
# TODO(#628): increase toward 7 after PipeNet data-movement lowering stops
# duplicating transfer bodies per participating coordinate.
GRID_DIM = 2


def get_transfer_k_tiles(grid_dim):
    return 2 * (grid_dim // 2)


def get_output_k_pairs(grid_dim):
    return get_transfer_k_tiles(grid_dim) // 2


# GRID_DIM=7 emits enough TTKernel code to exceed TT-Metal's default 90112-byte
# Tensix kernel config buffer.
KERNEL_CONFIG_BUFFER_RESERVE_BYTES = 128 * 1024


def make_ksplit_resource_allocation_kernel(grid_dim):
    row_upper_net = ttl.PipeNet(
        [
            ttl.Pipe((0, row_idx), (slice(row_idx, grid_dim), row_idx))
            for row_idx in range(grid_dim)
        ]
    )
    row_lower_net = ttl.PipeNet(
        [
            ttl.Pipe((0, row_idx), (slice(0, row_idx), row_idx))
            for row_idx in range(1, grid_dim)
        ]
    )
    col_upper_net = ttl.PipeNet(
        [
            ttl.Pipe(
                (col_idx, 0),
                (col_idx, slice(0, col_idx + 1)),
            )
            for col_idx in range(grid_dim)
        ]
    )
    col_lower_net = ttl.PipeNet(
        [
            ttl.Pipe(
                (col_idx, 0),
                (col_idx, slice(col_idx + 1, grid_dim)),
            )
            for col_idx in range(0, grid_dim - 1)
        ]
    )
    helper_row_even_net = ttl.PipeNet(
        [ttl.Pipe((0, row_idx), (grid_dim, row_idx)) for row_idx in range(grid_dim)]
    )
    helper_col_even_net = ttl.PipeNet(
        [ttl.Pipe((row_idx, 0), (grid_dim, row_idx)) for row_idx in range(grid_dim)]
    )

    @ttl.operation(grid=(grid_dim + 1, grid_dim), fp32_dest_acc_en=True)
    def ksplit_resource_allocation(input_tensor, output_tensor):
        _row_upper_net = row_upper_net
        _row_lower_net = row_lower_net
        _col_upper_net = col_upper_net
        _col_lower_net = col_lower_net
        _helper_row_even_net = helper_row_even_net
        _helper_col_even_net = helper_col_even_net

        half_k = input_tensor.shape[1] // (2 * TILE)
        tile11 = (1, 1)
        row_recv_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=tile11, block_count=half_k
        )
        col_recv_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=tile11, block_count=half_k
        )
        row_send_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=tile11, block_count=2
        )
        col_send_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=tile11, block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            output_tensor, shape=tile11, block_count=2
        )

        @ttl.compute()
        def compute():
            for _ in range(half_k):
                with (
                    col_recv_dfb.wait() as col_recv_blk,
                    row_recv_dfb.wait() as row_recv_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    out_blk.store(row_recv_blk + col_recv_blk)

        @ttl.datamovement()
        def post_receives_and_send():
            node_x, node_y = ttl.node(dims=2)
            for k_pair in range(half_k):
                even_k = 2 * k_pair
                odd_k = even_k + 1

                def recv_row(pipe):
                    ttl.copy(pipe, row_recv_blk).wait()

                def recv_col(pipe):
                    ttl.copy(pipe, col_recv_blk).wait()

                if row_lower_net.is_src():
                    with row_send_dfb.reserve() as row_send_blk:
                        ttl.copy(
                            input_tensor[node_y : node_y + 1, even_k : even_k + 1],
                            row_send_blk,
                        ).wait()

                    with row_send_dfb.wait() as row_send_blk:

                        def send_row(pipe):
                            ttl.copy(row_send_blk, pipe).wait()

                        if row_lower_net.is_dst():
                            with row_recv_dfb.reserve() as row_recv_blk:

                                def recv_row_then_send(pipe):
                                    recv_tx = ttl.copy(pipe, row_recv_blk)
                                    row_lower_net.if_src(send_row)
                                    helper_row_even_net.if_src(send_row)
                                    recv_tx.wait()

                                row_lower_net.if_dst(recv_row_then_send)
                        else:
                            row_lower_net.if_src(send_row)
                            helper_row_even_net.if_src(send_row)
                elif helper_row_even_net.is_src():
                    with row_send_dfb.reserve() as row_send_blk:
                        ttl.copy(
                            input_tensor[node_y : node_y + 1, even_k : even_k + 1],
                            row_send_blk,
                        ).wait()

                    with row_send_dfb.wait() as row_send_blk:

                        def send_row(pipe):
                            ttl.copy(row_send_blk, pipe).wait()

                        helper_row_even_net.if_src(send_row)
                elif row_lower_net.is_dst():
                    with row_recv_dfb.reserve() as row_recv_blk:
                        row_lower_net.if_dst(recv_row)
                elif helper_row_even_net.is_dst():
                    with row_recv_dfb.reserve() as row_recv_blk:
                        helper_row_even_net.if_dst(recv_row)

                if col_lower_net.is_src():
                    with col_send_dfb.reserve() as col_send_blk:
                        ttl.copy(
                            input_tensor[node_x : node_x + 1, even_k : even_k + 1],
                            col_send_blk,
                        ).wait()

                    with col_send_dfb.wait() as col_send_blk:

                        def send_col(pipe):
                            ttl.copy(col_send_blk, pipe).wait()

                        if col_lower_net.is_dst():
                            with col_recv_dfb.reserve() as col_recv_blk:

                                def recv_col_then_send(pipe):
                                    recv_tx = ttl.copy(pipe, col_recv_blk)
                                    col_lower_net.if_src(send_col)
                                    helper_col_even_net.if_src(send_col)
                                    recv_tx.wait()

                                col_lower_net.if_dst(recv_col_then_send)
                        else:
                            col_lower_net.if_src(send_col)
                            helper_col_even_net.if_src(send_col)
                elif helper_col_even_net.is_src():
                    with col_send_dfb.reserve() as col_send_blk:
                        ttl.copy(
                            input_tensor[node_x : node_x + 1, even_k : even_k + 1],
                            col_send_blk,
                        ).wait()

                    with col_send_dfb.wait() as col_send_blk:

                        def send_col(pipe):
                            ttl.copy(col_send_blk, pipe).wait()

                        helper_col_even_net.if_src(send_col)
                elif col_lower_net.is_dst():
                    with col_recv_dfb.reserve() as col_recv_blk:
                        col_lower_net.if_dst(recv_col)
                elif helper_col_even_net.is_dst():
                    with col_recv_dfb.reserve() as col_recv_blk:
                        helper_col_even_net.if_dst(recv_col)

                if row_upper_net.is_src():
                    with row_send_dfb.reserve() as row_send_blk:
                        ttl.copy(
                            input_tensor[node_y : node_y + 1, odd_k : odd_k + 1],
                            row_send_blk,
                        ).wait()

                    with row_send_dfb.wait() as row_send_blk:

                        def send_row(pipe):
                            ttl.copy(row_send_blk, pipe).wait()

                        if row_upper_net.is_dst():
                            with row_recv_dfb.reserve() as row_recv_blk:

                                def recv_row_then_send(pipe):
                                    recv_tx = ttl.copy(pipe, row_recv_blk)
                                    row_upper_net.if_src(send_row)
                                    recv_tx.wait()

                                row_upper_net.if_dst(recv_row_then_send)
                        else:
                            row_upper_net.if_src(send_row)
                elif row_upper_net.is_dst():
                    with row_recv_dfb.reserve() as row_recv_blk:
                        row_upper_net.if_dst(recv_row)

                if col_upper_net.is_src():
                    with col_send_dfb.reserve() as col_send_blk:
                        ttl.copy(
                            input_tensor[node_x : node_x + 1, odd_k : odd_k + 1],
                            col_send_blk,
                        ).wait()

                    with col_send_dfb.wait() as col_send_blk:

                        def send_col(pipe):
                            ttl.copy(col_send_blk, pipe).wait()

                        if col_upper_net.is_dst():
                            with col_recv_dfb.reserve() as col_recv_blk:

                                def recv_col_then_send(pipe):
                                    recv_tx = ttl.copy(pipe, col_recv_blk)
                                    col_upper_net.if_src(send_col)
                                    recv_tx.wait()

                                col_upper_net.if_dst(recv_col_then_send)
                        else:
                            col_upper_net.if_src(send_col)
                elif col_upper_net.is_dst():
                    with col_recv_dfb.reserve() as col_recv_blk:
                        col_upper_net.if_dst(recv_col)

        @ttl.datamovement()
        def write_output():
            node_x, node_y = ttl.node(dims=2)
            for k_pair in range(half_k):
                out_row = k_pair * grid_dim + node_y
                with out_dfb.wait() as out_blk:
                    ttl.copy(
                        out_blk,
                        output_tensor[out_row : out_row + 1, node_x : node_x + 1],
                    ).wait()

    return ksplit_resource_allocation


# FINAL-LABEL: module attributes
# FINAL-SAME: ttl.pipe_sram_scratch_bytes = 64 : i64
# FINAL-SAME: ttl.pipe_sync_semaphore_count = 11 : i64
# FINAL-NOT: ttl.pipe_global_semaphore_count
#
# CHECK-CPP-LABEL: // post_receives_and_send
# CHECK-CPP-DAG: {{(size_t|int32_t)}} [[READY:v[0-9]+]] = 10;
# CHECK-CPP: noc_inline_dw_write
# CHECK-CPP: get_semaphore([[READY]])
# CHECK-CPP: reinterpret_cast<tt_l1_ptr uint32_t*>
# CHECK-CPP: experimental::semaphore_wait
# CHECK-CPP: {{noc\.async_write\(}}
# CHECK-CPP: noc_semaphore_inc
#
# RUNTIME: PASS: ksplit_resource_allocation result verified
def make_expected_output(input_torch, grid_dim):
    output_k_pairs = get_output_k_pairs(grid_dim)
    output_torch = torch.zeros(
        output_k_pairs * grid_dim * TILE,
        (grid_dim + 1) * TILE,
        dtype=input_torch.dtype,
    )
    for k_pair in range(output_k_pairs):
        even_k = 2 * k_pair
        odd_k = even_k + 1
        for node_y in range(grid_dim):
            for node_x in range(grid_dim + 1):
                row_k = even_k if node_x < node_y or node_x == grid_dim else odd_k
                col_k = even_k if node_x == grid_dim or node_y > node_x else odd_k
                col_source_row = node_y if node_x == grid_dim else node_x

                row_tile = input_torch[
                    node_y * TILE : (node_y + 1) * TILE,
                    row_k * TILE : (row_k + 1) * TILE,
                ]
                col_tile = input_torch[
                    col_source_row * TILE : (col_source_row + 1) * TILE,
                    col_k * TILE : (col_k + 1) * TILE,
                ]
                output_row = k_pair * grid_dim + node_y
                output_torch[
                    output_row * TILE : (output_row + 1) * TILE,
                    node_x * TILE : (node_x + 1) * TILE,
                ] = (
                    row_tile + col_tile
                )
    return output_torch


def make_input_tensor(grid_dim):
    transfer_k_tiles = get_transfer_k_tiles(grid_dim)
    input_torch = torch.empty(
        grid_dim * TILE,
        transfer_k_tiles * TILE,
        dtype=torch.float32,
    )
    for source_row in range(grid_dim):
        for k_tile in range(transfer_k_tiles):
            tile_value = source_row * transfer_k_tiles + k_tile + 1
            input_torch[
                source_row * TILE : (source_row + 1) * TILE,
                k_tile * TILE : (k_tile + 1) * TILE,
            ] = tile_value
    return input_torch


def open_reproducer_device():
    if hasattr(ttnn, "device") and hasattr(
        ttnn.device, "get_max_worker_l1_unreserved_size"
    ):
        default_size = ttnn.device.get_max_worker_l1_unreserved_size()
        return ttnn.open_device(
            device_id=0,
            worker_l1_size=default_size - KERNEL_CONFIG_BUFFER_RESERVE_BYTES,
        )
    return ttnn.open_device(device_id=0)


def main():
    device = open_reproducer_device()
    try:
        grid_dim = GRID_DIM
        output_k_pairs = get_output_k_pairs(grid_dim)
        ksplit_resource_allocation = make_ksplit_resource_allocation_kernel(grid_dim)
        input_torch = make_input_tensor(grid_dim)
        output_torch = torch.zeros(
            output_k_pairs * grid_dim * TILE,
            (grid_dim + 1) * TILE,
            dtype=torch.float32,
        )

        input_tensor = to_dram(input_torch, device)
        output_tensor = to_dram(output_torch, device)
        output_context = (
            contextlib.redirect_stdout(io.StringIO())
            if os.environ.get("TTLANG_SUPPRESS_KERNEL_OUTPUT") == "1"
            else contextlib.nullcontext()
        )
        with output_context:
            ksplit_resource_allocation(input_tensor, output_tensor)

        ttnn.synchronize_device(device)
        result_torch = ttnn.to_torch(output_tensor).float()
        expected_torch = make_expected_output(input_torch, grid_dim).float()
        assert_allclose(result_torch, expected_torch, rtol=0.0, atol=0.0)
        print("PASS: ksplit_resource_allocation result verified")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
