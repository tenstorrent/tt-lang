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

The test uses GRID_DIM=7, which launches on an 8x7 worker grid. It transfers
six K tiles as three even/odd pairs, keeping the same row/column/helper PipeNet
structure, both-route semantics, float32 tensors, and compute-side DFB waits.
It fixes the schedule by posting loopback receives before sending and by popping
send DFB blocks before reusing them. The operation has no numeric output;
verification is successful kernel execution and device synchronization, which
detects invalid generated synchronization code or deadlock.
"""

import torch  # noqa: E402
import ttnn  # noqa: E402

import ttl  # noqa: E402
from ttlang_test_utils import to_dram  # noqa: E402


TILE = 32
GRID_DIM = 7
TRANSFER_K_TILES = 2 * (GRID_DIM // 2)
# GRID_DIM=7 emits enough TTKernel code to exceed TT-Metal's default 90112-byte
# Tensix kernel config buffer.
KERNEL_CONFIG_BUFFER_RESERVE_BYTES = 128 * 1024


def make_ksplit_resource_allocation_kernel():
    grid_dim = GRID_DIM
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

        @ttl.compute()
        def compute():
            for _ in range(half_k):
                with (
                    col_recv_dfb.wait() as _col_recv_blk,
                    row_recv_dfb.wait() as _row_recv_blk,
                ):
                    pass

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
            pass

    return ksplit_resource_allocation


# FINAL-LABEL: module attributes
# FINAL-SAME: ttl.pipe_sram_scratch_bytes = 32 : i64
# FINAL-SAME: ttl.pipe_sync_semaphore_count = 11 : i64
# FINAL-NOT: ttl.pipe_global_semaphore_count
#
# CHECK-CPP-LABEL: // post_receives_and_send
# CHECK-CPP-DAG: {{(size_t|int32_t)}} [[READY:v[0-9]+]] = 10;
# CHECK-CPP: noc_inline_dw_write
# CHECK-CPP: get_semaphore([[READY]])
# CHECK-CPP: reinterpret_cast<tt_l1_ptr uint32_t*>
# CHECK-CPP: experimental::semaphore_wait
# CHECK-CPP: noc_async_write
# CHECK-CPP: noc_semaphore_inc
#
# RUNTIME: PASS: ksplit_resource_allocation synchronized
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
        ksplit_resource_allocation = make_ksplit_resource_allocation_kernel()
        input_torch = torch.randn(
            GRID_DIM * TILE, TRANSFER_K_TILES * TILE, dtype=torch.float32
        )
        output_torch = torch.zeros(
            GRID_DIM * TILE, GRID_DIM * TILE, dtype=torch.float32
        )

        input_tensor = to_dram(input_torch, device)
        output_tensor = to_dram(output_torch, device)
        ksplit_resource_allocation(input_tensor, output_tensor)
        ttnn.synchronize_device(device)
        print("PASS: ksplit_resource_allocation synchronized")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
