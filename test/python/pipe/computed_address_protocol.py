# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_FINAL_MLIR=%t.p2p.final.mlir TTLANG_OP=point_to_point %python %s > %t.p2p.output 2>&1
# RUN: FileCheck %s --input-file=%t.p2p.final.mlir --check-prefix=P2P-FINAL
# RUN: FileCheck %s --input-file=%t.p2p.output --check-prefix=P2P-CPP
# RUN: env TTLANG_FINAL_MLIR=%t.allgather.final.mlir TTLANG_OP=all_gather %python %s > %t.allgather.output 2>&1
# RUN: FileCheck %s --input-file=%t.allgather.final.mlir --check-prefix=ALLGATHER-FINAL
# RUN: FileCheck %s --input-file=%t.allgather.output --check-prefix=ALLGATHER-CPP

"""Computed receiver-address protocol coverage for PipeNet lowering.

The point-to-point case covers computed addresses with sender-local capacity
release. The all-gather case covers computed addresses for a collective
multicast transfer where receiver-ready counters remain required. Both cases
execute on device and compare the result with torch expected values.
"""

import os

import torch  # noqa: E402
import ttnn  # noqa: E402

import ttl  # noqa: E402
from utils.correctness import assert_pcc  # noqa: E402

TILE = 32
ALL_GATHER_WIDTH = 4


def _make_input(shape):
    numel = 1
    for extent in shape:
        numel *= extent
    return torch.arange(numel, dtype=torch.float32).reshape(shape).to(torch.bfloat16)


def _device_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@ttl.operation(grid=(2, 1))
def point_to_point_computed_address(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])

    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)

        def send(pipe):
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 0], send_blk).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe).wait()

        net.if_src(send)

        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 0]).wait()

        if node_x == 1:
            net.if_dst(recv)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(ALL_GATHER_WIDTH, 1))
def row_all_gather_computed_address(inp, out):
    pipe0 = ttl.Pipe(src=(0, 0), dst=(slice(0, ALL_GATHER_WIDTH), 0))
    pipe1 = ttl.Pipe(src=(1, 0), dst=(slice(0, ALL_GATHER_WIDTH), 0))
    pipe2 = ttl.Pipe(src=(2, 0), dst=(slice(0, ALL_GATHER_WIDTH), 0))
    pipe3 = ttl.Pipe(src=(3, 0), dst=(slice(0, ALL_GATHER_WIDTH), 0))
    all_gather_net = ttl.PipeNet([pipe0, pipe1, pipe2, pipe3])

    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)
        if all_gather_net.is_dst():
            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe0, recv_blk)
                if node_x == 0:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe0).wait()
                recv_tx.wait()
            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe1, recv_blk)
                if node_x == 1:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 1], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe1).wait()
                recv_tx.wait()

            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 0]).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 1]).wait()

            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe2, recv_blk)
                if node_x == 2:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 2], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe2).wait()
                recv_tx.wait()
            with recv_dfb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe3, recv_blk)
                if node_x == 3:
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 3], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe3).wait()
                recv_tx.wait()

            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 2]).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 3]).wait()

    @ttl.datamovement()
    def dm_brisc():
        pass


# Point-to-point computed address: the receiver does not publish its reserved
# DFB address, and the sender uses the capacity protocol.
# P2P-FINAL-LABEL: func.func @dm
# P2P-FINAL-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
# P2P-FINAL: Semaphore({}).down
# P2P-FINAL: noc.async_write
# P2P-FINAL-NOT: noc_inline_dw_write
# P2P-FINAL-NOT: load_from_l1

# P2P-CPP: Semaphore({{.*}}).down(
# P2P-CPP: noc.async_write(
# P2P-CPP-NOT: noc_inline_dw_write

# Collective computed address: the sender still waits for receiver readiness,
# but the destination DFB address is computed instead of loaded from SRAM.
# ALLGATHER-FINAL-LABEL: func.func @dm
# ALLGATHER-FINAL-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
# ALLGATHER-FINAL-DAG: %[[STRIDE:.*]] = "emitc.constant"() <{value = 2048 : i32}>
# ALLGATHER-FINAL: experimental::semaphore_wait
# ALLGATHER-FINAL: get_compile_time_arg_val
# ALLGATHER-FINAL: %[[BASE:.*]] = literal "get_compile_time_arg_val(2)" : i32
# ALLGATHER-FINAL: %[[BASE_UI:.*]] = cast %[[BASE]] : i32 to ui32
# ALLGATHER-FINAL: %[[STRIDE_UI:.*]] = cast %[[STRIDE]] : i32 to ui32
# ALLGATHER-FINAL: %[[ADDR_UI:.*]] = add %[[BASE_UI]], %[[STRIDE_UI]]
# ALLGATHER-FINAL: %[[ADDR:.*]] = cast %[[ADDR_UI]] : ui32 to i32
# ALLGATHER-FINAL: async_write_multicast<Noc::McastMode::INCLUDE_SRC>
# ALLGATHER-FINAL-NOT: arith.remui
# ALLGATHER-FINAL-NOT: noc_inline_dw_write
# ALLGATHER-FINAL-NOT: load_from_l1

# ALLGATHER-CPP: experimental::semaphore_wait(
# ALLGATHER-CPP: get_compile_time_arg_val(
# ALLGATHER-CPP: noc0.async_write_multicast<Noc::McastMode::INCLUDE_SRC>(
# ALLGATHER-CPP-NOT: noc_inline_dw_write


def main():
    op_name = os.environ.get("TTLANG_OP", "point_to_point")
    device = ttnn.open_device(device_id=0)
    try:
        if op_name == "point_to_point":
            inp_torch = _make_input((TILE, TILE))
            out_torch = torch.zeros((TILE, TILE), dtype=torch.bfloat16)
            inp = _device_ttnn(inp_torch, device)
            out = _device_ttnn(out_torch, device)
            point_to_point_computed_address(inp, out)
            ttnn.synchronize_device(device)
            assert_pcc(inp_torch.float(), ttnn.to_torch(out).float())
            return

        inp_torch = _make_input((TILE, ALL_GATHER_WIDTH * TILE))
        out_torch = torch.zeros((TILE, ALL_GATHER_WIDTH * TILE), dtype=torch.bfloat16)
        inp = _device_ttnn(inp_torch, device)
        out = _device_ttnn(out_torch, device)
        row_all_gather_computed_address(inp, out)
        ttnn.synchronize_device(device)
        assert_pcc(inp_torch.float(), ttnn.to_torch(out).float())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
