# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_FINAL_MLIR=%t.p2p.bf16.final.mlir TTLANG_OP=point_to_point TTLANG_DTYPE=bf16 %python %s > %t.p2p.bf16.output 2>&1
# RUN: FileCheck %s --input-file=%t.p2p.bf16.final.mlir --check-prefix=P2P-FINAL
# RUN: FileCheck %s --input-file=%t.p2p.bf16.output --check-prefix=P2P-CPP
# RUN: env TTLANG_FINAL_MLIR=%t.p2p.fp32.final.mlir TTLANG_OP=point_to_point TTLANG_DTYPE=fp32 %python %s > %t.p2p.fp32.output 2>&1
# RUN: FileCheck %s --input-file=%t.p2p.fp32.final.mlir --check-prefix=P2P-FINAL
# RUN: FileCheck %s --input-file=%t.p2p.fp32.output --check-prefix=P2P-CPP
# RUN: env TTLANG_FINAL_MLIR=%t.allgather.bf16.final.mlir TTLANG_OP=all_gather TTLANG_DTYPE=bf16 %python %s > %t.allgather.bf16.output 2>&1
# RUN: FileCheck %s --input-file=%t.allgather.bf16.final.mlir --check-prefixes=ALLGATHER-FINAL,ALLGATHER-BF16-FINAL
# RUN: FileCheck %s --input-file=%t.allgather.bf16.output --check-prefix=ALLGATHER-CPP
# RUN: env TTLANG_FINAL_MLIR=%t.allgather.fp32.final.mlir TTLANG_OP=all_gather TTLANG_DTYPE=fp32 %python %s > %t.allgather.fp32.output 2>&1
# RUN: FileCheck %s --input-file=%t.allgather.fp32.final.mlir --check-prefixes=ALLGATHER-FINAL,ALLGATHER-FP32-FINAL
# RUN: FileCheck %s --input-file=%t.allgather.fp32.output --check-prefix=ALLGATHER-CPP

"""Computed receiver-address protocol coverage for PipeNet lowering.

The point-to-point case covers dynamic computed addresses with sender-local
capacity release. The all-gather case covers dynamic computed addresses for a
collective multicast transfer where receiver-ready counters remain required.
Both cases execute on device and compare the result with torch expected values.
"""

import os

import torch  # noqa: E402
import ttnn  # noqa: E402

import ttl  # noqa: E402
from ttlang_test_utils import to_dram, torch_dtype_from_env  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

TILE = 32
POINT_TO_POINT_ITERS = 2
ALL_GATHER_WIDTH = 4
ALL_GATHER_ITERS = 2


def _make_input(shape, dtype):
    numel = 1
    for extent in shape:
        numel *= extent
    return torch.arange(numel, dtype=torch.float32).reshape(shape).to(dtype)


def _assert_copy_matches(expected, actual, dtype):
    if dtype == torch.bfloat16:
        assert_allclose(actual.float(), expected.float(), rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual.float(), expected.float(), rtol=1e-5, atol=1e-5)


@ttl.operation(grid=(2, 1))
def point_to_point_computed_address(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])

    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

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

        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 0]).wait()

        for _iter_idx in range(POINT_TO_POINT_ITERS):
            net.if_src(send)
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
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=8)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)
        for _iter_idx in range(ALL_GATHER_ITERS):
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
# DFB address, and the sender uses the capacity protocol. recv_block_count=2
# and two iterations force dynamic sender-side receiver slot tracking.
# P2P-FINAL-LABEL: func.func @dm
# The receiver DFB is allocated at physical index 0 after DFB compaction.
# P2P-FINAL-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 0>
# P2P-FINAL-DAG: %[[P2P_BASE_ARG_INDEX:.*]] = "emitc.constant"() <{value = 2 : index}>
# Address calculation may move relative to the capacity wait; both must
# precede the payload write.
# P2P-FINAL-DAG: call_opaque "experimental::semaphore_wait_min"
# P2P-FINAL-DAG: call_opaque "get_common_arg_val"(%[[P2P_BASE_ARG_INDEX]])
# P2P-FINAL-DAG: = rem {{.*}} : (ui32, ui32) -> ui32
# P2P-FINAL: noc0.async_write
# P2P-FINAL-NOT: noc_inline_dw_write
# P2P-FINAL-NOT: load_from_l1

# P2P-CPP-DAG: experimental::semaphore_wait_min(
# P2P-CPP-DAG: get_common_arg_val<uint32_t>(
# P2P-CPP-DAG: {{.*}} % {{.*}};
# P2P-CPP: noc0.async_write(
# P2P-CPP-NOT: noc_inline_dw_write

# Collective computed address: the sender still waits for receiver readiness,
# but the destination DFB address is computed instead of loaded from SRAM.
# recv_block_count=8 and two iterations force dynamic sender-side receiver slot
# tracking across all four multicast sends.
# ALLGATHER-FINAL-LABEL: func.func @dm
# The receiver DFB is allocated at physical index 0 after DFB compaction.
# ALLGATHER-FINAL-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 0>
# ALLGATHER-BF16-FINAL-DAG: %[[STRIDE:.*]] = "emitc.constant"() <{value = 2048 : i32}>
# ALLGATHER-FP32-FINAL-DAG: %[[STRIDE:.*]] = "emitc.constant"() <{value = 4096 : i32}>
# ALLGATHER-FINAL-DAG: %[[ALLGATHER_BASE_ARG_INDEX:.*]] = "emitc.constant"() <{value = 2 : index}>
# ALLGATHER-FINAL-DAG: call_opaque "experimental::semaphore_wait"
# ALLGATHER-FINAL-DAG: call_opaque "get_common_arg_val"(%[[ALLGATHER_BASE_ARG_INDEX]])
# ALLGATHER-FINAL-DAG: %[[STRIDE_UI:.*]] = cast %[[STRIDE]] : i32 to ui32
# ALLGATHER-FINAL-DAG: mul {{.*}}, %[[STRIDE_UI]]
# ALLGATHER-FINAL-DAG: = rem {{.*}} : (ui32, ui32) -> ui32
# ALLGATHER-FINAL: async_write_multicast<NocOptions::MCAST_INCL_SRC>
# ALLGATHER-FINAL-NOT: noc_inline_dw_write
# ALLGATHER-FINAL-NOT: load_from_l1

# ALLGATHER-CPP-DAG: experimental::semaphore_wait(
# ALLGATHER-CPP-DAG: get_common_arg_val<uint32_t>(
# ALLGATHER-CPP-DAG: {{.*}} % {{.*}};
# ALLGATHER-CPP: noc0.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
# ALLGATHER-CPP-NOT: noc_inline_dw_write


def main():
    op_name = os.environ.get("TTLANG_OP", "point_to_point")
    dtype = torch_dtype_from_env("TTLANG_DTYPE")
    device = ttnn.open_device(device_id=0)
    try:
        if op_name == "point_to_point":
            inp_torch = _make_input((TILE, TILE), dtype)
            out_torch = torch.zeros((TILE, TILE), dtype=dtype)
            inp = to_dram(inp_torch, device)
            out = to_dram(out_torch, device)
            point_to_point_computed_address(inp, out)
            ttnn.synchronize_device(device)
            _assert_copy_matches(inp_torch, ttnn.to_torch(out), dtype)
            return

        inp_torch = _make_input((TILE, ALL_GATHER_WIDTH * TILE), dtype)
        out_torch = torch.zeros((TILE, ALL_GATHER_WIDTH * TILE), dtype=dtype)
        inp = to_dram(inp_torch, device)
        out = to_dram(out_torch, device)
        row_all_gather_computed_address(inp, out)
        ttnn.synchronize_device(device)
        _assert_copy_matches(inp_torch, ttnn.to_torch(out), dtype)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
