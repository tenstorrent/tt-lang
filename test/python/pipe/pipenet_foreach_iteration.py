# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-LOOPS < %t.output
# RUN: %python %s --report-kernel-size < %t.output | FileCheck %s --check-prefix=CHECK-SIZE
# RUN: FileCheck %s --check-prefix=CHECK-NO-DESCRIPTOR-ARRAYS < %t.output

"""Compile-only coverage for large table-driven PipeNet callback lowering."""

import re
import sys

import pytest
import torch
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)

NODE_GRID = (4, 8)
NODE_COUNT = NODE_GRID[0] * NODE_GRID[1]
PEER_COUNT = NODE_COUNT - 1
ALL_TO_ALL_EDGE_COUNT = NODE_COUNT * PEER_COUNT
SINGLE_RECEIVER_COLLECTIVE_COUNT = 7
MAX_LOCAL_PIPE_KERNEL_SOURCE_BYTES = 24 * 1024
MAX_DEVICE_PIPE_KERNEL_SOURCE_BYTES = 32 * 1024


class BFloat16Tensor:
    dtype = torch.bfloat16


NODES = [
    (node_x, node_y) for node_x in range(NODE_GRID[0]) for node_y in range(NODE_GRID[1])
]
ALL_TO_ALL_NET = ttl.PipeNet(
    [
        ttl.Pipe(src=source, dst=destination)
        for source in NODES
        for destination in NODES
        if source != destination
    ]
)

DEVICE_DOMAIN = ttl.DeviceDomain(NODE_GRID)
DEVICE_ALL_TO_ALL_NET = ttl.PipeNet(graph=ttl.TransferGraph.all_to_all(DEVICE_DOMAIN))

SINGLE_RECEIVER_COLLECTIVE_NET = ttl.PipeNet(
    [
        ttl.Pipe(src=(node, 0), dst=(slice(node, node + 1), 1))
        for node in range(SINGLE_RECEIVER_COLLECTIVE_COUNT)
    ]
)


@ttl.operation(grid=(SINGLE_RECEIVER_COLLECTIVE_COUNT, 2))
def compile_single_receiver_collective():
    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if SINGLE_RECEIVER_COLLECTIVE_NET.is_dst():
            with recv_dfb.wait() as _recv_blk:
                pass

    @ttl.datamovement()
    def send_dm():
        if SINGLE_RECEIVER_COLLECTIVE_NET.is_src():
            with send_dfb.reserve() as send_blk:
                SINGLE_RECEIVER_COLLECTIVE_NET.if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )

    @ttl.datamovement()
    def recv_dm():
        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()

        SINGLE_RECEIVER_COLLECTIVE_NET.if_dst(recv)


@ttl.operation(grid=NODE_GRID)
def compile_all_to_all():
    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(
        template, shape=(1, 1), block_count=PEER_COUNT
    )

    @ttl.compute()
    def compute():
        for _peer_index in range(PEER_COUNT):
            with recv_dfb.wait() as _recv_block:
                pass

    @ttl.datamovement()
    def send_dm():
        with send_dfb.reserve() as send_block:
            ALL_TO_ALL_NET.if_src(lambda pipe: ttl.copy(send_block, pipe).wait())

    @ttl.datamovement()
    def recv_dm():
        def receive(pipe):
            with recv_dfb.reserve() as recv_block:
                ttl.copy(pipe, recv_block).wait()

        ALL_TO_ALL_NET.if_dst(receive)


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def compile_device_all_to_all():
    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(
        template, shape=(1, 1), block_count=PEER_COUNT
    )

    @ttl.compute()
    def compute():
        for _peer_index in range(PEER_COUNT):
            with recv_dfb.wait() as _recv_block:
                pass

    @ttl.datamovement()
    def send_dm():
        with send_dfb.reserve() as send_block:
            DEVICE_ALL_TO_ALL_NET.if_src(lambda pipe: ttl.copy(send_block, pipe).wait())

    @ttl.datamovement()
    def recv_dm():
        def receive(pipe):
            with recv_dfb.reserve() as recv_block:
                ttl.copy(pipe, recv_block).wait()

        DEVICE_ALL_TO_ALL_NET.if_dst(receive)


def report_table_driven_kernel_size(output):
    """Report the largest pipe kernel source section in compiler output."""
    # Kernel logging appends one newline after the source text.
    pipe_kernel_sources = [
        match.group("source").removesuffix("\n")
        for match in re.finditer(
            r"^=== (?:send_dm|recv_dm) kernel written to [^\n]+ ===\n"
            r"(?P<source>.*?)^={60}$",
            output,
            re.DOTALL | re.MULTILINE,
        )
    ]
    assert len(pipe_kernel_sources) == 6
    local_kernel_bytes = max(
        len(kernel_source.encode()) for kernel_source in pipe_kernel_sources[:4]
    )
    device_kernel_bytes = max(
        len(kernel_source.encode()) for kernel_source in pipe_kernel_sources[4:]
    )
    assert local_kernel_bytes < MAX_LOCAL_PIPE_KERNEL_SOURCE_BYTES
    assert device_kernel_bytes < MAX_DEVICE_PIPE_KERNEL_SOURCE_BYTES
    print(
        "LOCAL-TABLE-DRIVEN-PIPE-KERNEL-SOURCE-BYTES: "
        f"{local_kernel_bytes} / {MAX_LOCAL_PIPE_KERNEL_SOURCE_BYTES}"
    )
    print(
        "DEVICE-TABLE-DRIVEN-PIPE-KERNEL-SOURCE-BYTES: "
        f"{device_kernel_bytes} / {MAX_DEVICE_PIPE_KERNEL_SOURCE_BYTES}"
    )


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--report-kernel-size":
        report_table_driven_kernel_size(sys.stdin.read())
    else:
        assert len(sys.argv) == 1
        assert len(ALL_TO_ALL_NET.pipes) == ALL_TO_ALL_EDGE_COUNT
        assert DEVICE_ALL_TO_ALL_NET.graph is not None
        assert DEVICE_ALL_TO_ALL_NET.graph.is_structured
        assert (
            sum(1 for _edge in DEVICE_ALL_TO_ALL_NET.graph.iter_edges())
            == ALL_TO_ALL_EDGE_COUNT
        )
        print(f"ALL-TO-ALL-EDGE-COUNT: {len(ALL_TO_ALL_NET.pipes)}")
        compile_single_receiver_collective()
        compile_all_to_all()
        compile_device_all_to_all()


# CHECK-INITIAL-NOT: ttl.if_src
# CHECK-INITIAL-NOT: ttl.if_dst
# CHECK-INITIAL-NOT: ttl.create_pipe
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL-SAME: name "DEVICE_ALL_TO_ALL_NET"
# CHECK-INITIAL-SAME: deviceTransfer
# CHECK-INITIAL: ^bb0(%{{.*}}: !ttl.selected_pipe_src):
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL-SAME: name "DEVICE_ALL_TO_ALL_NET"
# CHECK-INITIAL-SAME: deviceTransfer
# CHECK-INITIAL: ^bb0(%{{.*}}: !ttl.selected_pipe_dst):
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL-NOT: ttl.if_src
# CHECK-INITIAL-NOT: ttl.if_dst
# CHECK-INITIAL-NOT: ttl.create_pipe

# CHECK-CPP: ALL-TO-ALL-EDGE-COUNT: 992
# The generated kernels may compute record-table fields before their transport
# operations; these checks require the independent code-generation features.
# CHECK-CPP-DAG: {{noc[0-9]*\.async_write\(}}
# CHECK-CPP-DAG: {{noc[0-9]*\.async_write_multicast}}
# CHECK-CPP-DAG: experimental::constant_table_lookup<
# CHECK-CPP-DAG: tt::tt_fabric::RoutingPlaneConnectionManager
# CHECK-CPP-DAG: to_noc_fused_unicast_write_atomic_inc
# CHECK-CPP-DAG: send_payload_without_header_non_blocking_from_address
# CHECK-CPP-DAG: experimental::routing_plane_atomic_inc

# CHECK-LOOPS-COUNT-8: for (
# CHECK-LOOPS-NOT: for (

# CHECK-SIZE: LOCAL-TABLE-DRIVEN-PIPE-KERNEL-SOURCE-BYTES: {{[0-9]+}} / 24576
# CHECK-SIZE: DEVICE-TABLE-DRIVEN-PIPE-KERNEL-SOURCE-BYTES: {{[0-9]+}} / 32768

# Pipe-record fields must remain compile-time tables; only mutable progress
# state requires local arrays.
# CHECK-NO-DESCRIPTOR-ARRAYS: TTNN INTEROP: Compiling kernel
# CHECK-NO-DESCRIPTOR-ARRAYS-NOT: {{size_t v[0-9]+\[[0-9]+\];}}
# CHECK-NO-DESCRIPTOR-ARRAYS: Compiled kernel ready
