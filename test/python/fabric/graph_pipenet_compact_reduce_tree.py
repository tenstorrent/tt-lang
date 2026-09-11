# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir timeout 300 %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL --implicit-check-not=deviceTransfer --implicit-check-not='array<1920x' --implicit-check-not='array<960x' --implicit-check-not='array<480x' --implicit-check-not='array<240x' --implicit-check-not='array<120x' < %t.initial.mlir
# RUN: %python %s --check-generated-tables < %t.output | FileCheck %s --check-prefix=CHECK-TABLES

"""Compile a realistic graph PipeNet relation without Cartesian expansion."""

import re
import sys

import pytest
import torch
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)

DEVICE_EXTENTS = (4, 8)
ROOT_DEVICE = (1, 3)
PROGRAM_GRID = (12, 10)
EXPANDED_RECORD_COUNTS = frozenset({1_920, 960, 480, 240, 120})


class BFloat16Tensor:
    dtype = torch.bfloat16


def _nearest_parent(device):
    """Return the next device toward the root in a nearest-neighbor tree."""
    if device == ROOT_DEVICE:
        return None
    device_row, device_column = device
    root_column = ROOT_DEVICE[1]
    if device_column < root_column:
        return device_row, device_column + 1
    if device_column > root_column:
        return device_row, device_column - 1
    if device_row == 0:
        return ROOT_DEVICE
    if device_row == 3:
        return 2, root_column
    if device_row == 2:
        return ROOT_DEVICE
    raise ValueError(f"device {device} has no reduction-tree parent")


def _partition_tree_edges():
    """Partition tree edges by whether and when their source receives data."""
    edges = []
    for device_row in range(DEVICE_EXTENTS[0]):
        for device_column in range(DEVICE_EXTENTS[1]):
            device = (device_row, device_column)
            parent = _nearest_parent(device)
            if parent is not None:
                edges.append((device, parent))
    receiving_devices = {destination for _source, destination in edges}
    input_edges = tuple(edge for edge in edges if edge[0] not in receiving_devices)
    input_edge_set = set(input_edges)
    partial_edges = tuple(edge for edge in edges if edge not in input_edge_set)

    partial_groups = [[] for _group_index in range(4)]
    destination_counts = {}
    for edge in partial_edges:
        destination = edge[1]
        group_index = destination_counts.get(destination, 0)
        partial_groups[group_index].append(edge)
        destination_counts[destination] = group_index + 1
    groups = (input_edges, *(tuple(group) for group in partial_groups))
    assert tuple(len(group) for group in groups) == (8, 16, 4, 2, 1)
    return groups


DEVICE_DOMAIN = ttl.DeviceDomain(DEVICE_EXTENTS)
EDGE_GROUPS = _partition_tree_edges()
INPUT_NET = ttl.PipeNet(
    graph=ttl.TransferGraph.edges(DEVICE_DOMAIN, edges=EDGE_GROUPS[0])
)
PARTIAL_NET_0 = ttl.PipeNet(
    graph=ttl.TransferGraph.edges(DEVICE_DOMAIN, edges=EDGE_GROUPS[1])
)
PARTIAL_NET_1 = ttl.PipeNet(
    graph=ttl.TransferGraph.edges(DEVICE_DOMAIN, edges=EDGE_GROUPS[2])
)
PARTIAL_NET_2 = ttl.PipeNet(
    graph=ttl.TransferGraph.edges(DEVICE_DOMAIN, edges=EDGE_GROUPS[3])
)
PARTIAL_NET_3 = ttl.PipeNet(
    graph=ttl.TransferGraph.edges(DEVICE_DOMAIN, edges=EDGE_GROUPS[4])
)


@ttl.operation(
    grid=PROGRAM_GRID,
    device_domain=DEVICE_DOMAIN,
    options="--ttl-specialize-cores",
)
def compile_compact_reduce_tree():
    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=1)
    receive_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def send_data_movement():
        def send(pipe):
            with send_dfb.reserve() as send_block:
                pass
            with send_dfb.wait() as send_block:
                ttl.copy(send_block, pipe).wait()

        INPUT_NET.if_src(send)
        PARTIAL_NET_0.if_src(send)
        PARTIAL_NET_1.if_src(send)
        PARTIAL_NET_2.if_src(send)
        PARTIAL_NET_3.if_src(send)

    @ttl.datamovement()
    def receive_data_movement():
        def receive(pipe):
            with receive_dfb.reserve() as receive_block:
                ttl.copy(pipe, receive_block).wait()
            with receive_dfb.wait() as receive_block:
                pass

        INPUT_NET.if_dst(receive)
        PARTIAL_NET_0.if_dst(receive)
        PARTIAL_NET_1.if_dst(receive)
        PARTIAL_NET_2.if_dst(receive)
        PARTIAL_NET_3.if_dst(receive)


def check_generated_tables(output):
    """Reject constant tables sized as expanded graph/worker products."""
    table_sizes = {
        int(match.group("size"))
        for match in re.finditer(
            r"static const uint64_t __ttlang_constant_table_[A-F0-9]+"
            r"\[(?P<size>[0-9]+)\]",
            output,
        )
    }
    expanded_sizes = sorted(table_sizes & EXPANDED_RECORD_COUNTS)
    assert not expanded_sizes, f"found expanded PipeNet tables: {expanded_sizes}"
    print("EXPANDED-PIPE-TABLES: none")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--check-generated-tables":
        check_generated_tables(sys.stdin.read())
    else:
        assert len(sys.argv) == 1
        compile_compact_reduce_tree()


# Each source and destination operation stores five compact graph PipeNets.
# CHECK-INITIAL-COUNT-10: #ttl.pipenet_records<

# CHECK-TABLES: EXPANDED-PIPE-TABLES: none
