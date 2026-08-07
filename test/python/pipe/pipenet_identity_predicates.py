# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.identity.mlir TTLANG_CASE=identity %python %s
# RUN: FileCheck %s --input-file=%t.identity.mlir --check-prefix=IDENTITY --implicit-check-not=arith.cmpi
# RUN: FileCheck %s --input-file=%t.identity.mlir --check-prefix=IDENTITY-SCF
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.runtime.mlir TTLANG_CASE=runtime %python %s
# RUN: FileCheck %s --input-file=%t.runtime.mlir --check-prefix=RUNTIME

"""Frontend IR coverage for graph callback identity specialization."""

import os

import pytest
import torch
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)

DEVICE_DOMAIN = ttl.DeviceDomain((1, 2))
EXCHANGE_NET = ttl.PipeNet(graph=ttl.TransferGraph.all_to_all(DEVICE_DOMAIN))


class BFloat16Tensor:
    dtype = torch.bfloat16


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def compile_identity_predicates():
    template = BFloat16Tensor()
    src_zero_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    src_one_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    dst_zero_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    dst_one_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.compute()
    def idle_compute():
        pass

    @ttl.datamovement()
    def identity_dm():
        def select_destination(pipe):
            destination_device_index = pipe.destination_device_index
            selected_dfb = src_one_dfb
            if pipe.destination_device_index == 0:
                selected_dfb = src_zero_dfb
                branch_local_dfb = selected_dfb
                if destination_device_index <= 0:
                    nested_dfb = branch_local_dfb
            else:
                branch_local_dfb = selected_dfb
                nested_dfb = branch_local_dfb

            if destination_device_index != -1:
                operator_dfb = nested_dfb
            if destination_device_index < 2:
                operator_dfb = nested_dfb
            if 0 <= destination_device_index:
                operator_dfb = nested_dfb
            if destination_device_index > -1:
                operator_dfb = nested_dfb
            if destination_device_index >= 0:
                operator_dfb = nested_dfb

            with operator_dfb.reserve() as _reserved_block:
                pass
            with operator_dfb.wait() as _ready_block:
                pass

        EXCHANGE_NET.if_src(select_destination)

        def select_source(pipe):
            if pipe.source_device_index == 0:
                selected_dfb = dst_zero_dfb
            else:
                selected_dfb = dst_one_dfb

            if 0 == pipe.source_device_index:
                selected_dfb = dst_zero_dfb

            with selected_dfb.reserve() as _reserved_block:
                pass
            with selected_dfb.wait() as _ready_block:
                pass

        EXCHANGE_NET.if_dst(select_source)

    @ttl.datamovement()
    def idle_data_movement():
        pass


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def compile_runtime_predicate():
    template = BFloat16Tensor()
    runtime_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.compute()
    def idle_compute():
        pass

    @ttl.datamovement()
    def runtime_dm():
        def select_node(pipe):
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with runtime_dfb.reserve() as _reserved_block:
                    pass

        EXCHANGE_NET.if_src(select_node)

    @ttl.datamovement()
    def idle_data_movement():
        pass


if __name__ == "__main__":
    if os.environ["TTLANG_CASE"] == "identity":
        compile_identity_predicates()
    else:
        compile_runtime_predicate()


# IDENTITY-LABEL: func.func @identity_dm
# IDENTITY-DAG: %[[SRC_ZERO:.+]] = ttl.bind_cb{{.*}} {dfb_id = 0 : index}
# IDENTITY-DAG: %[[SRC_ONE:.+]] = ttl.bind_cb{{.*}} {dfb_id = 1 : index}
# IDENTITY-DAG: %[[DST_ZERO:.+]] = ttl.bind_cb{{.*}} {dfb_id = 2 : index}
# IDENTITY-DAG: %[[DST_ONE:.+]] = ttl.bind_cb{{.*}} {dfb_id = 3 : index}
# IDENTITY: %[[PIPE_0:.+]] = ttl.create_pipe
# IDENTITY-SAME: source = <coordinates = [0, 0]>, destination = <coordinates = [0, 1]>
# IDENTITY: %[[IS_SOURCE_0:.+]] = ttl.is_device <coordinates = [0, 0]>
# IDENTITY-NEXT: scf.if %[[IS_SOURCE_0]] {
# IDENTITY-NEXT: ttl.if_src %[[PIPE_0]]
# IDENTITY-NEXT: %{{.+}} = ttl.cb_reserve %[[SRC_ONE]]
# IDENTITY: ttl.cb_wait %[[SRC_ONE]]
# IDENTITY: %[[PIPE_1:.+]] = ttl.create_pipe
# IDENTITY-SAME: source = <coordinates = [0, 1]>, destination = <coordinates = [0, 0]>
# IDENTITY: %[[IS_SOURCE_1:.+]] = ttl.is_device <coordinates = [0, 1]>
# IDENTITY-NEXT: scf.if %[[IS_SOURCE_1]] {
# IDENTITY-NEXT: ttl.if_src %[[PIPE_1]]
# IDENTITY-NEXT: %{{.+}} = ttl.cb_reserve %[[SRC_ZERO]]
# IDENTITY: ttl.cb_wait %[[SRC_ZERO]]
# IDENTITY: %[[PIPE_2:.+]] = ttl.create_pipe
# IDENTITY-SAME: source = <coordinates = [0, 0]>, destination = <coordinates = [0, 1]>
# IDENTITY: %[[IS_DESTINATION_0:.+]] = ttl.is_device <coordinates = [0, 1]>
# IDENTITY-NEXT: scf.if %[[IS_DESTINATION_0]] {
# IDENTITY-NEXT: ttl.if_dst %[[PIPE_2]]
# IDENTITY-NEXT: %{{.+}} = ttl.cb_reserve %[[DST_ZERO]]
# IDENTITY: ttl.cb_wait %[[DST_ZERO]]
# IDENTITY: %[[PIPE_3:.+]] = ttl.create_pipe
# IDENTITY-SAME: source = <coordinates = [0, 1]>, destination = <coordinates = [0, 0]>
# IDENTITY: %[[IS_DESTINATION_1:.+]] = ttl.is_device <coordinates = [0, 0]>
# IDENTITY-NEXT: scf.if %[[IS_DESTINATION_1]] {
# IDENTITY-NEXT: ttl.if_dst %[[PIPE_3]]
# IDENTITY-NEXT: %{{.+}} = ttl.cb_reserve %[[DST_ONE]]
# IDENTITY: ttl.cb_wait %[[DST_ONE]]
# IDENTITY-NOT: scf.if
# IDENTITY: return

# The only SCF conditions are the four graph endpoint predicates. Identity
# predicates are specialized before emission and add no nested SCF regions.
# IDENTITY-SCF-COUNT-4: scf.if
# IDENTITY-SCF-NOT: scf.if

# RUNTIME-LABEL: func.func @runtime_dm
# RUNTIME: %[[RUNTIME_PIPE:.+]] = ttl.create_pipe
# RUNTIME: %[[IS_RUNTIME_SOURCE:.+]] = ttl.is_device
# RUNTIME-NEXT: scf.if %[[IS_RUNTIME_SOURCE]] {
# RUNTIME-NEXT: ttl.if_src %[[RUNTIME_PIPE]]
# RUNTIME: %[[NODE_X:.+]] = ttl.core_x
# RUNTIME: %[[RUNTIME_CONDITION:.+]] = arith.cmpi eq, %[[NODE_X]], %{{.+}}
# RUNTIME-NEXT: scf.if %[[RUNTIME_CONDITION]] {
# RUNTIME-NEXT: %{{.+}} = ttl.cb_reserve
