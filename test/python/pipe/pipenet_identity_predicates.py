# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.identity.mlir TTLANG_CASE=identity %python %s
# RUN: FileCheck %s --input-file=%t.identity.mlir --check-prefix=IDENTITY
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.runtime.mlir TTLANG_CASE=runtime %python %s
# RUN: FileCheck %s --input-file=%t.runtime.mlir --check-prefix=RUNTIME

"""Frontend IR coverage for runtime-selected graph callback identities."""

import os

import pytest
import torch
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)

DEVICE_DOMAIN = ttl.DeviceDomain((1, 2))
EXCHANGE_NET = ttl.PipeNet(graph=ttl.TransferGraph.all_to_all(DEVICE_DOMAIN))
ROOT_DEVICE_INDEX = 0


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
            destination = pipe.dst
            destination_x = destination[0]
            destination_y = destination[1]
            destination_device_index = pipe.destination_device_index
            if pipe.destination_device_index == 0:
                with src_zero_dfb.reserve() as _reserved_block:
                    pass
            else:
                with src_one_dfb.reserve() as _reserved_block:
                    pass

            if destination_device_index != 2:
                pass
            if destination_device_index < 2:
                pass
            if 0 <= destination_device_index:
                pass
            if destination_device_index > 0:
                pass
            if destination_device_index >= 0:
                pass

        EXCHANGE_NET.if_src(select_destination)

        def select_source(pipe):
            source_x, source_y = pipe.src
            if pipe.source_device_index == ROOT_DEVICE_INDEX:
                with dst_zero_dfb.reserve() as _reserved_block:
                    pass
            else:
                with dst_one_dfb.reserve() as _reserved_block:
                    pass

            if ROOT_DEVICE_INDEX == pipe.source_device_index:
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
            if pipe.destination_device_index == node_x:
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
# IDENTITY-NOT: ttl.create_pipe
# IDENTITY-NOT: ttl.is_device
# IDENTITY: ttl.pipenet_foreach_src attributes
# IDENTITY: ^bb0(%[[SRC_PIPE:.+]]: !ttl.selected_pipe_src):
# IDENTITY-NEXT: %[[DST_X:.+]], %[[DST_Y:.+]], %[[DST_END_X:.+]], %[[DST_END_Y:.+]] = ttl.selected_pipe_destination_coordinates %[[SRC_PIPE]]
# IDENTITY-NEXT: %[[DST_DEVICE:.+]] = ttl.selected_pipe_destination_device_index %[[SRC_PIPE]]
# IDENTITY: arith.cmpi ne, %[[DST_DEVICE]],
# IDENTITY: arith.cmpi slt, %[[DST_DEVICE]],
# IDENTITY: arith.cmpi sle,
# IDENTITY: arith.cmpi sgt, %[[DST_DEVICE]],
# IDENTITY: arith.cmpi sge, %[[DST_DEVICE]],
# IDENTITY: ttl.pipenet_foreach_dst attributes
# IDENTITY: ^bb0(%[[DST_PIPE:.+]]: !ttl.selected_pipe_dst):
# IDENTITY-NEXT: %[[SRC_X:.+]], %[[SRC_Y:.+]] = ttl.selected_pipe_source_coordinates %[[DST_PIPE]]
# IDENTITY-NEXT: %[[SRC_DEVICE:.+]] = ttl.selected_pipe_source_device_index %[[DST_PIPE]]
# IDENTITY: arith.cmpi eq, %[[SRC_DEVICE]],
# IDENTITY: return

# RUNTIME-LABEL: func.func @runtime_dm
# RUNTIME-NOT: ttl.create_pipe
# RUNTIME-NOT: ttl.is_device
# RUNTIME: ttl.pipenet_foreach_src attributes
# RUNTIME: ^bb0(%[[RUNTIME_PIPE:.+]]: !ttl.selected_pipe_src):
# RUNTIME: %[[NODE_X:.+]] = ttl.core_x
# RUNTIME: %[[DEVICE:.+]] = ttl.selected_pipe_destination_device_index %[[RUNTIME_PIPE]]
# RUNTIME-NEXT: %[[RUNTIME_CONDITION:.+]] = arith.cmpi eq, %[[DEVICE]], %[[NODE_X]]
# RUNTIME-NEXT: scf.if %[[RUNTIME_CONDITION]] {
# RUNTIME-NEXT: %{{.+}} = ttl.cb_reserve
