# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_FINAL_MLIR=%t.final.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-FINAL < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""Compile a graph PipeNet whose device edges cross launch-node endpoints."""

import pytest
import torch
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)


class BFloat16Tensor:
    dtype = torch.bfloat16


DEVICE_DOMAIN = ttl.DeviceDomain((1, 2))
EXCHANGE_NET = ttl.PipeNet(
    graph=ttl.TransferGraph.all_to_all(DEVICE_DOMAIN),
    pipes=[
        ttl.Pipe(src=(1, 0), dst=(0, 0)),
        ttl.Pipe(src=(2, 0), dst=(0, 0)),
    ],
)


@ttl.operation(grid=(3, 1), device_domain=DEVICE_DOMAIN)
def compile_cross_node_exchange():
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

        EXCHANGE_NET.if_src(send)

    @ttl.datamovement()
    def receive_data_movement():
        def receive(pipe):
            with receive_dfb.reserve() as receive_block:
                ttl.copy(pipe, receive_block).wait()
            with receive_dfb.wait() as receive_block:
                pass

        EXCHANGE_NET.if_dst(receive)


if __name__ == "__main__":
    compile_cross_node_exchange()


# The transfer graph and launch-node relation remain factorized in initial IR.
# Callback control flow does not encode endpoint placement.
# CHECK-INITIAL-LABEL: func.func @send_data_movement
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL-SAME: name "EXCHANGE_NET"
# CHECK-INITIAL-SAME: kind = all_to_all
# CHECK-INITIAL-SAME: pipes[<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0
# CHECK-INITIAL-SAME: <srcX = 2, srcY = 0, dstStartX = 0, dstStartY = 0
# CHECK-INITIAL-NOT: deviceTransfer
# CHECK-INITIAL-LABEL: func.func @receive_data_movement
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL-SAME: name "EXCHANGE_NET"
# CHECK-INITIAL-SAME: kind = all_to_all
# CHECK-INITIAL-SAME: pipes[<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0
# CHECK-INITIAL-SAME: <srcX = 2, srcY = 0, dstStartX = 0, dstStartY = 0
# CHECK-INITIAL-NOT: deviceTransfer
# CHECK-INITIAL-NOT: ttl.node

# Both mapped callbacks must retain their transport after PipeGraph expands
# the factorized graph records into device-specific transfer nodes.
# CHECK-FINAL-LABEL: func.func @send_data_movement()
# CHECK-FINAL-SAME: ttl.fabric_routes =
# CHECK-FINAL: call_opaque "experimental::routing_plane_fused_write_atomic_inc"
# CHECK-FINAL-LABEL: func.func @receive_data_movement()
# CHECK-FINAL-SAME: ttl.fabric_routes =
# CHECK-FINAL: call_opaque "experimental::routing_plane_atomic_inc"

# CHECK-CPP: Compiled kernel ready
