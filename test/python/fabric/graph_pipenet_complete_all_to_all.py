# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""Compile complete all-to-all endpoints with local and remote transfers."""

import pytest
import torch
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)


class BFloat16Tensor:
    dtype = torch.bfloat16


DEVICE_DOMAIN = ttl.DeviceDomain((1, 4))
ROW_ALL_TO_ALL = ttl.PipeNet(
    [
        ttl.Pipe.all_to_all(
            src=DEVICE_DOMAIN[0, :].at_node(0, 0),
            dst=DEVICE_DOMAIN[0, :].at_node(0, 0),
            include_self=True,
        )
    ]
)


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def compile_complete_all_to_all():
    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=1)
    receive_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def sender():
        def send(pipe):
            with send_dfb.reserve() as send_block:
                pass
            with send_dfb.wait() as send_block:
                ttl.copy(send_block, pipe).wait()

        ROW_ALL_TO_ALL.if_src(send)

    @ttl.datamovement()
    def receiver():
        def receive(pipe):
            with receive_dfb.reserve() as receive_block:
                ttl.copy(pipe, receive_block).wait()
            with receive_dfb.wait() as receive_block:
                pass

        ROW_ALL_TO_ALL.if_dst(receive)


if __name__ == "__main__":
    compile_complete_all_to_all()


# The first graph group contains four same-device edges. The second contains
# the twelve distinct-device edges. Both retain node (0, 0) endpoints.
# CHECK-INITIAL-LABEL: func.func @sender
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL-SAME: name "ROW_ALL_TO_ALL" mappings
# CHECK-INITIAL-SAME: source = <coordinates = [0, 0]>
# CHECK-INITIAL-SAME: destination = <coordinates = [0, 0]>
# CHECK-INITIAL-SAME: source = <coordinates = [0, 0]>
# CHECK-INITIAL-SAME: destination = <coordinates = [0, 1]>
# CHECK-INITIAL-SAME: pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0
# CHECK-INITIAL-LABEL: func.func @receiver
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL-SAME: name "ROW_ALL_TO_ALL" mappings
# CHECK-INITIAL-SAME: source = <coordinates = [0, 0]>
# CHECK-INITIAL-SAME: destination = <coordinates = [0, 0]>
# CHECK-INITIAL-SAME: source = <coordinates = [0, 0]>
# CHECK-INITIAL-SAME: destination = <coordinates = [0, 1]>

# CHECK-CPP: Compiled kernel ready
