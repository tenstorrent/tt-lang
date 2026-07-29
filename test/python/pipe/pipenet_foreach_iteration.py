# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_EMIT_RUNNER=%t.runner.py %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-LOOPS < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-SIZE < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-NO-DESCRIPTOR-ARRAYS < %t.output

"""Compile-only coverage for table-driven PipeNet callback lowering."""

import os
import runpy
from pathlib import Path

import pytest
import torch
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)

PIPE_COUNT = 7
MAX_PIPE_KERNEL_SOURCE_BYTES = 24 * 1024


class BFloat16Tensor:
    dtype = torch.bfloat16


UNICAST_NET = ttl.PipeNet(
    [ttl.Pipe(src=(node, 0), dst=(node, 1)) for node in range(PIPE_COUNT)]
)

SINGLETON_MULTICAST_NET = ttl.PipeNet(
    [
        ttl.Pipe(src=(node, 0), dst=(slice(node, node + 1), 1))
        for node in range(PIPE_COUNT)
    ]
)


@ttl.operation(grid=(PIPE_COUNT, 2))
def compile_pipenet_foreach_iteration():
    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if UNICAST_NET.is_dst():
            with recv_dfb.wait() as _recv_blk:
                pass

    @ttl.datamovement()
    def send_dm():
        if UNICAST_NET.is_src():
            with send_dfb.reserve() as send_blk:

                def send(pipe):
                    ttl.copy(send_blk, pipe).wait()

                UNICAST_NET.if_src(send)
        if SINGLETON_MULTICAST_NET.is_src():
            with send_dfb.reserve() as send_blk:
                SINGLETON_MULTICAST_NET.if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )

    @ttl.datamovement()
    def recv_dm():
        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()

        UNICAST_NET.if_dst(recv)
        SINGLETON_MULTICAST_NET.if_dst(recv)


def report_table_driven_kernel_size():
    runner = runpy.run_path(os.environ["TTLANG_EMIT_RUNNER"])
    pipe_kernel_paths = [
        Path(kernel_path)
        for kernel_path, thread_type in runner["KERNEL_PATHS"]
        if thread_type == "noc"
    ]
    largest_kernel_bytes = max(
        kernel_path.stat().st_size for kernel_path in pipe_kernel_paths
    )
    assert largest_kernel_bytes < MAX_PIPE_KERNEL_SOURCE_BYTES
    print(
        "TABLE-DRIVEN-PIPE-KERNEL-SOURCE-BYTES: "
        f"{largest_kernel_bytes} / {MAX_PIPE_KERNEL_SOURCE_BYTES}"
    )


if __name__ == "__main__":
    compile_pipenet_foreach_iteration()
    report_table_driven_kernel_size()


# CHECK-INITIAL-NOT: ttl.if_src
# CHECK-INITIAL-NOT: ttl.if_dst
# CHECK-INITIAL-NOT: ttl.create_pipe
# CHECK-INITIAL: ttl.pipenet_foreach_src
# CHECK-INITIAL-SAME: name "UNICAST_NET"
# CHECK-INITIAL: ^bb0(%{{.*}}: !ttl.selected_pipe_src):
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL-SAME: name "UNICAST_NET"
# CHECK-INITIAL: ^bb0(%{{.*}}: !ttl.selected_pipe_dst):
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL: ttl.pipenet_foreach_dst
# CHECK-INITIAL-SAME: name "SINGLETON_MULTICAST_NET"
# CHECK-INITIAL: ^bb0(%{{.*}}: !ttl.selected_pipe_dst):
# CHECK-INITIAL: ttl.copy
# CHECK-INITIAL-NOT: ttl.if_src
# CHECK-INITIAL-NOT: ttl.if_dst
# CHECK-INITIAL-NOT: ttl.create_pipe

# CHECK-CPP: {{noc[0-9]*\.async_write\(}}
# CHECK-CPP: {{noc[0-9]*\.async_write_multicast}}
# CHECK-CPP-COUNT-2: {{noc[0-9]*\.inline_dw_write}}
# CHECK-CPP: experimental::constant_table_lookup<

# CHECK-LOOPS-COUNT-4: for (
# CHECK-LOOPS-NOT: for (

# CHECK-SIZE: TABLE-DRIVEN-PIPE-KERNEL-SOURCE-BYTES: {{[0-9]+}} / 24576

# Pipe-record fields must remain compile-time tables; only mutable progress
# state requires local arrays.
# CHECK-NO-DESCRIPTOR-ARRAYS: TTNN INTEROP: Compiling kernel
# CHECK-NO-DESCRIPTOR-ARRAYS-NOT: {{size_t v[0-9]+\[[0-9]+\];}}
# CHECK-NO-DESCRIPTOR-ARRAYS: TABLE-DRIVEN-PIPE-KERNEL-SOURCE-BYTES:
