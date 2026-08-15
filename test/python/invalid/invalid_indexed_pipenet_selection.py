# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=non_pipenet not %python %s 2>&1 | FileCheck %s --check-prefix=NONPIPE
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=out_of_range not %python %s 2>&1 | FileCheck %s --check-prefix=RANGE
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=undefined_receiver not %python %s 2>&1 | FileCheck %s --check-prefix=UNDEF
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=dynamic_receiver not %python %s 2>&1 | FileCheck %s --check-prefix=DYNAMIC
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=zero_step not %python %s 2>&1 | FileCheck %s --check-prefix=ZEROSTEP

"""Invalid coverage for indexed PipeNet selection."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch  # noqa: E402
import ttl  # noqa: E402


class BFloat16Tensor:
    dtype = torch.bfloat16


# NONPIPE: PipeNet.is_src() receiver must be a compile-time PipeNet expression
@ttl.operation(grid=(1, 1))
def non_pipenet_receiver():
    not_a_pipenet = 0

    @ttl.datamovement()
    def dm():
        not_a_pipenet.is_src()


# RANGE: PipeNet.if_dst() receiver must be a compile-time PipeNet expression
@ttl.operation(grid=(1, 1))
def out_of_range_receiver():
    pipe_nets = [ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(0, 0))])]
    recv_dfb = ttl.make_dataflow_buffer_like(
        BFloat16Tensor(), shape=(1, 1), block_count=2
    )

    @ttl.datamovement()
    def dm():
        pipe_nets[1].if_dst(lambda pipe: ttl.copy(pipe, recv_dfb.reserve()).wait())


# UNDEF: PipeNet.is_src() receiver must be a compile-time PipeNet expression
@ttl.operation(grid=(1, 1))
def undefined_receiver():
    @ttl.datamovement()
    def dm():
        # `typo_net` is never bound in any enclosing scope.
        typo_net.is_src()


# DYNAMIC: PipeNet.is_src() receiver must be a compile-time PipeNet expression
@ttl.operation(grid=(2, 1))
def dynamic_receiver():
    pipe_nets = [
        ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))]),
        ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))]),
    ]

    @ttl.datamovement()
    def dm():
        node_col, _node_row = ttl.node(dims=2)
        pipe_nets[node_col].is_src()


# ZEROSTEP: range() arg 3 must not be zero
@ttl.operation(grid=(1, 1))
def zero_step_selection_loop():
    pipe_nets = [ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(0, 0))])]

    @ttl.datamovement()
    def dm():
        # A PipeNet-selection loop with a static zero step is invalid Python,
        # surfaced as a compile error rather than a dynamic-loop fallback.
        for net_index in range(0, len(pipe_nets), 0):
            pipe_nets[net_index].if_src(lambda pipe: None)


if __name__ == "__main__":
    match os.environ["CASE"]:
        case "non_pipenet":
            non_pipenet_receiver()
        case "out_of_range":
            out_of_range_receiver()
        case "undefined_receiver":
            undefined_receiver()
        case "dynamic_receiver":
            dynamic_receiver()
        case "zero_step":
            zero_step_selection_loop()
        case unknown:
            raise ValueError(f"unknown CASE={unknown}")
