# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s dispatch-condition-reset 2>&1 | FileCheck %s --check-prefix=RESET-CONDITION
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s dispatch-condition-reconfiguration 2>&1 | FileCheck %s --check-prefix=RECONFIGURATION-CONDITION
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s nested-loop-reset 2>&1 | FileCheck %s --check-prefix=NESTED-RESET

# RESET-CONDITION: error: synchronized DFB reset must execute a compile-time-known number of times on each launch node; it may depend on the launch node but not on runtime values
# RECONFIGURATION-CONDITION: error: DFB reconfiguration must execute a compile-time-known number of times on each launch node; it may depend on the launch node but not on runtime values
# NESTED-RESET: error: repeated synchronized DFB reset must execute in one loop, not in nested loops

"""Reject resets and reconfigurations the DFB lifecycle verifier cannot check.

A reset or reconfiguration must execute a compile-time-known number of times
on each launch node: at the kernel's top level, under conditions on the launch
node only, or in one loop with a compile-time trip count.
"""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import ttl_api

MODE = sys.argv[1]
HEADER = os.path.join(
    os.path.dirname(__file__), "..", "include", "scalar_result_op.hpp"
)


def _blackhole_compile_target(_runtime_args):
    return "blackhole"


ttl_api._device_target_arch = _blackhole_compile_target


def _participants():
    return (
        ttl.Kernel(ttl.KernelKind.COMPUTE),
        ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT),
        ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT),
    )


def make_dispatch_condition_reset():
    compute_kernel, reader_kernel, writer_kernel = _participants()
    reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))
    active = ttl.DispatchCondition(ttl.ScalarType.I32)

    @ttl.operation(grid=(1, 1))
    def dispatch_condition_reset(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            flag = ttl.call_extern_func(
                HEADER, "scalar_predicate", template_args=[1], condition_result=active
            )
            if flag:
                ttl.reset_dfbs(reset, dfbs=[dfb])

        @ttl.datamovement(kernel=reader_kernel)
        def reader():
            flag = ttl.call_extern_func(
                HEADER, "scalar_predicate", template_args=[1], condition_result=active
            )
            if flag:
                ttl.reset_dfbs(reset, dfbs=[dfb])

        @ttl.datamovement(kernel=writer_kernel)
        def writer():
            flag = ttl.call_extern_func(
                HEADER, "scalar_predicate", template_args=[1], condition_result=active
            )
            if flag:
                ttl.reset_dfbs(reset, dfbs=[dfb])

    return dispatch_condition_reset


def make_dispatch_condition_reconfiguration():
    compute_kernel, reader_kernel, writer_kernel = _participants()
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )
    active = ttl.DispatchCondition(ttl.ScalarType.I32)

    @ttl.operation(grid=(1, 1))
    def dispatch_condition_reconfiguration(inp, out):
        @ttl.compute(kernel=compute_kernel)
        def compute():
            flag = ttl.call_extern_func(
                HEADER, "scalar_predicate", template_args=[1], condition_result=active
            )
            if flag:
                ttl.reconfigure_dfbs(boundary)

        @ttl.datamovement(kernel=reader_kernel)
        def reader():
            flag = ttl.call_extern_func(
                HEADER, "scalar_predicate", template_args=[1], condition_result=active
            )
            if flag:
                ttl.reconfigure_dfbs(boundary)

        @ttl.datamovement(kernel=writer_kernel)
        def writer():
            flag = ttl.call_extern_func(
                HEADER, "scalar_predicate", template_args=[1], condition_result=active
            )
            if flag:
                ttl.reconfigure_dfbs(boundary)

    return dispatch_condition_reconfiguration


def make_nested_loop_reset():
    compute_kernel, reader_kernel, writer_kernel = _participants()
    reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))

    @ttl.operation(grid=(1, 1))
    def nested_loop_reset(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            for outer in range(2):
                for inner in range(2):
                    ttl.reset_all_dfbs(reset)

        @ttl.datamovement(kernel=reader_kernel)
        def reader():
            with dfb.reserve() as blk:
                ttl.copy(inp[0, 0], blk).wait()
            for outer in range(2):
                for inner in range(2):
                    ttl.reset_all_dfbs(reset)

        @ttl.datamovement(kernel=writer_kernel)
        def writer():
            for outer in range(2):
                for inner in range(2):
                    ttl.reset_all_dfbs(reset)

    return nested_loop_reset


FACTORIES = {
    "dispatch-condition-reset": make_dispatch_condition_reset,
    "dispatch-condition-reconfiguration": make_dispatch_condition_reconfiguration,
    "nested-loop-reset": make_nested_loop_reset,
}
operation = FACTORIES[MODE]()

tensors = [
    ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    for _ in range(2)
]
operation(*tensors)
