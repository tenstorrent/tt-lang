# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s 6 2>&1 | FileCheck %s --check-prefix=CHECKED
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s 7 2>&1 | FileCheck %s --check-prefix=BEYOND

"""Resets under k independent dispatch conditions yield 2^k alternatives.

Within the alternative budget the surplus pushed before the first reset is
rejected; beyond it the DFB is unknown and the program is accepted.
"""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import ttl_api

HEADER = os.path.join(
    os.path.dirname(__file__), "..", "include", "scalar_result_op.hpp"
)
RESET_COUNT = int(sys.argv[1])


def _blackhole_compile_target(_runtime_args):
    return "blackhole"


ttl_api._device_target_arch = _blackhole_compile_target


def make_operation():
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    participants = (compute_kernel, reader_kernel, writer_kernel)
    reset0 = ttl.DFBReset(participants=participants)
    reset1 = ttl.DFBReset(participants=participants)
    reset2 = ttl.DFBReset(participants=participants)
    reset3 = ttl.DFBReset(participants=participants)
    reset4 = ttl.DFBReset(participants=participants)
    reset5 = ttl.DFBReset(participants=participants)
    reset6 = ttl.DFBReset(participants=participants)
    condition0 = ttl.DispatchCondition(ttl.ScalarType.I32)
    condition1 = ttl.DispatchCondition(ttl.ScalarType.I32)
    condition2 = ttl.DispatchCondition(ttl.ScalarType.I32)
    condition3 = ttl.DispatchCondition(ttl.ScalarType.I32)
    condition4 = ttl.DispatchCondition(ttl.ScalarType.I32)
    condition5 = ttl.DispatchCondition(ttl.ScalarType.I32)
    condition6 = ttl.DispatchCondition(ttl.ScalarType.I32)

    @ttl.operation(grid=(1, 1))
    def budget_probe(input_tensor, output_tensor):
        scratch_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=(1, 1), block_count=1
        )

        @ttl.compute(kernel=compute_kernel)
        def compute():
            flag0 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition0,
            )
            if flag0:
                ttl.reset_dfbs(reset0, dfbs=[scratch_dfb])
            flag1 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition1,
            )
            if flag1:
                ttl.reset_dfbs(reset1, dfbs=[scratch_dfb])
            flag2 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition2,
            )
            if flag2:
                ttl.reset_dfbs(reset2, dfbs=[scratch_dfb])
            flag3 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition3,
            )
            if flag3:
                ttl.reset_dfbs(reset3, dfbs=[scratch_dfb])
            flag4 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition4,
            )
            if flag4:
                ttl.reset_dfbs(reset4, dfbs=[scratch_dfb])
            flag5 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition5,
            )
            if flag5:
                ttl.reset_dfbs(reset5, dfbs=[scratch_dfb])
            if RESET_COUNT > 6:
                flag6 = ttl.call_extern_func(
                    HEADER,
                    "scalar_predicate",
                    template_args=[1],
                    condition_result=condition6,
                )
                if flag6:
                    ttl.reset_dfbs(reset6, dfbs=[scratch_dfb])

        @ttl.datamovement(kernel=reader_kernel)
        def reader():
            # Two unpopped pushes before the first reset exceed the capacity.
            with scratch_dfb.reserve() as block:
                ttl.copy(input_tensor[0, 0], block).wait()
            with scratch_dfb.reserve() as block:
                ttl.copy(input_tensor[0, 0], block).wait()
            flag0 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition0,
            )
            if flag0:
                ttl.reset_dfbs(reset0, dfbs=[scratch_dfb])
            with scratch_dfb.reserve() as block:
                ttl.copy(input_tensor[0, 0], block).wait()
            flag1 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition1,
            )
            if flag1:
                ttl.reset_dfbs(reset1, dfbs=[scratch_dfb])
            with scratch_dfb.reserve() as block:
                ttl.copy(input_tensor[0, 0], block).wait()
            flag2 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition2,
            )
            if flag2:
                ttl.reset_dfbs(reset2, dfbs=[scratch_dfb])
            with scratch_dfb.reserve() as block:
                ttl.copy(input_tensor[0, 0], block).wait()
            flag3 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition3,
            )
            if flag3:
                ttl.reset_dfbs(reset3, dfbs=[scratch_dfb])
            with scratch_dfb.reserve() as block:
                ttl.copy(input_tensor[0, 0], block).wait()
            flag4 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition4,
            )
            if flag4:
                ttl.reset_dfbs(reset4, dfbs=[scratch_dfb])
            with scratch_dfb.reserve() as block:
                ttl.copy(input_tensor[0, 0], block).wait()
            flag5 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition5,
            )
            if flag5:
                ttl.reset_dfbs(reset5, dfbs=[scratch_dfb])
            with scratch_dfb.reserve() as block:
                ttl.copy(input_tensor[0, 0], block).wait()
            if RESET_COUNT > 6:
                flag6 = ttl.call_extern_func(
                    HEADER,
                    "scalar_predicate",
                    template_args=[1],
                    condition_result=condition6,
                )
                if flag6:
                    ttl.reset_dfbs(reset6, dfbs=[scratch_dfb])
                with scratch_dfb.reserve() as block:
                    ttl.copy(input_tensor[0, 0], block).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def writer():
            flag0 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition0,
            )
            if flag0:
                ttl.reset_dfbs(reset0, dfbs=[scratch_dfb])
            with scratch_dfb.wait() as block:
                ttl.copy(block, output_tensor[0, 0]).wait()
            flag1 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition1,
            )
            if flag1:
                ttl.reset_dfbs(reset1, dfbs=[scratch_dfb])
            with scratch_dfb.wait() as block:
                ttl.copy(block, output_tensor[0, 0]).wait()
            flag2 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition2,
            )
            if flag2:
                ttl.reset_dfbs(reset2, dfbs=[scratch_dfb])
            with scratch_dfb.wait() as block:
                ttl.copy(block, output_tensor[0, 0]).wait()
            flag3 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition3,
            )
            if flag3:
                ttl.reset_dfbs(reset3, dfbs=[scratch_dfb])
            with scratch_dfb.wait() as block:
                ttl.copy(block, output_tensor[0, 0]).wait()
            flag4 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition4,
            )
            if flag4:
                ttl.reset_dfbs(reset4, dfbs=[scratch_dfb])
            with scratch_dfb.wait() as block:
                ttl.copy(block, output_tensor[0, 0]).wait()
            flag5 = ttl.call_extern_func(
                HEADER,
                "scalar_predicate",
                template_args=[1],
                condition_result=condition5,
            )
            if flag5:
                ttl.reset_dfbs(reset5, dfbs=[scratch_dfb])
            with scratch_dfb.wait() as block:
                ttl.copy(block, output_tensor[0, 0]).wait()
            if RESET_COUNT > 6:
                flag6 = ttl.call_extern_func(
                    HEADER,
                    "scalar_predicate",
                    template_args=[1],
                    condition_result=condition6,
                )
                if flag6:
                    ttl.reset_dfbs(reset6, dfbs=[scratch_dfb])
                with scratch_dfb.wait() as block:
                    ttl.copy(block, output_tensor[0, 0]).wait()

    return budget_probe


def main():
    shape = (32, 32)
    tensors = [
        ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        for _ in range(2)
    ]
    make_operation()(*tensors)
    print("ACCEPTED")


# CHECKED: capacity-unsafe producer and consumer transactions
# CHECKED-NOT: ACCEPTED
# BEYOND: ACCEPTED
main()
