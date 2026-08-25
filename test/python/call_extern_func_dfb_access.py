# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_FINAL_MLIR=%t.final.mlir TTLANG_COMPILER_OPTIONS=--ttl-reuse-user-dfbs %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=FINAL < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""Verify typed external DFB inspection through compiler lowering."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/inspect_dfb.hpp"


@ttl.operation()
def inspect_aliased_dfbs(first: ttl.DFB, second: ttl.DFB, third: ttl.DFB):
    ttl.call_extern_func(
        FAKE_HEADER,
        "inspect_dfb",
        template_args=[ttl.dfb_descriptor(second)],
        func_args=[first],
        dfb_dependencies=[third],
        dfb_accesses=[
            ttl.DFBAccess.inspect(first),
            ttl.DFBAccess.inspect(second),
            ttl.DFBAccess.inspect(third),
        ],
        kernel=ttl.KernelKind.COMPUTE,
    )


@ttl.operation()
def complete_aliased_queue(producer: ttl.DFB, consumer: ttl.DFB):
    ttl.call_extern_func(
        FAKE_HEADER,
        "complete_queue_lifecycle",
        func_args=[producer, consumer],
        dfb_effects=[
            ttl.DFBEffect.reserve(producer, tiles=1),
            ttl.DFBEffect.push(producer, tiles=1),
            ttl.DFBEffect.wait(consumer, tiles=1),
            ttl.DFBEffect.pop(consumer, tiles=1),
        ],
        kernel=ttl.KernelKind.COMPUTE,
    )


@ttl.operation()
def process_aliased_dfbs(descriptor: ttl.DFB, later_queue: ttl.DFB):
    inspect_aliased_dfbs(descriptor, descriptor, descriptor)
    complete_aliased_queue(later_queue, later_queue)


@ttl.operation(grid=(1, 1))
def inspect_dfb(input_tensor):
    shared_allocation = ttl.make_dfb_allocation_group()
    descriptor = ttl.make_dataflow_buffer_like(
        input_tensor,
        shape=(1, 1),
        block_count=1,
        allocation_group=shared_allocation,
    )
    later_queue = ttl.make_dataflow_buffer_like(
        input_tensor,
        shape=(1, 1),
        block_count=1,
        allocation_group=shared_allocation,
    )
    process_aliased_dfbs(descriptor, later_queue)


# Distinct formal parameters retain separate dependency occurrences through
# nested composition when they adapt to the same caller DFB.
# INITIAL-LABEL: func.func @inspect_dfb__trisc
# INITIAL: %[[DESCRIPTOR:.*]] = ttl.bind_cb{{.*}}allocation_group = #ttl.dfb_allocation_group<0>{{.*}}dfb_id = 0 : index
# INITIAL: %[[LATER_QUEUE:.*]] = ttl.bind_cb{{.*}}allocation_group = #ttl.dfb_allocation_group<0>{{.*}}dfb_id = 1 : index
# INITIAL: ttl.opaque_call "inspect_dfb" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%[[DESCRIPTOR]] : !ttl.cb<{{.*}}>) dfb_dependencies(%[[DESCRIPTOR]] : !ttl.cb<{{.*}}>) dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>, #ttl.dfb_non_transactional_access<inspect, 1>, #ttl.dfb_non_transactional_access<inspect, 2>] (%[[DESCRIPTOR]]) {header = "/dev/null/inspect_dfb.hpp"}
# INITIAL: ttl.opaque_call "complete_queue_lifecycle" dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 1, 1>, #ttl.dfb_protocol_effect<pop, 1, 1>] (%[[LATER_QUEUE]], %[[LATER_QUEUE]]) {header = "/dev/null/inspect_dfb.hpp"}

# All three inspections complete synchronously, and the later queue has a
# complete protocol lifecycle, so the ordered logical DFBs reuse one index.
# FINAL: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}]
# FINAL-NOT: dfb_index = 1

# Access metadata does not add C++ arguments or queue operations.
# CHECK-CPP: inspect_dfb<ttlang::DFBDescriptor<{{.*}}>>(get_compile_time_arg_val(0));
# CHECK-CPP: complete_queue_lifecycle(get_compile_time_arg_val(0), get_compile_time_arg_val(0));


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    inspect_dfb(input_tensor)
