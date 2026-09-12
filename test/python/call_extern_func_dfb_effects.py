# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.specialized.initial.mlir TTLANG_COMPILER_OPTIONS=--ttl-specialize-cores %python %s > %t.specialized.output 2>&1
# RUN: FileCheck %s --check-prefix=INITIAL < %t.specialized.initial.mlir
# RUN: FileCheck %s --check-prefix=SPECIALIZED-CPP < %t.specialized.output

"""Verify external-call DFB metadata through unified-operation lowering."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/fake_shim.hpp"
EFFECT_TILES = 5
reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
writer = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)


@ttl.operation()
def external_source(source: ttl.DFB):
    ttl.call_extern_func(
        FAKE_HEADER,
        "external_source",
        dfb_dependencies=[source],
        dfb_effects=[
            ttl.DFBEffect.reserve(source, tiles=EFFECT_TILES - 1),
            ttl.DFBEffect.push(source, tiles=EFFECT_TILES - 1),
        ],
        kernel=writer,
    )


def make_external_stage(transaction_count):
    @ttl.operation()
    def external_stage(source: ttl.DFB, destination: ttl.DFB):
        ttl.call_extern_func(
            FAKE_HEADER,
            "external_stage",
            template_args=[ttl.get_dfb_id(source)],
            func_args=[source],
            dfb_dependencies=[destination],
            dfb_effects=[
                ttl.DFBEffect.repeat(
                    -(-transaction_count),
                    [
                        ttl.DFBEffect.repeat(
                            1,
                            [ttl.DFBEffect.wait(source, tiles=(EFFECT_TILES + 1) // 3)],
                        ),
                        ttl.DFBEffect.pop(source, tiles=EFFECT_TILES - 3),
                    ],
                ),
                ttl.DFBEffect.repeat(
                    0,
                    [ttl.DFBEffect.wait(destination, tiles=99)],
                ),
                ttl.DFBEffect.reserve(destination, tiles=(EFFECT_TILES * 2) % 3),
                ttl.DFBEffect.push(destination, tiles=+(EFFECT_TILES % 2)),
            ],
            unknown_dfb_access=True,
            kernel=reader,
        )

    return external_stage


external_stage = make_external_stage(2)


@ttl.operation(grid=(2, 1))
def external_metadata_kernel(inp):
    source = ttl.make_dataflow_buffer_like(inp, shape=(1, 2), block_count=2)
    destination = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    external_source(source)
    core_x, _ = ttl.node(dims=2)
    if core_x == 0:
        external_stage(source, destination)
    else:
        external_stage(source, destination)


# The composed call retains automatic and dependency-only DFB identity. The
# effect list order and distinct tile counts are preserved exactly.
# INITIAL-LABEL: func.func @external_metadata_kernel__ncrisc
# INITIAL-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "__main__.{{.*}}external_stage{{\[captures=[0-9a-f]+\]}}">
# INITIAL-DAG: %[[SOURCE:.*]] = ttl.bind_cb
# INITIAL-DAG: %[[DESTINATION:.*]] = ttl.bind_cb
# INITIAL: ttl.opaque_call "external_stage" template_args [#ttl.external_template_arg<dfb_index, 0>] template_dfbs(%[[SOURCE]] : !ttl.cb<{{.*}}>) dfb_dependencies(%[[DESTINATION]] : !ttl.cb<{{.*}}>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>, #ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>, #ttl.dfb_protocol_effect<reserve, 1, 1>, #ttl.dfb_protocol_effect<push, 1, 1>] (%[[SOURCE]]) {header = "/dev/null/fake_shim.hpp", unknown_dfb_access}

# Dependency-only operands and protocol metadata do not change the C++ call.
# CHECK-CPP: external_stage<[[SOURCE_INDEX:[0-9]+]]U>(get_compile_time_arg_val([[SOURCE_INDEX]]));
# CHECK-CPP-NOT: cb_reserve
# CHECK-CPP-NOT: cb_push
# CHECK-CPP-NOT: cb_wait
# CHECK-CPP-NOT: cb_pop

# Core specialization clones the selected call without adding C++ arguments or
# protocol calls.
# SPECIALIZED-CPP-COUNT-2: external_stage<{{[0-9]+}}U>(get_compile_time_arg_val({{[0-9]+}}));
# SPECIALIZED-CPP-NOT: cb_reserve
# SPECIALIZED-CPP-NOT: cb_push
# SPECIALIZED-CPP-NOT: cb_wait
# SPECIALIZED-CPP-NOT: cb_pop


if __name__ == "__main__":
    inp = ttnn.from_torch(
        torch.zeros((32, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    external_metadata_kernel(inp)
