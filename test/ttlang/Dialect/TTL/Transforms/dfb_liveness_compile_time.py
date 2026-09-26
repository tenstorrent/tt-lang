# REQUIRES: optimized
# RUN: %python %s

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Bounds protocol-graph and structural-order construction time, and the
# lifecycle verifier's handling of many resets under unresolved dispatch
# conditions, in optimized builds.

import os
import subprocess

PROTOCOL_GRAPH_TIMEOUT_SECONDS = 10
STRUCTURAL_ORDER_TIMEOUT_SECONDS = 5
CONDITIONAL_RESET_TIMEOUT_SECONDS = 5
CONDITIONAL_RESET_COUNT = 24
GRID_CONDITIONAL_RESET_TIMEOUT_SECONDS = 10
GRID_CONDITIONAL_RESET_COUNT = 6
GRID_CONDITIONAL_RESET_GRID = (4, 4)
GRID_CONDITIONAL_RESET_DFB_COUNT = 4
WIDE_CONDITION_RESET_TIMEOUT_SECONDS = 10
WIDE_CONDITION_LEAVES = 16
TRANSACTION_COUNT = 500
RESET_COUNT_PER_BRANCH = 128
RESET_NESTING_DEPTH = 8
DFB_TYPE = "!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>"
PROTOCOL_TYPE = "<[1, 1], !ttcore.tile<32x32, bf16>, 2>"
TENSOR_TYPE = "tensor<1x1x!ttcore.tile<32x32, bf16>>"


def build_stress_module() -> str:
    lines = [
        "module {",
        "  func.func @dfb_liveness_compile_time()",
        "      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,",
        "                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {",
        "    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} "
        f"{{dfb_id = 0 : index}} : {DFB_TYPE}",
    ]
    for transaction_index in range(TRANSACTION_COUNT):
        lines.extend(
            [
                f"    %reserved{transaction_index} = ttl.cb_reserve %dfb : "
                f"{PROTOCOL_TYPE} -> {TENSOR_TYPE}",
                f"    ttl.cb_push %dfb : {PROTOCOL_TYPE}",
                f"    %waited{transaction_index} = ttl.cb_wait %dfb : "
                f"{PROTOCOL_TYPE} -> {TENSOR_TYPE}",
                f"    ttl.cb_pop %dfb : {PROTOCOL_TYPE}",
            ]
        )
    lines.extend(["    return", "  }", "}"])
    return "\n".join(lines)


def build_structural_order_stress_module() -> str:
    operation_identity = "dfb_structural_order_compile_time"
    participant_list = (
        f'<kind = compute, identity = "compute", operation = "{operation_identity}">, '
        f'<kind = data_movement, identity = "reader", operation = "{operation_identity}">, '
        f'<kind = data_movement, identity = "writer", operation = "{operation_identity}">'
    )
    lines = [
        "module attributes {ttl.launch_grid = [12, 10], "
        "ttl.target_arch = #ttcore.arch<blackhole>} {"
    ]
    participant_specs = (
        ("compute", "compute", "compute", None),
        ("reader", "data_movement", "noc", 0),
        ("writer", "data_movement", "noc", 1),
    )
    for function_name, kernel_kind, thread_kind, noc_index in participant_specs:
        lines.extend(
            [
                f"  func.func @{function_name}()",
                f"      attributes {{ttl.kernel_thread = #ttkernel.thread<{thread_kind}>,",
                "                  ttl.logical_kernel = "
                f'#ttl.logical_kernel<kind = {kernel_kind}, identity = "{function_name}", '
                f'operation = "{operation_identity}">,',
            ]
        )
        if noc_index is not None:
            lines.append(f"                  ttl.noc_index = {noc_index} : i32,")
        lines.extend(
            [
                "                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {",
                "    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} "
                f"{{dfb_id = 0 : index}} : {DFB_TYPE}",
                "    %condition = arith.constant true",
            ]
        )
        lines.extend("    scf.if %condition {" for _ in range(RESET_NESTING_DEPTH))
        lines.append("    scf.if %condition {")
        reset_ordinal = 0
        for branch_name in ("then", "else"):
            if branch_name == "else":
                lines.append("    } else {")
            for _ in range(RESET_COUNT_PER_BRANCH):
                lines.append(
                    f"      ttl.reset_all_dfbs <{reset_ordinal}, "
                    f"participants[{participant_list}]>"
                )
                reset_ordinal += 1
        lines.append("    }")
        lines.extend("    }" for _ in range(RESET_NESTING_DEPTH))
        lines.extend(["    return", "  }"])
    lines.append("}")
    return "\n".join(lines)


def build_conditional_reset_module(
    reset_count: int = CONDITIONAL_RESET_COUNT,
    grid: tuple[int, int] = (1, 1),
    dfb_count: int = 1,
    condition_leaves: int = 1,
) -> str:
    """Every participant resets the DFBs under each of many dispatch conditions.

    The lifecycle verifier keeps an executed and a skipped alternative per
    unresolved reset; the number of alternatives is bounded, so the resets
    must not multiply into an exponential sequence set, and the per-node
    sequences must stay cheap across the grid and the DFBs. Each condition is
    the conjunction of `condition_leaves` dispatch conditions, so path
    conditions with many variables must stay cheap as well.
    """
    operation_identity = "dfb_conditional_reset_compile_time"
    participant_list = (
        f'<kind = compute, identity = "compute", operation = "{operation_identity}">, '
        f'<kind = data_movement, identity = "reader", operation = "{operation_identity}">, '
        f'<kind = data_movement, identity = "writer", operation = "{operation_identity}">'
    )
    dfb_names = [f"%dfb{dfb_index}" for dfb_index in range(dfb_count)]
    dfb_operands = ", ".join(dfb_names) + " : " + ", ".join([DFB_TYPE] * dfb_count)
    lines = [
        f"module attributes {{ttl.launch_grid = [{grid[0]}, {grid[1]}], "
        "ttl.target_arch = #ttcore.arch<blackhole>} {"
    ]
    for name, kind, identity, thread, extra in (
        ("compute", "compute", "compute", "compute", ""),
        ("reader", "data_movement", "reader", "noc", "ttl.noc_index = 0 : i32, "),
        ("writer", "data_movement", "writer", "noc", "ttl.noc_index = 1 : i32, "),
    ):
        lines.extend(
            [
                f"  func.func @{name}()",
                f"      attributes {{ttl.kernel_thread = #ttkernel.thread<{thread}>,",
                f"                  ttl.logical_kernel = #ttl.logical_kernel<kind = {kind}, "
                f'identity = "{identity}", operation = "{operation_identity}">,',
                f"                  {extra}ttl.base_cta_index = 2 : i32, ttl.crta_indices = []}} {{",
            ]
        )
        for dfb_index, dfb_name in enumerate(dfb_names):
            lines.append(
                f"    {dfb_name} = ttl.bind_cb {{cb_index = {dfb_index}, block_count = 2}} "
                f"{{dfb_id = {dfb_index} : index}} : {DFB_TYPE}"
            )
        lines.append("    %zero = arith.constant 0 : i32")
        for reset_index in range(reset_count):
            for dfb_index, dfb_name in enumerate(dfb_names):
                if name == "reader":
                    lines.extend(
                        [
                            f"    %reserved{reset_index}_{dfb_index} = ttl.cb_reserve "
                            f"{dfb_name} : {PROTOCOL_TYPE} -> {TENSOR_TYPE}",
                            f"    ttl.cb_push {dfb_name} : {PROTOCOL_TYPE}",
                        ]
                    )
                elif name == "writer":
                    lines.extend(
                        [
                            f"    %waited{reset_index}_{dfb_index} = ttl.cb_wait "
                            f"{dfb_name} : {PROTOCOL_TYPE} -> {TENSOR_TYPE}",
                            f"    ttl.cb_pop {dfb_name} : {PROTOCOL_TYPE}",
                        ]
                    )
            for leaf in range(condition_leaves):
                condition_index = reset_index * condition_leaves + leaf
                lines.extend(
                    [
                        f'    %flag{reset_index}_{leaf} = ttl.opaque_call "scalar_predicate" '
                        "template_args [#ttl.external_template_arg<signed_integer, 1>] () "
                        f"{{condition_result = #ttl.dispatch_condition<{condition_index}, i32>, "
                        'header = "predicate.hpp"} : () -> i32',
                        f"    %leaf{reset_index}_{leaf} = arith.cmpi ne, "
                        f"%flag{reset_index}_{leaf}, %zero : i32",
                    ]
                )
            lines.append(
                f"    %active{reset_index}_0 = arith.andi %leaf{reset_index}_0, %leaf{reset_index}_0 : i1"
            )
            for leaf in range(1, condition_leaves):
                lines.append(
                    f"    %active{reset_index}_{leaf} = arith.andi "
                    f"%active{reset_index}_{leaf - 1}, %leaf{reset_index}_{leaf} : i1"
                )
            lines.extend(
                [
                    f"    scf.if %active{reset_index}_{condition_leaves - 1} {{",
                    f"      ttl.reset_dfbs <{reset_index}, participants[{participant_list}]>"
                    f"({dfb_operands})",
                    "    }",
                ]
            )
        lines.extend(["    return", "  }"])
    lines.append("}")
    return "\n".join(lines)


for workload_name, module, timeout_seconds, pipeline in (
    (
        "protocol graph",
        build_stress_module(),
        PROTOCOL_GRAPH_TIMEOUT_SECONDS,
        "ttl-finalize-dfb-indices",
    ),
    (
        "structural order",
        build_structural_order_stress_module(),
        STRUCTURAL_ORDER_TIMEOUT_SECONDS,
        "ttl-finalize-dfb-indices",
    ),
    (
        "conditional reset",
        build_conditional_reset_module(),
        CONDITIONAL_RESET_TIMEOUT_SECONDS,
        "ttl-finalize-dfb-indices,ttl-verify-dfb-spsc,ttl-verify-dfb-lifecycle",
    ),
    (
        "grid conditional reset",
        build_conditional_reset_module(
            GRID_CONDITIONAL_RESET_COUNT,
            GRID_CONDITIONAL_RESET_GRID,
            GRID_CONDITIONAL_RESET_DFB_COUNT,
        ),
        GRID_CONDITIONAL_RESET_TIMEOUT_SECONDS,
        "ttl-finalize-dfb-indices,ttl-verify-dfb-spsc,ttl-verify-dfb-lifecycle",
    ),
    (
        "wide condition reset",
        build_conditional_reset_module(
            GRID_CONDITIONAL_RESET_COUNT,
            GRID_CONDITIONAL_RESET_GRID,
            GRID_CONDITIONAL_RESET_DFB_COUNT,
            WIDE_CONDITION_LEAVES,
        ),
        WIDE_CONDITION_RESET_TIMEOUT_SECONDS,
        "ttl-finalize-dfb-indices,ttl-verify-dfb-spsc,ttl-verify-dfb-lifecycle",
    ),
):
    try:
        result = subprocess.run(
            [
                "ttlang-opt",
                f"-pass-pipeline=builtin.module({pipeline})",
                "-o",
                os.devnull,
            ],
            input=module,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as timeout:
        raise AssertionError(
            f"DFB {workload_name} compilation exceeded {timeout_seconds} seconds"
        ) from timeout

    if result.returncode != 0:
        raise AssertionError(result.stderr)
