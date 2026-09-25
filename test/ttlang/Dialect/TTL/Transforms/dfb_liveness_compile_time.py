# REQUIRES: optimized
# RUN: %python %s

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Bounds protocol-graph, structural-order, and launch-location analysis time in
# optimized builds.

import os
import subprocess

PROTOCOL_GRAPH_TIMEOUT_SECONDS = 10
STRUCTURAL_ORDER_TIMEOUT_SECONDS = 5
SCF_RESULT_CHAIN_TIMEOUT_SECONDS = 3
TRANSACTION_COUNT = 500
RESET_COUNT_PER_BRANCH = 128
RESET_NESTING_DEPTH = 8
SCF_RESULT_CHAIN_DEPTH = 22
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


def build_scf_result_chain_stress_module() -> str:
    lines = ["module attributes {ttl.launch_grid = [13 : i64, 10 : i64]} {"]
    for function_name, thread_kind, is_producer in (
        ("producer", "noc", True),
        ("consumer", "compute", False),
    ):
        lines.extend(
            [
                f"  func.func @{function_name}() attributes "
                f"{{ttl.kernel_thread = #ttkernel.thread<{thread_kind}>}} {{",
                "    %buffer = ttl.bind_cb {cb_index = 0, block_count = 2} "
                f"{{dfb_id = 47 : index}} : {DFB_TYPE}",
                "    %core_x = ttl.core_x : index",
                "    %zero = arith.constant 0 : index",
                "    %one = arith.constant 1 : index",
                "    %two = arith.constant 2 : index",
                "    %is_x_zero = arith.cmpi eq, %core_x, %zero : index",
            ]
        )
        # Overlapping dependencies require one evaluator cache per location.
        for chain_index in range(SCF_RESULT_CHAIN_DEPTH):
            prior_count = "%two" if chain_index == 0 else f"%count{chain_index - 1}"
            condition = "%is_x_zero" if chain_index == 0 else f"%condition{chain_index}"
            selected_count = "%one" if chain_index < 2 else f"%count{chain_index - 2}"
            if chain_index:
                lines.append(
                    f"    {condition} = arith.cmpi eq, {prior_count}, %one : index"
                )
            lines.extend(
                [
                    f"    %count{chain_index} = scf.if {condition} -> index {{",
                    f"      scf.yield {selected_count} : index",
                    "    } else {",
                    "      scf.yield %one : index",
                    "    }",
                ]
            )
        lines.append(
            "    scf.for %iteration = %zero to "
            f"%count{SCF_RESULT_CHAIN_DEPTH - 1} step %one {{"
        )
        if is_producer:
            lines.extend(
                [
                    f"      %reserved = ttl.cb_reserve %buffer : {PROTOCOL_TYPE} -> {TENSOR_TYPE}",
                    f"      ttl.cb_push %buffer : {PROTOCOL_TYPE}",
                ]
            )
        else:
            lines.extend(
                [
                    f"      %waited = ttl.cb_wait %buffer : {PROTOCOL_TYPE} -> {TENSOR_TYPE}",
                    f"      ttl.cb_pop %buffer : {PROTOCOL_TYPE}",
                ]
            )
        lines.extend(["    }", "    func.return", "  }"])
    lines.append("}")
    return "\n".join(lines)


for workload_name, module, timeout_seconds in (
    ("protocol graph", build_stress_module(), PROTOCOL_GRAPH_TIMEOUT_SECONDS),
    (
        "structural order",
        build_structural_order_stress_module(),
        STRUCTURAL_ORDER_TIMEOUT_SECONDS,
    ),
    (
        "chained scf.if results",
        build_scf_result_chain_stress_module(),
        SCF_RESULT_CHAIN_TIMEOUT_SECONDS,
    ),
):
    try:
        result = subprocess.run(
            [
                "ttlang-opt",
                "-pass-pipeline=builtin.module(ttl-finalize-dfb-indices)",
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
