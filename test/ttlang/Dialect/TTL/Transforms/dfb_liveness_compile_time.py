# REQUIRES: optimized
# RUN: %python %s

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Bounds protocol-graph and structural-order construction time, and the
# lifecycle verifier's handling of resets repeated in a long generation loop,
# in optimized builds.

import os
import subprocess

PROTOCOL_GRAPH_TIMEOUT_SECONDS = 10
STRUCTURAL_ORDER_TIMEOUT_SECONDS = 5
GENERATION_RESET_TIMEOUT_SECONDS = 10
GENERATION_COUNT = 4096
GENERATION_RESET_COUNT = 6
GENERATION_RESET_GRID = (13, 10)
GENERATION_RESET_DFB_COUNT = 8
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


def build_generation_reset_module(
    generations: int = GENERATION_COUNT,
    resets_per_generation: int = GENERATION_RESET_COUNT,
    grid: tuple[int, int] = GENERATION_RESET_GRID,
    dfb_count: int = GENERATION_RESET_DFB_COUNT,
) -> str:
    """Every participant runs one generation loop with several resets.

    Before each reset the reader pushes a block into every DFB and the writer
    pops it. Each reset restores only the first DFB, so the others carry their
    state across it, and generation 0 pushes and pops one extra block. The
    lifecycle verifier must not grow with the number of generations.
    """
    operation_identity = "dfb_generation_reset_compile_time"
    participant_list = (
        f'<kind = compute, identity = "compute", operation = "{operation_identity}">, '
        f'<kind = data_movement, identity = "reader", operation = "{operation_identity}">, '
        f'<kind = data_movement, identity = "writer", operation = "{operation_identity}">'
    )
    dfb_names = [f"%dfb{dfb_index}" for dfb_index in range(dfb_count)]
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
        lines.extend(
            [
                "    %c0 = arith.constant 0 : index",
                "    %c1 = arith.constant 1 : index",
                f"    %generations = arith.constant {generations} : index",
                "    scf.for %generation = %c0 to %generations step %c1 {",
                "      %first = arith.cmpi eq, %generation, %c0 : index",
            ]
        )

        def transfer(dfb_name, suffix):
            if name == "reader":
                return [
                    f"        %reserved{suffix} = ttl.cb_reserve {dfb_name} : "
                    f"{PROTOCOL_TYPE} -> {TENSOR_TYPE}",
                    f"        ttl.cb_push {dfb_name} : {PROTOCOL_TYPE}",
                ]
            if name == "writer":
                return [
                    f"        %waited{suffix} = ttl.cb_wait {dfb_name} : "
                    f"{PROTOCOL_TYPE} -> {TENSOR_TYPE}",
                    f"        ttl.cb_pop {dfb_name} : {PROTOCOL_TYPE}",
                ]
            return []

        if name != "compute":
            lines.append("      scf.if %first {")
            lines.extend(transfer(dfb_names[0], "_first"))
            lines.append("      }")
        for reset_index in range(resets_per_generation):
            for dfb_index, dfb_name in enumerate(dfb_names):
                lines.extend(
                    line[2:]
                    for line in transfer(dfb_name, f"{reset_index}_{dfb_index}")
                )
            lines.append(
                f"      ttl.reset_dfbs <{reset_index}, participants[{participant_list}]>"
                f"({dfb_names[0]} : {DFB_TYPE})"
            )
        lines.extend(["    }", "    return", "  }"])
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
        "generation loop resets",
        build_generation_reset_module(),
        GENERATION_RESET_TIMEOUT_SECONDS,
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
