# REQUIRES: optimized
# RUN: %python %s

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Bounds cumulative external-effect ordering time in optimized builds.

import os
import subprocess


COMPILATION_TIMEOUT_SECONDS = 10
TRANSACTION_COUNT = 600
DFB_TYPE = "!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>"


def build_stress_module() -> str:
    lines = [
        "module attributes {ttl.launch_grid = [8, 8], "
        "ttl.target_arch = #ttcore.arch<blackhole>} {"
    ]
    function_specs = (
        ("cumulative_producer", 0, "reserve", "push"),
        ("cumulative_consumer", 1, "wait", "pop"),
    )
    for function_name, noc_index, acquire_effect, release_effect in function_specs:
        effects = []
        for _ in range(TRANSACTION_COUNT):
            effects.extend(
                [
                    f"#ttl.dfb_protocol_effect<{acquire_effect}, 0, 1>",
                    f"#ttl.dfb_protocol_effect<{release_effect}, 0, 1>",
                ]
            )
        lines.extend(
            [
                f"  func.func @{function_name}()",
                "      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,",
                f"                  ttl.noc_index = {noc_index} : i32,",
                "                  ttl.base_cta_index = 1 : i32, "
                "ttl.crta_indices = []} {",
                "    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} "
                f"{{dfb_id = 0 : index}} : {DFB_TYPE}",
                f'    ttl.opaque_call "{function_name}"',
                f"        dfb_dependencies(%dfb : {DFB_TYPE})",
                "        dfb_effects ["
                + ",\n                     ".join(effects)
                + "]",
                '        () {header = "effects.hpp"} : () -> ()',
                "    return",
                "  }",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


try:
    result = subprocess.run(
        [
            "ttlang-opt",
            "-pass-pipeline=builtin.module(ttl-finalize-dfb-indices)",
            "-o",
            os.devnull,
        ],
        input=build_stress_module(),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=COMPILATION_TIMEOUT_SECONDS,
        check=False,
    )
except subprocess.TimeoutExpired as timeout:
    raise AssertionError(
        f"DFB cumulative protocol compilation exceeded "
        f"{COMPILATION_TIMEOUT_SECONDS} seconds"
    ) from timeout

if result.returncode != 0:
    raise AssertionError(result.stderr)
