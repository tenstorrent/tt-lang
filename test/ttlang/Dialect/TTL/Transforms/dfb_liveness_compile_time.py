# RUN: %python %s ttlang-opt

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


COMPILATION_TIMEOUT_SECONDS = 10
TRANSACTION_COUNT = 500
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


compiler = sys.argv[1]
try:
    result = subprocess.run(
        [
            compiler,
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
        f"DFB liveness compilation exceeded {COMPILATION_TIMEOUT_SECONDS} seconds"
    ) from timeout

if result.returncode != 0:
    raise AssertionError(result.stderr)
