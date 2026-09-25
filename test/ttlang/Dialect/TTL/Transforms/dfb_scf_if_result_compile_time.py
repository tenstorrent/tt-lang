# REQUIRES: optimized
# RUN: %python %s

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Bounds launch-domain evaluation of chained `scf.if` results in optimized
# builds. Each result's condition reads the previous result and each yield
# reads the result two steps back, so an evaluator without a shared cache
# re-evaluates a subexpression tree that doubles with the chain depth. The
# same chain must still resolve per launch node: the surplus variant is
# rejected and the balanced variant is accepted.

import os
import subprocess

TIMEOUT_SECONDS = 5
CHAIN_DEPTH = 24
LOOP_COUNT = 8
DFB_TYPE = "!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>"
PROTOCOL_TYPE = "<[1, 1], !ttcore.tile<32x32, bf16>, 1>"
TENSOR_TYPE = "tensor<1x1x!ttcore.tile<32x32, bf16>>"


def add_kernel(lines, name, thread, loop_count, surplus):
    lines.extend(
        [
            f"  func.func @{name}() attributes "
            f"{{ttl.kernel_thread = #ttkernel.thread<{thread}>}} {{",
            f"    %buffer = ttl.bind_cb {{cb_index = 0, block_count = 1}} "
            f"{{dfb_id = 47 : index}} : {DFB_TYPE}",
            "    %core_x = ttl.core_x : index",
            "    %zero = arith.constant 0 : index",
            "    %one = arith.constant 1 : index",
            "    %two = arith.constant 2 : index",
            "    %is_x_zero = arith.cmpi eq, %core_x, %zero : index",
        ]
    )
    # count0 is 2 on the x == 0 column and 1 elsewhere; every later result
    # selects between the two previous results, so the chain keeps the value.
    for index in range(CHAIN_DEPTH):
        if index == 0:
            condition, then_value, else_value = "%is_x_zero", "%two", "%one"
        else:
            previous = f"%count{index - 1}"
            before = f"%count{index - 2}" if index >= 2 else "%one"
            condition = f"%condition{index}"
            lines.append(f"    {condition} = arith.cmpi eq, {previous}, %two : index")
            then_value, else_value = previous, before
        lines.extend(
            [
                f"    %count{index} = scf.if {condition} -> index {{",
                f"      scf.yield {then_value} : index",
                "    } else {",
                f"      scf.yield {else_value} : index",
                "    }",
            ]
        )
    bound = f"%count{CHAIN_DEPTH - 1}"
    # The producer loops to the chain's value; the consumer loops to the same
    # value or, for the surplus variant, one less on the x == 0 column.
    if name == "consumer" and surplus:
        lines.append(f"    {bound}_less = arith.subi {bound}, %one : index")
        lines.append(f"    %is_two = arith.cmpi eq, {bound}, %two : index")
        lines.append(
            f"    %consumer_bound = arith.select %is_two, {bound}_less, {bound} : index"
        )
        bound = "%consumer_bound"
    for loop_index in range(loop_count):
        lines.append(
            f"    scf.for %iteration{loop_index} = %zero to {bound} step %one {{"
        )
        if name == "producer":
            lines.extend(
                [
                    f"      %reserved{loop_index} = ttl.cb_reserve %buffer : "
                    f"{PROTOCOL_TYPE} -> {TENSOR_TYPE}",
                    f"      ttl.cb_push %buffer : {PROTOCOL_TYPE}",
                ]
            )
        else:
            lines.extend(
                [
                    f"      %waited{loop_index} = ttl.cb_wait %buffer : "
                    f"{PROTOCOL_TYPE} -> {TENSOR_TYPE}",
                    f"      ttl.cb_pop %buffer : {PROTOCOL_TYPE}",
                ]
            )
        lines.append("    }")
    lines.extend(["    func.return", "  }"])


def build_module(surplus):
    lines = ["module attributes {ttl.launch_grid = [13 : i64, 10 : i64]} {"]
    add_kernel(lines, "producer", "noc", LOOP_COUNT, surplus)
    add_kernel(lines, "consumer", "compute", LOOP_COUNT, surplus)
    lines.append("}")
    return "\n".join(lines) + "\n"


for surplus in (False, True):
    module = build_module(surplus)
    try:
        result = subprocess.run(
            [
                "ttlang-opt",
                "-pass-pipeline=builtin.module(ttl-finalize-dfb-indices,"
                "ttl-verify-dfb-spsc,ttl-verify-dfb-lifecycle)",
                "-o",
                os.devnull,
            ],
            input=module,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as timeout:
        raise AssertionError(
            f"chained scf.if evaluation exceeded {TIMEOUT_SECONDS} seconds"
        ) from timeout
    if surplus:
        expected = (
            "the producer pushes 16 block(s) per launch and the consumer pops "
            "8, leaving 8 outstanding block(s) for capacity 1"
        )
        if result.returncode == 0 or expected not in result.stderr:
            raise AssertionError(
                "the chained bound was not resolved per launch node:\n" + result.stderr
            )
    elif result.returncode != 0:
        raise AssertionError(result.stderr)
