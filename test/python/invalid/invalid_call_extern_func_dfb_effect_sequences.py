# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=dynamic not %python %s 2>&1 | FileCheck %s --check-prefix=DYNAMIC
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=negative not %python %s 2>&1 | FileCheck %s --check-prefix=NEGATIVE
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=boolean not %python %s 2>&1 | FileCheck %s --check-prefix=BOOLEAN
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=invalid_concat not %python %s 2>&1 | FileCheck %s --check-prefix=CONCAT

"""Reject invalid static DFB-effect sequence expressions."""

import os

import torch
import ttl
import ttnn

FAKE_HEADER = "/dev/null/fake_shim.hpp"


# DYNAMIC: TTLangCompileError: error: ttl.call_extern_func() DFB effect sequence repetition requires one list expression and one statically resolvable integer
@ttl.operation(grid=(1, 1))
def dynamic_repeat(input_tensor):
    source = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=1)
    node_x, _ = ttl.node(dims=2)
    ttl.call_extern_func(
        FAKE_HEADER,
        "external_stage",
        func_args=[source],
        dfb_effects=node_x * [ttl.DFBEffect.wait(source, tiles=1)],
        kernel=ttl.KernelKind.DATA_MOVEMENT,
    )


# NEGATIVE: TTLangCompileError: error: ttl.call_extern_func() DFB effect sequence repetition count must be nonnegative
@ttl.operation(grid=(1, 1))
def negative_repeat(input_tensor):
    source = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=1)
    ttl.call_extern_func(
        FAKE_HEADER,
        "external_stage",
        func_args=[source],
        dfb_effects=-1 * [ttl.DFBEffect.wait(source, tiles=1)],
        kernel=ttl.KernelKind.DATA_MOVEMENT,
    )


# BOOLEAN: TTLangCompileError: error: ttl.call_extern_func() DFB effect sequence repetition requires one list expression and one statically resolvable integer
@ttl.operation(grid=(1, 1))
def boolean_repeat(input_tensor):
    source = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=1)
    ttl.call_extern_func(
        FAKE_HEADER,
        "external_stage",
        func_args=[source],
        dfb_effects=True * [ttl.DFBEffect.wait(source, tiles=1)],
        kernel=ttl.KernelKind.DATA_MOVEMENT,
    )


# CONCAT: TTLangCompileError: error: ttl.call_extern_func() dfb_effects must be a list expression using list concatenation and nonnegative static repetition
@ttl.operation(grid=(1, 1))
def invalid_concatenation(input_tensor):
    source = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=1)
    ttl.call_extern_func(
        FAKE_HEADER,
        "external_stage",
        func_args=[source],
        dfb_effects=[ttl.DFBEffect.wait(source, tiles=1)] + 1,
        kernel=ttl.KernelKind.DATA_MOVEMENT,
    )


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    cases = {
        "dynamic": dynamic_repeat,
        "negative": negative_repeat,
        "boolean": boolean_repeat,
        "invalid_concat": invalid_concatenation,
    }
    cases[os.environ["CASE"]](input_tensor)
