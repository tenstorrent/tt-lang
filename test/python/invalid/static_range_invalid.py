# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=dynamic_bound not %python %s 2>&1 | FileCheck %s --check-prefix=DYNAMIC
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=non_integer not %python %s 2>&1 | FileCheck %s --check-prefix=NONINTEGER
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=keywords not %python %s 2>&1 | FileCheck %s --check-prefix=KEYWORDS
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=zero_step not %python %s 2>&1 | FileCheck %s --check-prefix=ZEROSTEP
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=arity not %python %s 2>&1 | FileCheck %s --check-prefix=ARITY
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=target not %python %s 2>&1 | FileCheck %s --check-prefix=TARGET
# RUN: env TTLANG_COMPILE_ONLY=1 CASE=for_else not %python %s 2>&1 | FileCheck %s --check-prefix=FORELSE

"""Invalid coverage for explicitly unrolled TT-Lang loops."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch  # noqa: E402
import ttl  # noqa: E402


class BFloat16Tensor:
    dtype = torch.bfloat16


def output_dfb():
    return ttl.make_dataflow_buffer_like(BFloat16Tensor(), shape=(1, 1), block_count=2)


# DYNAMIC: ttl.static_range() argument 1 must be a compile-time integer
@ttl.operation(grid=(1, 1))
def dynamic_bound():
    result_dfb = output_dfb()

    @ttl.compute()
    def compute():
        column_coordinate, row_coordinate = ttl.node(dims=2)
        for iteration_index in ttl.static_range(column_coordinate):
            with result_dfb.reserve() as result_block:
                result_block.store(result_block)


# NONINTEGER: ttl.static_range() argument 1 must be a compile-time integer
@ttl.operation(grid=(1, 1))
def non_integer_bound():
    result_dfb = output_dfb()

    @ttl.compute()
    def compute():
        for iteration_index in ttl.static_range(1.5):
            with result_dfb.reserve() as result_block:
                result_block.store(result_block)


# KEYWORDS: ttl.static_range() does not accept keyword arguments
@ttl.operation(grid=(1, 1))
def keyword_arguments():
    result_dfb = output_dfb()

    @ttl.compute()
    def compute():
        for iteration_index in ttl.static_range(start=0, stop=1):
            with result_dfb.reserve() as result_block:
                result_block.store(result_block)


# ZEROSTEP: ttl.static_range(): range() arg 3 must not be zero
@ttl.operation(grid=(1, 1))
def zero_step():
    result_dfb = output_dfb()

    @ttl.compute()
    def compute():
        for iteration_index in ttl.static_range(0, 1, 0):
            with result_dfb.reserve() as result_block:
                result_block.store(result_block)


# ARITY: ttl.static_range() requires one to three integer arguments
@ttl.operation(grid=(1, 1))
def invalid_arity():
    result_dfb = output_dfb()

    @ttl.compute()
    def compute():
        for iteration_index in ttl.static_range():
            with result_dfb.reserve() as result_block:
                result_block.store(result_block)


# TARGET: ttl.static_range() requires a simple loop variable
@ttl.operation(grid=(1, 1))
def invalid_target():
    result_dfb = output_dfb()

    @ttl.compute()
    def compute():
        for first_index, second_index in ttl.static_range(1):
            with result_dfb.reserve() as result_block:
                result_block.store(result_block)


# FORELSE: ttl.static_range() does not support a for-else clause
@ttl.operation(grid=(1, 1))
def invalid_for_else():
    result_dfb = output_dfb()

    @ttl.compute()
    def compute():
        for iteration_index in ttl.static_range(1):
            with result_dfb.reserve() as result_block:
                result_block.store(result_block)
        else:
            pass


if __name__ == "__main__":
    match os.environ["CASE"]:
        case "dynamic_bound":
            dynamic_bound()
        case "non_integer":
            non_integer_bound()
        case "keywords":
            keyword_arguments()
        case "zero_step":
            zero_step()
        case "arity":
            invalid_arity()
        case "target":
            invalid_target()
        case "for_else":
            invalid_for_else()
        case unknown:
            raise ValueError(f"unknown CASE={unknown}")
