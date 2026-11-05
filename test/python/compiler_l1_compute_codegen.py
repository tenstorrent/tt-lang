# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s > %t.output 2>&1
# RUN: FileCheck %s --implicit-check-not='CircularBuffer<' --implicit-check-not='cb_wait_front(' --implicit-check-not='cb_reserve_back(' < %t.output

"""Check address-based compute calls emitted from representative Python operations."""

import os
import re
import tempfile
from pathlib import Path

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn

import accumulation_strategy_dst as accumulation
import test_compiler_l1_compute as cases


def make_host_tensor(rows, columns):
    value = torch.zeros((rows, columns), dtype=torch.bfloat16)
    return ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def read_placement(final_ir):
    module = final_ir.read_text()
    offsets = [int(value) for value in re.findall(r"l1_payload_offset = (\d+)", module)]
    arena = int(re.search(r"ttl.l1_arena_bytes = (\d+)", module).group(1))
    return offsets, arena


if __name__ == "__main__":
    options = "--ttl-memory-model=compiler-sram"
    for label, kwargs in (
        ("add", {}),
        ("sub", {"subtract": True}),
        ("mul", {"multiply": True}),
    ):
        print("BEGIN", label, flush=True)
        cases._make_binary(**kwargs)(
            make_host_tensor(224, 32),
            make_host_tensor(224, 32),
            make_host_tensor(224, 32),
            options=options,
        )

    print("BEGIN matmul", flush=True)
    cases._make_matmul(2, 2, 2)(
        make_host_tensor(64, 64),
        make_host_tensor(64, 64),
        make_host_tensor(64, 64),
        options=options,
    )

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        print("BEGIN reduce", flush=True)
        cases._make_reduce(temporary, (1,), 1, False)(
            make_host_tensor(32, 32),
            make_host_tensor(32, 32),
            options=options,
        )
        print("BEGIN broadcast", flush=True)
        cases._make_unary(
            temporary,
            "ttl.block.broadcast(input_block, dims=[1], shape=(1, 1))",
        )(make_host_tensor(224, 32), make_host_tensor(224, 32), options=options)
        print("BEGIN transpose", flush=True)
        cases._make_unary(temporary, "ttl.math.transpose(input_block)")(
            make_host_tensor(224, 32), make_host_tensor(224, 32), options=options
        )

    with tempfile.TemporaryDirectory() as directory:
        final_ir = Path(directory) / "attention.mlir"
        os.environ["TTLANG_FINAL_MLIR"] = str(final_ir)
        print("BEGIN attention", flush=True)
        cases.l1_attention(
            make_host_tensor(32, 32),
            make_host_tensor(32, 32),
            make_host_tensor(32, 32),
            make_host_tensor(32, 32),
            options=options,
        )
        reused_offsets, reused_arena = read_placement(final_ir)
        assert len(reused_offsets) > len(set(reused_offsets))

        print("BEGIN attention without reuse", flush=True)
        cases.l1_attention(
            make_host_tensor(32, 32),
            make_host_tensor(32, 32),
            make_host_tensor(32, 32),
            make_host_tensor(32, 32),
            options=options + " --no-ttl-reuse-user-dfbs",
        )
        distinct_offsets, distinct_arena = read_placement(final_ir)
        assert len(distinct_offsets) == len(reused_offsets)
        assert len(distinct_offsets) == len(set(distinct_offsets))
        assert reused_arena < distinct_arena
        del os.environ["TTLANG_FINAL_MLIR"]

    print("BEGIN destination accumulation", flush=True)
    accumulation.dst_strategy_kernel(
        make_host_tensor(32, 32),
        make_host_tensor(32, 32),
        make_host_tensor(32, 32),
        options=options + " --ttl-accumulation-strategy=dst",
    )


# CHECK-LABEL: BEGIN add
# CHECK: ttlang::l1::target::add_tiles_init(
# CHECK: ttlang::l1::target::add_tiles(
# CHECK-LABEL: BEGIN sub
# CHECK: ttlang::l1::target::sub_tiles_init(
# CHECK: ttlang::l1::target::sub_tiles(
# CHECK-LABEL: BEGIN mul
# CHECK: ttlang::l1::target::mul_tiles_init(
# CHECK: ttlang::l1::target::mul_tiles(
# CHECK-LABEL: BEGIN matmul
# CHECK: l1_compute_context.matmulBlockInit(
# CHECK: l1_compute_context.matmulBlockInitShort(
# CHECK: ttlang::l1::target::matmul_block(
# CHECK-LABEL: BEGIN reduce
# CHECK: l1_compute_context.reduceInit<PoolType::SUM, ReduceDim::REDUCE_ROW>(
# CHECK: ttlang::l1::target::reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
# CHECK-LABEL: BEGIN broadcast
# CHECK: l1_compute_context.broadcastInit<BroadcastType::COL>(
# CHECK: ttlang::l1::target::unary_bcast<BroadcastType::COL>(
# CHECK-LABEL: BEGIN transpose
# CHECK: l1_compute_context.transposeInit(
# CHECK: ttlang::l1::target::transpose_wh_tile(
# CHECK-LABEL: BEGIN attention
# CHECK: exp_tile(
# CHECK: recip_tile(
# CHECK: ttlang::l1::target::matmul_block(
# CHECK-LABEL: BEGIN attention without reuse
# CHECK-LABEL: BEGIN destination accumulation
# CHECK: ttlang::l1::target::binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
# CHECK: ttlang::l1::target::binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
