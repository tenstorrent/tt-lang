# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TT_METAL_DPRINT_CORES=0,0 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Comprehensive runtime test for print() in kernel code.

Tests:
- Scalar: string, integer, float constants; core coordinate variables
- CB details
- Tile print: bf16 and f32
- DST labels after exp and add
- Thread conditioning: math, pack, unpack
- Print inside a loop (f32 kernel)
"""

import os

os.environ["TT_METAL_DPRINT_CORES"] = "0,0"

import torch
import ttnn
import ttl


# =============================================================================
# Kernel 1: bf16 + L1 -- all modes, DST labels, all threads, two inputs
# =============================================================================


@ttl.operation(grid=(1, 1))
def dprint_bf16_kernel(inp, inp2, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    inp2_dfb = ttl.make_dataflow_buffer_like(inp2, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as lhs, inp2_dfb.wait() as rhs, out_dfb.reserve() as o:
            # Scalar: string, integer, float
            print("bf16 compute start")
            print("answer:", 42)
            print("pi:", 3.14)

            # CB details
            print(inp_dfb)

            # Tile print (bf16, prints on pack and unpack threads)
            print(lhs, thread="pack")

            # DST labels after exp and add
            exp_lhs = ttl.exp(lhs)
            # dst printing does not work on hw sim
            # print(_dump_dst_registers=True, label="after first exp")
            exp_rhs = ttl.exp(rhs)
            # print(_dump_dst_registers=True, label="after second exp")
            add_result = exp_lhs + exp_rhs
            # print(_dump_dst_registers=True, label="after add")

            # Thread conditioning: all three
            print("pack thread", thread="pack")
            print("math thread", thread="math")
            print("unpack thread", thread="unpack")

            o.store(add_result)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        print("bf16 dm hello")
        print("core:", x, y)
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()
        with inp2_dfb.reserve() as blk2:
            tx2 = ttl.copy(inp2[0, 0], blk2)
            tx2.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            # Tensor pages print (bf16, from L1 CB after compute push)
            print(blk, num_pages=1)
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# =============================================================================
# Kernel 2: f32 + DRAM -- f32 tile print, print inside loop
# =============================================================================


@ttl.operation(grid=(1, 1), fp32_dest_acc_en=True, dst_full_sync_en=False)
def dprint_f32_kernel(inp_f32, out_f32):
    inp_dfb = ttl.make_dataflow_buffer_like(inp_f32, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out_f32, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        for i in range(2):
            with inp_dfb.wait() as tile, out_dfb.reserve() as o:
                print("f32 loop iter")
                # Tile print (f32)
                print(tile)
                result = ttl.exp(tile)
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        for i in range(2):
            print("f32 dm read iter")
            with inp_dfb.reserve() as blk:
                tx = ttl.copy(inp_f32[0, 0], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        for i in range(2):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out_f32[0, 0])
                tx.wait()


# =============================================================================
# FileCheck patterns -- bf16 kernel
#
# DPRINT output is interleaved across threads (TR0=unpack, TR1=math,
# TR2=pack, NC=NCRISC). Compute prints default to math thread.
# DM prints (NC) run concurrently and may appear anywhere.
# Use CHECK-DAG where ordering is non-deterministic.
# =============================================================================

# Scalar prints and thread conditioning run outside the fused compute
# body and may appear in any order relative to each other and DM output.
# CHECK-DAG: bf16 compute start
# CHECK-DAG: answer: 42
# CHECK-DAG: pi: 3.14
# CHECK-DAG: cb_id
# CHECK-DAG: bf16 dm hello
# CHECK-DAG: core:
# CHECK-DAG: math thread
# CHECK-DAG: unpack thread

# DST and tile dprints are inside the fused compute body (pack thread).
# They appear after the scalar/thread output since the fused compute
# runs all tile ops then stores.
# XXX: === after first exp ===
# XXX: DST[0] (ttl.tile_exp)
# XXX: === after second exp ===
# XXX: DST[0] (ttl.tile_exp)
# XXX: DST[1] (ttl.tile_exp)
# XXX: === after add ===
# XXX: DST[0] (ttl.tile_add)
# XXX: DST[1] (ttl.tile_exp)
# CHECK: pack thread

# Tensor pages print in dm_write (bf16 page from L1 CB)
# The page output starts with page number followed by BF16 values.
# CHECK: 0:

# =============================================================================
# FileCheck patterns -- f32 kernel (runs after bf16 kernel)
# =============================================================================

# Loop prints in compute (two iterations, math thread).
# Use thread prefix to avoid matching codegen DPRINT statements.
# CHECK-DAG: TR1: f32 loop iter
# CHECK-DAG: TR1: f32 loop iter

# Loop prints in DM (NCRISC, two iterations).
# CHECK-DAG: NC: f32 dm read iter
# CHECK-DAG: NC: f32 dm read iter


# =============================================================================
# Test execution
# =============================================================================

device = ttnn.open_device(device_id=0)

try:
    # --- bf16 kernel: L1 tensors ---
    inp_bf16 = ttnn.from_torch(
        torch.randn((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    inp2_bf16 = ttnn.from_torch(
        torch.randn((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out_bf16 = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    dprint_bf16_kernel(inp_bf16, inp2_bf16, out_bf16)

    # --- f32 kernel: DRAM tensors moved to L1 ---
    inp_f32 = ttnn.from_torch(
        torch.randn((32, 32), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_f32 = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inp_f32 = ttnn.to_memory_config(inp_f32, memory_config=ttnn.L1_MEMORY_CONFIG)
    out_f32 = ttnn.to_memory_config(out_f32, memory_config=ttnn.L1_MEMORY_CONFIG)

    dprint_f32_kernel(inp_f32, out_f32)

finally:
    ttnn.close_device(device)
