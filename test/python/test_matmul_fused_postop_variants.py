# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Adversarial tests for matmul + fused post-ops (#476, PR #486).

Tests exercise the generateMatmulCompute lowering with various
post-op patterns, operand orderings, and grid configurations to
verify that no elementwise ops are silently dropped.

Coverage matrix:
  - Unary post-ops: relu, gelu
  - Binary post-ops: mul, add, sub (both operand orderings)
  - Chained post-ops: matmul -> binary -> unary, matmul -> unary -> binary
  - Self-binary: matmul result used as both operands
  - Multi-node grid: fused post-ops with grid="auto"
"""

# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32


# ---------------------------------------------------------------------------
# Kernel builders (single-node, 1x1 grid)
# ---------------------------------------------------------------------------


@ttl.operation(grid=(1, 1))
def matmul_relu_kernel(A, B, out):
    """relu(A @ B)"""
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
            with out_dfb.reserve() as o_blk:
                o_blk.store(ttl.relu(a_blk @ b_blk))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(A[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(B[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def matmul_gelu_kernel(A, B, out):
    """gelu(A @ B)"""
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
            with out_dfb.reserve() as o_blk:
                o_blk.store(ttl.gelu(a_blk @ b_blk))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(A[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(B[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def matmul_relu_scale_bias_kernel(A, B, scale_tile, bias_tile, out):
    """relu(scale * (A @ B) + bias) -- linear layer + activation."""
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=1)
    bi_dfb = ttl.make_dataflow_buffer_like(bias_tile, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            sc_dfb.wait() as sc,
            bi_dfb.wait() as bi,
        ):
            with out_dfb.reserve() as o_blk:
                o_blk.store(ttl.relu(sc * (a_blk @ b_blk) + bi))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(A[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(B[0, 0], blk).wait()
        with sc_dfb.reserve() as blk:
            ttl.copy(scale_tile[0, 0], blk).wait()
        with bi_dfb.reserve() as blk:
            ttl.copy(bias_tile[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def matmul_scale_relu_bias_kernel(A, B, scale_tile, bias_tile, out):
    """scale * relu(A @ B) + bias -- different ordering than linear+activation."""
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=1)
    bi_dfb = ttl.make_dataflow_buffer_like(bias_tile, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            sc_dfb.wait() as sc,
            bi_dfb.wait() as bi,
        ):
            with out_dfb.reserve() as o_blk:
                o_blk.store(sc * ttl.relu(a_blk @ b_blk) + bi)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(A[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(B[0, 0], blk).wait()
        with sc_dfb.reserve() as blk:
            ttl.copy(scale_tile[0, 0], blk).wait()
        with bi_dfb.reserve() as blk:
            ttl.copy(bias_tile[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def matmul_sub_lhs_kernel(A, B, bias_tile, out):
    """(A @ B) - bias -- matmul result on LHS of subtraction."""
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    bi_dfb = ttl.make_dataflow_buffer_like(bias_tile, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            bi_dfb.wait() as bi,
        ):
            with out_dfb.reserve() as o_blk:
                o_blk.store((a_blk @ b_blk) - bi)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(A[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(B[0, 0], blk).wait()
        with bi_dfb.reserve() as blk:
            ttl.copy(bias_tile[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def matmul_sub_rhs_kernel(A, B, bias_tile, out):
    """bias - (A @ B) -- matmul result on RHS of subtraction."""
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    bi_dfb = ttl.make_dataflow_buffer_like(bias_tile, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            bi_dfb.wait() as bi,
        ):
            with out_dfb.reserve() as o_blk:
                o_blk.store(bi - (a_blk @ b_blk))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(A[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(B[0, 0], blk).wait()
        with bi_dfb.reserve() as blk:
            ttl.copy(bias_tile[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def matmul_self_add_kernel(A, B, out):
    """(A @ B) + (A @ B) -- same DST value as both operands of binary op."""
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
            mm = a_blk @ b_blk
            with out_dfb.reserve() as o_blk:
                o_blk.store(mm + mm)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(A[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(B[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def matmul_div_kernel(A, B, scale_tile, out):
    """(A @ B) / scale -- SFPU binary division, different compute than FPU ops."""
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            sc_dfb.wait() as sc,
        ):
            with out_dfb.reserve() as o_blk:
                o_blk.store((a_blk @ b_blk) / sc)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(A[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(B[0, 0], blk).wait()
        with sc_dfb.reserve() as blk:
            ttl.copy(scale_tile[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def matmul_gated_gelu_residual_kernel(
    A, B, scale_tile, bias_tile, gate_tile, residual_tile, out
):
    """residual + gelu(scale * (A @ B) + bias) * gate

    6 post-ops after matmul: mul, add, gelu, mul, add (with residual).
    Exercises deep chain with mixed unary/binary post-ops.
    """
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=1)
    bi_dfb = ttl.make_dataflow_buffer_like(bias_tile, shape=(1, 1), block_count=1)
    gt_dfb = ttl.make_dataflow_buffer_like(gate_tile, shape=(1, 1), block_count=1)
    res_dfb = ttl.make_dataflow_buffer_like(residual_tile, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            sc_dfb.wait() as sc,
            bi_dfb.wait() as bi,
            gt_dfb.wait() as gt,
            res_dfb.wait() as res,
        ):
            with out_dfb.reserve() as o_blk:
                o_blk.store(res + ttl.gelu(sc * (a_blk @ b_blk) + bi) * gt)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(A[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(B[0, 0], blk).wait()
        with sc_dfb.reserve() as blk:
            ttl.copy(scale_tile[0, 0], blk).wait()
        with bi_dfb.reserve() as blk:
            ttl.copy(bias_tile[0, 0], blk).wait()
        with gt_dfb.reserve() as blk:
            ttl.copy(gate_tile[0, 0], blk).wait()
        with res_dfb.reserve() as blk:
            ttl.copy(residual_tile[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


# ---------------------------------------------------------------------------
# Multi-node kernel: fused post-ops with grid="auto"
# ---------------------------------------------------------------------------


def _make_multinode_gated_gelu_residual(m_blk, k_blk, n_blk):
    """residual + gelu(scale * (A @ B) + bias) * gate, multi-node."""

    @ttl.operation(grid="auto")
    def kernel(
        a_tensor: ttnn.Tensor,
        b_tensor: ttnn.Tensor,
        scale_tensor: ttnn.Tensor,
        bias_tensor: ttnn.Tensor,
        gate_tensor: ttnn.Tensor,
        residual_tensor: ttnn.Tensor,
        y_tensor: ttnn.Tensor,
    ) -> None:
        grid_n, grid_m = ttl.grid_size(dims=2)

        m_blocks = a_tensor.shape[0] // TILE // m_blk
        n_blocks = b_tensor.shape[1] // TILE // n_blk
        k_blocks = a_tensor.shape[1] // TILE // k_blk

        m_blocks_per_node = -(-m_blocks // grid_m)
        n_blocks_per_node = -(-n_blocks // grid_n)

        a_dfb = ttl.make_dataflow_buffer_like(
            a_tensor, shape=(m_blk, k_blk), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b_tensor, shape=(k_blk, n_blk), block_count=2
        )
        sc_dfb = ttl.make_dataflow_buffer_like(
            scale_tensor, shape=(m_blk, n_blk), block_count=1
        )
        bi_dfb = ttl.make_dataflow_buffer_like(
            bias_tensor, shape=(m_blk, n_blk), block_count=1
        )
        gt_dfb = ttl.make_dataflow_buffer_like(
            gate_tensor, shape=(m_blk, n_blk), block_count=1
        )
        res_dfb = ttl.make_dataflow_buffer_like(
            residual_tensor, shape=(m_blk, n_blk), block_count=1
        )
        acc_dfb = ttl.make_dataflow_buffer_like(
            y_tensor, shape=(m_blk, n_blk), block_count=2
        )
        y_dfb = ttl.make_dataflow_buffer_like(
            y_tensor, shape=(m_blk, n_blk), block_count=2
        )

        @ttl.datamovement()
        def read():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_m
                if mb < m_blocks:
                    sm = mb * m_blk
                    em = (mb + 1) * m_blk
                    for local_n in range(n_blocks_per_node):
                        nb = node_n * n_blocks_per_node + local_n
                        if nb < n_blocks:
                            sn = nb * n_blk
                            en = (nb + 1) * n_blk
                            with sc_dfb.reserve() as blk:
                                ttl.copy(scale_tensor[sm:em, sn:en], blk).wait()
                            with bi_dfb.reserve() as blk:
                                ttl.copy(bias_tensor[sm:em, sn:en], blk).wait()
                            with gt_dfb.reserve() as blk:
                                ttl.copy(gate_tensor[sm:em, sn:en], blk).wait()
                            with res_dfb.reserve() as blk:
                                ttl.copy(residual_tensor[sm:em, sn:en], blk).wait()
                            for kb in range(k_blocks):
                                sk = kb * k_blk
                                ek = (kb + 1) * k_blk
                                with (
                                    a_dfb.reserve() as a_blk,
                                    b_dfb.reserve() as b_blk,
                                ):
                                    ttl.copy(a_tensor[sm:em, sk:ek], a_blk).wait()
                                    ttl.copy(b_tensor[sk:ek, sn:en], b_blk).wait()

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_m
                if mb < m_blocks:
                    for local_n in range(n_blocks_per_node):
                        nb = node_n * n_blocks_per_node + local_n
                        if nb < n_blocks:
                            # First K iteration
                            with (
                                a_dfb.wait() as a_blk,
                                b_dfb.wait() as b_blk,
                            ):
                                with acc_dfb.reserve() as acc_blk:
                                    acc_blk.store(a_blk @ b_blk)
                            # Remaining K iterations: accumulate
                            for _ in range(k_blocks - 1):
                                with (
                                    a_dfb.wait() as a_blk,
                                    b_dfb.wait() as b_blk,
                                    acc_dfb.wait() as pre_acc,
                                ):
                                    with acc_dfb.reserve() as acc_blk:
                                        acc_blk.store(pre_acc + a_blk @ b_blk)
                            # Apply full post-op chain after accumulation
                            with (
                                acc_dfb.wait() as acc_blk,
                                sc_dfb.wait() as sc,
                                bi_dfb.wait() as bi,
                                gt_dfb.wait() as gt,
                                res_dfb.wait() as res,
                            ):
                                with y_dfb.reserve() as y_blk:
                                    y_blk.store(res + ttl.gelu(sc * acc_blk + bi) * gt)

        @ttl.datamovement()
        def write():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_m
                if mb < m_blocks:
                    sm = mb * m_blk
                    em = (mb + 1) * m_blk
                    for local_n in range(n_blocks_per_node):
                        nb = node_n * n_blocks_per_node + local_n
                        if nb < n_blocks:
                            sn = nb * n_blk
                            en = (nb + 1) * n_blk
                            with y_dfb.wait() as y_blk:
                                ttl.copy(y_blk, y_tensor[sm:em, sn:en]).wait()

    return kernel


def _make_multinode_matmul_relu_bias(m_blk, k_blk, n_blk):
    """relu(A @ B) + bias with multi-node grid and K-loop accumulation."""

    @ttl.operation(grid="auto")
    def kernel(
        a_tensor: ttnn.Tensor,
        b_tensor: ttnn.Tensor,
        bias_tensor: ttnn.Tensor,
        y_tensor: ttnn.Tensor,
    ) -> None:
        grid_n, grid_m = ttl.grid_size(dims=2)

        m_blocks = a_tensor.shape[0] // TILE // m_blk
        n_blocks = b_tensor.shape[1] // TILE // n_blk
        k_blocks = a_tensor.shape[1] // TILE // k_blk

        m_blocks_per_node = -(-m_blocks // grid_m)
        n_blocks_per_node = -(-n_blocks // grid_n)

        a_dfb = ttl.make_dataflow_buffer_like(
            a_tensor, shape=(m_blk, k_blk), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b_tensor, shape=(k_blk, n_blk), block_count=2
        )
        bias_dfb = ttl.make_dataflow_buffer_like(
            bias_tensor, shape=(m_blk, n_blk), block_count=2
        )
        acc_dfb = ttl.make_dataflow_buffer_like(
            y_tensor, shape=(m_blk, n_blk), block_count=2
        )
        y_dfb = ttl.make_dataflow_buffer_like(
            y_tensor, shape=(m_blk, n_blk), block_count=2
        )

        @ttl.datamovement()
        def read():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_m
                if mb < m_blocks:
                    sm = mb * m_blk
                    em = (mb + 1) * m_blk
                    for local_n in range(n_blocks_per_node):
                        nb = node_n * n_blocks_per_node + local_n
                        if nb < n_blocks:
                            sn = nb * n_blk
                            en = (nb + 1) * n_blk
                            with bias_dfb.reserve() as bi_blk:
                                ttl.copy(bias_tensor[sm:em, sn:en], bi_blk).wait()
                            for kb in range(k_blocks):
                                sk = kb * k_blk
                                ek = (kb + 1) * k_blk
                                with (
                                    a_dfb.reserve() as a_blk,
                                    b_dfb.reserve() as b_blk,
                                ):
                                    ttl.copy(a_tensor[sm:em, sk:ek], a_blk).wait()
                                    ttl.copy(b_tensor[sk:ek, sn:en], b_blk).wait()

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_m
                if mb < m_blocks:
                    for local_n in range(n_blocks_per_node):
                        nb = node_n * n_blocks_per_node + local_n
                        if nb < n_blocks:
                            # First K iteration: matmul without accumulator
                            with (
                                a_dfb.wait() as a_blk,
                                b_dfb.wait() as b_blk,
                            ):
                                with acc_dfb.reserve() as acc_blk:
                                    acc_blk.store(a_blk @ b_blk)
                            # Remaining K iterations: accumulate
                            for _ in range(k_blocks - 1):
                                with (
                                    a_dfb.wait() as a_blk,
                                    b_dfb.wait() as b_blk,
                                    acc_dfb.wait() as pre_acc,
                                ):
                                    with acc_dfb.reserve() as acc_blk:
                                        acc_blk.store(pre_acc + a_blk @ b_blk)
                            # Apply relu + bias after full accumulation
                            with (
                                acc_dfb.wait() as acc_blk,
                                bias_dfb.wait() as bi,
                            ):
                                with y_dfb.reserve() as y_blk:
                                    y_blk.store(ttl.relu(acc_blk) + bi)

        @ttl.datamovement()
        def write():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_m
                if mb < m_blocks:
                    sm = mb * m_blk
                    em = (mb + 1) * m_blk
                    for local_n in range(n_blocks_per_node):
                        nb = node_n * n_blocks_per_node + local_n
                        if nb < n_blocks:
                            sn = nb * n_blk
                            en = (nb + 1) * n_blk
                            with y_dfb.wait() as y_blk:
                                ttl.copy(y_blk, y_tensor[sm:em, sn:en]).wait()

    return kernel


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _random_inputs(device, seed=42):
    """Return A, B torch tensors and their DRAM copies."""
    torch.manual_seed(seed)
    a_pt = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    b_pt = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    return a_pt, b_pt, to_dram(a_pt, device), to_dram(b_pt, device)


def _scalar_tile(val, device):
    """Create a uniform tile filled with a scalar value."""
    pt = torch.full((TILE, TILE), float(val), dtype=torch.bfloat16)
    return pt, to_dram(pt, device)


def _run_and_compare(result_tt, golden_pt, threshold=0.999):
    result = ttnn.to_torch(result_tt).reshape(TILE, TILE).float()
    golden = golden_pt.to(torch.bfloat16).float()
    assert_pcc(golden, result, threshold=threshold)


# ---------------------------------------------------------------------------
# Tests: unary post-ops
# ---------------------------------------------------------------------------


class TestUnaryPostOps:
    def test_matmul_relu(self, device):
        """relu(A @ B) -- most common ML pattern."""
        a_pt, b_pt, a_dev, b_dev = _random_inputs(device)
        out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

        matmul_relu_kernel(a_dev, b_dev, out_dev)

        golden = torch.relu(a_pt.float() @ b_pt.float())
        _run_and_compare(out_dev, golden)

    def test_matmul_gelu(self, device):
        """gelu(A @ B) -- different SFPU function, tests init switching."""
        a_pt, b_pt, a_dev, b_dev = _random_inputs(device, seed=123)
        out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

        matmul_gelu_kernel(a_dev, b_dev, out_dev)

        golden = torch.nn.functional.gelu(a_pt.float() @ b_pt.float())
        _run_and_compare(out_dev, golden, threshold=0.998)


# ---------------------------------------------------------------------------
# Tests: chained post-ops (unary + binary combinations)
# ---------------------------------------------------------------------------


class TestChainedPostOps:
    def test_relu_scale_bias(self, device):
        """relu(scale * (A @ B) + bias) -- linear layer + activation."""
        a_pt, b_pt, a_dev, b_dev = _random_inputs(device)
        sc_pt, sc_dev = _scalar_tile(0.5, device)
        bi_pt, bi_dev = _scalar_tile(10.0, device)
        out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

        matmul_relu_scale_bias_kernel(a_dev, b_dev, sc_dev, bi_dev, out_dev)

        golden = torch.relu(0.5 * (a_pt.float() @ b_pt.float()) + 10.0)
        _run_and_compare(out_dev, golden)

    def test_scale_relu_bias(self, device):
        """scale * relu(A @ B) + bias -- unary between two binary ops."""
        a_pt, b_pt, a_dev, b_dev = _random_inputs(device)
        sc_pt, sc_dev = _scalar_tile(2.0, device)
        bi_pt, bi_dev = _scalar_tile(-5.0, device)
        out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

        matmul_scale_relu_bias_kernel(a_dev, b_dev, sc_dev, bi_dev, out_dev)

        golden = 2.0 * torch.relu(a_pt.float() @ b_pt.float()) + (-5.0)
        _run_and_compare(out_dev, golden)


# ---------------------------------------------------------------------------
# Tests: subtraction operand ordering
# ---------------------------------------------------------------------------


class TestSubtractionOrdering:
    def test_matmul_minus_bias(self, device):
        """(A @ B) - bias -- matmul on LHS."""
        a_pt, b_pt, a_dev, b_dev = _random_inputs(device)
        bi_pt, bi_dev = _scalar_tile(7.0, device)
        out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

        matmul_sub_lhs_kernel(a_dev, b_dev, bi_dev, out_dev)

        golden = (a_pt.float() @ b_pt.float()) - 7.0
        _run_and_compare(out_dev, golden)

    def test_bias_minus_matmul(self, device):
        """bias - (A @ B) -- matmul on RHS, reversed operand order."""
        a_pt, b_pt, a_dev, b_dev = _random_inputs(device)
        bi_pt, bi_dev = _scalar_tile(7.0, device)
        out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

        matmul_sub_rhs_kernel(a_dev, b_dev, bi_dev, out_dev)

        golden = 7.0 - (a_pt.float() @ b_pt.float())
        _run_and_compare(out_dev, golden)


# ---------------------------------------------------------------------------
# Tests: self-binary and division
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_self_add(self, device):
        """(A @ B) + (A @ B) -- same DST value as both binary operands."""
        a_pt, b_pt, a_dev, b_dev = _random_inputs(device)
        out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

        matmul_self_add_kernel(a_dev, b_dev, out_dev)

        golden = 2.0 * (a_pt.float() @ b_pt.float())
        _run_and_compare(out_dev, golden)

    def test_div_postop(self, device):
        """(A @ B) / scale -- SFPU binary division, different compute than FPU."""
        a_pt, b_pt, a_dev, b_dev = _random_inputs(device)
        sc_pt, sc_dev = _scalar_tile(4.0, device)
        out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

        matmul_div_kernel(a_dev, b_dev, sc_dev, out_dev)

        golden = (a_pt.float() @ b_pt.float()) / 4.0
        _run_and_compare(out_dev, golden, threshold=0.998)

    def test_gated_gelu_residual(self, device):
        """residual + gelu(scale * (A @ B) + bias) * gate

        6 post-ops: mul, add, gelu, mul, add. Exercises deep chain with
        mixed unary/binary post-ops — a realistic gated linear unit pattern.
        """
        a_pt, b_pt, a_dev, b_dev = _random_inputs(device)
        scale_val, bias_val, gate_val = 0.5, 1.0, 0.8
        sc_pt, sc_dev = _scalar_tile(scale_val, device)
        bi_pt, bi_dev = _scalar_tile(bias_val, device)
        gt_pt, gt_dev = _scalar_tile(gate_val, device)
        # Use non-trivial residual to distinguish from zero
        torch.manual_seed(99)
        res_pt = torch.randn(TILE, TILE, dtype=torch.bfloat16)
        res_dev = to_dram(res_pt, device)
        out_dev = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

        matmul_gated_gelu_residual_kernel(
            a_dev, b_dev, sc_dev, bi_dev, gt_dev, res_dev, out_dev
        )

        mm = a_pt.float() @ b_pt.float()
        golden = (
            res_pt.float()
            + torch.nn.functional.gelu(scale_val * mm + bias_val) * gate_val
        )
        _run_and_compare(out_dev, golden, threshold=0.998)


# ---------------------------------------------------------------------------
# Tests: multi-node grid
# ---------------------------------------------------------------------------


# (M_blk, K_blk, N_blk) — block dimensions in tiles.
# Shapes where m_blk * n_blk > 8 trigger DST subblocking (bf16 capacity = 8).
MULTINODE_SHAPES = [
    pytest.param((1, 1, 1), id="1x1x1"),
    pytest.param((2, 2, 2), id="2x2x2"),
    pytest.param((1, 2, 4), id="1x2x4"),
    pytest.param((4, 2, 1), id="4x2x1"),
    pytest.param((4, 2, 4), id="4x2x4-subblock"),  # 16 output tiles
    pytest.param((3, 2, 4), id="3x2x4-subblock"),  # 12 output tiles
    pytest.param((4, 4, 4), id="4x4x4-subblock"),  # 16 output tiles, larger K
]


class TestMultiNode:
    @pytest.mark.parametrize("block_shape", MULTINODE_SHAPES)
    def test_multinode_relu_bias(self, device, block_shape):
        """relu(A @ B) + bias with grid="auto" and K-loop accumulation."""
        m_blk, k_blk, n_blk = block_shape
        total_m = m_blk * 4 * TILE
        total_k = k_blk * 2 * TILE
        total_n = n_blk * 4 * TILE

        torch.manual_seed(42)
        a_pt = torch.randn((total_m, total_k), dtype=torch.bfloat16)
        b_pt = torch.randn((total_k, total_n), dtype=torch.bfloat16)
        bias_pt = torch.randn((total_m, total_n), dtype=torch.bfloat16)

        a_dev = to_dram(a_pt, device)
        b_dev = to_dram(b_pt, device)
        bias_dev = to_dram(bias_pt, device)
        y_dev = to_dram(torch.zeros((total_m, total_n), dtype=torch.bfloat16), device)

        kernel = _make_multinode_matmul_relu_bias(m_blk, k_blk, n_blk)
        kernel(a_dev, b_dev, bias_dev, y_dev)

        result = ttnn.to_torch(y_dev).float()
        expected = torch.relu(a_pt.float() @ b_pt.float()) + bias_pt.float()
        assert_pcc(expected, result, threshold=0.99)

    @pytest.mark.parametrize("block_shape", MULTINODE_SHAPES)
    def test_multinode_gated_gelu_residual(self, device, block_shape):
        """residual + gelu(scale * (A @ B) + bias) * gate, multi-node."""
        m_blk, k_blk, n_blk = block_shape
        total_m = m_blk * 4 * TILE
        total_k = k_blk * 2 * TILE
        total_n = n_blk * 4 * TILE

        torch.manual_seed(42)
        a_pt = torch.randn((total_m, total_k), dtype=torch.bfloat16)
        b_pt = torch.randn((total_k, total_n), dtype=torch.bfloat16)
        scale_pt = torch.full((total_m, total_n), 0.5, dtype=torch.bfloat16)
        bias_pt = torch.randn((total_m, total_n), dtype=torch.bfloat16)
        gate_pt = torch.full((total_m, total_n), 0.8, dtype=torch.bfloat16)
        residual_pt = torch.randn((total_m, total_n), dtype=torch.bfloat16)

        a_dev = to_dram(a_pt, device)
        b_dev = to_dram(b_pt, device)
        scale_dev = to_dram(scale_pt, device)
        bias_dev = to_dram(bias_pt, device)
        gate_dev = to_dram(gate_pt, device)
        residual_dev = to_dram(residual_pt, device)
        y_dev = to_dram(torch.zeros((total_m, total_n), dtype=torch.bfloat16), device)

        kernel = _make_multinode_gated_gelu_residual(m_blk, k_blk, n_blk)
        kernel(a_dev, b_dev, scale_dev, bias_dev, gate_dev, residual_dev, y_dev)

        mm = a_pt.float() @ b_pt.float()
        expected = (
            residual_pt.float()
            + torch.nn.functional.gelu(scale_pt.float() * mm + bias_pt.float())
            * gate_pt.float()
        )
        result = ttnn.to_torch(y_dev).float()
        assert_pcc(expected, result, threshold=0.99)
