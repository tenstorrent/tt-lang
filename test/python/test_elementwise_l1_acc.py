# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Elementwise L1 accumulation via += across a loop.

`+=` on a reserved block emits ttl.store with {accumulate}, which the
compiler annotates and lowers to L1 packer accumulation
(`pack_reconfig_l1_acc` guards around the loop). Existing coverage in
`test_matmul_l1_acc.py` exercises the matmul and passthrough RHS shapes;
this file exercises non-matmul RHS shapes: FPU binary, SFPU unary, fused
binary+unary, op-chains, subblocked elementwise, mixed FPU/matmul
transitions, and the `.store()` followed by `+=` accumulate pattern.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v --tb=short

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32


def _run_acc_test(kernel, inputs, out_shape, golden, device, threshold=0.999):
    """Shared harness: move inputs to DRAM, run kernel, compare against golden."""
    dev_inputs = [to_dram(t, device) for t in inputs]
    out_dev = to_dram(torch.zeros(*out_shape, dtype=torch.bfloat16), device)
    kernel(*dev_inputs, out_dev)
    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result, threshold=threshold)


def _make_binary_add_kernel():
    """out = sum over K of (a[k] + b[k]). FPU binary add on RHS."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out):
        Kt = a.shape[0] // TILE
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            out_blk.store(ttl.math.fill(out_blk, 0))
            for _ in range(Kt):
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                out_blk += a_blk + b_blk
                a_blk.pop()
                b_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(Kt):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[kt : kt + 1, 0:1], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


def _make_binary_mul_kernel():
    """out = sum over K of (a[k] * b[k]). FPU binary mul on RHS."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out):
        Kt = a.shape[0] // TILE
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            out_blk.store(ttl.math.fill(out_blk, 0))
            for _ in range(Kt):
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                out_blk += a_blk * b_blk
                a_blk.pop()
                b_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(Kt):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[kt : kt + 1, 0:1], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


def _make_unary_relu_kernel():
    """out = sum over K of relu(a[k]). SFPU unary on RHS."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, out):
        Kt = a.shape[0] // TILE
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            out_blk.store(ttl.math.fill(out_blk, 0))
            for _ in range(Kt):
                a_blk = a_dfb.wait()
                out_blk += ttl.math.relu(a_blk)
                a_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(Kt):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


def _make_binary_then_unary_kernel():
    """out = sum over K of relu(a[k] + b[k]). FPU binary -> SFPU unary fusion."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out):
        Kt = a.shape[0] // TILE
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            out_blk.store(ttl.math.fill(out_blk, 0))
            for _ in range(Kt):
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                out_blk += ttl.math.relu(a_blk + b_blk)
                a_blk.pop()
                b_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(Kt):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[kt : kt + 1, 0:1], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


def _make_fpu_chain_kernel():
    """out = sum over K of (a[k] * b[k] + c[k]). Two back-to-back FPU binaries."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, c, out):
        Kt = a.shape[0] // TILE
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            out_blk.store(ttl.math.fill(out_blk, 0))
            for _ in range(Kt):
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                c_blk = c_dfb.wait()
                out_blk += a_blk * b_blk + c_blk
                a_blk.pop()
                b_blk.pop()
                c_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(Kt):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[kt : kt + 1, 0:1], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[kt : kt + 1, 0:1], blk).wait()
                with c_dfb.reserve() as blk:
                    ttl.copy(c[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


def _make_sfpu_chain_kernel():
    """out = sum over K of exp(relu(a[k])). Two chained SFPU unaries."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, out):
        Kt = a.shape[0] // TILE
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            out_blk.store(ttl.math.fill(out_blk, 0))
            for _ in range(Kt):
                a_blk = a_dfb.wait()
                out_blk += ttl.math.exp(ttl.math.relu(a_blk))
                a_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(Kt):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


def _make_subblocked_add_kernel(block_m, block_n):
    """out[block_m, block_n] = sum over K of (a[k] + b[k]) for (block_m x block_n) blocks."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out):
        Kt = a.shape[0] // TILE // block_m
        a_dfb = ttl.make_dataflow_buffer_like(
            a, shape=(block_m, block_n), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b, shape=(block_m, block_n), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(block_m, block_n), block_count=2
        )

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            out_blk.store(ttl.math.fill(out_blk, 0))
            for _ in range(Kt):
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                out_blk += a_blk + b_blk
                a_blk.pop()
                b_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(Kt):
                m_off = kt * block_m
                with a_dfb.reserve() as blk:
                    ttl.copy(a[m_off : m_off + block_m, 0:block_n], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[m_off : m_off + block_m, 0:block_n], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:block_m, 0:block_n]).wait()

    return kernel


def _make_mixed_elem_matmul_kernel(K1, K2):
    """out = sum over K1 of (a[k]+b[k]) + sum over K2 of (c[k] @ d[k]).

    Loop 1: elementwise FPU-binary accumulation. Loop 2: matmul
    K-accumulation. Same output reserve shared across both loops.
    """

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, c, d, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
        d_dfb = ttl.make_dataflow_buffer_like(d, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            out_blk.store(ttl.math.fill(out_blk, 0))
            for _ in range(K1):
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                out_blk += a_blk + b_blk
                a_blk.pop()
                b_blk.pop()
            for _ in range(K2):
                c_blk = c_dfb.wait()
                d_blk = d_dfb.wait()
                out_blk += c_blk @ d_blk
                c_blk.pop()
                d_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(K1):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[kt : kt + 1, 0:1], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[kt : kt + 1, 0:1], blk).wait()
            for kt in range(K2):
                with c_dfb.reserve() as blk:
                    ttl.copy(c[0:1, kt : kt + 1], blk).wait()
                with d_dfb.reserve() as blk:
                    ttl.copy(d[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


def _make_fused_linear_relu_kernel(block_m, block_n, grid="auto"):
    """out = sum over K of relu(a[k] @ b[k] + c). Fused linear + bias + relu.

    Bias `c` is shared across K iterations (one read per output block);
    `a @ b + c` is evaluated per K step, relu applied, and accumulated
    via += into the output reserve's L1 slot.
    """

    @ttl.operation(grid=grid)
    def kernel(a, b, c, out):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE

        M_num = Mt // block_m
        N_num = Nt // block_n

        grid_n, grid_m = ttl.grid_size(dims=2)
        m_per = -(-M_num // grid_m)
        n_per = -(-N_num // grid_n)

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(block_m, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, block_n), block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(
            c, shape=(block_m, block_n), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(block_m, block_n), block_count=2
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for lm in range(m_per):
                mb = node_m * m_per + lm
                if mb < M_num:
                    for ln in range(n_per):
                        nb = node_n * n_per + ln
                        if nb < N_num:
                            out_blk = out_dfb.reserve()
                            out_blk.store(ttl.math.fill(out_blk, 0))
                            c_blk = c_dfb.wait()
                            for _ in range(Kt):
                                a_blk = a_dfb.wait()
                                b_blk = b_dfb.wait()
                                out_blk += ttl.math.relu(a_blk @ b_blk + c_blk)
                                a_blk.pop()
                                b_blk.pop()
                            c_blk.pop()
                            out_blk.push()

        @ttl.datamovement()
        def reader():
            node_n, node_m = ttl.node(dims=2)
            for lm in range(m_per):
                mb = node_m * m_per + lm
                if mb < M_num:
                    m_off = mb * block_m
                    for ln in range(n_per):
                        nb = node_n * n_per + ln
                        if nb < N_num:
                            n_off = nb * block_n
                            with c_dfb.reserve() as blk:
                                ttl.copy(
                                    c[
                                        m_off : m_off + block_m,
                                        n_off : n_off + block_n,
                                    ],
                                    blk,
                                ).wait()
                            for kt in range(Kt):
                                with a_dfb.reserve() as blk:
                                    ttl.copy(
                                        a[m_off : m_off + block_m, kt : kt + 1],
                                        blk,
                                    ).wait()

        @ttl.datamovement()
        def writer():
            node_n, node_m = ttl.node(dims=2)
            for lm in range(m_per):
                mb = node_m * m_per + lm
                if mb < M_num:
                    m_off = mb * block_m
                    for ln in range(n_per):
                        nb = node_n * n_per + ln
                        if nb < N_num:
                            n_off = nb * block_n
                            for kt in range(Kt):
                                with b_dfb.reserve() as blk:
                                    ttl.copy(
                                        b[kt : kt + 1, n_off : n_off + block_n],
                                        blk,
                                    ).wait()
                            with out_dfb.wait() as blk:
                                ttl.copy(
                                    blk,
                                    out[
                                        m_off : m_off + block_m,
                                        n_off : n_off + block_n,
                                    ],
                                ).wait()

    return kernel


def _make_store_then_acc_kernel(Kt):
    """.store(a[0]+b[0]), then += (a[k]+b[k]) for K-1 iterations."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            a_blk = a_dfb.wait()
            b_blk = b_dfb.wait()
            out_blk.store(a_blk + b_blk)
            a_blk.pop()
            b_blk.pop()
            for _ in range(Kt - 1):
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                out_blk += a_blk + b_blk
                a_blk.pop()
                b_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(Kt):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[kt : kt + 1, 0:1], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


# ---------------------------------------------------------------------------
# 1. Binary FPU add: out += a + b
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Kt", [2, 4, 8], ids=[f"K{k}" for k in [2, 4, 8]])
@pytest.mark.requires_device
def test_binary_add(Kt, device):
    """FPU binary add on RHS: out = sum over K of (a[k] + b[k])."""
    a = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    b = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    golden = (a.float() + b.float()).reshape(Kt, TILE, TILE).sum(dim=0)
    _run_acc_test(_make_binary_add_kernel(), [a, b], (TILE, TILE), golden, device)


# ---------------------------------------------------------------------------
# 2. Binary FPU mul: out += a * b
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Kt", [2, 4, 8], ids=[f"K{k}" for k in [2, 4, 8]])
@pytest.mark.requires_device
def test_binary_mul(Kt, device):
    """FPU binary mul on RHS: out = sum over K of (a[k] * b[k])."""
    a = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    b = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    golden = (a.float() * b.float()).reshape(Kt, TILE, TILE).sum(dim=0)
    _run_acc_test(_make_binary_mul_kernel(), [a, b], (TILE, TILE), golden, device)


# ---------------------------------------------------------------------------
# 3. Unary SFPU: out += relu(a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Kt", [2, 4, 8], ids=[f"K{k}" for k in [2, 4, 8]])
@pytest.mark.requires_device
def test_unary_relu(Kt, device):
    """SFPU unary on RHS: out = sum over K of relu(a[k])."""
    a = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    golden = torch.relu(a.float()).reshape(Kt, TILE, TILE).sum(dim=0)
    _run_acc_test(_make_unary_relu_kernel(), [a], (TILE, TILE), golden, device)


# ---------------------------------------------------------------------------
# 4. Binary -> unary fusion: out += relu(a + b)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Kt", [2, 4], ids=[f"K{k}" for k in [2, 4]])
@pytest.mark.requires_device
def test_binary_then_unary(Kt, device):
    """FPU binary followed by SFPU unary: out = sum over K of relu(a[k] + b[k])."""
    a = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    b = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    golden = torch.relu(a.float() + b.float()).reshape(Kt, TILE, TILE).sum(dim=0)
    _run_acc_test(
        _make_binary_then_unary_kernel(), [a, b], (TILE, TILE), golden, device
    )


# ---------------------------------------------------------------------------
# 5. FPU chain: out += a * b + c (two back-to-back binary ops).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Kt", [2, 4], ids=[f"K{k}" for k in [2, 4]])
@pytest.mark.requires_device
def test_fpu_chain_mul_add(Kt, device):
    """Two FPU binaries chained: out = sum over K of (a[k] * b[k] + c[k])."""
    a = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    b = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    c = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    golden = (a.float() * b.float() + c.float()).reshape(Kt, TILE, TILE).sum(dim=0)
    _run_acc_test(_make_fpu_chain_kernel(), [a, b, c], (TILE, TILE), golden, device)


# ---------------------------------------------------------------------------
# 6. SFPU chain: out += exp(relu(a))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Kt", [2, 4], ids=[f"K{k}" for k in [2, 4]])
@pytest.mark.requires_device
def test_sfpu_chain_exp_relu(Kt, device):
    """Two SFPU unaries chained: out = sum over K of exp(relu(a[k]))."""
    # Inputs scaled down: exp(relu(x)) grows fast; N(0,1) tails blow up bf16.
    a = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16) * 0.5
    golden = torch.exp(torch.relu(a.float())).reshape(Kt, TILE, TILE).sum(dim=0)
    _run_acc_test(_make_sfpu_chain_kernel(), [a], (TILE, TILE), golden, device)


# ---------------------------------------------------------------------------
# 7. Subblocked elementwise: block size exceeds DST capacity so the subblock
# pass splits the compute. Exercises L1-acc x subblocking interaction on a
# non-matmul RHS (matmul already covers this in test_matmul_l1_acc.py).
# ---------------------------------------------------------------------------

# block sizes: 2x2=4 fits in bf16 DST (8); 4x4=16 and 8x8=64 force subblocking.
SUBBLOCK_PARAMS = [
    # (block_m, block_n, Kt)
    (2, 2, 2),
    (4, 4, 2),
    (8, 8, 2),
]


@pytest.mark.parametrize(
    "block_m,block_n,Kt",
    SUBBLOCK_PARAMS,
    ids=[f"blk{m}x{n}_K{k}" for m, n, k in SUBBLOCK_PARAMS],
)
@pytest.mark.requires_device
def test_subblocked_binary_add(block_m, block_n, Kt, device):
    """Subblocked elementwise add + L1 acc."""
    M = block_m * TILE * Kt
    N = block_n * TILE
    a = torch.randn(M, N, dtype=torch.bfloat16)
    b = torch.randn(M, N, dtype=torch.bfloat16)
    golden = (
        (a.float() + b.float()).reshape(Kt, block_m * TILE, block_n * TILE).sum(dim=0)
    )
    _run_acc_test(
        _make_subblocked_add_kernel(block_m, block_n),
        [a, b],
        (block_m * TILE, block_n * TILE),
        golden,
        device,
    )


# ---------------------------------------------------------------------------
# 8. Mixed consecutive += loops on the same reserve: elementwise then matmul.
# Exercises init/kernel-config reconfig between FPU-binary pack-acc and
# matmul pack-acc while sharing the output reserve's L1 slot.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_mixed_elementwise_then_matmul(device):
    """Loop1 elementwise +=, loop2 matmul += on same reserve."""
    K1, K2 = 3, 4
    a = torch.randn(K1 * TILE, TILE, dtype=torch.bfloat16)
    b = torch.randn(K1 * TILE, TILE, dtype=torch.bfloat16)
    c = torch.randn(TILE, K2 * TILE, dtype=torch.bfloat16)
    d = torch.randn(K2 * TILE, TILE, dtype=torch.bfloat16)
    elem_sum = (a.float() + b.float()).reshape(K1, TILE, TILE).sum(dim=0)
    matmul_sum = c.float() @ d.float()
    golden = elem_sum + matmul_sum
    _run_acc_test(
        _make_mixed_elem_matmul_kernel(K1, K2),
        [a, b, c, d],
        (TILE, TILE),
        golden,
        device,
    )


# ---------------------------------------------------------------------------
# 9. Multinode fused long chain: out += relu(a @ b + c) across K.
# Combines matmul, FPU binary add, SFPU unary, multicore partitioning, and
# L1 accumulation in a single compute expression -- the deepest RHS chain
# for `+=` currently exercised end-to-end.
# ---------------------------------------------------------------------------


FUSED_LINEAR_PARAMS = [
    # (Mt, Kt, Nt, block_m, block_n)
    (16, 4, 16, 8, 8),  # 2x2 output blocks, K=4.
    (32, 4, 32, 8, 8),  # 4x4 output blocks, K=4.
]


@pytest.mark.parametrize(
    "Mt,Kt,Nt,block_m,block_n",
    FUSED_LINEAR_PARAMS,
    ids=[
        f"tiles{mt}x{kt}x{nt}_blk{bm}x{bn}"
        for mt, kt, nt, bm, bn in FUSED_LINEAR_PARAMS
    ],
)
@pytest.mark.requires_device
def test_fused_linear_relu_multinode(Mt, Kt, Nt, block_m, block_n, device):
    """Multinode out += relu(a @ b + c): matmul + add + relu + L1 acc."""
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    c = torch.randn(M, N, dtype=torch.bfloat16)
    # Per K-tile partial: relu(a[:, kt*TILE:(kt+1)*TILE] @ b[...] + c); sum over tiles.
    partials = [
        torch.relu(
            a.float()[:, kt * TILE : (kt + 1) * TILE]
            @ b.float()[kt * TILE : (kt + 1) * TILE, :]
            + c.float()
        )
        for kt in range(Kt)
    ]
    golden = torch.stack(partials).sum(dim=0)
    _run_acc_test(
        _make_fused_linear_relu_kernel(block_m, block_n),
        [a, b, c],
        (M, N),
        golden,
        device,
    )


# ---------------------------------------------------------------------------
# 10. `.store()` then elementwise += accumulate. First iteration writes
# a[0]+b[0] with .store(); remaining K-1 iterations accumulate a[k]+b[k]
# with +=. Total is sum over K of (a[k]+b[k]).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Kt", [2, 4], ids=[f"K{k}" for k in [2, 4]])
@pytest.mark.requires_device
def test_store_then_elementwise_acc(Kt, device):
    """.store(a+b) then += (a+b) loop. Result = sum over K of (a+b)."""
    a = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    b = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    golden = (a.float() + b.float()).reshape(Kt, TILE, TILE).sum(dim=0)
    _run_acc_test(_make_store_then_acc_kernel(Kt), [a, b], (TILE, TILE), golden, device)


# ---------------------------------------------------------------------------
# 11. `+=` loop with no prior pack before it.
# ---------------------------------------------------------------------------


def _make_no_prior_value_kernel():
    """sum over K of a[k] with no prior pack before the loop."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, out):
        Kt = a.shape[0] // TILE
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            out_blk = out_dfb.reserve()
            for _ in range(Kt):
                a_blk = a_dfb.wait()
                out_blk += a_blk
                a_blk.pop()
            out_blk.push()

        @ttl.datamovement()
        def reader():
            for kt in range(Kt):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[kt : kt + 1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.parametrize("Kt", [2, 4, 8], ids=[f"K{k}" for k in [2, 4, 8]])
@pytest.mark.requires_device
def test_no_prior_value_iter0_overwrite(Kt, device):
    """`+=` loop with no prior pack before it."""
    a = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
    golden = a.float().reshape(Kt, TILE, TILE).sum(dim=0)
    _run_acc_test(_make_no_prior_value_kernel(), [a], (TILE, TILE), golden, device)
