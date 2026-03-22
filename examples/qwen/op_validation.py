# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1: Hardware op validation for Qwen 2.5 0.5B.

Tests critical operations on real Blackhole hardware to determine
which decompositions are needed for the model implementation.

Usage:
    source build/env/activate
    python examples/qwen/op_validation.py
"""

import sys
import torch
import traceback

import ttnn
import ttl

TILE = 32


def to_device(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def pcc(result, expected):
    r = result.float().flatten()
    e = expected.float().flatten()
    return torch.corrcoef(torch.stack([r, e]))[0, 1].item()


# ==========================================================================
# Test 1: Matmul with K-accumulation (prev + a @ b pattern)
# Exact pattern from matmul_acc.py - the confirmed working approach
# ==========================================================================
@ttl.kernel(grid=(1, 1))
def matmul_k_acc_kernel(A, B, C, Y):
    """Y = A @ B + C (matmul with bias, K-accumulation)"""
    Mt = A.shape[0] // TILE
    Kt = A.shape[1] // TILE
    Nt = B.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(C, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for m in range(Mt):
            for n in range(Nt):
                with c_dfb.reserve() as c_blk:
                    tx = ttl.copy(C[m, n], c_blk)
                    tx.wait()
                for k in range(Kt):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        tx_a = ttl.copy(A[m, k], a_blk)
                        tx_a.wait()
                        tx_b = ttl.copy(B[k, n], b_blk)
                        tx_b.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            for _ in range(Nt):
                # Pre-load bias into accumulator
                with c_dfb.wait() as c_blk:
                    with acc_dfb.reserve() as acc:
                        acc.store(c_blk)

                # K-loop: prev + a @ b (fused matmul accumulation)
                for _ in range(Kt):
                    with (
                        a_dfb.wait() as a_blk,
                        b_dfb.wait() as b_blk,
                        acc_dfb.wait() as prev,
                    ):
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + a_blk @ b_blk)

                # Write result
                with acc_dfb.wait() as acc_blk:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(acc_blk)

    @ttl.datamovement()
    def write():
        for m in range(Mt):
            for n in range(Nt):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, Y[m, n])
                    tx.wait()


def test_matmul_k_acc(device):
    M, K, N = 128, 96, 64  # 4x3x2 tiles
    A_t = torch.randn(M, K, dtype=torch.bfloat16)
    B_t = torch.randn(K, N, dtype=torch.bfloat16)
    C_t = torch.randn(M, N, dtype=torch.bfloat16)  # bias

    A = to_device(A_t, device)
    B = to_device(B_t, device)
    C = to_device(C_t, device)
    Y = to_device(torch.zeros(M, N, dtype=torch.bfloat16), device)

    matmul_k_acc_kernel(A, B, C, Y)

    result = ttnn.to_torch(Y)
    expected = (A_t.float() @ B_t.float() + C_t.float()).bfloat16()
    score = pcc(result, expected)
    print(f"  Test 1 - Matmul K-acc + bias [{M}x{K}]@[{K}x{N}]+bias: PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL (PCC={score:.6f})"
    print(" PASS")
    return True


# ==========================================================================
# Test 2: SiLU decomposition: x * sigmoid(x)
# ==========================================================================
@ttl.kernel(grid=(1, 1))
def silu_kernel(X, Y):
    Mt = X.shape[0] // TILE
    Nt = X.shape[1] // TILE
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for m in range(Mt):
            for n in range(Nt):
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(X[m, n], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            for _ in range(Nt):
                with x_dfb.wait() as x_blk, y_dfb.reserve() as y_blk:
                    y_blk.store(x_blk * ttl.math.sigmoid(x_blk))

    @ttl.datamovement()
    def write():
        for m in range(Mt):
            for n in range(Nt):
                with y_dfb.wait() as blk:
                    tx = ttl.copy(blk, Y[m, n])
                    tx.wait()


def test_silu(device):
    M, N = 64, 64  # 2x2 tiles
    X_t = torch.randn(M, N, dtype=torch.bfloat16)
    X = to_device(X_t, device)
    Y = to_device(torch.zeros(M, N, dtype=torch.bfloat16), device)

    silu_kernel(X, Y)

    result = ttnn.to_torch(Y)
    expected = torch.nn.functional.silu(X_t.float()).bfloat16()
    score = pcc(result, expected)
    print(f"  Test 2 - SiLU decomposition [x*sigmoid(x)]: PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")
    return True


# ==========================================================================
# Test 3: store(a @ b, acc=True) - N/S per spec but used in examples
# Pattern from single_node_matmul.py
# ==========================================================================
@ttl.kernel(grid=(1, 1))
def matmul_acc_true_kernel(A, B, Y):
    Mt = A.shape[0] // TILE
    Kt = A.shape[1] // TILE
    Nt = B.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for m in range(Mt):
            for n in range(Nt):
                for k in range(Kt):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        tx_a = ttl.copy(A[m, k], a_blk)
                        tx_a.wait()
                        tx_b = ttl.copy(B[k, n], b_blk)
                        tx_b.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            for _ in range(Nt):
                with y_dfb.reserve() as y_blk:
                    for _ in range(Kt):
                        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                            y_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def write():
        for m in range(Mt):
            for n in range(Nt):
                with y_dfb.wait() as blk:
                    tx = ttl.copy(blk, Y[m, n])
                    tx.wait()


def test_matmul_acc_true(device):
    M, K, N = 64, 96, 64
    A_t = torch.randn(M, K, dtype=torch.bfloat16)
    B_t = torch.randn(K, N, dtype=torch.bfloat16)

    A = to_device(A_t, device)
    B = to_device(B_t, device)
    Y = to_device(torch.zeros(M, N, dtype=torch.bfloat16), device)

    matmul_acc_true_kernel(A, B, Y)

    result = ttnn.to_torch(Y)
    expected = (A_t.float() @ B_t.float()).bfloat16()
    score = pcc(result, expected)
    print(f"  Test 3 - store(a@b, acc=True): PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")
    return True


# ==========================================================================
# Test 4: ttl.math.reduce_sum - N/S per spec but used in examples
# ==========================================================================
@ttl.kernel(grid=(1, 1))
def reduce_sum_kernel(X, scaler, Y):
    Nt = X.shape[1] // TILE
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, Nt), buffer_factor=1)
    s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(X[0:1, 0:Nt], blk)
            tx.wait()
        with s_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.compute()
    def compute():
        with x_dfb.wait() as x_blk, s_dfb.wait() as s_blk:
            with y_dfb.reserve() as y_blk:
                y_blk.store(ttl.math.reduce_sum(x_blk, s_blk, y_blk, dims=[0]))

    @ttl.datamovement()
    def write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, Y[0, 0])
            tx.wait()


def test_reduce_sum(device):
    N = 128  # 4 tiles
    X_t = torch.randn(TILE, N, dtype=torch.bfloat16)
    scaler_t = torch.ones(TILE, TILE, dtype=torch.bfloat16)

    X = to_device(X_t, device)
    scaler = to_device(scaler_t, device)
    Y = to_device(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    reduce_sum_kernel(X, scaler, Y)

    result = ttnn.to_torch(Y)
    # reduce_sum with dims=[0] reduces along the column dimension
    expected = X_t.float().sum(dim=1, keepdim=True).expand(-1, TILE).bfloat16()
    score = pcc(result, expected)
    print(f"  Test 4 - ttl.math.reduce_sum: PCC={score:.6f}", end="")
    assert score > 0.98, f" FAIL"
    print(" PASS")
    return True


# ==========================================================================
# Test 5: Broadcast (tile-level row broadcast, matching simple_bcast.py)
# dims=[0] copies row 0 of the tile to all 32 rows within the tile
# ==========================================================================
@ttl.kernel(grid=(1, 1))
def broadcast_add_kernel(A, B, C, Y):
    """Y = A * B + bcast_row(C). C has values only in row 0."""
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(C, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(A[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(B[0, 0], blk)
            tx.wait()
        with c_dfb.reserve() as blk:
            tx = ttl.copy(C[0, 0], blk)
            tx.wait()

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            c_dfb.wait() as c_blk,
            y_dfb.reserve() as y_blk,
        ):
            c_bcast = ttl.math.broadcast(c_blk, y_blk, dims=[0])
            y_blk.store(a_blk * b_blk + c_bcast)

    @ttl.datamovement()
    def write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, Y[0, 0])
            tx.wait()


def test_broadcast_add(device):
    A_t = torch.full((TILE, TILE), 2.0, dtype=torch.bfloat16)
    B_t = torch.full((TILE, TILE), 3.0, dtype=torch.bfloat16)
    C_t = torch.zeros((TILE, TILE), dtype=torch.bfloat16)
    C_t[0, :] = 1.0  # row 0 = 1.0, broadcast to all rows

    A = to_device(A_t, device)
    B = to_device(B_t, device)
    C = to_device(C_t, device)
    Y = to_device(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    broadcast_add_kernel(A, B, C, Y)

    result = ttnn.to_torch(Y)
    expected = torch.full((TILE, TILE), 7.0, dtype=torch.bfloat16)  # 2*3 + 1 = 7
    match = torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)
    print(f"  Test 5 - Broadcast row (a*b + bcast(c)): val={result[0,0].item():.1f} expect=7.0", end="")
    assert match, f" FAIL (got {result[0,0].item()})"
    print(" PASS")
    return True


# ==========================================================================
# Test 6: Sequential accumulation (sum 4 tiles via running total)
# This is the practical fallback for reduce_sum: accumulate tiles one by one
# ==========================================================================
@ttl.kernel(grid=(1, 1))
def seq_sum_kernel(X, Y):
    """Sum 4 tiles sequentially: acc = 0, acc += tile[i] for each tile."""
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for n in range(4):
            with x_dfb.reserve() as blk:
                tx = ttl.copy(X[0, n], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        # First tile: store directly as initial accumulator value
        with x_dfb.wait() as t0:
            with acc_dfb.reserve() as acc:
                acc.store(t0)

        # Remaining tiles: acc = prev + next
        for _ in range(3):
            with x_dfb.wait() as t_next, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + t_next)

        # Write final result
        with acc_dfb.wait() as final:
            with y_dfb.reserve() as y_blk:
                y_blk.store(final)

    @ttl.datamovement()
    def write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, Y[0, 0])
            tx.wait()


def test_seq_sum(device):
    N = 128  # 4 tiles
    X_t = torch.randn(TILE, N, dtype=torch.bfloat16)

    X = to_device(X_t, device)
    Y = to_device(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    seq_sum_kernel(X, Y)

    result = ttnn.to_torch(Y)
    expected = (X_t[:, :32].float() + X_t[:, 32:64].float()
                + X_t[:, 64:96].float() + X_t[:, 96:128].float()).bfloat16()
    score = pcc(result, expected)
    print(f"  Test 6 - Sequential sum (4 tiles): PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")
    return True


# ==========================================================================
# Main
# ==========================================================================
def main():
    device = ttnn.open_device(device_id=0)
    results = {}

    tests = [
        ("matmul_k_acc", test_matmul_k_acc),
        ("silu_decomp", test_silu),
        ("matmul_acc_true", test_matmul_acc_true),
        ("reduce_sum", test_reduce_sum),
        ("broadcast_add", test_broadcast_add),
        ("seq_sum", test_seq_sum),
    ]

    try:
        print("=" * 60)
        print("Phase 1: Hardware Op Validation")
        print("=" * 60)

        for name, test_fn in tests:
            try:
                test_fn(device)
                results[name] = "PASS"
            except Exception as e:
                results[name] = f"FAIL: {e}"
                traceback.print_exc()
                print()

        print("\n" + "=" * 60)
        print("Summary:")
        print("=" * 60)
        for name, status in results.items():
            print(f"  {name}: {status}")

        print("\n" + "=" * 60)
        print("Decision Points:")
        print("=" * 60)
        if results.get("matmul_acc_true", "").startswith("PASS"):
            print("  -> store(a@b, acc=True) WORKS: use for K-accumulation (simpler)")
        else:
            print("  -> store(a@b, acc=True) FAILS: use prev+a@b pattern")

        if results.get("reduce_sum", "").startswith("PASS"):
            print("  -> reduce_sum WORKS: use directly for RMSNorm/softmax")
        elif results.get("seq_sum", "").startswith("PASS"):
            print("  -> reduce_sum FAILS but seq_sum WORKS: use sequential accumulation")
        else:
            print("  -> Both reductions FAIL: use host-side reduction")

        passed = sum(1 for v in results.values() if v == "PASS")
        print(f"\n{passed}/{len(results)} tests passed")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
