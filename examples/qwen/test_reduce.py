# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validate reduce_sum and reduce_max on Blackhole hardware."""

import torch
import ttl
import ttnn

TILE = 32


def to_dev(t, dev):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# Test 1: reduce_sum dims=[0] (column-wise: sum across rows)
@ttl.kernel(grid=(1, 1))
def reduce_sum_cols_kernel(X, scaler, Y):
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(X[0, 0], blk)
            tx.wait()
        with s_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.compute()
    def compute():
        with x_dfb.wait() as x_blk, s_dfb.wait() as s_blk, y_dfb.reserve() as y_blk:
            y_blk.store(ttl.math.reduce_sum(x_blk, s_blk, y_blk, dims=[0]))

    @ttl.datamovement()
    def write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, Y[0, 0])
            tx.wait()


# Test 2: reduce_sum dims=[1] (row-wise: sum across columns)
@ttl.kernel(grid=(1, 1))
def reduce_sum_rows_kernel(X, scaler, Y):
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(X[0, 0], blk)
            tx.wait()
        with s_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.compute()
    def compute():
        with x_dfb.wait() as x_blk, s_dfb.wait() as s_blk, y_dfb.reserve() as y_blk:
            y_blk.store(ttl.math.reduce_sum(x_blk, s_blk, y_blk, dims=[1]))

    @ttl.datamovement()
    def write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, Y[0, 0])
            tx.wait()


# Test 3: reduce_max dims=[0] (column-wise max)
@ttl.kernel(grid=(1, 1))
def reduce_max_cols_kernel(X, scaler, Y):
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(X[0, 0], blk)
            tx.wait()
        with s_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.compute()
    def compute():
        with x_dfb.wait() as x_blk, s_dfb.wait() as s_blk, y_dfb.reserve() as y_blk:
            y_blk.store(ttl.math.reduce_max(x_blk, s_blk, y_blk, dims=[0]))

    @ttl.datamovement()
    def write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, Y[0, 0])
            tx.wait()


# Test 4: reduce_max dims=[1] (row-wise max)
@ttl.kernel(grid=(1, 1))
def reduce_max_rows_kernel(X, scaler, Y):
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(X[0, 0], blk)
            tx.wait()
        with s_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.compute()
    def compute():
        with x_dfb.wait() as x_blk, s_dfb.wait() as s_blk, y_dfb.reserve() as y_blk:
            y_blk.store(ttl.math.reduce_max(x_blk, s_blk, y_blk, dims=[1]))

    @ttl.datamovement()
    def write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, Y[0, 0])
            tx.wait()


def main():
    device = ttnn.open_device(device_id=0)
    try:
        X_t = torch.randn(TILE, TILE, dtype=torch.bfloat16)
        scaler_ones = torch.ones(TILE, TILE, dtype=torch.bfloat16)

        X = to_dev(X_t, device)
        sc = to_dev(scaler_ones, device)

        # Test 1: reduce_sum dims=[0]
        print("Test 1: reduce_sum dims=[0] (sum across rows)...", end="", flush=True)
        Y = to_dev(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        reduce_sum_cols_kernel(X, sc, Y)
        r = ttnn.to_torch(Y)
        expected = X_t.float().sum(dim=0)  # [32]
        pcc = torch.corrcoef(torch.stack([r[0].float(), expected]))[0, 1].item()
        print(f" PCC={pcc:.6f} {'PASS' if pcc > 0.98 else 'FAIL'}")

        # Test 2: reduce_sum dims=[1]
        print("Test 2: reduce_sum dims=[1] (sum across cols)...", end="", flush=True)
        Y = to_dev(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        reduce_sum_rows_kernel(X, sc, Y)
        r = ttnn.to_torch(Y)
        expected = X_t.float().sum(dim=1)  # [32]
        # Only first 16 valid per docs
        pcc = torch.corrcoef(torch.stack([r[0, :16].float(), expected[:16]]))[0, 1].item()
        print(f" PCC={pcc:.6f} (first 16) {'PASS' if pcc > 0.98 else 'FAIL'}")

        # Test 3: reduce_max dims=[0]
        print("Test 3: reduce_max dims=[0] (max across rows)...", end="", flush=True)
        Y = to_dev(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        reduce_max_cols_kernel(X, sc, Y)
        r = ttnn.to_torch(Y)
        expected = X_t.float().max(dim=0).values
        pcc = torch.corrcoef(torch.stack([r[0].float(), expected]))[0, 1].item()
        print(f" PCC={pcc:.6f} {'PASS' if pcc > 0.98 else 'FAIL'}")

        # Test 4: reduce_max dims=[1]
        print("Test 4: reduce_max dims=[1] (max across cols)...", end="", flush=True)
        Y = to_dev(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        reduce_max_rows_kernel(X, sc, Y)
        r = ttnn.to_torch(Y)
        expected = X_t.float().max(dim=1).values
        pcc = torch.corrcoef(torch.stack([r[0, :16].float(), expected[:16]]))[0, 1].item()
        print(f" PCC={pcc:.6f} (first 16) {'PASS' if pcc > 0.98 else 'FAIL'}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
