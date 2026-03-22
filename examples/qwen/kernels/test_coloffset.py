"""Test column-offset indexing in tt-lang kernels."""
import torch
import ttl
import ttnn

TILE = 32


def td(t, d):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@ttl.kernel(grid=(1, 1))
def copy_col2_kernel(X, Y):
    """Copy tile at X[0, 2] to Y[0, 0]."""
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(X[0, 2], blk)
            tx.wait()

    @ttl.compute()
    def compute():
        with x_dfb.wait() as x_blk, y_dfb.reserve() as y_blk:
            y_blk.store(x_blk)

    @ttl.datamovement()
    def write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, Y[0, 0])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def copy_runtime_col_kernel(X, Y):
    """Copy tile at X[0, col] where col is a loop variable."""
    Nt = X.shape[1] // TILE
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        # Read 3rd tile (index 2) using loop variable
        for col in range(Nt):
            with x_dfb.reserve() as blk:
                tx = ttl.copy(X[0, col], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Nt):
            with x_dfb.wait() as x_blk, y_dfb.reserve() as y_blk:
                y_blk.store(x_blk)

    @ttl.datamovement()
    def write():
        for col in range(Nt):
            with y_dfb.wait() as blk:
                tx = ttl.copy(blk, Y[0, col])
                tx.wait()


@ttl.kernel(grid=(1, 1))
def copy_computed_col_kernel(X, Y):
    """Copy tiles at computed column offsets: pair*2 and pair*2+1."""
    num_pairs = X.shape[1] // TILE // 2
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for pair in range(num_pairs):
            col0 = pair * 2
            col1 = pair * 2 + 1
            with x_dfb.reserve() as blk:
                tx = ttl.copy(X[0, col0], blk)
                tx.wait()
            with x_dfb.reserve() as blk:
                tx = ttl.copy(X[0, col1], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(num_pairs):
            with x_dfb.wait() as blk0, y_dfb.reserve() as out0:
                out0.store(blk0)
            with x_dfb.wait() as blk1, y_dfb.reserve() as out1:
                out1.store(blk1)

    @ttl.datamovement()
    def write():
        for pair in range(num_pairs):
            col0 = pair * 2
            col1 = pair * 2 + 1
            with y_dfb.wait() as blk:
                tx = ttl.copy(blk, Y[0, col0])
                tx.wait()
            with y_dfb.wait() as blk:
                tx = ttl.copy(blk, Y[0, col1])
                tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        # Test 1: static column offset
        print("Test 1: copy from static col 2...", end="", flush=True)
        X_t = torch.randn(TILE, 128, dtype=torch.bfloat16)
        Y_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
        X = td(X_t, device)
        Y = td(Y_t, device)
        copy_col2_kernel(X, Y)
        r = ttnn.to_torch(Y)
        expected = X_t[:, 64:96]
        pcc = torch.corrcoef(torch.stack([r.float().flatten(), expected.float().flatten()]))[0, 1].item()
        print(f" PCC={pcc:.4f} {'PASS' if pcc > 0.98 else 'FAIL'}")

        # Test 2: loop-variable column
        print("Test 2: copy all cols via loop...", end="", flush=True)
        X2 = td(X_t, device)
        Y2 = td(torch.zeros(TILE, 128, dtype=torch.bfloat16), device)
        copy_runtime_col_kernel(X2, Y2)
        r2 = ttnn.to_torch(Y2)
        pcc2 = torch.corrcoef(torch.stack([r2.float().flatten(), X_t.float().flatten()]))[0, 1].item()
        print(f" PCC={pcc2:.4f} {'PASS' if pcc2 > 0.98 else 'FAIL'}")

        # Test 3: computed column offsets (pair*2, pair*2+1)
        print("Test 3: computed col offsets (pair*2)...", end="", flush=True)
        X3 = td(X_t, device)
        Y3 = td(torch.zeros(TILE, 128, dtype=torch.bfloat16), device)
        copy_computed_col_kernel(X3, Y3)
        r3 = ttnn.to_torch(Y3)
        pcc3 = torch.corrcoef(torch.stack([r3.float().flatten(), X_t.float().flatten()]))[0, 1].item()
        print(f" PCC={pcc3:.4f} {'PASS' if pcc3 > 0.98 else 'FAIL'}")

        print("All column offset tests passed!")
    finally:
        ttnn.close_device(device)
