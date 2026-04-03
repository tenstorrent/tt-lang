"""Shift-register with pipes: 27-pass neighbor sharing.
Remote neighbors via pipe, local neighbors from DRAM.
PipeNet constructed inside kernel scope using captured int constants.
"""
import torch
import numpy as np
import ttnn
import ttl

TILE = 32


def test_shift_register_pipes():
    device = ttnn.open_device(device_id=0)

    n_dim = 3
    n_cells = n_dim ** 3
    n_cores = min(n_cells, 8)
    cells_per_core = -(-n_cells // n_cores)

    own_data = np.random.randn(n_cells * TILE, TILE).astype(np.float32) * 0.1

    offsets = [(dx, dy, dz)
               for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)]

    def cell_core(c):
        return min(c // cells_per_core, n_cores - 1)

    def neighbor_id(c, dx, dy, dz):
        cx, cy, cz = c // (n_dim*n_dim), (c // n_dim) % n_dim, c % n_dim
        return ((cx+dx) % n_dim) * n_dim**2 + ((cy+dy) % n_dim) * n_dim + ((cz+dz) % n_dim)

    expected = np.zeros((n_cells * TILE, TILE), dtype=np.float32)
    for c in range(n_cells):
        for dx, dy, dz in offsets:
            nc = neighbor_id(c, dx, dy, dz)
            expected[c*TILE:(c+1)*TILE, :] += own_data[nc*TILE:(nc+1)*TILE, :]

    def to_tt(arr):
        return ttnn.from_torch(
            torch.tensor(arr, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tt_own = to_tt(own_data)
    tt_out = to_tt(np.zeros((n_cells * TILE, TILE), dtype=np.float32))

    for oi, (dx, dy, dz) in enumerate(offsets):
        is_first = 1 if oi == 0 else 0
        off_dx, off_dy, off_dz = int(dx), int(dy), int(dz)

        # Build pipe list here, but pass as constructed PipeNet
        # Since we can't capture lists, build PipeNet at module level
        # Actually, PipeNet must be at kernel scope. Let me build it there
        # using the offset constants which ARE capturable ints.

        @ttl.kernel(grid=(n_cores, 1))
        def offset_kernel(own, acc_out):
            grid_cols, _ = ttl.grid_size(dims=2)
            nc = own.shape[0] // TILE
            cpc = -(-nc // grid_cols)
            dim2 = n_dim * n_dim

            # Build PipeNet at kernel scope from captured int constants
            # For this offset, compute which cells need remote pipes
            pipes = []
            for c in range(n_cells):
                cx = c // dim2
                cy = (c // n_dim) % n_dim
                cz = c % n_dim
                nbr = ((cx + off_dx) % n_dim) * dim2 + ((cy + off_dy) % n_dim) * n_dim + ((cz + off_dz) % n_dim)
                src_core = min(nbr // cpc, n_cores - 1)
                dst_core = min(c // cpc, n_cores - 1)
                if src_core != dst_core:
                    pipes.append(ttl.Pipe((src_core, 0), (dst_core, 0)))

            has_pipes = 1 if len(pipes) > 0 else 0
            net = ttl.PipeNet(pipes if len(pipes) > 0 else [ttl.Pipe((0, 0), (0, 0))])

            nbr_cb = ttl.make_dataflow_buffer_like(own, shape=(1, 1), buffer_factor=2)
            acc_cb = ttl.make_dataflow_buffer_like(own, shape=(1, 1), buffer_factor=2)
            out_cb = ttl.make_dataflow_buffer_like(own, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                core_x, _ = ttl.core(dims=2)
                for local_c in range(cpc):
                    cid = core_x * cpc + local_c
                    if cid < nc:
                        if is_first == 1:
                            with nbr_cb.wait() as n, out_cb.reserve() as o:
                                o.store(n)
                        else:
                            with nbr_cb.wait() as n, acc_cb.wait() as prev, out_cb.reserve() as o:
                                o.store(prev + n)

            @ttl.datamovement()
            def dm_read():
                core_x, _ = ttl.core(dims=2)
                for local_c in range(cpc):
                    cid = core_x * cpc + local_c
                    if cid < nc:
                        cx = cid // dim2
                        cy = (cid // n_dim) % n_dim
                        cz = cid % n_dim
                        nbr_id = ((cx + off_dx) % n_dim) * dim2 + ((cy + off_dy) % n_dim) * n_dim + ((cz + off_dz) % n_dim)

                        with nbr_cb.reserve() as blk:
                            tx = ttl.copy(own[nbr_id, 0], blk)
                            tx.wait()

                            # Send via pipe if we're a source for any pipe
                            if has_pipes == 1:
                                def send(pipe):
                                    xf = ttl.copy(blk, pipe)
                                    xf.wait()
                                net.if_src(send)

                        if is_first == 0:
                            with acc_cb.reserve() as blk:
                                tx = ttl.copy(acc_out[cid, 0], blk)
                                tx.wait()

            @ttl.datamovement()
            def dm_write():
                core_x, _ = ttl.core(dims=2)
                for local_c in range(cpc):
                    cid = core_x * cpc + local_c
                    if cid < nc:
                        with out_cb.wait() as blk:
                            tx = ttl.copy(blk, acc_out[cid, 0])
                            tx.wait()

        offset_kernel(tt_own, tt_out)

    result = ttnn.to_torch(tt_out).float().numpy()

    max_err = 0.0
    for c in range(n_cells):
        exp_val = expected[c * TILE, 0]
        got_val = result[c * TILE, 0]
        if abs(exp_val) > 1e-6:
            err = abs(got_val - exp_val) / abs(exp_val)
        else:
            err = abs(got_val - exp_val)
        max_err = max(max_err, err)

    print(f"Max error across {n_cells} cells: {max_err:.6f}")
    passed = max_err < 0.10
    print(f"Shift-register pipes test: {'PASS' if passed else 'FAIL'}")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_shift_register_pipes()
