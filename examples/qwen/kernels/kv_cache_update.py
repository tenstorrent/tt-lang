# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fully on-device KV cache update kernel. Zero host transfers.

Updates both K^T [64, 512] and V [512, 64] caches for 2 KV heads in one call.

V update (row insertion at pos): broadcast(new_vals, dims=[0]) + row_mask
K^T update (column insertion at pos): transpose → broadcast(dims=[1]) + col_mask

Factory function creates 16 variants (one per tile_slot = pos // 32).
"""

import torch
import ttl
import ttnn

TILE = 32


def _make_kv_cache_update_kernel(tile_slot):
    """Create a KV cache update kernel for a specific tile_slot.

    tile_slot = pos // 32 determines which tiles to modify:
      - V [512, 64]: row tile_slot (tiles V[tile_slot, 0] and V[tile_slot, 1])
      - K^T [64, 512]: column tile_slot (tiles K^T[0, tile_slot] and K^T[1, tile_slot])
    """

    @ttl.kernel(grid=(1, 1))
    def kv_cache_update(
        k_rot,         # [TILE, 128] — new K for 2 KV heads (row 0 has values)
        v_out,         # [TILE, 128] — new V for 2 KV heads (row 0 has values)
        kt_cache_0,    # [64, 512]  — K^T cache head 0 (read-modify-write)
        kt_cache_1,    # [64, 512]  — K^T cache head 1 (read-modify-write)
        v_cache_0,     # [512, 64]  — V cache head 0 (read-modify-write)
        v_cache_1,     # [512, 64]  — V cache head 1 (read-modify-write)
        row_mask,      # [TILE, TILE] — 1.0 in row (pos%32), 0.0 elsewhere
        inv_row_mask,  # [TILE, TILE] — 1.0 - row_mask
        col_mask,      # [TILE, TILE] — 1.0 in col (pos%32), 0.0 elsewhere
        inv_col_mask,  # [TILE, TILE] — 1.0 - col_mask
    ):
        Nt = 2  # head_dim / TILE = 64 / 32

        # Reader → Compute DFBs
        new_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
        old_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
        rm_dfb = ttl.make_dataflow_buffer_like(row_mask, shape=(1, 1), buffer_factor=2)
        irm_dfb = ttl.make_dataflow_buffer_like(inv_row_mask, shape=(1, 1), buffer_factor=2)

        # Compute-local DFBs
        trans_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
        masked_new_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
        zeroed_old_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)

        # Compute → Writer DFB
        result_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)

        @ttl.datamovement()
        def read():
            # === K^T column insertion (4 tiles: 2 heads × 2 dim-rows) ===
            # K^T head 0: tiles K^T[0, tile_slot] and K^T[1, tile_slot]
            for dim_row in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(k_rot[0, dim_row], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(kt_cache_0[dim_row, tile_slot], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(col_mask[0, 0], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(inv_col_mask[0, 0], blk)
                    tx.wait()

            # K^T head 1
            for dim_row in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(k_rot[0, Nt + dim_row], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(kt_cache_1[dim_row, tile_slot], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(col_mask[0, 0], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(inv_col_mask[0, 0], blk)
                    tx.wait()

            # === V row insertion (4 tiles: 2 heads × 2 dim-cols) ===
            # V head 0: tiles V[tile_slot, 0] and V[tile_slot, 1]
            for dim_col in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(v_out[0, dim_col], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(v_cache_0[tile_slot, dim_col], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(row_mask[0, 0], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(inv_row_mask[0, 0], blk)
                    tx.wait()

            # V head 1
            for dim_col in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(v_out[0, Nt + dim_col], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(v_cache_1[tile_slot, dim_col], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(row_mask[0, 0], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(inv_row_mask[0, 0], blk)
                    tx.wait()

        @ttl.compute()
        def compute():
            # === K^T column insertion (4 tiles) ===
            # New K values in row 0 → transpose → column 0 → broadcast cols → mask
            for _ in range(4):
                with new_dfb.wait() as new_val, old_dfb.wait() as old_val:
                    with rm_dfb.wait() as cm, irm_dfb.wait() as icm:
                        # transpose: row 0 values → column 0
                        with trans_dfb.reserve() as tr:
                            tr.store(ttl.math.transpose(new_val, tr))
                        # broadcast column 0 → all columns
                        with trans_dfb.wait() as tr_val:
                            with bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(tr_val, bc, dims=[1]))
                        # masked_new = broadcast * col_mask
                        with bcast_dfb.wait() as bc_val:
                            with masked_new_dfb.reserve() as mn:
                                mn.store(bc_val * cm)
                        # zeroed_old = old * inv_col_mask
                        with zeroed_old_dfb.reserve() as zo:
                            zo.store(old_val * icm)
                        # result = zeroed_old + masked_new
                        with zeroed_old_dfb.wait() as zo, masked_new_dfb.wait() as mn:
                            with result_dfb.reserve() as res:
                                res.store(zo + mn)

            # === V row insertion (4 tiles) ===
            # New V values in row 0 → broadcast rows → mask
            for _ in range(4):
                with new_dfb.wait() as new_val, old_dfb.wait() as old_val:
                    with rm_dfb.wait() as rm, irm_dfb.wait() as irm:
                        # broadcast row 0 → all rows
                        with bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(new_val, bc, dims=[0]))
                        # masked_new = broadcast * row_mask
                        with bcast_dfb.wait() as bc_val:
                            with masked_new_dfb.reserve() as mn:
                                mn.store(bc_val * rm)
                        # zeroed_old = old * inv_row_mask
                        with zeroed_old_dfb.reserve() as zo:
                            zo.store(old_val * irm)
                        # result = zeroed_old + masked_new
                        with zeroed_old_dfb.wait() as zo, masked_new_dfb.wait() as mn:
                            with result_dfb.reserve() as res:
                                res.store(zo + mn)

        @ttl.datamovement()
        def write():
            # K^T head 0
            for dim_row in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, kt_cache_0[dim_row, tile_slot])
                    tx.wait()
            # K^T head 1
            for dim_row in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, kt_cache_1[dim_row, tile_slot])
                    tx.wait()
            # V head 0
            for dim_col in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, v_cache_0[tile_slot, dim_col])
                    tx.wait()
            # V head 1
            for dim_col in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, v_cache_1[tile_slot, dim_col])
                    tx.wait()

    return kv_cache_update


# Lazy kernel cache
_kv_cache_update_cache = {}


def get_kv_cache_update_kernel(tile_slot):
    """Get (or compile) the KV cache update kernel for a tile_slot."""
    if tile_slot not in _kv_cache_update_cache:
        _kv_cache_update_cache[tile_slot] = _make_kv_cache_update_kernel(tile_slot)
    return _kv_cache_update_cache[tile_slot]


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_kv_cache_update(device):
    """Test K^T column insertion + V row insertion at various positions."""
    print("KV cache update tests (K^T col + V row):")

    head_dim = 64
    max_seq = 512
    num_kv_heads = 2

    for pos in [0, 1, 5, 31, 32, 33, 63, 100, 255, 511]:
        tile_slot = pos // TILE
        sub_pos = pos % TILE
        print(f"  pos={pos:3d} (slot={tile_slot}, sub={sub_pos})...", end="", flush=True)

        # Initialize caches
        kt_cache_0_t = torch.randn(head_dim, max_seq, dtype=torch.bfloat16) * 0.1
        kt_cache_1_t = torch.randn(head_dim, max_seq, dtype=torch.bfloat16) * 0.1
        v_cache_0_t = torch.randn(max_seq, head_dim, dtype=torch.bfloat16) * 0.1
        v_cache_1_t = torch.randn(max_seq, head_dim, dtype=torch.bfloat16) * 0.1

        # New K/V values
        k_rot_t = torch.zeros(TILE, num_kv_heads * head_dim, dtype=torch.bfloat16)
        v_out_t = torch.zeros(TILE, num_kv_heads * head_dim, dtype=torch.bfloat16)
        k_rot_t[0, :] = torch.randn(num_kv_heads * head_dim, dtype=torch.bfloat16) * 5.0
        v_out_t[0, :] = torch.randn(num_kv_heads * head_dim, dtype=torch.bfloat16) * 5.0

        # Masks
        rm_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
        rm_t[sub_pos, :] = 1.0
        irm_t = 1.0 - rm_t
        cm_t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
        cm_t[:, sub_pos] = 1.0
        icm_t = 1.0 - cm_t

        # Upload
        kt_c0 = _to_device(kt_cache_0_t, device)
        kt_c1 = _to_device(kt_cache_1_t, device)
        v_c0 = _to_device(v_cache_0_t, device)
        v_c1 = _to_device(v_cache_1_t, device)
        k_rot = _to_device(k_rot_t, device)
        v_out = _to_device(v_out_t, device)
        rm = _to_device(rm_t, device)
        irm = _to_device(irm_t, device)
        cm = _to_device(cm_t, device)
        icm = _to_device(icm_t, device)

        # Run
        kernel = get_kv_cache_update_kernel(tile_slot)
        kernel(k_rot, v_out, kt_c0, kt_c1, v_c0, v_c1, rm, irm, cm, icm)

        # Read back
        kt_c0_r = ttnn.to_torch(kt_c0)
        kt_c1_r = ttnn.to_torch(kt_c1)
        v_c0_r = ttnn.to_torch(v_c0)
        v_c1_r = ttnn.to_torch(v_c1)

        # Expected: K^T column pos updated, V row pos updated
        kt_0_exp = kt_cache_0_t.clone()
        kt_0_exp[:, pos] = k_rot_t[0, :head_dim]  # column insertion
        kt_1_exp = kt_cache_1_t.clone()
        kt_1_exp[:, pos] = k_rot_t[0, head_dim:]
        v_0_exp = v_cache_0_t.clone()
        v_0_exp[pos, :] = v_out_t[0, :head_dim]  # row insertion
        v_1_exp = v_cache_1_t.clone()
        v_1_exp[pos, :] = v_out_t[0, head_dim:]

        def pcc(a, b):
            return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()

        # Check affected tiles only
        ts = tile_slot * TILE
        te = ts + TILE
        pcc_kt0 = pcc(kt_c0_r[:, ts:te], kt_0_exp[:, ts:te])
        pcc_kt1 = pcc(kt_c1_r[:, ts:te], kt_1_exp[:, ts:te])
        pcc_v0 = pcc(v_c0_r[ts:te, :], v_0_exp[ts:te, :])
        pcc_v1 = pcc(v_c1_r[ts:te, :], v_1_exp[ts:te, :])

        ok = all(p > 0.99 for p in [pcc_kt0, pcc_kt1, pcc_v0, pcc_v1])
        print(f" KT0={pcc_kt0:.4f} KT1={pcc_kt1:.4f} V0={pcc_v0:.4f} V1={pcc_v1:.4f}"
              f" {'PASS' if ok else 'FAIL'}")

    print("Done.")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_kv_cache_update(device)
    finally:
        ttnn.close_device(device)
