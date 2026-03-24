# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Trace-compatible KV cache update: single kernel for all positions.

Pre-computed full-width masks [TILE, 512] encode both the target slot AND
sub-tile position. For non-target slots, mask=0/inv_mask=1 → old unchanged.
For target slot, mask has the row/col pattern → modification happens.

8 cores, 2 tile-slots each. All tensors are pre-allocated, data updated
per token via copy_host_to_device_tensor.
"""

import torch
import ttl
import ttnn

TILE = 32
CACHE_TILES = 16
NUM_UPDATE_CORES = 8
SLOTS_PER_CORE = CACHE_TILES // NUM_UPDATE_CORES  # 2


@ttl.kernel(grid=(1, NUM_UPDATE_CORES))
def kv_cache_update_traced(
    k_rot,           # [TILE, 128] — new K
    v_out,           # [TILE, 128] — new V
    kt_cache_0,      # [64, 512]  — K^T cache head 0
    kt_cache_1,      # [64, 512]  — K^T cache head 1
    v_cache_0,       # [512, 64]  — V cache head 0
    v_cache_1,       # [512, 64]  — V cache head 1
    row_masks_full,  # [TILE, 512] — row mask per slot (zero except target)
    irow_masks_full, # [TILE, 512] — inv row mask per slot (ones except target)
    col_masks_full,  # [TILE, 512] — col mask per slot
    icol_masks_full, # [TILE, 512] — inv col mask per slot
):
    Nt = 2  # head_dim / TILE

    new_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    old_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    rm_dfb = ttl.make_dataflow_buffer_like(row_masks_full, shape=(1, 1), buffer_factor=2)
    irm_dfb = ttl.make_dataflow_buffer_like(irow_masks_full, shape=(1, 1), buffer_factor=2)
    trans_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    masked_new_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    zeroed_old_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    result_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)

    y_size, x_size = ttl.grid_size(dims=2)

    @ttl.datamovement()
    def read():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        slot_start = nid * SLOTS_PER_CORE

        for slot_local in range(SLOTS_PER_CORE):
            ts = slot_start + slot_local

            # K^T head 0: 2 dim-row tiles
            for dim_row in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(k_rot[0, dim_row], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(kt_cache_0[dim_row, ts], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(col_masks_full[0, ts], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(icol_masks_full[0, ts], blk)
                    tx.wait()

            # K^T head 1: 2 dim-row tiles
            for dim_row in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(k_rot[0, Nt + dim_row], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(kt_cache_1[dim_row, ts], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(col_masks_full[0, ts], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(icol_masks_full[0, ts], blk)
                    tx.wait()

            # V head 0: 2 dim-col tiles
            for dim_col in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(v_out[0, dim_col], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(v_cache_0[ts, dim_col], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(row_masks_full[0, ts], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(irow_masks_full[0, ts], blk)
                    tx.wait()

            # V head 1: 2 dim-col tiles
            for dim_col in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(v_out[0, Nt + dim_col], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(v_cache_1[ts, dim_col], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(row_masks_full[0, ts], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(irow_masks_full[0, ts], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(SLOTS_PER_CORE):
            # K^T: 4 tiles (transpose + col broadcast + mask)
            for _ in range(4):
                with new_dfb.wait() as new_val, old_dfb.wait() as old_val:
                    with rm_dfb.wait() as cm, irm_dfb.wait() as icm:
                        with trans_dfb.reserve() as tr:
                            tr.store(ttl.math.transpose(new_val, tr))
                        with trans_dfb.wait() as tr_val:
                            with bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(tr_val, bc, dims=[1]))
                        with bcast_dfb.wait() as bc_val:
                            with masked_new_dfb.reserve() as mn:
                                mn.store(bc_val * cm)
                        with zeroed_old_dfb.reserve() as zo:
                            zo.store(old_val * icm)
                        with zeroed_old_dfb.wait() as zo, masked_new_dfb.wait() as mn:
                            with result_dfb.reserve() as res:
                                res.store(zo + mn)

            # V: 4 tiles (row broadcast + mask)
            for _ in range(4):
                with new_dfb.wait() as new_val, old_dfb.wait() as old_val:
                    with rm_dfb.wait() as rm, irm_dfb.wait() as irm:
                        with bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(new_val, bc, dims=[0]))
                        with bcast_dfb.wait() as bc_val:
                            with masked_new_dfb.reserve() as mn:
                                mn.store(bc_val * rm)
                        with zeroed_old_dfb.reserve() as zo:
                            zo.store(old_val * irm)
                        with zeroed_old_dfb.wait() as zo, masked_new_dfb.wait() as mn:
                            with result_dfb.reserve() as res:
                                res.store(zo + mn)

    @ttl.datamovement()
    def write():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        slot_start = nid * SLOTS_PER_CORE

        for slot_local in range(SLOTS_PER_CORE):
            ts = slot_start + slot_local
            # K^T head 0
            for dim_row in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, kt_cache_0[dim_row, ts])
                    tx.wait()
            # K^T head 1
            for dim_row in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, kt_cache_1[dim_row, ts])
                    tx.wait()
            # V head 0
            for dim_col in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, v_cache_0[ts, dim_col])
                    tx.wait()
            # V head 1
            for dim_col in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, v_cache_1[ts, dim_col])
                    tx.wait()


@ttl.kernel(grid=(1, NUM_UPDATE_CORES))
def kv_cache_update_stacked(
    k_rot,           # [TILE, 128] — new K
    v_out,           # [TILE, 128] — new V
    kt_stacked,      # [128, 512] — K^T both heads stacked (head0 rows 0-63, head1 rows 64-127)
    v_stacked,       # [1024, 64] — V both heads stacked (head0 rows 0-511, head1 rows 512-1023)
    row_masks_full,  # [TILE, 512]
    irow_masks_full, # [TILE, 512]
    col_masks_full,  # [TILE, 512]
    icol_masks_full, # [TILE, 512]
):
    """KV cache update for stacked (both-heads) tensors."""
    Nt = 2  # head_dim / TILE
    V_TILE_ROWS = CACHE_TILES  # 16 tile rows per V head

    new_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    old_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    rm_dfb = ttl.make_dataflow_buffer_like(row_masks_full, shape=(1, 1), buffer_factor=2)
    irm_dfb = ttl.make_dataflow_buffer_like(irow_masks_full, shape=(1, 1), buffer_factor=2)
    trans_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    masked_new_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    zeroed_old_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)
    result_dfb = ttl.make_dataflow_buffer_like(k_rot, shape=(1, 1), buffer_factor=2)

    y_size, x_size = ttl.grid_size(dims=2)

    @ttl.datamovement()
    def read():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        slot_start = nid * SLOTS_PER_CORE

        for slot_local in range(SLOTS_PER_CORE):
            ts = slot_start + slot_local

            # K^T head 0: rows 0..Nt-1 of kt_stacked
            for dim_row in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(k_rot[0, dim_row], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(kt_stacked[dim_row, ts], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(col_masks_full[0, ts], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(icol_masks_full[0, ts], blk)
                    tx.wait()

            # K^T head 1: rows Nt..2*Nt-1 of kt_stacked
            for dim_row in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(k_rot[0, Nt + dim_row], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(kt_stacked[Nt + dim_row, ts], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(col_masks_full[0, ts], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(icol_masks_full[0, ts], blk)
                    tx.wait()

            # V head 0: rows 0..V_TILE_ROWS-1 of v_stacked
            for dim_col in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(v_out[0, dim_col], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(v_stacked[ts, dim_col], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(row_masks_full[0, ts], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(irow_masks_full[0, ts], blk)
                    tx.wait()

            # V head 1: rows V_TILE_ROWS..2*V_TILE_ROWS-1 of v_stacked
            for dim_col in range(Nt):
                with new_dfb.reserve() as blk:
                    tx = ttl.copy(v_out[0, Nt + dim_col], blk)
                    tx.wait()
                with old_dfb.reserve() as blk:
                    tx = ttl.copy(v_stacked[V_TILE_ROWS + ts, dim_col], blk)
                    tx.wait()
                with rm_dfb.reserve() as blk:
                    tx = ttl.copy(row_masks_full[0, ts], blk)
                    tx.wait()
                with irm_dfb.reserve() as blk:
                    tx = ttl.copy(irow_masks_full[0, ts], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(SLOTS_PER_CORE):
            # K^T: 4 tiles (transpose + col broadcast + mask)
            for _ in range(4):
                with new_dfb.wait() as new_val, old_dfb.wait() as old_val:
                    with rm_dfb.wait() as cm, irm_dfb.wait() as icm:
                        with trans_dfb.reserve() as tr:
                            tr.store(ttl.math.transpose(new_val, tr))
                        with trans_dfb.wait() as t_val:
                            with bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(t_val, bc, dims=[1]))
                        with bcast_dfb.wait() as bc_val:
                            with masked_new_dfb.reserve() as mn:
                                mn.store(bc_val * cm)
                        with zeroed_old_dfb.reserve() as zo:
                            zo.store(old_val * icm)
                        with zeroed_old_dfb.wait() as zo, masked_new_dfb.wait() as mn:
                            with result_dfb.reserve() as res:
                                res.store(zo + mn)

            # V: 4 tiles (row broadcast + mask)
            for _ in range(4):
                with new_dfb.wait() as new_val, old_dfb.wait() as old_val:
                    with rm_dfb.wait() as rm, irm_dfb.wait() as irm:
                        with bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(new_val, bc, dims=[0]))
                        with bcast_dfb.wait() as bc_val:
                            with masked_new_dfb.reserve() as mn:
                                mn.store(bc_val * rm)
                        with zeroed_old_dfb.reserve() as zo:
                            zo.store(old_val * irm)
                        with zeroed_old_dfb.wait() as zo, masked_new_dfb.wait() as mn:
                            with result_dfb.reserve() as res:
                                res.store(zo + mn)

    @ttl.datamovement()
    def write():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        slot_start = nid * SLOTS_PER_CORE

        for slot_local in range(SLOTS_PER_CORE):
            ts = slot_start + slot_local
            # K^T head 0
            for dim_row in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, kt_stacked[dim_row, ts])
                    tx.wait()
            # K^T head 1
            for dim_row in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, kt_stacked[Nt + dim_row, ts])
                    tx.wait()
            # V head 0
            for dim_col in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, v_stacked[ts, dim_col])
                    tx.wait()
            # V head 1
            for dim_col in range(Nt):
                with result_dfb.wait() as blk:
                    tx = ttl.copy(blk, v_stacked[V_TILE_ROWS + ts, dim_col])
                    tx.wait()


def build_full_masks(pos):
    """Build [TILE, 512] mask tensors for a given position."""
    tile_slot = pos // TILE
    sub_pos = pos % TILE

    row_m = torch.zeros(TILE, CACHE_TILES * TILE, dtype=torch.bfloat16)
    irow_m = torch.ones(TILE, CACHE_TILES * TILE, dtype=torch.bfloat16)
    col_m = torch.zeros(TILE, CACHE_TILES * TILE, dtype=torch.bfloat16)
    icol_m = torch.ones(TILE, CACHE_TILES * TILE, dtype=torch.bfloat16)

    # Only the target slot's tile gets the mask pattern
    ts = tile_slot * TILE
    te = ts + TILE
    row_m[sub_pos, ts:te] = 1.0
    irow_m[sub_pos, ts:te] = 0.0
    col_m[ts:te, sub_pos] = 1.0  # Wait, col_m is [TILE, 512] not [512, TILE]
    # col_mask: within the target slot's [TILE, TILE] subtile, column sub_pos = 1.0
    # The [TILE, TILE] subtile at position tile_slot is col_m[:, ts:te]
    # We want col_m[row, ts + sub_pos] = 1.0 for all rows
    col_m[:, ts + sub_pos] = 1.0
    icol_m[:, ts + sub_pos] = 0.0

    return row_m, irow_m, col_m, icol_m


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_traced_kv_update(device):
    """Test trace-compatible KV cache update."""
    from kv_cache_update import get_kv_cache_update_kernel

    print("Traced KV cache update test:")

    HD, MAX_SEQ, KV = 64, 512, 128

    for pos in [0, 5, 31, 32, 100, 511]:
        print(f"  pos={pos:3d}...", end="", flush=True)

        kt0 = torch.randn(HD, MAX_SEQ, dtype=torch.bfloat16) * 0.1
        kt1 = torch.randn(HD, MAX_SEQ, dtype=torch.bfloat16) * 0.1
        v0 = torch.randn(MAX_SEQ, HD, dtype=torch.bfloat16) * 0.1
        v1 = torch.randn(MAX_SEQ, HD, dtype=torch.bfloat16) * 0.1
        k_rot_t = torch.zeros(TILE, KV, dtype=torch.bfloat16)
        v_out_t = torch.zeros(TILE, KV, dtype=torch.bfloat16)
        k_rot_t[0, :] = torch.randn(KV, dtype=torch.bfloat16) * 5.0
        v_out_t[0, :] = torch.randn(KV, dtype=torch.bfloat16) * 5.0

        # Reference: original kernel
        ref_kt0 = _to_device(kt0.clone(), device)
        ref_kt1 = _to_device(kt1.clone(), device)
        ref_v0 = _to_device(v0.clone(), device)
        ref_v1 = _to_device(v1.clone(), device)
        sub_pos = pos % TILE
        rm = torch.zeros(TILE, TILE, dtype=torch.bfloat16); rm[sub_pos, :] = 1.0
        irm = 1.0 - rm
        cm = torch.zeros(TILE, TILE, dtype=torch.bfloat16); cm[:, sub_pos] = 1.0
        icm = 1.0 - cm
        ref_kern = get_kv_cache_update_kernel(pos // TILE)
        ref_kern(_to_device(k_rot_t, device), _to_device(v_out_t, device),
                 ref_kt0, ref_kt1, ref_v0, ref_v1,
                 _to_device(rm, device), _to_device(irm, device),
                 _to_device(cm, device), _to_device(icm, device))
        ref_kt0_t = ttnn.to_torch(ref_kt0)
        ref_v0_t = ttnn.to_torch(ref_v0)

        # Traced kernel
        row_m, irow_m, col_m, icol_m = build_full_masks(pos)
        trc_kt0 = _to_device(kt0.clone(), device)
        trc_kt1 = _to_device(kt1.clone(), device)
        trc_v0 = _to_device(v0.clone(), device)
        trc_v1 = _to_device(v1.clone(), device)
        kv_cache_update_traced(
            _to_device(k_rot_t, device), _to_device(v_out_t, device),
            trc_kt0, trc_kt1, trc_v0, trc_v1,
            _to_device(row_m, device), _to_device(irow_m, device),
            _to_device(col_m, device), _to_device(icol_m, device),
        )
        trc_kt0_t = ttnn.to_torch(trc_kt0)
        trc_v0_t = ttnn.to_torch(trc_v0)

        pcc_kt = torch.corrcoef(torch.stack([
            ref_kt0_t.float().flatten(), trc_kt0_t.float().flatten()
        ]))[0, 1].item()
        pcc_v = torch.corrcoef(torch.stack([
            ref_v0_t.float().flatten(), trc_v0_t.float().flatten()
        ]))[0, 1].item()
        ok = pcc_kt > 0.99 and pcc_v > 0.99
        print(f" KT={pcc_kt:.4f} V={pcc_v:.4f} {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_traced_kv_update(device)
    finally:
        ttnn.close_device(device)
