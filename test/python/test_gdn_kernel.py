# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
test_gdn_kernel.py - Gated Delta Rule (GDN) kernel in pure TT-Lang.

Implements the recurrent gated delta rule per timestep:
    S_t = alpha * S_{t-1} + beta * k ⊗ (v - (alpha * S_{t-1})^T @ k)
    o_t = S_t^T @ q

Single fused kernel using implicit intermediates (BlockExprs).
The `with` context manager keeps input blocks alive for reuse,
eliminating duplicate DFBs and intermediate buffers.

Usage:
    run-test.sh --hw test_gdn_kernel.py
"""

import torch
import torch.nn.functional as F
import ttnn
import ttl

TILE = 32
D = 32


# ---- Helpers ----


def pcc(a, b):
    a_f, b_f = a.flatten().float(), b.flatten().float()
    if a_f.std() < 1e-8 and b_f.std() < 1e-8:
        return 1.0
    if a_f.std() < 1e-8 or b_f.std() < 1e-8:
        return 0.0
    return torch.corrcoef(torch.stack([a_f, b_f]))[0, 1].item()


def vec_to_tile(vec):
    t = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    t[: len(vec), 0] = vec.bfloat16()
    return t


def tile_to_vec(tile, d):
    return tile[:d, 0].clone()


def scalar_to_tile(val):
    return torch.full((TILE, TILE), float(val), dtype=torch.bfloat16)


def to_tt(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---- PyTorch reference ----


def gdn_step_ref(S, q, k, v, alpha, beta):
    S_sc = alpha * S
    e = S_sc.T @ k
    delta = v - e
    S_new = S_sc + beta * torch.outer(k, delta)
    o = S_new.T @ q
    return S_new, o


# ---- TT-Lang kernel (fused, implicit intermediates) ----


@ttl.operation(grid=(1, 1))
def gdn_step(state_in, q_in, k_in, v_in, alpha_in, beta_in, state_out, out):
    """Fused GDN step: state update + output read in one kernel.

    All intermediate values (s_scaled, st, e, delta, ...) are implicit
    BlockExprs that live in DST registers. The `with` context keeps
    input blocks alive so k and s_scaled can be reused without
    duplicate DFBs.
    """
    si = ttl.make_dataflow_buffer_like(state_in, shape=(1, 1), block_count=2)
    qi = ttl.make_dataflow_buffer_like(q_in, shape=(1, 1), block_count=2)
    ki = ttl.make_dataflow_buffer_like(k_in, shape=(1, 1), block_count=2)
    vi = ttl.make_dataflow_buffer_like(v_in, shape=(1, 1), block_count=2)
    ai = ttl.make_dataflow_buffer_like(alpha_in, shape=(1, 1), block_count=2)
    bi = ttl.make_dataflow_buffer_like(beta_in, shape=(1, 1), block_count=2)
    so = ttl.make_dataflow_buffer_like(state_out, shape=(1, 1), block_count=2)
    oo = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    # One thread-local DFB to bridge state between the two store phases
    sn_local = ttl.make_dataflow_buffer_like(state_in, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        # Phase 1: compute S_new, store to state_out and local buffer
        with (
            si.wait() as s,
            ai.wait() as a,
            ki.wait() as k,
            vi.wait() as v,
            bi.wait() as b,
        ):
            s_scaled = a * s
            st = ttl.transpose(s_scaled)
            e = st @ k
            delta = v - e
            dt = ttl.transpose(delta)
            outer = k @ dt
            s_new = s_scaled + b * outer

            with so.reserve() as so_blk:
                so_blk.store(s_new)
            with sn_local.reserve() as snl:
                snl.store(s_new)

        # Phase 2: compute output from local copy of S_new
        with sn_local.wait() as sn, qi.wait() as q:
            snt = ttl.transpose(sn)
            output = snt @ q
            with oo.reserve() as oo_blk:
                oo_blk.store(output)

    @ttl.datamovement()
    def dm_read():
        with si.reserve() as blk:
            ttl.copy(state_in[0, 0], blk).wait()
        with qi.reserve() as blk:
            ttl.copy(q_in[0, 0], blk).wait()
        with ki.reserve() as blk:
            ttl.copy(k_in[0, 0], blk).wait()
        with vi.reserve() as blk:
            ttl.copy(v_in[0, 0], blk).wait()
        with ai.reserve() as blk:
            ttl.copy(alpha_in[0, 0], blk).wait()
        with bi.reserve() as blk:
            ttl.copy(beta_in[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with so.wait() as blk:
            ttl.copy(blk, state_out[0, 0]).wait()
        with oo.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


# ---- Test ----

import pytest


def test_gdn():
    device = ttnn.open_device(device_id=0)
    try:
        seq_len = 8
        torch.manual_seed(42)

        Q = F.normalize(torch.randn(seq_len, D), p=2, dim=-1)
        K = F.normalize(torch.randn(seq_len, D), p=2, dim=-1)
        V = torch.randn(seq_len, D)
        g_vals = -torch.abs(torch.randn(seq_len))
        alpha_vals = torch.exp(g_vals)
        beta_vals = torch.sigmoid(torch.randn(seq_len))

        # -- PyTorch reference (float32) --
        S_ref = torch.zeros(D, D)
        ref_outputs = []
        for t in range(seq_len):
            S_ref, o_ref = gdn_step_ref(
                S_ref, Q[t], K[t], V[t], alpha_vals[t].item(), beta_vals[t].item()
            )
            ref_outputs.append(o_ref)

        # -- TT-Lang (bfloat16 on device) --
        S_tile = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
        ttl_outputs = []

        for t in range(seq_len):
            q_tile = vec_to_tile(Q[t])
            k_tile = vec_to_tile(K[t])
            v_tile = vec_to_tile(V[t])
            a_tile = scalar_to_tile(alpha_vals[t].item())
            b_tile = scalar_to_tile(beta_vals[t].item())

            s_out_tt = to_tt(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
            o_out_tt = to_tt(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

            gdn_step(
                to_tt(S_tile, device),
                to_tt(q_tile, device),
                to_tt(k_tile, device),
                to_tt(v_tile, device),
                to_tt(a_tile, device),
                to_tt(b_tile, device),
                s_out_tt,
                o_out_tt,
            )

            S_tile = ttnn.to_torch(s_out_tt)
            o_tile = ttnn.to_torch(o_out_tt)
            ttl_outputs.append(tile_to_vec(o_tile, D))

        # -- Compare PCC --
        print("=" * 50)
        print("GDN Kernel PCC Results  (D=32, seq=8)")
        print("=" * 50)

        all_pass = True
        for t in range(seq_len):
            p = pcc(ref_outputs[t], ttl_outputs[t].float())
            status = "PASS" if p > 0.95 else "FAIL"
            if p <= 0.95:
                all_pass = False
            print(f"  step {t}: PCC = {p:.6f}  [{status}]")

        all_ref = torch.stack(ref_outputs)
        all_ttl = torch.stack([o.float() for o in ttl_outputs])
        overall = pcc(all_ref, all_ttl)
        print(f"\n  overall PCC:     {overall:.6f}")

        S_ref_tile = torch.zeros(TILE, TILE)
        S_ref_tile[:D, :D] = S_ref
        sp = pcc(S_ref_tile, S_tile.float())
        print(f"  final state PCC: {sp:.6f}")

        if all_pass and overall > 0.92:
            print("\nPASS")
        else:
            print("\nFAIL")
        assert overall > 0.92, f"overall PCC too low: {overall:.6f}"

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_gdn()
