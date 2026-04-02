# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 SwiGLU MLP -- Hardware version.

All matmuls use (1,1)@(1,1) blocks with explicit K-loop accumulation,
following the proven pattern from matmul_acc.py.

Config: hidden=128 (4 tiles), mlp=384 (12 tiles)
"""

import torch
import ttnn
import ttl

TILE = 32
SEQ_T = 1
HID_T = 4
MLP_T = 12


# ============================================================================
# Phase 1: Gate + Up + SwiGLU
#   gate = x @ W_gate : (1,4) @ (4,12) via K-loop of (1,1)@(1,1)
#   up = x @ W_up     : same
#   swiglu = silu(gate) * up
# ============================================================================
@ttl.operation(grid=(1, 1))
def swiglu_phase1(x, w_gate, w_up, swiglu_out):
    a_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(w_gate, shape=(1, 1), buffer_factor=2)
    # Accumulator ping-pong for gate projection
    acc_dfb = ttl.make_dataflow_buffer_like(swiglu_out, shape=(1, 1), buffer_factor=2)
    # Bias (zero) to initialize accumulator
    bias_dfb = ttl.make_dataflow_buffer_like(swiglu_out, shape=(1, 1), buffer_factor=2)
    # Gate and up results
    gate_dfb = ttl.make_dataflow_buffer_like(swiglu_out, shape=(1, 1), buffer_factor=2)
    up_dfb = ttl.make_dataflow_buffer_like(swiglu_out, shape=(1, 1), buffer_factor=2)
    # SwiGLU output
    out_dfb = ttl.make_dataflow_buffer_like(swiglu_out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        # For each output column n:
        for _ in range(MLP_T):
            # Gate projection: sum over k of x[0,k] @ w_gate[k,n]
            with bias_dfb.wait() as z:
                with acc_dfb.reserve() as acc:
                    acc.store(z)
            for _ in range(HID_T):
                with a_dfb.wait() as a, b_dfb.wait() as b:
                    with acc_dfb.wait() as prev, acc_dfb.reserve() as acc:
                        acc.store(prev + a @ b)
            with acc_dfb.wait() as gate_val, gate_dfb.reserve() as g:
                g.store(gate_val)

            # Up projection: same loop
            with bias_dfb.wait() as z:
                with acc_dfb.reserve() as acc:
                    acc.store(z)
            for _ in range(HID_T):
                with a_dfb.wait() as a, b_dfb.wait() as b:
                    with acc_dfb.wait() as prev, acc_dfb.reserve() as acc:
                        acc.store(prev + a @ b)
            with acc_dfb.wait() as up_val, up_dfb.reserve() as u:
                u.store(up_val)

            # SwiGLU: silu(gate) * up
            with gate_dfb.wait() as gv, up_dfb.wait() as uv, out_dfb.reserve() as o:
                silu = gv * ttl.math.sigmoid(gv)
                o.store(silu * uv)

    @ttl.datamovement()
    def dm_read():
        for n in range(MLP_T):
            # Bias (zero) for gate acc init
            with bias_dfb.reserve() as blk:
                tx = ttl.copy(swiglu_out[0, 0], blk)  # zeros
                tx.wait()
            # Gate K-loop: stream x tiles and w_gate column n
            for k in range(HID_T):
                with a_dfb.reserve() as ablk, b_dfb.reserve() as bblk:
                    tx_a = ttl.copy(x[0, k], ablk)
                    tx_a.wait()
                    tx_b = ttl.copy(w_gate[k, n], bblk)
                    tx_b.wait()

            # Bias (zero) for up acc init
            with bias_dfb.reserve() as blk:
                tx = ttl.copy(swiglu_out[0, 0], blk)  # zeros
                tx.wait()
            # Up K-loop
            for k in range(HID_T):
                with a_dfb.reserve() as ablk, b_dfb.reserve() as bblk:
                    tx_a = ttl.copy(x[0, k], ablk)
                    tx_a.wait()
                    tx_b = ttl.copy(w_up[k, n], bblk)
                    tx_b.wait()

    @ttl.datamovement()
    def dm_write():
        for n in range(MLP_T):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, swiglu_out[0, n])
                tx.wait()


# ============================================================================
# Phase 2: Down projection
#   out = swiglu @ W_down : (1,12) @ (12,4) via K-loop of (1,1)@(1,1)
# ============================================================================
@ttl.operation(grid=(1, 1))
def down_proj_phase2(swiglu_in, w_down, bias, out):
    a_dfb = ttl.make_dataflow_buffer_like(swiglu_in, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(w_down, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        for _ in range(HID_T):  # Output columns
            with bias_dfb.wait() as z:
                with acc_dfb.reserve() as acc:
                    acc.store(z)
            for _ in range(MLP_T):  # K dimension
                with a_dfb.wait() as a, b_dfb.wait() as b:
                    with acc_dfb.wait() as prev, acc_dfb.reserve() as acc:
                        acc.store(prev + a @ b)
            with acc_dfb.wait() as result, out_dfb.reserve() as o:
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        for n in range(HID_T):
            with bias_dfb.reserve() as blk:
                tx = ttl.copy(bias[0, 0], blk)
                tx.wait()
            for k in range(MLP_T):
                with a_dfb.reserve() as ablk, b_dfb.reserve() as bblk:
                    tx_a = ttl.copy(swiglu_in[0, k], ablk)
                    tx_a.wait()
                    tx_b = ttl.copy(w_down[k, n], bblk)
                    tx_b.wait()

    @ttl.datamovement()
    def dm_write():
        for n in range(HID_T):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, n])
                tx.wait()


# ============================================================================
# Test
# ============================================================================
def main():
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    seq, hid, mlp = SEQ_T * TILE, HID_T * TILE, MLP_T * TILE

    def d(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x_t = torch.randn(seq, hid, dtype=torch.bfloat16) * 0.1
    wg_t = torch.randn(hid, mlp, dtype=torch.bfloat16) * 0.02
    wu_t = torch.randn(hid, mlp, dtype=torch.bfloat16) * 0.02
    wd_t = torch.randn(mlp, hid, dtype=torch.bfloat16) * 0.02

    sw_dev = d(torch.zeros(seq, mlp, dtype=torch.bfloat16))
    out_dev = d(torch.zeros(seq, hid, dtype=torch.bfloat16))
    bias_dev = d(torch.zeros(seq, hid, dtype=torch.bfloat16))

    print("FLUX.2 SwiGLU MLP on Hardware (tile-by-tile K-loop)\n")

    print("Phase 1: Gate + Up + SwiGLU...")
    swiglu_phase1(d(x_t), d(wg_t), d(wu_t), sw_dev)

    sw_result = ttnn.to_torch(sw_dev)
    x_f = x_t.float()
    gate_ref = x_f @ wg_t.float()
    up_ref = x_f @ wu_t.float()
    sw_exp = (gate_ref * torch.sigmoid(gate_ref) * up_ref).bfloat16()
    sw_corr = torch.corrcoef(
        torch.stack([sw_result.float().flatten(), sw_exp.float().flatten()])
    )[0, 1].item()
    print(f"  SwiGLU correlation: {sw_corr:.6f}")

    print("\nPhase 2: Down projection...")
    down_proj_phase2(sw_dev, d(wd_t), bias_dev, out_dev)

    result = ttnn.to_torch(out_dev)
    full_exp = (sw_exp.float() @ wd_t.float()).bfloat16()
    full_corr = torch.corrcoef(
        torch.stack([result.float().flatten(), full_exp.float().flatten()])
    )[0, 1].item()
    print(f"  Full MLP correlation: {full_corr:.6f}")

    print(f"\nExpected[0,:8]: {full_exp[0,:8]}")
    print(f"Result[0,:8]:   {result[0,:8]}")

    if sw_corr > 0.95 and full_corr > 0.95:
        print("\nPASSED: SwiGLU MLP on hardware")
    else:
        print(f"\nFAILED")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
