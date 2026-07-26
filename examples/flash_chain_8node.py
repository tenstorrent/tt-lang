# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Flash chain 8-node example using liveness-based DFB reuse.

Computes single-query scaled dot-product attention over eight K/V shards and
combines per-node streaming-softmax statistics with a binary tree reduction.

Compute-local constants and temporary tile results remain SSA values so
ttl-insert-intermediate-dfbs can materialize the DFBs required by lowering.
The current configuration uses all 32 physical DFB slots after materialization
and liveness-based allocation.

The shard compute and tree reduction are fused in one TTL operation so per-node
statistics stay in operation-local DFBs instead of round-tripping through DRAM.
The tree reduction mirrors the optimized implementation's ordered receive
staging where current TTL permits it: one receive DFB per payload field is reused
across tree levels in FIFO order, and one send-ready DFB per payload field is
reused across disjoint source roles. The compute state DFBs are reused for
merged state after their local shard values have been consumed.

High-level optimized algorithm:

  for each node's contiguous K/V shard:
    for each K/V chunk:
      update local streaming-softmax statistics `(m, l, o)`
    publish the local `(m, l, o)` payload to the node's tree parent

  reduce `(m, l, o)` triples with the associative online-softmax combine:
    m = max(m_left, m_right)
    l = exp((m_left - m) * scale) * l_left
      + exp((m_right - m) * scale) * l_right
    o = exp((m_left - m) * scale) * o_left
      + exp((m_right - m) * scale) * o_right

  s0: 1 -> 0, 3 -> 2, 5 -> 4, 7 -> 6
  s1: 2 -> 0, 6 -> 4
  s2: 4 -> 0
  node 0 normalizes `o / l` and writes the final output tile.

The `m` state stores unscaled score maxima. The attention scale is applied only
to exponent arguments, matching the optimized SDPA kernels.

Nodes 2, 4, and 6 both receive a child payload and send the merged result to
their parent. Their sender thread waits on `m_send`, `l_send`, and `o_send`, so
compute writes those DFBs only after merging the local state with the child
payload.

Issue tenstorrent/tt-lang#671 tracks structured block subviews for packed
payloads. Until those subviews are supported, the `(m, l, o)` fields require
separate state, send, and receive DFBs.
"""

import math
import os

import torch
import ttnn
import ttl


TILE = ttnn.TILE_SIZE  # Last known working: 32; optimized target: 32.
PNHt = 1  # Last known working: 1; optimized target: 1 tile with an 8-row Q/stat tile.
DHt = 18  # Last known working: 18; optimized target: 18.
vDHt = 8  # Last known working: 8; first program-size failure: 16; optimized target: 16.
Sk_chunk_t = 4  # Last known working: 4; optimized target: 4.
N_CHUNKS = 1  # Last known working: 1; optimized target: 1.
NNODES = 8  # Last known working: 8; optimized target: 8 nodes per reduction group.
KERNEL_CONFIG_BUFFER_SIZE = 96 * 1024

St_per_node = Sk_chunk_t * N_CHUNKS
HEAD_DIM = DHt * TILE
HEAD_DIM_V = vDHt * TILE
SEQ_PER_NODE = St_per_node * TILE
SEQ = SEQ_PER_NODE * NNODES
Q_ROWS = PNHt * TILE
SCALE = 1.0 / math.sqrt(HEAD_DIM)
PCC_THRESHOLD = 0.99


@ttl.operation(grid=(NNODES, 1), fp32_dest_acc_en=False)
def flash_chain_8node(q, k, v, final_out):
    # Each PipeNet transfer currently targets one homogeneous DFB payload. The
    # fields use separate PipeNets until issue tenstorrent/tt-lang#671 provides
    # structured block subviews for packed payloads.
    s0_m = ttl.PipeNet([ttl.Pipe(src=(2 * i + 1, 0), dst=(2 * i, 0)) for i in range(4)])
    s0_l = ttl.PipeNet([ttl.Pipe(src=(2 * i + 1, 0), dst=(2 * i, 0)) for i in range(4)])
    s0_o = ttl.PipeNet([ttl.Pipe(src=(2 * i + 1, 0), dst=(2 * i, 0)) for i in range(4)])
    s1_m = ttl.PipeNet([ttl.Pipe(src=(4 * i + 2, 0), dst=(4 * i, 0)) for i in range(2)])
    s1_l = ttl.PipeNet([ttl.Pipe(src=(4 * i + 2, 0), dst=(4 * i, 0)) for i in range(2)])
    s1_o = ttl.PipeNet([ttl.Pipe(src=(4 * i + 2, 0), dst=(4 * i, 0)) for i in range(2)])
    s2_m = ttl.PipeNet([ttl.Pipe(src=(4, 0), dst=(0, 0))])
    s2_l = ttl.PipeNet([ttl.Pipe(src=(4, 0), dst=(0, 0))])
    s2_o = ttl.PipeNet([ttl.Pipe(src=(4, 0), dst=(0, 0))])

    q_cb = ttl.make_dataflow_buffer_like(k, shape=(PNHt, DHt), block_count=2)
    k_cb = ttl.make_dataflow_buffer_like(k, shape=(Sk_chunk_t, DHt), block_count=2)
    v_cb = ttl.make_dataflow_buffer_like(k, shape=(Sk_chunk_t, vDHt), block_count=2)
    score_cb = ttl.make_dataflow_buffer_like(k, shape=(PNHt, Sk_chunk_t), block_count=2)
    scalar_tmp_cb = ttl.make_dataflow_buffer_like(k, shape=(PNHt, 1), block_count=2)
    m_new_cb = ttl.make_dataflow_buffer_like(k, shape=(PNHt, 1), block_count=2)
    alpha_cb = ttl.make_dataflow_buffer_like(k, shape=(PNHt, 1), block_count=2)
    exp_scores_cb = ttl.make_dataflow_buffer_like(
        k, shape=(PNHt, Sk_chunk_t), block_count=2
    )

    m_state_cb = ttl.make_dataflow_buffer_like(k, shape=(PNHt, 1), block_count=2)
    l_state_cb = ttl.make_dataflow_buffer_like(k, shape=(PNHt, 1), block_count=2)
    o_state_cb = ttl.make_dataflow_buffer_like(k, shape=(PNHt, vDHt), block_count=2)

    # Send-ready DFBs decouple compute from the sender's wait. Compute-carried
    # tree state uses the local state DFBs after local shard state is consumed.
    m_send = ttl.make_dataflow_buffer_like(k, shape=(PNHt, 1), block_count=2)
    l_send = ttl.make_dataflow_buffer_like(k, shape=(PNHt, 1), block_count=2)
    o_send = ttl.make_dataflow_buffer_like(k, shape=(PNHt, vDHt), block_count=2)

    # Pipe receive buffers are filled by recv DM and consumed by compute in
    # tree-level order on every destination node.
    m_recv = ttl.make_dataflow_buffer_like(k, shape=(PNHt, 1), block_count=3)
    l_recv = ttl.make_dataflow_buffer_like(k, shape=(PNHt, 1), block_count=3)
    o_recv = ttl.make_dataflow_buffer_like(k, shape=(PNHt, vDHt), block_count=3)
    out_cb = ttl.make_dataflow_buffer_like(k, shape=(PNHt, vDHt), block_count=2)

    @ttl.compute()
    def compute():
        m0 = m_state_cb.reserve()
        m0.store(ttl.block.fill(-1e30, shape=(PNHt, 1)))
        l0 = l_state_cb.reserve()
        l0.store(ttl.block.fill(0, shape=(PNHt, 1)))
        o0 = o_state_cb.reserve()
        o0.store(ttl.block.fill(0, shape=(PNHt, vDHt)))

        q_blk = q_cb.wait()
        for _ in range(N_CHUNKS):
            k_blk = k_cb.wait()
            v_blk = v_cb.wait()
            score_w = score_cb.reserve()
            score_w.store(q_blk @ ttl.transpose(k_blk))
            score_for_reduce = score_cb.wait()
            cm_w = scalar_tmp_cb.reserve()
            cm_w.store(ttl.math.reduce_max(score_for_reduce, dims=[1]))
            score_replay = score_cb.reserve()
            score_replay.store(score_for_reduce)
            m_old = m_state_cb.wait()
            chunk_max = scalar_tmp_cb.wait()
            m_new_out = m_new_cb.reserve()
            m_new_out.store(ttl.math.max(m_old, chunk_max))
            m_new = m_new_cb.wait()
            alpha_out = alpha_cb.reserve()
            alpha_out.store(ttl.exp(ttl.sub(m_old, m_new) * SCALE))
            m_next = m_state_cb.reserve()
            m_next.store(m_new)

            m_broadcast = ttl.block.broadcast(m_new, dims=[1], shape=(PNHt, Sk_chunk_t))
            qk = score_cb.wait()
            exp_scores_out = exp_scores_cb.reserve()
            exp_scores_out.store(ttl.exp(ttl.sub(qk, m_broadcast) * SCALE))
            exp_scores = exp_scores_cb.wait()
            cs_w = scalar_tmp_cb.reserve()
            cs_w.store(ttl.math.reduce_sum(exp_scores, dims=[1]))
            exp_scores_replay = exp_scores_cb.reserve()
            exp_scores_replay.store(exp_scores)

            l_old = l_state_cb.wait()
            chunk_sum = scalar_tmp_cb.wait()
            alpha = alpha_cb.wait()
            l_next = l_state_cb.reserve()
            l_next.store(ttl.add(ttl.mul(alpha, l_old), chunk_sum))
            alpha_replay = alpha_cb.reserve()
            alpha_replay.store(alpha)

            o_old = o_state_cb.wait()
            alpha_for_broadcast = alpha_cb.wait()
            alpha_broadcast = ttl.block.broadcast(
                alpha_for_broadcast, dims=[1], shape=(PNHt, vDHt)
            )
            o_corrected = ttl.mul(alpha_broadcast, o_old)
            exp_scores_for_matmul = exp_scores_cb.wait()
            partial_v = exp_scores_for_matmul @ v_blk
            o_next = o_state_cb.reserve()
            o_next.store(ttl.add(o_corrected, partial_v))

        # Tree reduction starts here after local shard accumulation.
        m_final = m_state_cb.wait()
        l_final = l_state_cb.wait()
        o_final = o_state_cb.wait()
        if s0_m.is_src():
            m_send_out = m_send.reserve()
            m_send_out.store(m_final)
            l_send_out = l_send.reserve()
            l_send_out.store(l_final)
            o_send_out = o_send.reserve()
            o_send_out.store(o_final)

        if s0_m.is_dst():
            if s1_m.is_dst():
                m_b0 = m_recv.wait()
                l_b0 = l_recv.wait()
                o_b0 = o_recv.wait()
                m_merge_out = scalar_tmp_cb.reserve()
                m_merge_out.store(ttl.math.max(m_final, m_b0))
                m_new = scalar_tmp_cb.wait()
                a_scale0 = ttl.exp(ttl.sub(m_final, m_new) * SCALE)
                b_scale0 = ttl.exp(ttl.sub(m_b0, m_new) * SCALE)
                l_new = ttl.add(ttl.mul(a_scale0, l_final), ttl.mul(b_scale0, l_b0))
                a_scale0_bc = ttl.block.broadcast(
                    a_scale0, dims=[1], shape=(PNHt, vDHt)
                )
                b_scale0_bc = ttl.block.broadcast(
                    b_scale0, dims=[1], shape=(PNHt, vDHt)
                )
                o_new = ttl.add(
                    ttl.mul(a_scale0_bc, o_final), ttl.mul(b_scale0_bc, o_b0)
                )
                m_state_out = m_state_cb.reserve()
                m_state_out.store(m_new)
                l_state_out = l_state_cb.reserve()
                l_state_out.store(l_new)
                o_state_out = o_state_cb.reserve()
                o_state_out.store(o_new)
            if s1_m.is_src():
                m_b0 = m_recv.wait()
                l_b0 = l_recv.wait()
                o_b0 = o_recv.wait()
                m_merge_out = scalar_tmp_cb.reserve()
                m_merge_out.store(ttl.math.max(m_final, m_b0))
                m_new = scalar_tmp_cb.wait()
                a_scale0 = ttl.exp(ttl.sub(m_final, m_new) * SCALE)
                b_scale0 = ttl.exp(ttl.sub(m_b0, m_new) * SCALE)
                l_new = ttl.add(ttl.mul(a_scale0, l_final), ttl.mul(b_scale0, l_b0))
                a_scale0_bc = ttl.block.broadcast(
                    a_scale0, dims=[1], shape=(PNHt, vDHt)
                )
                b_scale0_bc = ttl.block.broadcast(
                    b_scale0, dims=[1], shape=(PNHt, vDHt)
                )
                o_new = ttl.add(
                    ttl.mul(a_scale0_bc, o_final), ttl.mul(b_scale0_bc, o_b0)
                )
                m_send_out = m_send.reserve()
                m_send_out.store(m_new)
                l_send_out = l_send.reserve()
                l_send_out.store(l_new)
                o_send_out = o_send.reserve()
                o_send_out.store(o_new)

        if s1_m.is_dst():
            if s2_m.is_dst():
                m_a1 = m_state_cb.wait()
                l_a1 = l_state_cb.wait()
                o_a1 = o_state_cb.wait()
                m_b1 = m_recv.wait()
                l_b1 = l_recv.wait()
                o_b1 = o_recv.wait()
                m_merge_out = scalar_tmp_cb.reserve()
                m_merge_out.store(ttl.math.max(m_a1, m_b1))
                m_new = scalar_tmp_cb.wait()
                a_scale1 = ttl.exp(ttl.sub(m_a1, m_new) * SCALE)
                b_scale1 = ttl.exp(ttl.sub(m_b1, m_new) * SCALE)
                l_new = ttl.add(ttl.mul(a_scale1, l_a1), ttl.mul(b_scale1, l_b1))
                a_scale1_bc = ttl.block.broadcast(
                    a_scale1, dims=[1], shape=(PNHt, vDHt)
                )
                b_scale1_bc = ttl.block.broadcast(
                    b_scale1, dims=[1], shape=(PNHt, vDHt)
                )
                o_new = ttl.add(ttl.mul(a_scale1_bc, o_a1), ttl.mul(b_scale1_bc, o_b1))
                m_state_out = m_state_cb.reserve()
                m_state_out.store(m_new)
                l_state_out = l_state_cb.reserve()
                l_state_out.store(l_new)
                o_state_out = o_state_cb.reserve()
                o_state_out.store(o_new)
            if s2_m.is_src():
                m_a1 = m_state_cb.wait()
                l_a1 = l_state_cb.wait()
                o_a1 = o_state_cb.wait()
                m_b1 = m_recv.wait()
                l_b1 = l_recv.wait()
                o_b1 = o_recv.wait()
                m_merge_out = scalar_tmp_cb.reserve()
                m_merge_out.store(ttl.math.max(m_a1, m_b1))
                m_new = scalar_tmp_cb.wait()
                a_scale1 = ttl.exp(ttl.sub(m_a1, m_new) * SCALE)
                b_scale1 = ttl.exp(ttl.sub(m_b1, m_new) * SCALE)
                l_new = ttl.add(ttl.mul(a_scale1, l_a1), ttl.mul(b_scale1, l_b1))
                a_scale1_bc = ttl.block.broadcast(
                    a_scale1, dims=[1], shape=(PNHt, vDHt)
                )
                b_scale1_bc = ttl.block.broadcast(
                    b_scale1, dims=[1], shape=(PNHt, vDHt)
                )
                o_new = ttl.add(ttl.mul(a_scale1_bc, o_a1), ttl.mul(b_scale1_bc, o_b1))
                m_send_out = m_send.reserve()
                m_send_out.store(m_new)
                l_send_out = l_send.reserve()
                l_send_out.store(l_new)
                o_send_out = o_send.reserve()
                o_send_out.store(o_new)

        if s2_m.is_dst():
            m_a2 = m_state_cb.wait()
            l_a2 = l_state_cb.wait()
            o_a2 = o_state_cb.wait()
            m_b3 = m_recv.wait()
            l_b3 = l_recv.wait()
            o_b3 = o_recv.wait()
            m_merge_out = scalar_tmp_cb.reserve()
            m_merge_out.store(ttl.math.max(m_a2, m_b3))
            m_fin = scalar_tmp_cb.wait()
            aa3 = ttl.exp(ttl.sub(m_a2, m_fin) * SCALE)
            ab3 = ttl.exp(ttl.sub(m_b3, m_fin) * SCALE)
            l_fin = ttl.add(ttl.mul(aa3, l_a2), ttl.mul(ab3, l_b3))
            aa3_bc = ttl.block.broadcast(aa3, dims=[1], shape=(PNHt, vDHt))
            ab3_bc = ttl.block.broadcast(ab3, dims=[1], shape=(PNHt, vDHt))
            o_unnorm = ttl.add(ttl.mul(aa3_bc, o_a2), ttl.mul(ab3_bc, o_b3))

            out_w = out_cb.reserve()
            l_bc = ttl.block.broadcast(l_fin, dims=[1], shape=(PNHt, vDHt))
            out_w.store(ttl.mul(o_unnorm, ttl.math.recip(l_bc)))

    @ttl.datamovement()
    def dm_sender():
        # Each non-root node sends once. Separate PipeNets are required because
        # each selected pipe currently binds one receiver DFB.
        s0_m.if_src(lambda pipe: ttl.copy(m_send.wait(), pipe).wait())
        s0_l.if_src(lambda pipe: ttl.copy(l_send.wait(), pipe).wait())
        s0_o.if_src(lambda pipe: ttl.copy(o_send.wait(), pipe).wait())

        s1_m.if_src(lambda pipe: ttl.copy(m_send.wait(), pipe).wait())
        s1_l.if_src(lambda pipe: ttl.copy(l_send.wait(), pipe).wait())
        s1_o.if_src(lambda pipe: ttl.copy(o_send.wait(), pipe).wait())

        s2_m.if_src(lambda pipe: ttl.copy(m_send.wait(), pipe).wait())
        s2_l.if_src(lambda pipe: ttl.copy(l_send.wait(), pipe).wait())
        s2_o.if_src(lambda pipe: ttl.copy(o_send.wait(), pipe).wait())

    @ttl.datamovement()
    def dm_recv_io():
        node_x, _ = ttl.node(dims=2)

        # Load this node's K/V slice before waiting for child statistics.
        q_dst = q_cb.reserve()
        ttl.copy(q[0:PNHt, 0:DHt], q_dst)
        k_base = node_x * St_per_node
        for chunk_index in range(N_CHUNKS):
            key_row = k_base + chunk_index * Sk_chunk_t
            k_dst = k_cb.reserve()
            ttl.copy(k[key_row : key_row + Sk_chunk_t, 0:DHt], k_dst)
            v_dst = v_cb.reserve()
            ttl.copy(v[key_row : key_row + Sk_chunk_t, 0:vDHt], v_dst)

        # Destination nodes receive child payloads into FIFO DFBs. Compute
        # consumes the same DFBs in tree-level order.
        s0_m.if_dst(lambda pipe: ttl.copy(pipe, m_recv.reserve()).wait())
        s0_l.if_dst(lambda pipe: ttl.copy(pipe, l_recv.reserve()).wait())
        s0_o.if_dst(lambda pipe: ttl.copy(pipe, o_recv.reserve()).wait())

        s1_m.if_dst(lambda pipe: ttl.copy(pipe, m_recv.reserve()).wait())
        s1_l.if_dst(lambda pipe: ttl.copy(pipe, l_recv.reserve()).wait())
        s1_o.if_dst(lambda pipe: ttl.copy(pipe, o_recv.reserve()).wait())

        s2_m.if_dst(lambda pipe: ttl.copy(pipe, m_recv.reserve()).wait())
        s2_l.if_dst(lambda pipe: ttl.copy(pipe, l_recv.reserve()).wait())
        s2_o.if_dst(lambda pipe: ttl.copy(pipe, o_recv.reserve()).wait())

        if s2_m.is_dst():
            out_blk = out_cb.wait()
            ttl.copy(out_blk, final_out[0:PNHt, 0:vDHt])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _to_dram(device, tensor):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _compute_pcc(golden, actual):
    golden_flat = torch.nan_to_num(golden.float().flatten())
    actual_flat = torch.nan_to_num(actual.float().flatten())

    if torch.any(golden_flat.bool()).item() != torch.any(actual_flat.bool()).item():
        return 0.0

    if torch.equal(golden_flat, actual_flat):
        return 1.0

    pcc = torch.corrcoef(torch.stack([golden_flat, actual_flat]))[0, 1].item()
    if not math.isfinite(pcc):
        return 1.0
    return pcc


def main():
    torch.manual_seed(2026)
    compile_only = os.getenv("TTLANG_COMPILE_ONLY") == "1"
    max_worker_l1_size = ttnn.device.get_max_worker_l1_unreserved_size()
    worker_l1_size = max_worker_l1_size - KERNEL_CONFIG_BUFFER_SIZE
    device = ttnn.open_device(device_id=0, worker_l1_size=worker_l1_size)
    try:
        q_torch = torch.randn(Q_ROWS, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        k_torch = torch.randn(SEQ, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        v_torch = torch.randn(SEQ, HEAD_DIM_V, dtype=torch.bfloat16) * 0.1

        q_ref = q_torch.float().unsqueeze(0).unsqueeze(0)
        k_ref = k_torch.float().unsqueeze(0).unsqueeze(0)
        v_ref = v_torch.float().unsqueeze(0).unsqueeze(0)
        o_ref = (
            torch.nn.functional.scaled_dot_product_attention(
                q_ref,
                k_ref,
                v_ref,
                scale=SCALE,
            )
            .squeeze(0)
            .squeeze(0)
            .to(torch.bfloat16)
        )

        q_dram = _to_dram(device, q_torch)
        k_dram = _to_dram(device, k_torch)
        v_dram = _to_dram(device, v_torch)

        final_dram = _to_dram(
            device,
            torch.zeros(Q_ROWS, HEAD_DIM_V, dtype=torch.bfloat16),
        )

        print("Running flash_chain_8node...")
        flash_chain_8node(q_dram, k_dram, v_dram, final_dram)
        ttnn.synchronize_device(device)

        if compile_only:
            print("COMPILE_ONLY PASS")
            return

        out = ttnn.to_torch(final_dram).reshape(Q_ROWS, HEAD_DIM_V).to(torch.bfloat16)
        pcc = _compute_pcc(o_ref, out)
        print(f"flash_chain8  PCC={pcc:.6f}")
        print(f"  out[0,:4] = {out[0,:4].tolist()}")
        print(f"  ref[0,:4] = {o_ref[0,:4].tolist()}")
        if pcc < PCC_THRESHOLD:
            raise AssertionError(
                f"correctness failed: PCC={pcc}, threshold={PCC_THRESHOLD}"
            )
        print("CHAIN8 PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
