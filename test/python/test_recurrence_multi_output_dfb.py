# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Loop-carried recurrence whose updated value feeds both the loop-carried
state DFB and a broadcast.

`m_new = max(m_old, reduce_max(x))` is stored to the state DFB and broadcast
for the subtract, so convert-ttl-to-compute fuses it into a multi-output
compute that writes the state DFB and the broadcast staging DFB. The staging
DFB is produced and re-read in the compute thread; its cb_push must stay
ordered before its cb_wait, otherwise the compute pipeline blocks on device.

Kernel: y[c] = x[c] - max(reduce_max(x[0..c])) (running per-row maximum over
chunks, subtracted from the current chunk).
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32
N_CHUNKS = 4
WT = 2  # tiles per chunk along the reduced dimension


@ttl.operation(grid=(1, 1))
def running_max_subtract(x, neg_inf, out):
    x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, WT), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, WT), block_count=2)
    cm_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    m_state_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    seed_cb = ttl.make_dataflow_buffer_like(neg_inf, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        seed = seed_cb.wait()
        m0 = m_state_cb.reserve()
        m0.store(seed)
        for _ in range(N_CHUNKS):
            x_blk = x_cb.wait()
            cm_w = cm_cb.reserve()
            cm_w.store(ttl.math.reduce_max(x_blk, dims=[1]))
            cm = cm_cb.wait()
            m_old = m_state_cb.wait()
            # m_new feeds the state store and the broadcast: multi-output.
            m_new = ttl.math.max(m_old, cm)
            m_next = m_state_cb.reserve()
            m_next.store(m_new)
            m_bc = ttl.block.broadcast(m_new, dims=[1], shape=(1, WT))
            y_w = out_cb.reserve()
            y_w.store(ttl.sub(x_blk, m_bc))
        _ = m_state_cb.wait()

    @ttl.datamovement()
    def dm():
        seed_dst = seed_cb.reserve()
        ttl.copy(neg_inf[0:1, 0:1], seed_dst)
        for c in range(N_CHUNKS):
            x_dst = x_cb.reserve()
            ttl.copy(x[c : c + 1, 0:WT], x_dst)
            y_blk = out_cb.wait()
            ttl.copy(y_blk, out[c : c + 1, 0:WT])

    @ttl.datamovement()
    def dm_unused():
        pass


@pytest.mark.parametrize(
    "dtype, threshold",
    [(torch.bfloat16, 0.99), (torch.float32, 0.9999)],
    ids=["bf16", "fp32"],
)
def test_running_max_subtract(device, dtype, threshold):
    torch.manual_seed(0)
    x = torch.randn(N_CHUNKS * TILE, WT * TILE, dtype=dtype)

    m = torch.full((TILE, 1), -1e30, dtype=torch.float32)
    y_ref = torch.empty_like(x, dtype=torch.float32)
    for c in range(N_CHUNKS):
        xc = x[c * TILE : (c + 1) * TILE, :].float()
        m = torch.maximum(m, xc.amax(dim=1, keepdim=True))
        y_ref[c * TILE : (c + 1) * TILE, :] = xc - m

    x_dram = to_dram(x, device)
    neg_inf_dram = to_dram(torch.full((TILE, TILE), -1e30, dtype=dtype), device)
    out_dram = to_dram(torch.zeros(N_CHUNKS * TILE, WT * TILE, dtype=dtype), device)

    running_max_subtract(x_dram, neg_inf_dram, out_dram)
    ttnn.synchronize_device(device)

    y = ttnn.to_torch(out_dram).float()
    assert_pcc(y_ref, y, threshold)
