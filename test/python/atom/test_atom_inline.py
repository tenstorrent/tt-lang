# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""@ttl.atom-in-@ttl.atom inlining. A small DFB-parameter helper atom is
inlined into a larger atom; the helper takes its DFBs as ttl.DFB
parameters. An inlined callee may also declare its own scratch DFBs when
inlined at the body top level: the decls are hoisted, and the scratch
buffers of sequential sibling callees reuse one CB index. A callee that
declares buffers but is inlined inside a for/if is rejected."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1


@ttl.atom()
def _exp_block(inp: ttl.DFB, out: ttl.DFB):
    """Per-tile exp; declares no DFBs of its own (takes them as params)."""
    x = inp.wait()
    r = out.reserve()
    r.store(ttl.exp(x))


@ttl.atom(grid=(1, 1))
def atom_outer_exp(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    a_blk = a_cb.reserve()
    ttl.copy(in_t[0:1, 0:1], a_blk)

    _exp_block(a_cb, out_cb)  # inlined at decoration time

    out_done = out_cb.wait()
    ttl.copy(out_done, out_t[0:1, 0:1])


@ttl.atom()
def _exp_via_scratch(in_cb: ttl.DFB, out_t):
    """Inlined helper that declares its own scratch DFB: compute writes
    exp(x) into the scratch, data movement drains it to ``out_t``."""
    scratch = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
    x = in_cb.wait()
    s = scratch.reserve()
    s.store(ttl.exp(x))
    done = scratch.wait()
    ttl.copy(done, out_t[0:1, 0:1])


@ttl.atom(grid=(1, 1))
def atom_two_scratch(in0, in1, out0, out1):
    """Inlines two sibling scratch-declaring helpers; their scratch DFBs run
    sequentially, so they share one CB index."""
    a0 = ttl.make_dataflow_buffer_like(in0, shape=(1, 1), block_count=2)
    a1 = ttl.make_dataflow_buffer_like(in1, shape=(1, 1), block_count=2)

    b0 = a0.reserve()
    ttl.copy(in0[0:1, 0:1], b0)
    b1 = a1.reserve()
    ttl.copy(in1[0:1, 0:1], b1)

    _exp_via_scratch(a0, out0)
    _exp_via_scratch(a1, out1)


@ttl.atom()
def _scratch_dm_then_compute(in_t, out_t):
    """Two scratch DFBs whose textual spans are disjoint but whose runtime
    lifetimes overlap across threads: ``a`` is data-movement-produced (copied
    from a tensor) and compute-consumed; ``b`` is compute-produced and
    data-movement-consumed. A statement-span reuse would merge a and b (their
    text spans don't overlap), giving one CB two producers on two threads and
    interleaving the streams. Site-based reuse keeps them distinct."""
    a = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
    b = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)

    ad = a.reserve()
    ttl.copy(in_t[0:1, 0:1], ad)  # data movement: in_t -> a
    av = a.wait()  # compute consumes a
    bd = b.reserve()
    bd.store(ttl.exp(av))  # compute produces b
    bv = b.wait()
    ttl.copy(bv, out_t[0:1, 0:1])  # data movement: b -> out_t


@ttl.atom(grid=(1, 1))
def atom_cross_thread_scratch(in_t, out_t):
    _scratch_dm_then_compute(in_t, out_t)


def test_dfb_callee_in_loop_rejected():
    """A callee that declares its own DFB cannot be inlined inside a loop:
    its decl could not be hoisted to the body top level."""

    @ttl.atom()
    def _declares_dfb(inp: ttl.DFB, out: ttl.DFB):
        scratch = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        x = inp.wait()
        s = scratch.reserve()
        s.store(x)
        done = scratch.wait()
        r = out.reserve()
        r.store(done)

    with pytest.raises(ValueError, match="atom body top level"):

        @ttl.atom(grid=(1, 1))
        def _outer_bad(in_t, out_t):
            a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
            out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)
            for _ in range(2):
                _declares_dfb(a_cb, out_cb)


def _dfbs(n):
    """n fresh bf16 (1,1) double-buffered DFBs with sequential CB indices."""
    from ttl.dataflow_buffer import DataflowBuffer, _reset_cb_counter

    _reset_cb_counter()
    return [DataflowBuffer(None, (1, 1), 2, dtype="bf16") for _ in range(n)]


def test_reuse_overlays_sibling_sites():
    """Scratch from two different inline sites overlays onto one CB index;
    bridge DFBs keep distinct indices below the overlay."""
    from ttl.atom import _reuse_inlined_dfb_indices

    a0, a1, s0, s1 = _dfbs(4)
    dfbs = {"a0": a0, "a1": a1, "s0": s0, "s1": s1}
    # s0 from site 0, s1 from site 1 (distinct sibling sites).
    _reuse_inlined_dfb_indices(dfbs, {"s0": 0, "s1": 1})

    assert s0._cb_index == s1._cb_index
    assert {a0._cb_index, a1._cb_index} == {0, 1}
    assert len({a0._cb_index, a1._cb_index, s0._cb_index}) == 3


def test_reuse_keeps_same_site_distinct():
    """Two scratch DFBs from the SAME inline site stay distinct, even though
    their textual lifetimes are disjoint. This is the case a statement-span
    analysis would wrongly merge: across the compute/data-movement split the
    two buffers can be live concurrently (e.g. one DM-produced, one
    compute-produced), so sharing a CB index would interleave their streams."""
    from ttl.atom import _reuse_inlined_dfb_indices

    s0, s1 = _dfbs(2)
    dfbs = {"s0": s0, "s1": s1}
    _reuse_inlined_dfb_indices(dfbs, {"s0": 0, "s1": 0})

    assert s0._cb_index != s1._cb_index


def test_atom_outer_exp(device):
    tile = ttnn.TILE_SIZE
    inp_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)
    expected = torch.exp(inp_t.float()).to(torch.bfloat16)

    in_t = to_l1(inp_t, device)
    out_t = to_l1(torch.zeros(tile, tile, dtype=torch.bfloat16), device)

    atom_outer_exp(in_t, out_t)

    got = ttnn.to_torch(out_t).reshape(tile, tile).to(torch.bfloat16)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_atom_two_scratch(device):
    tile = ttnn.TILE_SIZE

    def rand():
        return (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)

    in0_t, in1_t = rand(), rand()
    in0 = to_l1(in0_t, device)
    in1 = to_l1(in1_t, device)
    out0 = to_l1(torch.zeros(tile, tile, dtype=torch.bfloat16), device)
    out1 = to_l1(torch.zeros(tile, tile, dtype=torch.bfloat16), device)

    atom_two_scratch(in0, in1, out0, out1)

    got0 = ttnn.to_torch(out0).reshape(tile, tile).to(torch.bfloat16)
    got1 = ttnn.to_torch(out1).reshape(tile, tile).to(torch.bfloat16)
    assert_allclose(got0, torch.exp(in0_t.float()).to(torch.bfloat16), rtol=2e-2, atol=2e-2)
    assert_allclose(got1, torch.exp(in1_t.float()).to(torch.bfloat16), rtol=2e-2, atol=2e-2)


def test_atom_cross_thread_scratch(device):
    """The cross-thread hazard runs correctly: the two same-site scratch DFBs
    must not share a CB index, or the DM-produced and compute-produced streams
    would interleave."""
    tile = ttnn.TILE_SIZE
    inp_t = (torch.randn(tile, tile, dtype=torch.bfloat16) * 0.5).clamp(-1.0, 1.0)
    expected = torch.exp(inp_t.float()).to(torch.bfloat16)

    in_t = to_l1(inp_t, device)
    out_t = to_l1(torch.zeros(tile, tile, dtype=torch.bfloat16), device)

    atom_cross_thread_scratch(in_t, out_t)

    got = ttnn.to_torch(out_t).reshape(tile, tile).to(torch.bfloat16)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)
