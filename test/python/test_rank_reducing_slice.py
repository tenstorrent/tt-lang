# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for rank-reducing ttl.tensor_slice / ttl.copy (issue #629).

A slice result (CB block) may be lower rank than the source tensor. The leading
(tensor_rank - cb_rank) dims are squeezed with scalar subscripts; the trailing
dims map to the CB shape. Each squeezed scalar index selects one slot in its
dim and contributes its offset to the per-tile tensor coordinate, so a
(B, N, S, D) tensor can be read into a rank-2 block via ``x_t[b, n, s0:s1, 0:D]``
for any b, n -- not just size-1 leading dims. This lets higher-rank tensors feed
2D compute ops (broadcast / reduce / transpose) without a reshape.

The supported addressing is identical to a same-rank slice: the squeezed coords
are linearized against the full tensor tile grid. The only constraint kept at the
Python layer is that a squeezed position must use scalar (not range) syntax.
"""

import importlib.util
import tempfile
from typing import Callable

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from conftest import temp_kernel_files
from ttl.diagnostics import TTLangCompileError
from ttlang_test_utils import assert_allclose, to_dram

TILE_SIZE = 32


def _t(n: int) -> int:
    """Tile count -> element count."""
    return n * TILE_SIZE


# =============================================================================
# Kernel loading
# =============================================================================

_kernel_cache: dict[tuple, Callable] = {}


def _load_kernel(cache_key, code, prefix):
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=prefix
    ) as f:
        f.write(code)
        temp_path = f.name

    temp_kernel_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("rr_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.kernel
    _kernel_cache[cache_key] = kernel
    return kernel


def _scalars(sel) -> str:
    """Scalar subscripts for squeezed leading dims, e.g. (2, 1) -> '2, 1, '."""
    return "".join(f"{i}, " for i in sel)


# =============================================================================
# 1. Read a full 2D plane out of an N-D tensor, selecting leading slots.
# =============================================================================

# (leading element/tile dims, selected index per leading dim).
PLANE_CASES = [
    ((2,), (1,)),  # 3D, select n=1
    ((4, 2), (2, 1)),  # 4D, select b=2 n=1 -- the core #629 case
    ((1, 1), (0, 0)),  # 4D, size-1 squeeze (original supported case)
    ((3, 1), (2, 0)),  # 4D, mixed: one leading dim > 1, one == 1
    ((2, 2, 2), (1, 0, 1)),  # 5D, select nonzero slots
]


def _plane_id(case):
    leading, sel = case
    return "L" + "x".join(map(str, leading)) + "_sel" + "_".join(map(str, sel))


def _select_plane_code(sel, st, dt):
    lead = _scalars(sel)
    return f"""\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=({st}, {dt}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({st}, {dt}), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = inp_dfb.wait()
        o = out_dfb.reserve()
        o.store(ttl.math.neg(x))
        x.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        blk = inp_dfb.reserve()
        ttl.copy(inp[{lead}0:{st}, 0:{dt}], blk).wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        ob = out_dfb.wait()
        ttl.copy(ob, out[0:{st}, 0:{dt}]).wait()
        ob.pop()
"""


@pytest.mark.parametrize("case", PLANE_CASES, ids=[_plane_id(c) for c in PLANE_CASES])
def test_select_plane_from_nd(device, case):
    """Read one 2D plane from a rank-N tensor by scalar-indexing the leading
    dims, then negate. Verifies the selected sub-block (not just slot 0)."""
    leading, sel = case
    st, dt = 2, 2
    shape = tuple(leading) + (_t(st), _t(dt))

    code = _select_plane_code(sel, st, dt)
    kernel = _load_kernel(("plane", case), code, "kernel_rr_plane_")

    inp_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros((_t(st), _t(dt)), dtype=torch.bfloat16)
    expected = -inp_torch[sel]  # torch scalar-indexes the same leading slots

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# 2. Issue #629 literal example: read a (1, 1) block via x_t[b, n, s, c].
# =============================================================================


def _single_tile_code(b, n, st, dt):
    return f"""\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(x_t, out):
    x_cb = ttl.make_dataflow_buffer_like(x_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        for _s in range({st}):
            for _c in range({dt}):
                x = x_cb.wait()
                o = out_cb.reserve()
                o.store(ttl.math.neg(x))
                x.pop()
                o.push()

    @ttl.datamovement()
    def dm_read():
        b = {b}
        n = {n}
        for s in range({st}):
            for c in range({dt}):
                buf = x_cb.reserve()
                ttl.copy(x_t[b, n, s, c], buf).wait()
                buf.push()

    @ttl.datamovement()
    def dm_write():
        for s in range({st}):
            for c in range({dt}):
                ob = out_cb.wait()
                ttl.copy(ob, out[s, c]).wait()
                ob.pop()
"""


@pytest.mark.parametrize("sel", [(0, 0), (1, 1)], ids=["sel0_0", "sel1_1"])
def test_629_single_tile_read(device, sel):
    """The issue's ``x_t[b, n, s, c]`` pattern: read a 4D tensor one (1, 1) tile
    at a time into a rank-2 block, looping over the trailing tile grid."""
    b, n = sel
    st, dt = 2, 2
    shape = (2, 2, _t(st), _t(dt))

    code = _single_tile_code(b, n, st, dt)
    kernel = _load_kernel(("single_tile", sel), code, "kernel_rr_629_")

    x_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros((_t(st), _t(dt)), dtype=torch.bfloat16)
    expected = -x_torch[b, n]

    x_t = to_dram(x_torch, device)
    out = to_dram(out_torch, device)

    kernel(x_t, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# 3. Read a multi-tile sub-range at a nonzero S offset within a selected plane.
# =============================================================================


def test_select_subrange(device):
    """Read x_t[b, n, s0:s0+chunk, 0:D] -- nonzero batch/head and a nonzero tile
    offset along S -- into a multi-tile block."""
    b, n = 1, 0
    s0, chunk, dt = 1, 2, 2
    st = 4
    shape = (2, 2, _t(st), _t(dt))

    code = f"""\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(x_t, out):
    x_cb = ttl.make_dataflow_buffer_like(x_t, shape=({chunk}, {dt}), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=({chunk}, {dt}), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = x_cb.wait()
        o = out_cb.reserve()
        o.store(ttl.math.neg(x))
        x.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        blk = x_cb.reserve()
        ttl.copy(x_t[{b}, {n}, {s0}:{s0 + chunk}, 0:{dt}], blk).wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        ob = out_cb.wait()
        ttl.copy(ob, out[0:{chunk}, 0:{dt}]).wait()
        ob.pop()
"""
    kernel = _load_kernel(("subrange",), code, "kernel_rr_subrange_")

    x_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros((_t(chunk), _t(dt)), dtype=torch.bfloat16)
    expected = -x_torch[b, n, _t(s0) : _t(s0 + chunk), :]

    x_t = to_dram(x_torch, device)
    out = to_dram(out_torch, device)

    kernel(x_t, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# 4. Two independently selected planes added together (2D compute consumer).
# =============================================================================


def test_select_plane_binary_add(device):
    """Add two 2D planes squeezed from different (b, n) slots of two 4D tensors,
    exercising two rank-reducing reads feeding a single 2D op."""
    sel_a = (0, 1)
    sel_b = (2, 0)
    st, dt = 2, 2
    shape = (3, 2, _t(st), _t(dt))

    code = f"""\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=({st}, {dt}), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=({st}, {dt}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({st}, {dt}), block_count=2)

    @ttl.compute()
    def compute_fn():
        l = lhs_dfb.wait()
        r = rhs_dfb.wait()
        o = out_dfb.reserve()
        o.store(l + r)
        l.pop()
        r.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        lb = lhs_dfb.reserve()
        ttl.copy(lhs[{sel_a[0]}, {sel_a[1]}, 0:{st}, 0:{dt}], lb).wait()
        lb.push()
        rb = rhs_dfb.reserve()
        ttl.copy(rhs[{sel_b[0]}, {sel_b[1]}, 0:{st}, 0:{dt}], rb).wait()
        rb.push()

    @ttl.datamovement()
    def dm_write():
        ob = out_dfb.wait()
        ttl.copy(ob, out[0:{st}, 0:{dt}]).wait()
        ob.pop()
"""
    kernel = _load_kernel(("plane_add",), code, "kernel_rr_planeadd_")

    lhs_torch = torch.rand(shape, dtype=torch.bfloat16)
    rhs_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros((_t(st), _t(dt)), dtype=torch.bfloat16)
    expected = lhs_torch[sel_a] + rhs_torch[sel_b]

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# 5. Iterate every plane: read and write rank reduction across all leading slots.
# =============================================================================


def test_iterate_all_planes(device):
    """Loop over every (b, n) plane of a 4D tensor, reading and writing each via
    rank-reducing slices. Exercises the leading offset for all slots on both the
    read and write paths."""
    b_dim, n_dim = 2, 3
    st, dt = 2, 2
    shape = (b_dim, n_dim, _t(st), _t(dt))

    code = f"""\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=({st}, {dt}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({st}, {dt}), block_count=2)

    @ttl.compute()
    def compute_fn():
        for _ in range({b_dim}):
            for _ in range({n_dim}):
                x = inp_dfb.wait()
                o = out_dfb.reserve()
                o.store(ttl.math.neg(x))
                x.pop()
                o.push()

    @ttl.datamovement()
    def dm_read():
        for b in range({b_dim}):
            for n in range({n_dim}):
                blk = inp_dfb.reserve()
                ttl.copy(inp[b, n, 0:{st}, 0:{dt}], blk).wait()
                blk.push()

    @ttl.datamovement()
    def dm_write():
        for b in range({b_dim}):
            for n in range({n_dim}):
                ob = out_dfb.wait()
                ttl.copy(ob, out[b, n, 0:{st}, 0:{dt}]).wait()
                ob.pop()
"""
    kernel = _load_kernel(("all_planes",), code, "kernel_rr_allplanes_")

    inp_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(shape, dtype=torch.bfloat16)
    expected = -inp_torch

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# 6. Write-side selection: store a 2D block into one plane of a 4D output.
# =============================================================================


def test_write_select_plane(device):
    """Write a 2D block into a specific (b, n) plane of a 4D output; the other
    planes must remain untouched (zero)."""
    b, n = 1, 2
    b_dim, n_dim = 2, 3
    st, dt = 2, 2
    in_shape = (_t(st), _t(dt))
    out_shape = (b_dim, n_dim, _t(st), _t(dt))

    code = f"""\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=({st}, {dt}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({st}, {dt}), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = inp_dfb.wait()
        o = out_dfb.reserve()
        o.store(ttl.math.neg(x))
        x.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        blk = inp_dfb.reserve()
        ttl.copy(inp[0:{st}, 0:{dt}], blk).wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        ob = out_dfb.wait()
        ttl.copy(ob, out[{b}, {n}, 0:{st}, 0:{dt}]).wait()
        ob.pop()
"""
    kernel = _load_kernel(("write_plane",), code, "kernel_rr_writeplane_")

    inp_torch = torch.rand(in_shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(out_shape, dtype=torch.bfloat16)
    expected = torch.zeros(out_shape, dtype=torch.bfloat16)
    expected[b, n] = -inp_torch

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# 7. Streaming chunks along S within a selected plane (cache pattern).
# =============================================================================


def test_chunked_stream_in_plane(device):
    """Stream a selected (b, n) plane into a rank-2 block in S-chunks, mirroring
    a cache indexed natively as 4D while compute stays rank-2."""
    b, n = 1, 1
    st, dt = 4, 2
    chunk = 2
    shape = (2, 2, _t(st), _t(dt))

    code = f"""\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=({chunk}, {dt}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({chunk}, {dt}), block_count=2)

    @ttl.compute()
    def compute_fn():
        for _ in range(0, {st}, {chunk}):
            x = inp_dfb.wait()
            o = out_dfb.reserve()
            o.store(ttl.math.neg(x))
            x.pop()
            o.push()

    @ttl.datamovement()
    def dm_read():
        for kc in range(0, {st}, {chunk}):
            blk = inp_dfb.reserve()
            ttl.copy(inp[{b}, {n}, kc:kc + {chunk}, 0:{dt}], blk).wait()
            blk.push()

    @ttl.datamovement()
    def dm_write():
        for kc in range(0, {st}, {chunk}):
            ob = out_dfb.wait()
            ttl.copy(ob, out[kc:kc + {chunk}, 0:{dt}]).wait()
            ob.pop()
"""
    kernel = _load_kernel(("chunked",), code, "kernel_rr_chunk_")

    inp_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros((_t(st), _t(dt)), dtype=torch.bfloat16)
    expected = -inp_torch[b, n]

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# 8. Partial rank reduction: a 4D tensor read into a 3D DFB.
# =============================================================================


def test_4d_into_3d_dfb(device):
    """Squeeze only the leading dim of a 4D tensor into a rank-3 block: select
    batch b, keep (N, S, D) as the block. Verifies partial (not full-to-2D) rank
    reduction with a multi-dim trailing block."""
    b = 1
    b_dim, n_dim = 2, 2
    st, dt = 2, 2
    shape = (b_dim, n_dim, _t(st), _t(dt))
    out_shape = (n_dim, _t(st), _t(dt))

    code = f"""\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=({n_dim}, {st}, {dt}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({n_dim}, {st}, {dt}), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = inp_dfb.wait()
        o = out_dfb.reserve()
        o.store(ttl.math.neg(x))
        x.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        blk = inp_dfb.reserve()
        ttl.copy(inp[{b}, 0:{n_dim}, 0:{st}, 0:{dt}], blk).wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        ob = out_dfb.wait()
        ttl.copy(ob, out[0:{n_dim}, 0:{st}, 0:{dt}]).wait()
        ob.pop()
"""
    kernel = _load_kernel(("4d_into_3d",), code, "kernel_rr_4d3d_")

    inp_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(out_shape, dtype=torch.bfloat16)
    expected = -inp_torch[b]

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# 9. Edge cases for 1,1 collapse.
# =============================================================================


def test_collapse_to_single_tile(device):
    """Fully collapse (1, 1, 32, 32) -> rank-2 (1, 1) with all-scalar subscripts.
    A single-tile CB does not require range syntax on the trailing dims."""
    shape = (1, 1, TILE_SIZE, TILE_SIZE)
    code = """\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = inp_dfb.wait()
        o = out_dfb.reserve()
        o.store(ttl.math.neg(x))
        x.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        blk = inp_dfb.reserve()
        ttl.copy(inp[0, 0, 0, 0], blk).wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        ob = out_dfb.wait()
        ttl.copy(ob, out[0, 0]).wait()
        ob.pop()
"""
    kernel = _load_kernel(("single_tile_collapse",), code, "kernel_rr_single_")

    inp_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros((TILE_SIZE, TILE_SIZE), dtype=torch.bfloat16)
    expected = -inp_torch[0, 0]

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_collapse_mixed_tile_extents(device):
    """Collapse (1, 1, 32, 64) -> (1, 2): squeezed leading dims, a single-tile
    range (0:1) on S, and multi-tile D. Multi-tile CBs require range syntax even
    for the size-1 trailing dim."""
    st, dt = 1, 2
    shape = (1, 1, _t(st), _t(dt))
    code = """\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 2), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 2), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = inp_dfb.wait()
        o = out_dfb.reserve()
        o.store(ttl.math.neg(x))
        x.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        blk = inp_dfb.reserve()
        ttl.copy(inp[0, 0, 0:1, 0:2], blk).wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        ob = out_dfb.wait()
        ttl.copy(ob, out[0:1, 0:2]).wait()
        ob.pop()
"""
    kernel = _load_kernel(("mixed_tile",), code, "kernel_rr_mixed_")

    inp_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros((_t(st), _t(dt)), dtype=torch.bfloat16)
    expected = (-inp_torch).reshape(_t(st), _t(dt))

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# 10. Negative: a squeezed position must be a scalar, not a range.
# =============================================================================


def test_squeeze_index_must_be_scalar(device):
    """A range subscript in a squeezed leading position is rejected: a range does
    not reduce rank, so it is ambiguous against a lower-rank CB."""
    st, dt = 2, 2
    shape = (1, 1, _t(st), _t(dt))
    code = f"""\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=({st}, {dt}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({st}, {dt}), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = inp_dfb.wait()
        o = out_dfb.reserve()
        o.store(ttl.math.neg(x))
        x.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        blk = inp_dfb.reserve()
        ttl.copy(inp[0:1, 0, 0:{st}, 0:{dt}], blk).wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        ob = out_dfb.wait()
        ttl.copy(ob, out[0:{st}, 0:{dt}]).wait()
        ob.pop()
"""
    kernel = _load_kernel(("bad_squeeze",), code, "kernel_rr_badsqueeze_")

    inp = to_dram(torch.rand(shape, dtype=torch.bfloat16), device)
    out = to_dram(torch.zeros((_t(st), _t(dt)), dtype=torch.bfloat16), device)

    with pytest.raises(TTLangCompileError, match="scalar"):
        kernel(inp, out)


def test_3d_range_into_2d_dfb_rejected(device):
    """Reading a 3D block into a 2D DFB is rejected when the batch dim uses a
    range: that dim does not collapse to a scalar, so it cannot be squeezed to
    fit the rank-2 block."""
    n_dim = 2
    st, dt = 2, 2
    shape = (n_dim, _t(st), _t(dt))
    code = f"""\
import ttl

@ttl.operation(grid=(1, 1))
def kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=({st}, {dt}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({st}, {dt}), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = inp_dfb.wait()
        o = out_dfb.reserve()
        o.store(ttl.math.neg(x))
        x.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        blk = inp_dfb.reserve()
        ttl.copy(inp[0:{n_dim}, 0:{st}, 0:{dt}], blk).wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        ob = out_dfb.wait()
        ttl.copy(ob, out[0:{st}, 0:{dt}]).wait()
        ob.pop()
"""
    kernel = _load_kernel(("3d_range_2d",), code, "kernel_rr_3d2d_")

    inp = to_dram(torch.rand(shape, dtype=torch.bfloat16), device)
    out = to_dram(torch.zeros((_t(st), _t(dt)), dtype=torch.bfloat16), device)

    with pytest.raises(TTLangCompileError, match="scalar"):
        kernel(inp, out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
