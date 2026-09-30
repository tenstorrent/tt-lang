# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Local top-k selection, the single-core stage of the metal TopK kernel.

The shape is the ``(1, 1, 32, 64)`` family from
``tests/ttnn/unit_tests/operations/reduce/test_topk.py``: one tile-row of
scores, two tiles wide. ``k`` is 32 so every column of the result tile is a
selected element. The identity index tensor is an input, matching that
test's ``indices_tensor`` argument. The full GLM indexer is a multi-core
pipeline and is outside this kernel.

Golden values and indices come from ``torch.topk`` on the same scores.
``topk_fused_kernel`` places one or two elementwise ops before a stable
top-k, after it, and on both sides. ``topk_scoped_kernel`` places a stable
top-k in the function body, in a loop, and in a second loop nested under
that loop, so each fuse/defuse pair is lowered in a different region.
"""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_l1

ROWS = 32
WIDTH = 64
K = 32


@ttl.operation(grid=(1, 1))
def topk_largest_kernel(scores, indices, out_values, out_indices):
    """Select the 32 largest scores in each row, largest first."""
    scores_dfb = ttl.make_dataflow_buffer_like(scores, shape=(1, 2), block_count=1)
    indices_dfb = ttl.make_dataflow_buffer_like(indices, shape=(1, 2), block_count=1)
    values_dfb = ttl.make_dataflow_buffer_like(out_values, shape=(1, 1), block_count=1)
    out_indices_dfb = ttl.make_dataflow_buffer_like(
        out_indices, shape=(1, 1), block_count=1
    )

    @ttl.compute()
    def select_largest():
        scores_blk = scores_dfb.wait()
        indices_blk = indices_dfb.wait()
        values_blk = values_dfb.reserve()
        indices_out_blk = out_indices_dfb.reserve()
        top_values, top_indices = ttl.math.topk(
            scores_blk, K, indices=indices_blk, stable=True
        )
        values_blk.store(top_values)
        indices_out_blk.store(top_indices)
        scores_blk.pop()
        indices_blk.pop()
        values_blk.push()
        indices_out_blk.push()

    @ttl.datamovement()
    def dm_read():
        scores_blk = scores_dfb.reserve()
        ttl.copy(scores[0:1, 0:2], scores_blk).wait()
        scores_blk.push()
        indices_blk = indices_dfb.reserve()
        ttl.copy(indices[0:1, 0:2], indices_blk).wait()
        indices_blk.push()

    @ttl.datamovement()
    def dm_write():
        values_blk = values_dfb.wait()
        ttl.copy(values_blk, out_values[0, 0]).wait()
        values_blk.pop()
        indices_blk = out_indices_dfb.wait()
        ttl.copy(indices_blk, out_indices[0, 0]).wait()
        indices_blk.pop()


@ttl.operation(grid=(1, 1))
def topk_smallest_kernel(scores, indices, out_values, out_indices):
    """Select the 32 smallest scores in each row, smallest first."""
    scores_dfb = ttl.make_dataflow_buffer_like(scores, shape=(1, 2), block_count=1)
    indices_dfb = ttl.make_dataflow_buffer_like(indices, shape=(1, 2), block_count=1)
    values_dfb = ttl.make_dataflow_buffer_like(out_values, shape=(1, 1), block_count=1)
    out_indices_dfb = ttl.make_dataflow_buffer_like(
        out_indices, shape=(1, 1), block_count=1
    )

    @ttl.compute()
    def select_smallest():
        scores_blk = scores_dfb.wait()
        indices_blk = indices_dfb.wait()
        values_blk = values_dfb.reserve()
        indices_out_blk = out_indices_dfb.reserve()
        top_values, top_indices = ttl.math.topk(
            scores_blk, K, indices=indices_blk, largest=False, stable=True
        )
        values_blk.store(top_values)
        indices_out_blk.store(top_indices)
        scores_blk.pop()
        indices_blk.pop()
        values_blk.push()
        indices_out_blk.push()

    @ttl.datamovement()
    def dm_read():
        scores_blk = scores_dfb.reserve()
        ttl.copy(scores[0:1, 0:2], scores_blk).wait()
        scores_blk.push()
        indices_blk = indices_dfb.reserve()
        ttl.copy(indices[0:1, 0:2], indices_blk).wait()
        indices_blk.push()

    @ttl.datamovement()
    def dm_write():
        values_blk = values_dfb.wait()
        ttl.copy(values_blk, out_values[0, 0]).wait()
        values_blk.pop()
        indices_blk = out_indices_dfb.wait()
        ttl.copy(indices_blk, out_indices[0, 0]).wait()
        indices_blk.pop()


@ttl.operation(grid=(1, 1))
def topk_fused_kernel(
    scores,
    indices,
    before_values,
    before_indices,
    after_values,
    after_indices,
    around_values,
    around_indices,
):
    """Place one or two elementwise ops before, after, and around top-k.

    Before doubles the scores and then selects. After selects the raw scores
    and doubles the selected values. Around negates the scores, selects, and
    doubles the selected values. Indices stay the top-k indices.
    """
    scores_dfb = ttl.make_dataflow_buffer_like(scores, shape=(1, 2), block_count=1)
    indices_dfb = ttl.make_dataflow_buffer_like(indices, shape=(1, 2), block_count=1)
    before_values_dfb = ttl.make_dataflow_buffer_like(
        before_values, shape=(1, 1), block_count=1
    )
    before_indices_dfb = ttl.make_dataflow_buffer_like(
        before_indices, shape=(1, 1), block_count=1
    )
    after_values_dfb = ttl.make_dataflow_buffer_like(
        after_values, shape=(1, 1), block_count=1
    )
    after_indices_dfb = ttl.make_dataflow_buffer_like(
        after_indices, shape=(1, 1), block_count=1
    )
    around_values_dfb = ttl.make_dataflow_buffer_like(
        around_values, shape=(1, 1), block_count=1
    )
    around_indices_dfb = ttl.make_dataflow_buffer_like(
        around_indices, shape=(1, 1), block_count=1
    )

    @ttl.compute()
    def select_placed():
        scores_blk = scores_dfb.wait()
        indices_blk = indices_dfb.wait()

        doubled = scores_blk * 2.0
        before_values_blk = before_values_dfb.reserve()
        before_indices_blk = before_indices_dfb.reserve()
        before_top_values, before_top_indices = ttl.math.topk(
            doubled, K, indices=indices_blk, stable=True
        )
        before_values_blk.store(before_top_values)
        before_indices_blk.store(before_top_indices)
        before_values_blk.push()
        before_indices_blk.push()

        after_values_blk = after_values_dfb.reserve()
        after_indices_blk = after_indices_dfb.reserve()
        after_top_values, after_top_indices = ttl.math.topk(
            scores_blk, K, indices=indices_blk, stable=True
        )
        after_values_blk.store(after_top_values * 2.0)
        after_indices_blk.store(after_top_indices)
        after_values_blk.push()
        after_indices_blk.push()

        flipped = ttl.math.neg(scores_blk)
        around_values_blk = around_values_dfb.reserve()
        around_indices_blk = around_indices_dfb.reserve()
        around_top_values, around_top_indices = ttl.math.topk(
            flipped, K, indices=indices_blk, stable=True
        )
        around_values_blk.store(around_top_values * 2.0)
        around_indices_blk.store(around_top_indices)
        around_values_blk.push()
        around_indices_blk.push()

        scores_blk.pop()
        indices_blk.pop()

    @ttl.datamovement()
    def dm_read():
        scores_blk = scores_dfb.reserve()
        ttl.copy(scores[0:1, 0:2], scores_blk).wait()
        scores_blk.push()
        indices_blk = indices_dfb.reserve()
        ttl.copy(indices[0:1, 0:2], indices_blk).wait()
        indices_blk.push()

    @ttl.datamovement()
    def dm_write():
        before_values_blk = before_values_dfb.wait()
        ttl.copy(before_values_blk, before_values[0, 0]).wait()
        before_values_blk.pop()
        before_indices_blk = before_indices_dfb.wait()
        ttl.copy(before_indices_blk, before_indices[0, 0]).wait()
        before_indices_blk.pop()
        after_values_blk = after_values_dfb.wait()
        ttl.copy(after_values_blk, after_values[0, 0]).wait()
        after_values_blk.pop()
        after_indices_blk = after_indices_dfb.wait()
        ttl.copy(after_indices_blk, after_indices[0, 0]).wait()
        after_indices_blk.pop()
        around_values_blk = around_values_dfb.wait()
        ttl.copy(around_values_blk, around_values[0, 0]).wait()
        around_values_blk.pop()
        around_indices_blk = around_indices_dfb.wait()
        ttl.copy(around_indices_blk, around_indices[0, 0]).wait()
        around_indices_blk.pop()


@ttl.operation(grid=(1, 1))
def topk_scoped_kernel(
    scores,
    bias,
    indices,
    outer_values,
    outer_indices,
    loop_values,
    loop_indices,
    inner_abs_values,
    inner_abs_indices,
    inner_neg_values,
    inner_neg_indices,
):
    """Select top-k from three nested regions.

    The function body selects the raw scores. The outer loop selects the
    doubled scores on its first iteration. The inner loop, reached on the
    second iteration, selects ``abs(scores + bias)`` and then ``-scores``.
    Each selection is stable, so each region lowers its own fuse/defuse pair.
    """
    scores_dfb = ttl.make_dataflow_buffer_like(scores, shape=(1, 2), block_count=1)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 2), block_count=1)
    indices_dfb = ttl.make_dataflow_buffer_like(indices, shape=(1, 2), block_count=1)
    outer_values_dfb = ttl.make_dataflow_buffer_like(
        outer_values, shape=(1, 1), block_count=1
    )
    outer_indices_dfb = ttl.make_dataflow_buffer_like(
        outer_indices, shape=(1, 1), block_count=1
    )
    loop_values_dfb = ttl.make_dataflow_buffer_like(
        loop_values, shape=(1, 1), block_count=1
    )
    loop_indices_dfb = ttl.make_dataflow_buffer_like(
        loop_indices, shape=(1, 1), block_count=1
    )
    inner_abs_values_dfb = ttl.make_dataflow_buffer_like(
        inner_abs_values, shape=(1, 1), block_count=1
    )
    inner_abs_indices_dfb = ttl.make_dataflow_buffer_like(
        inner_abs_indices, shape=(1, 1), block_count=1
    )
    inner_neg_values_dfb = ttl.make_dataflow_buffer_like(
        inner_neg_values, shape=(1, 1), block_count=1
    )
    inner_neg_indices_dfb = ttl.make_dataflow_buffer_like(
        inner_neg_indices, shape=(1, 1), block_count=1
    )

    @ttl.compute()
    def select_scoped():
        scores_blk = scores_dfb.wait()
        bias_blk = bias_dfb.wait()
        indices_blk = indices_dfb.wait()

        outer_values_blk = outer_values_dfb.reserve()
        outer_indices_blk = outer_indices_dfb.reserve()
        outer_top_values, outer_top_indices = ttl.math.topk(
            scores_blk, K, indices=indices_blk, stable=True
        )
        outer_values_blk.store(outer_top_values)
        outer_indices_blk.store(outer_top_indices)
        outer_values_blk.push()
        outer_indices_blk.push()

        for step in range(2):
            if step == 0:
                scaled = scores_blk * 2.0
                loop_values_blk = loop_values_dfb.reserve()
                loop_indices_blk = loop_indices_dfb.reserve()
                loop_top_values, loop_top_indices = ttl.math.topk(
                    scaled, K, indices=indices_blk, stable=True
                )
                loop_values_blk.store(loop_top_values)
                loop_indices_blk.store(loop_top_indices)
                loop_values_blk.push()
                loop_indices_blk.push()
            else:
                for inner in range(2):
                    if inner == 0:
                        shifted = ttl.math.abs(scores_blk + bias_blk)
                        abs_values_blk = inner_abs_values_dfb.reserve()
                        abs_indices_blk = inner_abs_indices_dfb.reserve()
                        abs_top_values, abs_top_indices = ttl.math.topk(
                            shifted, K, indices=indices_blk, stable=True
                        )
                        abs_values_blk.store(abs_top_values)
                        abs_indices_blk.store(abs_top_indices)
                        abs_values_blk.push()
                        abs_indices_blk.push()
                    else:
                        flipped = ttl.math.neg(scores_blk)
                        neg_values_blk = inner_neg_values_dfb.reserve()
                        neg_indices_blk = inner_neg_indices_dfb.reserve()
                        neg_top_values, neg_top_indices = ttl.math.topk(
                            flipped, K, indices=indices_blk, stable=True
                        )
                        neg_values_blk.store(neg_top_values)
                        neg_indices_blk.store(neg_top_indices)
                        neg_values_blk.push()
                        neg_indices_blk.push()

        scores_blk.pop()
        bias_blk.pop()
        indices_blk.pop()

    @ttl.datamovement()
    def dm_read():
        scores_blk = scores_dfb.reserve()
        ttl.copy(scores[0:1, 0:2], scores_blk).wait()
        scores_blk.push()
        bias_blk = bias_dfb.reserve()
        ttl.copy(bias[0:1, 0:2], bias_blk).wait()
        bias_blk.push()
        indices_blk = indices_dfb.reserve()
        ttl.copy(indices[0:1, 0:2], indices_blk).wait()
        indices_blk.push()

    @ttl.datamovement()
    def dm_write():
        outer_values_blk = outer_values_dfb.wait()
        ttl.copy(outer_values_blk, outer_values[0, 0]).wait()
        outer_values_blk.pop()
        outer_indices_blk = outer_indices_dfb.wait()
        ttl.copy(outer_indices_blk, outer_indices[0, 0]).wait()
        outer_indices_blk.pop()
        loop_values_blk = loop_values_dfb.wait()
        ttl.copy(loop_values_blk, loop_values[0, 0]).wait()
        loop_values_blk.pop()
        loop_indices_blk = loop_indices_dfb.wait()
        ttl.copy(loop_indices_blk, loop_indices[0, 0]).wait()
        loop_indices_blk.pop()
        abs_values_blk = inner_abs_values_dfb.wait()
        ttl.copy(abs_values_blk, inner_abs_values[0, 0]).wait()
        abs_values_blk.pop()
        abs_indices_blk = inner_abs_indices_dfb.wait()
        ttl.copy(abs_indices_blk, inner_abs_indices[0, 0]).wait()
        abs_indices_blk.pop()
        neg_values_blk = inner_neg_values_dfb.wait()
        ttl.copy(neg_values_blk, inner_neg_values[0, 0]).wait()
        neg_values_blk.pop()
        neg_indices_blk = inner_neg_indices_dfb.wait()
        ttl.copy(neg_indices_blk, inner_neg_indices[0, 0]).wait()
        neg_indices_blk.pop()


def make_topk_wide_kernel(width_tiles, k, largest):
    """Select ``k`` scores from a ``width_tiles``-tile row.

    Widths above two tiles exercise the merge and rebuild iterations that
    the two-tile kernels above never reach. ``k`` below 32 exercises the
    tile-wide network with a sliced result.
    """
    output_tiles = (k + 31) // 32

    @ttl.operation(grid=(1, 1))
    def topk_wide_kernel(scores, indices, out_values, out_indices):
        scores_dfb = ttl.make_dataflow_buffer_like(
            scores, shape=(1, width_tiles), block_count=1
        )
        indices_dfb = ttl.make_dataflow_buffer_like(
            indices, shape=(1, width_tiles), block_count=1
        )
        values_dfb = ttl.make_dataflow_buffer_like(
            out_values, shape=(1, output_tiles), block_count=1
        )
        out_indices_dfb = ttl.make_dataflow_buffer_like(
            out_indices, shape=(1, output_tiles), block_count=1
        )

        @ttl.compute()
        def select_wide():
            scores_blk = scores_dfb.wait()
            indices_blk = indices_dfb.wait()
            values_blk = values_dfb.reserve()
            indices_out_blk = out_indices_dfb.reserve()
            top_values, top_indices = ttl.math.topk(
                scores_blk, k, indices=indices_blk, largest=largest, stable=True
            )
            values_blk.store(top_values)
            indices_out_blk.store(top_indices)
            scores_blk.pop()
            indices_blk.pop()
            values_blk.push()
            indices_out_blk.push()

        @ttl.datamovement()
        def dm_read():
            scores_blk = scores_dfb.reserve()
            ttl.copy(scores[0:1, 0:width_tiles], scores_blk).wait()
            scores_blk.push()
            indices_blk = indices_dfb.reserve()
            ttl.copy(indices[0:1, 0:width_tiles], indices_blk).wait()
            indices_blk.push()

        @ttl.datamovement()
        def dm_write():
            values_blk = values_dfb.wait()
            ttl.copy(values_blk, out_values[0:1, 0:output_tiles]).wait()
            values_blk.pop()
            indices_blk = out_indices_dfb.wait()
            ttl.copy(indices_blk, out_indices[0:1, 0:output_tiles]).wait()
            indices_blk.pop()

    return topk_wide_kernel


def _scores(width=WIDTH):
    """32 rows of a permutation of the bf16-exact integers [-width/2, width/2)."""
    base = torch.arange(-(width // 2), width // 2, dtype=torch.float32)
    generator = torch.Generator().manual_seed(2005)
    rows = [base[torch.randperm(width, generator=generator)] for _ in range(ROWS)]
    return torch.stack(rows).to(torch.bfloat16)


def _bias():
    return torch.full((ROWS, WIDTH), 40, dtype=torch.bfloat16)


def _identity_indices(width=WIDTH):
    columns = torch.arange(width, dtype=torch.int64)
    return columns.unsqueeze(0).expand(ROWS, width).to(torch.uint16).contiguous()


def _run(kernel, scores, indices, device, bias=None):
    scores_dev = to_l1(scores, device)
    indices_dev = to_l1(indices, device)
    out_values = to_l1(torch.zeros((ROWS, K), dtype=torch.bfloat16), device)
    out_indices = to_l1(torch.zeros((ROWS, K), dtype=torch.uint16), device)
    if bias is None:
        kernel(scores_dev, indices_dev, out_values, out_indices)
    else:
        kernel(scores_dev, to_l1(bias, device), indices_dev, out_values, out_indices)
    got_values = ttnn.to_torch(out_values).reshape(ROWS, K).to(torch.bfloat16)
    got_indices = ttnn.to_torch(out_indices).reshape(ROWS, K).to(torch.int64)
    return got_values, got_indices


def _assert_matches_torch(scores, got_values, got_indices, largest, k=K):
    expected_values, expected_indices = torch.topk(
        scores.float(), k, dim=-1, largest=largest, sorted=True
    )
    expected_values = expected_values.to(torch.bfloat16)
    assert torch.equal(got_values, expected_values)
    assert torch.equal(got_indices, expected_indices.to(torch.int64))
    gathered = torch.gather(scores.float(), -1, got_indices)
    assert torch.equal(gathered.to(torch.bfloat16), got_values)


def test_topk_largest_matches_torch(device):
    scores = _scores()
    got_values, got_indices = _run(
        topk_largest_kernel, scores, _identity_indices(), device
    )
    _assert_matches_torch(scores, got_values, got_indices, largest=True)


def _assert_scaled_topk(scores, got_values, got_indices, scale):
    expected_values, expected_indices = torch.topk(
        scores.float(), K, dim=-1, largest=True, sorted=True
    )
    expected_values = (expected_values * scale).to(torch.bfloat16)
    assert torch.equal(got_indices, expected_indices.to(torch.int64))
    assert torch.equal(got_values, expected_values)
    gathered = torch.gather(scores.float(), -1, got_indices) * scale
    assert torch.equal(gathered.to(torch.bfloat16), got_values)


def test_topk_before_after_and_around_matches_torch(device):
    scores = _scores()
    indices = _identity_indices()

    def device_pair(dtype):
        return to_l1(torch.zeros((ROWS, K), dtype=dtype), device)

    before_values, before_indices = device_pair(torch.bfloat16), device_pair(
        torch.uint16
    )
    after_values, after_indices = device_pair(torch.bfloat16), device_pair(torch.uint16)
    around_values, around_indices = device_pair(torch.bfloat16), device_pair(
        torch.uint16
    )
    topk_fused_kernel(
        to_l1(scores, device),
        to_l1(indices, device),
        before_values,
        before_indices,
        after_values,
        after_indices,
        around_values,
        around_indices,
    )

    def from_device(values_dev, indices_dev):
        got_values = ttnn.to_torch(values_dev).reshape(ROWS, K).to(torch.bfloat16)
        got_indices = ttnn.to_torch(indices_dev).reshape(ROWS, K).to(torch.int64)
        return got_values, got_indices

    doubled = (scores.float() * 2).to(torch.bfloat16)
    _assert_matches_torch(
        doubled, *from_device(before_values, before_indices), largest=True
    )
    _assert_scaled_topk(scores, *from_device(after_values, after_indices), scale=2)
    flipped = (-scores.float()).to(torch.bfloat16)
    _assert_scaled_topk(flipped, *from_device(around_values, around_indices), scale=2)


def test_topk_nested_scopes_match_torch(device):
    scores = _scores()
    bias = _bias()
    indices = _identity_indices()

    def device_pair(dtype):
        return to_l1(torch.zeros((ROWS, K), dtype=dtype), device)

    outer_values, outer_indices = device_pair(torch.bfloat16), device_pair(torch.uint16)
    loop_values, loop_indices = device_pair(torch.bfloat16), device_pair(torch.uint16)
    abs_values, abs_indices = device_pair(torch.bfloat16), device_pair(torch.uint16)
    neg_values, neg_indices = device_pair(torch.bfloat16), device_pair(torch.uint16)
    topk_scoped_kernel(
        to_l1(scores, device),
        to_l1(bias, device),
        to_l1(indices, device),
        outer_values,
        outer_indices,
        loop_values,
        loop_indices,
        abs_values,
        abs_indices,
        neg_values,
        neg_indices,
    )

    expected = [
        scores,
        (scores.float() * 2).to(torch.bfloat16),
        (scores.float() + bias.float()).abs().to(torch.bfloat16),
        (-scores.float()).to(torch.bfloat16),
    ]
    got = [
        (outer_values, outer_indices),
        (loop_values, loop_indices),
        (abs_values, abs_indices),
        (neg_values, neg_indices),
    ]
    for expected_scores, (values_dev, indices_dev) in zip(expected, got):
        got_values = ttnn.to_torch(values_dev).reshape(ROWS, K).to(torch.bfloat16)
        got_indices = ttnn.to_torch(indices_dev).reshape(ROWS, K).to(torch.int64)
        _assert_matches_torch(expected_scores, got_values, got_indices, largest=True)


def test_topk_smallest_matches_torch(device):
    scores = _scores()
    got_values, got_indices = _run(
        topk_smallest_kernel, scores, _identity_indices(), device
    )
    _assert_matches_torch(scores, got_values, got_indices, largest=False)


@pytest.mark.parametrize(
    "width_tiles, k, largest",
    [
        (4, 32, True),
        (8, 32, True),
        (8, 32, False),
        (8, 64, True),
        (4, 8, True),
        (2, 16, False),
    ],
)
def test_topk_wide_rows_match_torch(device, width_tiles, k, largest):
    width = width_tiles * 32
    scores = _scores(width)
    indices = _identity_indices(width)
    scores_dev = to_l1(scores, device)
    indices_dev = to_l1(indices, device)
    out_values = to_l1(torch.zeros((ROWS, k), dtype=torch.bfloat16), device)
    out_indices = to_l1(torch.zeros((ROWS, k), dtype=torch.uint16), device)
    make_topk_wide_kernel(width_tiles, k, largest)(
        scores_dev, indices_dev, out_values, out_indices
    )
    got_values = ttnn.to_torch(out_values).reshape(ROWS, k).to(torch.bfloat16)
    got_indices = ttnn.to_torch(out_indices).reshape(ROWS, k).to(torch.int64)
    _assert_matches_torch(scores, got_values, got_indices, largest=largest, k=k)
