# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the per-core specialization path.

Per-core specialization is opt-in via the compiler option `specialize_cores`
(enable with --ttl-specialize-cores; disabled by default). It is a single
module pass (ttkernel-specialize-cores) run at the TTKernel level right before
EmitC: for each kernel that branches on a core coordinate it emits one clone
per launch coordinate, replacing the coordinate reads with constants and
tagging each clone with ttl.core_coord. The runtime bridge
(_compile_ttnn_kernel) turns each ttl.core_coord into a per-kernel core
range for dispatch.

The matmul kernel below addresses its operands through the coordinate but never
branches on it, so specialization skips cloning for it. That test therefore
verifies that:
  * The specialized result matches a torch reference (numerical correctness).
  * The specialized result matches the default (unspecialized) path.
  * The dumped final MLIR shows these data-addressing kernels are NOT cloned
    (no `ttl.core_coord`), i.e. specialization safely declines to clone them.

A second kernel (branch_swap_*) *does* branch on the coordinate: its reader
selects a different source column depending on core_x, so specialization
clones the reader per core and const-folds the branch. That test exercises the
full clone-and-dispatch path end to end (the MLIR-only clone/fold checks live in
the lit test test/ttlang/Dialect/TTKernel/Transforms/specialize_cores.mlir).
"""

import os
import re

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import torch

import ttl
from ttlang_test_utils import assert_pcc, to_dram

TILE_SIZE = 32

# Small grid keeps the clone count (3 kernels * GRID_X * GRID_Y) modest.
GRID_X = 2
GRID_Y = 2


def _assert_not_cloned(final_mlir_path):
    """Confirm the dumped final MLIR shows no per-core clones.

    The kernels here use the coordinate only for addressing, never for a branch,
    so specialization must decline to clone them: no `ttl.core_coord` clone tag
    and no `_c<x>_<y>` clone suffixes should appear.
    """
    with open(final_mlir_path) as fd:
        mlir = fd.read()
    assert (
        "ttl.core_coord" not in mlir
    ), "data-addressing kernels must not be cloned (found ttl.core_coord)"
    clones = re.findall(r"func\.func @\w+_c\d+_\d+", mlir)
    assert not clones, f"expected no per-core clones, got {clones}"


# The motivating case from the specialization epic: a 2D grid matmul where
# core (x, y) computes output tile out[y, x] = a[y, 0] @ b[0, x]. The reader
# and writer address a, b, and out through their ttl.node coordinate, so
# specialization must const-fold each clone's core_x / core_y to the right
# tile. A single K tile per core is used because multicore K-accumulation is
# not supported yet (see the matmul_multinode TODO and issue #652).
@ttl.operation(grid=(GRID_X, GRID_Y))
def matmul_default(a, b, out):
    """Per-core single-tile matmul, compiled through the normal pipeline."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def mm_compute():
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        with out_dfb.reserve() as o:
            o.store(a_blk @ b_blk)
        a_blk.pop()
        b_blk.pop()

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        with a_dfb.reserve() as a_blk:
            tx = ttl.copy(a[y, 0], a_blk)
            tx.wait()
        with b_dfb.reserve() as b_blk:
            tx = ttl.copy(b[0, x], b_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        with out_dfb.wait() as o:
            tx = ttl.copy(o, out[y, x])
            tx.wait()


# Separate op object so its compilation cache is independent from the default
# matmul. This one is always invoked with options="--ttl-specialize-cores".
@ttl.operation(grid=(GRID_X, GRID_Y))
def matmul_specialized(a, b, out):
    """Identical body to matmul_default; invoked with specialization on."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def mm_compute():
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        with out_dfb.reserve() as o:
            o.store(a_blk @ b_blk)
        a_blk.pop()
        b_blk.pop()

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        with a_dfb.reserve() as a_blk:
            tx = ttl.copy(a[y, 0], a_blk)
            tx.wait()
        with b_dfb.reserve() as b_blk:
            tx = ttl.copy(b[0, x], b_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        with out_dfb.wait() as o:
            tx = ttl.copy(o, out[y, x])
            tx.wait()


def _make_matmul_inputs(device):
    M = GRID_Y * TILE_SIZE
    N = GRID_X * TILE_SIZE
    K = TILE_SIZE  # single K tile per core
    a_torch = torch.randn((M, K), dtype=torch.bfloat16)
    b_torch = torch.randn((K, N), dtype=torch.bfloat16)
    expected = (a_torch.float() @ b_torch.float()).to(torch.bfloat16)
    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    return a, b, expected


def test_specialize_cores_matmul_matches_reference(device, monkeypatch, tmp_path):
    """Specialized per-core matmul matches torch and the unspecialized path."""
    a, b, expected = _make_matmul_inputs(device)
    out_default = to_dram(torch.zeros_like(expected), device)
    out_spec = to_dram(torch.zeros_like(expected), device)

    # Unspecialized baseline (specialization off by default).
    matmul_default(a, b, out_default)
    default_result = ttnn.to_torch(out_default)

    # Specialized path (opt in). Dump the final MLIR so we can confirm the
    # pass ran.
    final_mlir = tmp_path / "matmul_specialized_final.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))
    matmul_specialized(a, b, out_spec, options="--ttl-specialize-cores")
    spec_result = ttnn.to_torch(out_spec)

    # Numerical correctness vs torch, and equivalence to the default path.
    assert_pcc(expected, spec_result, threshold=0.999)
    assert_pcc(default_result, spec_result, threshold=0.999)

    # These kernels address through the coordinate but never branch on it, so
    # specialization must leave them un-cloned.
    _assert_not_cloned(str(final_mlir))


# A kernel that actually branches on the core coordinate. The reader swaps the
# two tile-columns: core (x, y) reads source column `1 - x`, so the output is
# the input with its two column-blocks swapped. The branch (`if x == 0`) lowers
# to an scf.if on the coordinate, so specialization clones the reader per core
# and const-folds each clone's branch. The compute and writer never branch, so
# they stay whole-grid -- this also exercises the mixed cloned/uncloned kernel
# validation path. The column-swap assumes GRID_X == 2.
@ttl.operation(grid=(GRID_X, GRID_Y))
def branch_swap_default(a, out):
    """Reader branches on core_x to swap columns; normal (unspecialized) path."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with a_dfb.wait() as a_tile, out_dfb.reserve() as o:
            o.store(a_tile)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        with a_dfb.reserve() as blk:
            # Default (x == 1) reads column 0; column-0 cores read column 1.
            tx = ttl.copy(a[y, 0], blk)
            if x == 0:
                tx = ttl.copy(a[y, 1], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[y, x])
            tx.wait()


# Separate op object so its compilation cache is independent from the default
# branch kernel. This one is always invoked with options="--ttl-specialize-cores".
@ttl.operation(grid=(GRID_X, GRID_Y))
def branch_swap_specialized(a, out):
    """Identical body to branch_swap_default; invoked with specialization on."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with a_dfb.wait() as a_tile, out_dfb.reserve() as o:
            o.store(a_tile)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[y, 0], blk)
            if x == 0:
                tx = ttl.copy(a[y, 1], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[y, x])
            tx.wait()


def _make_swap_inputs(device):
    assert GRID_X == 2, "branch_swap column-swap reference assumes GRID_X == 2"
    shape = (GRID_Y * TILE_SIZE, GRID_X * TILE_SIZE)
    a_torch = torch.randn(shape, dtype=torch.bfloat16)
    left = a_torch[:, :TILE_SIZE]
    right = a_torch[:, TILE_SIZE:]
    expected = torch.cat([right, left], dim=1).contiguous()
    a = to_dram(a_torch, device)
    return a, expected


def _assert_reader_cloned(final_mlir_path):
    """Confirm the branching reader was cloned once per core.

    Only the reader branches on the coordinate, so there must be exactly one
    clone per launch coordinate (compute and writer stay whole-grid), each
    tagged with `ttl.core_coord`, and the clones must cover every coordinate.
    """
    with open(final_mlir_path) as fd:
        mlir = fd.read()
    assert (
        "ttl.core_coord" in mlir
    ), "a kernel that branches on the coordinate must be cloned (no ttl.core_coord)"
    clones = re.findall(r"func\.func @\w+_c(\d+)_(\d+)", mlir)
    coords = {(int(x), int(y)) for x, y in clones}
    expected_coords = {(x, y) for y in range(GRID_Y) for x in range(GRID_X)}
    assert coords == expected_coords, (
        f"reader clones cover {sorted(coords)}, expected " f"{sorted(expected_coords)}"
    )
    assert len(clones) == GRID_X * GRID_Y, (
        f"expected exactly {GRID_X * GRID_Y} clones (reader only), got "
        f"{len(clones)}: {clones}"
    )


def test_specialize_cores_branch_matches_reference(device, monkeypatch, tmp_path):
    """A coordinate-branching kernel is cloned per core and stays correct."""
    a, expected = _make_swap_inputs(device)
    out_default = to_dram(torch.zeros_like(expected), device)
    out_spec = to_dram(torch.zeros_like(expected), device)

    # Unspecialized baseline: the runtime coordinate read + scf.if already
    # produce the swap correctly on the default path.
    branch_swap_default(a, out_default)
    default_result = ttnn.to_torch(out_default)

    # Specialized path (opt in). Dump the final MLIR so we can confirm the
    # reader was cloned per core.
    final_mlir = tmp_path / "branch_specialized_final.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))
    branch_swap_specialized(a, out_spec, options="--ttl-specialize-cores")
    spec_result = ttnn.to_torch(out_spec)

    # Numerical correctness vs torch, and equivalence to the default path.
    assert_pcc(expected, spec_result)
    assert_pcc(default_result, spec_result)

    # The reader branches on core_x, so it must be cloned once per core.
    _assert_reader_cloned(str(final_mlir))


def _make_branch_swap_op():
    """Return a fresh column-swap op with its own compilation cache.

    The cache is keyed on tensor properties, not on env vars like
    TTLANG_EMIT_RUNNER / TTLANG_COMPILE_ONLY, so tests that toggle those must
    use distinct op objects or the cache hit skips the env-dependent codegen.
    """

    @ttl.operation(grid=(GRID_X, GRID_Y))
    def branch_swap(a, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute_fn():
            with a_dfb.wait() as a_tile, out_dfb.reserve() as o:
                o.store(a_tile)

        @ttl.datamovement()
        def dm_read():
            x, y = ttl.node(dims=2)
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[y, 0], blk)
                if x == 0:
                    tx = ttl.copy(a[y, 1], blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            x, y = ttl.node(dims=2)
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[y, x])
                tx.wait()

    return branch_swap


def test_specialize_cores_emit_runner_no_crash(device, monkeypatch, tmp_path):
    """Regression: TTLANG_EMIT_RUNNER must stay clone-aligned.

    The emit block indexed thread_tensor_indices (one per original thread)
    against kernel_paths (one per clone), so it raised IndexError.
    """
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    runner_path = tmp_path / "runner.py"
    monkeypatch.setenv("TTLANG_EMIT_RUNNER", str(runner_path))
    a, _ = _make_swap_inputs(device)
    out = to_dram(
        torch.zeros((GRID_Y * TILE_SIZE, GRID_X * TILE_SIZE), dtype=torch.bfloat16),
        device,
    )
    _make_branch_swap_op()(a, out, options="--ttl-specialize-cores")
    assert runner_path.exists(), "no runner emitted"


@pytest.mark.xfail(
    reason="emitted runner template cannot express per-clone dispatch: it "
    "re-derives reader/writer from a positional counter and passes only the "
    "whole-grid core_ranges, so every clone lands on every core and tt-metal "
    "rejects it with TT_FATAL: Illegal NOC usage. Needs per-kernel core-range "
    "and NOC-role constants in emit_runner_file.",
    strict=True,
    raises=RuntimeError,
)
def test_specialize_cores_emit_runner_executes(device, monkeypatch, tmp_path):
    """The emitted runner, run standalone on a cold cache, must reproduce the swap.

    Compile-only so the op never executes and warms the program cache; otherwise
    the runner's custom_program_hash reuses the cached program and masks the
    misdispatch. Strict xfail so it flags when the template is fixed.
    """
    import importlib.util

    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    runner_path = tmp_path / "runner.py"
    monkeypatch.setenv("TTLANG_EMIT_RUNNER", str(runner_path))
    a, expected = _make_swap_inputs(device)
    out = to_dram(torch.zeros_like(expected), device)
    _make_branch_swap_op()(a, out, options="--ttl-specialize-cores")

    spec = importlib.util.spec_from_file_location("emitted_runner", str(runner_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("TTLANG_COMPILE_ONLY", raising=False)
    module.run([a, out], device=device)
    assert_pcc(expected, ttnn.to_torch(out))


if __name__ == "__main__":
    # Manual repro: run the specialized path directly (no pytest) and print a
    # summary plus the specialized-vs-torch comparison.
    from ttlang_test_utils import require_hardware

    print("=== Per-core specialization repro ===")
    require_hardware()

    final_mlir = "/tmp/specialize_cores_final.mlir"
    os.environ["TTLANG_FINAL_MLIR"] = final_mlir

    dev = ttnn.open_device(device_id=0)
    try:
        a, b, expected = _make_matmul_inputs(dev)
        out = to_dram(torch.zeros_like(expected), dev)

        print(
            f"Grid: {GRID_X}x{GRID_Y} "
            f"(these kernels address through the coordinate but never branch on "
            f"it, so specialization is a no-op: no clones expected)"
        )
        matmul_specialized(a, b, out, options="--ttl-specialize-cores")

        result = ttnn.to_torch(out)
        assert_pcc(expected, result, threshold=0.999)
        _assert_not_cloned(final_mlir)
        print(f"OK: specialized result matches torch reference (no clones).")
        print(f"Final MLIR written to {final_mlir}")
    finally:
        ttnn.close_device(dev)
