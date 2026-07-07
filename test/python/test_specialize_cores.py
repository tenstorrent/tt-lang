# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the per-core specialization path.

`ttl-specialize-cores` is exposed as the opt-in compiler option
`specialize_cores` (enable with `--ttl-specialize-cores`; disabled by default
because it is not yet compatible with PipeNet collectives or L1 accumulation).
That pass specializes each kernel function for its launch grid: it tags every
clone with `ttl.core_coord` (the coordinates it serves) and const-folds
`ttl.core_x` / `ttl.core_y` to a representative coordinate. The runtime bridge
(`_compile_ttnn_kernel`) turns each `ttl.core_coord` into a
`KernelSpec.core_ranges` so every specialized binary is dispatched only to the
cores it serves.

To avoid emitting one near-identical binary per core, the pass groups
coordinates with identical control flow (via `LaunchNodeDomainAnalysis`) into a
single clone. The compute kernel here never touches the coordinate, so it
de-duplicates to a single whole-grid clone; the data-movement kernels address
`a`, `b`, and `out` through their `ttl.node` coordinate, so they stay per-core.

What is checked:
  * The specialized result matches a torch reference (numerical correctness).
  * The specialized result matches the default (unspecialized) path.
  * The dumped final MLIR shows specialization actually happened: coordinates
    are const-folded away, the coordinate-free compute kernel de-duplicates to
    one clone, and each data-movement kernel stays per-core.
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


@ttl.operation(grid=(GRID_X, GRID_Y))
def add_kernel_default(a, b, out):
    """Per-core elementwise add, compiled through the normal pipeline."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with (
            a_dfb.wait() as a_tile,
            b_dfb.wait() as b_tile,
            out_dfb.reserve() as out_tile,
        ):
            out_tile.store(a_tile + b_tile)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        with a_dfb.reserve() as a_blk:
            tx = ttl.copy(a[y, x], a_blk)
            tx.wait()
        with b_dfb.reserve() as b_blk:
            tx = ttl.copy(b[y, x], b_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        with out_dfb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()


# Separate op object so its compilation cache is independent from the default
# op. This one is always invoked with options="--ttl-specialize-cores".
@ttl.operation(grid=(GRID_X, GRID_Y))
def add_kernel_specialized(a, b, out):
    """Identical body to add_kernel_default; invoked with specialization on."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with (
            a_dfb.wait() as a_tile,
            b_dfb.wait() as b_tile,
            out_dfb.reserve() as out_tile,
        ):
            out_tile.store(a_tile + b_tile)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        with a_dfb.reserve() as a_blk:
            tx = ttl.copy(a[y, x], a_blk)
            tx.wait()
        with b_dfb.reserve() as b_blk:
            tx = ttl.copy(b[y, x], b_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        with out_dfb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()


def _make_inputs(device):
    shape = (GRID_Y * TILE_SIZE, GRID_X * TILE_SIZE)
    a_torch = torch.randn(shape, dtype=torch.bfloat16)
    b_torch = torch.randn(shape, dtype=torch.bfloat16)
    expected = (a_torch.float() + b_torch.float()).to(torch.bfloat16)
    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    return a, b, expected


def _assert_specialized_mlir(final_mlir_path, compute_prefix):
    """Confirm the dumped final MLIR reflects a specialized program."""
    with open(final_mlir_path) as fd:
        mlir = fd.read()
    assert "ttl.core_coord" in mlir, "no ttl.core_coord clones in final MLIR"
    # Coordinates should have been const-folded away.
    assert "ttl.core_x" not in mlir, "ttl.core_x should be folded to a constant"
    assert "ttl.core_y" not in mlir, "ttl.core_y should be folded to a constant"
    # The compute kernel does not use the coordinate, so control-flow de-dup
    # collapses it to a single whole-grid clone.
    compute_clones = re.findall(rf"func\.func @{compute_prefix}_c\d+_\d+", mlir)
    assert len(compute_clones) == 1, (
        f"expected the coordinate-free compute kernel to de-duplicate to one "
        f"clone, got {compute_clones}"
    )
    # Data-movement kernels address through the coordinate, so each stays a
    # separate per-core clone.
    n_cores = GRID_X * GRID_Y
    for dm_prefix in ("dm_read", "dm_write"):
        dm_clones = re.findall(rf"func\.func @{dm_prefix}_c\d+_\d+", mlir)
        assert (
            len(dm_clones) == n_cores
        ), f"expected {n_cores} per-core {dm_prefix} clones, got {dm_clones}"


def test_specialize_cores_matches_reference(device, monkeypatch, tmp_path):
    """Specialized per-core dispatch matches torch and the unspecialized path."""
    a, b, expected = _make_inputs(device)
    out_default = to_dram(torch.zeros_like(expected), device)
    out_spec = to_dram(torch.zeros_like(expected), device)

    # Unspecialized baseline (specialization off by default).
    add_kernel_default(a, b, out_default)
    default_result = ttnn.to_torch(out_default)

    # Specialized path (opt in). Dump the final MLIR so we can confirm the
    # pass ran.
    final_mlir = tmp_path / "specialized_final.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))
    add_kernel_specialized(a, b, out_spec, options="--ttl-specialize-cores")
    spec_result = ttnn.to_torch(out_spec)

    # Numerical correctness vs torch, and equivalence to the default path.
    assert_pcc(expected, spec_result)
    assert_pcc(default_result, spec_result)

    # Structural confirmation that specialization actually happened.
    _assert_specialized_mlir(str(final_mlir), compute_prefix="compute_fn")


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

    # Structural confirmation that specialization actually happened.
    _assert_specialized_mlir(str(final_mlir), compute_prefix="mm_compute")


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
        a, b, expected = _make_inputs(dev)
        out = to_dram(torch.zeros_like(expected), dev)

        n_cores = GRID_X * GRID_Y
        print(
            f"Grid: {GRID_X}x{GRID_Y} "
            f"(expect 1 deduped compute + {2 * n_cores} per-core "
            f"data-movement kernels)"
        )
        add_kernel_specialized(a, b, out, options="--ttl-specialize-cores")

        result = ttnn.to_torch(out)
        assert_pcc(expected, result)
        _assert_specialized_mlir(final_mlir, compute_prefix="compute_fn")
        print(f"OK: specialized result matches torch reference.")
        print(f"Final MLIR with clones written to {final_mlir}")
    finally:
        ttnn.close_device(dev)
