# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the per-core specialization path.

Per-core specialization is opt-in via the compiler option `specialize_cores`
(enable with `--ttl-specialize-cores`; disabled by default). It is a single
module pass (`ttkernel-specialize-cores`) run at the TTKernel level right
before EmitC: for each kernel that branches on a core coordinate it emits one
clone per launch coordinate, replacing the coordinate reads with constants and
tagging each clone with `ttl.core_coord`. The runtime bridge
(`_compile_ttnn_kernel`) turns each `ttl.core_coord` into a per-kernel core
range for dispatch.

Op bodies are built by `_make_matmul_op` / `_make_branch_swap_op` so default
and specialized runs get distinct op objects (and compilation caches) without
duplicating the kernel source. Env-var-dependent emit-runner tests call the
factory again for a fresh cache, because the key does not include
`TTLANG_EMIT_RUNNER` / `TTLANG_COMPILE_ONLY`.

Coverage:
  * matmul: addresses through the coordinate but never branches, so
    specialization skips cloning. Checks torch PCC, match vs unspecialized,
    and final MLIR has no `ttl.core_coord` / `_c<x>_<y>` clones.
  * branch_swap: reader branches on `core_x` to swap columns, so the reader is
    cloned per core. Checks torch PCC, match vs unspecialized, and that clones
    cover the launch grid (MLIR-only clone/fold checks live in
    `test/ttlang/Dialect/TTKernel/Transforms/specialize_cores.mlir`).
  * emit_runner_no_crash: `TTLANG_EMIT_RUNNER` must not IndexError when
    kernels are cloned (per-clone tensor indices / core ranges).
  * emit_runner_executes: emitted runner, run cold, reproduces the swap
    (per-kernel core ranges and NOC roles baked into the template).
  * subset_dfb: an extra DFB is reserved and waited only on column-0 cores.
    Specialization folds that use away on the other cores; the kernel still
    runs on device. Clone `ttl.used_dfb_indices` values are checked against a
    fixture with fixed indices (column-0 keeps 0 and 1; column-1 keeps 0). The
    pass itself is covered by
    `test/ttlang/Dialect/TTKernel/Transforms/specialize_cores_dfb_use_elision.mlir`.
"""

import os
import re

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import torch

import ttl
import ttl.dialects.ttl as ttl_dialect
import ttl.ttl_api as ttl_api
from ttl.ir import Context, Module
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
#
# Two op objects share this body so each has an independent compilation cache:
# matmul_default runs unspecialized; matmul_specialized is always invoked with
# options="--ttl-specialize-cores".
def _make_matmul_op():
    @ttl.operation(grid=(GRID_X, GRID_Y))
    def matmul(a, b, out):
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

    return matmul


matmul_default = _make_matmul_op()
matmul_specialized = _make_matmul_op()


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
#
# Fresh op objects share this body so each has an independent compilation
# cache. The cache is keyed on tensor properties, not on env vars like
# TTLANG_EMIT_RUNNER / TTLANG_COMPILE_ONLY, so tests that toggle those must
# call _make_branch_swap_op() again rather than reuse branch_swap_*.
def _make_branch_swap_op():
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

    return branch_swap


branch_swap_default = _make_branch_swap_op()
branch_swap_specialized = _make_branch_swap_op()


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


# An extra DFB is live only on column-0 cores. The reader and compute both
# branch on `x == 0`, so specialization clones them and folds the extra
# reserve/wait away on column 1. Column 0 writes `a + extra`; column 1 writes
# `a`. Extra is one tile-column so only the live cores address it.
def _make_subset_dfb_op():
    @ttl.operation(grid=(GRID_X, GRID_Y))
    def subset_dfb(a, extra, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        extra_dfb = ttl.make_dataflow_buffer_like(extra, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute_fn():
            x, _y = ttl.node(dims=2)
            if x == 0:
                with (
                    a_dfb.wait() as a_tile,
                    extra_dfb.wait() as extra_tile,
                    out_dfb.reserve() as o,
                ):
                    o.store(a_tile + extra_tile)
            else:
                with a_dfb.wait() as a_tile, out_dfb.reserve() as o:
                    o.store(a_tile)

        @ttl.datamovement()
        def dm_read():
            x, y = ttl.node(dims=2)
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[y, x], blk)
                tx.wait()
            if x == 0:
                with extra_dfb.reserve() as extra_blk:
                    tx = ttl.copy(extra[y, 0], extra_blk)
                    tx.wait()

        @ttl.datamovement()
        def dm_write():
            x, y = ttl.node(dims=2)
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[y, x])
                tx.wait()

    return subset_dfb


subset_dfb_specialized = _make_subset_dfb_op()

# Clone names and DFB slots for the specialized subset-DFB kernel. Column-0
# keeps the shared tile (0) and the extra tile (1); column-1 keeps only 0.
# Compute also uses the output tile (2).
_SUBSET_DFB_SPECIALIZED_MLIR = """
module {
  func.func @dm_read_c0_0() attributes {
    ttl.used_dfb_indices = array<i32: 0, 1>
  } {
    return
  }
  func.func @dm_read_c1_0() attributes {
    ttl.used_dfb_indices = array<i32: 0>
  } {
    return
  }
  func.func @compute_fn_c0_0() attributes {
    ttl.used_dfb_indices = array<i32: 0, 1, 2>
  } {
    return
  }
  func.func @compute_fn_c1_0() attributes {
    ttl.used_dfb_indices = array<i32: 0, 2>
  } {
    return
  }
}
"""


def test_specialize_cores_used_dfb_indices_match_cloned_funcs():
    """Column-0 clones keep DFB 1; column-1 clones keep only the shared slots."""
    context = Context()
    ttl_dialect.ensure_dialects_registered(context)
    with context:
        module = Module.parse(_SUBSET_DFB_SPECIALIZED_MLIR)
        assert ttl_api._get_kernel_optional_i32_array_attr(
            module, "dm_read_c0_0", "ttl.used_dfb_indices"
        ) == [0, 1]
        assert ttl_api._get_kernel_optional_i32_array_attr(
            module, "dm_read_c1_0", "ttl.used_dfb_indices"
        ) == [0]
        assert ttl_api._get_kernel_optional_i32_array_attr(
            module, "compute_fn_c0_0", "ttl.used_dfb_indices"
        ) == [0, 1, 2]
        assert ttl_api._get_kernel_optional_i32_array_attr(
            module, "compute_fn_c1_0", "ttl.used_dfb_indices"
        ) == [0, 2]


def test_specialize_cores_subset_dfb_runs_on_device(device):
    """A DFB used only on column-0 cores stays correct after specialization."""
    assert GRID_X == 2, "subset DFB reference assumes a two-column launch grid"
    a_shape = (GRID_Y * TILE_SIZE, GRID_X * TILE_SIZE)
    extra_shape = (GRID_Y * TILE_SIZE, TILE_SIZE)
    a_torch = torch.randn(a_shape, dtype=torch.bfloat16)
    extra_torch = torch.randn(extra_shape, dtype=torch.bfloat16)
    expected = a_torch.clone()
    expected[:, :TILE_SIZE] = a_torch[:, :TILE_SIZE] + extra_torch

    a = to_dram(a_torch, device)
    extra = to_dram(extra_torch, device)
    out = to_dram(torch.zeros(a_shape, dtype=torch.bfloat16), device)

    subset_dfb_specialized(a, extra, out, options="--ttl-specialize-cores")

    assert_pcc(expected, ttnn.to_torch(out))


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


def test_specialize_cores_emit_runner_executes(device, monkeypatch, tmp_path):
    """The emitted runner, run standalone on a cold cache, must reproduce the swap.

    Compile-only so the op never executes and warms the program cache; otherwise
    the runner's custom_program_hash could reuse a cached in-process program and
    mask a template bug. The emitted runner must carry per-kernel core ranges
    and NOC roles so clones dispatch correctly when built cold.
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
